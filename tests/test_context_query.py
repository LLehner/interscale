"""A context-only global embedding: the transformer's queries from a position, not from the cell (`model.global_component.parameters.context_only: query`; decoder_plan.md, D1).

In the plain transformer a cell's token rides in its own residual stream, so its global embedding carries the cell (near
visible cells the global head copies the local one). With ``context_only: query`` every cell is a *query stream*: it starts
as one learned token plus the cell's node encoding and, in each layer, reads the cells' input tokens through that layer's
weights. Pinned here:

* **the switch** -- off by default, then no state is added; on, the model differs by the one learned token and every other
  weight is initialised as before, without a draw from the global generator;
* **no trace of the cell** -- a cell's global embedding does not move, bit for bit, with its own expression: at one and
  two layers, with node encodings, an attention bias or RoPE, through a pooled patch, and behind a GCN under the long-range
  mask (where the cells the GCN mixed it into are blocked as keys too). The plain transformer's does. The one exception is
  ``attention.center_qk``: the mean over all cells that queries and keys are centred on holds the cell at weight 1 / S,
  and the dependence is shown to fall with S;
* **what it is** -- the read-out equals its definition written out with plain tensor operations; the queries are the
  learned token plus the node encodings laid out as the tokens (also when a long slide is subsampled); the attention
  never reads a blocked pair and its rows sum to 1; gradients reach the token, the layers and CLS;
* **refusals** -- an unknown kind, a graph-level prediction, any attention backend but torch's dense one, a forward pass
  without its queries, and the readouts that compose attention as token mixing;
* **through the real model and trainer** -- a global-only model with the own-cell skip and a dual model with the additive
  decoder train, infer, and carry ``ctx-query`` in their checkpoint names.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.module.base import GlobalModule, gather_tokens
from interscale.tl.utils import get_model_filename_prefix

G, E = 6, 16
SPACING = 100.0
SIDE = 6


def _cfg(context="query", *, layers=2, center=False, bias=("fourier",), rotary=(), node=("sinusoidal",), lrm=False):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 128, 4, layers, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.node, params.pe.bias, params.pe.rotary = list(node), list(bias), list(rotary)
    params.pe.fourier.max_wavelength = params.pe.rope.max_wavelength = params.pe.distance.max_dist = 2000.0
    params.attention.center_qk = center
    params.long_range_attention = lrm
    params.context_only = context
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um, cfg.dataset.name = "spatial", 1.0, "d"
    cfg.dataset.sample_key = ["sample"]
    return cfg


def _stir(module, scale=0.1, seed=7):
    """Every parameter moved by a little seeded noise.

    A fresh module is degenerate: the node-encoding projections and the attention-bias tables start at zero (a switched-on
    encoding changes nothing at step 0), the layer norms are identities, and the query biases are zero, so a position, a norm
    or a bias that was left out would change nothing a test could see. After this none of them is.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=gen))
    return module


def _module(seed=0, stir=True, mask_percentage=0.0, mask_strategy="node", **kwargs):
    """A global-only transformer module (Linear tokens) as the config builds it, with no parameter left at its degenerate start."""
    torch.manual_seed(seed)
    module = GlobalModule.from_config(
        _cfg(**kwargs),
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=mask_percentage,
        mask_strategy=mask_strategy,
        type_gex_embedding="Linear",
    ).eval()
    return _stir(module) if stir else module


def _graph(side=SIDE, seed=0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    batch = Batch.from_data_list([data])
    batch.obs_names = torch.arange(n)
    return batch


def _rows(module, batch):
    """``(global embeddings [N, E] in cell order, attention maps or None)`` of a global-only module's pass."""
    tokens = module.input_embedding(batch.x.float())
    with torch.no_grad(), module.recording_attention() as record:
        out, padding, index, _, _ = module.step(batch, tokens)
    rows = gather_tokens(out, padding)
    order = torch.as_tensor(index[0], dtype=torch.long)
    cells = torch.empty_like(rows)
    cells[order] = rows
    return cells, record[0]["maps"]


def _changed(batch, cell, amount=3.0):
    other = batch.clone()
    other.x = batch.x.clone()
    other.x[cell] += amount
    return other


CELL = (SIDE // 2) * SIDE + SIDE // 2


# --------------------------------------------------------------------------- the switch


def test_off_by_default_adds_no_state_and_forward_takes_no_queries():
    assert _cfg().model.global_component.parameters.context_only == "query"
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    default = get_global_component_cfg(get_cfg_defaults(), "self-attn-transformer")
    assert default.model.global_component.parameters.context_only == "none"
    plain = _module(context="none")
    assert plain.context_only == "none" and not any("context_token" in key for key in plain.state_dict())
    batch = _graph()
    tokens = plain.input_embedding(batch.x.float())
    given = plain.common_step_local_to_global(batch, tokens, eval_step=True)
    assert given.queries is None and len(given) == 5  # the five fields are unpacked by position all over
    with pytest.raises(ValueError, match="context_only is off"):
        plain.forward(given[0], given[1], given[3], rotary=given[4], queries=torch.zeros_like(given[0]))


def test_on_the_model_differs_by_one_token_and_every_other_weight_is_initialised_as_before():
    plain, query = _module(context="none", stir=False), _module(context="query", stir=False)
    extra = set(query.state_dict()) - set(plain.state_dict())
    assert extra == {"context_token"} and set(plain.state_dict()) <= set(query.state_dict())
    assert all(torch.equal(plain.state_dict()[k], query.state_dict()[k]) for k in plain.state_dict())
    # ... drawing nothing from the global generator
    torch.manual_seed(5)
    _module(context="query", seed=1, stir=False)
    after_query = torch.rand(3)
    torch.manual_seed(5)
    _module(context="none", seed=1, stir=False)
    assert torch.equal(after_query, torch.rand(3))


# --------------------------------------------------------------------------- no trace of the cell


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"layers": 1},
        {"layers": 3},
        {"node": ()},
        {"bias": ()},
        {"bias": (), "rotary": ("rope",), "node": ()},
        {"bias": ("fourier",), "rotary": ("rope",)},
    ],
    ids=["two-layers", "one-layer", "three-layers", "no-node-encoding", "no-bias", "rope-only", "bias-rope-node"],
)
def test_a_cells_embedding_does_not_move_with_its_own_expression_while_the_plain_transformers_does(kwargs):
    batch = _graph()
    query, plain = _module(context="query", **kwargs), _module(context="none", **kwargs)
    before, _ = _rows(query, batch)
    after, _ = _rows(query, _changed(batch, CELL))
    assert torch.equal(after[CELL], before[CELL])  # bit for bit: no path from the cell's token to its embedding
    others = [i for i in range(batch.num_nodes) if i != CELL]
    assert (after[others] - before[others]).abs().max() > 0  # and the others do read it
    plain_before, _ = _rows(plain, batch)
    plain_after, _ = _rows(plain, _changed(batch, CELL))
    assert (plain_after[CELL] - plain_before[CELL]).abs().max() > 1e-2  # the cell's token rides in its stream


def test_every_cell_is_blind_to_itself_not_just_one():
    batch = _graph()
    module = _module()
    before, _ = _rows(module, batch)
    for cell in (0, 7, CELL, batch.num_nodes - 1):
        after, _ = _rows(module, _changed(batch, cell, amount=-0.7))
        assert torch.equal(after[cell], before[cell]), cell


def test_with_center_qk_the_cell_enters_only_through_the_sequence_mean_at_weight_one_over_s():
    """Queries and keys are centred on the mean over all cells' tokens, the cell's own included: a 1 / S effect."""
    effects = {}
    for side in (4, 8):
        batch = _graph(side)
        cell = (side // 2) * side + side // 2
        module = _module(center=True)
        before, _ = _rows(module, batch)
        after, _ = _rows(module, _changed(batch, cell))
        effects[side] = (after[cell] - before[cell]).abs().max().item()
        plain_before, _ = _rows(_module(context="none", center=True), batch)
        plain_after, _ = _rows(_module(context="none", center=True), _changed(batch, cell))
        assert effects[side] < 0.2 * (plain_after[cell] - plain_before[cell]).abs().max().item()
    assert 0 < effects[8] < 0.6 * effects[4]  # four times the cells, well under the effect
    # no centring: nothing at all
    batch = _graph(8)
    off, _ = _rows(_module(center=False), batch)
    assert torch.equal(_rows(_module(center=False), _changed(batch, 36))[0][36], off[36])


# --------------------------------------------------------------------------- behind a GCN and with patches


def _config_combined(context, *, lrm):
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = _cfg(context, layers=1, lrm=lrm, bias=("fourier",))
    cfg.model.local_component.name = "GCN"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg.model.decoder.dual_decoder = True
    cfg.model.n_embed = E
    cfg.model.global_component.parameters.n_heads = 2
    cfg.dataset.mask_percentage = 0.0
    return cfg


def _build(cfg, seed=0):
    from interscale.model.combined_model import CombinedModel

    rng = np.random.default_rng(0)
    n = SIDE * SIDE
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    CombinedModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    net = CombinedModel(adata, cfg=cfg)
    _stir(net.module)
    return net


def _global_rows(net, batch):
    """The global embeddings of a combined model's pass over ``batch``, in cell order, and the tokens it read."""
    module = net.module.eval()
    with torch.no_grad():
        tokens = module.local_module.forward(batch.x, batch.edge_index)
        out, padding, index, _, _ = module.global_module.step(batch, tokens)
    rows = gather_tokens(out, padding)
    cells = torch.empty_like(rows)
    cells[torch.as_tensor(index[0], dtype=torch.long)] = rows
    return cells


def test_behind_a_gcn_the_long_range_mask_keeps_the_neighbours_that_hold_the_cell_out_of_its_embedding():
    """The GCN mixes a cell into the tokens within 2 hops; with the mask on none of them is a key of its read-out."""
    batch = _graph()
    masked = _build(_config_combined("query", lrm=True))
    assert masked.module.local_module.receptive_field_hops == 2
    before = _global_rows(masked, batch)
    after = _global_rows(masked, _changed(batch, CELL))
    assert torch.equal(after[CELL], before[CELL])
    assert (after - before).abs().max() > 0  # cells beyond the reach read the changed neighbourhood
    # Without the mask the cell's neighbours' GCN tokens, which contain it, are keys: it is not alone any more.
    open_ = _build(_config_combined("query", lrm=False))
    before_open = _global_rows(open_, batch)
    after_open = _global_rows(open_, _changed(batch, CELL))
    assert (after_open[CELL] - before_open[CELL]).abs().max() > 0


def test_a_pooled_patch_reads_the_other_patches_only():
    cfg = _cfg(layers=2, bias=("fourier",))
    cfg.model.global_component.parameters.pooling.kind = "grid"
    cfg.model.global_component.parameters.pooling.size_um = 3 * SPACING
    torch.manual_seed(0)
    module = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type="linear",
        mask_percentage=0.0,
        mask_strategy="node",
        type_gex_embedding="Linear",
    ).eval()
    _stir(module)
    batch = _graph()

    def rows(b):
        with torch.no_grad():
            out, padding, index, _, _ = module.step(b, module.input_embedding(b.x.float()))
        cells = torch.empty(b.num_nodes, E)
        cells[torch.as_tensor(index[0], dtype=torch.long)] = gather_tokens(out, padding)
        return cells

    before, after = rows(batch), rows(_changed(batch, 0))
    patch = (batch.pos[:, 0] // (3 * SPACING) == 0) & (batch.pos[:, 1] // (3 * SPACING) == 0)  # cell 0's patch
    assert patch[0] and patch.sum() == 9
    assert torch.equal(after[patch], before[patch])  # every cell of the patch reads the *other* patches
    assert (after[~patch] - before[~patch]).abs().max() > 0


# --------------------------------------------------------------------------- what the read-out is


def _written_out(module, batch):
    """The read-out from the definition, with plain tensor operations and the module's own weights (one graph)."""
    tokens = module.input_embedding(batch.x.float())
    given = module.common_step_local_to_global(batch, tokens, eval_step=True)
    padded, padding, _, mask, _ = given
    cls = module.cls_embedding.expand(1, 1, -1)
    memory = module.norm_input(torch.cat([padded, cls], dim=0))  # [S + 1, 1, E]
    state = module.norm_input(torch.cat([module.context_token.view(1, 1, -1) + given.queries, cls], dim=0))
    heads, maps = module.n_heads, []
    for layer in module.transformer_encoder.layers:
        attn = layer.self_attn
        w_q, w_k, w_v = attn.in_proj_weight.chunk(3)
        b_q, b_k, b_v = attn.in_proj_bias.chunk(3)
        q_in, k_in = layer.norm1(state), layer.norm1(memory)
        v_in = k_in
        if layer.center_qk:
            from interscale.module.global_modules.transformer_encoder_layer import cell_positions, center_tokens

            cells = cell_positions(k_in, padding_mask_with_cls(padding))
            q_in, k_in = center_tokens(q_in, cells), center_tokens(k_in, cells)
        length = state.shape[0]
        q = (q_in @ w_q.T + b_q).view(length, heads, -1).transpose(0, 1)  # [H, L, hd]
        k = (k_in @ w_k.T + b_k).view(length, heads, -1).transpose(0, 1)
        v = (v_in @ w_v.T + b_v).view(length, heads, -1).transpose(0, 1)
        logits = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        blocked = (
            mask
            if mask.dtype != torch.bool
            else torch.zeros_like(mask, dtype=logits.dtype).masked_fill(mask, float("-inf"))
        )
        weights = torch.softmax(logits + blocked, dim=-1)
        maps.append(weights)
        mixed = (weights @ v).transpose(0, 1).reshape(length, -1)
        state = state + attn.out_proj(mixed).unsqueeze(1)
        state = state + layer.linear2(layer.activation(layer.linear1(layer.norm2(state))))
    return module.transformer_encoder.norm(state), torch.stack(maps)


def padding_mask_with_cls(padding):
    return torch.cat([padding, padding.new_zeros(padding.shape[0], 1)], dim=1)


@pytest.mark.parametrize(
    "kwargs", [{}, {"center": True}, {"layers": 1, "bias": (), "node": ()}], ids=["default", "centred", "plain"]
)
def test_the_read_out_equals_its_definition_written_out(kwargs):
    module = _module(**kwargs)
    batch = _graph()
    tokens = module.input_embedding(batch.x.float())
    with torch.no_grad(), module.recording_attention() as record:
        out, padding, _, _, _ = module.step(batch, tokens)
        expected, expected_maps = _written_out(module, batch)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5)
    maps = record[0]["maps"]  # [layers, B * heads, S + 1, S + 1]
    assert torch.allclose(maps, expected_maps, atol=1e-5)


def test_the_queries_are_the_node_encodings_laid_out_as_the_tokens_also_for_a_subsampled_slide():
    """A slide longer than the sequence keeps a subset of its cells; the queries follow the cells that were kept."""
    import random

    cfg = _cfg()
    cfg.model.global_component.parameters.max_seq_len = 20
    torch.manual_seed(0)
    module = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type="linear",
        mask_percentage=0.0,
        mask_strategy="node",
        type_gex_embedding="Linear",
    ).eval()
    _stir(module)
    batch = _graph()  # 36 cells, 20 tokens
    tokens = module.input_embedding(batch.x.float())
    random.seed(1)
    given = module.common_step_local_to_global(batch, tokens, eval_step=True)
    kept = torch.as_tensor(given.index_nodes[0], dtype=torch.long)
    assert len(kept) == 20 and given.queries.shape == given.padded_emb.shape
    encoding = module.positional_encoding(batch, dtype=tokens.dtype, device=tokens.device, inputs={})
    assert torch.allclose(given.queries[:, 0], encoding[kept], atol=1e-6)
    # ... and without a node encoding the queries are zero: only the learned token is left
    bare = _module(node=())
    zero = bare.common_step_local_to_global(batch, bare.input_embedding(batch.x.float()), eval_step=True)
    assert not zero.queries.any()


def test_the_attention_never_reads_a_blocked_pair_and_its_rows_sum_to_one():
    module = _module(layers=2)
    batch = _graph()
    _, maps = _rows(module, batch)
    n = batch.num_nodes
    assert maps.shape == (2, 4, n + 1, n + 1)  # layers, heads, queries (CLS last), keys
    diagonal = torch.diagonal(maps[..., :n, :n], dim1=-2, dim2=-1)
    assert diagonal.abs().max() == 0  # a cell never reads its own token
    assert torch.allclose(maps.sum(-1), torch.ones(2, 4, n + 1), atol=1e-5)
    assert maps[..., :n, n].min() > 0  # the CLS key stays open for every cell


def test_the_long_range_mask_blocks_what_the_gcn_mixed_in_for_the_read_out_too():
    batch = _graph()
    net = _build(_config_combined("query", lrm=True))
    module = net.module.eval()
    with torch.no_grad(), module.global_module.recording_attention() as record:
        tokens = module.local_module.forward(batch.x, batch.edge_index)
        module.global_module.step(batch, tokens)
    maps = record[0]["maps"][0]  # [heads, S + 1, S + 1]
    ii, jj = torch.meshgrid(torch.arange(SIDE), torch.arange(SIDE), indexing="ij")
    hops = (ii.flatten()[:, None] - ii.flatten()[None, :]).abs() + (jj.flatten()[:, None] - jj.flatten()[None, :]).abs()
    near = hops <= 2
    n = SIDE * SIDE
    assert maps[:, :n, :n][:, near].abs().max() == 0
    assert maps[:, :n, :n][:, ~near].min() > 0


def test_gradients_reach_the_token_the_layers_and_cls():
    module = _module().train()
    batch = _graph()
    tokens = module.input_embedding(batch.x.float())
    out, padding, _, _, _ = module.step(batch, tokens)
    (gather_tokens(out, padding) * torch.randn(1, E, generator=torch.Generator().manual_seed(2))).sum().backward()
    assert module.context_token.grad.abs().sum() > 0
    assert module.cls_embedding.grad.abs().sum() > 0
    assert module.input_embedding.weight.grad.abs().sum() > 0
    for layer in module.transformer_encoder.layers:
        assert layer.self_attn.in_proj_weight.grad.abs().sum() > 0 and layer.linear1.weight.grad.abs().sum() > 0


def test_padding_is_never_read_a_short_slide_batched_with_a_long_one_reads_what_it_reads_alone():
    """Two slides of different sizes are padded to one length; the padded keys must carry no attention."""
    module = _module(layers=2)
    short, long_ = _graph(4), _graph(6)
    both = Batch.from_data_list([short.get_example(0), long_.get_example(0)])
    both.obs_names = torch.arange(both.num_nodes)
    with torch.no_grad():
        alone = module.step(short, module.input_embedding(short.x.float()))
        together = module.step(both, module.input_embedding(both.x.float()))
    rows_alone = gather_tokens(alone[0], alone[1])
    mine = (together[3].shape[0], together[1].shape)  # (B * heads, padded + 1): the short slide is padded
    assert together[1][0].sum() == 36 - 16, mine  # 20 padded positions in the first graph
    rows_together = gather_tokens(together[0], together[1])[:16]
    order_alone = torch.as_tensor(alone[2][0], dtype=torch.long)
    order_together = torch.as_tensor(together[2][0], dtype=torch.long)
    assert torch.allclose(rows_alone[order_alone.argsort()], rows_together[order_together.argsort()], atol=1e-5)


def test_the_module_itself_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="context_only must be one of"):
        _module(context="stream")


def test_later_layers_query_with_what_the_earlier_ones_read():
    """Two layers are not one layer twice: the second layer's queries differ between cells (they read different places)."""
    module = _module(layers=1, node=())
    deeper = _module(layers=2, node=())
    batch = _graph()
    one, _ = _rows(module, batch)
    two, _ = _rows(deeper, batch)
    assert not torch.allclose(one, two, atol=1e-3)
    assert two.std(0).min() > 0  # cells differ: each read another place


# --------------------------------------------------------------------------- refusals


def test_the_config_wants_a_known_kind_a_node_level_prediction_and_torchs_dense_attention():
    from interscale.config import _validate_context_only

    cfg = _cfg()
    cfg.dataset.prediction_level = "node"
    _validate_context_only(cfg)
    _validate_context_only(_cfg("none"))
    bad = _cfg()
    bad.model.global_component.parameters.context_only = "stream"
    with pytest.raises(ValueError, match="context_only must be one of"):
        _validate_context_only(bad)
    graph = _cfg()
    graph.dataset.prediction_level = "graph"
    with pytest.raises(ValueError, match="prediction_level node"):
        _validate_context_only(graph)
    for kind in ("performer", "entmax", "topk", "linear", "exphormer"):
        backend = _cfg()
        backend.dataset.prediction_level = "node"
        backend.model.global_component.parameters.attention.kind = kind
        with pytest.raises(ValueError, match="torch's dense attention"):
            _validate_context_only(backend)
    wide = _cfg()
    wide.dataset.prediction_level = "node"
    wide.model.global_component.parameters.attention.head_dim = 32  # not n_embed / n_heads
    with pytest.raises(ValueError, match="default head width"):
        _validate_context_only(wide)


def test_config_loading_runs_the_check():
    from interscale.config import _validate

    cfg = _cfg()
    cfg.model.global_component.parameters.attention.kind = "performer"
    with pytest.raises(ValueError):
        _validate(cfg)


def test_a_module_built_on_another_attention_backend_refuses_the_option():
    cfg = _cfg()
    cfg.model.global_component.parameters.attention.head_dim = 32
    with pytest.raises(ValueError, match="vendored dense attention"):
        GlobalModule.from_config(
            cfg,
            local_mask_hops=0,
            n_input=G,
            n_output=G,
            n_embed=E,
            decoder_type="linear",
            mask_percentage=0.0,
            mask_strategy="node",
            type_gex_embedding="Linear",
        )


def test_forward_refuses_missing_or_foreign_queries():
    module = _module()
    batch = _graph()
    given = module.common_step_local_to_global(batch, module.input_embedding(batch.x.float()), eval_step=True)
    with pytest.raises(ValueError, match="needs the queries"):
        module.forward(given[0], given[1], given[3], rotary=given[4])
    with pytest.raises(ValueError, match="another batch"):
        module.forward(given[0], given[1], given[3], rotary=given[4], queries=given.queries[:-1])


def test_the_readouts_that_compose_attention_as_token_mixing_refuse_it():
    from interscale.evaluation.attention_check import attention_readouts, check_attention
    from interscale.evaluation.attention_flow import patch_flow

    module = _module()
    with pytest.raises(ValueError, match="token mixing"):
        check_attention(module, [], prediction_task="regression", prediction_level="node", device="cpu")
    with pytest.raises(ValueError, match="token mixing"):
        attention_readouts(
            module, [], prediction_task="regression", prediction_level="node", device="cpu", spatial_unit_um=1.0
        )
    with pytest.raises(ValueError, match="token mixing"):
        patch_flow(
            module,
            [[]],
            prediction_task="regression",
            prediction_level="node",
            mask_strategy="gene",
            device="cpu",
            spatial_unit_um=1.0,
            patch_um=100.0,
            kinds=("rollout",),
        )


def test_the_checkpoint_name_carries_it():
    names = {context: get_model_filename_prefix(_cfg(context), False, True) for context in ("none", "query")}
    assert names["none"] != names["query"] and "ctx-query" in names["query"] and "ctx" not in names["none"]
    combined = _config_combined("query", lrm=True)
    assert "var-lrm+ctx-query_" in get_model_filename_prefix(combined, True, True)


def test_a_model_without_a_transformer_refuses_to_train_with_the_option():
    import types

    from interscale.train._training import NodeMaskingTrainingPlan

    local_only = types.SimpleNamespace(_cfg=_cfg(), module=types.SimpleNamespace())
    with pytest.raises(ValueError, match="context_only"):
        NodeMaskingTrainingPlan._refuse_unsupported_options(local_only)
    has_it = types.SimpleNamespace(_cfg=_cfg(), module=types.SimpleNamespace(context_only="query"))
    NodeMaskingTrainingPlan._refuse_unsupported_options(has_it)
    inside = types.SimpleNamespace(
        _cfg=_cfg(), module=types.SimpleNamespace(global_module=types.SimpleNamespace(context_only="query"))
    )
    NodeMaskingTrainingPlan._refuse_unsupported_options(inside)
    plain = types.SimpleNamespace(_cfg=_cfg("none"), module=types.SimpleNamespace())
    NodeMaskingTrainingPlan._refuse_unsupported_options(plain)


# --------------------------------------------------------------------------- the readouts that stay meaningful


def test_the_attention_diagnostics_read_the_read_outs_attention():
    from interscale.evaluation.attention_diagnostics import attention_diagnostics

    module = _module()
    graphs = [_graph(seed=k).get_example(0) for k in range(2)]
    for graph in graphs:
        graph.mask = torch.rand(graph.num_nodes, generator=torch.Generator().manual_seed(3)) < 0.3
    found = attention_diagnostics(
        module,
        graphs,
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        spatial_unit_um=1.0,
        radii_um=(300.0,),
    )
    assert found and all(np.isfinite(v) for v in found.values())
    assert any("span" in key for key in found) and any("effective_keys" in key for key in found)


# --------------------------------------------------------------------------- through the real models and the trainer


def _adata(tmp_path, per=36):
    samples = ["s1", "s2", "s3", "s4"]
    rng = np.random.default_rng(0)
    n = per * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(n, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, per))
    split = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    path = tmp_path / "context.h5ad"
    adata.write_h5ad(path)
    return path


def _yaml(tmp_path, *, combined):
    h5ad = _adata(tmp_path)
    local = "  local_component:\n    name: GCN\n" if combined else ""
    decoder = (
        "    type: linear\n    dual_decoder: True\n    additive: True\n"
        if combined
        else "    type: linear\n    own_cell_skip: True\n"
    )
    weights = "  loss_weights:\n    local: 0.0\n    global: 1.0\n" if combined else ""
    lrm = "True" if combined else "False"
    path = tmp_path / "context.yaml"
    path.write_text(
        f"model:\n  n_embed: 8\n{local}  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 64\n      num_layers: 2\n"
        f"      n_heads: 2\n      long_range_attention: {lrm}\n      type_gex_embedding: Linear\n"
        "      context_only: query\n      pe:\n        node: [sinusoidal]\n"
        f"  decoder:\n{decoder}  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 3\n"
        f"  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n{weights}"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: contexttest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
    )
    return path


def _train(tmp_path, *, combined):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    cfg = load_config(str(_yaml(tmp_path, combined=combined)))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    model_cls = CombinedModel if combined else GlobalModel
    model_cls._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    net = model_cls(adata, cfg=cfg)
    transformer = getattr(net.module, "global_module", net.module)
    assert transformer.context_only == "query"
    start = transformer.context_token.detach().clone()
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    net.train(max_epochs=3, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    rows = [row for row in net.history_.history if "val_loss" in row]
    assert rows and all(np.isfinite(row["val_loss"]) for row in rows)
    assert not torch.equal(transformer.context_token.detach().cpu(), start)  # the learned token trains
    return net, adata, cfg


@pytest.mark.parametrize("combined", [False, True], ids=["global-only-with-skip", "dual-additive"])
def test_a_model_trains_through_the_trainer_and_names_its_checkpoint(tmp_path, combined):
    net, _, _ = _train(tmp_path, combined=combined)
    assert any("ctx-query" in p.name for p in (tmp_path / "out").iterdir())


def test_inference_runs_the_read_out_too_and_reports_the_same_predictions(tmp_path, monkeypatch):
    """``get_model_output`` decodes outside ``_common_step``: the clean pass is the query stream as well."""
    import random

    import interscale.model.global_model as global_model

    net, adata, cfg = _train(tmp_path, combined=False)
    module = net.module.eval()
    graph = Data(
        x=torch.tensor(adata.X[:36]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pos=torch.tensor(adata.obsm["spatial"][:36], dtype=torch.float32),
        obs_names=torch.arange(1, 37),
    )
    monkeypatch.setattr(global_model, "prepare_a2d_dataset", lambda cfg: lambda adata: [[graph], None])
    monkeypatch.setattr(global_model, "attach_positional_inputs", lambda pyg, cfg: None)
    saved = {}
    monkeypatch.setattr(net, "save_evaluation_results", lambda adata, prefix, **out: saved.update(out) or adata)
    random.seed(0)
    net.get_model_output(adata[:36].copy())
    found = saved["y_pred_global_df"].to_numpy(dtype=float)
    assert np.isfinite(found).all() and found.shape == (36, G)

    # what the module itself gives on the clean pass: the read-out's embedding through the decoder, plus the skip
    random.seed(0)
    tokens = module.gex_embedding(graph)
    _, emb, padding, kept, _ = module.evaluate(graph.clone(), tokens)
    expected = module.predict(emb, padding, "node", module.own_cell_input(graph, tokens), torch.as_tensor(kept[0]))
    assert np.allclose(found[np.argsort(kept[0])], expected.detach().numpy()[np.argsort(kept[0])], atol=1e-5)


# --------------------------------------------------------------------------- what it can and cannot learn


def _toy_slide(seed, cells=30):
    gen = torch.Generator().manual_seed(seed)
    data = Data(
        x=torch.randn(cells, G, generator=gen),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pos=torch.rand(cells, 2, generator=gen) * 1000,
    )
    data.mask = torch.zeros(cells, dtype=torch.bool)
    batch = Batch.from_data_list([data])
    batch.obs_names = torch.arange(cells)
    return batch


def _fit_toy(context, task, steps=300):
    """Held-out correlation of a read-out trained on ``task``: the other cells' mean of gene 0, or the cell's own gene 0."""
    module = _module(context=context, layers=2, stir=False).train()  # from the default start, as a model is trained
    head = torch.nn.Linear(E, 1)
    optimiser = torch.optim.Adam([*module.parameters(), *head.parameters()], lr=5e-3)

    def target(batch):
        x = batch.x[:, 0]
        return (x.sum() - x) / (len(x) - 1) if task == "context" else x

    def predict(batch):
        out, padding, index, _, _ = module.step(batch, module.input_embedding(batch.x.float()))
        cells = torch.empty(batch.num_nodes)
        cells[torch.as_tensor(index[0], dtype=torch.long)] = head(gather_tokens(out, padding)).squeeze(-1)
        return cells

    train = [_toy_slide(k) for k in range(24)]
    for step in range(steps):
        batch = train[step % len(train)]
        loss = ((predict(batch) - target(batch)) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    module.eval()
    test = [_toy_slide(100 + k) for k in range(6)]
    with torch.no_grad():
        predicted = torch.cat([predict(b) for b in test])
        wanted = torch.cat([target(b) for b in test])
    return torch.corrcoef(torch.stack([predicted, wanted]))[0, 1].item()


def test_the_attribution_instrument_finds_nothing_of_the_receiver_in_its_own_prediction():
    """Integrated gradients from a baseline cell, the sender ranks' instrument: the receiver contributes to everyone's prediction but its own."""
    from interscale.evaluation.masked_probing import _model_heads, draw_masks
    from interscale.evaluation.receptive_field import _frozen_parameters, _program_attribution

    data = _graph().get_example(0)
    data.obs_names = torch.arange(data.num_nodes, dtype=torch.float32) + 1
    response = 5
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[response])  # the response gene hidden in every cell
    own = {}
    for context in ("query", "none"):
        module = _module(context=context, mask_percentage=1.0, mask_strategy="gene")
        with _frozen_parameters(module):
            found = _program_attribution(
                module,
                Batch.from_data_list([data]),
                [CELL],
                [np.isin(np.arange(G), [response])],
                None,
                "regression",
                8,
                _model_heads,
            )
        contribution, change = found["global"]
        assert contribution[0].abs().max() > 0 and torch.isfinite(contribution[0]).all()
        assert contribution[0].sum().item() == pytest.approx(change[0].item(), rel=0.1)  # the contributions add up
        own[context] = contribution[0, CELL].abs().item()
    assert own["query"] == 0.0  # a context-only prediction holds nothing of its own cell
    assert own["none"] > 0.0  # the plain transformer's does


def test_the_read_out_learns_what_the_context_says_and_cannot_learn_the_cell_itself():
    """A target the other cells determine is learned; a copy of the cell's own gene, which the plain transformer learns, is not."""
    assert _fit_toy("none", "mirror") > 0.9  # the plain transformer copies its own token
    assert abs(_fit_toy("query", "mirror")) < 0.25  # the query stream has no token of its own to copy
    assert _fit_toy("query", "context") > 0.3  # but it reads the context
