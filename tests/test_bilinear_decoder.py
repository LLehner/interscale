"""The bilinear part (``model.decoder.bilinear``; roadmap "The user's questions for next week", point 1, 2026-11-12):
``y_i = sum_j a_ij (x_j V) U^T`` with ``a_ij`` the last layer's attention and ``B = U V^T`` the readable gene-to-gene map.

Pinned here:

* **the part** -- equals the hand-computed sum over the kept cells' attention rows and visible genes, in the prediction's rows;
* **no hidden value** -- a hidden entry's value never reaches it (cell masking and gene masking);
* **the switch** -- off, no new state and no random draw; on, ``U`` zero so the model starts as without it;
* **the gradient** -- the part trains ``U``, ``V`` and the attention (the last layer's query / key projections);
* **the refusals** -- validator (rank, head, backend, pooling, hierarchy), the module (CombinedModel), the tag;
* **the readouts** -- ``gene_map``, ``components``, ``planted_recovery``, ``compare_maps``, ``pair_weights`` on known matrices.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

E, G, H, R = 8, 6, 2, 3


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _graph(side, seed, *, strategy="node", hidden_genes=(0, 1)):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * 100.0
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= 101.0).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) * 3 + 1.0, edge_index=edges, pos=pos)
    if strategy == "gene":
        rows = torch.arange(n) % 3 == 0
        data.gene_mask = torch.zeros(n, G, dtype=torch.bool)
        data.gene_mask[:, list(hidden_genes)] = rows[:, None]
        data.mask = data.gene_mask.any(1)
    else:
        data.mask = torch.arange(n) % 4 == 0
    data.obs_names = torch.arange(n)
    return data


def _cfg(rank=R, head=-1, strategy="node", **over):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    cfg.model.decoder.bilinear.rank = rank
    cfg.model.decoder.bilinear.head = head
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 64, H, 2, 0.0
    params.type_gex_embedding = "Linear"
    cfg.optim.loss = "SCELoss"
    cfg.dataset.mask_strategy = strategy
    cfg.dataset.mask_percentage = 0.25 if strategy == "node" else 1.0
    if strategy == "gene":
        cfg.dataset.mask_genes = ["g0", "g1"]
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    for key, value in over.items():
        node, leaf = key.rsplit(".", 1)
        target = cfg
        for part in node.split("."):
            target = target[part]
        target[leaf] = value
    return cfg


def _model(seed=0, **kwargs):
    from interscale.config import _validate
    from interscale.model.global_model import GlobalModel

    cfg = _cfg(**kwargs)
    _validate(cfg)
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(seed)
    return GlobalModel(adata, cfg=cfg)


def _by_hand(module, batch, out, head):
    """The part from the returned map and the masked batch, cell by cell, in the prediction's rows."""
    masked, _, _ = module._common_step_masking(batch)
    visible = module.visible_entries(masked)
    x = torch.where(visible, masked.x.float(), torch.zeros_like(masked.x.float()))
    sent = x @ module.bilinear_sender  # [N, r]
    attn = out.view.attn[-1]  # [B*H, S+1, S+1]
    n_graphs = int(masked.batch.max()) + 1
    seq = attn.shape[1] - 1
    maps = attn.view(n_graphs, H, seq + 1, seq + 1)[:, :, :seq, :seq]
    a = maps.mean(1) if head < 0 else maps[:, head]
    want = torch.zeros(len(out.y_pred), G)
    node_idx = out.view.padded_node_idx
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(masked.batch.long(), minlength=n_graphs).cumsum(0)[:-1]])
    # which token row each kept cell is: graph b, position seq - n_b + k
    row = 0
    for b in range(n_graphs):
        n_b = int((masked.batch == b).sum())
        for k in range(n_b):
            cell = node_idx[row]
            assert cell == starts[b] + k  # no subsampling: every cell kept, in order
            received = torch.zeros(R)
            for kk in range(n_b):
                received += a[b, seq - n_b + k, seq - n_b + kk] * sent[starts[b] + kk]
            want[row] = received @ module.bilinear_receiver.T
            row += 1
    return want


@pytest.mark.parametrize("head", [-1, 1])
def test_the_part_is_the_attention_weighted_bilinear_read_of_the_neighbours_visible_genes(head):
    net = _model(head=head)
    module = _stir(net.module)
    module.eval()
    batch = Batch.from_data_list([_graph(4, 1), _graph(5, 2)])
    out = module._common_step(batch, "regression", "node")
    parts = out.parts["global"]
    assert set(parts) == {"context", "bilinear"}
    assert torch.allclose(sum(parts.values()), out.y_pred, atol=1e-5)
    assert out.view.attn is not None and out.view.attn.shape[0] == 2  # both layers' maps, with the pass
    want = _by_hand(module, batch, out, head)
    assert torch.allclose(parts["bilinear"], want, atol=1e-5)
    assert parts["bilinear"].abs().sum() > 0


@pytest.mark.parametrize("strategy", ["node", "gene"])
def test_no_hidden_value_reaches_the_part(strategy):
    net = _model(strategy=strategy)
    module = _stir(net.module)
    module.eval()
    graphs = [_graph(4, 1, strategy=strategy), _graph(5, 2, strategy=strategy)]
    out = module._common_step(Batch.from_data_list(graphs), "regression", "node")
    changed = [g.clone() for g in graphs]
    for g in changed:
        hidden = g.gene_mask if strategy == "gene" else g.mask[:, None].expand_as(g.x)
        g.x = torch.where(hidden, g.x * 7 + 3, g.x)
    out2 = module._common_step(Batch.from_data_list(changed), "regression", "node")
    assert torch.allclose(out2.parts["global"]["bilinear"], out.parts["global"]["bilinear"], atol=1e-6)
    # and the visible values do reach it
    moved = [g.clone() for g in graphs]
    for g in moved:
        hidden = g.gene_mask if strategy == "gene" else g.mask[:, None].expand_as(g.x)
        g.x = torch.where(hidden, g.x, g.x * 2)
    out3 = module._common_step(Batch.from_data_list(moved), "regression", "node")
    assert not torch.allclose(out3.parts["global"]["bilinear"], out.parts["global"]["bilinear"], atol=1e-4)


def test_off_no_new_state_no_random_draw_and_on_the_model_starts_as_without_it():
    on = _model(seed=3)
    off = _model(seed=3, rank=0)
    extra = set(on.module.state_dict()) - set(off.module.state_dict())
    assert extra == {"bilinear_sender", "bilinear_receiver"}
    assert torch.equal(on.module.bilinear_receiver.detach(), torch.zeros(G, R))
    for name in off.module.state_dict():
        assert torch.equal(on.module.state_dict()[name], off.module.state_dict()[name]), name
    assert on.module.return_attention_maps and not off.module.return_attention_maps
    on.module.eval(), off.module.eval()
    batch = Batch.from_data_list([_graph(4, 1)])
    torch.manual_seed(0)
    a = on.module._common_step(batch, "regression", "node")
    torch.manual_seed(0)
    b = off.module._common_step(batch, "regression", "node")
    assert torch.allclose(a.y_pred, b.y_pred, atol=1e-6)
    assert torch.equal(a.parts["global"]["bilinear"], torch.zeros_like(a.y_pred))
    assert b.parts is None
    assert off.module.bilinear_input(batch, None, [list(range(16))], torch.zeros(1, 17, dtype=torch.bool)) is None


def test_the_part_trains_u_v_and_the_attention():
    net = _model()
    module = _stir(net.module)
    module.train()
    out = module._common_step(Batch.from_data_list([_graph(4, 1), _graph(5, 2)]), "regression", "node")
    loss = (out.parts["global"]["bilinear"] ** 2).mean()
    loss.backward()
    assert module.bilinear_receiver.grad.abs().sum() > 0
    assert module.bilinear_sender.grad.abs().sum() > 0
    last = module.transformer_encoder.layers[-1].self_attn
    in_proj = getattr(last, "in_proj_weight", None)
    assert in_proj is not None and in_proj.grad is not None and in_proj.grad.abs().sum() > 0
    # The context part's decoder gets nothing from this loss: the part is its own path.
    assert module.decoder.decoder.weight.grad is None or module.decoder.decoder.weight.grad.abs().sum() == 0


def test_inference_reads_the_same_part_from_the_clean_pass():
    net = _model()
    module = _stir(net.module)
    module.eval()
    graph = _graph(4, 1)
    batch = Batch.from_data_list([graph])
    embedding = module.gex_embedding(batch)
    _, h, padding, index_nodes = module.encode(batch, embedding)
    got = module.bilinear_input(batch, module.last_attention_map(), index_nodes, padding, masked=False)
    assert got.shape == (16, R)
    # by hand: the clean map over the clean genes
    attn = module.last_attention_map().view(1, H, 17, 17)[:, :, :16, :16].mean(1)[0]
    want = attn @ (graph.x @ module.bilinear_sender)
    assert torch.allclose(got, want, atol=1e-5)
    _, h2, padding2, index2 = module.encode(batch, embedding)  # a second pass reads its own map, not a stale one
    assert torch.allclose(module.bilinear_input(batch, module.last_attention_map(), index2, padding2, masked=False), got)


def test_a_step_refuses_without_the_part_and_without_a_map():
    net = _model()
    module = net.module
    batch = Batch.from_data_list([_graph(4, 1)])
    module.bilinear_input = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="bilinear part needs"):
        module._common_step(batch, "regression", "node")
    fresh = _model().module
    with pytest.raises(ValueError, match="kept none"):
        fresh.bilinear_input(batch, None, [list(range(16))], torch.zeros(1, 17, dtype=torch.bool))


def test_the_validator_the_module_refusals_and_the_tag():
    from interscale.config import _validate_bilinear
    from interscale.tl.utils import _decoder_tag

    _validate_bilinear(_cfg())
    with pytest.raises(ValueError, match="rank must be >= 0"):
        _validate_bilinear(_cfg(rank=-1))
    with pytest.raises(ValueError, match="the part is off"):
        _validate_bilinear(_cfg(rank=0, head=1))
    with pytest.raises(ValueError, match="below n_heads"):
        _validate_bilinear(_cfg(head=H))
    with pytest.raises(ValueError, match="attention.kind dense"):
        _validate_bilinear(_cfg(**{"model.global_component.parameters.attention.kind": "performer"}))
    with pytest.raises(ValueError, match="pooling"):
        _validate_bilinear(_cfg(**{"model.global_component.parameters.pooling.kind": "grid"}))
    with pytest.raises(ValueError, match="hierarchical"):
        _validate_bilinear(_cfg(**{"model.global_component.parameters.hierarchy.kind": "coarse_to_fine"}))
    assert _decoder_tag(_cfg()) == f"dec-linear-bil{R}_"
    assert _decoder_tag(_cfg(head=1)) == f"dec-linear-bil{R}h1_"
    assert _decoder_tag(_cfg(rank=0)) == ""

    from interscale.module.global_modules import TransformerNodeEncoderHook

    common = dict(max_seq_len=64, n_heads=H, num_layers=1, dim_feedforward=16, dropout_global=0.0, n_input=G, n_output=G,
                  n_embed=E, decoder_type="linear", dropout_decoder=0.0, decoder_hidden_dims=[], mask_percentage=0.25,
                  mask_strategy="node", bilinear_rank=R, long_range_attention=False)
    with pytest.raises(ValueError, match="built for a GlobalModel"):
        TransformerNodeEncoderHook(type_gex_embedding=None, return_attention_maps=True, **common)
    with pytest.raises(ValueError, match="return_attention_maps"):
        TransformerNodeEncoderHook(type_gex_embedding="Linear", return_attention_maps=False, **common)


def test_the_readouts_on_known_matrices():
    from interscale.evaluation.bilinear import compare_maps, components, gene_map, pair_weights, planted_recovery

    net = _model()
    module = net.module
    names = [f"g{k}" for k in range(G)]
    with torch.no_grad():
        module.bilinear_sender.zero_()
        module.bilinear_receiver.zero_()
        module.bilinear_sender[1, 0] = 1.0  # sender g1 ...
        module.bilinear_receiver[4, 0] = 2.0  # ... moves receiver g4 by 2
        module.bilinear_sender[2, 1] = 0.5
        module.bilinear_receiver[5, 1] = -1.0
    B = gene_map(module, names)
    assert B.shape == (G, G) and B.loc["g4", "g1"] == 2.0 and B.loc["g5", "g2"] == -0.5 and B.loc["g0"].abs().sum() == 0
    rec = planted_recovery(B, [("g1", "g4"), ("g0", "g5"), ("g1", "g0")])
    assert rec.loc[0, "hit"] and rec.loc[0, "rank"] == 1 and rec.loc[0, "share"] == 1.0
    assert not rec.loc[1, "hit"] and rec.loc[1, "rank"] == 2 and rec.loc[1, "top_sender"] == "g2"
    assert not rec.loc[2, "hit"] and rec.loc[2, "rank"] == G  # a zero row
    comp = components(module, names, top=2)
    first = comp[(comp.component == 0) & (comp["rank"] == 0)]
    assert set(first.gene) == {"g4", "g1"} and (first.loading > 0).all()  # the largest component: g1 -> g4, signs positive
    raw = components(module, names, top=1, rotation="raw")
    assert len(raw) == 2 * R
    same = compare_maps(B, B)
    assert same["cosine"] == pytest.approx(1.0) and same["top_overlap"] == 1.0
    flipped = compare_maps(B, -B)
    assert flipped["cosine"] == pytest.approx(-1.0)
    # pair weights: a_ij * (x_j B)[gene]
    module.eval()
    graph = _graph(4, 1)
    batch = Batch.from_data_list([graph])
    _, _, padding, index_nodes = module.encode(batch, module.gex_embedding(batch))
    from interscale.tl.padding import pad_like

    x_pad = pad_like(graph.x, torch.zeros(16, dtype=torch.long), index_nodes, 16)
    w = pair_weights(module, module.last_attention_map(), x_pad, gene=4)
    a = module.last_attention_map().view(1, H, 17, 17)[:, :, :16, :16].mean(1)
    assert torch.allclose(w, a * (2.0 * graph.x[:, 1])[None, None, :], atol=1e-6)
