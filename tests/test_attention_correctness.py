"""Is attention computed correctly, and do its readouts read what they claim? (roadmap.md, T4)

Pinned here:

* **the vendored attention is torch's** -- ``MultiHeadAttentionWithEdits`` against the stock
  ``nn.MultiheadAttention`` with the same weights (outputs, per-head maps, input gradients) under every mask
  form the encoder hands on, and ``CustomTransformerEncoderLayer`` against the stock ``nn.TransformerEncoderLayer``;
  the maps come back per head whatever ``average_attn_weights`` says;
* **the gradient hook** -- the gradient it stores is autograd's gradient of the target with respect to the very
  map the layer used;
* **relevance** (Chefer) -- the heads averaged within each graph (one graph: the old numbers), a given target
  traced instead of the summed output, on the output's device;
* **check_attention** -- zero differences on real models (dense, RoPE, centred queries, a bias), the torch checks
  only where torch can take the inputs, a broken softmax caught;
* **attention_readouts** -- rows and columns mapped to the right cells (a sender planted at the top of its
  receiver's map ranks 1), the readouts' identities for one layer (relevance = rule 5, rollout ∝ attention), no
  RNG consumed, the module left as found.
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Data

from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits

E, H, G = 8, 2, 5
SPACING = 100.0
SIDE = 6


def _pair(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    vendored = MultiHeadAttentionWithEdits(E, H, dropout=dropout)
    stock = torch.nn.MultiheadAttention(E, H, dropout=dropout)
    stock.load_state_dict(vendored.state_dict())
    return vendored, stock


def _masks(n, b):
    blocked = torch.zeros(b * H, n, n, dtype=torch.bool)
    blocked[:, torch.arange(n), torch.arange(n)] = True
    blocked[:, 0, 3] = True
    bias = torch.randn(b * H, n, n, generator=torch.Generator().manual_seed(3)).masked_fill(blocked, float("-inf"))
    padding = torch.zeros(b, n, dtype=torch.bool)
    padding[1, :2] = True
    float_padding = torch.zeros(b, n).masked_fill(padding, float("-inf"))
    return {
        "none": {},
        "blocked": {"attn_mask": blocked},
        "bias": {"attn_mask": bias},
        "padding": {"key_padding_mask": padding},
        "blocked+padding": {"attn_mask": blocked, "key_padding_mask": padding},
        "bias+float padding": {"attn_mask": bias, "key_padding_mask": float_padding},
    }


# --------------------------------------------------------------------------- the vendored attention is torch's


@pytest.mark.parametrize("case", list(_masks(6, 2)))
@pytest.mark.parametrize("training", [False, True])
def test_the_vendored_attention_is_torchs_stock_attention(case, training):
    vendored, stock = _pair()
    vendored.train(training), stock.train(training)
    n, b = 6, 2
    masks = _masks(n, b)[case]
    x = torch.randn(n, b, E, requires_grad=True)
    out, weights = vendored(x, x, x, average_attn_weights=False, **masks)
    out_ref, weights_ref = stock(x, x, x, need_weights=True, average_attn_weights=False, **masks)
    assert torch.allclose(out, out_ref, atol=1e-6)
    assert weights.shape == (b * H, n, n)  # per head, flat over graphs and heads
    assert torch.allclose(weights, weights_ref.flatten(0, 1), atol=1e-6, equal_nan=True)
    (grad,) = torch.autograd.grad((out * torch.arange(E)).sum(), x)
    (grad_ref,) = torch.autograd.grad((out_ref * torch.arange(E)).sum(), x)
    assert torch.allclose(grad, grad_ref, atol=1e-5)


def test_the_vendored_attention_returns_its_maps_per_head_whatever_averaging_asks():
    """The edit keeps the per-head map (what relevance needs); ``average_attn_weights`` is ignored."""
    vendored, stock = _pair()
    x = torch.randn(5, 1, E)
    _, averaged = vendored(x, x, x, average_attn_weights=True)
    _, per_head = stock(x, x, x, need_weights=True, average_attn_weights=False)
    assert torch.allclose(averaged, per_head.flatten(0, 1), atol=1e-6)


def test_queries_keys_and_values_from_different_tokens_are_torchs_too():
    """``attention.center_qk`` hands the attention centred queries and keys and the raw tokens as values."""
    vendored, stock = _pair()
    qk, v = torch.randn(6, 2, E), torch.randn(6, 2, E)
    out, weights = vendored(qk, qk, v, average_attn_weights=False)
    out_ref, weights_ref = stock(qk, qk, v, need_weights=True, average_attn_weights=False)
    assert torch.allclose(out, out_ref, atol=1e-6) and torch.allclose(weights, weights_ref.flatten(0, 1), atol=1e-6)


def test_the_custom_encoder_layer_is_torchs_stock_layer():
    from interscale.module.global_modules.transformer_encoder_layer import CustomTransformerEncoderLayer

    torch.manual_seed(0)
    custom = CustomTransformerEncoderLayer(E, H, dim_feedforward=16, dropout=0.0, norm_first=True).eval()
    stock = torch.nn.TransformerEncoderLayer(E, H, dim_feedforward=16, dropout=0.0, norm_first=True).eval()
    stock.load_state_dict(custom.state_dict())
    n, b = 6, 2
    x = torch.randn(n, b, E)
    for masks in _masks(n, b).values():
        kwargs = {"src_mask": masks.get("attn_mask"), "src_key_padding_mask": masks.get("key_padding_mask")}
        if kwargs["src_mask"] is not None and kwargs["src_key_padding_mask"] is not None:
            # The encoder hands its layers masks of one type; make the padding float like the bias.
            if kwargs["src_mask"].dtype != kwargs["src_key_padding_mask"].dtype:
                continue
        assert torch.allclose(custom(x, **kwargs), stock(x, **kwargs), atol=1e-5, equal_nan=True)


# --------------------------------------------------------------------------- the gradient hook


def _transformer(num_layers=2, rope=False, seed=0):
    """A GlobalModel's transformer on a lattice graph, ready for ``forward``."""
    return _build(_config(num_layers=num_layers, rope=rope), seed=seed).module


def test_the_hook_stores_autograds_gradient_with_respect_to_the_map_used():
    module = _transformer(num_layers=2).eval()
    batch = _batch([_lattice()])
    with module.hooking_attention():
        out = module._common_step(batch, "regression", "node")
    target = (out.y_pred[3] ** 2).sum()
    layers = module.transformer_encoder.layers
    maps = [layer.get_attn_output_weights() for layer in layers]
    expected = torch.autograd.grad(target, maps, retain_graph=True)
    target.backward()
    for layer, grad in zip(layers, expected, strict=True):
        assert layer.get_attn_gradients() is not None
        assert torch.allclose(layer.get_attn_gradients(), grad)


def test_hooking_attention_hooks_masked_passes_only_inside_and_restores_the_flag():
    module = _transformer().eval()
    batch = _batch([_lattice()])
    module._common_step(batch, "regression", "node")
    assert module.transformer_encoder.layers[0].get_attn_output_weights() is None  # a training-style pass keeps no map
    with module.hooking_attention():
        assert module._hook_attention
        module._common_step(batch, "regression", "node")
        assert module.transformer_encoder.layers[0].get_attn_output_weights() is not None
    assert not module._hook_attention
    with pytest.raises(RuntimeError), module.hooking_attention():
        raise RuntimeError
    assert not module._hook_attention


# --------------------------------------------------------------------------- relevance


def test_relevance_averages_heads_within_each_graph():
    """Two graphs in one batch: each graph's relevance is the one it gets alone (the old code mixed them)."""
    from interscale.tl.self_attn_relevance import SelfAttentionRelevance

    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(E, H, dim_feedforward=16, dropout=0.0, norm_first=True)
    from interscale.module.global_modules.transformer_encoder_layer import CustomTransformerEncoderLayer

    custom = CustomTransformerEncoderLayer(E, H, dim_feedforward=16, dropout=0.0, norm_first=True)
    custom.load_state_dict(layer.state_dict())
    encoder = torch.nn.TransformerEncoder(custom, 2).eval()
    for block in encoder.layers:
        block.register_hook = block.keep_maps = True
    relevance = SelfAttentionRelevance(encoder)
    x = torch.randn(5, 2, E)
    both = relevance.generate_relevance(encoder(x))
    assert both.shape == (2, 5, 5)
    for g in range(2):
        alone = relevance.generate_relevance(encoder(x[:, g : g + 1]))
        assert alone.shape == (5, 5)
        assert torch.allclose(both[g], alone, atol=1e-6)


def test_relevance_of_one_graph_without_a_target_is_the_old_formula():
    from interscale.tl.self_attn_relevance import SelfAttentionRelevance

    module = _transformer(num_layers=2).eval()
    batch = _batch([_lattice()])
    with module.hooking_attention():
        out = module._common_step(batch, "regression", "node")
    found = module.self_attn_relevance.generate_relevance(out.view.global_embedding)
    eye = torch.eye(found.shape[0])
    old = eye.clone()
    for layer in module.transformer_encoder.layers:
        cam = SelfAttentionRelevance.avg_heads(layer.get_attn_output_weights(), layer.get_attn_gradients())
        old += cam @ old
    assert torch.allclose(found, old, atol=1e-6)


def test_relevance_traces_the_target_it_is_given():
    module = _transformer(num_layers=1).eval()
    batch = _batch([_lattice()])
    with module.hooking_attention():
        out = module._common_step(batch, "regression", "node")
    embedding = out.view.global_embedding
    summed = module.self_attn_relevance.generate_relevance(embedding)
    own = module.self_attn_relevance.generate_relevance(embedding, target=(out.y_pred[4] ** 2).sum())
    assert not torch.allclose(summed, own)
    layer = module.transformer_encoder.layers[0]
    cam = (layer.get_attn_gradients() * layer.get_attn_output_weights()).clamp(min=0).mean(0)
    assert torch.allclose(own, torch.eye(len(own)) + cam, atol=1e-6)  # one layer: I + rule 5 of this target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_relevance_runs_on_the_outputs_device():
    module = _transformer(num_layers=1).cuda().eval()
    batch = _batch([_lattice()]).to("cuda")
    with module.hooking_attention():
        out = module._common_step(batch, "regression", "node")
    found = module.self_attn_relevance.generate_relevance(out.view.global_embedding)
    assert found.device.type == "cuda"


# --------------------------------------------------------------------------- check_attention


def _lattice(seed=0, hidden=(14, 15, 20)):
    ii, jj = torch.meshgrid(torch.arange(SIDE), torch.arange(SIDE), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = SIDE * SIDE
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) * 3 + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    data.mask[list(hidden)] = True
    data.obs_names = torch.arange(n) + 1  # prepare_geome_dataset's names: the AnnData position + 1
    return data


def _batch(graphs):
    from torch_geometric.data import Batch

    return Batch.from_data_list(graphs)


def _config(model="GlobalModel", *, num_layers=2, rope=False, center=False, bias=False):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    if model == "CombinedModel":
        cfg.model.local_component.name = "GCN"
        cfg = get_local_component_cfg(cfg, "GCN")
        cfg.model.decoder.dual_decoder = True
    cfg.model.n_embed = E
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 64, H, num_layers, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.node = ["sinusoidal"]
    if rope:
        params.pe.rotary = ["rope"]
        params.pe.rope.min_wavelength = 100.0
    if bias:
        params.pe.bias = ["fourier"]
    params.attention.center_qk = center
    cfg.dataset.mask_strategy = "node"
    cfg.dataset.mask_percentage = 0.3
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    return cfg


def _build(cfg, model="GlobalModel", seed=0):
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    n = SIDE * SIDE
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    cls = {"GlobalModel": GlobalModel, "CombinedModel": CombinedModel}[model]
    cls._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return cls(adata, cfg=cfg)


def _check(module):
    from interscale.evaluation.attention_check import check_attention

    return check_attention(module, [_lattice()], prediction_task="regression", prediction_level="node", device="cpu")


@pytest.mark.parametrize(
    "model, options",
    [
        ("GlobalModel", {}),
        ("GlobalModel", {"rope": True}),
        ("GlobalModel", {"center": True}),
        ("GlobalModel", {"bias": True}),
        ("CombinedModel", {"rope": True}),
        ("CombinedModel", {}),
    ],
)
def test_check_attention_finds_no_difference_on_the_real_models(model, options):
    table = _check(_build(_config(model, **options), model).module)
    assert set(table["layer"]) == {0, 1}
    assert (table["max_abs_diff"] <= 1e-5).all(), table
    assert (table["max_abs"] > 0).all()
    checks = set(zip(table["check"], table["quantity"], strict=True))
    assert {("reference", "weights"), ("reference", "output")} <= checks
    # torch's stock modules cannot rotate; its layer cannot centre queries and keys apart from values.
    assert (("torch", "output") in checks) == (not options.get("rope"))
    assert (("torch_layer", "output") in checks) == (not options.get("rope") and not options.get("center"))


@pytest.mark.parametrize("case", list(_masks(6, 2)))
def test_the_reference_is_the_vendored_attention_under_every_mask_form(case):
    """``reference_attention`` itself, on padded batches of two graphs (the real-model checks run one graph)."""
    from interscale.evaluation.attention_check import reference_attention

    vendored, _ = _pair()
    vendored.eval()
    x = torch.randn(6, 2, E)
    masks = _masks(6, 2)[case]
    angles = torch.rand(2 * H, 6, E // H // 2, generator=torch.Generator().manual_seed(5)) * 6
    for rotary in (None, (angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2))):
        out, weights = vendored(x, x, x, average_attn_weights=False, rotary=rotary, **masks)
        ref_out, ref_weights = reference_attention(vendored, x, x, x, rotary=rotary, **masks)
        assert torch.allclose(out, ref_out, atol=1e-6)
        assert torch.allclose(weights, ref_weights, atol=1e-6, equal_nan=True)


def test_check_attention_catches_a_broken_softmax(monkeypatch):
    import interscale.module.global_modules.transformer_utils as utils

    module = _build(_config()).module
    softmax = utils.softmax
    monkeypatch.setattr(utils, "softmax", lambda x, dim: softmax(1.5 * x, dim=dim))
    table = _check(module)
    worst = table.groupby(["check", "quantity"])["max_abs_diff"].max()
    assert worst[("reference", "weights")] > 1e-3 and worst[("torch", "output")] > 1e-3


def test_check_attention_leaves_the_module_and_the_rng_as_found():
    module = _build(_config()).module.train()
    states = torch.get_rng_state(), random.getstate()
    _check(module)
    assert module.training
    assert torch.equal(torch.get_rng_state(), states[0]) and random.getstate() == states[1]
    assert all(len(layer.self_attn._forward_hooks) == 0 for layer in module.transformer_encoder.layers)


# --------------------------------------------------------------------------- attention_readouts


def _readouts(module, graphs, edges, **kwargs):
    from interscale.evaluation.attention_check import attention_readouts

    options = {"queries": 0, "edge_queries": 4, "tolerance_um": 1e9, "radii_um": (150.0,)} | kwargs
    return attention_readouts(
        module, graphs, prediction_task="regression", prediction_level="node", device="cpu",
        spatial_unit_um=1.0, edges=edges, **options,
    )


def _top_key(module, graph, receiver, layer=-1):
    """The visible cell with the most head-mean attention from ``receiver`` in ``layer``, found independently."""
    module.eval()
    with torch.no_grad(), module.recording_attention() as record:
        module._common_step(_batch([graph]), "regression", "node")
    entry = record[0]
    nodes = torch.as_tensor(entry["index_nodes"][0])
    spots = (~entry["src_padding_mask"][0].bool()).nonzero().squeeze(1)
    row = entry["maps"][layer].mean(0)[spots[(nodes == receiver).nonzero().item()]][spots]
    row = row.clone()
    row[graph.mask[nodes]] = -1.0  # hidden cells are no senders
    return int(nodes[row.argmax()])


def test_a_sender_at_the_top_of_its_receivers_map_ranks_first():
    """Rows and columns land on the right cells: with every cell compared, the top key ranks 1 by attention."""
    module = _build(_config(num_layers=1)).module
    graph = _lattice()
    receiver = 14
    sender = _top_key(module, graph, receiver)
    edges = pd.DataFrame({"sender": [sender], "receiver": [receiver], "range_class": ["planted"]})
    tables = _readouts(module, [graph], edges)
    ranks = tables["edges"].set_index(["readout", "range_class"])["matched_rank"]
    assert ranks[("attn_last", "planted")] == 1.0 and ranks[("attn_mean", "planted")] == 1.0
    assert ranks[("rollout", "planted")] == 1.0  # one layer: (A + I) / 2 off the diagonal is A / 2
    from interscale.evaluation.attention_check import READOUTS

    assert set(tables["edges"]["readout"]) == set(READOUTS)


def test_rollout_composes_the_head_mean_maps_with_the_residual_path_last_layer_leftmost():
    from interscale.evaluation.attention_check import _rollout

    maps = torch.softmax(torch.randn(2, H, 4, 4, generator=torch.Generator().manual_seed(0)), dim=-1)
    eye = torch.eye(4, dtype=torch.float64)
    first, second = (0.5 * maps[k].mean(0).double() + 0.5 * eye for k in range(2))
    assert torch.allclose(_rollout(maps), second @ first)
    assert torch.allclose(_rollout(maps).sum(-1), torch.ones(4, dtype=torch.float64))  # rows stay distributions


def test_the_report_keeps_the_number_of_pairs_behind_each_rank():
    from interscale.evaluation.report import compact_tables

    module = _build(_config(num_layers=1)).module
    edges = pd.DataFrame({"sender": [0, 1], "receiver": [14, 20], "range_class": ["a", "b"]})
    tables = _readouts(module, [_lattice()], edges)
    compact = compact_tables({f"readout_{name}": table for name, table in tables.items()})
    ranks, pairs = compact["readout_edges"], compact["readout_pairs"]
    assert ranks.shape == pairs.shape and list(ranks.index) == list(pairs.index)
    assert (pairs.loc[:, ["a", "b"]] >= 1).all().all() and pairs["any"].eq(pairs["a"] + pairs["b"]).all()


def test_one_layer_readouts_agree_where_they_are_one_quantity():
    module = _build(_config(num_layers=1)).module
    edges = pd.DataFrame({"sender": [0], "receiver": [14], "range_class": ["planted"]})
    tables = _readouts(module, [_lattice()], edges, queries=3)
    agree = tables["agreement"].set_index(["readout_a", "readout_b"])["spearman"]
    assert agree[("attn_last", "attn_mean")] == pytest.approx(1.0)
    assert agree[("attn_last", "rollout")] == pytest.approx(1.0)
    assert agree[("grad_attn_last", "relevance")] == pytest.approx(1.0)  # one layer: relevance = I + rule 5
    # Each query's own target, not the summed output (degenerate at init: nearly no gradient reaches the maps).
    assert not agree[("relevance", "relevance_sum")] == pytest.approx(1.0, abs=1e-3)
    span = tables["span"].set_index(["readout", "metric"])["value"]
    assert 0 < span[("attn_last", "span_um")] < SIDE * SPACING * 2
    assert 0 <= span[("relevance", "within_150um")] <= 1


def test_attention_readouts_run_on_the_dual_decoder_model_and_leave_it_as_found():
    module = _build(_config("CombinedModel", rope=True), "CombinedModel").module.train()
    edges = pd.DataFrame({"sender": [0, 1], "receiver": [14, 20], "range_class": ["a", "b"]})
    states = torch.get_rng_state(), random.getstate()
    tables = _readouts(module, [_lattice(), _lattice(seed=1)], edges, queries=2)
    assert torch.equal(torch.get_rng_state(), states[0]) and random.getstate() == states[1]
    assert module.training
    assert all(p.grad is None for p in module.parameters())
    ranks = tables["edges"]
    assert ranks["matched_rank"].between(0, 1).all() and (ranks.groupby("readout")["n_pairs"].max() > 0).all()
    assert len(module.global_module.norm_input._forward_pre_hooks) == 0


def test_attention_readouts_refuse_a_backend_without_dense_maps():
    cfg = _config()
    cfg.model.global_component.parameters.attention.kind = "linear"
    module = _build(cfg).module
    with pytest.warns(RuntimeWarning, match="dense maps"):
        assert _readouts(module, [_lattice()], None) == {}
