"""The sparse-key penalty (``optim.aux_loss_weights.sparse_keys``; ``tl/density.py``; hierarchical attention's cell level).

Pinned here:

* **the density** -- relative to each section's own median, with a radius per section from its nearest-neighbour distance, so two
  sections at different cell sizes read alike; the border and a hole read sparse; bins are ordered and decode to their centres;
* **the penalty is the attention the pass uses** -- equal to ``sum_k a_qk s_k`` over the exported cell-level attention of every
  layer, per query and head, for dense candidates, routed ones, and routed blocks recomputed in the backward pass;
* **it trains attention and nothing else** -- a gradient reaches the attention projections, the forward pass is unchanged by the
  option (bit for bit), and off by default nothing is computed;
* **the term refuses** a model without a hierarchy and a bin column that does not reach the graphs.
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.tl.density import bin_values, density_bins, relative_density, sparsity_weights

E, G, H = 8, 6, 2
SPACING = 100.0
N_BINS = 20


# --------------------------------------------------------------------------- the density


def _grid(side, spacing):
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], 1).astype(float) * spacing


def test_relative_density_is_per_section_and_the_border_and_holes_read_sparse():
    a, b = _grid(30, 10.0), _grid(30, 25.0)  # two sections, cells 2.5 times apart
    coords = np.concatenate([a, b])
    sections = ["a"] * len(a) + ["b"] * len(b)
    rel, radii = relative_density(coords, sections, radius_factor=5.0)
    assert radii["a"] == pytest.approx(50.0) and radii["b"] == pytest.approx(125.0)
    assert np.allclose(rel[: len(a)], rel[len(a) :])  # the same layout reads the same, whatever the spacing
    centre = (np.abs(a - 145.0) < 40).all(1)
    corner = (a == 0).all(1)
    edge = (a[:, 0] == 0) & (a[:, 1] == 140.0)
    assert np.allclose(rel[: len(a)][centre], 1.0, atol=0.05)
    assert 0.25 < rel[: len(a)][corner].item() < 0.4  # about a quarter disc at the corner
    assert 0.45 < rel[: len(a)][edge].item() < 0.65  # about half a disc at the middle of an edge
    # A hole of 90 µm: a cell on its rim loses about a third of its disc (circle overlap), so it reads ~0.8 of the interior.
    keep = np.hypot(*(a - 145.0).T) > 90
    rel_hole, _ = relative_density(a[keep], ["a"] * keep.sum())
    d = np.hypot(*(a[keep] - 145.0).T)
    rim, inner = (d <= 100), (d > 130) & (d <= 140)
    assert rel_hole[rim].mean() < 0.85 * rel_hole[inner].mean()


def test_bins_are_ordered_and_decode_to_their_centres():
    rel = np.array([0.0, 0.049, 0.05, 0.5, 0.999, 1.0, 3.0])
    cat = density_bins(rel, n_bins=N_BINS)
    assert list(cat.codes) == [0, 0, 1, 10, 19, 20, 20]
    assert cat.ordered and len(cat.categories) == N_BINS + 1
    assert list(pd.get_dummies(pd.Series(cat)).columns) == list(cat.categories)  # one-hot columns in code order
    values = bin_values(N_BINS)
    assert values[0] == pytest.approx(0.025) and values[10] == pytest.approx(0.525) and values[-1] == 1.0
    w = sparsity_weights(N_BINS, threshold=0.6)
    assert np.all(np.diff(w) <= 0) and w[-1] == 0.0 and w[0] == pytest.approx(1 - 0.025 / 0.6)
    assert np.all(w[values >= 0.6] == 0.0)
    with pytest.raises(ValueError, match="positive"):
        sparsity_weights(N_BINS, threshold=0.0)


# --------------------------------------------------------------------------- the model


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _lattice(side, seed):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.rand(n, generator=gen) < 0.3  # cells hidden whole
    # density bins: every code from sparse to dense occurs
    codes = torch.randint(0, N_BINS + 1, (n,), generator=gen)
    data.density_bin = torch.nn.functional.one_hot(codes, N_BINS + 1).float()
    return data


def _batch(sides=(6, 5)):
    batch = Batch.from_data_list([_lattice(side, seed=side) for side in sides])
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _config(hierarchy, *, weight=0.1, layers=2):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 200, H, layers, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.bias = ["distance"]
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    params.attention.visible_keys_only = True
    params.hierarchy.kind = "coarse_to_fine"
    for key, value in hierarchy.items():
        node = params.hierarchy
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = value
    cfg.dataset.mask_strategy = "node"
    cfg.dataset.mask_percentage = 0.3
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.extra_obs_keys = ["density_bin"]
    cfg.optim.aux_loss_weights.sparse_keys = weight
    cfg.optim.sparse_keys.n_bins = N_BINS
    cfg.optim.sparse_keys.threshold = 0.6
    return cfg


def _build(cfg, seed=0):
    from interscale.config import _validate
    from interscale.model.global_model import GlobalModel

    _validate(cfg)
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obs["density_bin"] = density_bins(rng.uniform(0, 1.5, 40), N_BINS)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return GlobalModel(adata, cfg=cfg).module


def _step(module, batch):
    random.seed(0)
    return module._common_step(batch, "regression", "node")


def _expected(module, batch):
    """``sum_k a_qk s_k`` over every exported cell-level row, divided by the rows (queries x heads x layers x graphs)."""
    from interscale.evaluation.hierarchy_readouts import cell_rows

    weights_of_code = torch.as_tensor(sparsity_weights(N_BINS, 0.6), dtype=torch.float32)
    total, rows = 0.0, 0
    with torch.no_grad(), module.recording_hierarchy() as record:
        _step(module, batch)
    graphs = batch.to_data_list()
    for i, entry in enumerate(record):  # layer by layer, one entry per graph
        graph = graphs[i % len(graphs)]
        s_graph = weights_of_code[graph.density_bin.argmax(1)]
        nodes = torch.arange(graph.num_nodes) if entry["nodes"] is None else torch.as_tensor(entry["nodes"])
        s = s_graph[nodes]
        keys, weights = cell_rows(entry)  # [n, K], [H, n, K]; key n is CLS, -1 padding
        n = len(nodes)
        is_cell = (keys >= 0) & (keys < n)
        s_keys = torch.where(is_cell, s[keys.clamp(0, n - 1)], torch.zeros(()))
        total += float((weights * s_keys[None]).sum())
        rows += weights.shape[0] * n
    return total / rows


ROUTE_ALL = {"candidates.mode": "routed", "candidates.topk": [100], "candidates.cap_per_query": 4096}
SETTINGS = {
    "dense": {"levels": [2, 6]},
    "routed": {"levels": [2, 6], **ROUTE_ALL},
    "routed, blocks recomputed": {"levels": [2, 6], **ROUTE_ALL, "candidates.chunk_pairs": 300},
}


@pytest.mark.parametrize("name", list(SETTINGS))
def test_the_penalty_is_the_sparsity_weighted_key_mass_of_the_attention_the_pass_uses(name):
    module = _stir(_build(_config(SETTINGS[name])))
    batch = _batch()
    module.eval()
    expected = _expected(module, batch)
    # The pass that trains: in training mode (dropout 0), so routed blocks are recomputed in the backward pass when chunked.
    module.train()
    out = _step(module, batch)
    penalty = out.view.attn_penalty
    assert penalty is not None and penalty.requires_grad
    assert 0.0 < float(penalty) < 1.0
    assert float(penalty) == pytest.approx(expected, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("name", list(SETTINGS))
def test_the_penalty_reaches_the_attention_and_leaves_the_forward_pass_unchanged(name):
    on = _stir(_build(_config(SETTINGS[name])))
    off = _build(_config(SETTINGS[name], weight=0.0))
    off.load_state_dict(on.state_dict())
    batch = _batch()
    on.train(), off.train()
    out_on, out_off = _step(on, batch), _step(off, batch)
    assert out_off.view.attn_penalty is None  # off: nothing computed
    assert torch.equal(out_on.y_pred, out_off.y_pred)
    on.zero_grad(set_to_none=True)
    out_on.view.attn_penalty.backward()
    grads = {n: p.grad for n, p in on.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any("in_proj" in n for n in grads), sorted(grads)
    assert not any("decoder" in n for n in grads)  # attention only


def test_the_term_weighs_the_penalty_into_the_total_and_refuses_what_it_cannot_read():
    from interscale.train.aux_losses import build_aux_losses

    cfg = _config(SETTINGS["routed"], weight=0.25)
    module = _stir(_build(cfg))
    module.train()
    out = _step(module, _batch())
    composite = build_aux_losses(cfg, module)
    total, reported = composite(out, None)
    assert float(reported["sparse_keys"]) == pytest.approx(float(out.view.attn_penalty))
    assert float(total) == pytest.approx(0.25 * float(out.view.attn_penalty))

    no_hier = _config({})
    no_hier.model.global_component.parameters.hierarchy.kind = "none"
    with pytest.raises(ValueError, match="hierarchical attention"):
        build_aux_losses(no_hier, None)
    missing = _config(SETTINGS["routed"])
    missing.dataset.extra_obs_keys = []
    with pytest.raises(ValueError, match="extra_obs_keys"):
        build_aux_losses(missing, None)


def test_a_cell_without_a_bin_weighs_nothing_and_too_many_bins_are_refused():
    module = _stir(_build(_config(SETTINGS["dense"])))
    batch = _batch()
    batch.density_bin = torch.zeros_like(batch.density_bin)  # no cell has a bin
    module.train()
    assert float(_step(module, batch).view.attn_penalty) == 0.0
    wide = _batch()
    wide.density_bin = torch.nn.functional.one_hot(torch.full((wide.num_nodes,), N_BINS + 3), N_BINS + 4).float()
    with pytest.raises(ValueError, match="n_bins"):
        _step(module, wide)
