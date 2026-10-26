"""Hierarchical coarse-to-fine attention (``module/global_modules/hierarchy.py``, ``hierarchical_attention.py``;
``.claude/hierarchical_attention_plan.md``).

Pinned here:

* **the hierarchy** (plan tests 1, 10-11) -- every cell and cluster has exactly one parent, children partition each parent,
  kdtree clusters are balanced, the build is deterministic and cached, a patch top level puts every cell in one patch and
  merges small ones, the shifted twin has other boundaries;
* **the must-pass regression** (plan test 2) -- with prior soft, lambda 0, dense candidates and singleton clusters (or no
  coarse level) the hierarchical model is the dense transformer: same weights at one seed, same predictions and gradients;
* **the priors** (plan tests 3-4) -- the soft prior with size correction sums to 1 and is what a content-free level reads; the
  factorized modes give each child cluster exactly its parent's weight (renormalised over the selected parents when routed);
* **routing** (plan test 5) -- with every parent selected the routed model is the dense one in every prior mode, blocks and
  recomputation included; with pruning each query reads only the children of its candidates, and no tensor over the cells'
  pairs is ever formed;
* **invariances** (plan test 7) -- the output does not depend on the order of the cells;
* **no leakage** -- no hidden value reaches a prediction (cell masking, programme masking), a visible one does;
* **the switch** -- off by default, the config refuses what the hierarchy cannot carry, checkpoint names tell arms apart,
  exploration draws nothing from the global generator, a model trains end to end through ``train()``.
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch.utils._python_dispatch import TorchDispatchMode
from torch_geometric.data import Batch, Data

from interscale.module.global_modules.hierarchical_attention import SparseRows
from interscale.module.global_modules.hierarchy import HierarchyCache, build_hierarchy, kd_parts, patch_labels

E, G, H = 8, 6, 2
SPACING = 100.0


def _stir(module, seed=0, scale=0.1):
    """Every parameter moved off its initialisation by seeded noise (tests-stir-the-init)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _grid(side, spacing=SPACING):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * spacing


def _lattice(side, seed, programme=True):
    pos = _grid(side)
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.ones(n, dtype=torch.bool) if programme else torch.zeros(n, dtype=torch.bool)
    if programme:
        data.gene_mask = torch.zeros(n, G, dtype=torch.bool)
        data.gene_mask[:, [0, 1]] = True  # programme masking: a gene set hidden in every cell
    return data


def _batch(sides=(6, 5), programme=True):
    batch = Batch.from_data_list([_lattice(side, seed=side, programme=programme) for side in sides])
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _config(hierarchy=None, *, bias=("distance",), strategy="gene", dropout=0.0, max_seq_len=200, layers=2):
    """The windowed global-only model of the window sweep at toy width (no RoPE: the hierarchy does not carry it), with
    ``hierarchy`` (a dict of settings, dotted keys for nested ones) switched on when given."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    cfg.model.decoder.own_cell_skip = True
    cfg.model.decoder.own_cell_skip_input = "raw"
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = max_seq_len, H, layers, dropout
    params.type_gex_embedding = "Linear"
    params.attention.center_qk = True
    params.pe.bias = list(bias)
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    params.pe.fourier.max_wavelength = 2000.0
    params.pe.fourier.isotropic = False
    if hierarchy is not None:
        params.hierarchy.kind = "coarse_to_fine"
        for key, value in hierarchy.items():
            node = params.hierarchy
            *path, last = key.split(".")
            for part in path:
                node = node[part]
            node[last] = value
    cfg.dataset.mask_strategy = strategy
    cfg.dataset.mask_percentage = 1.0 if strategy == "gene" else 0.3
    if strategy == "gene":
        cfg.dataset.mask_genes = ["g0", "g1"]
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
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
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return GlobalModel(adata, cfg=cfg).module


def _run(module, batch, weight=None):
    """One masked pass of the model's own step, then the backward of a fixed weighting of its prediction."""
    random.seed(0)
    module.zero_grad(set_to_none=True)
    out = module._common_step(batch, "regression", "node")
    weight = torch.randn(out.y_pred.shape, generator=torch.Generator().manual_seed(3)) if weight is None else weight
    (out.y_pred * weight).sum().backward()
    return out.y_pred.detach(), {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


def _record(module, batch):
    """Every layer's export of one masked pass (no gradient)."""
    transformer = module
    with torch.no_grad(), transformer.recording_hierarchy() as record:
        random.seed(0)
        module._common_step(batch, "regression", "node")
    return record


SOFT_ZERO = {"prior_mode": "soft", "lambda_init": 0.0, "lambda_learnable": "fixed"}


# --------------------------------------------------------------------------- the hierarchy


@pytest.mark.parametrize("builder", ["kdtree", "random"])
@pytest.mark.parametrize("top", ["tree", "patches"])
def test_every_cell_and_cluster_has_one_parent_and_children_partition_each_parent(builder, top):
    pos = _grid(12).numpy() + np.random.default_rng(0).normal(scale=5.0, size=(144, 2))
    hier = build_hierarchy(pos, levels=(4, 16), builder=builder, top_mode=top, neighbors=3)
    assert hier.n_levels == 2 and hier.counts[-1] == 144
    for level in range(1, hier.n_levels + 1):
        up = hier.up[level]
        assert len(up) == hier.counts[level] and up.min() >= 0 and up.max() < hier.counts[level - 1]
        table = hier.children[level - 1]
        listed = table[table >= 0]
        # Every child is listed exactly once, under its own parent.
        assert torch.equal(listed.sort().values, torch.arange(hier.counts[level]))
        rows = torch.arange(hier.counts[level - 1])[:, None].expand_as(table)[table >= 0]
        assert torch.equal(up[listed], rows)
        assert torch.equal(hier.n_children[level - 1], (table >= 0).sum(1))
    # Nested: a cell's cluster at level 1 lies in its cluster at level 0.
    assert torch.equal(hier.up[1][hier.assign[1]], hier.assign[0])
    # Neighbour lists start with the cluster itself.
    for level in range(hier.n_levels):
        assert torch.equal(hier.neighbours[level][:, 0], torch.arange(hier.counts[level]))


def test_kdtree_clusters_are_balanced_and_the_build_is_deterministic():
    pos = np.random.default_rng(1).uniform(0, 1000, size=(1000, 2))
    a = build_hierarchy(pos, levels=(8, 64), builder="kdtree")
    b = build_hierarchy(pos, levels=(8, 64), builder="kdtree")
    for level in range(2):
        assert torch.equal(a.assign[level], b.assign[level])
        sizes = a.sizes[level]
        assert sizes.max() - sizes.min() <= 2  # balanced by cell count
    assert a.counts[:2] == [8, 64]
    # The same holds for cells per cluster, and it gives the same partition as the matching count.
    c = build_hierarchy(pos, cells_per_cluster=(125.0, 15.625), builder="kdtree")
    assert torch.equal(c.assign[0], a.assign[0])
    r1 = build_hierarchy(pos, levels=(8,), builder="random", seed=3)
    r2 = build_hierarchy(pos, levels=(8,), builder="random", seed=3)
    assert torch.equal(r1.assign[0], r2.assign[0])
    # Random clusters are not spatial: their spread is about the slide's, a kdtree cluster's far below it.
    assert float(r1.spread[0].median()) > 2.5 * float(a.spread[0].median())


def test_a_kdtree_split_is_balanced_across_the_longer_side():
    pos = np.stack([np.arange(100.0) * 3.0, np.zeros(100)], 1)  # a line along x
    parts = kd_parts(pos, np.arange(100), 4)
    assert [len(p) for p in parts] == [25, 25, 25, 25]
    assert [int(p.min()) for p in parts] == [0, 25, 50, 75]  # in order along the line


def test_patches_put_every_cell_in_one_patch_and_merge_small_ones():
    rng = np.random.default_rng(2)
    pos = np.concatenate([rng.uniform(0, 1000, size=(300, 2)), [[2000.0, 2000.0], [2010.0, 1990.0]]])
    labels = patch_labels(pos, size_um=250.0, n_patches=0, offset=0.0, min_cells=5)
    assert labels.shape == (302,) and labels.min() == 0
    counts = np.bincount(labels)
    assert counts.min() >= 5  # the two far cells joined the nearest larger patch
    hier = build_hierarchy(pos, levels=(16, 60), top_mode="patches", patch_size_um=250.0, min_cells=5)
    assert hier.sizes[0].min() >= 5


def test_the_shifted_twin_has_its_boundaries_elsewhere():
    pos = _grid(16).numpy()
    plain = build_hierarchy(pos, levels=(16,))
    shifted = build_hierarchy(pos, levels=(16,), shift=True)
    # Pairs of horizontal neighbours split by the plain partition: the shifted one keeps many of them together.
    left, right = np.arange(256)[np.arange(256) % 16 < 15], np.arange(256)[np.arange(256) % 16 < 15] + 1
    cut_plain = plain.assign[0][left] != plain.assign[0][right]
    cut_shifted = shifted.assign[0][left] != shifted.assign[0][right]
    assert cut_plain.any() and (cut_plain & ~cut_shifted).sum() > 0.5 * cut_plain.sum()
    assert shifted.sizes[0].max() - shifted.sizes[0].min() <= 2


def test_the_cache_builds_a_slide_once():
    cache = HierarchyCache()
    pos = _grid(6)
    first = cache.get(pos, levels=(4,))
    assert cache.get(pos.clone(), levels=(4,)) is first
    assert cache.get(pos, levels=(9,)) is not first
    assert cache.get(pos + 1.0, levels=(4,)) is not first
    assert first.build_seconds >= 0 and "l0_cells_median" in first.stats()


# --------------------------------------------------------------------------- the must-pass regression


@pytest.mark.parametrize(
    "hierarchy",
    [
        {**SOFT_ZERO, "cells_per_cluster": [1.0]},
        {
            **SOFT_ZERO,
            "cells_per_cluster": [1.0],
            "candidates.mode": "routed",
            "candidates.topk": [100],
            "candidates.cap_per_query": 4096,
        },
        {"prior_mode": "none"},  # no coarse level at all
    ],
    ids=["singletons, dense", "singletons, routed over every parent", "no coarse level"],
)
@pytest.mark.parametrize("bias", [("distance",), ("distance", "fourier")])
def test_with_lambda_zero_and_singletons_the_hierarchical_model_is_the_dense_one(hierarchy, bias):
    batch = _batch()
    dense = _build(_config(bias=bias))
    hier = _build(_config(hierarchy, bias=bias))
    shared = dense.state_dict()
    assert all(torch.equal(value, hier.state_dict()[name]) for name, value in shared.items())
    _stir(dense)
    missing, unexpected = hier.load_state_dict(dense.state_dict(), strict=False)
    assert not unexpected
    dense.eval(), hier.eval()
    pred_dense, grads_dense = _run(dense, batch)
    pred_hier, grads_hier = _run(hier, batch)
    assert torch.allclose(pred_hier, pred_dense, atol=1e-5)
    for name, value in grads_dense.items():
        assert torch.allclose(grads_hier[name], value, atol=1e-5, rtol=1e-4), name


def test_off_by_default_the_model_is_unchanged():
    from interscale.module.global_modules.transformer_encoder_layer import CustomTransformerEncoderLayer

    module = _build(_config())
    assert module.hierarchy is None
    assert all(type(layer) is CustomTransformerEncoderLayer for layer in module.transformer_encoder.layers)
    assert not any("hier" in name or "pool_" in name for name in module.state_dict())


# --------------------------------------------------------------------------- the priors


def _content_free(module):
    """Queries of zero (a level's logits are its biases and prior alone)."""
    with torch.no_grad():
        for layer in module.transformer_encoder.layers:
            layer.self_attn.in_proj_weight[:E] = 0.0
            layer.self_attn.in_proj_bias[:E] = 0.0
    return module


def _prior(a_parent, hier, level, size_correction=True):
    """``P_ij = A_{p(i), p(j)} / n_{p(j)}`` over a level's tokens and CLS, from the parent level's attention ``[H, ., n_par + 1]``."""
    n_par = hier.counts[level - 1]
    pq = hier.up[level]
    pk = torch.cat([pq, pq.new_tensor([n_par])])
    sizes = torch.cat([hier.n_children[level - 1], hier.n_children[level - 1].new_ones(1)]).float()
    p = a_parent[:, pq][:, :, pk]
    return p / sizes[pk] if size_correction else p


def test_the_soft_prior_sums_to_one_and_a_content_free_level_reads_it():
    batch = _batch((8,))
    cfg = _config({"prior_mode": "soft", "levels": [4, 16], "lambda_init": 1.0, "lambda_learnable": "fixed"}, bias=())
    module = _content_free(_stir(_build(cfg))).eval()
    _content_free(module)
    entry = _record(module, batch)[0]
    hier = entry["hierarchy"]
    a0, a1 = entry["attention"][0], entry["attention"][1]
    p = _prior(a0, hier, 1)
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[:2]), atol=1e-5)
    expected = (p + 1e-6) / (p + 1e-6).sum(-1, keepdim=True)
    assert torch.allclose(a1, expected, atol=1e-5)


def _cluster_mass(weights, keys_parent, n_par):
    """``[H, Q, n_par + 1]``: each query's weight summed over the keys of each parent cluster (CLS last)."""
    out = torch.zeros(weights.shape[0], weights.shape[1], n_par + 1)
    return out.scatter_add(-1, keys_parent.expand_as(weights), weights)


@pytest.mark.parametrize("level", [1, 2])
def test_factorized_gives_each_child_cluster_exactly_its_parents_weight(level):
    batch = _batch((8,))
    module = _stir(_build(_config({"prior_mode": "factorized", "levels": [4, 16]}))).eval()
    entry = _record(module, batch)[0]
    hier = entry["hierarchy"]
    a_par, a = entry["attention"][level - 1], entry["attention"][level]
    n_par = hier.counts[level - 1]
    assert torch.allclose(a.sum(-1), torch.ones(a.shape[:2]), atol=1e-5)
    pk = torch.cat([hier.up[level], hier.up[level].new_tensor([n_par])])
    mass = _cluster_mass(a, pk, n_par)
    parent_rows = a_par[:, hier.up[level]]  # [H, n, n_par + 1]
    assert torch.allclose(mass, parent_rows, atol=1e-5)


def test_factorized_percell_rows_sum_to_one_and_the_direction_prior_moves_them():
    batch = _batch((8,))
    plain = _stir(_build(_config({"prior_mode": "factorized_percell", "levels": [4, 16]}))).eval()
    entry = _record(plain, batch)[0]
    assert torch.allclose(entry["attention"][2].sum(-1), torch.ones(entry["attention"][2].shape[:2]), atol=1e-5)
    directed = _build(_config({"prior_mode": "soft", "levels": [4, 16], "direction_prior": True, "mu_init": 2.0}))
    _stir(directed).eval()
    with_mu = _record(directed, batch)[0]["attention"][2]
    with torch.no_grad():
        for layer in directed.transformer_encoder.layers:
            layer.prior_mu.zero_()
    without = _record(directed, batch)[0]["attention"][2]
    assert not torch.allclose(with_mu, without, atol=1e-4)


# --------------------------------------------------------------------------- routing


ROUTE_ALL = {"candidates.mode": "routed", "candidates.topk": [100], "candidates.cap_per_query": 4096}


@pytest.mark.parametrize(
    "mode",
    [
        {"prior_mode": "none", "readout": "concat"},
        {"prior_mode": "soft"},
        {"prior_mode": "soft", "direction_prior": True},
        {"prior_mode": "soft", "size_correction": False, "readout": "transformer"},
        {"prior_mode": "factorized"},
        {"prior_mode": "factorized_percell"},
        {
            "prior_mode": "soft",
            "pooling": "attn_pool",
            "pool_features": ["count", "spread"],
            "share_level_params": False,
        },
        {"prior_mode": "soft", "shift": True, "coarse_bias": "own", "prior_refresh": "once"},
    ],
    ids=["none", "soft", "soft+direction", "soft, no size correction", "factorized", "percell", "attn_pool", "shift"],
)
@pytest.mark.parametrize("chunk_pairs", [2**24, 300], ids=["one block", "blocks"])
def test_routed_over_every_parent_is_the_dense_model(mode, chunk_pairs):
    batch = _batch()
    settings = {**mode, "levels": [2, 6]}
    dense = _stir(_build(_config(settings)))
    routed = _build(_config({**settings, **ROUTE_ALL, "candidates.chunk_pairs": chunk_pairs}))
    routed.load_state_dict(dense.state_dict())
    # Training mode (dropout 0): the routed cells are recomputed block by block in the backward pass.
    dense.train(), routed.train()
    pred_dense, grads_dense = _run(dense, batch)
    pred_routed, grads_routed = _run(routed, batch)
    assert torch.allclose(pred_routed, pred_dense, atol=1e-5)
    assert set(grads_routed) == set(grads_dense)
    # The direction prior adds a few float32 roundings (in float64 the two agree to 1e-12, checked 2026-10-26).
    atol = 1e-4 if mode.get("direction_prior") else 1e-5
    for name, value in grads_dense.items():
        assert torch.allclose(grads_routed[name], value, atol=atol, rtol=1e-4), name


def test_a_pruned_query_reads_only_the_children_of_its_candidates_and_factorized_renormalises_over_them():
    batch = _batch((12,))
    settings = {
        "prior_mode": "factorized",
        "levels": [9],
        "neighbors": 0,
        "candidates.mode": "routed",
        "candidates.topk": [1],
    }
    module = _stir(_build(_config(settings))).eval()
    entry = _record(module, batch)[0]
    hier, a0, rows = entry["hierarchy"], entry["attention"][0], entry["attention"][1]
    assert isinstance(rows, SparseRows)
    n, n_par = hier.counts[1], hier.counts[0]
    keys, weights = rows.rows(torch.arange(n))
    cells = (keys >= 0) & (keys < n)
    parent_of_key = torch.where(cells, hier.up[1][keys.clamp(0, n - 1)], torch.full_like(keys, n_par))
    # Candidates: the own parent and its single top target, so at most two clusters' cells per query.
    for i in range(n):
        allowed = {int(hier.up[1][i]), int(a0[:, hier.up[1][i], :n_par].amax(0).argmax())}
        read = set(parent_of_key[i][cells[i] & (weights[:, i] > 0).any(0)].tolist())
        assert read <= allowed
    mass = _cluster_mass(weights, parent_of_key, n_par)  # [H, n, n_par + 1]
    for i in range(0, n, 7):
        chosen = sorted(set(parent_of_key[i][cells[i]].tolist()))
        parent = int(hier.up[1][i])
        row = a0[:, parent]  # [H, n_par + 1]
        total = row[:, chosen].sum(-1) + row[:, n_par]
        for c in chosen:
            assert torch.allclose(mass[:, i, c], row[:, c] / total, atol=1e-5)


class _Shapes(TorchDispatchMode):
    """Every tensor any op returns, forward and backward, by shape."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.shapes.append(tuple(t.shape))
        return out


def test_a_routed_pass_forms_no_matrix_over_the_cells_pairs():
    batch = _batch((13,))  # 169 cells
    settings = {
        "prior_mode": "factorized_percell",
        "levels": [4, 16],
        "neighbors": 1,
        "direction_prior": True,
        "candidates.mode": "routed",
        "candidates.topk": [1],
        "candidates.chunk_pairs": 2000,
    }
    module = _stir(_build(_config(settings)))
    module.train()
    with _Shapes() as seen:
        _run(module, batch)
    pairs = [s for s in seen.shapes if sum(d in (169, 170) for d in s) >= 2]
    assert not pairs, pairs[:5]


def test_exploration_draws_nothing_from_the_global_generator():
    batch = _batch()
    settings = {
        "prior_mode": "soft",
        "levels": [2, 6],
        "candidates.mode": "routed",
        "candidates.topk": [1],
        "candidates.explore_n": 2,
        "candidates.explore_strategy": "prior_sampled",
    }
    module = _stir(_build(_config(settings))).train()
    state = torch.random.get_rng_state()
    _run(module, batch)
    assert torch.equal(torch.random.get_rng_state(), state)


# --------------------------------------------------------------------------- invariances, leakage


def test_the_output_does_not_depend_on_the_order_of_the_cells():
    graph = _lattice(7, seed=1)
    perm = torch.randperm(49, generator=torch.Generator().manual_seed(0))
    shuffled = Data(x=graph.x[perm], edge_index=torch.argsort(perm)[graph.edge_index], pos=graph.pos[perm])
    shuffled.mask, shuffled.gene_mask = graph.mask[perm], graph.gene_mask[perm]
    module = _stir(_build(_config({"prior_mode": "factorized_percell", "levels": [3, 9], "readout": "concat"}))).eval()
    one, other = Batch.from_data_list([graph]), Batch.from_data_list([shuffled])
    one.obs_names, other.obs_names = torch.arange(49), perm
    pred, _ = _run(module, one)
    pred_shuffled, _ = _run(module, other)
    assert torch.allclose(pred_shuffled, pred[perm], atol=1e-5)


@pytest.mark.parametrize("strategy", ["node", "gene"])
@pytest.mark.parametrize("candidates", ["dense", "routed"])
def test_no_hidden_value_reaches_a_prediction_and_a_visible_one_does(strategy, candidates):
    batch = _batch((6,), programme=strategy == "gene")
    n = batch.num_nodes
    if strategy == "node":
        batch.mask = torch.zeros(n, dtype=torch.bool)
        batch.mask[:12] = True  # two rows hidden whole
    settings = {
        "prior_mode": "soft",
        "levels": [2, 6],
        "readout": "transformer",
        "pooling": "attn_pool",
        "candidates.mode": candidates,
        "candidates.topk": [1],
    }
    module = _stir(_build(_config(settings, strategy=strategy))).eval()

    def predict(b):
        with torch.no_grad():
            random.seed(0)
            return module._common_step(b, "regression", "node").y_pred

    before = predict(batch)
    hidden = batch.clone()
    hidden.x = batch.x.clone()
    if strategy == "node":
        hidden.x[:12] += torch.rand(12, G) * 50.0
    else:
        hidden.x[:, [0, 1]] = torch.rand(n, 2) * 50.0
    assert torch.equal(predict(hidden), before)
    visible = batch.clone()
    visible.x = batch.x.clone()
    visible.x[30, 3] += 50.0
    after = predict(visible)
    assert not torch.allclose(after[:12], before[:12])


# --------------------------------------------------------------------------- the switch


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"pe.rotary": ["rope"]}, "rope"),
        ({"attention.kind": "fused"}, "attention.kind dense"),
        ({"pe.bias": ["spectral"]}, "spectral"),
        ({"hierarchy.levels": [4], "hierarchy.cells_per_cluster": [10.0]}, "not both"),
        ({"hierarchy.levels": [4], "hierarchy.prior_mode": "none"}, "readout"),
        ({"hierarchy.levels": [4], "hierarchy.prior_mode": "sharp"}, "prior_mode"),
        ({"hierarchy.levels": [4], "hierarchy.candidates.mode": "routed", "hierarchy.candidates.topk": [0]}, "topk"),
        ({"hierarchy.levels": [4], "pooling.kind": "grid", "pooling.size_um": 100.0}, "pooling"),
    ],
)
def test_the_config_refuses_what_the_hierarchy_cannot_carry(change, match):
    from interscale.config import _validate

    cfg = _config({"prior_mode": "soft", "levels": [4]})
    params = cfg.model.global_component.parameters
    for key, value in change.items():
        node = params
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = value
    with pytest.raises(ValueError, match=match):
        _validate(cfg)


def test_checkpoint_names_tell_the_arms_apart():
    from interscale.tl.utils import get_model_filename_prefix

    arms = [
        None,
        {"prior_mode": "soft", "levels": [16, 256]},
        {"prior_mode": "soft", "levels": [16]},
        {"prior_mode": "soft", "cells_per_cluster": [156.0]},
        {"prior_mode": "factorized", "levels": [16, 256]},
        {"prior_mode": "factorized_percell", "levels": [16, 256]},
        {"prior_mode": "soft", "levels": [16, 256], "direction_prior": True},
        {"prior_mode": "none", "levels": [16, 256], "readout": "concat"},
        {"prior_mode": "soft", "levels": [16, 256], "builder": "random"},
        {"prior_mode": "soft", "levels": [16, 256], "pooling": "attn_pool"},
        {"prior_mode": "soft", "levels": [16, 256], "shift": True},
        {"prior_mode": "soft", "levels": [16, 256], "top_level.mode": "patches"},
        {"prior_mode": "soft", "levels": [16, 256], "candidates.mode": "routed", "candidates.topk": [4]},
        {"prior_mode": "soft", "levels": [16, 256], "candidates.mode": "routed", "candidates.topk": [8]},
        {
            "prior_mode": "soft",
            "levels": [16, 256],
            "candidates.mode": "routed",
            "candidates.topk": [8],
            "candidates.explore_n": 2,
        },
        {"prior_mode": "soft", "levels": [16, 256], "lambda_init": 2.0},
    ]
    names = [get_model_filename_prefix(_config(arm), local_component=False, global_component=True) for arm in arms]
    assert len(set(names)) == len(names)
    assert "hier-" not in names[0]


def test_a_hierarchical_model_trains_end_to_end_under_programme_masking(tmp_path):
    """Through ``train()``: masks drawn and redrawn, routed attention with exploration, checkpoint saved and reloadable."""
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    n = 60 * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, 60))
    adata.obs["split"] = pd.Categorical(
        [{"s1": "train", "s2": "train", "s3": "val", "s4": "test"}[s] for s in adata.obs["sample"]]
    )
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "hier.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "hier.yaml"
    path.write_text(
        f"model:\n  n_embed: {E}\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 64\n      num_layers: 2\n      n_heads: 2\n      type_gex_embedding: Linear\n"
        "      attention:\n        center_qk: True\n"
        "      pe:\n        bias: [distance]\n        distance:\n          kind: scales\n"
        "          scale_min_um: 50.0\n          scale_max_um: 500.0\n"
        "      hierarchy:\n        kind: coarse_to_fine\n        levels: [3, 12]\n        prior_mode: factorized\n"
        "        readout: concat\n        pooling: attn_pool\n        candidates:\n          mode: routed\n"
        "          topk: [1]\n          explore_n: 1\n"
        f"  decoder:\n    type: linear\n    own_cell_skip: True\n    own_cell_skip_input: raw\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 3\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: hiertest\n  prediction_task: regression\n  prediction_level: node\n"
        "  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: gene\n  mask_percentage: 1.0\n"
        "  mask_genes: ['g2', 'g5']\n  spatial_key: spatial\n  batch_size: 1\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 120\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    net = GlobalModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg,
        num_workers=0,
        batch_size=1,
        mask_percentage=1.0,
        mask_strategy="gene",
        learning_type="node",
        **gene_set_mask_kwargs(cfg, adata),
    )
    net.train(max_epochs=3, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    losses = [row["val_loss"] for row in net.history_.history if "val_loss" in row]
    assert losses and all(np.isfinite(losses))
    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert any("hier-kdtree-l3x12-fac" in name for name in saved), saved
    layer = net.module.transformer_encoder.layers[0]
    assert layer.readout_weight.abs().sum() > 0 and layer.pool_query.abs().sum() > 0  # both trained from zero


# --------------------------------------------------------------------------- the readouts (plan T8, T10; tests 8-9)


def test_read_vec_points_to_the_sender_and_splits_exactly_into_coarse_and_refinement():
    from interscale.evaluation.hierarchy_readouts import decompose, read_vectors

    pos = _grid(6)
    n = len(pos)
    keys = torch.arange(n + 1).expand(n, -1)
    weights = torch.zeros(1, n, n + 1)
    sender = 20
    weights[0, :, sender] = 1.0  # every cell reads only the planted sender
    weights[0, sender, sender], weights[0, sender, n] = 0.0, 1.0  # the sender itself reads CLS
    read, mass = read_vectors(keys, weights, pos)
    expected = pos[sender] - pos
    expected[sender] = 0.0
    assert torch.allclose(read[0], expected)
    assert torch.allclose(mass[0, sender], torch.tensor(0.0))
    # Mixed weights over a hierarchy: coarse + refinement is read_vec, exactly.
    hier = build_hierarchy(pos.numpy(), levels=(4,))
    rand = torch.rand(2, n, n + 1, generator=torch.Generator().manual_seed(0)).softmax(-1)
    read, _ = read_vectors(keys, rand, pos)
    coarse, refine = decompose(keys, rand, pos, hier.up[-1], hier.centroid[0])
    assert torch.allclose(coarse + refine, read, atol=1e-5)


@pytest.mark.parametrize("candidates", ["dense", "routed"])
def test_the_readouts_run_on_a_masked_pass_and_find_a_sender_the_attention_reads(candidates, tmp_path):
    from interscale.evaluation.hierarchy_readouts import hierarchy_readouts, plot_fields

    graph = _lattice(8, seed=2)
    settings = {"prior_mode": "factorized", "levels": [4, 16], "candidates.mode": candidates, "candidates.topk": [2]}
    module = _stir(_build(_config(settings)))
    edges = pd.DataFrame(
        {"sender": [9, 9, 50], "receiver": [10, 18, 41], "weight": [1.0, 1.0, 1.0], "range_class": ["short"] * 3}
    )
    graph.obs_names = torch.arange(64) + 1  # prepare_geome_dataset names nodes by AnnData position + 1
    state = torch.random.get_rng_state()
    tables = hierarchy_readouts(
        module,
        [graph],
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        edges=edges,
        distance_bins_um=[0, 150, 400, 2000],
        export_dir=tmp_path,
        graph_names=["slide"],
    )
    assert torch.equal(torch.random.get_rng_state(), state)
    assert module.training  # mode restored
    planted = tables["hier_planted"]
    assert set(planted["metric"]) >= {"sender_lift", "pair_auprc", "arrow_angle_deg", "sender_in_keys"}
    lift = planted[(planted.metric == "sender_lift") & (planted.head == "mean")]["value"]
    assert np.isfinite(lift).all() and (lift > 0).all()
    if candidates == "routed":
        assert planted["metric"].str.startswith("routing_recall_l").any()
    health = tables["hier_health"]
    assert {"coarse_entropy", "never_selected", "lambda"} - set(health["metric"]) == {"lambda"}  # factorized: no lambda
    assert len(tables["hier_profile"]) == 3 * 2 and len(tables["hier_stats"]) == 1
    fields = np.load(tmp_path / "slide.npz")
    assert np.allclose(fields["coarse_vec"] + fields["refine_vec"], fields["read_vec"], atol=1e-5)
    plot_fields(tmp_path / "slide.npz", tmp_path / "slide.png", senders=graph.pos[[9, 50]].numpy())
    assert (tmp_path / "slide.png").stat().st_size > 0
    # The planted pairs travel with the fields, so a plot can mark the senders of chosen classes by itself.
    assert set(fields["planted_class"]) == {"short"} and fields["planted_sender"].dtype.kind == "i"
    plot_fields(tmp_path / "slide.npz", tmp_path / "classes.png", classes=["short"])
    assert (tmp_path / "classes.png").stat().st_size > 0


def test_the_readouts_read_the_dense_model_alike():
    """The dense transformer and its singleton-hierarchy twin compute one function, so their planted tables agree."""
    from interscale.evaluation.hierarchy_readouts import hierarchy_readouts

    graph = _lattice(7, seed=3)
    graph.obs_names = torch.arange(49) + 1
    edges = pd.DataFrame(
        {"sender": [9, 9, 30], "receiver": [10, 17, 38], "weight": [1.0, 2.0, 1.0], "range_class": ["a", "a", "b"]}
    )
    dense = _stir(_build(_config()))
    twin = _build(_config({**SOFT_ZERO, "cells_per_cluster": [1.0]}))
    twin.load_state_dict(dense.state_dict(), strict=False)
    common = {
        "prediction_task": "regression",
        "prediction_level": "node",
        "device": torch.device("cpu"),
        "edges": edges,
    }
    a = hierarchy_readouts(dense, [graph], distance_bins_um=[0, 150, 2000], **common)
    b = hierarchy_readouts(twin, [graph], distance_bins_um=[0, 150, 2000], **common)
    planted_a = a["hier_planted"].set_index(["layer", "range_class", "head", "metric"])["value"]
    planted_b = b["hier_planted"].set_index(["layer", "range_class", "head", "metric"])["value"].loc[planted_a.index]
    assert np.allclose(planted_a.to_numpy(), planted_b.to_numpy(), atol=1e-4, equal_nan=True)
    assert np.allclose(a["hier_profile"]["share"], b["hier_profile"]["share"], atol=1e-5)
