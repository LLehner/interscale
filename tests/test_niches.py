"""Niche patches and their three uses (``tl/niches.py``; lung_plan.md L2-L3, masking_plan.md M3).

Pinned here:

* **the patches** -- a patch is a connected piece of one label on the graph's own edges (two places of one niche are two
  patches, no patch spans two labels); small patches merge into the neighbour they share most edges with (or the nearest one
  when they have no edge), large ones are cut into kdtree pieces inside them; an unlabelled cell is a label of its own; a
  batch is numbered graph by graph;
* **the hierarchy's top level** (``top_level.mode: obs_label``) -- one cluster per label, per patch or per kdtree piece of
  a patch (plan test 10), the finer levels nested inside, cached by labels and edges too; through the model: the top level
  is the patches of the kept cells, a label change moves the prediction while no hidden value does, singleton pieces give the
  dense model exactly, an unbalanced level warns that routing is starved; the config and checkpoint names;
* **the mask unit** (``dataset.mask_block_kind: label``) -- whole patches drawn as discs are, the disc draw unchanged random
  number for random number, redrawn for train and fixed for val, pieces cut to ``max_cells``; the evaluation draws them too;
  the config and checkpoint names;
* **pooled tokens** (``pooling.kind: label``) -- one token per patch, every cell of a patch shares its global embedding, no
  hidden value reaches a prediction, a patch of one cell per label is the per-cell model; the config and checkpoint names.
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.module.global_modules.hierarchy import HierarchyCache, build_hierarchy
from interscale.tl.niches import (
    batch_label_patches,
    graph_label_patches,
    label_components,
    label_patches,
    merge_small_patches,
    niche_codes,
    split_large_patches,
)

E, G, H = 8, 6, 2
SPACING = 100.0


def _lattice_edges(pos):
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    return edges[:, edges[0] != edges[1]]


def _stripes(side=10):
    """Positions, edges and labels of a lattice: label 0 in two outer stripes (x < 3, x >= 7), label 1 between them, and
    one cell of label 0 alone inside the middle stripe."""
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    x = ii.flatten()
    labels = ((x >= 3) & (x < 7)).long()
    island = int(((ii.flatten() == 5) & (jj.flatten() == 5)).nonzero())
    labels[island] = 0
    return pos, _lattice_edges(pos), labels, island


def _one_hot(labels, n_labels=None):
    n_labels = int(labels.max()) + 1 if n_labels is None else n_labels
    return torch.nn.functional.one_hot(labels, n_labels).float()


def _connected(cells, edge_index):
    """Whether ``cells`` are connected through edges among themselves."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cells = np.asarray(cells)
    inside = np.isin(edge_index[0], cells) & np.isin(edge_index[1], cells)
    where = {c: k for k, c in enumerate(cells)}
    src = [where[c] for c in edge_index[0][inside]]
    dst = [where[c] for c in edge_index[1][inside]]
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(cells), len(cells)))
    return connected_components(graph, directed=True, connection="weak")[0] == 1


def _same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])


# --------------------------------------------------------------------------- the patches


def test_a_patch_is_a_connected_piece_of_one_label_and_two_places_of_a_niche_are_two_patches():
    pos, edges, labels, island = _stripes()
    patch = label_components(edges.numpy(), labels.numpy(), 100)
    assert patch.max() + 1 == 4  # left stripe, middle stripe, right stripe, the island
    assert patch[0] == 0 and np.all(np.diff(np.unique(patch, return_index=True)[1]) > 0)  # numbered by first cell
    for p in range(4):
        cells = np.flatnonzero(patch == p)
        assert len(np.unique(labels.numpy()[cells])) == 1
        assert _connected(cells, edges.numpy())
    left, right = patch[0], patch[99]
    assert left != right and labels[0] == labels[99]
    assert (patch == patch[island]).sum() == 1


def test_small_patches_merge_into_the_neighbour_they_share_most_edges_with():
    pos, edges, labels, island = _stripes()
    patch = merge_small_patches(label_components(edges.numpy(), labels.numpy(), 100), edges.numpy(), 5, pos.numpy())
    assert patch.max() + 1 == 3 and np.bincount(patch).min() >= 5
    middle = patch[50]
    assert patch[island] == middle  # its only neighbour
    assert set(labels.numpy()[patch == middle]) == {0, 1}  # a merged patch may hold two labels
    assert all(_connected(np.flatnonzero(patch == p), edges.numpy()) for p in range(3))
    # More shared edges beat a larger neighbour: one cell with one edge to a patch of 10 and two to a patch of 3.
    chain = [(k, k + 1) for k in range(9)] + [(10, 11), (11, 12), (13, 9), (13, 10), (13, 11)]
    both = torch.tensor(chain + [(b, a) for a, b in chain]).t()
    beside = merge_small_patches(np.array([0] * 10 + [1] * 3 + [2]), both.numpy(), 2)
    assert beside[13] == beside[10] != beside[0]
    # Ties in shared edges go to the larger neighbour: a cell between a big and a small patch, one edge to each.
    line = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    tied = merge_small_patches(np.array([0, 0, 0, 1, 2]), line.numpy(), 2)
    assert tied[3] == tied[0] and tied[4] == tied[0]


def test_a_patch_without_an_edge_to_another_joins_the_nearest_one_or_stays_without_positions():
    pos, edges, labels, island = _stripes()
    cut = edges[:, (edges[0] != island) & (edges[1] != island)].numpy()
    components = label_components(cut, labels.numpy(), 100)
    merged = merge_small_patches(components, cut, 5, pos.numpy())
    assert merged[island] == merged[50]  # the middle stripe's centroid is nearest
    kept = merge_small_patches(components, cut, 5)
    assert (kept == kept[island]).sum() == 1


def test_a_large_patch_is_cut_into_kdtree_pieces_inside_it():
    pos, edges, labels, _ = _stripes()
    components = label_components(edges.numpy(), labels.numpy(), 100)
    pieces = split_large_patches(components, pos.numpy(), 10)
    sizes = np.bincount(pieces)
    assert sizes.max() <= 10
    for p in range(pieces.max() + 1):
        assert len(np.unique(components[pieces == p])) == 1  # a piece never leaves its patch
    for c in range(components.max() + 1):
        assert len(np.unique(pieces[components == c])) == math.ceil((components == c).sum() / 10)
    patch = label_patches(edges.numpy(), labels.numpy(), 100, pos=pos.numpy(), min_cells=5, max_cells=10)
    assert np.bincount(patch).max() <= 10 and np.bincount(patch).min() >= 5
    with pytest.raises(ValueError, match="pass pos"):
        label_patches(edges.numpy(), labels.numpy(), 100, max_cells=10)


def test_an_unlabelled_cell_is_a_label_of_its_own():
    pos, edges, labels, _ = _stripes()
    data = Data(edge_index=edges, pos=pos, niche=_one_hot(labels))
    data.niche[:10] = 0.0  # the first row of cells: no label (NaN in obs)
    codes = niche_codes(data, "niche")
    assert (codes[:10] == -1).all() and torch.equal(codes[10:], labels[10:])
    patch = graph_label_patches(data, "niche")
    assert len(torch.unique(patch[:10])) == 1 and not torch.isin(patch[:10], patch[10:]).any()
    with pytest.raises(AttributeError, match="extra_obs_keys"):
        niche_codes(data, "region")


def test_a_batch_is_numbered_graph_by_graph():
    pos, edges, labels, _ = _stripes()
    small = Data(x=torch.zeros(100, 1), edge_index=edges, pos=pos, niche=_one_hot(labels))
    other_labels = (pos[:, 1] >= 500).long()
    other = Data(x=torch.zeros(100, 1), edge_index=edges, pos=pos, niche=_one_hot(other_labels))
    batch = Batch.from_data_list([small, other])
    patch, n = batch_label_patches(batch, "niche", max_cells=40)
    first = graph_label_patches(small, "niche", max_cells=40)
    second = graph_label_patches(other, "niche", max_cells=40)
    assert torch.equal(patch[:100], first) and torch.equal(patch[100:], second + int(first.max()) + 1)
    assert n == int(first.max()) + int(second.max()) + 2
    assert torch.equal(batch_label_patches(batch, "niche", max_cells=40)[0], patch)  # deterministic


# --------------------------------------------------------------------------- the hierarchy's top level


def _structure(hier):
    for level in range(1, hier.n_levels + 1):
        table = hier.children[level - 1]
        assert torch.equal(table[table >= 0].sort().values, torch.arange(hier.counts[level]))
    for level in range(1, hier.n_levels):
        assert torch.equal(hier.up[level][hier.assign[level]], hier.assign[level - 1])


def test_the_top_level_is_one_cluster_per_label_per_patch_or_per_piece():
    pos, edges, labels, island = _stripes()
    common = {"labels": labels.numpy(), "edge_index": edges.numpy(), "top_mode": "obs_label"}
    whole = build_hierarchy(pos.numpy(), cells_per_cluster=(10.0, 2.0), split_by_space="none", **common)
    assert whole.counts[0] == 2 and _same_partition(whole.assign[0], labels)
    patches = build_hierarchy(pos.numpy(), cells_per_cluster=(10.0, 2.0), split_by_space="components", **common)
    assert patches.counts[0] == 4 and _same_partition(patches.assign[0], label_components(edges, labels.numpy(), 100))
    pieces = build_hierarchy(pos.numpy(), cells_per_cluster=(10.0, 2.0), split_by_space="kdtree", **common)
    assert int(pieces.sizes[0].max()) <= 10
    for c in range(pieces.counts[0]):
        assert len(np.unique(patches.assign[0][pieces.assign[0] == c])) == 1
    merged = build_hierarchy(pos.numpy(), levels=(1, 20), split_by_space="components", min_cells=5, **common)
    assert merged.counts[0] == 3 and int(merged.sizes[0].min()) >= 5
    for hier in (whole, patches, pieces, merged):
        _structure(hier)
    shifted = build_hierarchy(pos.numpy(), cells_per_cluster=(10.0, 2.0), split_by_space="kdtree", shift=True, **common)
    assert int(shifted.sizes[0].max()) <= 10 and not torch.equal(shifted.assign[0], pieces.assign[0])
    with pytest.raises(ValueError, match="label for each"):
        build_hierarchy(pos.numpy(), levels=(4,), top_mode="obs_label")
    with pytest.raises(ValueError, match="edge_index"):
        build_hierarchy(pos.numpy(), levels=(4,), top_mode="obs_label", labels=labels.numpy())


def test_the_cache_keys_a_label_top_level_by_its_labels_and_edges():
    pos, edges, labels, _ = _stripes()
    cache = HierarchyCache()
    settings = {"levels": (4, 20), "top_mode": "obs_label", "split_by_space": "components"}
    first = cache.get(pos, labels=labels, edge_index=edges, **settings)
    assert cache.get(pos.clone(), labels=labels.clone(), edge_index=edges.clone(), **settings) is first
    assert cache.get(pos, labels=1 - labels, edge_index=edges, **settings) is not first
    assert cache.get(pos, labels=labels, edge_index=edges[:, :-2], **settings) is not first


def _graph(side, seed, labels=None):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    labels = (ii.flatten() >= side // 2).long() if labels is None else labels
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=_lattice_edges(pos), pos=pos)
    data.niche = _one_hot(labels, 3)
    data.mask = torch.zeros(n, dtype=torch.bool)
    return data


def _batch(graphs):
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _config(*, hierarchy=None, pool=None, max_seq_len=200):
    """The windowed global-only model with the raw own-cell skip at toy width, under cell masking, the niche label attached."""
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
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = max_seq_len, H, 2, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.bias = ["distance"]
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    for node, values in ((params.hierarchy, hierarchy), (params.pooling, pool)):
        for key, value in (values or {}).items():
            *path, last = key.split(".")
            target = node
            for part in path:
                target = target[part]
            target[last] = value
    cfg.dataset.mask_strategy = "node"
    cfg.dataset.mask_percentage = 0.3
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.extra_obs_keys = ["niche"]
    cfg.dataset.name = "d"
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


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _predict(module, batch):
    with torch.no_grad():
        random.seed(0)
        return module._common_step(batch, "regression", "node").y_pred


NICHE_TOP = {
    "kind": "coarse_to_fine",
    "prior_mode": "none",
    "readout": "concat",
    "cells_per_cluster": [6.0, 2.0],
    "top_level.mode": "obs_label",
    "top_level.obs_key": "niche",
    "top_level.split_by_space": "kdtree",
}


def test_through_the_model_the_top_level_is_the_patches_of_the_kept_cells():
    graphs = [_graph(6, seed=1), _graph(5, seed=2)]
    module = _stir(_build(_config(hierarchy=NICHE_TOP))).eval()
    with torch.no_grad(), module.recording_hierarchy() as record:
        random.seed(0)
        module._common_step(_batch(graphs), "regression", "node")
    seen = {entry["graph"]: entry for entry in record if entry["layer"] == 0}
    for g, graph in enumerate(graphs):
        entry = seen[g]
        expected = graph_label_patches(graph, "niche", max_cells=6)[torch.as_tensor(entry["nodes"])]
        assert _same_partition(entry["hierarchy"].assign[0].cpu(), expected)
        assert int(entry["hierarchy"].sizes[0].max()) <= 6


def test_the_label_moves_the_prediction_and_no_hidden_value_does():
    graph = _graph(6, seed=3)
    graph.mask[:12] = True  # two rows hidden whole
    batch = _batch([graph])
    module = _stir(_build(_config(hierarchy=NICHE_TOP))).eval()
    before = _predict(module, batch)
    hidden = batch.clone()
    hidden.x = batch.x.clone()
    hidden.x[:12] += torch.rand(12, G) * 50.0
    assert torch.equal(_predict(module, hidden), before)
    relabelled = batch.clone()
    relabelled.niche = _one_hot((batch.pos[:, 1] >= 300).long(), 3)  # the same slide cut the other way
    assert not torch.allclose(_predict(module, relabelled), before)


def test_singleton_pieces_of_a_label_give_the_dense_model():
    # Nothing hidden whole: a hidden singleton is a learned token at level 0, which CLS reads (as in the regression of
    # test_hierarchical_attention, run under programme masking).
    batch = _batch([_graph(6, seed=4), _graph(5, seed=5)])
    singletons = {
        "kind": "coarse_to_fine",
        "prior_mode": "soft",
        "lambda_init": 0.0,
        "lambda_learnable": "fixed",
        "cells_per_cluster": [1.0],
        "top_level.mode": "obs_label",
        "top_level.obs_key": "niche",
        "top_level.split_by_space": "kdtree",
    }
    dense = _build(_config())
    hier = _build(_config(hierarchy=singletons))
    _stir(dense)
    _, unexpected = hier.load_state_dict(dense.state_dict(), strict=False)
    assert not unexpected
    dense.eval(), hier.eval()
    assert torch.allclose(_predict(hier, batch), _predict(dense, batch), atol=1e-5)


def test_a_level_starved_by_one_wide_parent_warns():
    labels = torch.zeros(36, dtype=torch.long)
    labels[:6] = 1  # one patch of 6 cells and one of 30
    graph = _graph(6, seed=6, labels=labels)
    settings = {
        "kind": "coarse_to_fine",
        "prior_mode": "soft",
        "levels": [2, 36],
        "top_level.mode": "obs_label",
        "top_level.obs_key": "niche",
        "candidates.mode": "routed",
        "candidates.topk": [2],
        "candidates.cap_per_query": 16,
    }
    module = _build(_config(hierarchy=settings)).eval()
    with pytest.warns(RuntimeWarning, match="cap_per_query"):
        _predict(module, _batch([graph]))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"top_level.obs_key": ""}, "obs_key"),
        ({"top_level.obs_key": "region"}, "extra_obs_keys"),
        ({"top_level.split_by_space": "voronoi"}, "split_by_space"),
        ({"cells_per_cluster": []}, "coarse level"),
        ({"top_level.split_by_space": "none", "direction_prior": True, "prior_mode": "soft"}, "direction"),
        ({"top_level.mode": "tree"}, "only top_level.mode obs_label"),
    ],
)
def test_the_config_refuses_a_label_top_level_it_cannot_build(change, match):
    from interscale.config import _validate

    _validate(_config(hierarchy=NICHE_TOP))
    with pytest.raises(ValueError, match=match):
        _validate(_config(hierarchy={**NICHE_TOP, **change}))


def test_checkpoint_names_tell_the_label_top_levels_apart():
    from interscale.tl.utils import get_model_filename_prefix

    arms = [
        {**NICHE_TOP, "top_level.mode": "tree", "top_level.obs_key": "", "top_level.split_by_space": "components"},
        NICHE_TOP,
        {**NICHE_TOP, "top_level.split_by_space": "components"},
        {**NICHE_TOP, "top_level.split_by_space": "none"},
        {**NICHE_TOP, "top_level.min_cells": 5},
    ]
    names = [get_model_filename_prefix(_config(hierarchy=arm), False, True) for arm in arms]
    assert len(set(names)) == len(names)
    assert "-ob-niche-kd" in names[1] and "-ob-" not in names[0]


# --------------------------------------------------------------------------- the mask unit


def _old_disc_draw(pos, radius, fraction, generator):
    """The disc draw as it was before niche patches shared its loop."""
    pos = pos.double()
    target = max(1, math.ceil(fraction * pos.shape[0]))
    mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    while int(mask.sum()) < target:
        free = torch.where(~mask)[0]
        centre = free[torch.randint(len(free), (1,), generator=generator)]
        mask |= (pos - pos[centre]).norm(dim=1) <= radius
    return mask


def _reference_unit_draw(units, fraction, generator):
    """A cell drawn among the uncovered ones, its whole unit hidden, until the fraction is covered."""
    target = max(1, math.ceil(fraction * len(units)))
    mask = torch.zeros(len(units), dtype=torch.bool)
    while int(mask.sum()) < target:
        free = torch.where(~mask)[0]
        centre = free[torch.randint(len(free), (1,), generator=generator)]
        mask |= units == units[centre]
    return mask


def test_units_are_drawn_whole_until_the_fraction_is_covered_and_discs_are_unchanged():
    from interscale.tl.masking import sample_block_mask, sample_unit_mask

    pos, edges, labels, _ = _stripes()
    units = torch.as_tensor(label_patches(edges.numpy(), labels.numpy(), 100, pos=pos.numpy(), max_cells=10))
    for seed in range(5):
        mask = sample_unit_mask(units, 0.3, torch.Generator().manual_seed(seed))
        assert mask.sum() >= 30 and mask.sum() < 30 + 10
        for u in torch.unique(units):
            assert mask[units == u].all() or not mask[units == u].any()
        assert torch.equal(mask, _reference_unit_draw(units, 0.3, torch.Generator().manual_seed(seed)))
        disc = sample_block_mask(pos, 250.0, 0.3, torch.Generator().manual_seed(seed))
        assert torch.equal(disc, _old_disc_draw(pos, 250.0, 0.3, torch.Generator().manual_seed(seed)))


def _niche_graph():
    pos, edges, labels, _ = _stripes()
    data = Data(x=torch.rand(100, 5), edge_index=edges, pos=pos)
    data.niche = _one_hot(labels)
    return data


def _dm(seed=0, **blocks):
    """Train (two graphs) and val (one), the masks drawn after seeding the global generator (the graphs drawn before)."""
    from interscale.geome_dataloader import GraphAnnDataModule

    datas = [[_niche_graph(), _niche_graph()], [_niche_graph()]]
    torch.manual_seed(seed)
    dm = GraphAnnDataModule(
        datas=datas,
        batch_size=1,
        num_workers=0,
        mask_percentage=0.0,
        mask_strategy="node",
        **{"mask_block_fraction": 0.3, "mask_block_kind": "label", "mask_block_label": "niche", **blocks},
    )
    dm.setup(stage="fit")
    return dm


@pytest.mark.parametrize("max_cells", [0, 10])
def test_the_datamodule_hides_whole_patches_redrawn_for_train_and_fixed_for_val(max_cells):
    from interscale.tl.masking import sample_unit_mask

    units = graph_label_patches(_niche_graph(), "niche", max_cells=max_cells)
    torch.manual_seed(0)
    expected = [sample_unit_mask(units, 0.3) for _ in range(3)]  # train, train, val: the global generator, in order
    dm = _dm(seed=0, mask_block_max_cells=max_cells)
    assert all(torch.equal(d.mask, m) for d, m in zip(dm.train_data + dm.val_data, expected, strict=True))
    for data in dm.train_data + dm.val_data:
        assert data.mask.sum() >= 30
        for u in torch.unique(units):
            assert data.mask[units == u].all() or not data.mask[units == u].any()
    train, val = [d.mask.clone() for d in dm.train_data], [d.mask.clone() for d in dm.val_data]
    changed = False
    for _ in range(5):
        dm.resample_train_mask()
        changed |= any(not torch.equal(before, d.mask) for before, d in zip(train, dm.train_data, strict=True))
    assert changed
    assert all(torch.equal(before, d.mask) for before, d in zip(val, dm.val_data, strict=True))
    with pytest.raises(ValueError, match="Niche blocks need"):
        _dm(mask_block_label=None)


def _block_cfg(**dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.extra_obs_keys = ["niche"]
    cfg.dataset.mask_block_fraction = 0.3
    cfg.dataset.mask_block_kind = "label"
    cfg.dataset.mask_block_label_key = "niche"
    for key, value in dataset.items():
        setattr(cfg.dataset, key, value)
    return cfg


@pytest.mark.parametrize(
    ("dataset", "match"),
    [
        ({"mask_block_radius_um": 500.0}, "radius to 0"),
        ({"mask_block_label_key": ""}, "mask_block_label_key"),
        ({"extra_obs_keys": []}, "extra_obs_keys"),
        ({"mask_block_kind": "ring"}, "mask_block_kind"),
        ({"mask_block_max_cells": -1}, "not be negative"),
        ({"mask_block_fraction": 0.0}, "no niche patch would be drawn"),
        ({"mask_block_kind": "disc", "mask_block_radius_um": 500.0}, "mask_block_kind: label"),
    ],
)
def test_the_config_refuses_niche_blocks_it_cannot_draw(dataset, match):
    from interscale.config import _validate_block_mask

    _validate_block_mask(_block_cfg())
    with pytest.raises(ValueError, match=match):
        _validate_block_mask(_block_cfg(**dataset))


def test_niche_blocks_reach_the_datamodule_and_the_checkpoint_name():
    from interscale.config import _validate
    from interscale.geome_dataloader import block_mask_kwargs
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _block_cfg(mask_block_max_cells=50, mask_percentage=0.0)
    _validate(cfg)
    assert block_mask_kwargs(cfg) == {
        "mask_block_fraction": 0.3,
        "mask_block_kind": "label",
        "mask_block_label": "niche",
        "mask_block_max_cells": 50,
    }
    cfg.dataset.name = "d"
    name = get_model_filename_prefix(cfg, True, True)
    assert "blklab-niche-mx50f0.3" in name
    discs = _block_cfg(mask_block_kind="disc", mask_block_label_key="", mask_block_radius_um=500.0)
    discs.dataset.name = "d"
    assert "blk500f0.3" in get_model_filename_prefix(discs, True, True)


def test_the_deep_regime_accepts_niche_blocks():
    from interscale.config import _validate_regime_loss

    cfg = _block_cfg(mask_percentage=0.0)
    cfg.optim.regime_loss.kind = "depth"
    cfg.optim.regime_loss.global_cells = "deep"
    cfg.model.decoder.dual_decoder = True
    _validate_regime_loss(cfg)


def test_the_evaluation_draws_niche_patches_too_and_its_discs_are_unchanged():
    from interscale.evaluation.masked_probing import draw_masks
    from interscale.tl.masking import sample_block_mask, sample_unit_mask

    graphs = [_niche_graph(), _niche_graph()]
    draw_masks(graphs, "blocks", 0.3, seed=7, label="niche", max_cells=10)
    units = graph_label_patches(graphs[0], "niche", max_cells=10)
    assert torch.equal(graphs[0].mask, sample_unit_mask(units, 0.3, torch.Generator().manual_seed(7)))
    for data in graphs:
        assert data.mask.sum() >= 30 and torch.equal(data.gene_mask, data.mask[:, None].expand_as(data.gene_mask))
        for u in torch.unique(units):
            assert data.mask[units == u].all() or not data.mask[units == u].any()
    draw_masks(graphs, "cells+blocks", 0.0, seed=7, label="niche", block_rate=0.3)
    assert graphs[0].mask.sum() >= 30
    discs = [_niche_graph()]
    draw_masks(discs, "blocks", 0.3, seed=7, radius=250.0)
    assert torch.equal(discs[0].mask, sample_block_mask(discs[0].pos, 250.0, 0.3, torch.Generator().manual_seed(7)))
    with pytest.raises(ValueError, match="'blocks' needs"):
        draw_masks(discs, "blocks", 0.3, seed=7)


# --------------------------------------------------------------------------- pooled tokens


NICHE_POOL = {"kind": "label", "label_key": "niche"}


def test_one_token_per_patch_and_every_cell_of_a_patch_shares_its_embedding():
    from interscale.module.base import gather_tokens
    from interscale.module.global_modules.pooling import PatchPooling

    graphs = [_graph(6, seed=7), _graph(5, seed=8)]
    batch = _batch(graphs)
    pooling = PatchPooling(E, kind="label", label="niche", max_cells=9)
    plan = pooling.plan(batch)
    patch, n = batch_label_patches(batch, "niche", max_cells=9)
    assert torch.equal(plan.patch, patch) and plan.n_patches == n
    assert torch.allclose(plan.centroid[0], batch.pos[patch == 0].mean(0))
    stripes = _batch([_niche_graph()])
    assert PatchPooling(E, kind="label", label="niche").plan(stripes).n_patches == 4  # the island is a patch
    assert PatchPooling(E, kind="label", label="niche", min_cells=5).plan(stripes).n_patches == 3

    module = _build(_config(pool={**NICHE_POOL, "max_cells": 9})).eval()
    with module.recording_attention() as record, torch.no_grad():
        out = module._common_step(_batch(graphs[:1]), "regression", "node")
    one, _ = batch_label_patches(_batch(graphs[:1]), "niche", max_cells=9)
    assert record[0]["maps"].shape[-1] == int(one.max()) + 2  # the patches and CLS, not 36 cells
    embedding = gather_tokens(out.view.global_embedding, out.view.src_padding_mask)
    for p in range(int(one.max()) + 1):
        rows = embedding[one == p]
        assert torch.equal(rows, rows[:1].expand_as(rows))


def test_no_hidden_value_reaches_a_pooled_prediction_and_a_visible_one_does():
    graph = _graph(6, seed=9)
    graph.mask[:12] = True
    batch = _batch([graph])
    module = _stir(_build(_config(pool=NICHE_POOL))).eval()
    before = _predict(module, batch)
    hidden = batch.clone()
    hidden.x = batch.x.clone()
    hidden.x[:12] += torch.rand(12, G) * 50.0
    assert torch.equal(_predict(module, hidden), before)
    visible = batch.clone()
    visible.x = batch.x.clone()
    visible.x[30, 3] += 50.0
    after = _predict(module, visible)
    assert not torch.allclose(after[:12], before[:12])  # through its patch token and attention
    relabelled = batch.clone()
    relabelled.niche = _one_hot((batch.pos[:, 1] >= 300).long(), 3)
    assert not torch.allclose(_predict(module, relabelled), before)


def test_a_label_per_cell_pools_into_the_per_cell_model():
    graphs = [_graph(5, seed=10), _graph(4, seed=11)]
    for g in graphs:
        g.niche = torch.eye(g.num_nodes, 25)
    batch = _batch(graphs)
    plain = _build(_config()).eval()
    pooled = _build(_config(pool=NICHE_POOL)).eval()
    assert set(pooled.state_dict()) - set(plain.state_dict()) == {"pooling.fill"}
    assert torch.allclose(_predict(plain, batch), _predict(pooled, batch), atol=1e-5)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"label_key": ""}, "label_key"),
        ({"label_key": "region"}, "extra_obs_keys"),
        ({"size_um": 200.0}, "no size_um"),
        ({"min_cells": 0}, "min_cells"),
        ({"kind": "grid", "size_um": 200.0}, "set kind label"),
    ],
)
def test_the_config_refuses_niche_pooling_it_cannot_build(change, match):
    from interscale.config import _validate

    _validate(_config(pool=NICHE_POOL))
    with pytest.raises(ValueError, match=match):
        _validate(_config(pool={**NICHE_POOL, **change}))


def test_checkpoint_names_tell_niche_pooling_apart():
    from interscale.tl.utils import get_model_filename_prefix

    arms = [
        NICHE_POOL,
        {**NICHE_POOL, "min_cells": 5},
        {**NICHE_POOL, "max_cells": 50},
        {"kind": "grid", "size_um": 200.0},
    ]
    names = [get_model_filename_prefix(_config(pool=arm), False, True) for arm in arms]
    assert len(set(names)) == len(names)
    assert (
        "pool-label-niche_" in names[0] and "pool-label-niche-mc5_" in names[1] and "pool-label-niche-mx50_" in names[2]
    )
