"""A local and a global gene set from Moran's I at the local component's reach and across the slide (masking_plan.md, M2).

``evaluation.gene_scales.scale_sets`` and its parts. Pinned here:

* **the reach** -- ``h`` hops on the graph, made symmetric, never the cell itself, never across slides; a hand count on a
  lattice;
* **the statistics** -- the local one is the textbook Moran's I with row-standardised hop weights (pooled over slides); the
  global one is the textbook statistic with inverse-distance weights over every other cell of the slide (floored at the
  nearest-neighbour distance), estimated from sampled centres; a slide-wide gradient keeps its global Moran's I, a pattern on
  the scale of a cell loses it;
* **the selection** -- candidates are significant (Bonferroni over genes) and above the floor; the smallest fall is global,
  the largest local; the relative rule divides a gene's size out, the absolute one does not (a weak local gene lands in the
  global set); too few candidates split in half with a warning;
* **end to end** -- on constructed fields of known scale and size the sets are what they must be, noise is no candidate,
  the result is reproducible and leaves every global generator alone;
* **the output** -- ``var`` columns a ``dataset.mask_genes`` selector names; the figure renders.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from interscale.evaluation.gene_scales import (
    GLOBAL_SET,
    LOCAL_SET,
    NONE,
    ScaleSets,
    global_moran,
    hop_neighbourhoods,
    local_moran,
    local_moran_null,
    plot_scale_sets,
    scale_sets,
    select_scale_sets,
    write_scale_sets,
)
from interscale.tl.masking import resolve_mask_genes


def lattice(side: int, n_slides: int = 1, spacing: float = 10.0):
    """``(pos [N, 2], graph [N])``: ``n_slides`` square lattices of ``side`` x ``side`` cells, far apart."""
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    one = np.stack([xs.ravel(), ys.ravel()], 1).astype(float) * spacing
    pos = np.concatenate([one + s * 1e5 for s in range(n_slides)])
    graph = np.repeat(np.arange(n_slides), side * side)
    return pos, graph


def knn_adjacency(pos, graph, k: int):
    """A directed k-NN graph per slide (as squidpy builds it before symmetrising), sparse ``[N, N]``."""
    rows, cols = [], []
    for g in np.unique(graph):
        idx = np.flatnonzero(graph == g)
        _, j = cKDTree(pos[idx]).query(pos[idx], k=k + 1)
        rows.append(np.repeat(idx, k))
        cols.append(idx[j[:, 1:]].ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pos), len(pos)))


def rook_adjacency(side: int):
    """The 4-neighbour lattice graph of :func:`lattice` (one slide), symmetric."""
    pos, graph = lattice(side)
    d = np.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    return sp.csr_matrix((d > 0) & (d <= 10.0 + 1e-9)).astype(float), pos, graph


def textbook_moran(z_by_slide: list[np.ndarray], w_by_slide: list[np.ndarray]) -> np.ndarray:
    """Pooled Moran's I with row-standardised dense weights, every cell a centre: ``sum z W z / sum z^2`` over slides."""
    total = sum((z * ((w / w.sum(1, keepdims=True)) @ z)).sum(0) for z, w in zip(z_by_slide, w_by_slide))
    norm = sum((z**2).sum(0) for z in z_by_slide)
    return total / norm


def zscore(values):
    return (values - values.mean(0)) / values.std(0)


def fields(side: int, n_slides: int, seed: int = 0):
    """Genes of known scale on lattices: a slide-wide gradient (global), smooth blobs of ~1 cell (local) at full and at
    weak size, and white noise. Returns ``(values [N, G], pos, graph, genes)``."""
    rng = np.random.default_rng(seed)
    pos, graph = lattice(side, n_slides)
    columns = {"gradient": [], "wide": [], "patch": [], "weak_patch": [], "noise_a": [], "noise_b": []}
    for s in range(n_slides):
        x = np.arange(side) / side
        gradient = np.add.outer(np.cos(np.pi * x), 0.5 * np.sin(np.pi * x)).ravel()
        wide = gaussian_filter(rng.normal(size=(side, side)), sigma=side / 4, mode="wrap").ravel()
        patch = gaussian_filter(rng.normal(size=(side, side)), sigma=0.8, mode="wrap").ravel()
        columns["gradient"].append(zscore(gradient[:, None])[:, 0] + 0.3 * rng.normal(size=side * side))
        columns["wide"].append(zscore(wide[:, None])[:, 0] + 0.3 * rng.normal(size=side * side))
        columns["patch"].append(zscore(patch[:, None])[:, 0] + 0.3 * rng.normal(size=side * side))
        # The same kind of pattern, buried in noise: a small Moran's I, but still local.
        columns["weak_patch"].append(0.6 * zscore(patch[:, None])[:, 0] + rng.normal(size=side * side))
        columns["noise_a"].append(rng.normal(size=side * side))
        columns["noise_b"].append(rng.normal(size=side * side))
    genes = list(columns)
    values = np.stack([np.concatenate(columns[g]) for g in genes], 1) + 5.0
    return values, pos, graph, genes


# ------------------------------------------------------------------------------- the reach


def test_hop_neighbourhoods_count_a_lattice_by_hand():
    adjacency, pos, graph = rook_adjacency(7)
    one, two = hop_neighbourhoods(adjacency, graph, 2)
    centre = 3 * 7 + 3
    assert one[centre].nnz == 4  # the 4 rook neighbours
    assert two[centre].nnz == 12  # 4 + 8 at Manhattan distance 2
    assert one.diagonal().sum() == 0 and two.diagonal().sum() == 0
    assert (abs(two - two.T)).nnz == 0
    corner = 0
    assert two[corner].nnz == 5  # (1,0) (0,1) (2,0) (1,1) (0,2)


def test_hop_neighbourhoods_symmetrise_a_directed_graph():
    pos, graph = lattice(6)
    directed = knn_adjacency(pos, graph, 3)
    assert (abs(directed - directed.T)).nnz > 0
    (one,) = hop_neighbourhoods(directed, graph, 1)
    assert (abs(one - one.T)).nnz == 0
    assert np.all(one.toarray()[directed.toarray() > 0] == 1)


def test_hop_neighbourhoods_refuse_an_edge_between_slides():
    pos, graph = lattice(4, n_slides=2)
    adjacency = knn_adjacency(pos, graph, 3).tolil()
    adjacency[0, 20] = 1.0
    with pytest.raises(ValueError, match="different graphs"):
        hop_neighbourhoods(adjacency.tocsr(), graph, 2)
    with pytest.raises(ValueError, match="hops"):
        hop_neighbourhoods(knn_adjacency(pos, graph, 3), graph, 0)


# ------------------------------------------------------------------------------- the statistics


def test_local_moran_is_the_textbook_statistic_pooled_over_slides():
    rng = np.random.default_rng(1)
    pos, graph = lattice(6, n_slides=2)
    values = rng.normal(size=(len(pos), 4)) + np.arange(4)
    values[graph == 1] *= 3.0  # a different scale on the second slide: each is z-scored on its own
    adjacency = knn_adjacency(pos, graph, 4)
    hoods = hop_neighbourhoods(adjacency, graph, 2)
    found = local_moran(values, graph, hoods)
    for h, hood in enumerate(hoods):
        dense = hood.toarray()
        zs = [zscore(values[graph == g]) for g in (0, 1)]
        ws = [dense[np.ix_(graph == g, graph == g)] for g in (0, 1)]
        np.testing.assert_allclose(found[h], textbook_moran(zs, ws), atol=1e-12)


def test_global_moran_with_every_centre_is_the_inverse_distance_statistic():
    rng = np.random.default_rng(2)
    pos, graph = lattice(5, n_slides=2)
    pos = pos + rng.normal(scale=0.5, size=pos.shape)
    values = rng.normal(size=(len(pos), 3))
    found = global_moran(values, graph, pos, n_centres=None)
    zs, ws = [], []
    for g in (0, 1):
        p = pos[graph == g]
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        floor = np.median(np.sort(d, axis=1)[:, 1])
        w = 1.0 / np.maximum(d, floor)
        np.fill_diagonal(w, 0.0)
        zs.append(zscore(values[graph == g]))
        ws.append(w)
    np.testing.assert_allclose(found, textbook_moran(zs, ws), atol=1e-12)


def test_global_moran_reads_the_slide_scale_not_the_cell_scale():
    values, pos, graph, genes = fields(side=24, n_slides=2)
    found = dict(zip(genes, global_moran(values, graph, pos, n_centres=None)))
    assert found["gradient"] > 0.3
    assert found["wide"] > found["patch"] + 0.1
    assert abs(found["noise_a"]) < 0.02 and abs(found["patch"]) < 0.05


def test_global_moran_sampled_centres_estimate_it_reproducibly():
    values, pos, graph, _ = fields(side=24, n_slides=2)
    every = global_moran(values, graph, pos, n_centres=None)
    a = global_moran(values, graph, pos, n_centres=200, seed=5)
    b = global_moran(values, graph, pos, n_centres=200, seed=5)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, every, atol=0.05)
    assert not np.array_equal(a, global_moran(values, graph, pos, n_centres=200, seed=6))


def test_local_null_holds_a_noise_gene():
    values, pos, graph, genes = fields(side=20, n_slides=2)
    hoods = hop_neighbourhoods(knn_adjacency(pos, graph, 8), graph, 2)
    mean, sd = local_moran_null(values, graph, hoods, n_permutations=20, seed=0)
    assert np.all(np.abs(mean) < 0.05) and np.all(sd > 0)
    with pytest.raises(ValueError, match="permutations"):
        local_moran_null(values, graph, hoods, n_permutations=1)


# ------------------------------------------------------------------------------- the selection


def table_of(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows, index=["i_local", "i_global", "z_local"]).T


def test_selection_takes_the_two_ends_of_the_relative_fall():
    table = table_of(
        {
            "smooth": [0.9, 0.6, 50.0],  # fall 0.33
            "smooth2": [0.8, 0.4, 50.0],  # 0.5
            "mid": [0.8, 0.2, 50.0],  # 0.75
            "local": [0.8, 0.02, 50.0],  # 0.975
            "weak_local": [0.08, 0.0, 10.0],  # 1.0, small but significant
            "noise": [0.01, 0.0, 0.5],  # not significant
        }
    )
    out = select_scale_sets(table, 2, rule="relative", min_local_i=0.05, z=3.0)
    assert list(out.index[out["scale_set"] == GLOBAL_SET]) == ["smooth", "smooth2"]
    assert set(out.index[out["scale_set"] == LOCAL_SET]) == {"local", "weak_local"}
    assert out.loc["noise", "scale_set"] == NONE and not out.loc["noise", "candidate"]
    assert np.isnan(out.loc["noise", "drop"]) and out.loc["smooth", "rank"] == 1


def test_absolute_rule_puts_a_weak_local_gene_into_the_global_set():
    """The reason the relative rule is the default: an absolute difference is small for a small pattern of any scale."""
    table = table_of(
        {
            "smooth": [0.9, 0.6, 50.0],
            "smooth2": [0.8, 0.45, 50.0],
            "local": [0.8, 0.02, 50.0],
            "weak_local": [0.08, 0.0, 10.0],
        }
    )
    relative = select_scale_sets(table, 1, rule="relative")
    absolute = select_scale_sets(table, 1, rule="absolute")
    assert relative.loc["weak_local", "scale_set"] == LOCAL_SET
    assert absolute.loc["weak_local", "scale_set"] == GLOBAL_SET
    assert absolute.loc["local", "scale_set"] == LOCAL_SET


def test_absolute_rule_is_the_size_of_the_difference_either_way():
    """A gene more autocorrelated across the slide than within the reach is far from its local value under ``absolute``
    (the user's rule as stated), and the most global gene under ``relative`` (its fall is negative)."""
    table = table_of({"flat": [0.9, 0.85, 50.0], "rising": [0.2, 0.5, 50.0], "local": [0.8, 0.0, 50.0]})
    absolute = select_scale_sets(table, 1, rule="absolute")
    assert absolute.loc["flat", "scale_set"] == GLOBAL_SET and absolute.loc["rising", "drop"] == pytest.approx(0.3)
    relative = select_scale_sets(table, 1, rule="relative")
    assert relative.loc["rising", "scale_set"] == GLOBAL_SET and relative.loc["rising", "drop"] < 0


def test_selection_filters_by_significance_and_floor():
    table = table_of({f"g{k}": [0.3, 0.1, 3.5] for k in range(100)} | {"strong": [0.3, 0.1, 10.0], "low": [0.04, 0.0, 30.0]})
    out = select_scale_sets(table, 1, rule="relative", min_local_i=0.05, z=3.0)
    # 3.5 SDs is below the Bonferroni bar for 102 genes (about 4.4); "low" is under the floor.
    assert out["candidate"].sum() == 1 and out.loc["strong", "candidate"] and not out.loc["low", "candidate"]


def test_selection_splits_too_few_candidates_in_half():
    table = table_of({"a": [0.9, 0.6, 50.0], "b": [0.8, 0.1, 50.0], "c": [0.7, 0.05, 50.0]})
    with pytest.warns(UserWarning, match="3 candidate genes"):
        out = select_scale_sets(table, 5)
    assert (out["scale_set"] == GLOBAL_SET).sum() == 1 and (out["scale_set"] == LOCAL_SET).sum() == 1
    with pytest.raises(ValueError, match="rule"):
        select_scale_sets(table, 1, rule="ratio")


# ------------------------------------------------------------------------------- end to end


def test_scale_sets_on_fields_of_known_scale():
    values, pos, graph, genes = fields(side=24, n_slides=3)
    adjacency = knn_adjacency(pos, graph, 8)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    result = scale_sets(values, graph, pos, adjacency, genes, hops=2, n=2, n_permutations=10, n_centres=300, seed=0)
    assert np.array_equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    table = result.table
    assert isinstance(result, ScaleSets)
    assert set(result.genes(GLOBAL_SET)) == {"gradient", "wide"}
    assert set(result.genes(LOCAL_SET)) == {"patch", "weak_patch"}
    assert not table.loc[["noise_a", "noise_b"], "candidate"].any()
    # The weak pattern: a far smaller Moran's I than the strong one, the same fall.
    assert table.loc["weak_patch", "i_local"] < 0.5 * table.loc["patch", "i_local"]
    # 8 neighbours, 2 hops on a lattice: 24 cells, out to 2 * sqrt(2) spacings.
    assert result.reach_cells == pytest.approx(24, abs=1)
    assert result.reach_um == pytest.approx(20 * np.sqrt(2), rel=0.01)
    again = scale_sets(values, graph, pos, adjacency, genes, hops=2, n=2, n_permutations=10, n_centres=300, seed=0)
    pd.testing.assert_frame_equal(table, again.table)


def test_scale_sets_refuse_mismatched_inputs():
    values, pos, graph, genes = fields(side=8, n_slides=1)
    adjacency = knn_adjacency(pos, graph, 4)
    with pytest.raises(ValueError, match="genes"):
        scale_sets(values, graph, pos, adjacency, genes[:-1], n=1, n_permutations=2)
    with pytest.raises(ValueError, match="local_form"):
        scale_sets(values, graph, pos, adjacency, genes, n=1, local_form="mean", n_permutations=2)


def test_local_form_reach_reads_the_whole_neighbourhood():
    values, pos, graph, genes = fields(side=16, n_slides=1)
    adjacency = knn_adjacency(pos, graph, 8)
    by_max = scale_sets(values, graph, pos, adjacency, genes, n=1, local_form="max", n_permutations=4, n_centres=100)
    by_reach = scale_sets(values, graph, pos, adjacency, genes, n=1, local_form="reach", n_permutations=4, n_centres=100)
    np.testing.assert_allclose(by_reach.table["i_local"], by_reach.table["i_reach"])
    assert np.all(by_max.table["i_local"] >= by_max.table["i_reach"] - 1e-12)
    assert set(by_reach.table["hop_local"].dropna()) == {2.0}


# ------------------------------------------------------------------------------- the output


def test_written_columns_are_named_by_a_mask_genes_selector():
    values, pos, graph, genes = fields(side=16, n_slides=2)
    result = scale_sets(values, graph, pos, knn_adjacency(pos, graph, 8), genes, n=2, n_permutations=6, n_centres=100)
    var = pd.DataFrame(index=pd.Index(genes + ["absent"]))
    write_scale_sets(var, result)
    assert var.loc["absent", "scale_set"] == NONE and np.isnan(var.loc["absent", "scale_i_local"])
    columns = resolve_mask_genes(["scale_set=global"], var)
    assert [genes[c] for c in columns] == sorted(result.genes(GLOBAL_SET), key=genes.index)
    assert {"scale_i_local", "scale_i_global", "scale_drop", "scale_z_local"} <= set(var.columns)


def test_figure_renders_with_and_without_a_truth():
    import matplotlib

    matplotlib.use("Agg")
    values, pos, graph, genes = fields(side=12, n_slides=1)
    result = scale_sets(values, graph, pos, knn_adjacency(pos, graph, 8), genes, n=1, n_permutations=4, n_centres=50)
    truth = pd.Series({"gradient": 500.0, "wide": 300.0, "patch": 10.0, "weak_patch": 10.0})
    assert len(plot_scale_sets(result).axes) == 1
    assert len(plot_scale_sets(result, truth=truth, title="t").axes) == 2
