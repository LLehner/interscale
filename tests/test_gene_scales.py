"""Moran's I against a growing neighbourhood, and which part of a decomposable prediction carries each gene.

roadmap.md, task 1.

``evaluation.gene_scales``. Pinned here:

* **the neighbours** -- never the centre, never another slide, nearest first; cells at one position stay each other's
  neighbours; a small slide pads; the centres are a private, reproducible draw;
* **the statistic** -- with every cell a centre it is the textbook Moran's I (row-standardised k-NN weights), cumulative
  and per shell; sampled centres estimate it, float32 sums agree with float64; every slide is z-scored on its own (an
  offset between slides is no autocorrelation) and the sums are pooled; a slide where a gene is constant counts nothing;
  the cumulative curve is the count-weighted mean of the shells up to it;
* **the null and the scales** -- the permutation null sits near 0 and holds a noise gene; a smooth field has a longer
  scale than a rough one and alternating values come out ``anti``; the half point is interpolated on log k;
* **the shares** -- per gene, a part's share of the variance of the prediction (1 for the part that holds it all,
  negative for one against the others), over the kept entries only;
* **the folds and the model** -- every cell hidden once at the rate; through a real additive dual model every cell gets
  one row at its cell, the parts add up to the prediction, the graphs' masks come back, and a cell's own values never
  reach its row while its visible neighbours do; a head without parts is refused;
* **the tables, the figures and the report** -- the summary correlates the scale with the share, the design table counts
  per class, the figures render, and the report's summary carries every arm's gene scales.
"""

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from interscale.evaluation import report
from interscale.evaluation.gene_scales import (
    ANTI,
    CONSTANT,
    NONE,
    TRUTH,
    VARIABLE,
    FoldParts,
    GeneScales,
    MoranCurves,
    MoranNull,
    assignment,
    cell_folds,
    collect_fold_parts,
    curves_from_ranks,
    gene_cosine,
    gene_scales,
    half_point,
    length_scales,
    moran_by_rank,
    moran_curves,
    part_shares,
    permutation_null,
    pick_genes,
    plot_part_curves,
    plot_part_maps,
    plot_scale_curves,
    sample_neighbours,
    scale_tables,
)


def _random_slide(n=80, seed=0):
    """Random positions (no ties) on one slide and three genes: noise, a gradient, a wave."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 10.0, (n, 2))
    values = np.column_stack(
        [
            rng.normal(size=n),
            pos[:, 0] + rng.normal(scale=0.5, size=n),
            np.sin(pos[:, 1]) + rng.normal(scale=0.2, size=n),
        ]
    )
    return pos, np.zeros(n, dtype=int), values


def _lattice(side=24, slides=1, spacing=100.0):
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    one = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float) * spacing
    return np.tile(one, (slides, 1)), np.repeat(np.arange(slides), side * side)


def _field(side, sigma, rng):
    """A Gaussian random field on the lattice, smoothed at ``sigma`` lattice steps, z-scored."""
    f = gaussian_filter(rng.normal(size=(side, side)), sigma=sigma)
    return ((f - f.mean()) / f.std()).ravel()


# --------------------------------------------------------------------------- the neighbours


def test_the_neighbours_are_the_nearest_other_cells_of_the_same_slide():
    rng = np.random.default_rng(0)
    one = rng.uniform(0.0, 10.0, (40, 2))
    pos, graph = np.concatenate([one, one]), np.repeat([0, 1], 40)  # two slides on the same coordinates
    nb = sample_neighbours(pos, graph, k_max=6, n_centres=None)
    assert np.array_equal(nb.centre, np.arange(80))
    assert (nb.neighbour != nb.centre[:, None]).all()
    assert (graph[nb.neighbour] == graph[nb.centre][:, None]).all()
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    d[graph[:, None] != graph[None]] = np.inf
    np.fill_diagonal(d, np.inf)
    np.testing.assert_allclose(nb.distance, np.sort(d, axis=1)[:, :6])
    np.testing.assert_allclose(np.take_along_axis(d, nb.neighbour, axis=1), nb.distance)
    assert np.allclose(nb.reach([1, 6], unit=2.0), 2.0 * np.median(nb.distance[:, [0, 5]], axis=0))


def test_cells_at_one_position_stay_each_others_neighbours():
    pos = np.array([[0, 0], [0, 0], [0, 0], [5, 0], [9, 0], [20, 0]], dtype=float)
    nb = sample_neighbours(pos, np.zeros(6, dtype=int), k_max=3, n_centres=None)
    for c in range(3):
        assert c not in nb.neighbour[c]
        assert set(nb.neighbour[c, :2]) == {0, 1, 2} - {c}
        assert nb.distance[c, :2].tolist() == [0.0, 0.0]
    # more cells at one spot than neighbours asked for: the centre may miss the k + 1 found; it is never a neighbour
    stacked = sample_neighbours(np.zeros((5, 2)), np.zeros(5, dtype=int), k_max=2, n_centres=None)
    assert (stacked.neighbour != stacked.centre[:, None]).all() and (stacked.distance == 0).all()
    assert (stacked.neighbour >= 0).all()


def test_a_small_slide_pads_and_counts_nothing_beyond_its_cells():
    rng = np.random.default_rng(1)
    big, small = rng.uniform(0, 10, (50, 2)), rng.uniform(20, 21, (3, 2))
    pos, graph = np.concatenate([big, small]), np.repeat([0, 1], [50, 3])
    values = np.column_stack([pos[:, 0], rng.normal(size=53)])
    nb = sample_neighbours(pos, graph, k_max=5, n_centres=None)
    rows = graph[nb.centre] == 1
    assert (nb.neighbour[rows, 2:] == -1).all() and np.isinf(nb.distance[rows, 2:]).all()
    assert (nb.neighbour[rows, :2] >= 50).all()
    pooled = moran_curves(values, graph, nb, [2, 5], dtype=np.float64)
    alone = moran_curves(
        values[:50], graph[:50], sample_neighbours(big, np.zeros(50), k_max=5, n_centres=None), [2, 5], dtype=np.float64
    )
    np.testing.assert_allclose(pooled.shell[1], alone.shell[1])  # ranks 3..5: the small slide has none
    assert not np.allclose(pooled.shell[0], alone.shell[0])  # ranks 1..2: it has


def test_the_centres_are_a_private_reproducible_draw():
    pos, graph = _lattice(side=10, slides=3)
    np.random.seed(3)
    torch.manual_seed(3)
    numpy_state, torch_state = np.random.get_state()[1].copy(), torch.get_rng_state().clone()
    a = sample_neighbours(pos, graph, k_max=4, n_centres=60, seed=1)
    b = sample_neighbours(pos, graph, k_max=4, n_centres=60, seed=1)
    c = sample_neighbours(pos, graph, k_max=4, n_centres=60, seed=2)
    assert np.array_equal(a.centre, b.centre) and not np.array_equal(a.centre, c.centre)
    # 60 of each slide's 100 cells, none twice (drawn with replacement, some would repeat: p of none about 1e-8)
    assert len(a.centre) == 180 and len(np.unique(a.centre)) == 180
    assert (np.bincount(graph[a.centre]) == 60).all()
    assert np.array_equal(np.random.get_state()[1], numpy_state) and torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(ValueError, match="k_max"):
        sample_neighbours(pos, graph, k_max=0)


# --------------------------------------------------------------------------- the statistic


@pytest.mark.parametrize(("lo", "hi"), [(0, 3), (3, 8), (0, 8)])
def test_with_every_cell_a_centre_it_is_the_textbook_statistic(lo, hi):
    """Moran's I = (N / S0) z'Wz / z'z, weights 1 / (hi - lo) on each cell's neighbours ranked lo + 1 .. hi."""
    pos, graph, values = _random_slide()
    n = len(pos)
    nb = sample_neighbours(pos, graph, k_max=8, n_centres=None)
    curves = curves_from_ranks(moran_by_rank(values, graph, nb, dtype=np.float64), [3, 8])
    found = curves["cumulative" if lo == 0 else "shell"][[3, 8].index(hi)]
    _, j = cKDTree(pos).query(pos, k=hi + 1)
    w = np.zeros((n, n))
    for i in range(n):
        w[i, j[i, 1 + lo : 1 + hi]] = 1.0 / (hi - lo)
    z = values - values.mean(axis=0)
    textbook = (n / w.sum()) * np.einsum("ig,ij,jg->g", z, w, z) / (z**2).sum(axis=0)
    np.testing.assert_allclose(found, textbook, atol=1e-12)
    assert textbook[1] > 0.5 and abs(textbook[0]) < 0.3  # the gradient is autocorrelated, the noise is not


def test_sampled_centres_estimate_it_and_float32_sums_agree():
    pos, graph = _lattice(side=30)
    rng = np.random.default_rng(2)
    values = np.column_stack([_field(30, 3.0, rng), rng.normal(size=900)])
    ks = [4, 16, 64]
    every = sample_neighbours(pos, graph, k_max=64, n_centres=None)
    exact = moran_curves(values, graph, every, ks, dtype=np.float64)
    for seed in range(5):
        sampled = moran_curves(
            values, graph, sample_neighbours(pos, graph, k_max=64, n_centres=300, seed=seed), ks, dtype=np.float64
        )
        # the ratio estimator: the centres' own z ** 2 cancel (dividing by the slide's variance was up to 0.15 off here)
        assert np.abs(sampled.shell[:, 0] - exact.shell[:, 0]).max() < 0.04, seed
        assert np.abs(sampled.shell[:, 1] - exact.shell[:, 1]).max() < 0.1, seed  # the noise gene: sampling noise only
    assert exact.shell[0, 0] > 0.8
    np.testing.assert_allclose(moran_curves(values, graph, every, ks).shell, exact.shell, atol=1e-5)
    np.testing.assert_allclose(exact.ks, ks)
    np.testing.assert_allclose(exact.reach_um, every.reach(ks))


def test_each_slide_is_standardised_on_its_own_and_the_sums_pooled():
    rng = np.random.default_rng(5)
    pos_a, pos_b = rng.uniform(0, 10, (60, 2)), rng.uniform(0, 10, (90, 2))
    smooth_a, smooth_b = np.sin(pos_a[:, 0]), np.cos(pos_b[:, 1])
    values_a = np.column_stack([rng.normal(size=60) + 5.0, smooth_a])  # a slide offset on gene 0
    values_b = np.column_stack([rng.normal(size=90), smooth_b])
    pos, graph = np.concatenate([pos_a, pos_b]), np.repeat([0, 1], [60, 90])
    values = np.concatenate([values_a, values_b])
    nb = sample_neighbours(pos, graph, k_max=6, n_centres=None)
    pooled = moran_by_rank(values, graph, nb, dtype=np.float64)
    a = moran_by_rank(
        values_a, np.zeros(60), sample_neighbours(pos_a, np.zeros(60), k_max=6, n_centres=None), dtype=np.float64
    )
    b = moran_by_rank(
        values_b, np.zeros(90), sample_neighbours(pos_b, np.zeros(90), k_max=6, n_centres=None), dtype=np.float64
    )
    np.testing.assert_allclose(pooled.total, a.total + b.total, atol=1e-10)
    np.testing.assert_allclose(pooled.norm, a.norm + b.norm)
    np.testing.assert_allclose(pooled.norm[:, 1], 150.0)  # every cell a centre: the cells per rank
    i = curves_from_ranks(pooled, [6])["cumulative"][0]
    assert abs(i[0]) < 0.15  # the offset between the slides is not autocorrelation
    # z-scored over both slides at once, the offset would make every neighbour look alike
    assert curves_from_ranks(moran_by_rank(values, np.zeros(150), nb, dtype=np.float64), [6])["cumulative"][0][0] > 0.8


def test_a_gene_constant_in_a_slide_counts_nothing_there():
    rng = np.random.default_rng(6)
    pos_a, pos_b = rng.uniform(0, 10, (40, 2)), rng.uniform(0, 10, (50, 2))
    values = np.column_stack([np.r_[np.full(40, 3.0), np.sin(pos_b[:, 0])], np.full(90, 2.0)])
    pos, graph = np.concatenate([pos_a, pos_b]), np.repeat([0, 1], [40, 50])
    ks = [2, 4]
    nb = sample_neighbours(pos, graph, k_max=4, n_centres=None)
    pooled = moran_curves(values, graph, nb, ks, dtype=np.float64)
    alone = moran_curves(
        values[40:], graph[40:], sample_neighbours(pos_b, np.ones(50), k_max=4, n_centres=None), ks, dtype=np.float64
    )
    np.testing.assert_allclose(pooled.shell[:, 0], alone.shell[:, 0])
    assert np.isnan(pooled.shell[:, 1]).all() and np.isnan(pooled.cumulative[:, 1]).all()
    null = MoranNull(
        mean={f: np.zeros((2, 2)) for f in ("shell", "cumulative")},
        sd={f: np.full((2, 2), 0.01) for f in ("shell", "cumulative")},
    )
    assert length_scales(pooled, null, ["a", "b"]).loc["b", "pattern"] == CONSTANT


def test_the_cumulative_curve_is_the_weighted_mean_of_the_shells():
    pos, graph, values = _random_slide(n=120, seed=7)
    ks = [2, 5, 9, 16]
    sums = moran_by_rank(values, graph, sample_neighbours(pos, graph, k_max=16, n_centres=60, seed=1), dtype=np.float64)
    curves = curves_from_ranks(sums, ks)
    edges = np.cumsum(sums.norm, axis=0)[np.array(ks) - 1]
    weights = np.diff(np.vstack([np.zeros((1, 3)), edges]), axis=0)
    weighted = np.cumsum(curves["shell"] * weights, axis=0) / np.cumsum(weights, axis=0)
    np.testing.assert_allclose(curves["cumulative"], weighted, atol=1e-12)
    with pytest.raises(ValueError, match="ks must increase"):
        curves_from_ranks(sums, [4, 4])
    with pytest.raises(ValueError, match="ks must increase"):
        curves_from_ranks(sums, [4, 32])


# --------------------------------------------------------------------------- the null and the scales


def test_a_smooth_field_has_a_longer_scale_than_a_rough_one_and_alternation_is_anti():
    side = 30
    pos, graph = _lattice(side=side)
    rng = np.random.default_rng(8)
    xs, ys = pos[:, 0] / 100, pos[:, 1] / 100
    checker = np.where((xs + ys) % 2 == 0, 1.0, -1.0) + 0.05 * rng.normal(size=side * side)
    values = np.column_stack([_field(side, 1.0, rng), _field(side, 4.0, rng), checker, rng.normal(size=side * side)])
    res = gene_scales(
        {TRUTH: values},
        graph,
        pos,
        ["rough", "smooth", "checker", "noise"],
        ks=(4, 8, 16, 32, 64, 128, 256),
        n_centres=None,
        n_permutations=5,
    )
    t = res.table
    assert t.loc["rough", "pattern"] == VARIABLE and t.loc["smooth", "pattern"] == VARIABLE
    assert t.loc["smooth", "ell_um"] > 1.5 * t.loc["rough", "ell_um"]
    assert t.loc["smooth", "ell_k"] > t.loc["rough", "ell_k"]
    assert t.loc["checker", "pattern"] == ANTI and t.loc["checker", "i_first"] < -0.9
    assert t.loc["noise", "pattern"] == NONE and np.isnan(t.loc["noise", "ell_um"])
    assert list(t.columns) == ["pattern", "i_first", "z_first", "i_peak", "k_peak", "ell_k", "ell_um", "i_last", "auc"]


def test_the_null_sits_near_zero_and_holds_a_noise_gene():
    rng = np.random.default_rng(9)
    pos, graph = rng.uniform(0, 10, (300, 2)), np.zeros(300, dtype=int)
    values = rng.normal(size=(300, 6))
    nb = sample_neighbours(pos, graph, k_max=16, n_centres=None)
    np.random.seed(1)
    numpy_state = np.random.get_state()[1].copy()
    null = permutation_null(values, graph, nb, [4, 16], n_permutations=20, seed=0)
    assert np.array_equal(np.random.get_state()[1], numpy_state)
    assert null.mean["shell"].shape == (2, 6) and (null.sd["shell"] > 0).all()
    assert (np.abs(null.mean["shell"]) < 4 * null.sd["shell"] / math.sqrt(20) + 0.01).all()
    observed = moran_curves(values, graph, nb, [4, 16])
    assert (np.abs(observed.shell - null.mean["shell"]) < 4 * null.sd["shell"]).mean() >= 0.9
    again = permutation_null(values, graph, nb, [4, 16], n_permutations=20, seed=0)
    np.testing.assert_allclose(again.sd["shell"], null.sd["shell"])
    with pytest.raises(ValueError, match="at least 2"):
        permutation_null(values, graph, nb, [4], n_permutations=1)


def test_the_null_shuffles_within_each_slide():
    rng = np.random.default_rng(12)
    pos, graph = rng.uniform(0, 10, (200, 2)), np.repeat([0, 1], 100)
    values = np.column_stack([np.repeat([0.0, 5.0], 100), rng.normal(size=200)])  # gene 0: constant within each slide
    nb = sample_neighbours(pos, graph, k_max=4, n_centres=None)
    null = permutation_null(values, graph, nb, [4], n_permutations=3)
    assert np.isnan(null.mean["shell"][0, 0]) and np.isfinite(null.mean["shell"][0, 1])


def test_the_half_point_is_interpolated_on_log_k_after_the_peak():
    ks = np.array([4, 8, 16])
    assert half_point([0.8, 0.6, 0.2], ks) == pytest.approx(8 * 2**0.5)
    assert half_point([0.8, 0.4, 0.1], ks) == pytest.approx(8.0)
    assert half_point([0.8, np.nan, 0.2], ks) == pytest.approx(4 * 4 ** (2 / 3))
    assert half_point([0.8, 0.7, 0.6], ks) == np.inf
    # a curve that rises first (a lattice's diagonal neighbours) halves from its peak, not from its first value
    assert half_point([0.1, 0.6, 0.2], ks) == pytest.approx(8 * 2**0.75)
    assert half_point([-0.1, 0.5, 0.2], ks) == pytest.approx(8 * 2 ** (5 / 6))
    assert half_point([np.nan, 0.5, 0.2], ks) == pytest.approx(8 * 2 ** (5 / 6))
    assert np.isnan(half_point([-0.1, -0.2, -0.3], ks)) and np.isnan(half_point([np.nan] * 3, ks))
    # a dip before the peak is not where the curve halves
    assert half_point([0.5, 0.1, 0.9, 0.3], np.array([4, 8, 16, 32])) == pytest.approx(16 * 2**0.75)


def test_length_scales_read_the_pattern_and_convert_to_um():
    ks = np.array([4, 8, 16, 32])
    reach = np.array([100.0, 141.0, 200.0, 283.0])
    genes = ["decays", "flat", "anti", "constant", "stays", "rises", "marginal", "narrow"]
    shell = np.column_stack(
        [
            [0.8, 0.6, 0.2, 0.0],
            [0.002, 0.001, 0.0, 0.0],
            [-0.5, -0.1, 0.0, 0.0],
            [np.nan] * 4,
            [0.9, 0.85, 0.8, 0.7],
            [0.01, 0.6, 0.2, 0.05],
            [0.032, 0.0, 0.0, 0.0],  # 3.2 null SDs: above 3, not above 3 corrected for the 4 shells (3.40)
            [0.02, 0.0, 0.0, 0.0],  # 20 of its own SDs, 2 of the floor
        ]
    )
    curves = MoranCurves(ks, reach, shell, shell.copy())
    sd = np.full((4, 8), 0.01)
    sd[:, 7] = 0.001  # an SD from a few shuffles that came out low
    null = MoranNull(
        mean={f: np.zeros((4, 8)) for f in ("shell", "cumulative")}, sd={f: sd.copy() for f in ("shell", "cumulative")}
    )
    t = length_scales(curves, null, genes, z=3.0)
    assert t["pattern"].tolist() == [VARIABLE, NONE, ANTI, CONSTANT, VARIABLE, VARIABLE, NONE, NONE]
    assert t.loc["decays", "ell_k"] == pytest.approx(8 * 2**0.5)
    assert t.loc["decays", "ell_um"] == pytest.approx(math.sqrt(141.0 * 200.0))
    assert t.loc["stays", "ell_k"] == np.inf and t.loc["stays", "ell_um"] == np.inf
    assert t.loc["rises", "ell_k"] == pytest.approx(8 * 2**0.75)
    assert t.loc["rises", "k_peak"] == 8 and t.loc["rises", "i_peak"] == 0.6 and t.loc["decays", "k_peak"] == 4
    assert t.loc["decays", "z_first"] == pytest.approx(80.0) and t.loc["narrow", "z_first"] == pytest.approx(2.0)
    assert t.loc["decays", "i_last"] == 0.0 and t.loc["stays", "i_last"] == 0.7
    assert t.loc["decays", "auc"] == pytest.approx(0.4)
    assert np.isnan(t.loc["flat", "ell_k"]) and np.isnan(t.loc["anti", "ell_um"])
    with pytest.raises(ValueError, match="names"):
        length_scales(curves, null, ["too", "few"])


# --------------------------------------------------------------------------- the shares


def test_a_parts_share_is_its_part_of_the_variance_of_each_genes_prediction():
    rng = np.random.default_rng(10)
    n = 50
    s = rng.normal(size=n)
    local = np.column_stack([np.full(n, 2.0), s, 0.5 * s, -s, np.ones(n)])
    glob = np.column_stack([s, np.full(n, -1.0), 0.5 * s, 2.0 * s, np.ones(n)])
    shares = part_shares({"local": local, "global": glob})
    np.testing.assert_allclose(shares["share_global"].to_numpy()[:4], [1.0, 0.0, 0.5, 2.0], atol=1e-12)
    np.testing.assert_allclose(shares["share_local"].to_numpy()[:4], [0.0, 1.0, 0.5, -1.0], atol=1e-12)
    np.testing.assert_allclose((shares["share_local"] + shares["share_global"]).to_numpy()[:4], 1.0)
    np.testing.assert_allclose(shares["norm_global"].to_numpy()[3], 2.0)
    assert shares.iloc[4].isna().all()  # a prediction that does not move has no shares
    # entries outside `keep` change nothing
    keep = np.ones((n, 5), dtype=bool)
    keep[:10] = False
    noisy = local.copy()
    noisy[:10] = rng.normal(scale=100.0, size=(10, 5))
    kept = part_shares({"local": noisy, "global": glob}, keep)
    np.testing.assert_allclose(
        kept.to_numpy()[:4], part_shares({"local": local[10:], "global": glob[10:]}).to_numpy()[:4], atol=1e-12
    )
    with pytest.raises(ValueError, match="no parts"):
        part_shares({})


def test_the_gene_cosine_is_centred_on_the_given_mean():
    true = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    pred = np.array([[2.0, 2.0], [3.0, 2.0], [4.0, 2.0]])
    np.testing.assert_allclose(gene_cosine(pred, true, np.array([3.0, 2.0]))[0], 1.0)
    assert np.isnan(gene_cosine(pred, true, np.array([3.0, 2.0]))[1])  # a gene at its mean everywhere
    assert gene_cosine(pred, true)[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- the folds and the model

G, E = 6, 16


def _graphs(sizes=(14, 18), seed=0, names=True):
    """Lattice slides with a 4-nearest-neighbour graph, positive expression, positions in µm and unique cell names."""
    rng = np.random.default_rng(seed)
    datas, start = [], 1
    for n in sizes:
        side = math.ceil(math.sqrt(n))
        pos = np.array([(k % side, k // side) for k in range(n)], dtype=float) * 100.0
        _, j = cKDTree(pos).query(pos, k=5)
        src = np.repeat(np.arange(n), 4)
        dst = j[:, 1:].ravel()
        edge_index = torch.tensor(np.stack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long)
        d = Data(
            x=torch.tensor(rng.random((n, G)) + 0.5, dtype=torch.float32),
            edge_index=edge_index,
            pos=torch.tensor(pos, dtype=torch.float32),
        )
        if names:
            d.obs_names = torch.arange(start, start + n, dtype=torch.float64)
        start += n
        datas.append(d)
    return datas


def _stir(module, scale=0.1, seed=7):
    """Every parameter moved by seeded noise: a fresh module's zero biases and identity norms hide missing paths."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=gen))
    return module


def _module(additive=True, residual=False, rate=0.3):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder = True
    cfg.model.decoder.additive = additive
    cfg.model.decoder.global_residual = residual
    cfg.optim.loss_weights["local"], cfg.optim.loss_weights["global"] = 0.0, 1.0
    cfg.dataset.name = "d"
    torch.manual_seed(0)
    module = DualDecoderCombinedModule(
        cfg=cfg,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=rate,
        mask_strategy="node",
    ).eval()
    return _stir(module)


FOLD = {
    "n_folds": 3,
    "prediction_task": "regression",
    "prediction_level": "node",
    "batch_size": 2,
    "device": "cpu",
    "seed": 0,
}


def test_the_folds_hide_every_cell_once_at_the_rate():
    data_list = [Data(x=torch.zeros(n, 2)) for n in (10, 23, 7)]
    np.random.seed(2)
    numpy_state = np.random.get_state()[1].copy()
    folds = cell_folds(data_list, 3, seed=0)
    assert np.array_equal(np.random.get_state()[1], numpy_state)
    for data, fold in zip(data_list, folds, strict=True):
        assert fold.shape == (data.num_nodes,) and set(np.unique(fold)) <= {0, 1, 2}
        counts = np.bincount(fold, minlength=3)
        assert counts.max() - counts.min() <= 1
    assert all(np.array_equal(a, b) for a, b in zip(folds, cell_folds(data_list, 3, seed=0), strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(folds, cell_folds(data_list, 3, seed=1), strict=True))
    with pytest.raises(ValueError, match="at least 2 folds"):
        cell_folds(data_list, 1)


def test_every_cell_gets_one_row_at_its_cell_and_the_parts_add_up():
    data_list = _graphs()
    data_list[0].mask = torch.zeros(14, dtype=torch.bool)  # a mask of the caller's comes back; the other graph has none
    before = data_list[0].mask.clone()
    fold = collect_fold_parts(_module(), data_list, **FOLD)
    assert len(fold.true) == 32 and sorted(np.rint(fold.cells).astype(int).tolist()) == list(range(1, 33))
    for r, name in enumerate(np.rint(fold.cells).astype(int)):
        g, j = (0, name - 1) if name <= 14 else (1, name - 15)
        assert fold.graph[r] == g
        np.testing.assert_allclose(fold.true[r], data_list[g].x[j].numpy(), atol=1e-6)
        np.testing.assert_allclose(fold.pos[r], data_list[g].pos[j].numpy())
    assert set(fold.parts) == {"local", "global"}
    np.testing.assert_allclose(fold.parts["local"] + fold.parts["global"], fold.prediction, atol=1e-5)
    assert np.abs(fold.parts["global"]).max() > 1e-3 and np.abs(fold.parts["local"]).max() > 1e-3
    assert torch.equal(data_list[0].mask, before) and "gene_mask" not in data_list[0]
    assert "mask" not in data_list[1] and "gene_mask" not in data_list[1]


def test_a_cells_own_values_never_reach_its_row_but_its_neighbours_do():
    data_list, module = _graphs(), _module()
    base = collect_fold_parts(module, data_list, **FOLD)
    folds = cell_folds(data_list, 3, seed=0)
    j = 5
    row = int(np.flatnonzero(np.rint(base.cells).astype(int) == j + 1)[0])

    def changed(cell):
        moved = [d.clone() for d in data_list]
        moved[0].x[cell] += 10.0
        return collect_fold_parts(module, moved, **FOLD)

    own = changed(j)
    np.testing.assert_allclose(own.prediction[row], base.prediction[row], atol=1e-6)
    for part in ("local", "global"):
        np.testing.assert_allclose(own.parts[part][row], base.parts[part][row], atol=1e-6)
    assert np.abs(own.prediction - base.prediction).max() > 1e-3  # where the cell is visible, it is read
    edges = data_list[0].edge_index.numpy()
    neighbours = edges[1][edges[0] == j]
    visible = [int(k) for k in neighbours if folds[0][k] != folds[0][j]]
    assert visible
    other = changed(visible[0])
    assert np.abs(other.parts["local"][row] - base.parts["local"][row]).max() > 1e-4
    assert np.abs(other.prediction[row] - base.prediction[row]).max() > 1e-4


def test_a_prediction_without_parts_or_cells_without_names_is_refused():
    with pytest.raises(ValueError, match="not a sum of parts"):
        collect_fold_parts(_module(additive=False), _graphs(), **FOLD)
    with pytest.raises(ValueError, match="obs_names"):
        collect_fold_parts(_module(), _graphs(names=False), **FOLD)
    residual = collect_fold_parts(_module(additive=False, residual=True), _graphs(), **FOLD)
    np.testing.assert_allclose(residual.parts["local"] + residual.parts["global"], residual.prediction, atol=1e-5)


def test_the_references_of_the_same_passes_come_along():
    data_list = _graphs()
    fold = collect_fold_parts(_module(), data_list, spatial_unit_um=1.0, references={"kernels_um": (150.0,)}, **FOLD)
    assert set(fold.references) == {"ref_slide_mean", "ref_kernel_150um"}
    assert all(value.shape == fold.true.shape for value in fold.references.values())
    # a hidden cell's slide mean is the mean of the visible cells of its slide in its own fold
    folds = cell_folds(data_list, 3, seed=0)
    name = int(np.rint(fold.cells[0]))
    g, j = (0, name - 1) if name <= 14 else (1, name - 15)
    visible = folds[g] != folds[g][j]
    np.testing.assert_allclose(fold.references["ref_slide_mean"][0], data_list[g].x[visible].mean(0).numpy(), atol=1e-5)


# --------------------------------------------------------------------------- the tables, the figures and the report


def _separable(side=24, seed=11):
    """Two lattice slides: rough genes on a "local part", smooth genes on a "global part", noise genes on neither."""
    rng = np.random.default_rng(seed)
    pos, graph = _lattice(side=side, slides=2)
    sigmas = {"r1": 0.7, "r2": 0.9, "r3": 1.1, "s1": 4.0, "s2": 5.0, "s3": 6.0}
    genes = [*sigmas, "n1", "n2"]
    truth = np.zeros((len(pos), len(genes)))
    local = np.zeros_like(truth)
    glob = np.zeros_like(truth)
    for k, gene in enumerate(genes):
        for g in (0, 1):
            rows = graph == g
            if gene in sigmas:
                f = _field(side, sigmas[gene], rng)
                held, other = (local, glob) if gene.startswith("r") else (glob, local)
                held[rows, k] = f
                other[rows, k] = 0.05 * rng.normal(size=rows.sum())
                truth[rows, k] = f + 0.3 * rng.normal(size=rows.sum())
            else:
                truth[rows, k] = rng.normal(size=rows.sum())
                local[rows, k], glob[rows, k] = 0.05 * rng.normal(size=(2, rows.sum()))
    planted = pd.Series({gene: 100.0 * s for gene, s in sigmas.items()})
    design = pd.Series(
        {
            **dict.fromkeys(("r1", "r2", "r3"), "local"),
            **dict.fromkeys(("s1", "s2", "s3"), "global"),
            "n1": "none",
            "n2": "none",
        }
    )
    return pos, graph, genes, truth, local, glob, planted, design


def test_the_tables_correlate_the_scale_with_the_share():
    pos, graph, genes, truth, local, glob, planted, design = _separable()
    shares = part_shares({"local": local, "global": glob})
    res = gene_scales(
        {TRUTH: truth, "local": local, "global": glob},
        graph,
        pos,
        genes,
        columns=shares,
        n_centres=None,
        n_permutations=5,
        ks=(4, 8, 16, 32, 64, 128, 256),
    )
    tables = scale_tables(res, planted_um=planted, design=design)
    summary = tables["gene_scales_summary"].iloc[0]
    assert summary["n_genes"] == 8 and summary["n_variable"] == 6
    assert summary["rho_ell_share"] > 0.7 and summary["global_majority"] == pytest.approx(0.5)
    assert summary["rho_ell_planted"] > 0.7 and summary["rho_share_planted"] > 0.7
    by_design = tables["gene_scales_by_design"]
    assert set(by_design.index) == {"local", "global", "none"}
    assert by_design.loc["global", "global_majority"] == 1.0 and by_design.loc["local", "global_majority"] == 0.0
    assert by_design.loc["none", "n_variable"] == 0
    per_gene = tables["gene_scales"]
    assert per_gene["share_global"].is_monotonic_decreasing
    assert set(per_gene.index[:3]) == {"s1", "s2", "s3"} and per_gene.loc["s1", "planted_um"] == 400.0
    # the parts' own curves: the local part of a rough gene drops, the global part of a smooth one stays up
    j_r, j_s = genes.index("r1"), genes.index("s3")
    assert res.curves["local"].shell[3, j_r] < 0.5 * res.curves["local"].shell[0, j_r]
    assert res.curves["global"].shell[3, j_s] > 0.5 * res.curves["global"].shell[0, j_s]
    assert "design" not in scale_tables(res) and "gene_scales_by_design" not in scale_tables(res)


def test_a_scale_beyond_the_sizes_ranks_as_the_longest():
    genes = ["a", "b", "c", "d", "e"]
    table = pd.DataFrame(
        {
            "pattern": [VARIABLE] * 4 + [NONE],
            "ell_um": [100.0, 200.0, np.inf, np.inf, np.nan],
            "share_global": [0.1, 0.2, 0.8, 0.9, 0.95],
        },
        index=pd.Index(genes, name="gene"),
    )
    summary = scale_tables(GeneScales(genes, {}, MoranNull({}, {}), table))["gene_scales_summary"].iloc[0]
    assert summary["n_rho"] == 4 and summary["rho_ell_share"] == pytest.approx(
        np.corrcoef([1, 2, 3.5, 3.5], [1, 2, 3, 4])[0, 1]
    )
    assert summary["global_majority"] == 0.5 and summary["n_variable"] == 4
    # planted scales in an order of their own: each correlation reads its own pair of columns
    planted = pd.Series({"a": 300.0, "b": 100.0, "c": 400.0, "d": 200.0, "e": 50.0})
    found = scale_tables(GeneScales(genes, {}, MoranNull({}, {}), table), planted_um=planted)[
        "gene_scales_summary"
    ].iloc[0]
    assert found["n_planted"] == 4
    assert found["rho_share_planted"] == pytest.approx(np.corrcoef([1, 2, 3, 4], [3, 1, 4, 2])[0, 1])
    assert found["rho_ell_planted"] == pytest.approx(np.corrcoef([1, 2, 3.5, 3.5], [3, 1, 4, 2])[0, 1])


def test_the_classes_follow_the_share_or_the_given_design():
    pos, graph, genes, truth, local, glob, _, design = _separable()
    res = gene_scales(
        {TRUTH: truth},
        graph,
        pos,
        genes,
        columns=part_shares({"local": local, "global": glob}),
        n_centres=None,
        n_permutations=3,
        ks=(4, 16, 64),
    )
    by_share = assignment(res)
    assert (
        by_share[["r1", "r2", "r3"]].tolist() == ["local"] * 3
        and by_share[["s1", "s2", "s3"]].tolist() == ["global"] * 3
    )
    assert by_share[["n1", "n2"]].tolist() == ["not variable"] * 2
    assert assignment(res, classes=design)[["n1", "s1"]].tolist() == ["not variable", "global"]
    picked = pick_genes(res, 4)
    assert len(picked) == 4 and set(picked[:2]) <= {"s1", "s2", "s3"} and set(picked[2:]) <= {"r1", "r2", "r3"}
    plain = gene_scales({TRUTH: truth}, graph, pos, genes, n_centres=None, n_permutations=3, ks=(4, 16, 64))
    with pytest.raises(ValueError, match="share_global"):
        assignment(plain)


def test_the_figures_render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, graph, genes, truth, local, glob, planted, design = _separable(side=16)
    shares = part_shares({"local": local, "global": glob})
    res = gene_scales(
        {TRUTH: truth, "local": local, "global": glob},
        graph,
        pos,
        genes,
        columns=shares,
        n_centres=None,
        n_permutations=3,
        ks=(4, 8, 16, 32, 64),
    )
    res.table["planted_um"] = planted.reindex(res.table.index)
    fig = plot_scale_curves(res, title="a run")
    assert len(fig.axes[0].lines) >= len(genes)
    fig = plot_scale_curves(
        gene_scales({TRUTH: truth}, graph, pos, genes, n_centres=None, n_permutations=3, ks=(4, 16)),
        classes=design,
        form="cumulative",
    )
    assert fig.axes[0].get_legend() is not None
    fig = plot_part_curves(res, ["r1", "s1", "n1"], title="parts")
    assert sum(ax.get_visible() for ax in fig.axes) >= 3
    fold = FoldParts(truth, local + glob, {"local": local, "global": glob}, graph, pos, np.arange(len(pos)), {})
    fig = plot_part_maps(fold, ["r1", "s1"], genes, shares=res.table, graph=1)
    assert len(fig.axes) == 2 * 4 + 1  # genes x (expression, prediction, two parts) and the colour bar
    with pytest.raises(ValueError, match="none of the genes"):
        plot_part_curves(res, ["missing"])
    plt.close("all")


def test_the_report_summary_carries_every_arms_gene_scales(tmp_path):
    path = tmp_path / "results.md"
    for arm, rho in (("a", 0.6), ("b", 0.1)):
        summary = pd.DataFrame([{"n_genes": 8, "n_variable": 6, "rho_ell_share": rho, "global_majority": 0.5}])
        by_design = pd.DataFrame(
            {"design": ["local", "global"], "n_genes": [3, 3], "global_majority": [0.0, 1.0]}
        ).set_index("design")
        report.append_sections(
            path, arm, report.compact_tables({"gene_scales_summary": summary, "gene_scales_by_design": by_design})
        )
    sections = report.read_sections(path)
    assert ("a", "gene_scales_summary") in sections and ("b", "gene_scales_by_design") in sections
    found = dict(report.summary_tables(sections, ["a", "b"]))
    titles = [t for t in found if "gene scales" in t]
    assert len(titles) == 2
    table = found[next(t for t in titles if "by design" not in t)]
    assert table.loc["a", "rho_ell_share"] == pytest.approx(0.6) and table.loc["b", "rho_ell_share"] == pytest.approx(
        0.1
    )
    by_design = found[next(t for t in titles if "by design" in t)]
    assert by_design.loc[("b", "global"), "global_majority"] == 1.0
    assert "gene_scales_summary" in report.NOTES and "gene_scales" in report.NOTES
