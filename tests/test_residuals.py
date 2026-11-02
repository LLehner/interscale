"""Label residuals before the scale sets (yardsticks_plan.md, Y0): ``evaluation.residuals``.

Pinned here:

* **the blocks** -- one-hot labels (NaN its own level), composition shares that sum to 1 over the neighbours, never the cell
  itself, never across graphs; a cell without neighbours gets its graph's mean;
* **the fits** -- a label residual is the value minus its per-graph label mean (``fit="graph"``) or the closed-form
  pooled fit with graph intercepts; graphs are never mixed under ``graph``; stages are cumulative; R² and adjusted R² as
  written; a gene a stage explains exactly becomes 0, not rounding noise;
* **the order-free split** -- Shapley shares add up to the full R², do not depend on the order, equal each block's own R²
  for orthogonal blocks and split two copies of one block evenly;
* **the scale sets on residuals** -- a pure label gene loses its Moran's I, a gradient within the labels keeps it; the
  summary lines the two stages up;
* every global generator is left alone.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from interscale.evaluation.gene_scales import hop_neighbourhoods, local_moran
from interscale.evaluation.residuals import (
    GRAPH,
    UNLABELLED,
    Block,
    composition_block,
    knn_neighbourhood,
    label_block,
    label_summary,
    neighbourhood_mean,
    residual_scale_sets,
    residualise,
    set_crosstab,
    shapley_r2,
)


def _cells(n_per=60, n_graphs=3, seed=0):
    rng = np.random.default_rng(seed)
    graph = np.repeat(np.arange(n_graphs), n_per)
    pos = rng.uniform(0, 10, size=(len(graph), 2))
    labels = rng.choice(["a", "b", "c"], size=len(graph))
    return rng, graph, pos, labels


# ------------------------------------------------------------------------------- blocks


def test_label_block_one_hot_and_nan_level():
    block = label_block(np.array(["x", None, "y", "x", np.nan], dtype=object), name="t")
    assert block.kind == "label" and block.name == "t"
    assert set(block.levels) == {"x", "y", UNLABELLED}
    assert np.array_equal(block.design.sum(axis=1), np.ones(5))
    col = list(block.levels).index(UNLABELLED)
    assert np.array_equal(block.design[:, col], [0, 1, 0, 0, 1])


def test_knn_neighbourhood_excludes_self_and_other_graphs():
    _, graph, pos, _ = _cells()
    a = knn_neighbourhood(pos, graph, 5)
    assert a.diagonal().sum() == 0
    rows, cols = a.nonzero()
    assert np.all(graph[rows] == graph[cols])
    assert np.array_equal(np.asarray(a.sum(axis=1)).ravel(), np.full(len(graph), 5.0))
    # the nearest other cell of each graph is among them
    for i in [0, 70, 150]:
        same = np.flatnonzero(graph == graph[i])
        d = np.linalg.norm(pos[same] - pos[i], axis=1)
        d[same == i] = np.inf
        assert a[i, same[d.argmin()]] == 1


def test_knn_neighbourhood_with_interleaved_graphs():
    rng = np.random.default_rng(5)
    graph = rng.permutation(np.repeat([0, 1, 2], 30))
    pos = rng.uniform(0, 10, size=(90, 2))
    a = knn_neighbourhood(pos, graph, 3).toarray()
    for i in range(90):
        same = np.flatnonzero((graph == graph[i]) & (np.arange(90) != i))
        d = np.linalg.norm(pos[same] - pos[i], axis=1)
        assert set(np.flatnonzero(a[i])) == set(same[np.argsort(d)[:3]])


def test_composition_rows_sum_to_one_and_count_neighbours():
    graph = np.array([0, 0, 0, 0, 1, 1])
    labels = np.array(["a", "b", "b", "a", "a", "b"])
    a = sp.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],  # no neighbour: its graph's mean
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
    )
    block = composition_block(labels, a, graph, name="near")
    lv = list(block.levels)
    comp = pd.DataFrame(block.design, columns=lv)
    assert np.allclose(comp.sum(axis=1), 1.0)
    assert np.allclose(comp.loc[0, ["a", "b"]], [0.0, 1.0])
    assert np.allclose(comp.loc[1, ["a", "b"]], [1.0, 0.0])
    assert np.allclose(comp.loc[2, ["a", "b"]], [0.5, 0.5])  # graph 0 holds a, b, b, a
    assert np.allclose(comp.loc[3, ["a", "b"]], [1 / 3, 2 / 3])
    assert np.allclose(comp.loc[4, ["a", "b"]], [0.0, 1.0])


def test_neighbourhood_ignores_the_diagonal_and_refuses_crossing_graphs():
    graph = np.array([0, 0, 1])
    feats = np.array([[1.0], [3.0], [5.0]])
    with_self = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float))
    assert np.allclose(neighbourhood_mean(with_self, feats, graph).ravel(), [3.0, 1.0, 5.0])
    crossing = sp.csr_matrix(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float))
    with pytest.raises(ValueError, match="different graphs"):
        neighbourhood_mean(crossing, feats, graph)


def test_block_refuses_bad_designs():
    with pytest.raises(ValueError, match="NaN"):
        Block("x", np.array([[np.nan]]), "label")
    with pytest.raises(ValueError, match=r"\[N, P\]"):
        Block("x", np.zeros(3), "label")


# ------------------------------------------------------------------------------- fits


def test_label_residual_is_value_minus_graph_label_mean():
    rng, graph, _, labels = _cells()
    values = rng.normal(size=(len(graph), 4)) + 3.0 * (labels == "a")[:, None]
    res = residualise(values, graph, [label_block(labels)], list("pqrs"))
    expect = values.copy()
    for g in np.unique(graph):
        for lab in np.unique(labels):
            m = (graph == g) & (labels == lab)
            expect[m] -= values[m].mean(axis=0)
    assert np.allclose(res.values["label"], expect, atol=1e-5)
    centred = values.copy()
    for g in np.unique(graph):
        centred[graph == g] -= values[graph == g].mean(axis=0)
    assert np.allclose(res.values[GRAPH], centred, atol=1e-5)
    assert np.allclose(res.r2[GRAPH], 0.0)
    r2 = 1 - (expect**2).sum(0) / (centred**2).sum(0)
    assert np.allclose(res.r2["label"], r2)
    # 2 free label coefficients per graph (3 levels, one taken by the intercept)
    assert res.n_params["label"] == 2 * 3
    dof_tot = len(graph) - 3
    assert np.allclose(res.r2_adjusted["label"], 1 - (1 - r2) * dof_tot / (dof_tot - 6))


def test_pooled_fit_matches_closed_form_with_graph_intercepts():
    rng, graph, _, labels = _cells()
    values = rng.normal(size=(len(graph), 3)) + 2.0 * (labels == "b")[:, None] + graph[:, None]
    res = residualise(values, graph, [label_block(labels)], list("pqr"), fit="pooled")
    x = np.hstack([np.eye(3)[graph], label_block(labels).design[:, 1:]])  # graph dummies + 2 labels
    coef, *_ = np.linalg.lstsq(x, values, rcond=None)
    assert np.allclose(res.values["label"], values - x @ coef, atol=1e-5)
    assert res.n_params["label"] == 2
    # pooled and per-graph differ when the label effect differs between graphs
    values2 = values + 5.0 * ((labels == "b") & (graph == 0))[:, None]
    a = residualise(values2, graph, [label_block(labels)], list("pqr"), fit="pooled")
    b = residualise(values2, graph, [label_block(labels)], list("pqr"), fit="graph")
    assert not np.allclose(a.values["label"], b.values["label"], atol=1e-3)


def test_graph_fit_never_mixes_graphs():
    rng, graph, _, labels = _cells()
    values = rng.normal(size=(len(graph), 2))
    a = residualise(values, graph, [label_block(labels)], ["p", "q"])
    other = values.copy()
    other[graph == 2] = rng.normal(size=((graph == 2).sum(), 2)) * 10 + 7
    b = residualise(other, graph, [label_block(labels)], ["p", "q"])
    keep = graph != 2
    assert np.array_equal(a.values["label"][keep], b.values["label"][keep])


def test_stages_are_cumulative_and_r2_rises():
    rng, graph, pos, labels = _cells()
    near = composition_block(labels, knn_neighbourhood(pos, graph, 6), graph, name="near")
    values = rng.normal(size=(len(graph), 5))
    res = residualise(values, graph, [label_block(labels), near], list("abcde"), keep=["near"])
    assert res.stages == [GRAPH, "label", "near"]
    assert list(res.values) == ["near"]
    assert np.all(res.r2["near"] >= res.r2["label"] - 1e-12)
    assert np.allclose(res.delta_r2()["near"], res.r2["near"] - res.r2["label"])
    joint = np.hstack([label_block(labels).design, near.design])
    one = residualise(values, graph, [Block("both", joint, "x")], list("abcde"))
    assert np.allclose(one.values["both"], res.values["near"], atol=1e-5)


def test_exactly_explained_gene_becomes_constant():
    rng, graph, pos, labels = _cells()
    marker = (labels == "a").astype(float) * 2.0 + graph  # a label mean per graph and nothing else
    values = np.column_stack([marker, rng.normal(size=len(graph))])
    res = residualise(values, graph, [label_block(labels)], ["m", "n"])
    assert np.all(res.values["label"][:, 0] == 0.0)
    assert np.isclose(res.r2.loc["m", "label"], 1.0)
    curves = local_moran(res.values["label"], graph, hop_neighbourhoods(knn_neighbourhood(pos, graph, 4), graph, 1))
    assert np.isnan(curves[0, 0]) and np.isfinite(curves[0, 1])


def test_residualise_refuses_bad_input():
    rng, graph, _, labels = _cells()
    values = rng.normal(size=(len(graph), 2))
    block = label_block(labels)
    with pytest.raises(ValueError, match="fit must be"):
        residualise(values, graph, [block], ["p", "q"], fit="nope")
    with pytest.raises(ValueError, match="distinct"):
        residualise(values, graph, [block, block], ["p", "q"])
    with pytest.raises(ValueError, match="distinct"):
        residualise(values, graph, [Block(GRAPH, block.design, "label")], ["p", "q"])
    with pytest.raises(ValueError, match="genes"):
        residualise(values, graph, [block], ["p"])
    with pytest.raises(ValueError, match="rows"):
        residualise(values, graph, [Block("x", block.design[:-1], "label")], ["p", "q"])
    with pytest.raises(ValueError, match="unknown stages"):
        residualise(values, graph, [block], ["p", "q"], keep=["foo"])


# ------------------------------------------------------------------------------- order-free split


def test_shapley_adds_up_and_ignores_order():
    rng, graph, pos, labels = _cells(n_per=80)
    region = np.where(pos[:, 0] > 5, "left", "right")
    near = composition_block(labels, knn_neighbourhood(pos, graph, 6), graph, name="near")
    blocks = [label_block(labels), label_block(region, name="region"), near]
    values = (
        rng.normal(size=(len(graph), 3))
        + 2.0 * (labels == "a")[:, None]
        + 1.5 * (region == "left")[:, None] * np.array([1, 0, 1])
        + 3.0 * near.design[:, [0]] * np.array([0, 1, 1])
    )
    a = shapley_r2(values, graph, blocks, list("pqr"))
    assert np.allclose(a[["label", "region", "near"]].sum(axis=1), a["total"])
    full = residualise(values, graph, blocks, list("pqr"))
    assert np.allclose(a["total"], full.r2["near"])
    b = shapley_r2(values, graph, blocks[::-1], list("pqr"))
    assert np.allclose(a[["label", "region", "near"]], b[["label", "region", "near"]])
    # the planted structure: gene q has no region effect, gene p no composition effect
    assert a.loc["q", "region"] < a.loc["p", "region"] and a.loc["p", "near"] < a.loc["q", "near"]


def test_shapley_orthogonal_blocks_and_duplicates():
    # Two balanced, crossed factors within each graph are orthogonal after centring.
    graph = np.repeat([0, 1], 40)
    f1 = np.tile(np.repeat(["u", "v"], 20), 2)
    f2 = np.tile(np.tile(["s", "t"], 20), 2)
    rng = np.random.default_rng(3)
    values = rng.normal(size=(80, 2)) + 2 * (f1 == "u")[:, None] + (f2 == "s")[:, None]
    b1, b2 = label_block(f1, "f1"), label_block(f2, "f2")
    s = shapley_r2(values, graph, [b1, b2], ["p", "q"])
    alone1 = residualise(values, graph, [b1], ["p", "q"]).r2["f1"]
    alone2 = residualise(values, graph, [b2], ["p", "q"]).r2["f2"]
    assert np.allclose(s["f1"], alone1) and np.allclose(s["f2"], alone2)
    dup = shapley_r2(values, graph, [b1, Block("copy", b1.design, "label")], ["p", "q"])
    assert np.allclose(dup["f1"], dup["copy"]) and np.allclose(dup["f1"] * 2, alone1)


def test_shapley_refuses_too_many_or_no_blocks():
    rng, graph, _, labels = _cells()
    values = rng.normal(size=(len(graph), 1))
    with pytest.raises(ValueError, match="takes 1 to"):
        shapley_r2(values, graph, [], ["p"])
    many = [label_block(labels, name=f"b{i}") for i in range(7)]
    with pytest.raises(ValueError, match="takes 1 to"):
        shapley_r2(values, graph, many, ["p"])


# ------------------------------------------------------------------------------- scale sets on residuals


def _two_label_tissue(seed=0):
    """Two slides of 30 x 30 cells; label = left/right half (blocks of tissue). Gene 'marker' is the label alone,
    'within' a smooth gradient inside each half that the label does not explain, the rest noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(30.0), np.arange(30.0))
    pos1 = np.column_stack([xs.ravel(), ys.ravel()])
    pos = np.vstack([pos1, pos1])
    graph = np.repeat([0, 1], len(pos1))
    labels = np.where(pos[:, 0] < 15, "L", "R")
    marker = 3.0 * (labels == "L") + 0.3 * rng.normal(size=len(pos))
    within = np.sin(pos[:, 1] / 4.0) + 0.3 * rng.normal(size=len(pos))
    noise = rng.normal(size=(len(pos), 8))
    values = np.column_stack([marker, within, noise])
    genes = ["marker", "within"] + [f"n{i}" for i in range(8)]
    return values, graph, pos, labels, genes


def test_scale_sets_on_residuals_remove_the_label_and_keep_the_rest():
    values, graph, pos, labels, genes = _two_label_tissue()
    adjacency = knn_neighbourhood(pos, graph, 4)
    res = residualise(values, graph, [label_block(labels)], genes)
    state = np.random.get_state()
    torch_state = torch.get_rng_state()
    table, found = residual_scale_sets(
        res, graph, pos, adjacency, genes, hops=2, n=1, n_permutations=5, n_centres=200, seed=1
    )
    assert np.array_equal(torch.get_rng_state(), torch_state)
    assert all(np.array_equal(x, y) for x, y in zip(np.random.get_state()[1:2], state[1:2], strict=True))
    assert set(found) == {GRAPH, "label"}
    summary = label_summary(table, res, after="label")
    assert summary.loc["marker", "i_local_before"] > 0.5
    assert summary.loc["marker", "kept_local"] < 0.2
    assert summary.loc["marker", "r2"] > 0.9
    assert summary.loc["within", "kept_local"] > 0.8
    assert summary.loc["marker", "i_global_before"] > summary.loc["marker", "i_global_after"]
    assert table[GRAPH, "candidate"]["marker"] and not table["label", "candidate"]["marker"]
    assert table["label", "candidate"]["within"]
    # the marker was in a set (two candidates, one per set) and is no candidate after; the noise never moves
    assert summary.loc["marker", "set_before"] != "none" and summary.loc["marker", "set_after"] == "none"
    assert summary.loc["marker", "moved"]
    assert not summary.loc[[f"n{i}" for i in range(8)], "moved"].any()
    tab = set_crosstab(summary)
    assert tab.loc["All", "All"] == len(genes)


def test_residual_scale_sets_refuses_stages_not_kept():
    values, graph, pos, labels, genes = _two_label_tissue()
    res = residualise(values, graph, [label_block(labels)], genes, keep=["label"])
    with pytest.raises(ValueError, match="no residuals kept"):
        residual_scale_sets(res, graph, pos, knn_neighbourhood(pos, graph, 4), genes, stages=[GRAPH])
