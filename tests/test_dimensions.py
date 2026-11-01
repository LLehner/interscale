"""The decoder-side readout of the embedding dimensions (``evaluation.dimensions``; roadmap task 2, ED0-ED2).

Pinned here:

* **ED0** -- the collector's rows are exactly what the linear decoder reads (its output is the step's context part), with the
  step's truth and hidden entries, for a GlobalModel and a LocalModel; with folds every cell is hidden once and the graphs' masks
  come back; a nonlinear decoder is refused; no global generator is touched;
* **ED1** -- on a planted readout the dimension that writes the signal tops all three readings, the shares sum to 1, ablating a
  dead dimension costs nothing; the singular directions do not move under a rotation of the dimensions, single ones do;
* **ED2** -- the planted dimension writes and tracks its genes, a noise one tracks nothing; the twin comparison lifts a learned
  dimension above an untrained one; every gene's priority is its row of ``W Σ_z^{1/2}`` (the variance of ``W z``), the planted genes
  first;
* **the same readout on a prediction** -- on a linear decoder's ``W z + b`` it equals the decoder readout (directions, loadings, signs,
  priority); per-group centring ignores a constant per group; the overlap of a subspace with itself is 1, with an orthogonal one 0;
* **varimax** -- recovers a sparse loading matrix from any rotation of it, keeps the subspace and each gene's summed squares, is
  blind to rescaled genes under Kaiser's normalisation; an option of both table builders, off by default.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interscale.evaluation.dimensions import (
    DimensionReadout,
    collect_dimensions,
    dimension_genes,
    dimension_importance,
    dimension_tables,
    effective_directions,
    gene_priority,
    prediction_readout,
    prediction_tables,
    rotated_tables,
    standardised_loadings,
    subspace_overlap,
    varimax,
    track_correlations,
    track_summary,
)

GENES = [f"g{k}" for k in range(8)]


def _planted(n=400, seed=0, noise=0.3, learned=True):
    """Dimension 0 writes genes 0-2, dimension 1 genes 3-5, dimension 2 nothing, dimension 3 is dead; the truth follows them."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    z[:, 3] = 0.0
    w = np.zeros((8, 4))
    w[:3, 0] = [2.0, 1.5, -1.0]  # one gene written with the opposite sign
    w[3:6, 1] = [0.6, 0.5, 0.4]
    w[:, 2] = rng.normal(scale=0.05, size=8)
    b = np.full(8, 1.0)
    true = z @ w.T + b + rng.normal(scale=noise, size=(n, 8))
    if not learned:  # an untrained twin: dimensions unrelated to the truth
        z = rng.normal(size=(n, 4))
    hidden = np.ones((n, 8), bool)
    hidden[: n // 4] = False  # visible cells: not read
    return DimensionReadout(z=z, true=true, hidden=hidden, weight=w, bias=b, graph=np.zeros(n, int))


# --------------------------------------------------------------------------- ED1


def test_the_planted_dimension_tops_every_reading_and_a_dead_one_costs_nothing():
    r = _planted()
    table = dimension_importance(r, center=r.bias)
    assert table["dim"].iloc[0] == 0 and np.isclose(table["share"].sum(), 1.0)
    by_dim = table.set_index("dim")
    assert by_dim.loc[0, "loading"] > by_dim.loc[1, "loading"] > by_dim.loc[2, "loading"]
    assert by_dim.loc[0, "ablation"] > by_dim.loc[1, "ablation"] > 0
    assert abs(by_dim.loc[3, "ablation"]) < 1e-12 and by_dim.loc[3, "share"] == 0
    assert np.isclose(table["cumulative_share"].iloc[-1], 1.0)


def test_a_shift_of_a_dimension_the_bias_takes_back_changes_no_reading():
    r = _planted()
    shifted = DimensionReadout(
        z=r.z + np.array([5.0, -3.0, 0.0, 0.0]), true=r.true, hidden=r.hidden, weight=r.weight,
        bias=r.bias - r.weight @ np.array([5.0, -3.0, 0.0, 0.0]), graph=r.graph,
    )
    assert np.allclose(shifted.z @ shifted.weight.T + shifted.bias, r.z @ r.weight.T + r.bias)
    a = dimension_importance(r, center=r.bias).set_index("dim")
    b = dimension_importance(shifted, center=r.bias).set_index("dim")
    assert np.allclose(a[["loading", "share", "ablation"]], b.loc[a.index, ["loading", "share", "ablation"]], atol=1e-9)


def test_the_standardised_loadings_read_the_hidden_cells_only():
    r = _planted()
    r.true[: len(r.true) // 4] += 1000.0  # visible cells: never read
    s = standardised_loadings(r)
    rows = r.hidden.any(1)
    want = r.weight[0, 0] * r.z[rows, 0].std() / r.true[rows, 0].std()
    assert np.isclose(s[0, 0], want)


def test_the_singular_directions_do_not_move_under_a_rotation_single_dimensions_do():
    r = _planted()
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))
    turned = DimensionReadout(z=r.z @ q, true=r.true, hidden=r.hidden, weight=r.weight @ q, bias=r.bias, graph=r.graph)
    assert np.allclose(turned.z @ turned.weight.T, r.z @ r.weight.T)  # the same predictions
    d0, g0 = effective_directions(r, GENES)
    d1, g1 = effective_directions(turned, GENES)
    assert np.allclose(d0["singular_value"], d1["singular_value"])
    assert np.allclose(np.abs(g0["loading"]), np.abs(g1["loading"]))
    assert not np.allclose(standardised_loadings(r), standardised_loadings(turned))
    assert set(g0[g0["direction"] == 0]["gene"].head(3)) == {"g0", "g1", "g2"}


# --------------------------------------------------------------------------- ED2


def test_the_planted_dimension_writes_and_tracks_its_genes_and_noise_tracks_nothing():
    r = _planted()
    rows, overlap = dimension_genes(r, GENES, top=3)
    write0 = rows[(rows["dim"] == 0) & (rows["kind"] == "write")]["gene"].tolist()
    track0 = rows[(rows["dim"] == 0) & (rows["kind"] == "track")]["gene"].tolist()
    assert write0 == ["g0", "g1", "g2"] and set(track0) == {"g0", "g1", "g2"}
    assert overlap.set_index("dim").loc[0, "shared"] == 3
    rho = track_correlations(r)
    assert np.all(np.abs(rho[2]) < 0.2) and rho[0, 0] > 0.8
    assert np.all(np.isnan(rho[3]))  # a dead dimension has no spread


def test_the_track_correlation_reads_each_genes_hidden_cells():
    r = _planted()
    r.hidden[:, 0] = False
    r.hidden[200:, 0] = True  # gene 0 hidden in another set of cells than the rest
    rho = track_correlations(r)
    cells = r.hidden[:, 0]
    import pandas as pd

    want = pd.Series(r.z[cells, 0]).corr(pd.Series(r.true[cells, 0]), method="spearman")
    assert np.isclose(rho[0, 0], want)


def test_a_learned_dimension_tracks_its_genes_above_an_untrained_twin():
    summary = track_summary(_planted(), twin=_planted(learned=False))
    twin_alone = track_summary(_planted(learned=False))
    assert np.isclose(summary["twin_mean"].iloc[0], twin_alone["track_strength"].mean())
    assert np.allclose(summary["lift"], summary["track_strength"] - summary["twin_mean"], equal_nan=True)
    assert summary.set_index("dim").loc[0, "lift"] > 0.5
    assert summary.set_index("dim").loc[2, "lift"] < 0.2
    tables = dimension_tables(_planted(), GENES, center=np.ones(8), twin=_planted(learned=False), n_dims=2)
    assert set(tables) == {
        "dim_importance", "dim_directions", "dim_direction_genes", "dim_genes", "dim_overlap", "dim_tracks", "dim_gene_priority"
    }
    assert set(tables["dim_genes"]["dim"]) == {0, 1}


# --------------------------------------------------------------------------- ED0 through the models


def _vk():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graphs(vk):
    graphs = [vk._lattice(6, seed=6, hidden=vk.HIDDEN), vk._lattice(5, seed=5, hidden=[0, 1, 2, 7])]
    for i, g in enumerate(graphs):
        g.obs_names = torch.arange(g.num_nodes) + 100 * i  # unique names across the list
    return graphs


def _global(vk):
    return vk._stir(vk._build(vk._config("dense", visible=False))).eval()


def _local(vk):
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.model.local_model import LocalModel

    cfg = vk._config("dense", visible=False)
    cfg.model.local_component.name = "GCN"
    cfg = get_local_component_cfg(cfg, "GCN")
    adata = vk._build.__globals__["AnnData"](X=np.zeros((40, vk.G), dtype=np.float32))
    import pandas as pd

    adata.var_names = [f"g{k}" for k in range(vk.G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 500, (40, 2))
    LocalModel._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(0)
    return vk._stir(LocalModel(adata, cfg=cfg).module).eval()


@pytest.mark.parametrize("kind", ["global", "local"])
def test_the_rows_are_what_the_decoder_reads_with_the_steps_truth_and_hidden_entries(kind):
    from torch_geometric.data import Batch

    vk = _vk()
    module = _global(vk) if kind == "global" else _local(vk)
    graphs = _graphs(vk)
    state = torch.random.get_rng_state()
    r = collect_dimensions(module, graphs, batch_size=2, device=torch.device("cpu"))
    assert torch.equal(state, torch.random.get_rng_state())
    with torch.no_grad():
        random.seed(0)
        out = module._common_step(Batch.from_data_list(graphs), "regression", "node")
    decoded = r.z @ r.weight.T + r.bias
    assert np.allclose(decoded, out.y_pred.double().numpy(), atol=1e-5)  # no other part here: the decoder alone
    assert np.array_equal(r.hidden, out.entry_mask.numpy()) and np.allclose(r.true, out.y_true.numpy())
    assert set(np.unique(r.graph)) == {0, 1}


def test_with_folds_every_cell_is_hidden_once_and_the_masks_come_back():
    vk = _vk()
    module = _global(vk)
    graphs = _graphs(vk)
    masks = [g.mask.clone() for g in graphs]
    r = collect_dimensions(module, graphs, batch_size=2, device=torch.device("cpu"), folds=3)
    assert len(r.z) == sum(g.num_nodes for g in graphs) and r.hidden.all()
    assert len(set(zip(r.graph.tolist(), r.cells.tolist(), strict=True))) == len(r.z)
    assert all(torch.equal(g.mask, m) for g, m in zip(graphs, masks, strict=True))
    assert not any("gene_mask" in g for g in graphs)


def test_a_nonlinear_decoder_is_refused():
    vk = _vk()
    cfg = vk._config("dense", visible=False)
    cfg.model.decoder.type = "nonlinear"
    module = vk._build(cfg)
    with pytest.raises(ValueError, match="linear decoder"):
        collect_dimensions(module, _graphs(vk), batch_size=2, device=torch.device("cpu"))


# --------------------------------------------------------------------------- gene priority and the readout on predictions


def test_every_genes_priority_is_its_row_of_the_whitened_decoder_and_the_planted_genes_lead():
    r = _planted()
    table = gene_priority(r, GENES)
    rows = r.hidden.any(axis=1)
    centred = r.z[rows] - r.z[rows].mean(axis=0)
    root = np.linalg.cholesky(centred.T @ centred / (len(centred) - 1) + 1e-300 * np.eye(4))
    expected = ((r.weight @ root) ** 2).sum(axis=1)  # any square root: row norms agree
    by_gene = table.set_index("gene")
    assert np.allclose(by_gene.loc[GENES, "variance"], expected)
    assert np.allclose(by_gene.loc[GENES, "variance"], (r.z[rows] @ r.weight.T).var(axis=0, ddof=1))
    assert list(table["gene"].head(3)) == ["g0", "g1", "g2"] and list(table["rank"]) == list(range(1, 9))
    assert np.isclose(table["share"].sum(), 1.0) and np.isclose(table["cumulative_share"].iloc[-1], 1.0)
    truth = np.where(r.hidden[rows], r.true[rows], np.nan)
    assert np.allclose(by_gene.loc[GENES, "std_ratio"], np.sqrt(expected) / np.nanstd(truth, axis=0, ddof=1))
    assert "dim_gene_priority" in dimension_tables(r, GENES, center=r.bias)


def test_the_readout_of_a_linear_decoders_prediction_is_the_decoder_readout():
    r = _planted()
    pred = r.z @ r.weight.T + r.bias
    read = prediction_readout(pred, r.hidden, GENES, true=r.true)
    d0, g0 = effective_directions(r, GENES)
    rank = 3  # dimension 3 is dead: the remaining singular values are rounding
    assert np.allclose(read["directions"]["singular_value"][:rank], d0["singular_value"][:rank])
    top = lambda g: g[g["direction"] < rank].sort_values(["direction", "rank"])  # noqa: E731
    assert list(top(read["genes"])["gene"]) == list(top(g0)["gene"])
    assert np.allclose(top(read["genes"])["loading"], top(g0)["loading"])  # signs too: both fixed the same way
    a, b = read["priority"], gene_priority(r, GENES)
    assert list(a["gene"]) == list(b["gene"])
    assert np.allclose(a[["share", "variance", "std_ratio"]], b[["share", "variance", "std_ratio"]])


def test_a_constant_per_group_changes_nothing_within_groups_but_moves_the_pooled_readout():
    r = _planted()
    pred = r.z @ r.weight.T + r.bias
    groups = np.arange(len(pred)) % 2
    shifted = pred + np.where(groups[:, None] == 1, 4.0, -1.0) * np.linspace(1, 2, 8)
    a = prediction_readout(pred, r.hidden, GENES, groups=groups)
    b = prediction_readout(shifted, r.hidden, GENES, groups=groups)
    assert np.allclose(a["priority"]["variance"], b["priority"]["variance"])
    assert np.allclose(a["singular_values"], b["singular_values"])
    pooled = prediction_readout(shifted, r.hidden, GENES)
    assert pooled["singular_values"][0] > 2 * a["singular_values"][0]
    # the visible rows are not read: changing them changes nothing
    noisy = shifted.copy()
    noisy[~r.hidden.any(axis=1)] += 100.0
    c = prediction_readout(noisy, r.hidden, GENES, groups=groups)
    assert np.allclose(c["priority"]["variance"], b["priority"]["variance"])


def test_subspace_overlap_is_one_for_the_same_span_and_zero_for_an_orthogonal_one():
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(8, 8)))
    turned = q[:, :3] @ np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    assert np.isclose(subspace_overlap(q[:, :3], turned, 3), 1.0)
    assert np.isclose(subspace_overlap(q[:, :3], q[:, 3:6], 3), 0.0)
    assert np.isclose(subspace_overlap(q[:, :2], q[:, [0, 5]], 2), 0.5)


def test_prediction_tables_read_every_head_and_the_truth():
    r = _planted()
    pred = r.z @ r.weight.T + r.bias
    other = pred[:, ::-1].copy()
    tables = prediction_tables({"global": pred, "ref_nearest_8": other}, r.true, r.hidden, GENES, n_directions=2)
    assert set(tables["pred_directions"]["head"]) == {"global", "ref_nearest_8", "truth"}
    assert tables["pred_directions"].groupby("head").size().max() == 2
    assert set(tables["pred_direction_genes"]["direction"]) == {0, 1}
    wide = tables["pred_gene_priority"]
    assert len(wide) == 8 and wide["gene"].iloc[0] == "g0"
    assert {"global_share", "ref_nearest_8_std_ratio", "truth_share"} <= set(wide.columns)
    assert np.allclose(wide.set_index("gene")["truth_std_ratio"], 1.0)
    ov = tables["pred_overlap"].set_index(["head", "against", "k"])["overlap"]
    assert ov.loc[("global", "truth", 1)] > 0.9  # the truth follows the planted dimension
    assert len(tables["pred_overlap"]) == 3 * 3


# --------------------------------------------------------------------------- varimax


def _blocks(seed=0):
    """Three gene programs on 12 genes, four genes each, of different strength; returned with a random rotation of them."""
    rng = np.random.default_rng(seed)
    sparse = np.zeros((12, 3))
    for c, scale in enumerate((3.0, 2.0, 1.0)):
        sparse[4 * c : 4 * c + 4, c] = scale * rng.uniform(0.7, 1.3, size=4)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return sparse, sparse @ q


def _matched(found, sparse):
    """``found``'s columns put in ``sparse``'s order and sign (by the largest absolute inner product)."""
    out = np.zeros_like(sparse)
    for c in range(sparse.shape[1]):
        dots = sparse[:, c] @ found
        j = int(np.argmax(np.abs(dots)))
        out[:, c] = np.sign(dots[j]) * found[:, j]
    return out


def test_varimax_recovers_a_sparse_structure_from_any_rotation():
    sparse, mixed = _blocks()
    assert not np.allclose(np.abs(mixed), np.abs(sparse), atol=0.1)  # the mixing hides the programs
    found, rotation = varimax(mixed)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
    assert np.allclose(_matched(found, sparse), sparse, atol=1e-4)  # to the convergence tolerance


def test_the_rotation_keeps_the_subspace_and_every_genes_summed_squares():
    _, mixed = _blocks(seed=1)
    found, _ = varimax(mixed)
    assert np.allclose((found**2).sum(axis=1), (mixed**2).sum(axis=1))
    qa, qb = np.linalg.qr(mixed)[0], np.linalg.qr(found)[0]
    assert np.isclose(subspace_overlap(qa, qb, 3), 1.0)
    one, eye = varimax(mixed[:, :1])
    assert np.allclose(one, mixed[:, :1]) and np.allclose(eye, 1.0)


def test_kaisers_normalisation_ignores_how_each_gene_is_scaled_and_plain_varimax_does_not():
    _, mixed = _blocks(seed=2)
    # Not a simple structure (a perfectly simple one is found whatever the genes' scales): every gene loads a little everywhere.
    mixed = mixed + np.random.default_rng(6).normal(scale=0.5, size=mixed.shape)
    scale = np.random.default_rng(5).uniform(0.05, 3.0, size=(12, 1))
    _, r0 = varimax(mixed, kaiser=True)
    _, r1 = varimax(mixed * scale, kaiser=True)
    assert np.allclose(np.abs(r0), np.abs(r1), atol=1e-6)
    _, p0 = varimax(mixed)
    _, p1 = varimax(mixed * scale)
    assert not np.allclose(np.abs(p0), np.abs(p1), atol=1e-3)


def test_the_rotated_tables_find_the_programs_in_a_prediction_and_are_an_option():
    sparse, _ = _blocks(seed=3)
    rng = np.random.default_rng(4)
    pred = rng.normal(size=(600, 3)) @ sparse.T  # three independent programs
    hidden = np.ones_like(pred, dtype=bool)
    genes = [f"g{k}" for k in range(12)]
    read = prediction_readout(pred, hidden, genes)
    components, gene_rows = rotated_tables(read["basis"], read["singular_values"], genes, k=3, top=4)
    assert np.isclose(components["share"].sum(), 1.0) and np.isclose(components["share_of_rotated"].sum(), 1.0)
    assert list(components["variance"]) == sorted(components["variance"], reverse=True)
    for c, block in enumerate((range(0, 4), range(4, 8), range(8, 12))):
        assert set(gene_rows[gene_rows["component"] == c]["gene"]) == {f"g{k}" for k in block}
        assert (gene_rows[gene_rows["component"] == c]["loading"] > 0).all()  # the sign is fixed
    plain = prediction_tables({"global": pred}, pred, hidden, genes)
    turned = prediction_tables({"global": pred}, pred, hidden, genes, rotation="varimax", rotate_k=3)
    assert "pred_rotated" not in plain and {"pred_rotated", "pred_rotated_genes"} <= set(turned)
    assert set(turned["pred_rotated"]["head"]) == {"global", "truth"}
    assert np.allclose(plain["pred_overlap"]["overlap"], turned["pred_overlap"]["overlap"])
    with pytest.raises(ValueError, match="varimax"):
        prediction_tables({"global": pred}, pred, hidden, genes, rotation="promax")
    r = _planted()
    assert "dim_rotated" not in dimension_tables(r, GENES, center=r.bias)
    dims = dimension_tables(r, GENES, center=r.bias, rotation="varimax", rotate_k=2)
    assert set(dims["dim_rotated_genes"][dims["dim_rotated_genes"]["component"] == 0]["gene"].head(3)) == {"g0", "g1", "g2"}
    with pytest.raises(ValueError, match="varimax"):
        dimension_tables(r, GENES, center=r.bias, rotation="quartimax")


def _varimax_criterion(loadings):
    return float(((loadings**2).var(axis=0)).sum())


def test_varimax_is_at_the_maximum_of_its_own_criterion_on_a_structure_that_is_not_simple():
    rng = np.random.default_rng(7)
    loadings = rng.normal(size=(20, 4)) + np.repeat(np.eye(4) * 2.0, 5, axis=0)
    found, _ = varimax(loadings @ np.linalg.qr(rng.normal(size=(4, 4)))[0])
    best = _varimax_criterion(found)
    for _ in range(500):  # no nearby rotation does better (quartimax, e.g., lands where small steps gain ~0.05)
        step = rng.normal(scale=0.01, size=(4, 4))
        q, _ = np.linalg.qr(np.eye(4) + (step - step.T))
        assert _varimax_criterion(found @ q) <= best + 1e-9


def test_the_rotated_components_are_sorted_and_their_shares_read_the_whole_prediction():
    q, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(30, 6)))
    genes = [f"g{k}" for k in range(30)]
    components, _ = rotated_tables(q, np.ones(6), genes, k=6)  # equal singular values: varimax picks any order
    assert list(components["variance"]) == sorted(components["variance"], reverse=True)
    part, _ = rotated_tables(q, np.array([3.0, 2.0, 1.0, 1.0, 0.5, 0.5]), genes, k=2)
    whole = 9.0 + 4.0 + 1.0 + 1.0 + 0.25 + 0.25
    assert np.isclose(part["share"].sum(), 13.0 / whole)
    assert np.allclose(part["share"], part["variance"] / whole)
    assert np.isclose(part["share_of_rotated"].sum(), 1.0)
