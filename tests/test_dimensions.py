"""The decoder-side readout of the embedding dimensions (``evaluation.dimensions``; roadmap task 2, ED0-ED2).

Pinned here:

* **ED0** -- the collector's rows are exactly what the linear decoder reads (its output is the step's context part), with the
  step's truth and hidden entries, for a GlobalModel and a LocalModel; with folds every cell is hidden once and the graphs' masks
  come back; a nonlinear decoder is refused; no global generator is touched;
* **ED1** -- on a planted readout the dimension that writes the signal tops all three readings, the shares sum to 1, ablating a
  dead dimension costs nothing; the singular directions do not move under a rotation of the dimensions, single ones do;
* **ED2** -- the planted dimension writes and tracks its genes, a noise one tracks nothing; the twin comparison lifts a learned
  dimension above an untrained one.
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
    standardised_loadings,
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
    assert set(tables) == {"dim_importance", "dim_directions", "dim_direction_genes", "dim_genes", "dim_overlap", "dim_tracks"}
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
