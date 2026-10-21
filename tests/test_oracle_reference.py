"""The oracle reference: a ridge on the true drivers of a synthetic dataset (``masked_probing.oracle_features``; roadmap.md, T1).

A programme masked in every cell leaves no visible trace of itself, so the bar for a model is what the cell's other genes
and surroundings say. The oracle adds what the generator knows -- distances to the sources, flags, receivers -- to separate
"the model fails" from "the task cannot be solved from these inputs". Pinned here, on toy graphs with a known driver:

* **the features** -- numbers become bumps (and an indicator where missing), flags stay, other categoricals are one-hot, a
  gate multiplies a key's features, a batch picks its cells' rows by ``obs_names``; nothing reads expression;
* **the reference** -- ``ref_oracle`` exists only with drivers, recovers a smooth, a gated and a partly missing function of
  them where the context references cannot, reads no hidden value, and survives a save and load;
* **the report** -- it is shown beside the references but is never "the best reference" a model is compared with.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation import masked_probing as mp
from interscale.evaluation import report
from interscale.evaluation.masked_probing import FittedReferences, oracle_features

SIDE, SPACING, G, N_GRAPHS = 8, 100.0, 6, 4
CELLS = SIDE * SIDE


def _obs(seed=0):
    """Per cell: its distance to its graph's source, a flag, a gate, and a distance missing on half the graphs."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    pos = np.stack([ii.ravel(), jj.ravel()], axis=1) * SPACING
    rows = []
    for g in range(N_GRAPHS):
        source = rng.uniform(0, SIDE * SPACING, size=2)
        d = np.hypot(*(pos - source).T)
        gate = rng.random(CELLS) < 0.3
        other = (
            np.where(np.arange(CELLS) % 2 == 0, np.nan, rng.uniform(0, 800, CELLS)) if g % 2 else np.full(CELLS, np.nan)
        )
        rows.append(pd.DataFrame({"dist_S": d, "flag": d < 250.0, "gate": gate, "dist_O": other, "graph": g}))
    return pd.concat(rows, ignore_index=True), pos


def _graphs(obs, pos, response, seed=1):
    """One graph per block of ``obs``; gene 0 (the response) is hidden in every cell, the others are noise."""
    rng = np.random.default_rng(seed)
    graphs = []
    for g in range(N_GRAPHS):
        rows = np.arange(g * CELLS, (g + 1) * CELLS)
        x = torch.tensor(rng.normal(size=(CELLS, G)), dtype=torch.float32) + 2.0
        x[:, 0] = torch.tensor(response[rows], dtype=torch.float32)
        data = Data(x=x, pos=torch.tensor(pos, dtype=torch.float32), edge_index=torch.zeros((2, 0), dtype=torch.long))
        data.gene_mask = torch.zeros(CELLS, G, dtype=torch.bool)
        data.gene_mask[:, 0] = True
        data.mask = torch.ones(CELLS, dtype=torch.bool)
        data.obs_names = torch.tensor(rows + 1, dtype=torch.float64)  # a cell's row in obs plus the dataset's offset
        graphs.append(data)
    return graphs


def _fit(graphs, drivers, **kwargs):
    return FittedReferences(
        graphs[:-1],
        spatial_unit_um=1.0,
        kernels_um=(300.0,),
        drivers=drivers,
        alphas=(0.01, 1.0),
        folds=2,
        **kwargs,
    )


def _predict(fitted, graph, name):
    batch = Batch.from_data_list([graph])
    return fitted(batch, batch.gene_mask.bool())[name][:, 0].numpy()


def _r2(pred, true):
    return 1.0 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()


# --------------------------------------------------------------------------- the features


def test_numbers_become_bumps_flags_stay_and_missing_values_get_an_indicator():
    obs, _ = _obs()
    bumps = oracle_features(obs, ["dist_S"], n_basis=10)
    assert bumps.n_features == 10
    assert oracle_features(obs, ["flag"]).n_features == 1  # a flag is itself
    assert oracle_features(obs, ["dist_O"], n_basis=10).n_features == 11  # bumps and the indicator of the missing
    assert oracle_features(obs, ["dist_S", "flag"], n_basis=10).n_features == 11
    assert oracle_features(obs, ["dist_S"], gates={"dist_S": "gate"}, n_basis=10).n_features == 20  # and gated


def test_a_batch_picks_its_cells_rows_and_the_features_read_no_expression():
    obs, pos = _obs()
    drivers = oracle_features(obs, ["dist_S", "flag"], n_basis=8)
    graphs = _graphs(obs, pos, np.zeros(len(obs)))
    batch = Batch.from_data_list([graphs[2]])
    features = drivers(batch, batch.gene_mask.bool())
    assert features.shape == (CELLS, 9) and features.dtype == torch.float64
    whole = oracle_features(obs, ["dist_S", "flag"], n_basis=8)
    expected = whole(Batch.from_data_list([graphs[2]]), None)
    assert torch.equal(features, expected)
    # A graph's rows are not another's: the features of two graphs differ (their sources do).
    other = drivers(Batch.from_data_list([graphs[0]]), None)
    assert not torch.equal(other, features)
    batch.x = batch.x + 100.0  # expression is never read
    assert torch.equal(drivers(batch, None), features)


def test_categoricals_and_string_flags_are_read_as_what_they_are():
    obs, _ = _obs()
    obs["gate_text"] = pd.Categorical(np.where(obs["gate"], "True", "False"))  # as sweep_common's cast leaves a flag
    obs["state"] = pd.Categorical(np.where(obs["dist_S"] < 200, "near", np.where(obs["dist_S"] < 400, "mid", "far")))
    assert oracle_features(obs, ["gate_text"]).n_features == 1
    assert oracle_features(obs, ["state"]).n_features == 3  # one-hot
    gated = oracle_features(obs, ["dist_S"], gates={"dist_S": "gate_text"}, n_basis=6)
    plain = oracle_features(obs, ["dist_S"], gates={"dist_S": "gate"}, n_basis=6)
    graph = Data(x=torch.zeros(4, 2), obs_names=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(gated(graph, None), plain(graph, None))


@pytest.mark.parametrize(
    "call, message",
    [
        pytest.param(lambda obs: oracle_features(obs, ["nope"]), "lacks the driver columns", id="missing key"),
        pytest.param(
            lambda obs: oracle_features(obs, ["dist_S"], gates={"dist_S": "nope"}),
            "lacks the driver columns",
            id="missing gate",
        ),
        pytest.param(
            lambda obs: oracle_features(obs, ["dist_S"], gates={"flag": "gate"}), "not drivers", id="stray gate"
        ),
        pytest.param(
            lambda obs: oracle_features(obs, ["dist_S"], gates={"dist_S": "dist_O"}), "not a boolean", id="numeric gate"
        ),
    ],
)
def test_the_features_refuse_what_they_cannot_read(call, message):
    obs, _ = _obs()
    with pytest.raises(ValueError, match=message):
        call(obs)


def test_cells_outside_obs_are_refused():
    obs, _ = _obs()
    drivers = oracle_features(obs, ["dist_S"])
    graph = Data(x=torch.zeros(2, 2), obs_names=torch.tensor([1.0, float(len(obs) + 5)]))
    with pytest.raises(ValueError, match="outside"):
        drivers(graph, None)


# --------------------------------------------------------------------------- the reference


def _smooth(obs):
    return 3.0 * np.exp(-((obs["dist_S"].to_numpy() / 300.0) ** 2)) + 0.5


def test_the_oracle_exists_only_with_drivers():
    obs, pos = _obs()
    graphs = _graphs(obs, pos, _smooth(obs))
    assert "ref_oracle" not in _fit(graphs, None).models
    assert "ref_oracle" in _fit(graphs, oracle_features(obs, ["dist_S"])).models


def test_it_recovers_a_smooth_function_of_a_driver_where_the_context_cannot():
    obs, pos = _obs()
    graphs = _graphs(obs, pos, _smooth(obs))
    fitted = _fit(graphs, oracle_features(obs, ["dist_S"], n_basis=16))
    truth = graphs[-1].x[:, 0].numpy()
    assert _r2(_predict(fitted, graphs[-1], "ref_oracle"), truth) > 0.95
    assert _r2(_predict(fitted, graphs[-1], "ref_own+context"), truth) < 0.3  # nothing visible shows the response


def test_it_recovers_a_response_that_exists_only_where_a_gate_is_on_and_only_with_the_gate():
    obs, pos = _obs()
    response = _smooth(obs) * obs["gate"].to_numpy()
    graphs = _graphs(obs, pos, response)
    truth = graphs[-1].x[:, 0].numpy()
    gated = _fit(graphs, oracle_features(obs, ["dist_S", "gate"], gates={"dist_S": "gate"}, n_basis=16))
    plain = _fit(graphs, oracle_features(obs, ["dist_S", "gate"], n_basis=16))
    assert _r2(_predict(gated, graphs[-1], "ref_oracle"), truth) > 0.95
    assert (
        _r2(_predict(plain, graphs[-1], "ref_oracle"), truth)
        < _r2(_predict(gated, graphs[-1], "ref_oracle"), truth) - 0.05
    )


def test_it_recovers_a_driver_that_exists_on_some_graphs_only():
    """`dist_O` is missing on every cell of even graphs and half of the odd ones; the response is 5 wherever it is missing."""
    obs, pos = _obs()
    d = obs["dist_O"].to_numpy()
    response = np.where(np.isnan(d), 5.0, 1.0 + 2.0 * np.exp(-((np.nan_to_num(d) / 300.0) ** 2)))
    graphs = _graphs(obs, pos, response)
    fitted = _fit(graphs, oracle_features(obs, ["dist_O"], n_basis=16))
    assert _r2(_predict(fitted, graphs[-1], "ref_oracle"), graphs[-1].x[:, 0].numpy()) > 0.9


def test_it_reads_no_hidden_value():
    obs, pos = _obs()
    graphs = _graphs(obs, pos, _smooth(obs))
    fitted = _fit(graphs, oracle_features(obs, ["dist_S"]))
    before = _predict(fitted, graphs[-1], "ref_oracle")
    changed = graphs[-1].clone()
    changed.x = graphs[-1].x.clone()
    changed.x[:, 0] += 50.0  # the hidden response
    assert np.array_equal(_predict(fitted, changed, "ref_oracle"), before)
    changed.x[:, 3] += 50.0  # a visible gene: the own cell's part of the reference moves
    assert not np.array_equal(_predict(fitted, changed, "ref_oracle"), before)


def test_a_saved_fit_needs_its_drivers_back_and_only_then():
    obs, pos = _obs()
    graphs = _graphs(obs, pos, _smooth(obs))
    drivers = oracle_features(obs, ["dist_S"], n_basis=12)
    fitted = _fit(graphs, drivers)
    state = fitted.state()
    reloaded = FittedReferences.from_state(state, drivers=drivers)
    assert np.allclose(_predict(reloaded, graphs[-1], "ref_oracle"), _predict(fitted, graphs[-1], "ref_oracle"))
    with pytest.raises(ValueError, match="true drivers"):
        FittedReferences.from_state(state)
    with pytest.raises(ValueError, match="no oracle"):
        FittedReferences.from_state(_fit(graphs, None).state(), drivers=drivers)
    assert FittedReferences.from_state(_fit(graphs, None).state()).drivers is None


# --------------------------------------------------------------------------- the report


def test_the_oracle_is_shown_beside_the_references_but_is_never_the_best_one():
    assert report.ORACLE_REFERENCE == mp.ORACLE_REFERENCE
    table = pd.DataFrame(
        {"n_hidden": [100, 100], "global": [0.5, 0.7], "ref_own": [0.4, 0.6], "ref_oracle": [0.99, 0.99]},
        index=pd.Index(["interaction=M1>MC1", "interaction=M2>MC2"], name="group"),
    )
    per_arm = {"arm": table}
    versus_best = report._versus_bar(per_arm, report.BEST_REFERENCE, "interaction=", report.TIE_MARGIN)
    assert versus_best.loc[("arm", "global"), "bar mean"] == pytest.approx(0.5)  # ref_own, not the oracle
    assert versus_best.loc[("arm", "global"), "wins"] == 2
    versus_oracle = report._versus_bar(per_arm, report.ORACLE_REFERENCE, "interaction=", report.TIE_MARGIN)
    assert versus_oracle.loc[("arm", "global"), "losses"] == 2

    by_depth = pd.DataFrame(
        {"all": [100, 0.5, 0.4, 0.99]},
        index=pd.MultiIndex.from_tuples(
            [("g", "n_hidden"), ("g", "global"), ("g", "ref_own"), ("g", "ref_oracle")], names=["group", "head"]
        ),
    )
    summary = report._group_depth_summary({"arm": by_depth}, None)
    best = summary.loc[("g", "reference", "best"), "all"]
    assert best == pytest.approx(0.4) and summary.loc[("g", "reference", "oracle"), "all"] == pytest.approx(0.99)
