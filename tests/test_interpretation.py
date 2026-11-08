"""From a model to biology (``evaluation.interpretation``, ``interpretation_report``).

Pinned on a module of known dependence: type B's target 0 reads type A's sender 3 within 150 µm; its target 2 reads sender 4 of the cell 400 µm away, whatever its type; every
target 1 the cell's own sender 5. So a state flow A -> B exists and C -> B does not, the far region keeps the distant read and loses the near one, and the own read never moves a
thing. The graphs and the global generators are left alone. Then enrichment against its hypergeometric definition and with the panel as universe, known pairs against the
expression-matched null, the attention tables on planted biology, and programmes on an embedding with a known cell-type part and a known spatial part.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.context_test import _hidden_entries
from interscale.evaluation.dimensions import DimensionReadout
from interscale.evaluation.interpretation import (
    _changed,
    attention_biology,
    attention_pairs,
    centred_cosine,
    centred_sums,
    context_use,
    cosine_from_sums,
    edge_recovery,
    enrichment,
    flow_matrix,
    gene_flow_long,
    gene_flow_tables,
    gene_flows,
    gene_influence,
    gene_spatial,
    lr_cluster_p,
    lr_support,
    lr_support_by_type,
    match_programmes,
    matched_cells,
    moran,
    moran_genes,
    pick_receivers,
    programme_cards,
    programme_class,
    programmes,
    type_flows,
)
from interscale.evaluation.interpretation_report import (
    condition_by_section,
    influence_read,
    lr_row,
    md_table,
    planted_against_twin,
    twin_of,
    write_run_report,
    write_summary,
    write_tables,
)
from interscale.module.base._step_output import StepOutput, ViewOutput

SIDE, SPACING, G = 18, 50.0, 8
GENES = [f"g{k}" for k in range(G)]
HIDDEN = [0, 1, 2]


def _pos():
    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float) * SPACING


def _labels(seed=0):
    return np.random.default_rng(seed).choice(["A", "B", "C"], size=SIDE * SIDE, p=[0.4, 0.4, 0.2])


def _near(pos, labels):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return (d <= 150.0) & (d > 0) & (labels == "A")[None, :]


def _partner(pos):
    period = SIDE * SPACING
    target = np.column_stack([np.mod(pos[:, 0] + 400.0, period), pos[:, 1]])
    return np.argmin(np.linalg.norm(target[:, None] - pos[None], axis=-1), axis=1)


class _Known(torch.nn.Module):
    registered_global_component = True

    def __init__(self, labels):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.labels = labels

    def _common_step(self, batch, task, level):
        x, pos = batch.x.double(), batch.pos.double().numpy()
        pred = torch.zeros_like(x)
        near = torch.as_tensor(_near(pos, self.labels), dtype=torch.float64)
        is_b = torch.as_tensor(self.labels == "B")
        reads = (near @ x[:, 3]) / near.sum(1).clamp_min(1.0)
        pred[is_b, 0] = reads[is_b]
        pred[is_b, 2] = x[torch.as_tensor(_partner(pos)), 4][is_b]
        pred[:, 1] = x[:, 5]
        return StepOutput(y_pred=pred.float(), y_true=batch.x, entry_mask=batch.gene_mask.bool(), views=[ViewOutput()])


def _graph(labels, seed=0):
    rng = np.random.default_rng(seed)
    pos = _pos()
    x = rng.random((len(pos), G))
    near = _near(pos, labels)
    x[:, 0] = (near @ x[:, 3]) / np.maximum(near.sum(1), 1)
    x[:, 2] = x[_partner(pos), 4]
    x[:, 1] = x[:, 5]
    d = Data(x=torch.tensor(x, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32))
    d.edge_index = torch.zeros((2, 0), dtype=torch.long)
    mask = torch.zeros(len(pos), G, dtype=torch.bool)
    mask[:, HIDDEN] = True
    d.gene_mask = mask
    return d


# ----------------------------------------------------------------------------------------------- biology resources


def test_enrichment_is_the_hypergeometric_on_the_panel_and_corrects_within_a_library():
    panel = [f"g{k}" for k in range(100)]
    sets = {
        "lib": {
            "hit": [f"g{k}" for k in range(10)] + ["not_on_panel"],
            "miss": [f"g{k}" for k in range(50, 60)],
            "off": ["x", "y"],
        }
    }
    found = enrichment([f"g{k}" for k in range(8)] + ["g50", "g51"], panel, sets)
    top = found.iloc[0]
    assert top["set"] == "hit" and top["overlap"] == 8 and top["set_size"] == 10  # cut to the panel
    assert top["p"] == pytest.approx(hypergeom.sf(7, 100, 10, 10))
    assert top["q"] == pytest.approx(min(top["p"] * 2, 1.0))  # two sets tested: "off" has no gene on the panel
    assert (found["q"] >= found["p"]).all() and "off" not in set(found["set"])
    # A gene outside the panel is no hit.
    assert enrichment(["x", "y", "g0", "g1"], panel, sets).iloc[0]["hits"] == 2


def test_known_pairs_stand_out_only_when_they_rank_above_matched_targets():
    rng = np.random.default_rng(0)
    targets, ligands = [f"t{k}" for k in range(60)], [f"l{k}" for k in range(10)]
    score = pd.DataFrame(rng.random((10, 60)), index=ligands, columns=targets)
    pairs = [(f"l{k}", f"t{k}") for k in range(10)]
    random = lr_support(score, pairs, n_perm=300)
    assert random["n_pairs"] == 10 and 0.2 < random["auc"] < 0.8 and random["p"] > 0.01
    planted = score.copy()
    for k in range(10):
        planted.loc[f"l{k}", f"t{k}"] = 5.0
    found = lr_support(planted, pairs, n_perm=300)
    assert found["auc"] == pytest.approx(1.0) and found["p"] < 0.01 and found["pairs"]["score"].iloc[0] == 5.0
    assert found["null_mean"] == pytest.approx(0.5, abs=0.05)
    # Expression explains it: the receptors are simply the most expressed, and every target's score follows expression. The matched null takes that away.
    expression = pd.Series(np.arange(60.0), index=targets)
    follows = pd.DataFrame(
        np.tile(expression.to_numpy(), (10, 1)) + rng.random((10, 60)), index=ligands, columns=targets
    )
    top = [(f"l{k}", f"t{59 - k}") for k in range(10)]
    unmatched, matched = (
        lr_support(follows, top, n_perm=300),
        lr_support(follows, top, expression=expression, n_perm=300),
    )
    assert unmatched["p"] < 0.01 and matched["null_mean"] > 0.8 and matched["p"] > unmatched["p"]
    assert lr_support(score, [("nobody", "t0")])["n_pairs"] == 0


# ----------------------------------------------------------------------------------------------- choosing cells


def test_rare_types_keep_their_receivers_and_the_zones_stay_untouched():
    pos, labels = _pos(), _labels()
    rng = np.random.default_rng(0)
    receivers, keep = pick_receivers(pos, labels, np.ones(len(pos), bool), 0.0, rng=rng, per_type=20)
    assert keep.sum() == len(receivers) and all((labels[receivers] == t).sum() == 20 for t in "ABC")
    receivers, keep = pick_receivers(pos, labels, np.ones(len(pos), bool), 120.0, rng=rng, per_type=5, max_cover=0.3)
    # Each type always gets its first receiver, so the limit can be passed by those; the rarest chooses first.
    assert keep.mean() <= 0.5 and (labels[receivers] == "C").sum() == 5
    d = np.linalg.norm(pos[:, None] - pos[receivers][None], axis=-1).min(axis=1)
    assert keep[d <= 120.0].all() and not keep[d > 120.0].any()
    candidates = labels != "C"
    receivers, _ = pick_receivers(pos, labels, candidates, 0.0, rng=rng, per_type=10)
    assert candidates[receivers].all()


def test_matched_cells_are_the_nearest_unused_others():
    pos = np.array([[0.0, 0], [10, 0], [1, 0], [11, 0], [50, 0], [51, 0]])
    got = matched_cells(pos, [0, 1], [2, 3, 4, 5], rng=np.random.default_rng(0))
    assert sorted(got.tolist()) == [2, 3]
    assert matched_cells(pos, [0, 1, 4], [2], rng=np.random.default_rng(0)).tolist() == [2]  # it runs out


# ----------------------------------------------------------------------------------------------- type flows


def _flows(**kw):
    labels = _labels()
    module, graph = _Known(labels), _graph(labels)
    return (
        labels,
        module,
        graph,
        type_flows(
            module, graph, labels, GENES, **({"n_draws": 1, "per_type": 40, "far_per_type": 12, "far_radius": 180.0, "seed": 1} | kw)
        ),
    )


def test_flows_find_who_affects_whom_and_leave_the_graph_and_generators_alone():
    labels = _labels()
    module, graph = _Known(labels), _graph(labels)
    before = graph.x.clone()
    torch_state, np_state = torch.random.get_rng_state(), np.random.get_state()[1].copy()
    flows, targets = type_flows(
        module, graph, labels, GENES, n_draws=1, per_type=40, far_per_type=12, far_radius=180.0, seed=1
    )
    assert torch.equal(torch_state, torch.random.get_rng_state()) and np.array_equal(np_state, np.random.get_state()[1])
    assert torch.equal(graph.x, before)
    state = flow_matrix(flows, kind="state", region="context")
    assert state.loc["A", "B"] > 0.05
    assert state.loc["C", "B"] < state.loc["A", "B"]  # C reaches B only through the distant read, A through both
    # Only B's targets depend on anything but the own row, so nobody affects A or C.
    assert abs(state.loc["A", "A"]) < 1e-9 and abs(state.loc["A", "C"]) < 1e-9
    top = targets[
        (targets["kind"] == "state")
        & (targets["sender"] == "A")
        & (targets["receiver"] == "B")
        & (targets["region"] == "context")
    ]
    assert top.iloc[0]["gene"] in ("g0", "g2")
    # The own read: receivers keep their rows, so target 1 never moves.
    assert not len(targets[(targets["gene"] == "g1") & (targets["delta_cos"].abs() > 1e-9)])


def test_the_far_region_keeps_the_distant_read_and_loses_the_near_one():
    # The far zones fit one B receiver per draw on this grid; a target's score within type needs at least 3 receivers, pooled over draws.
    labels, _, _, (flows, targets) = _flows(n_draws=6)
    use = context_use(flows).set_index(["region", "receiver"])
    assert use.loc[("context", "B"), "state"] > 0.05
    assert use.loc[("far", "B"), "state"] > 0.02  # target 2 reads 400 µm away, beyond the zone
    assert abs(use.loc[("context", "A"), "state"]) < 1e-9
    far = targets[
        (targets["region"] == "far") & (targets["kind"] == "all_state") & (targets["receiver"] == "B")
    ].set_index("gene")["delta_cos"]
    assert far["g2"] > 0.1
    # Target 0 reads within 150 µm, all inside the kept 180 µm zone: exactly nothing.
    assert "g0" not in far.index or abs(far["g0"]) < 1e-9


def test_hidden_entries_are_never_handed_around():
    labels = _labels()
    graph = _graph(labels)
    copy = _changed(graph, torch.as_tensor(~_hidden_entries(graph, "gene")), [0, 1], [5, 6])
    assert torch.equal(copy.x[0, HIDDEN], graph.x[0, HIDDEN]) and torch.equal(copy.x[1, HIDDEN], graph.x[1, HIDDEN])
    visible = [k for k in range(G) if k not in HIDDEN]
    assert torch.equal(copy.x[0, visible], graph.x[5, visible]) and torch.equal(copy.x[1, visible], graph.x[6, visible])
    assert torch.equal(copy.x[2:], graph.x[2:])


def test_identity_takes_its_null_off_and_sections_add_as_sums():
    labels, _, _, (flows, _) = _flows()
    raw = flow_matrix(flows, kind="identity", region="context", minus_null=False)
    net = flow_matrix(flows, kind="identity", region="context")
    null = flow_matrix(flows, kind="identity_null", region="context", minus_null=False)
    assert net.loc["A", "B"] == pytest.approx(raw.loc["A", "B"] - null.loc["A", "B"])
    # A sum over sections is a ratio of sums, not a mean of ratios.
    part = flows[(flows["kind"] == "state") & (flows["region"] == "context")].copy()
    twice = pd.concat([part, part.assign(sum_delta=part["sum_delta"] * 3, n_scored=part["n_scored"] * 9)])
    expected = (
        4
        * part.set_index(["sender", "receiver"]).loc[("A", "B"), "sum_delta"]
        / (10 * part.set_index(["sender", "receiver"]).loc[("A", "B"), "n_scored"])
    )
    assert flow_matrix(twice, kind="state").loc["A", "B"] == pytest.approx(expected)
    # Context use: the summed drop over the cells scored, per region and receiver type.
    use = context_use(flows).set_index(["region", "receiver"])
    one = flows[(flows["kind"] == "all_state") & (flows["region"] == "context") & (flows["receiver"] == "B")]
    assert use.loc[("context", "B"), "state"] == pytest.approx(one["sum_delta"].sum() / one["n_scored"].sum())
    assert (flows["n_scored"] <= flows["receivers"]).all()


def test_the_centred_cosine_is_a_cosine_after_centring_and_its_sums_pool_exactly():
    rng = np.random.default_rng(4)
    pred, truth = rng.normal(size=(30, 6)), rng.normal(size=(30, 6))
    valid = rng.random((30, 6)) < 0.7
    a = np.where(valid, pred - np.where(valid, pred, 0).sum(0) / valid.sum(0), 0)
    b = np.where(valid, truth - np.where(valid, truth, 0).sum(0) / valid.sum(0), 0)
    expected = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    assert np.allclose(centred_cosine(pred, truth, valid, axis=1), np.where(valid.sum(1) >= 2, expected, np.nan), equal_nan=True)
    one = valid.copy()
    one[0] = False
    one[0, 2] = True  # a cell with one hidden gene: its cosine would be a sign, not a score
    assert np.isnan(centred_cosine(pred, truth, one, axis=1)[0])
    # A level per gene and one scale of the prediction cancel (centred on its own mean).
    assert np.allclose(
        centred_cosine(5.0 * pred + rng.normal(size=6), truth, valid, axis=0), centred_cosine(pred, truth, valid, axis=0)
    )
    # Fixed centres: sums over parts give the cosine over all rows.
    cp, ct = rng.normal(size=6), rng.normal(size=6)
    whole = centred_cosine(pred, truth, valid, axis=0, centre_pred=cp, centre_truth=ct)
    pooled = sum(centred_sums(pred[k : k + 10], truth[k : k + 10], valid[k : k + 10], cp, ct) for k in (0, 10, 20))
    assert np.allclose(cosine_from_sums(pooled, valid.sum(0)), whole)


class _Miscalibrated(_Known):
    """The same reads, every prediction 5 times too large and shifted per gene: what a model with R² far below 0 looks like."""

    def _common_step(self, batch, task, level):
        out = super()._common_step(batch, task, level)
        shift = torch.arange(out.y_pred.shape[1], dtype=out.y_pred.dtype)
        return StepOutput(y_pred=5.0 * out.y_pred + shift, y_true=out.y_true, entry_mask=out.entry_mask, views=out.views)


def test_flows_do_not_count_a_level_or_scale_error_of_the_model():
    labels = _labels()
    graph = _graph(labels)
    kw = {"n_draws": 1, "per_type": 40, "far_per_type": 12, "far_radius": 180.0, "seed": 1}
    right, right_targets = type_flows(_Known(labels), graph, labels, GENES, **kw)
    wrong, wrong_targets = type_flows(_Miscalibrated(labels), graph, labels, GENES, **kw)
    assert flow_matrix(right, kind="state").loc["A", "B"] > 0.05
    for kind in ("state", "identity", "identity_null"):
        pd.testing.assert_frame_equal(flow_matrix(right, kind=kind), flow_matrix(wrong, kind=kind), atol=1e-9)
    assert np.allclose(right_targets["delta_cos"], wrong_targets["delta_cos"], atol=1e-9)
    genes_right = gene_flows(_Known(labels), [graph], [labels], GENES, ["g3", "g4"], per_type=40, seed=2).delta_cos()
    genes_wrong = gene_flows(_Miscalibrated(labels), [graph], [labels], GENES, ["g3", "g4"], per_type=40, seed=2).delta_cos()
    assert np.allclose(genes_right, genes_wrong, atol=1e-9, equal_nan=True)


def _type_specific_pairs(rng, *, planted: float):
    """Random scores; each receptor is detected in one receiver type, every other target in all four: the setting where a maximum over types is biased."""
    types, ligands = ["T0", "T1", "T2", "T3"], [f"L{k}" for k in range(24)]
    receptors, others = [f"R{k}" for k in range(24)], [f"O{k}" for k in range(40)]
    rows = []
    for k, ligand in enumerate(ligands):
        for t, receiver in enumerate(types):
            for target in others + [receptors[k]] * (t == k % 4):
                score = rng.normal() + (planted if target == receptors[k] else 0.0)
                rows.append({"sender": ligand, "receiver": receiver, "target": target, "delta_cos": score, "expressed": 0.5})
    return pd.DataFrame(rows), list(zip(ligands, receptors, strict=True))


def test_known_pairs_within_types_are_unbiased_where_a_maximum_over_types_is_not():
    # Random scores, so known pairs must sit at the null. Over 200 such seeds the within-type test gave p < 0.05 for 6.5% and AUC 0.503 ± 0.060 (2026-11-07).
    old, new = [], []
    for seed in range(5):
        long, pairs = _type_specific_pairs(np.random.default_rng(seed), planted=0.0)
        # The old test: one score per target, its largest over the types, against expression-matched targets. A receptor has one type to be maximised over.
        pooled = long.pivot_table(index="sender", columns="target", values="delta_cos", aggfunc="max")
        old.append(lr_support(pooled, pairs, expression=pd.Series(0.5, index=pooled.columns), n_perm=200))
        new.append(lr_support_by_type(long, pairs, n_perm=200, seed=seed))
    assert np.mean([r["auc"] for r in old]) < 0.35 and np.mean([r["null_mean"] for r in old]) == pytest.approx(0.5, abs=0.02)
    assert len(old[0]["null_draws"]) == 200
    assert abs(np.mean([r["auc"] for r in new]) - 0.5) < 0.06
    assert all(r["n_pairs"] == len(pairs) and r["n_types"] == 4 for r in new)
    assert set(new[0]["pairs"].columns) == {"ligand", "receptor", "receiver", "score", "auc"}
    found = lr_support_by_type(_type_specific_pairs(np.random.default_rng(6), planted=3.0)[0], pairs, n_perm=400)
    assert found["auc"] > 0.8 and found["p"] < 0.01
    assert lr_support_by_type(long.iloc[:0], pairs)["n_pairs"] == 0


def _shared_receptors(rng, *, gene_effect: float, planted: float = 0.0, distinct: bool = False):
    """Every ligand paired with the same few receptors in every type (as ITGB1 on CosMx), each target with an effect of its own whatever the sender: what a
    random model's high-variance targets do. ``distinct``: each pair its own receptor instead."""
    types, ligands = ["T0", "T1", "T2", "T3"], [f"L{k}" for k in range(16)]
    receptors = [f"R{k}" for k in range(len(ligands) if distinct else 4)]
    targets = receptors + [f"O{k}" for k in range(40)]
    effect = {t: gene_effect * rng.normal() for t in targets}
    pairs = [(ligand, receptors[k]) for k, ligand in enumerate(ligands)] if distinct else [(ligand, r) for ligand in ligands for r in receptors]
    planted_pairs = set(pairs)
    rows = [
        {"sender": ligand, "receiver": receiver, "target": target, "expressed": 0.5,
         "delta_cos": effect[target] + 0.3 * rng.normal() + (planted if (ligand, target) in planted_pairs else 0.0)}
        for ligand in ligands for receiver in types for target in targets
    ]  # fmt: skip
    return pd.DataFrame(rows), pairs


def test_known_pairs_that_share_genes_are_not_read_as_independent():
    # Random gene effects shared by all of a receptor's pairs: no pair is special, so p must stay calibrated. The per-pair null rejects far too often
    # (the CosMx twin's 0.470 against 0.501 ± 0.008, notes.md 2026-11-08).
    independent, clustered = [], []
    for seed in range(12):
        long, pairs = _shared_receptors(np.random.default_rng(seed), gene_effect=1.0)
        found = lr_support_by_type(long, pairs, n_perm=100, seed=seed)
        independent.append(found["p_independent"])
        clustered.append(found["p"])
        assert found["cluster_sd"] > 2 * found["null_sd"]
    independent, clustered = np.asarray(independent), np.asarray(clustered)
    assert np.mean((independent < 0.05) | (independent > 0.95)) > 0.4  # either tail: shared genes push the mean far from the null
    assert np.mean(clustered < 0.05) <= 2 / 12
    # Each pair its own receptor: the bootstrap's spread is the null's, and a planted effect is still found.
    long, pairs = _shared_receptors(np.random.default_rng(20), gene_effect=0.0, distinct=True)
    plain = lr_support_by_type(long, pairs, n_perm=200)
    assert 0.5 < plain["cluster_sd"] / plain["null_sd"] < 2.0
    long, pairs = _shared_receptors(np.random.default_rng(21), gene_effect=0.0, planted=1.0, distinct=True)
    assert lr_support_by_type(long, pairs, n_perm=200)["p"] < 0.01
    assert np.isnan(lr_cluster_p(pd.DataFrame({"ligand": ["a"], "receptor": ["b"], "auc": [0.5]}), 0.5, 0.01)["p"])


def test_the_far_region_keeps_a_distant_sender_and_drops_a_near_one():
    # g0 reads g3 within 150 of the receiver (B cells); g2 reads g4 at 400 away. Beyond 160 only the partner remains. One receiver per type and graph: the
    # zones of the rarest type, chosen first, would otherwise fill the cover and leave B none.
    labels = _labels()
    graphs = [_graph(labels, seed=k) for k in (0, 3, 6, 9)]  # seeds where g0 varies enough within B to be read (the floor is per graph)
    near = gene_flows(_Known(labels), graphs, [labels] * 4, GENES, ["g3", "g4"], within_labels=False, per_type=40, seed=3)
    far = gene_flows(_Known(labels), graphs, [labels] * 4, GENES, ["g3", "g4"], within_labels=False, far_radius=160.0, per_type=1, seed=3)
    b = near.receivers.index("B")
    assert near.delta_cos()[0, b, 0] > 0.3 and near.delta_cos()[1, b, 2] > 0.3
    assert far.delta_cos()[0, b, 0] == pytest.approx(0.0, abs=1e-9)  # every cell g0 reads lies in the kept zone: nothing it reads changed
    assert far.delta_cos()[1, b, 2] > 0.05  # four receivers: a noisy cosine, but the partner's loss stays
    table = gene_influence(near)
    assert list(table.index) == ["g3", "g4"] and (table["reads"] > 0).all() and (table["top"] >= table["influence"]).all()


def test_moran_of_many_genes_is_moran_of_each_and_types_come_out():
    rng = np.random.default_rng(4)
    pos = _pos()
    labels = _labels(4)
    sections = np.where(pos[:, 0] < pos[:, 0].mean(), "s0", "s1")
    smooth = np.sin(pos[:, 0] / 200.0) + np.cos(pos[:, 1] / 230.0)
    marker = (labels == "A").astype(float) * 3.0  # only the type: patterned raw, flat within type
    noise = rng.normal(size=len(pos))
    x = np.column_stack([smooth + 0.1 * rng.normal(size=len(pos)), marker + 0.01 * rng.normal(size=len(pos)), noise])
    many = moran_genes(x, pos, sections, k=8, seed=1)
    for j in range(3):
        assert many[j] == pytest.approx(moran(x[:, j], pos, sections, k=8, seed=1), abs=1e-12)
    table = gene_spatial(x, ["smooth", "marker", "noise"], labels, sections, pos, n_centres=200)
    assert table.loc["smooth", "moran_local"] > 0.5 and table.loc["smooth", "moran_global"] > table.loc["noise", "moran_global"]
    assert abs(table.loc["marker", "moran_local"]) < 0.1 and abs(table.loc["noise", "moran_local"]) < 0.1


def _influence_toy(seed=0, *, learned=True):
    """200 genes: 40 spatially variable (20 local, 20 global), expression unrelated to space. Learned: SVGs influential, global ones far, local ones near."""
    rng = np.random.default_rng(seed)
    genes = [f"g{k}" for k in range(200)]
    local = np.r_[rng.uniform(0.5, 0.9, 20), rng.uniform(0.0, 0.1, 20), rng.uniform(0.0, 0.2, 160)]
    glob = np.r_[rng.uniform(0.0, 0.02, 20), rng.uniform(0.05, 0.1, 20), rng.uniform(0.0, 0.02, 160)]
    near = rng.normal(0, 0.01, 200) + (np.r_[np.full(20, 0.08), np.full(20, 0.03), np.zeros(160)] if learned else 0.0)
    far = rng.normal(0, 0.01, 200) + (np.r_[np.full(20, 0.0), np.full(20, 0.08), np.zeros(160)] if learned else 0.0)
    return pd.DataFrame(
        {"influence_near": near, "influence_far": far, "mean": rng.uniform(0, 3, 200), "detected": 0.5, "moran_local": local, "moran_global": glob,
         "moran_local_raw": local + 0.1},
        index=pd.Index(genes, name="gene"),
    )  # fmt: skip


def test_the_influence_read_finds_spatially_variable_genes_among_the_top_and_separates_scales():
    found = influence_read(_influence_toy(0), _influence_toy(1, learned=False))
    top = found["top"].set_index("influence")
    assert top.loc["learned_near", "observed"] > 0.9 and top.loc["learned_near", "p"] < 0.01
    assert top.loc["learned_near", "expected"] == pytest.approx(top.loc["learned_near", "base_rate"], abs=0.15)  # expression says nothing here
    sep = found["separation"]
    assert sep["lean_far_global"] > sep["lean_far_local"] and sep["p"] < 0.01
    corr = found["correlations"].set_index(["influence", "spatial"])
    # Nothing learned: the top genes are spatially variable at about the base rate, and influence does not follow Moran's I.
    empty = influence_read(_influence_toy(2, learned=False), _influence_toy(3, learned=False))
    nothing = empty["top"].set_index("influence")
    assert nothing.loc["learned_near", "p"] > 0.01
    rho = corr.loc[("learned_near", "moran_local"), "within_expression"]
    assert rho > 0.1 and rho > empty["correlations"].set_index(["influence", "spatial"]).loc[("learned_near", "moran_local"), "within_expression"] + 0.1


def test_the_report_and_summary_read_influence_against_the_twin(tmp_path):
    for k, name in enumerate(("rand50_s42", "rand50_s43")):
        write_tables(tmp_path / "runs" / name, {"gene_influence": _influence_toy(k).reset_index()})
    write_tables(tmp_path / "runs" / "rand50_s42_untrained", {"gene_influence": _influence_toy(9, learned=False).reset_index()})
    report = write_run_report(tmp_path / "runs" / "rand50_s43").read_text()
    assert "## 7. Which genes matter" in report and "minus the untrained twin's (rand50_s42_untrained)" in report and "influence_vs_moran.png" in report
    text = write_summary(tmp_path).read_text()
    assert "Which genes matter, and are they the spatially variable ones" in text and "rank-biased overlap" in text


def test_a_target_constant_within_its_type_gets_no_score():
    labels = _labels()
    graph = _graph(labels)
    x = graph.x.clone()
    x[:, 0] = 0.5 + 1e-4 * torch.rand(x.shape[0], generator=torch.Generator().manual_seed(0))  # no pattern within any type to lose
    graph.x = x
    flows = gene_flows(_Known(labels), [graph], [labels], GENES, ["g3", "g4"], per_type=40, seed=2)
    assert np.isnan(flows.delta_cos()[:, :, 0]).all() and np.isfinite(flows.delta_cos()[:, :, 2]).any()
    _, targets = type_flows(_Known(labels), graph, labels, GENES, n_draws=1, per_type=40, far_per_type=12, far_radius=180.0, seed=1)
    assert "g0" not in set(targets["gene"])


def test_known_pairs_within_types_are_matched_on_detection_in_that_type():
    # Receptors are well detected; half the other targets too. The score follows detection, not the pair: an unmatched null would call that support.
    rng = np.random.default_rng(7)
    rows, pairs = [], []
    for k in range(24):
        ligand, receptor, receiver = f"L{k}", f"R{k}", f"T{k % 4}"
        pairs.append((ligand, receptor))
        for target in [f"O{j}" for j in range(40)] + [receptor]:
            detected = 0.9 if target == receptor or int(target[1:]) % 2 == 0 else 0.1
            rows.append({"sender": ligand, "receiver": receiver, "target": target, "delta_cos": detected + 0.3 * rng.normal(), "expressed": detected})
    found = lr_support_by_type(pd.DataFrame(rows), pairs, n_perm=400)
    assert found["null_mean"] > 0.6  # the matched null knows well-detected targets rank high
    assert abs(found["auc"] - found["null_mean"]) < 3 * found["null_sd"] and found["p"] > 0.01


def test_the_long_gene_flow_table_keeps_what_the_tables_score():
    labels = _labels()
    flows = gene_flows(_Known(labels), [_graph(labels)], [labels], GENES, ["g3", "g4"], per_type=40, seed=2)
    long = gene_flow_long(flows)
    score, _ = gene_flow_tables(flows)
    assert (long["expressed"] >= 0.1).all() and np.isfinite(long["delta_cos"]).all()
    best = long.groupby(["sender", "target"])["delta_cos"].max().unstack()
    pd.testing.assert_frame_equal(best.reindex_like(score), score, check_names=False)


# ----------------------------------------------------------------------------------------------- gene flows


def test_gene_flows_name_the_sender_and_the_target():
    labels = _labels()
    module, graph = _Known(labels), _graph(labels)
    before = graph.x.clone()
    flows = gene_flows(
        module, [graph], [labels], GENES, ["g3", "g4", "g5", "g6", "g0"], within_labels=True, per_type=40, seed=2
    )
    assert torch.equal(graph.x, before)
    assert flows.senders == ["g3", "g4", "g5", "g6"]  # g0 is hidden in every cell: no sender
    dr2 = flows.delta_cos()
    b = flows.receivers.index("B")
    assert dr2[flows.senders.index("g3"), b, 0] > 0.05 and dr2[flows.senders.index("g4"), b, 2] > 0.1
    assert abs(dr2[flows.senders.index("g5"), b, 1]) < 1e-9  # the own row is kept
    assert np.nanmax(abs(dr2[flows.senders.index("g6")])) < 1e-9  # a gene nothing reads
    score, top = gene_flow_tables(flows, top=5)
    assert set(score.columns) <= set(GENES) and score.loc["g4", "g2"] > 0.1
    assert top.iloc[0]["sender"] in ("g3", "g4") and top.iloc[0]["receiver"] == "B"
    recovered = edge_recovery(score, [("g3", "g0"), ("g4", "g2")])
    assert (recovered["best_rank"] == 1).all() and recovered["chance_top1"].iloc[0] == pytest.approx(1 / 4)


def test_a_target_nobody_expresses_is_not_scored_and_within_type_keeps_a_markers_type():
    labels = _labels()
    module = _Known(labels)
    graph = _graph(labels)
    x = graph.x.clone()
    x[:, 3] = torch.as_tensor(labels == "A", dtype=x.dtype) + 0.2 * x[:, 3]  # a marker of A: it hardly varies within A
    x[:, 0] = 0.0
    for t in "ABC":
        x[np.flatnonzero(labels == t)[:2], 0] = 1.0  # target 0 varies, but is detected in two cells of each type
    graph.x = x
    within = gene_flows(module, [graph], [labels], GENES, ["g3"], within_labels=True, per_type=40, seed=3)
    across = gene_flows(module, [graph], [labels], GENES, ["g3"], within_labels=False, per_type=40, seed=3)
    assert 0 < within.expressed[:, 0].max() < 0.1 and np.isfinite(within.delta_cos()[0, :, 0]).any()
    score, _ = gene_flow_tables(within)
    assert "g0" not in score.columns
    b = within.receivers.index("B")
    # The model reads A's cells only. Permuting within A keeps every A cell an A cell's value; permuting across brings others' low values in: a far larger change in what it reads.
    assert abs(across.sum_shift[0, b, 0]) > 5 * abs(within.sum_shift[0, b, 0])


# ----------------------------------------------------------------------------------------------- attention


def test_attention_biology_finds_what_sits_in_sources_within_type():
    rng = np.random.default_rng(0)
    n = 3000
    pos = rng.uniform(0, 1000, size=(n, 2))
    sections = np.where(np.arange(n) < n // 2, "s1", "s2")
    # Types in patches, so the smoothed flow carries the type: a marker correlates with it across a section and not within the type.
    labels = np.where(np.sin(pos[:, 0] / 150) * np.sin(pos[:, 1] / 150) > 0, "A", "B")
    field = np.sin(pos[:, 0] / 40) + np.cos(pos[:, 1] / 40)
    out = np.clip(field, 0, None) + 0.1 * rng.normal(size=n) + (labels == "A") * 3.0
    inflow = np.clip(-field, 0, None)
    x = rng.normal(size=(n, 4))
    x[:, 0] += 2 * field  # follows the net flow within a type
    x[:, 1] += 3 * (labels == "A")  # a marker of the type that sources are enriched for, and nothing else
    tables = attention_biology(out, inflow, x, ["up", "marker", "n1", "n2"], labels, sections, pos, k=8, n_top=2)
    genes = tables["genes"].set_index("gene")["r"]
    assert genes["up"] > 0.3 and abs(genes["marker"]) < 0.1 and abs(genes["n1"]) < 0.1
    across = np.corrcoef(x[:, 1], out - inflow)[0, 1]
    assert across > 0.5  # the marker follows the flow across types: only the within-type reading removes it
    assert tables["genes"].iloc[0]["gene"] == "up" and tables["top_up"].iloc[0]["gene"] == "up"
    by_type = tables["by_type"].groupby("cell_type")["mean_net"].mean()
    assert by_type["A"] > by_type["B"]
    assert set(tables["by_type"]["section"]) == {"s1", "s2"}


def test_attention_pairs_read_above_what_colocation_explains():
    rng = np.random.default_rng(0)
    n = 1200
    pos = rng.uniform(0, 1000, size=(n, 2))
    labels = rng.choice(["A", "B", "C"], size=n)
    receivers = np.flatnonzero(labels == "B")[:200]
    tree_a = np.flatnonzero(labels == "A")
    sources = np.stack(
        [tree_a[np.argsort(np.linalg.norm(pos[tree_a] - pos[r], axis=1))[:3]] for r in receivers]
    )  # B reads the nearest As
    table = attention_pairs(
        receivers, sources, np.ones(sources.shape), labels, pos, grid_um=200.0, n_perm=40
    ).set_index(["source", "receiver"])
    assert table.loc[("A", "B"), "z"] > 5 and table.loc[("A", "B"), "log2_ratio"] > 0.5
    assert table.loc[("C", "B"), "flow"] == 0 and table.loc[("A", "C"), "flow"] == 0
    assert table.loc[("A", "B"), "median_distance_um"] < 150
    assert not len(attention_pairs(np.zeros(0, int), np.zeros((0, 3), int), np.zeros((0, 3)), labels, pos))


# ----------------------------------------------------------------------------------------------- programmes


def _readout(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1000, size=(n, 2))
    # Cell types in patches: the identity programme is spatially autocorrelated, what the type leaves of it is not.
    patches = (pos // 60).astype(int)
    patch_type = np.random.default_rng(seed + 1).choice(["A", "B", "C"], size=(18, 18))
    labels = patch_type[patches[:, 0], patches[:, 1]]
    sections = np.where(np.arange(n) < n // 2, "s1", "s2")
    conditions = np.where(sections == "s1", "ND", "T1D")
    identity = (labels == "A") * 3.0 + rng.normal(scale=0.2, size=n)
    spatial = np.sin(pos[:, 0] / 70) * np.cos(pos[:, 1] / 70) + 0.1 * rng.normal(size=n)
    disease = (conditions == "T1D") * (labels == "B") * 2.0 + rng.normal(scale=0.2, size=n)
    noise = rng.normal(scale=0.05, size=n)
    z = np.column_stack([identity, spatial, disease, noise])
    genes = (
        [f"id{k}" for k in range(10)]
        + [f"sp{k}" for k in range(10)]
        + [f"RPL{k}" for k in range(10)]
        + [f"bg{k}" for k in range(30)]
    )
    weight = np.zeros((len(genes), 4))
    weight[:10, 0], weight[10:20, 1], weight[20:30, 2] = 1.0, 1.0, 1.0
    weight += 0.01 * rng.normal(size=weight.shape)
    weight[30:, 3] = 0.1
    r = DimensionReadout(
        z=z,
        true=rng.normal(size=(n, len(genes))),
        hidden=np.ones((n, len(genes)), bool),
        weight=weight,
        bias=np.zeros(len(genes)),
        graph=(sections == "s2").astype(int),
        pos=pos,
    )
    return r, genes, labels, sections, conditions


def test_programmes_recover_the_parts_and_scores_track_them():
    r, genes, labels, sections, _ = _readout()
    loadings, share, scores = programmes(r, k=3)
    assert loadings.shape == (60, 3) and scores.shape == (2400, 3) and np.all(np.diff(share) <= 1e-9)
    assert share.sum() <= 1.0 + 1e-9
    assert np.allclose(
        scores.mean(axis=0), 0.0, atol=1e-9
    )  # the embedding's mean is the decoder's bias, not a programme
    spatial_truth = r.z[:, 1]
    best = np.abs([np.corrcoef(scores[:, c], spatial_truth)[0, 1] for c in range(3)]).max()
    assert best > 0.95
    for group in (range(0, 10), range(10, 20), range(20, 30)):
        chosen = np.argmax(np.abs(loadings[list(group)]).sum(axis=0))
        assert (loadings[:, chosen] ** 2)[list(group)].sum() > 0.8 * (loadings[:, chosen] ** 2).sum()


def test_the_cards_say_what_each_programme_is():
    r, genes, labels, sections, conditions = _readout()
    sets = {"lib": {"Identity set": [f"id{k}" for k in range(10)], "Other": [f"bg{k}" for k in range(10)]}}
    out = programme_cards(
        r,
        genes,
        labels,
        sections,
        conditions=conditions,
        gene_sets=sets,
        handling_genes=[f"RPL{k}" for k in range(10)],
        k=3,
        top=10,
    )
    cards = out["cards"]
    by = {card["top_genes"][:2]: card for card in cards.to_dict("records")}
    assert by["id"]["class"] == "identity" and by["id"]["top_type"] == "A" and by["id"]["eta2"] > 0.8
    assert "Identity set" in by["id"]["pathway"] and by["id"]["handling_share"] == 0
    assert (
        by["sp"]["class"] in ("spatial state", "niche")
        and by["sp"]["moran_within_near"] > 0.5
        and by["sp"]["eta2"] < 0.1
    )
    assert by["RP"]["class"] == "handling" and by["RP"]["handling_share"] > 0.8
    assert (
        by["RP"]["condition_type"] == "B"
        and by["RP"]["condition_effect"] * np.sign(out["loadings"][20, by["RP"]["programme"]]) > 1
    )
    assert by["id"]["moran_within_near"] < 0.1  # what the type leaves is noise
    raw = out["scores"][:, by["id"]["programme"]]
    assert moran(raw, r.pos, sections, k=8) > 0.3  # the score itself is not: it follows the patches
    assert set(out["by_type"]["cell_type"]) == {"A", "B", "C"}
    with pytest.raises(ValueError, match="one entry per kept row"):
        programme_cards(r, genes, labels[:5], sections, k=3)


def test_the_class_rules_apply_in_their_stated_order():
    assert programme_class(0.001, 0.9, 0.9, 0.9, 0.9) == "weak"
    assert programme_class(0.1, 0.9, 0.9, 0.9, 0.9) == "handling"
    assert programme_class(0.1, 0.9, 0.5, 0.5, 0.0) == "niche"  # a type's programme that also varies within the type
    assert programme_class(0.1, 0.9, 0.5, 0.0, 0.0) == "spatial state"
    assert programme_class(0.1, 0.9, 0.0, 0.0, 0.0) == "identity"
    assert programme_class(0.1, 0.1, 0.0, 0.0, 0.0) == "other"
    assert programme_class(float("nan"), 0.9, 0.0, 0.0, 0.0) == "weak"


def test_programmes_match_over_runs_through_a_permutation_and_a_sign():
    a = np.random.default_rng(0).normal(size=(50, 5))
    b = a[:, [2, 0, 1, 4, 3]] * np.array([1, -1, 1, 1, -1])
    matched = match_programmes(a, b)
    assert matched["b"].tolist() == [1, 2, 0, 4, 3] and np.allclose(matched["cosine"], 1.0)
    assert matched["sign"].tolist() == [-1, 1, 1, -1, 1]


# ----------------------------------------------------------------------------------------------- the report


def _toy_run(folder, seed=0, *, labels=None, sections=1):
    labels = _labels(seed) if labels is None else labels
    module, graph = _Known(labels), _graph(labels, seed)
    frames, targets = [], []
    for condition in ("ND", "T1D"):
        for k in range(sections):
            flows, top = type_flows(
                module, graph, labels, GENES, n_draws=1, per_type=30, far_per_type=8, far_radius=180.0, seed=seed + 10 * k
            )
            frames.append(flows.assign(section=f"s_{condition}" + (f"{k}" if sections > 1 else ""), condition=condition))
            targets.append(top)
    flows = pd.concat(frames)
    r, genes, lab, sections, conditions = _readout(seed=seed)
    cards = programme_cards(
        r,
        genes,
        lab,
        sections,
        conditions=conditions,
        gene_sets={"controls": {"T1D_x": [f"id{k}" for k in range(5)]}},
        handling_genes=[f"RPL{k}" for k in range(10)],
        k=3,
        top=10,
    )
    gene = gene_flows(module, [graph], [labels], GENES, ["g3", "g4", "g6"], per_type=30, seed=seed)
    score, top_genes = gene_flow_tables(gene)
    lr = lr_support(score, [("g3", "g0"), ("g4", "g2")], n_perm=20)
    write_tables(
        folder,
        {
            "flows": flows,
            "flow_targets": pd.concat(targets),
            "context_use": context_use(flows),
            "gene_top": top_genes,
            "lr_summary": pd.DataFrame([{k: v for k, v in lr.items() if k != "pairs"}]),
            "lr_pairs": lr["pairs"],
            "programme_cards": cards["cards"],
            "programme_by_type": cards["by_type"],
            "programme_enrichment": cards["enrichment"],
            "programme_loadings": cards["loadings"],
            "attention_pairs": pd.DataFrame(
                [
                    {"source": a, "receiver": b, "log2_ratio": 0.5 if a == "A" else -0.2, "condition": c}
                    for a in "ABC"
                    for b in "ABC"
                    for c in ("ND", "T1D")
                ]
            ),
            "attention_by_type": pd.DataFrame(
                [
                    {
                        "section": "s",
                        "cell_type": t,
                        "mean_net": 0.1,
                        "share_top_decile": 0.2,
                        "share_bottom_decile": 0.0,
                    }
                    for t in "ABC"
                ]
            ),
            "attention_genes": pd.DataFrame({"gene": GENES, "r": np.linspace(-0.3, 0.3, G)}),
        },
    )
    np.savez(folder / "tables" / "programme_map.npz", pos=r.pos, scores=cards["scores"], name="s1")


def test_a_report_is_rebuilt_from_its_tables_and_the_summary_reads_over_runs(tmp_path):
    for k, name in enumerate(("rand50_f0", "rand50_f0_untrained", "rand50_f1")):
        _toy_run(tmp_path / "runs" / name, seed=k)
    path = write_summary(tmp_path)
    text = path.read_text()
    assert (
        "rand50_f0_untrained" in text
        and "Flows that repeat" in text
        and "Programmes found again" in text
        and "one donor pair per run" in text
    )
    report = (tmp_path / "runs" / "rand50_f0" / "report.md").read_text()
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "How far this goes"):
        assert heading in report
    assert "Spearman" in report and "T1D_x" in report
    figures = {p.name for p in (tmp_path / "runs" / "rand50_f0" / "figures").glob("*.png")}
    assert {
        "flows_context.png",
        "flows_far.png",
        "attention_pairs.png",
        "programmes_by_type.png",
        "programmes_map.png",
    } <= figures


def test_the_planted_check_is_read_against_the_twin(tmp_path):
    def recovery(name, ranks):
        write_tables(tmp_path / "runs" / name, {"edge_recovery": pd.DataFrame(
            {"target": [f"t{k}" for k in range(len(ranks))], "best_rank": ranks, "senders_scored": 200, "planted": 1, "chance_top1": 0.005}
        )})  # fmt: skip

    recovery("full_s42", [1, 1, 1, 1, 7])
    recovery("full_s43", [1, 3, 1, 1, 1])
    recovery("full_s42_untrained", [1, 200, 1, 199, 2])
    assert twin_of(tmp_path / "runs" / "full_s42").name == "full_s42_untrained"
    assert twin_of(tmp_path / "runs" / "full_s43").name == "full_s42_untrained"  # another seed of the same arm
    assert twin_of(tmp_path / "runs" / "full_s42_untrained") is None
    found = planted_against_twin(tmp_path / "runs" / "full_s42")
    assert found["first"] == pytest.approx(0.8) and found["twin_first"] == pytest.approx(0.4) and found["beyond_twin"] == pytest.approx(0.4)
    report = write_run_report(tmp_path / "runs" / "full_s42").read_text()
    assert "untrained twin (full_s42_untrained): 40%" in report
    assert "planted_beyond_twin" in write_summary(tmp_path).read_text()


def test_old_known_pair_tables_get_the_clustered_p(tmp_path):
    rng = np.random.default_rng(3)
    pairs = pd.DataFrame({"ligand": [f"L{k % 20}" for k in range(120)], "receptor": [f"R{k % 3}" for k in range(120)], "receiver": "A",
                          "score": rng.normal(size=120), "auc": np.r_[np.full(40, 0.2), rng.random(80)]})  # fmt: skip
    write_tables(tmp_path, {"lr_pairs": pairs, "lr_summary": pd.DataFrame([{"n_pairs": 120, "auc": pairs["auc"].mean(), "null_mean": 0.5, "null_sd": 0.026, "p": 1.0}])})
    row = lr_row(tmp_path / "tables")
    assert row["p_independent"] == 1.0 and row["cluster_sd"] > 0.026 and 0 < row["p"] < 1


def test_condition_differences_use_sections_as_replicates(tmp_path):
    def flows(section, condition, value):
        return pd.DataFrame(
            [{"sender": a, "receiver": b, "kind": "state", "region": "context", "sum_delta": value if (a, b) == ("A", "B") else 0.1, "n_scored": 1.0,
              "section": section, "condition": condition} for a in "AB" for b in "AB"]
        )  # fmt: skip

    table = pd.concat([flows(f"nd{k}", "ND", 0.1 + 0.01 * k) for k in range(3)] + [flows(f"t{k}", "T1D", 0.5 + 0.01 * k) for k in range(3)])
    found = condition_by_section(table).set_index(["sender", "receiver"])
    assert found.loc[("A", "B"), "separated"] and found.loc[("A", "B"), "p"] == pytest.approx(0.1)
    assert found.loc[("A", "B"), "diff"] == pytest.approx(0.4) and not found.loc[("A", "A"), "separated"]
    assert condition_by_section(table[table["section"].isin(["nd0", "t0"])]).empty  # one section each: a donor pair, read over runs instead


def test_the_summary_reads_conditions_over_sections_and_gene_flows_over_runs(tmp_path):
    for k, name in enumerate(("rand50_s42", "rand50_s42_untrained", "rand50_s43")):
        _toy_run(tmp_path / "runs" / name, seed=k, sections=3)
    text = write_summary(tmp_path).read_text()
    assert "sections as replicates" in text and "share their donors" in text and "one donor pair per run" not in text
    assert "Gene flows that repeat over runs" in text
    assert {"flows_over_runs.png", "condition_over_runs.png"} <= {f.name for f in (tmp_path / "figures").glob("*.png")}
    assert "figures/flows_over_runs.png" in text and "figures/condition_over_runs.png" in text
    assert "3 ND, 3 T1D test sections" in (tmp_path / "runs" / "rand50_s42" / "report.md").read_text()


def test_a_run_with_no_tables_still_writes_a_report(tmp_path):
    folder = tmp_path / "runs" / "empty" / "tables"
    folder.mkdir(parents=True)
    text = write_run_report(folder.parent).read_text()
    assert "No flow table" in text and "No programme table" in text


def test_the_markdown_table_escapes_and_rounds():
    assert "a/b" in md_table(pd.DataFrame({"x": ["a|b"], "y": [1.23456]}))
    assert "1.23" in md_table(pd.DataFrame({"y": [1.23456]}))
    assert md_table(pd.DataFrame()) == "_(empty)_"
