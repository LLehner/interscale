"""Gene rankings of interaction (``evaluation.gene_ranking``; ``.claude/ccc_model_plan.md``, E7).

Pinned on a grid with known dependence: in type B, target g0 reads type A neighbours' g3 within 150 µm (a cross-gene interaction), g2 reads g4 of the cell
400 µm away (any type), g1 is the cell's own g5 (no interaction), and g6 is a smooth field the model reads from the neighbours' own g6 (the same gene in
other cells: smoothness, not an interaction). So the responders are g0 and g2, the drivers g3 and g4, the pairs g3 -> g0 and g4 -> g2; g6 responds only
through itself and must lose its score once the same gene is taken off; g1 never moves. Then the folds, the truth builders, the metrics by hand and the
model-free bars on planted data.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.gene_ranking import (
    ContextFlows,
    co_varying_groups,
    context_flows,
    group_pair_scores,
    group_sets,
    group_table,
    hide_cells,
    robust_z,
    driver_table,
    gene_folds,
    hide_genes,
    merge_context_flows,
    merge_gene_flows,
    neighbour_ridge,
    pair_table,
    pooled_pair_scores,
    programme_truth,
    ranking_metrics,
    responder_table,
    roles_truth,
    show_everything,
    spillover_flags,
    within_group_moran,
    within_group_variance,
)
from interscale.evaluation.interpretation import EVERYONE, GeneFlows, blind_to_own, gene_flows, scored_entries
from interscale.module.base._step_output import StepOutput, ViewOutput

SIDE, SPACING, G = 18, 50.0, 8
GENES = [f"g{k}" for k in range(G)]


def _pos():
    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float) * SPACING


def _labels(seed=0):
    return np.random.default_rng(seed).choice(["A", "B", "C"], size=SIDE * SIDE, p=[0.4, 0.4, 0.2])


def _within(pos, radius=150.0):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return (d <= radius) & (d > 0)


def _partner(pos):
    period = SIDE * SPACING
    target = np.column_stack([np.mod(pos[:, 0] + 400.0, period), pos[:, 1]])
    return np.argmin(np.linalg.norm(target[:, None] - pos[None], axis=-1), axis=1)


def _mean_over(weights, values):
    return (weights @ values) / np.maximum(weights.sum(1), 1)


class _Known(torch.nn.Module):
    """g0 (type B) = A neighbours' g3; g2 (type B) = the partner's g4; g6 = the neighbours' g6; g1 = the cell's own g5 unless ``blind``."""

    registered_global_component = True

    def __init__(self, labels, *, blind=False):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.labels = labels
        self.blind = blind
        if blind:
            self.context_only, self.own_cell_skip = "query", False

    def _common_step(self, batch, task, level):
        x, pos = batch.x.double(), batch.pos.double().numpy()
        near = torch.as_tensor(_within(pos), dtype=torch.float64)
        from_a = near * torch.as_tensor(self.labels == "A", dtype=torch.float64)[None, :]
        is_b = torch.as_tensor(self.labels == "B")
        pred = torch.zeros_like(x)
        pred[is_b, 0] = ((from_a @ x[:, 3]) / from_a.sum(1).clamp_min(1.0))[is_b]
        pred[is_b, 2] = x[torch.as_tensor(_partner(pos)), 4][is_b]
        pred[:, 6] = (near @ x[:, 6]) / near.sum(1).clamp_min(1.0)
        if not self.blind:
            pred[:, 1] = x[:, 5]
        gene_mask = getattr(batch, "gene_mask", None)
        entry = gene_mask.bool() if gene_mask is not None else None
        return StepOutput(y_pred=pred.float(), y_true=batch.x, entry_mask=entry, views=[ViewOutput()])


def _graph(labels, seed=0, *, hidden=(0, 1, 2)):
    rng = np.random.default_rng(seed)
    pos = _pos()
    x = rng.random((len(pos), G))
    near = _within(pos)
    from_a = near & (labels == "A")[None, :]
    x[:, 6] = _mean_over(near.astype(float), rng.random(len(pos)) * 3.0)  # a smooth field
    x[:, 0] = _mean_over(from_a.astype(float), x[:, 3])
    x[:, 2] = x[_partner(pos), 4]
    x[:, 1] = x[:, 5]
    d = Data(x=torch.tensor(x, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32))
    d.edge_index = torch.zeros((2, 0), dtype=torch.long)
    mask = torch.zeros(len(pos), G, dtype=torch.bool)
    mask[:, list(hidden)] = True
    d.gene_mask = mask
    return d


def _at(table, receiver, gene, column):
    return float(table[(table["receiver"] == receiver) & (table["gene"] == gene)][column].iloc[0])


# ----------------------------------------------------------------------------------------------- responders


def test_context_flows_find_the_responders_and_leave_graph_and_generators_alone():
    labels = _labels()
    module, graph = _Known(labels), _graph(labels)
    before = graph.x.clone()
    torch_state, np_state = torch.random.get_rng_state(), np.random.get_state()[1].copy()
    ctx = context_flows(module, [graph], [labels], GENES, per_type=40, seed=1)
    assert torch.equal(torch_state, torch.random.get_rng_state()) and np.array_equal(np_state, np.random.get_state()[1])
    assert torch.equal(graph.x, before)
    table = responder_table(ctx)
    assert _at(table, "B", "g0", "responder") > 0.1 and _at(table, "B", "g2", "responder") > 0.1
    # The own read: receivers keep their rows, so g1 never moves.
    assert abs(_at(table, "B", "g1", "responder")) < 1e-9 and abs(_at(table, "A", "g1", "responder")) < 1e-9
    # Only hidden entries are scored: a visible gene has no score.
    assert np.isnan(_at(table, "B", "g3", "responder"))
    # Sender types: B's g0 depends on A, not on C (identity minus its location-matched null).
    types = ctx.sender_types()
    a, c, b = ctx.senders.index("A"), ctx.senders.index("C"), ctx.receivers.index("B")
    assert types[a, b, 0] > types[c, b, 0] + 0.05


def test_every_entry_needs_a_model_blind_to_its_own_cell():
    labels = _labels()
    clean = show_everything([_graph(labels)])
    assert not blind_to_own(_Known(labels)) and blind_to_own(_Known(labels, blind=True))
    with pytest.raises(ValueError, match="never read a cell's own values"):
        scored_entries(clean[0], _Known(labels), "all")
    with pytest.raises(ValueError, match="never read a cell's own values"):
        gene_flows(_Known(labels), clean, [labels], GENES, ["g3"], targets="all", per_type=40)
    assert scored_entries(clean[0], _Known(labels, blind=True), "all").all()
    assert getattr(clean[0], "gene_mask", None) is None and not clean[0].mask.any()


def test_the_same_gene_in_other_cells_is_taken_off_and_cross_gene_pairs_stay():
    labels = _labels()
    module, graphs = _Known(labels, blind=True), show_everything([_graph(labels, seed=s) for s in range(2)])
    ctx = context_flows(module, graphs, [labels] * 2, GENES, targets="all", per_type=40, seed=2)
    flows = gene_flows(module, graphs, [labels] * 2, GENES, GENES, targets="all", per_type=40, seed=2)
    assert flows.base is not None and flows.base.shape == (len(flows.receivers), G)
    table = responder_table(ctx, flows)
    # g6 responds only through the same gene in other cells: its score goes once that is taken off.
    assert _at(table, "B", "g6", "responder") > 0.1
    assert _at(table, "B", "g6", "self") > 0.5 * _at(table, "B", "g6", "responder")
    assert abs(_at(table, "B", "g6", "responder_cross")) < 0.5 * _at(table, "B", "g6", "responder")
    # g0 reads another gene: its own gene in other cells carries nothing.
    assert abs(_at(table, "B", "g0", "self")) < 1e-9 and _at(table, "B", "g0", "responder_cross") > 0.1
    pairs = pair_table(flows, top=3)
    top_b = pairs[pairs["receiver"] == "B"]
    assert set(zip(top_b["sender"], top_b["target"]))  >= {("g3", "g0"), ("g4", "g2")}
    assert not ((pairs["sender"] == pairs["target"]).any())  # g6 -> g6 is no pair
    pooled = pooled_pair_scores(flows)
    assert pooled.idxmax() in {("g3", "g0"), ("g4", "g2")}
    drivers = driver_table(flows).set_index(["receiver", "sender"])
    assert drivers.loc[("B", "g3"), "positive"] > 0.1 and drivers.loc[("B", "g4"), "positive"] > 0.1
    # g6 drives only itself, which a driver score leaves out.
    assert drivers.loc[("B", "g6"), "positive"] < 0.05 * drivers.loc[("B", "g3"), "positive"]
    assert (EVERYONE, "g3") in drivers.index


# ----------------------------------------------------------------------------------------------- folds


def test_folds_hide_every_gene_once_per_partition_and_merges_average_where_read():
    folds = gene_folds(10, 0.2, partitions=2, seed=0)
    assert len(folds) == 10
    for p in range(2):
        assert sorted(np.concatenate(folds[5 * p : 5 * p + 5]).tolist()) == list(range(10))
    labels = _labels()
    graph = _graph(labels)
    hidden = hide_genes([graph], folds[0])[0]
    assert hidden.gene_mask[:, folds[0]].all() and not hidden.gene_mask[:, np.setdiff1d(np.arange(G), folds[0])[:1]].any()
    # As the gene-masking dataloader sets it (apply_mask requires it): the cells with a hidden entry.
    assert torch.equal(hidden.mask, hidden.gene_mask.any(dim=1))
    assert torch.equal(graph.gene_mask, _graph(labels).gene_mask)  # the original untouched
    nan = np.nan
    one = GeneFlows(["g3"], ["B"], ["g0", "g1"], np.array([[[0.2, nan]]]), np.zeros((1, 1, 2)), np.array([[5.0, 0.0]]), np.ones((1, 2)), np.array([[0.5, nan]]))
    two = GeneFlows(["g4", "g3"], ["B"], ["g0", "g1"], np.array([[[nan, 0.3]], [[0.4, 0.1]]]), np.zeros((2, 1, 2)), np.array([[0.0, 5.0]]), np.ones((1, 2)),
                    np.array([[nan, 0.6]]))  # fmt: skip
    merged = merge_gene_flows([one, two])
    assert merged.senders == ["g3", "g4"]
    np.testing.assert_allclose(merged.drop[0, 0], [0.3, 0.1])  # g3 -> g0 read twice: averaged
    np.testing.assert_allclose(merged.drop[1, 0], [nan, 0.3])
    np.testing.assert_allclose(merged.base[0], [0.5, 0.6])
    np.testing.assert_allclose(merged.count[0], [5.0, 5.0])
    ctx = ContextFlows(["all_state"], [EVERYONE], ["B"], ["g0"], np.array([[[[0.2]]]]), np.array([[0.5]]), np.array([[3.0]]), np.ones((1, 1)))
    ctx_nan = ContextFlows(["all_state"], [EVERYONE], ["B"], ["g0"], np.array([[[[nan]]]]), np.array([[nan]]), np.array([[0.0]]), np.ones((1, 1)))
    assert merge_context_flows([ctx, ctx_nan]).responder()[0, 0] == pytest.approx(0.2)


def test_fold_reads_of_a_channel_style_model_find_the_cross_gene_pairs():
    labels = _labels()
    module = _Known(labels)
    graphs = [_graph(labels, seed=s, hidden=()) for s in range(2)]
    parts, ctx_parts = [], []
    for fold in [np.array([0, 2, 6]), np.array([1, 3, 4, 5, 7])]:
        masked = hide_genes(graphs, fold)
        visible = [GENES[g] for g in range(G) if g not in set(fold.tolist())]
        parts.append(gene_flows(module, masked, [labels] * 2, GENES, visible, per_type=40, seed=3))
        ctx_parts.append(context_flows(module, masked, [labels] * 2, GENES, kinds=("all_state",), per_type=40, seed=3))
    flows, ctx = merge_gene_flows(parts), merge_context_flows(ctx_parts)
    table = responder_table(ctx, flows)
    # Hidden in every cell, a target is never a sender of its own: nothing is taken off.
    assert np.isnan(_at(table, "B", "g0", "self")) and _at(table, "B", "g0", "responder_cross") == _at(table, "B", "g0", "responder")
    pooled = pooled_pair_scores(flows)
    assert pooled[("g3", "g0")] > 0.1 and pooled[("g4", "g2")] > 0.1


# ----------------------------------------------------------------------------------------------- truth and metrics


def test_ranking_metrics_by_hand():
    scores = pd.Series({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.1, "e": np.nan})
    out = ranking_metrics(scores, {"a", "c", "e"}, negatives={"b"})
    assert out["n_pos"] == 2 and out["n_scored"] == 4 and out["base_rate"] == 0.5
    assert out["auprc"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)  # precision at a = 1, at c = 2/3
    assert out["auroc"] == pytest.approx(0.75)  # a beats b and d, c beats d only
    assert out["auroc_hard"] == pytest.approx(0.5)  # a beats b, c does not
    assert out["precision_at_k"] == pytest.approx(0.5) and out["k"] == 2
    assert out["median_rank"] == pytest.approx(2.0)
    assert np.isnan(ranking_metrics(scores, set())["auprc"])


def test_truth_from_roles_and_from_planted_programmes():
    var = pd.DataFrame(
        {
            "role": ["source", "response", "response", "sender", "receiver", "spatial", "noise", "receptor"],
            "interaction": ["M7>MC7/MC8", "M7>MC7/MC8", "M7>MC7/MC8", "S1>R1", "S1>R1", "", "", "M12>MC16"],
        },
        index=["M7", "MC7", "MC8", "S1", "R1", "SP1", "H1", "MR16"],
    )
    truth = roles_truth(var)
    assert truth["responders"] == {"MC7", "MC8", "R1"} and truth["drivers"] == {"M7", "S1"}
    assert truth["pairs"] == {("M7", "MC7"), ("M7", "MC8"), ("S1", "R1")}
    assert truth["hard_negatives"] == {"SP1"}
    programmes = pd.DataFrame(
        {
            "program": ["Immune>Ductal@step20x2", "null:Mast>Endothelial@exp50x2"],
            "sender": ["Immune", "Mast"],
            "receiver": ["Ductal", "Endothelial"],
            "genes": ["G1, G2", "G3"],
            "spike": [True, False],
        }
    )
    planted = programme_truth(programmes)
    assert planted["responders"] == {"Ductal": {"G1", "G2"}} and planted["controls"] == {"G3"}
    assert planted["sender_of"] == {("Ductal", "G1"): "Immune", ("Ductal", "G2"): "Immune"}


# ----------------------------------------------------------------------------------------------- bars


def _planted_sections(n, seed):
    """Sections where B's g0 = A neighbours' g3 (+ noise) and g6 is a smooth field; returned as one dict of stacked cells."""
    rng = np.random.default_rng(seed)
    xs, poss, labs, graphs = [], [], [], []
    for k in range(n):
        labels = _labels(seed + k)
        pos = _pos()
        x = rng.normal(size=(len(pos), G))
        near = _within(pos)
        x[:, 6] = _mean_over(near.astype(float), rng.normal(size=len(pos)) * 3.0)
        from_a = near & (labels == "A")[None, :]
        x[:, 0] = np.where(labels == "B", _mean_over(from_a.astype(float), x[:, 3]) * 3.0 + 0.3 * rng.normal(size=len(pos)), x[:, 0])
        x[:, 7] = np.where(labels == "A", x[:, 7] + 10.0, x[:, 7])  # a marker of A
        xs.append(x), poss.append(pos), labs.append(labels), graphs.append(np.full(len(pos), f"s{seed}_{k}"))
    return {"x": np.vstack(xs), "pos": np.vstack(poss), "labels": np.concatenate(labs), "graphs": np.concatenate(graphs)}


def test_the_neighbour_ridge_finds_the_cross_gene_dependence_and_not_a_gene_from_its_own_neighbours():
    train, test = _planted_sections(3, seed=0), _planted_sections(2, seed=10)
    ridge = neighbour_ridge(train, test, GENES, k=24, seed=0)
    b = ridge.receivers.index("B")
    # The neighbour means do not tell the sender's type, so A's g3 is diluted by B's and C's (0.2-0.3 here, not 1): a weak bar on purpose, cheap
    # on a 1,000-gene panel. It still ranks the planted target first.
    assert ridge.responder[b, 0] > 0.15 and int(np.nanargmax(ridge.responder[b])) == 0
    # g6 is smooth, but its own neighbours are left out and no other gene carries it.
    assert ridge.responder[b, 6] < 0.3
    assert int(np.argmax(ridge.pairs[b, :, 0])) == 3
    assert np.all(np.diag(ridge.pairs[b]) == 0.0)
    assert int(np.argmax(ridge.drivers()[b])) == 3


def test_moran_variance_and_spillover_bars():
    data = _planted_sections(2, seed=3)
    moran = within_group_moran(data["x"], data["pos"], data["labels"], data["graphs"], k=6)
    assert moran.loc["B", 6] > 0.3 and abs(moran.loc["B", 5]) < 0.15
    variance = within_group_variance(data["x"], data["labels"], data["graphs"])
    assert variance.loc["B", 0] > variance.loc["C", 0]  # only B's g0 carries the planted signal on top of noise
    flags = spillover_flags(data["x"], data["labels"])
    assert flags.loc["B", 7] and flags.loc["C", 7] and not flags.loc["A", 7]


# ----------------------------------------------------------------------------------------------- fixes: strata, sender sets, groups, robust z


def test_strata_constant_draws_as_before_and_singleton_strata_move_nothing():
    labels = _labels()
    module = _Known(labels, blind=True)
    graphs = show_everything([_graph(labels, seed=s, hidden=()) for s in range(2)])
    kw = dict(per_type=40, seed=3, targets="all")
    plain = gene_flows(module, graphs, [labels] * 2, GENES, ["g3", "g4"], **kw)
    const = gene_flows(module, graphs, [labels] * 2, GENES, ["g3", "g4"], strata=[np.zeros(len(labels))] * 2, **kw)
    np.testing.assert_allclose(plain.drop, const.drop, equal_nan=True)
    ctx_plain = context_flows(module, graphs, [labels] * 2, GENES, targets="all", kinds=("all_state",), per_type=40, seed=3)
    ctx_const = context_flows(module, graphs, [labels] * 2, GENES, targets="all", kinds=("all_state",), per_type=40, seed=3,
                              strata=[np.zeros(len(labels))] * 2)  # fmt: skip
    np.testing.assert_allclose(ctx_plain.responder(), ctx_const.responder(), equal_nan=True)
    # every cell its own stratum: nothing can be permuted, so nothing drops
    own = [np.arange(len(labels))] * 2
    alone = gene_flows(module, graphs, [labels] * 2, GENES, ["g3", "g4"], strata=own, **kw)
    assert np.nanmax(np.abs(alone.drop)) < 1e-9
    ctx_alone = context_flows(module, graphs, [labels] * 2, GENES, targets="all", kinds=("all_state",), per_type=40, seed=3, strata=own)
    assert np.nanmax(np.abs(ctx_alone.responder())) < 1e-9
    assert plain.drop[0, plain.receivers.index("B"), 0] > 0.1  # the planted pair is read without strata


def test_sender_sets_share_one_permutation_and_leave_member_targets_out(monkeypatch):
    from interscale.evaluation import interpretation

    labels = _labels()
    module = _Known(labels, blind=True)
    graphs = show_everything([_graph(labels, seed=0, hidden=())])
    kw = dict(per_type=40, seed=5, targets="all")
    single = gene_flows(module, graphs, [labels], GENES, ["g3"], **kw)
    as_set = gene_flows(module, graphs, [labels], GENES, [], sender_sets={"S": ["g3"]}, **kw)
    assert as_set.senders == ["S"]
    keep = np.isfinite(as_set.drop[0])
    np.testing.assert_allclose(as_set.drop[0][keep], single.drop[0][keep])
    assert np.isnan(as_set.drop[0, :, GENES.index("g3")]).all()  # the member's own column is not read
    seen = []
    real = interpretation._predict

    def spy(module_, data_list, **k):
        seen.append(data_list[0].x.clone())
        return real(module_, data_list, **k)

    monkeypatch.setattr(interpretation, "_predict", spy)
    gene_flows(module, graphs, [labels], GENES, [], sender_sets={"S": ["g3", "g5"]}, **kw)
    original = graphs[0].x
    changed = seen[-1]
    before = {(round(float(a), 6), round(float(b), 6)) for a, b in original[:, [3, 5]].tolist()}
    after = {(round(float(a), 6), round(float(b), 6)) for a, b in changed[:, [3, 5]].tolist()}
    assert after <= before  # each cell shows another cell's (g3, g5) combination, never a mixed one
    assert not torch.equal(changed[:, 3], original[:, 3])


def test_group_table_sufficiency_and_credited_pairs_by_hand():
    nan = np.nan
    senders = ["a", "b", "c", "G0", "G0|keep:a", "G0|keep:b"]
    genes = ["a", "b", "c", "t"]
    drop = np.full((6, 1, 4), nan)
    drop[0, 0, 3], drop[1, 0, 3], drop[2, 0, 3] = 0.05, 0.04, 0.3  # alone, a and b move t little: they stand in for each other
    drop[3, 0, 3] = 0.4  # the whole group {a, b}
    drop[4, 0, 3] = 0.0  # a kept: nothing lost, a carries it alone
    drop[5, 0, 3] = 0.4  # b kept: everything lost, b carries nothing
    flows = GeneFlows(senders, ["all"], genes, drop, np.zeros((6, 1, 4)), np.ones((1, 4)), np.ones((1, 4)), np.full((1, 4), 0.5))
    table = group_table(flows, [["a", "b"]], min_expressed=0.0)
    row = table[(table["group"] == "G0") & (table["target"] == "t")].iloc[0]
    assert row["drop"] == pytest.approx(0.4) and row["best"] == "a" and row["best_sufficiency"] == pytest.approx(1.0)
    assert "a:1.00" in row["sufficiency"] and "b:0.00" in row["sufficiency"]
    assert set(table["group"]) == {"G0", "c"}
    credited = group_pair_scores(flows, [["a", "b"]])
    assert credited.senders == ["a", "b", "c"]
    np.testing.assert_allclose(credited.drop[:, 0, 3], [0.4, 0.0, 0.3])
    assert group_sets([["a", "b"]]) == {"G0": ["a", "b"], "G0|keep:a": ["b"], "G0|keep:b": ["a"]}


def test_robust_z_against_the_other_senders():
    rng = np.random.default_rng(0)
    drop = rng.normal(0.0, 0.01, size=(30, 1, 2))
    drop[0, 0, 1] = 0.2
    flows = GeneFlows([f"s{k}" for k in range(30)], ["all"], ["x", "y"], drop, np.zeros_like(drop), np.ones((1, 2)), np.ones((1, 2)))
    z = robust_z(flows, min_expressed=0.0)
    assert z.drop[0, 0, 1] > 10 and np.nanmax(np.abs(z.drop[1:, 0, 1])) < 5


def test_hide_cells_and_co_varying_groups():
    labels = _labels()
    graph = _graph(labels, hidden=())
    hidden = hide_cells([graph], 0.2, seed=0)[0]
    assert hidden.mask.float().mean().item() == pytest.approx(0.2, abs=0.01)
    assert torch.equal(hidden.gene_mask.all(dim=1), hidden.mask) and torch.equal(hidden.gene_mask.any(dim=1), hidden.mask)
    rng = np.random.default_rng(1)
    pos = _pos()
    field = np.sin(pos[:, 0] / 200.0)
    x = np.column_stack([field + rng.normal(0, 0.05, len(pos)), 2 * field + rng.normal(0, 0.05, len(pos)), -field + rng.normal(0, 0.05, len(pos)),
                         rng.normal(0, 1, len(pos))])  # fmt: skip
    groups = co_varying_groups(x, pos, np.zeros(len(pos)), ["a", "b", "c", "noise"], k=8)
    assert groups == [["a", "b", "c"]]
