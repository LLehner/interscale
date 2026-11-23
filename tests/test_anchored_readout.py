"""``evaluation.anchored``: the flipped task's readouts on a toy GlobalModel (stirred off its initialisation)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_visible_keys_backends import G, _build, _config, _lattice, _stir

from interscale.evaluation import anchored

GRID, R_DISC, R_MAX = 3, 150.0, 400.0


def _module():
    cfg = _config("dense", visible=False, max_seq_len=200, center_qk=False)
    cfg.dataset.mask_strategy, cfg.dataset.mask_percentage = "gene", 1.0
    return _stir(_build(cfg)).eval()


def _graphs():
    graphs = [_lattice(10, seed=1), _lattice(9, seed=2)]
    graphs[0].x[:, 5] = 2.0  # a gene constant everywhere: replacing it changes nothing
    graphs[1].x[:, 5] = 2.0
    return graphs


def test_disc_views_carry_the_training_draw():
    g = _graphs()[0]
    views = anchored.disc_views(g, [1, 2], grid=GRID, r_disc=R_DISC, r_max=R_MAX)
    assert len(views) == 9
    for _, view, disc, ring, _ in views:
        assert np.array_equal(view.key_mask.numpy(), disc)
        assert np.array_equal(view.gene_mask[:, [1, 2]].all(1).numpy(), ring >= 0)
        assert not view.gene_mask[:, [0, 3, 4, 5]].any()


def test_interventions_touch_only_the_disc_and_never_the_targets_own_column(monkeypatch):
    seen = []
    real = anchored._predict

    def spy(module, data_list, **kw):
        seen.append(data_list[0])
        return real(module, data_list, **kw)

    monkeypatch.setattr(anchored, "_predict", spy)
    graphs = _graphs()
    read = anchored.read_discs(_module(), graphs[:1], [f"g{k}" for k in range(G)], ["g1", "g2"], grid=GRID, r_disc=R_DISC,
                               r_max=R_MAX, senders=["g0", "g5"])
    original = graphs[0].x
    for data in seen:
        disc = data.key_mask.numpy()
        changed = (data.x != original).any(dim=1).numpy()
        assert not changed[~disc].any()  # receivers and every other cell keep their rows
    # responder passes (whole donor rows) keep the target's column: g1 and g2 are each kept in one of them
    responder_passes = [d for d in seen if (d.x != original).sum() > (d.key_mask.sum() * 2)]
    assert responder_passes
    assert all(((d.x[:, 1] == original[:, 1]).all() or (d.x[:, 2] == original[:, 2]).all()) for d in responder_passes)
    tables = read.tables()
    assert set(tables) == {"anch_base", "anch_pairs", "anch_groups", "anch_curves"}
    assert np.isfinite(read.base).all()


def test_a_constant_sender_has_no_effect_and_no_self_pairs():
    read = anchored.read_discs(_module(), _graphs(), [f"g{k}" for k in range(G)], ["g1", "g2"], grid=GRID, r_disc=R_DISC,
                               r_max=R_MAX, senders=["g5", "g0", "g1"], responders=False)
    pairs = read.tables()["anch_pairs"]
    const = pairs[pairs["sender"] == "g5"]
    assert np.allclose(const["shift"], 0.0) and np.allclose(const["drop"], 0.0, atol=1e-9)
    assert (pairs[pairs["sender"] == "g0"]["shift"] > 0).any()
    assert not (pairs["sender"] == pairs["target"]).any()


def test_curve_agreement_against_the_average():
    import pandas as pd

    bins = np.arange(5)
    curves = pd.DataFrame({"graph": "a", "disc": 0, "target": "t", "bin": bins, "pred": bins * 1.0, "truth": 0.0, "cells": 1})
    out = anchored.curve_agreement(curves, {("a", 0): {"t": bins * 2.0}}, {"t": -bins * 1.0})
    assert out["corr_model"].iloc[0] == pytest.approx(1.0) and out["corr_average"].iloc[0] == pytest.approx(-1.0)


GENES = [f"g{k}" for k in range(G)]


def _halves(graphs):
    return [np.where(g.pos[:, 0].numpy() < float(g.pos[:, 0].median()), "a", "b") for g in graphs]


def test_batched_passes_give_the_same_read():
    graphs = _graphs()
    kw = dict(grid=GRID, r_disc=R_DISC, r_max=R_MAX, senders=["g0", "g3"], groups=[["g0", "g3", "g1"]])
    one = anchored.read_discs(_module(), graphs, GENES, ["g1", "g2"], batch_size=1, **kw).tables()
    four = anchored.read_discs(_module(), graphs, GENES, ["g1", "g2"], batch_size=4, **kw).tables()
    for name in ("anch_base", "anch_pairs", "anch_groups"):
        a, b = one[name].select_dtypes("number").to_numpy(), four[name].select_dtypes("number").to_numpy()
        assert np.allclose(a, b, atol=1e-5, equal_nan=True), name


def test_the_first_scheme_draws_as_the_single_scheme_read_and_label_donors_keep_their_label():
    graphs = _graphs()
    kw = dict(grid=GRID, r_disc=R_DISC, r_max=R_MAX, senders=["g0", "g3"], responders=True)
    old = anchored.read_discs(_module(), graphs, GENES, ["g1", "g2"], **kw).tables()
    new = anchored.read_discs(_module(), graphs, GENES, ["g1", "g2"], donor_labels={"any": None, "half": _halves(graphs)}, **kw).tables()
    assert set(new["anch_pairs"]["scheme"]) == {"any", "half"}
    for name in ("anch_base", "anch_pairs"):
        a = old[name].drop(columns="scheme").select_dtypes("number").to_numpy()
        b = new[name][new[name]["scheme"] == "any"].drop(columns="scheme").select_dtypes("number").to_numpy()
        assert np.allclose(a, b, equal_nan=True), name
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b"] * 50)
    disc = np.zeros(100, bool)
    disc[:20] = True
    rows, donors = anchored._donor_rows(rng, disc, labels, 100)
    assert (labels[donors] == labels[rows]).all() and not disc[donors].any()


def test_group_readout_and_sufficiency():
    graphs = _graphs()
    # g5 is constant: the group {g0, g5} acts as g0 alone, so keeping g0 leaves nothing to drop (sufficiency 1)
    read = anchored.read_discs(_module(), graphs, GENES, ["g1", "g2"], grid=GRID, r_disc=R_DISC, r_max=R_MAX,
                               senders=["g0", "g5"], groups=[["g0", "g5"], ["g4"], ["g1", "g0"]], responders=False)
    assert read.groups == [["g0", "g5"], ["g1", "g0"]]  # singletons are the per-sender passes
    t = read.tables()
    grp, pairs = t["anch_groups"], t["anch_pairs"]
    whole = grp[(grp["members"] == "g0 g5") & (grp["kept"] == "")].sort_values(["target", "ring"])["drop"].to_numpy()
    single = pairs[pairs["sender"] == "g0"].sort_values(["target", "ring"])["drop"].to_numpy()
    assert np.allclose(whole, single, atol=1e-9)
    keep_g0 = grp[(grp["members"] == "g0 g5") & (grp["kept"] == "g0")]["drop"]
    assert np.allclose(keep_g0, 0.0, atol=1e-9)
    # the target's own gene is never replaced: for target g1 the group {g1, g0} replaces g0 only
    own = grp[(grp["members"] == "g1 g0") & (grp["target"] == "g1") & (grp["kept"] == "")].sort_values("ring")["drop"].to_numpy()
    g0_on_g1 = pairs[(pairs["sender"] == "g0") & (pairs["target"] == "g1")].sort_values("ring")["drop"].to_numpy()
    assert np.allclose(own, g0_on_g1, atol=1e-9)
    ranks = anchored.rank_tables(t, local_rings=[0])
    assert set(ranks) == {"rank_pairs", "rank_senders", "rank_responders", "rank_groups"}
    g = ranks["rank_groups"]
    row = g[(g["members"] == "g0 g5") & (g["target"] == "g2")].iloc[0]
    if row["total"] > 0:
        assert row["most_sufficient"] == "g0" and "g0:1.00" in row["sufficiency"]
    s = ranks["rank_senders"]
    assert list(s["rank"]) == list(range(1, len(s) + 1)) and s["total"].is_monotonic_decreasing
