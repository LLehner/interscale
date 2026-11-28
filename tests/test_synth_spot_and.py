"""The planted-AND control of synth_spot (``and_programme=True``; .claude/attribution_plan.md, S13, 2026-11-27).

Opt-in and appended: every other gene (counts, expected values, obs, edges) is the default dataset's exactly; R5 is an AND of an S5 and an S6
neighbour on the 4-neighbourhood, R6 the matched control (2.5 -> 5 -> 10, a constant step in log space), both read from the same senders;
S5 and S6 never share a spot; every edge joins a receiver to a sender among its four lattice neighbours.
"""

import numpy as np
import pytest

from interscale.evaluation.synth_spot import AND_LEVELS, AND_PROGRAMMES, SIGNAL_SCALE, make_synth_spot

SMALL = dict(n_slides_per_condition=3, n_side=16, n_cropped_per_condition=1, n_with_holes_per_condition=1, max_holes=2,
             n_val_per_condition=1, n_test_per_condition=1, n_hv=2, n_nb=2, n_zinb=2, n_poi=2, n_hk=4)  # fmt: skip


@pytest.fixture(scope="module")
def pair():
    return make_synth_spot(**SMALL), make_synth_spot(**SMALL, and_programme=True)


def test_the_and_programme_is_opt_in_and_leaves_every_other_gene_unchanged(pair):
    base, new = pair
    g = list(base.var_names)
    assert list(new.var_names) == g + ["S5", "S6", "R5", "R6"]
    np.testing.assert_array_equal(new[:, g].layers["counts"], base.layers["counts"])
    np.testing.assert_array_equal(new[:, g].layers["expected"], base.layers["expected"])
    for col in base.obs.columns.drop(["total_counts", "n_genes_by_counts"]):
        assert new.obs[col].astype(str).equals(base.obs[col].astype(str)), col
    eb, en = base.uns["synthetic"]["interaction_edges"], new.uns["synthetic"]["interaction_edges"]
    key = ["sender", "receiver", "program"]
    assert set(map(tuple, eb[key].astype(str).values)) <= set(map(tuple, en[key].astype(str).values))
    assert "and_programme" not in base.uns["synthetic"]["params"] and new.uns["synthetic"]["params"]["and_programme"]


def test_r5_is_an_and_and_r6_doubles_per_sender_type(pair):
    _, new = pair
    obs = new.obs
    n5, n6 = obs["n_S5_nbrs"].to_numpy(), obs["n_S6_nbrs"].to_numpy()
    k = (n5 > 0).astype(int) + (n6 > 0).astype(int)
    expected = new[:, ["R5", "R6"]].layers["expected"]
    for j, gene in enumerate(("R5", "R6")):
        np.testing.assert_allclose(expected[:, j], np.asarray(AND_LEVELS[gene])[k] * SIGNAL_SCALE)
    # the rule itself, not only the table: R5 moves only with both sender types, R6 with each; same endpoints
    by = {gene: [np.unique(expected[k == m, j]) for m in range(3)] for j, gene in enumerate(("R5", "R6"))}
    assert all(len(v) == 1 for vs in by.values() for v in vs)
    r5, r6 = ([float(v[0]) for v in by[g]] for g in ("R5", "R6"))
    assert r5[0] == r5[1] < r5[2] and r6[0] < r6[1] < r6[2] and r5[0] == r6[0] and r5[2] == r6[2]
    state = obs["R56_state"].astype(str).to_numpy()
    assert ((state == "both") == ((n5 > 0) & (n6 > 0))).all() and (state == "both").any()
    assert not (obs["is_S5"].to_numpy() & obs["is_S6"].to_numpy()).any()  # disjoint senders: hiding one spot never removes both
    # R6 is additive where the models read it (log-normalised): the two steps of the expected value are equal in log
    lo, mid, hi = (np.log(np.asarray(AND_LEVELS["R6"]) * SIGNAL_SCALE)).tolist()
    assert mid - lo == pytest.approx(hi - mid)


def test_and_edges_join_receivers_to_their_lattice_neighbours(pair):
    _, new = pair
    e = new.uns["synthetic"]["interaction_edges"]
    pos = new.obsm["spatial"]
    slides = new.obs["slide"].astype(str).to_numpy()
    for gene, program in AND_PROGRAMMES.items():
        sub = e[e["program"] == program]
        assert len(sub)
        s, r = sub["sender"].to_numpy(), sub["receiver"].to_numpy()
        assert (slides[s] == slides[r]).all()
        np.testing.assert_allclose(np.linalg.norm(pos[s] - pos[r], axis=1), 100.0)
        is5, is6 = new.obs["is_S5"].to_numpy(), new.obs["is_S6"].to_numpy()
        assert (is5[s] | is6[s]).all()
        both = (new.obs["R56_state"].astype(str).to_numpy() == "both")
        if gene == "R5":
            assert both[r].all() and set(np.flatnonzero(both)) == set(r)
        else:
            assert (new.obs["R56_state"].astype(str).to_numpy()[r] != "none").all()


def test_the_strong_and_keeps_every_other_gene_and_the_default_levels_are_unchanged(pair):
    """weekend_plan.md A2 (2026-11-28): ``and_rate`` / ``and_high`` change only S5 / S6 / R5 / R6; the defaults are the 2026-11-27 control;
    R6 stays a constant log step (the geometric middle); out-of-range values are refused."""
    from interscale.evaluation.synth_spot import and_levels

    base, new = pair
    assert and_levels() == {g: tuple(map(float, v)) for g, v in AND_LEVELS.items()}
    strong = make_synth_spot(**SMALL, and_programme=True, and_rate=0.15, and_high=40.0)
    g = list(base.var_names)
    np.testing.assert_array_equal(strong[:, g].layers["counts"], base.layers["counts"])
    lv = strong.uns["synthetic"]["params"]["and_levels"]
    np.testing.assert_allclose(lv["R5"], [2.5, 2.5, 40.0])
    np.testing.assert_allclose(lv["R6"], [2.5, 10.0, 40.0])
    assert np.log(lv["R6"][1]) - np.log(lv["R6"][0]) == pytest.approx(np.log(lv["R6"][2]) - np.log(lv["R6"][1]))
    assert strong.obs["is_S5"].mean() > new.obs["is_S5"].mean()  # more senders at rate 0.15 than 0.08
    expected = strong[:, ["R5"]].layers["expected"][:, 0] / SIGNAL_SCALE
    both = strong.obs["R56_state"].astype(str).to_numpy() == "both"
    np.testing.assert_allclose(expected[both], 40.0)
    with pytest.raises(ValueError, match="and_rate"):
        make_synth_spot(**SMALL, and_programme=True, and_rate=0.6)
    with pytest.raises(ValueError, match="and_high"):
        make_synth_spot(**SMALL, and_programme=True, and_high=2.0)
