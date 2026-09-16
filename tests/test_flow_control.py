"""The attention-flow control: does net flow follow signalling, or merely co-occurrence?

A probe score cannot answer that -- nothing in it constrains which cells the transformer attends
to -- so this is a separate instrument, and its job is to be *falsifiable*. The tests below are
mostly about the ways a control can quietly stop controlling: no pairs configured, pairs written
for another dataset, or a summary read without knowing it came from a single pair.
"""

import numpy as np
import pandas as pd
import pytest

from interscale.evaluation.flow_control import (
    net_flow_control,
    net_flow_control_report,
    parse_pairs,
)

TYPES = ["senderA", "receiverA", "senderB", "receiverB", "stroma"]


def _flow(entries: dict[tuple[str, str], float], types=TYPES) -> pd.DataFrame:
    """An antisymmetric net-flow matrix with the named sender->receiver values set."""
    frame = pd.DataFrame(0.0, index=types, columns=types)
    for (sender, receiver), value in entries.items():
        frame.loc[sender, receiver] = value
        frame.loc[receiver, sender] = -value
    return frame


# --------------------------------------------------------------------------- parsing


def test_pairs_parse_from_strings():
    assert parse_pairs(["a>b", " c > d "]) == [("a", "b"), ("c", "d")]


@pytest.mark.parametrize("bad", ["ab", "a>b>c", "a>", ">b"])
def test_a_malformed_pair_is_rejected_by_name(bad):
    with pytest.raises(ValueError, match="sender>receiver"):
        parse_pairs([bad])


# --------------------------------------------------------------------------- the control itself


def test_a_null_pair_with_no_flow_scores_zero_and_low_percentile():
    flow = _flow({("senderA", "receiverA"): 0.9})

    result = net_flow_control(flow, null_pairs=["senderB>receiverA"])

    assert result["null_flow"] == pytest.approx(0.0)
    assert result["null_percentile"] < 0.5
    assert result["n_null"] == 1


def test_a_null_pair_carrying_flow_is_caught():
    """The failure this exists for: attention tracking who-is-near-whom rather than signalling."""
    flow = _flow({("senderA", "receiverA"): 0.9, ("senderB", "receiverA"): 0.8})

    result = net_flow_control(flow, null_pairs=["senderB>receiverA"])

    assert result["null_flow"] == pytest.approx(0.8)
    assert result["null_percentile"] > 0.8, "a violated null pair must stand out among all pairs"


def test_direction_does_not_rescue_a_null_pair():
    """A null pair is violated by flow either way, so it is scored on absolute value."""
    forward = net_flow_control(_flow({("senderB", "receiverA"): 0.8}), ["senderB>receiverA"])
    reverse = net_flow_control(_flow({("senderB", "receiverA"): -0.8}), ["senderB>receiverA"])

    assert forward["null_flow"] == reverse["null_flow"] == pytest.approx(0.8)


def test_signal_pairs_are_scored_signed_so_direction_is_a_claim():
    flow = _flow({("senderA", "receiverA"): 0.9})

    right_way = net_flow_control(flow, ["senderB>receiverA"], ["senderA>receiverA"])
    wrong_way = net_flow_control(flow, ["senderB>receiverA"], ["receiverA>senderA"])

    assert right_way["signal_flow"] == pytest.approx(0.9)
    assert wrong_way["signal_flow"] == pytest.approx(-0.9)


def test_separation_is_the_gate_quantity():
    """Genuine pairs strengthening while the null pair stays flat."""
    good = _flow({("senderA", "receiverA"): 0.9, ("senderB", "receiverA"): 0.05})
    confounded = _flow({("senderA", "receiverA"): 0.9, ("senderB", "receiverA"): 0.85})

    kwargs = dict(null_pairs=["senderB>receiverA"], signal_pairs=["senderA>receiverA"])
    assert net_flow_control(good, **kwargs)["separation"] > 0.8
    assert net_flow_control(confounded, **kwargs)["separation"] < 0.1


def test_percentile_is_scale_free():
    """Absolute magnitudes are not comparable across runs or normalisations; a rank is."""
    small = _flow({("senderA", "receiverA"): 0.9, ("senderB", "receiverA"): 0.1})
    scaled = small * 1000

    a = net_flow_control(small, ["senderB>receiverA"])
    b = net_flow_control(scaled, ["senderB>receiverA"])

    assert a["null_percentile"] == pytest.approx(b["null_percentile"])
    assert a["null_flow"] != pytest.approx(b["null_flow"])


# --------------------------------------------------------------------------- staying honest


def test_no_null_pairs_is_an_error_not_a_pass():
    """With none, null_flow is 0.0 by construction and would read as a passing control."""
    with pytest.raises(ValueError, match="at least one null pair"):
        net_flow_control(_flow({}), null_pairs=[])


def test_pairs_from_another_dataset_fail_with_what_is_available():
    with pytest.raises(KeyError, match="different dataset"):
        net_flow_control(_flow({}), null_pairs=["Tcell>Bcell"])


def test_n_null_is_reported_so_a_mean_is_never_read_blind():
    result = net_flow_control(_flow({}), null_pairs=["senderB>receiverA", "stroma>receiverB"])

    assert result["n_null"] == 2


def test_the_control_knows_no_cell_types():
    """Same code, entirely different annotation -- nothing dataset-specific is baked in."""
    types = ["tumour", "fibroblast", "Tcell"]
    flow = _flow({("tumour", "Tcell"): 0.7}, types=types)

    result = net_flow_control(flow, ["fibroblast>Tcell"], ["tumour>Tcell"])

    assert result["separation"] == pytest.approx(0.7)


# --------------------------------------------------------------------------- the report helper


def test_report_accepts_both_flow_dict_shapes():
    flow = _flow({("senderA", "receiverA"): 0.9})
    plain = {"healthy": flow}
    with_std = {"healthy": {"mean": flow, "std": flow * 0}}

    a = net_flow_control_report(plain, ["senderB>receiverA"])
    b = net_flow_control_report(with_std, ["senderB>receiverA"])

    assert a.index.tolist() == b.index.tolist() == ["healthy"]
    assert a.loc["healthy", "null_flow"] == b.loc["healthy", "null_flow"]


def test_report_has_one_row_per_group():
    flows = {
        "healthy": _flow({("senderA", "receiverA"): 0.9}),
        "diseased": _flow({("senderA", "receiverA"): 0.2, ("senderB", "receiverA"): 0.6}),
    }

    table = net_flow_control_report(flows, ["senderB>receiverA"], ["senderA>receiverA"])

    assert table.index.name == "group"
    assert sorted(table.index) == ["diseased", "healthy"]
    assert table.loc["healthy", "separation"] > table.loc["diseased", "separation"]


# --------------------------------------------------------------------------- orientation contract


def test_the_orientation_matches_the_real_flow_computation():
    """Pin the sign convention against `compute_hierarchical_net_flow`, which inverts it.

    That function treats information as the OPPOSITE of attention ("change sign to consider
    information as opposite of attention"), so A attending to B is reported as flow FROM B. If
    someone ever straightens that sign out, every `flow_signal_pairs` entry silently reverses
    meaning while every magnitude stays identical -- a failure no other test here would catch.
    """
    from anndata import AnnData

    from interscale.evaluation.net_streams import compute_hierarchical_net_flow

    n = 4
    adata = AnnData(X=np.abs(np.random.default_rng(0).normal(size=(n, 3))).astype(np.float32) + 0.1)
    adata.obs["cell_type"] = pd.Categorical(["A", "A", "B", "B"])
    adata.obs["win"] = pd.Categorical(["w1"] * n)
    adata.obs["smp"] = pd.Categorical(["s1"] * n)
    adata.obs_names = [f"c{i}" for i in range(n)]

    attention = np.zeros((n, n), dtype=np.float32)
    attention[0:2, 2:4] = 1.0  # A attends to B, and B does not attend back
    adata.obsm["_attn_matrix"] = attention

    flows = compute_hierarchical_net_flow(
        adata, window_key="win", sample_key="smp", condition_key=None, cell_type_col="cell_type"
    )
    flow = flows["all_samples"]["mean"]

    # A attends to B  =>  information flows B -> A  =>  positive at [B, A].
    assert flow.loc["B", "A"] > 0
    assert flow.loc["A", "B"] < 0

    # And the control reads it the same way round: "B>A" is the claim that holds.
    assert net_flow_control(flow, ["A>B"], ["B>A"])["signal_flow"] > 0
