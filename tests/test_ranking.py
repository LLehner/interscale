"""R1 and R4 (ranking_plan.md): ``evaluation.ranking``, and R3's ``dataset.drop_genes``.

Pinned here:

* **R1** -- ranks by effect, larger first; the expression-matched z against the gene's own bin (an abundant gene with a large effect
  is ordinary among abundant genes); the control floor and who clears it;
* **R4** -- RBO by hand, 1 for identical rankings, 0 for disjoint ones, top-weighted; the stability tables; RRA's rho by hand, a gene
  near the top in every run first, a gene at random not significant;
* **R3** -- ``drop_genes`` removes the selected genes (names, ``column=value``), records them, refuses a target gene and an unknown
  selector, leaves the data alone when empty; the checkpoint name carries the set (and does not without one).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from interscale.evaluation.ranking import rank_by_effect, rank_stability, rbo, robust_rank_aggregation

# ------------------------------------------------------------------------------- R1


def test_rank_by_effect_with_matched_bins_and_a_floor():
    genes = [f"g{k}" for k in range(40)]
    mean = pd.Series(np.repeat([1.0, 10.0], 20), index=genes)
    effect = pd.Series(np.r_[np.linspace(0.0, 0.1, 20), np.linspace(0.5, 0.6, 20)], index=genes)
    effect["g5"] = 0.4  # a low-expression gene with a large effect for its bin
    t = rank_by_effect(effect, mean_expression=mean, n_bins=2)
    assert t.index[0] == "g39" and t.loc["g39", "rank"] == 1
    assert t.loc["g5", "rank"] > 20  # by effect: behind every abundant gene
    assert t.loc["g5", "rank_bin_z"] == 1  # within its bin: far above its peers
    low = effect[mean == 1.0]
    median = low.median()
    assert t.loc["g5", "bin_z"] == pytest.approx((0.4 - median) / (1.4826 * (low - median).abs().median()))
    assert set(t["bin"].dropna().unique()) == {0, 1}
    controls = ["g0", "g1", "g2"]
    f = rank_by_effect(effect, controls=controls, floor_quantile=1.0)
    assert f["floor"].iloc[0] == pytest.approx(effect[controls].max())
    assert not f.loc["g2", "above_floor"] and f.loc["g10", "above_floor"] and f.loc["g0", "control"]
    with pytest.raises(ValueError, match="control"):
        rank_by_effect(effect, controls=["nope"])


# ------------------------------------------------------------------------------- R4


def test_rbo_by_hand_and_its_limits():
    assert rbo([1, 2, 3], [1, 3, 2], p=0.9) == pytest.approx(0.729 + 0.1 / 0.9 * (0.9 + 0.405 + 0.729))
    assert rbo(list(range(50)), list(range(50))) == pytest.approx(1.0)
    assert rbo(list(range(10)), list(range(10, 20))) == 0.0
    a = list(range(100))
    top_swapped = [1, 0] + a[2:]
    tail_swapped = a[:98] + [99, 98]
    assert rbo(a, tail_swapped) > rbo(a, top_swapped)  # top-weighted
    with pytest.raises(ValueError, match="p must"):
        rbo(a, a, p=1.0)


def test_rank_stability_tables():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=60), index=[f"g{k}" for k in range(60)])
    scores = pd.DataFrame(
        {"s42": base, "s43": base + 0.01 * rng.normal(size=60), "s44": rng.normal(size=60)}, index=base.index
    )
    pairs, genes = rank_stability(scores, top_k=10)
    same = pairs[(pairs.run_a == "s42") & (pairs.run_b == "s43")].iloc[0]
    other = pairs[(pairs.run_a == "s42") & (pairs.run_b == "s44")].iloc[0]
    assert same.rbo > 0.8 and same.spearman > 0.99 and same.jaccard > 0.8
    assert other.rbo < same.rbo and abs(other.spearman) < 0.5
    top = base.sort_values(ascending=False).index[0]
    assert genes.loc[top, "top_k_share"] >= 2 / 3 and genes["n_runs"].eq(3).all()
    with pytest.raises(ValueError, match="two runs"):
        rank_stability(scores[["s42"]])
    small = pd.DataFrame({"a": [5.0, 4.0, 3.0, 2.0, 1.0], "b": [5.0, 1.0, 4.0, 3.0, 2.0]}, index=list("vwxyz"))
    p2, g2 = rank_stability(small, top_k=3)
    assert p2.loc[0, "jaccard"] == pytest.approx(2 / 4)  # {v, w, x} vs {v, x, y}
    assert (
        g2.loc["x", "top_k_share"] == 1.0 and g2.loc["w", "top_k_share"] == 0.5
    )  # x is 3rd in a: top-3 includes rank 3


def test_rra_by_hand_and_on_planted_ranks():
    scores = pd.DataFrame({"a": [3.0, 2.0, 1.0], "b": [3.0, 1.0, 2.0], "c": [1.0, 3.0, 2.0]}, index=["x", "y", "z"])
    out = robust_rank_aggregation(scores)
    r = np.sort(np.array([1, 1, 3]) / 3)  # x: ranks 1, 1, 3 of 3
    want = min(beta.cdf(r[k], k + 1, 3 - k) for k in range(3))
    assert out.loc["x", "rho"] == pytest.approx(want) and out.index[0] == "x"
    assert out.loc["x", "p"] == pytest.approx(min(1.0, 3 * want))
    rng = np.random.default_rng(1)
    many = pd.DataFrame(rng.normal(size=(200, 5)), index=[f"g{k}" for k in range(200)])
    many.loc["g0"] = 10.0  # first in every run
    agg = robust_rank_aggregation(many)
    assert agg.index[0] == "g0" and agg.loc["g0", "p"] < 1e-6
    assert agg["p"].iloc[100:].min() > 0.05
    many.loc["g1", 0] = np.nan  # a run that does not rank a gene
    assert robust_rank_aggregation(many).loc["g1", "n_runs"] == 4


# ------------------------------------------------------------------------------- R3


def _adata():
    from anndata import AnnData

    a = AnnData(X=np.arange(20, dtype=np.float32).reshape(4, 5))
    a.var_names = ["A", "B", "C", "D", "E"]
    a.var["role"] = ["sender", "sender", "noise", "response", "noise"]
    return a


def _cfg(**dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    for key, value in dataset.items():
        cfg.dataset[key] = value
    return cfg


def test_drop_genes_removes_records_and_refuses_targets():
    from interscale.tl import drop_genes

    a = _adata()
    assert drop_genes(a, _cfg()) is a
    out = drop_genes(a, _cfg(drop_genes=["A", "role=noise"]))
    assert list(out.var_names) == ["B", "D"] and out.uns["dropped_genes"] == ["A", "C", "E"]
    assert np.array_equal(out.X, a.X[:, [1, 3]]) and a.n_vars == 5
    with pytest.raises(ValueError, match="target genes"):
        drop_genes(a, _cfg(drop_genes=["D"], mask_genes=["role=response"]))
    with pytest.raises(ValueError, match="matches no gene"):
        drop_genes(a, _cfg(drop_genes=["Z"]))


def test_the_checkpoint_name_carries_the_dropped_set():
    from interscale.tl.utils import get_model_filename_prefix

    plain = get_model_filename_prefix(_cfg(), True, False)
    one = get_model_filename_prefix(_cfg(drop_genes=["A", "B"]), True, False)
    same = get_model_filename_prefix(_cfg(drop_genes=["B", "A"]), True, False)
    other = get_model_filename_prefix(_cfg(drop_genes=["A", "C"]), True, False)
    assert "drop" not in plain and "drop2-" in one and one == same and one != other
