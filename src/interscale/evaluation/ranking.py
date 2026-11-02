"""R1 and R4: ranking genes by effect size, and how stable a ranking is across seeds and folds (``.claude/ranking_plan.md``).

**R1** (:func:`rank_by_effect`). A p-value ranks by sample size as much as by effect, so genes are ranked by an effect size (a
deletion's or the context test's ΔR², a unique R²). Two adjustments say whether an effect is more than a gene's expression level
buys: the **expression-matched z** (the effect against the genes of the same mean-expression bin: their median and 1.4826 MAD) and
the **control floor** (a quantile of named control genes' effects, e.g. CosMx's negative probes).

**R4** (:func:`rbo`, :func:`rank_stability`, :func:`robust_rank_aggregation`). Rank-biased overlap (Webber et al. 2010, the
extrapolated form): top-weighted agreement of two rankings, 1 identical, 0 disjoint; ``p`` sets the weight's depth (0.9: about the
first 10 ranks carry 86%). Per pair of runs also Spearman of the scores and Jaccard of the top-k; per gene the share of runs that put
it in the top-k. Robust rank aggregation (Kolde et al. 2012): a gene's normalised ranks over the runs, the smallest over k of the
Beta(k, m - k + 1) CDF at its k-th smallest rank (how unlikely so many good ranks are under uniform ranks), times m (Bonferroni).

Every function takes arrays or tables of scores by gene; nothing names a dataset.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr

#: MAD to standard deviation for normal data.
MAD_SCALE = 1.4826


# ------------------------------------------------------------------------------- R1


def _bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Quantile bins of ``values`` (fewer when values tie); NaN stays NaN."""
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=min(int(n_bins), max(1, int(values.notna().sum()))), labels=False, duplicates="drop")


def rank_by_effect(
    effect,
    *,
    mean_expression=None,
    controls=None,
    n_bins: int = 10,
    floor_quantile: float = 0.95,
) -> pd.DataFrame:
    """Genes ranked by an effect size (larger first), with the expression-matched z and the control floor.

    Parameters
    ----------
    effect
        A Series of effects by gene (NaN: not ranked).
    mean_expression
        A Series of each gene's mean expression (same genes), for the matched z; None: no z.
    controls
        Names of control genes (e.g. negative probes); their effects' ``floor_quantile`` is the floor; None: no floor.
    n_bins
        Mean-expression bins (quantiles).

    Returns
    -------
    pd.DataFrame
        By gene, best first: ``effect``, ``rank`` (1 = largest); with ``mean_expression``: ``bin``, ``bin_z``, ``rank_bin_z``;
        with ``controls``: ``floor`` and ``above_floor`` (controls are ranked too, marked by ``control``).
    """
    effect = pd.Series(effect, dtype=np.float64)
    out = pd.DataFrame({"effect": effect})
    out["rank"] = effect.rank(ascending=False, method="min")
    if mean_expression is not None:
        mean = pd.Series(mean_expression, dtype=np.float64).reindex(effect.index)
        bins = _bins(mean, n_bins)
        out["bin"] = bins
        z = pd.Series(np.nan, index=effect.index)
        for b in bins.dropna().unique():
            here = (bins == b) & effect.notna()
            values = effect[here]
            median = float(values.median())
            mad = MAD_SCALE * float((values - median).abs().median())
            if mad <= 0:  # a bin of (nearly) one value: its spread over all genes, not zero
                mad = MAD_SCALE * float((effect - effect.median()).abs().median()) or 1.0
            z[here] = (values - median) / mad
        out["bin_z"] = z
        out["rank_bin_z"] = z.rank(ascending=False, method="min")
    if controls is not None:
        controls = [c for c in controls if c in effect.index]
        if not controls:
            raise ValueError("none of the control genes is among the ranked genes.")
        floor = float(effect[controls].quantile(float(floor_quantile)))
        out["control"] = effect.index.isin(controls)
        out["floor"] = floor
        out["above_floor"] = effect > floor
    return out.sort_values("rank", kind="mergesort")


# ------------------------------------------------------------------------------- R4


def rbo(a, b, *, p: float = 0.9, depth: int | None = None) -> float:
    """Extrapolated rank-biased overlap of two rankings (best first), evaluated to ``depth`` (default: the shorter one)."""
    if not 0.0 < float(p) < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")
    a, b = list(a), list(b)
    k = min(len(a), len(b))
    if depth is not None:
        k = min(k, int(depth))
    if k == 0:
        return float("nan")
    seen_a, seen_b = set(), set()
    overlap, total = 0, 0.0
    for d in range(1, k + 1):
        x, y = a[d - 1], b[d - 1]
        if x == y:
            overlap += 1
        else:
            overlap += (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        total += overlap / d * float(p) ** d
    return float(overlap / k * float(p) ** k + (1.0 - float(p)) / float(p) * total)


def _order(scores: pd.Series, ascending: bool) -> list:
    """The genes of ``scores`` best first, NaN left out."""
    return list(scores.dropna().sort_values(ascending=ascending, kind="mergesort").index)


def rank_stability(
    scores: pd.DataFrame, *, top_k: int = 20, p: float = 0.9, ascending: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agreement of the rankings in ``scores`` (genes x runs; larger is better unless ``ascending``).

    Returns ``(pairs, genes)``: per pair of runs ``rbo``, ``spearman`` (the scores, genes both rank) and ``jaccard`` (top-k sets);
    per gene ``top_k_share`` (of the runs that rank it), ``mean_rank``, ``sd_rank``, ``n_runs``.
    """
    if scores.shape[1] < 2:
        raise ValueError("stability needs at least two runs (columns).")
    orders = {run: _order(scores[run], ascending) for run in scores.columns}
    rows = []
    for a, b in itertools.combinations(scores.columns, 2):
        both = scores[[a, b]].dropna()
        rho = spearmanr(both[a], both[b]).statistic if len(both) > 2 else np.nan
        top_a, top_b = set(orders[a][: int(top_k)]), set(orders[b][: int(top_k)])
        union = top_a | top_b
        rows.append(
            {
                "run_a": a,
                "run_b": b,
                "rbo": rbo(orders[a], orders[b], p=p),
                "spearman": float(rho),
                "jaccard": len(top_a & top_b) / len(union) if union else np.nan,
                "n_genes": len(both),
            }
        )
    ranks = scores.rank(ascending=ascending, method="min")
    in_top = ranks.le(int(top_k)) & scores.notna()
    genes = pd.DataFrame(
        {
            "top_k_share": in_top.sum(axis=1) / scores.notna().sum(axis=1).replace(0, np.nan),
            "mean_rank": ranks.mean(axis=1),
            "sd_rank": ranks.std(axis=1, ddof=1),
            "n_runs": scores.notna().sum(axis=1),
        }
    ).sort_values(["top_k_share", "mean_rank"], ascending=[False, True], kind="mergesort")
    return pd.DataFrame(rows), genes


def robust_rank_aggregation(scores: pd.DataFrame, *, ascending: bool = False) -> pd.DataFrame:
    """Kolde et al.'s RRA over the runs (columns) of ``scores``: per gene ``rho``, ``p`` (rho times the runs that rank it, at most
    1), ``mean_norm_rank`` and ``rank`` (1 = most consistently near the top). A gene a run does not rank is left out of that run.
    """
    n_genes = scores.notna().sum(axis=0)
    norm = scores.rank(ascending=ascending, method="average") / n_genes.replace(0, np.nan)
    rho = np.full(len(scores), np.nan)
    m_used = np.zeros(len(scores), dtype=int)
    for i, row in enumerate(norm.to_numpy()):
        r = np.sort(row[np.isfinite(row)])
        m = len(r)
        m_used[i] = m
        if m == 0:
            continue
        k = np.arange(1, m + 1)
        rho[i] = float(np.min(beta.cdf(r, k, m - k + 1)))
    out = pd.DataFrame(
        {
            "rho": rho,
            "p": np.minimum(1.0, rho * np.maximum(m_used, 1)),
            "mean_norm_rank": norm.mean(axis=1).to_numpy(),
            "n_runs": m_used,
        },
        index=scores.index,
    )
    out["rank"] = out["rho"].rank(method="min")
    return out.sort_values(["rho", "mean_norm_rank"], kind="mergesort")
