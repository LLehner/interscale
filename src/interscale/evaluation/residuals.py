"""Model-free yardstick Y0: how much of each gene's spatial pattern its cells' labels explain.

``.claude/yardsticks_plan.md``, Y0. Moran's I of a cell-type marker is mostly the type's own clustering, so on real tissue a
gene that :func:`~interscale.evaluation.gene_scales.scale_sets` calls "global" can be a region or cell-type marker, which is
not communication. This module takes labels out of the expression before the scale sets are drawn, and splits each gene's
variance among the labels.

**Blocks** (:class:`Block`) are design matrices over the cells: :func:`label_block` (one-hot of a label: the cell's own type,
or a region / niche label) and :func:`composition_block` (the share of each label among a cell's neighbours, for any 0/1
neighbourhood matrix: the GCN's hop neighbourhood of :func:`~interscale.evaluation.gene_scales.hop_neighbourhoods` or
:func:`knn_neighbourhood`). Later yardsticks add other blocks over the same neighbourhoods (:func:`neighbourhood_mean`).

**Fits** (:func:`residualise`). Every fit has an intercept per graph (slide): the first stage, ``graph``, is each gene centred
within its graph, and every later stage adds one block (cumulative). ``fit="graph"`` fits the coefficients within each graph
(a label's mean per slide), ``"pooled"`` once over all cells, with the graph intercepts. Least squares with the minimum-norm
solution, since one-hot columns are collinear with the intercept. R² per gene is against the within-graph variance, pooled over
graphs. :func:`shapley_r2` splits the full R² among the blocks without an order (Shapley over the blocks: the LMG
decomposition; 2^B fits).

**Composition explains interaction too.** A gene a cell expresses because of the types around it is explained by composition,
and that is the signal InterScale looks for; near-composition also catches segmentation spillover. Read the composition stages
as a split of the variance; rebuild the scale sets on the residual of the cell's own label (or a region label) only.

:func:`residual_scale_sets` runs the scale sets on chosen stages, :func:`label_summary` lines a gene's statistics up before and
after a stage. Nothing names a dataset: labels, neighbourhoods and genes come from the caller. No global random generator is
touched (the scale sets use private ones).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from interscale.evaluation.gene_scales import NONE, ScaleSets, sample_neighbours, scale_sets

#: How the coefficients are fitted (the intercepts are per graph in both).
FITS = ("graph", "pooled")
#: The first stage of :func:`residualise`: each gene centred within its graph.
GRAPH = "graph"
#: The level an unlabelled cell (NaN) gets in :func:`label_block`.
UNLABELLED = "<unlabelled>"
#: A gene whose residual sum of squares in a graph is below this share of its within-graph sum is explained exactly there:
#: its residual is set to 0, so the scale sets read it as constant rather than as rounding noise.
EXACT = 1e-10
#: The most blocks :func:`shapley_r2` takes (2^B fits).
MAX_SHAPLEY_BLOCKS = 6


# ------------------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class Block:
    """A design matrix over the cells.

    Attributes
    ----------
    name
        Its name, the stage's name in :func:`residualise` and a column of :func:`shapley_r2`.
    design
        ``[N, P]`` covariates.
    kind
        ``label``, ``composition`` or another word a caller chooses.
    levels
        The names of the ``P`` columns.
    """

    name: str
    design: np.ndarray
    kind: str
    levels: tuple = ()

    def __post_init__(self):
        if np.ndim(self.design) != 2:
            raise ValueError(f"block {self.name!r}: the design must be [N, P], got shape {np.shape(self.design)}.")
        if not np.all(np.isfinite(self.design)):
            raise ValueError(f"block {self.name!r}: the design holds NaN or inf.")


def _codes(labels) -> tuple[np.ndarray, list[str]]:
    """Integer codes and level names of ``labels``; a missing label is its own level (:data:`UNLABELLED`)."""
    series = pd.Series(np.asarray(labels, dtype=object))
    series = series.where(series.notna(), UNLABELLED).astype(str)
    cat = pd.Categorical(series)
    return np.asarray(cat.codes, dtype=np.int64), [str(c) for c in cat.categories]


def label_block(labels, name: str = "label") -> Block:
    """One-hot of ``labels`` (``[N]``, any type; NaN is a level of its own)."""
    codes, levels = _codes(labels)
    design = np.zeros((len(codes), len(levels)))
    design[np.arange(len(codes)), codes] = 1.0
    return Block(name, design, "label", tuple(levels))


def knn_neighbourhood(pos, graph, k: int) -> sp.csr_matrix:
    """``[N, N]`` 0/1: each cell's ``k`` nearest other cells of its graph (fewer in a smaller graph); never itself."""
    graph = np.asarray(graph)
    nb = sample_neighbours(pos, graph, k_max=int(k), n_centres=None)
    rows = np.repeat(nb.centre, nb.k_max)
    cols = nb.neighbour.ravel()
    keep = cols >= 0
    n = len(graph)
    return sp.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))


def _check_neighbourhood(neighbourhood, graph: np.ndarray) -> sp.csr_matrix:
    a = sp.csr_matrix(neighbourhood, dtype=np.float64)
    n = len(graph)
    if a.shape != (n, n):
        raise ValueError(f"the neighbourhood is {a.shape}, the cells are {n}.")
    a = a.copy()
    a.setdiag(0.0)
    a.eliminate_zeros()
    rows, cols = a.nonzero()
    if np.any(graph[rows] != graph[cols]):
        raise ValueError("the neighbourhood joins cells of different graphs (slides); build it per slide.")
    if a.nnz and a.data.min() < 0:
        raise ValueError("the neighbourhood has negative weights.")
    return a


def neighbourhood_mean(neighbourhood, features, graph) -> np.ndarray:
    """``[N, P]`` the weighted mean of ``features`` over each cell's neighbours (the cell itself left out).

    A cell without a neighbour gets the mean of its graph's cells, so it adds no information of its own.
    """
    graph = np.asarray(graph)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(graph):
        raise ValueError(f"features must be [N, P] with one graph per row, got {features.shape} and {graph.shape}.")
    a = _check_neighbourhood(neighbourhood, graph)
    weight = np.asarray(a.sum(axis=1)).ravel()
    out = np.asarray(a @ features)
    has = weight > 0
    out[has] /= weight[has, None]
    for g in pd.unique(graph[~has]):
        rows = graph == g
        out[rows & ~has] = features[rows].mean(axis=0)
    return out


def composition_block(labels, neighbourhood, graph, name: str = "composition") -> Block:
    """The share of each level of ``labels`` among each cell's neighbours (rows sum to 1)."""
    one_hot = label_block(labels)
    return Block(name, neighbourhood_mean(neighbourhood, one_hot.design, graph), "composition", one_hot.levels)


# ------------------------------------------------------------------------------- fits


def _groups(graph: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(graph == g) for g in pd.unique(graph)]


def _centre(x: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    for rows in groups:
        out[rows] = x[rows] - x[rows].mean(axis=0)
    return out


def _residual(y: np.ndarray, x: np.ndarray, groups: list[np.ndarray], fit: str) -> tuple[np.ndarray, int]:
    """Residual of the centred ``y`` on the centred ``x`` and the number of fitted coefficients (the ranks)."""
    if x.shape[1] == 0:
        return y.copy(), 0
    out = np.empty_like(y)
    if fit == "pooled":
        coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        return y - x @ coef, int(rank)
    total = 0
    for rows in groups:
        coef, _, rank, _ = np.linalg.lstsq(x[rows], y[rows], rcond=None)
        out[rows] = y[rows] - x[rows] @ coef
        total += int(rank)
    return out, total


def _exact_zero(res: np.ndarray, centred: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Set a gene's residual to 0 within a graph where the fit explains it exactly (:data:`EXACT`)."""
    for rows in groups:
        ss_res = (res[rows] ** 2).sum(axis=0)
        ss_tot = (centred[rows] ** 2).sum(axis=0)
        exact = ss_res <= EXACT * ss_tot
        if exact.any():
            res[np.ix_(rows, np.flatnonzero(exact))] = 0.0
    return res


def _r2(res: np.ndarray, ss_tot: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - (res**2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)


def _check(values, graph, blocks, genes, fit):
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}.")
    graph = np.asarray(graph)
    values = np.asarray(values, dtype=np.float64)
    genes = [str(g) for g in genes]
    if values.shape != (len(graph), len(genes)):
        raise ValueError(f"values are {values.shape}, the cells {len(graph)} and the genes {len(genes)}.")
    names = [b.name for b in blocks]
    if len(set(names)) != len(names) or GRAPH in names:
        raise ValueError(f"block names must be distinct and not {GRAPH!r}, got {names}.")
    for b in blocks:
        if len(b.design) != len(graph):
            raise ValueError(f"block {b.name!r} has {len(b.design)} rows, the cells are {len(graph)}.")
    return values, graph, genes


@dataclass
class Residuals:
    """The expression with the blocks taken out, stage by stage.

    Attributes
    ----------
    stages
        ``graph`` (each gene centred within its graph), then each block's name: the stage after that block and every block
        before it.
    values
        Per stage, the ``[N, G]`` residual (float32; a gene a stage explains exactly within a graph is 0 there).
    r2
        ``[G, stages]`` R² of each stage against the within-graph variance, pooled over graphs (``graph`` is 0).
    r2_adjusted
        The same, adjusted for the fitted coefficients (``n_params``) and the graph intercepts.
    n_params
        Per stage, the coefficients fitted (ranks, summed over graphs under ``fit="graph"``).
    settings
        ``fit``, the cells, the graphs and each block's kind and levels.
    """

    stages: list
    values: dict
    r2: pd.DataFrame
    r2_adjusted: pd.DataFrame
    n_params: dict
    settings: dict

    def delta_r2(self) -> pd.DataFrame:
        """``[G, blocks]`` what each block adds to the R² of the stages before it (depends on the order)."""
        return self.r2.diff(axis=1).iloc[:, 1:]


def residualise(values, graph, blocks, genes, *, fit: str = "graph", keep=None) -> Residuals:
    """The residual of ``values`` after the graph intercepts and then each of ``blocks`` in turn (cumulative).

    Parameters
    ----------
    values
        ``[N, G]`` expression (e.g. ``layers['log1p_norm']``).
    graph
        ``[N]`` each cell's graph (slide).
    blocks
        :class:`Block` s, in the order they are added.
    genes
        The ``G`` gene names.
    fit
        :data:`FITS`.
    keep
        The stages whose residuals are kept (default: all); R² is computed for all.
    """
    values, graph, genes = _check(values, graph, blocks, genes, fit)
    groups = _groups(graph)
    centred = _centre(values, groups)
    ss_tot = (centred**2).sum(axis=0)
    stages = [GRAPH] + [b.name for b in blocks]
    keep = set(stages if keep is None else keep)
    unknown = keep - set(stages)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; the stages are {stages}.")
    out, r2, r2_adj, n_params = {}, {}, {}, {}
    dof_tot = len(graph) - len(groups)
    design = np.zeros((len(graph), 0))
    for stage, block in zip(stages, [None, *blocks], strict=True):
        if block is not None:
            design = np.hstack([design, _centre(np.asarray(block.design, dtype=np.float64), groups)])
        res, n = _residual(centred, design, groups, fit)
        res = _exact_zero(res, centred, groups)
        r2[stage] = _r2(res, ss_tot)
        dof_res = dof_tot - n
        r2_adj[stage] = 1.0 - (1.0 - r2[stage]) * dof_tot / dof_res if dof_res > 0 else np.full(len(genes), np.nan)
        n_params[stage] = n
        if stage in keep:
            out[stage] = res.astype(np.float32)
    index = pd.Index(genes, name="gene")
    settings = {
        "fit": fit,
        "n_cells": len(graph),
        "n_graphs": len(groups),
        "blocks": {b.name: {"kind": b.kind, "levels": list(b.levels)} for b in blocks},
    }
    return Residuals(
        stages=stages,
        values=out,
        r2=pd.DataFrame(r2, index=index)[stages],
        r2_adjusted=pd.DataFrame(r2_adj, index=index)[stages],
        n_params=n_params,
        settings=settings,
    )


def shapley_r2(values, graph, blocks, genes, *, fit: str = "graph") -> pd.DataFrame:
    """``[G, blocks + total]``: each block's Shapley share of the R² of all blocks together (the LMG decomposition).

    For every subset ``S`` of the blocks the R² of the graph intercepts plus ``S``; a block's share is its gain over each
    ``S`` without it, averaged with the Shapley weights ``|S|! (B - |S| - 1)! / B!``. The shares add up to ``total``, the R²
    of all blocks, and do not depend on the order of ``blocks``. ``2^B`` fits; at most :data:`MAX_SHAPLEY_BLOCKS` blocks.
    """
    values, graph, genes = _check(values, graph, blocks, genes, fit)
    n_blocks = len(blocks)
    if not 1 <= n_blocks <= MAX_SHAPLEY_BLOCKS:
        raise ValueError(f"shapley_r2 takes 1 to {MAX_SHAPLEY_BLOCKS} blocks, got {n_blocks}.")
    groups = _groups(graph)
    centred = _centre(values, groups)
    ss_tot = (centred**2).sum(axis=0)
    designs = [_centre(np.asarray(b.design, dtype=np.float64), groups) for b in blocks]
    r2 = {}
    for size in range(n_blocks + 1):
        for subset in itertools.combinations(range(n_blocks), size):
            x = np.hstack([designs[i] for i in subset]) if subset else np.zeros((len(graph), 0))
            res, _ = _residual(centred, x, groups, fit)
            r2[subset] = np.zeros(len(genes)) if not subset else _r2(_exact_zero(res, centred, groups), ss_tot)
    shares = shapley_from_subsets(r2, [b.name for b in blocks])
    shares["total"] = r2[tuple(range(n_blocks))]
    return pd.DataFrame(shares, index=pd.Index(genes, name="gene"))


def shapley_from_subsets(values: dict, names: list) -> dict:
    """Each player's Shapley share of a set function given on every subset.

    ``values`` maps every sorted tuple of player indices (``()`` included) to an array (one value per gene); ``names`` names
    the players. A player's share is its gain over each subset without it, weighted ``|S|! (B - |S| - 1)! / B!``; the shares add
    up to ``values[all] - values[()]``.
    """
    n = len(names)
    missing = [s for size in range(n + 1) for s in itertools.combinations(range(n), size) if s not in values]
    if missing:
        raise ValueError(f"the set function is missing subsets {missing[:3]}{' ...' if len(missing) > 3 else ''}.")
    shares = {}
    for b in range(n):
        others = [i for i in range(n) if i != b]
        total = np.zeros_like(np.asarray(values[()], dtype=np.float64))
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                with_b = tuple(sorted((*subset, b)))
                total = total + weight * (np.asarray(values[with_b]) - np.asarray(values[subset]))
        shares[names[b]] = total
    return shares


# ------------------------------------------------------------------------------- the scale sets on residuals

#: The columns of a stage's scale sets that :func:`residual_scale_sets` keeps.
SCALE_COLUMNS = ("i_local", "i_global", "z_local", "candidate", "drop", "scale_set")


def residual_scale_sets(
    residuals: Residuals, graph, pos, adjacency, genes, *, stages=None, **kwargs
) -> tuple[pd.DataFrame, dict]:
    """:func:`~interscale.evaluation.gene_scales.scale_sets` on the residual of each of ``stages`` (default: all kept).

    Returns the per-gene table with columns ``(stage, column)`` for :data:`SCALE_COLUMNS` and the :class:`ScaleSets` of
    each stage. ``kwargs`` go to ``scale_sets`` (``hops``, ``n``, ``rule``, ``unit_um``, ``seed`` ...); every stage gets
    the same, so their centres and shuffles are the same draws.
    """
    stages = list(residuals.values) if stages is None else list(stages)
    missing = [s for s in stages if s not in residuals.values]
    if missing:
        raise ValueError(f"no residuals kept for {missing}; kept: {list(residuals.values)}.")
    found: dict[str, ScaleSets] = {}
    parts = []
    for stage in stages:
        result = scale_sets(residuals.values[stage], graph, pos, adjacency, genes, **kwargs)
        found[stage] = result
        part = result.table[list(SCALE_COLUMNS)].copy()
        part.columns = pd.MultiIndex.from_product([[stage], part.columns])
        parts.append(part)
    return pd.concat(parts, axis=1), found


def label_summary(table: pd.DataFrame, residuals: Residuals, *, before: str = GRAPH, after: str) -> pd.DataFrame:
    """Per gene, its scale statistics before and after a stage, and what the stage explains.

    Columns: ``i_local_before`` / ``_after`` / ``kept_local`` (after over before), the same for ``i_global``,
    ``drop_before`` / ``_after``, ``set_before`` / ``_after``, ``moved`` (the set changed), and ``r2`` (the R² of ``after``
    over ``before``'s).
    """
    b, a = table[before], table[after]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(
            {
                "i_local_before": b["i_local"],
                "i_local_after": a["i_local"],
                "kept_local": a["i_local"] / b["i_local"],
                "i_global_before": b["i_global"],
                "i_global_after": a["i_global"],
                "kept_global": a["i_global"] / b["i_global"],
                "drop_before": b["drop"],
                "drop_after": a["drop"],
                "set_before": b["scale_set"].fillna(NONE),
                "set_after": a["scale_set"].fillna(NONE),
                "r2": residuals.r2[after] - residuals.r2[before],
            }
        )
    out["moved"] = out["set_before"] != out["set_after"]
    return out


def set_crosstab(summary: pd.DataFrame) -> pd.DataFrame:
    """Genes per (set before, set after) of :func:`label_summary`."""
    return pd.crosstab(summary["set_before"], summary["set_after"], margins=True)
