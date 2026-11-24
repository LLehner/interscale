"""Sender fields against hidden genes (2.3, ``.claude/comparison.md``, 2026-11-24): the readout of the field-correlation term, for any model.

A receiver's sender field of gene ``s`` is ``f_s(i) = sum_j a_ij x_js``: what the cells it reads express, weighted by its attention (the last
layer's head mean, :func:`attention_rows.attention_rows`, under the graph's own masks). :func:`field_pairs` hides each gene fold in every cell
(as the channel-trained models were trained), and correlates every visible gene's field with every hidden gene over the cells, centred per graph
(or per graph and ``groups``, e.g. a region), sums pooled over graphs. ``r[s, t]`` ranks sender -> target pairs; the same with a Gaussian kernel
of distance in place of the attention (``kernel_um``) is the model-free bar: what fields of that width give without any learning.
"""

from __future__ import annotations

import numpy as np

from interscale.evaluation.attention_rows import attention_rows
from interscale.evaluation.gene_ranking import hide_genes


def _centre(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for g in np.unique(groups):
        rows = groups == g
        out[rows] -= out[rows].mean(axis=0)
    return out


def _kernel_rows(pos: np.ndarray, sigma: float) -> np.ndarray:
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    k = np.exp(-d2 / (2.0 * sigma**2))
    np.fill_diagonal(k, 0.0)
    return k / np.maximum(k.sum(axis=1, keepdims=True), 1e-12)


def field_pairs(module, graphs, n_genes: int, folds, *, groups=None, kernel_um: float | None = None, unit_um: float = 1.0, model: bool = True) -> dict:
    """``{"model": r [G, G], "kernel": r [G, G] or None, "count": [G, G]}``: Pearson ``r`` of sender field ``s`` (rows) with hidden target ``t``
    (columns) over all cells of ``graphs``, NaN where a pair was never read (a sender in the target's fold).

    ``folds``: gene index arrays; each is hidden in every cell of every graph in turn (``hide_genes``). ``groups``: one array per graph to centre
    within (default: per graph). ``kernel_um`` (in µm; positions times ``unit_um``): also the model-free fields. ``model`` False: the kernel
    only (no model pass)."""
    sums = {name: np.zeros((n_genes, n_genes)) for name in ("model_xy", "model_xx", "kernel_xy", "kernel_xx")}
    yy = np.zeros((n_genes, n_genes))
    count = np.zeros((n_genes, n_genes))
    groups = [None] * len(graphs) if groups is None else list(groups)
    for fold in folds:
        fold = np.asarray(fold, dtype=np.int64)
        senders = np.setdiff1d(np.arange(n_genes), fold)
        for graph, grp in zip(hide_genes(graphs, fold), groups, strict=True):
            x = graph.x.double().cpu().numpy()
            n = x.shape[0]
            grp = np.zeros(n, dtype=np.int64) if grp is None else np.asarray(grp)
            visible = x[:, senders]
            y = _centre(x[:, fold], grp)
            fields = {}
            if model:
                rows = np.nan_to_num(attention_rows(module, graph, np.arange(n), layer=-1, eval_step=False).astype(np.float64))
                fields["model"] = rows @ visible
            if kernel_um:
                pos = graph.pos[:, :2].double().cpu().numpy() * float(unit_um)
                fields["kernel"] = _kernel_rows(pos, float(kernel_um)) @ visible
            for name, f in fields.items():
                fc = _centre(f, grp)
                sums[f"{name}_xy"][np.ix_(senders, fold)] += fc.T @ y
                sums[f"{name}_xx"][np.ix_(senders, fold)] += (fc**2).sum(axis=0)[:, None]
            yy[np.ix_(senders, fold)] += (y**2).sum(axis=0)[None, :]
            count[np.ix_(senders, fold)] += n
    out = {"count": count}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in ("model", "kernel"):
            r = sums[f"{name}_xy"] / np.sqrt(sums[f"{name}_xx"] * yy)
            r[count == 0] = np.nan
            out[name] = r if (name == "model" and model) or (name == "kernel" and kernel_um) else None
    return out


def pair_table(r: np.ndarray, genes) -> "pd.DataFrame":  # noqa: F821
    """Long table ``sender, target, r, score`` (``score = |r|``), the diagonal and unread pairs left out, best first."""
    import pandas as pd

    genes = [str(g) for g in genes]
    s, t = np.nonzero(np.isfinite(r))
    keep = s != t
    s, t = s[keep], t[keep]
    table = pd.DataFrame({"sender": [genes[i] for i in s], "target": [genes[j] for j in t], "r": r[s, t]})
    table["score"] = table["r"].abs()
    return table.sort_values("score", ascending=False, ignore_index=True)


__all__ = ["field_pairs", "pair_table"]
