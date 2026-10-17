"""How alike two representations of the same cells are, and how many directions each uses.

Read on the features :func:`~interscale.evaluation.online_probes.collect_features` gathered from a
masked pass, per query group -- ``"visible"``, and the depth bins of the wholly hidden cells:

* **linear CKA** (Kornblith et al., 2019) between two feature sets: 1 when one is a rotation and
  scaling of the other, near 0 when they share no structure. ``global`` against ``smoother`` near 1
  says the global tokens are a spatial smoother of the visible expression; ``global`` against
  ``local`` on visible cells near 1 says the global component copies the local one.
* **participation ratio** of each set, ``(sum l)^2 / sum l^2`` over the eigenvalues ``l`` of its
  correlation matrix: the number of directions the cells spread along, 1 when every cell sits on
  one line. Hidden cells whose tokens collapse onto the mask token and its position read low.

Every column is z-scored over the split first (the sets come in unrelated units), and a set wider
than ``max_dims`` is reduced to its leading principal components over the split, so that hundreds
of structure-free columns -- a smoother's background genes -- do not drown what the set shares.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from interscale.evaluation._spatial_weights import VISIBLE, bin_labels, query_groups
from interscale.evaluation.online_probes import ProbeBatchFeatures

#: Feature sets compared by default: the model's own and the model-free ones a hidden cell can have.
BASE_SETS = ("local", "global", "pe", "smoother", "neighbours")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA of two ``[N, *]`` representations of the same cells; NaN when either is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0)
    y = y - y.mean(0)
    cross = np.linalg.norm(y.T @ x) ** 2
    norm = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(cross / norm) if norm > 0 else float("nan")


def participation_ratio(x: np.ndarray) -> float:
    """``(sum l)^2 / sum l^2`` over the covariance eigenvalues of ``[N, D]``; NaN when constant."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean(0)
    eig = np.clip(np.linalg.eigvalsh(x.T @ x / (len(x) - 1)), 0.0, None)
    total = eig.sum()
    return float(total**2 / (eig**2).sum()) if total > 0 else float("nan")


def _prepared(x: np.ndarray, max_dims: int) -> np.ndarray:
    """Columns z-scored (constant ones zeroed), then the leading ``max_dims`` principal components."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    x = np.where(sd > 0, (x - x.mean(0)) / np.where(sd > 0, sd, 1.0), 0.0)
    if max_dims > 0 and x.shape[1] > max_dims:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        x = x @ vt[:max_dims].T
    return x


def representation_table(
    features: ProbeBatchFeatures,
    *,
    depth_bins_um,
    sets: tuple[str, ...] = BASE_SETS,
    max_dims: int = 64,
    min_cells: int = 10,
) -> pd.DataFrame:
    """CKA between every pair of ``sets`` present, and each one's participation ratio, per query group.

    Needs ``features.depth_um`` (collect with ``spatial_unit_um``). Groups with fewer than
    ``min_cells`` cells are left out.

    Returns
    -------
    pandas.DataFrame
        ``query, metric, a, b, value, n_cells``: ``metric`` is ``cka`` (``a`` against ``b``) or
        ``participation_ratio`` (of ``a``; ``b`` empty), for ``"visible"``, ``"hidden"`` (every
        wholly hidden cell) and each depth bin.
    """
    if features.depth_um is None:
        raise ValueError("representation_table groups cells by depth; collect the features with spatial_unit_um.")
    present = [name for name in sets if name in features.embeddings]
    prepared = {name: _prepared(features.embeddings[name], max_dims) for name in present}
    depth = np.asarray(features.depth_um, dtype=float)
    hidden = depth > 0
    group = query_groups(depth, hidden, depth_bins_um)
    selections = [(VISIBLE, group == VISIBLE), ("hidden", hidden)]
    selections += [(label, group == label) for label in bin_labels(depth_bins_um)]
    rows = []
    for label, pick in selections:
        n = int(pick.sum())
        if n < min_cells:
            continue
        for a in present:
            rows.append(
                {
                    "query": label,
                    "metric": "participation_ratio",
                    "a": a,
                    "b": "",
                    "value": participation_ratio(prepared[a][pick]),
                    "n_cells": n,
                }
            )
        for a, b in itertools.combinations(present, 2):
            cka = linear_cka(prepared[a][pick], prepared[b][pick])
            rows.append({"query": label, "metric": "cka", "a": a, "b": b, "value": cka, "n_cells": n})
    return pd.DataFrame(rows, columns=["query", "metric", "a", "b", "value", "n_cells"])
