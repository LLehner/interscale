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

* **redundancy** (:func:`redundancy_table`): the share of one representation's variance a ridge reads from another, fitted on
  the training split and scored on the test split, per query group. ``global`` from ``local`` near 1 says the global tokens
  hold nothing the local ones do not (the copying of a dual model); near 0, nothing of the local tokens is in them. Beside
  it ``global`` from ``smoother`` and from ``pe`` say how much a spatial average of the visible expression, or the
  position alone, already gives.

Every column is z-scored over the split first (the sets come in unrelated units), and a set wider
than ``max_dims`` is reduced to its leading principal components over the split, so that hundreds
of structure-free columns -- a smoother's background genes -- do not drown what the set shares.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

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


#: ``(target, source)``: what :func:`redundancy_table` reads by default -- both ways between the model's two embeddings, and
#: the global one from the two model-free sets that already carry space.
REDUNDANCY_PAIRS = (("global", "local"), ("local", "global"), ("global", "smoother"), ("global", "pe"))


class _Standardiser:
    """Columns z-scored with the statistics of the data it was fitted on (constant ones zeroed), then reduced to the leading
    principal components of that data when wider than ``max_dims``: a fitted version of :func:`_prepared`.
    """

    def __init__(self, x: np.ndarray, max_dims: int = 0):
        x = np.asarray(x, dtype=np.float64)
        self.mean, sd = x.mean(0), x.std(0)
        self.scale = np.where(sd > 0, sd, 1.0)
        self.live = sd > 0
        self.components = None
        if max_dims > 0 and x.shape[1] > max_dims:
            z = np.where(self.live, (x - self.mean) / self.scale, 0.0)
            self.components = np.linalg.svd(z, full_matrices=False)[2][:max_dims]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = np.where(self.live, (np.asarray(x, dtype=np.float64) - self.mean) / self.scale, 0.0)
        return z if self.components is None else z @ self.components.T


def redundancy_table(
    train: ProbeBatchFeatures,
    test: ProbeBatchFeatures,
    *,
    depth_bins_um,
    pairs: tuple[tuple[str, str], ...] = REDUNDANCY_PAIRS,
    max_dims: int = 64,
    alphas=(0.1, 1.0, 10.0, 100.0, 1e3),
    min_cells: int = 30,
) -> pd.DataFrame:
    """How much of one representation a ridge reads from another, per query group: fitted on ``train``, scored on ``test``.

    For each ``(target, source)`` of ``pairs`` present in both splits and each query group (``visible``, ``hidden``, and the
    depth bins of the wholly hidden cells, as :func:`representation_table`), the source is z-scored and, when wider than
    ``max_dims``, reduced to its leading principal components (on the group's training cells); a ridge with the penalty
    chosen by leave-one-out (:class:`~sklearn.linear_model.RidgeCV`) maps it to the target, z-scored the same way. The
    ``r2`` is ``1 - SSE / SST`` pooled over the target's columns on the group's test cells, so every column weighs alike:
    1 when the source says all of the target, 0 when it says nothing a constant would not, negative when it misleads. A
    group needs ``min_cells`` cells in both splits. Fitting on the group itself asks the question where it matters: whether
    the global tokens of a *hidden* cell are a function of its local tokens, not whether a map learnt on visible cells
    carries over.

    Needs ``depth_um`` in both (collect with ``spatial_unit_um``).

    Returns
    -------
    pandas.DataFrame
        ``query, target, source, r2, n_train, n_test``.
    """
    if train.depth_um is None or test.depth_um is None:
        raise ValueError("redundancy_table groups cells by depth; collect the features with spatial_unit_um.")

    def selections(features):
        depth = np.asarray(features.depth_um, dtype=float)
        hidden = depth > 0
        group = query_groups(depth, hidden, depth_bins_um)
        found = {VISIBLE: group == VISIBLE, "hidden": hidden}
        found.update({label: group == label for label in bin_labels(depth_bins_um)})
        return found

    rows_train, rows_test = selections(train), selections(test)
    columns = ["query", "target", "source", "r2", "n_train", "n_test"]
    rows = []
    for target, source in pairs:
        if not all(name in features.embeddings for features in (train, test) for name in (target, source)):
            continue
        for label in rows_train:
            fit, score = rows_train[label], rows_test[label]
            if fit.sum() < min_cells or score.sum() < min_cells:
                continue
            x_fit = train.embeddings[source][fit]
            y_fit = train.embeddings[target][fit]
            read_source, read_target = _Standardiser(x_fit, max_dims), _Standardiser(y_fit)
            model = RidgeCV(alphas=[float(a) for a in alphas]).fit(read_source(x_fit), read_target(y_fit))
            truth = read_target(test.embeddings[target][score])
            error = ((model.predict(read_source(test.embeddings[source][score])) - truth) ** 2).sum()
            spread = ((truth - truth.mean(0)) ** 2).sum()
            rows.append(
                {
                    "query": label,
                    "target": target,
                    "source": source,
                    "r2": float(1.0 - error / spread) if spread > 0 else float("nan"),
                    "n_train": int(fit.sum()),
                    "n_test": int(score.sum()),
                }
            )
    return pd.DataFrame(rows, columns=columns)
