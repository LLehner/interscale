"""Linear probes on embeddings from masked inference passes, for a trained model after the fact.

A probe that predicts a cell's expression from an embedding of that same cell's expression mostly
inverts the encoder: on synth_data_0 a structure-free gene probed at R2 0.33 that way and ~0.02
once restricted to hidden cells (see ``online_probes``). So every probe here reads embeddings from
a pass in which the values it predicts were hidden from the model, drawn with a private seed:

* ``"entries"`` -- each (cell, gene) entry hidden independently, as gene masking trains. The gene
  vector is scored on each cell's hidden entries. Whatever stayed visible -- a receiver's receptor
  while its response is hidden -- is fair input, as it was in training.
* ``"cells"`` -- whole cells hidden. A hidden cell's embedding knows only its context, so the gene
  vector, labels and covariates read from it measure what the context says about the cell.

Regression is scored by **cosine similarity** between predicted and true vectors, per cell: over the
whole gene vector (its hidden entries), and over a vector of covariates. Two versions, side by
side:

* ``cosine_raw`` -- of the vectors as they are. Expression vectors share a large mean profile, so
  even predicting the training mean scores high (0.91 on synth_spot's log1p data); the ``mean``
  row gives that floor.
* ``cosine_centred`` -- of the deviations from the training mean. Predicting the mean scores 0,
  so this is the share of a cell's departure from the average cell the probe got right.

Per gene group (``gene_groups``, e.g. the planted programmes) the centred cosine is pooled over
all hidden entries of the group's genes, since a group of a few genes leaves only one or two
hidden entries per cell. Groups without structure -- background genes -- are the negative control:
they should sit at 0 for every representation.

Everything is dataset-agnostic: genes, groups, labels and covariates are arguments. The feature
sets are whatever :func:`~interscale.evaluation.online_probes.collect_features` gathered --
``local``, ``global``, ``pe`` and the input baselines -- plus ``local+global``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from interscale.evaluation.online_probes import ProbeBatchFeatures

MASK_KINDS = ("entries", "cells")


def draw_masks(data_list, kind: str, rate: float, seed: int) -> None:
    """Hide entries or whole cells of every graph in ``data_list``, in place, from a private generator.

    Writes ``data.gene_mask`` ``[N, G]`` and ``data.mask`` ``[N]`` (its row-wise OR), the attributes
    the gene-masking pipeline reads. ``"cells"`` hides every gene of a hidden cell. The global
    torch generator is not touched.
    """
    if kind not in MASK_KINDS:
        raise ValueError(f"kind must be one of {MASK_KINDS}, got {kind!r}.")
    gen = torch.Generator().manual_seed(int(seed))
    for data in data_list:
        n, g = data.x.shape
        if kind == "entries":
            gene_mask = torch.rand(n, g, generator=gen) < rate
        else:
            gene_mask = (torch.rand(n, generator=gen) < rate)[:, None].expand(n, g).clone()
        data.gene_mask = gene_mask
        data.mask = gene_mask.any(dim=1)


def feature_sets(features: ProbeBatchFeatures) -> dict[str, np.ndarray]:
    """The collected representations, plus ``local+global`` when both are there."""
    sets = dict(features.embeddings)
    if "local" in sets and "global" in sets:
        sets["local+global"] = np.concatenate([sets["local"], sets["global"]], axis=1)
    return sets


def _ridge_predictions(x_train, y_train, rows_train, x_eval, rows_eval, alpha) -> np.ndarray:
    """Per gene: fit on the train cells where it is hidden, predict the eval cells where it is hidden.

    ``y_train``, ``rows_train`` ``[N_train, G]``; returns ``[N_eval, G]``, NaN where not hidden. When
    every gene is hidden in the same cells (``"cells"`` masks) this is one multi-output fit.
    """
    scaler = StandardScaler().fit(x_train)
    xt, xe = scaler.transform(x_train), scaler.transform(x_eval)
    pred = np.full(rows_eval.shape, np.nan)
    same_train = (rows_train == rows_train[:, :1]).all()
    same_eval = (rows_eval == rows_eval[:, :1]).all()
    if same_train and same_eval:
        tr, ev = rows_train[:, 0], rows_eval[:, 0]
        if tr.any() and ev.any():
            # Reshaped: Ridge returns a flat array when y has a single column.
            pred[ev] = Ridge(alpha=alpha).fit(xt[tr], y_train[tr]).predict(xe[ev]).reshape(int(ev.sum()), -1)
        return pred
    for g in range(rows_train.shape[1]):
        tr, ev = rows_train[:, g], rows_eval[:, g]
        if tr.sum() >= 2 and ev.any():
            pred[ev, g] = Ridge(alpha=alpha).fit(xt[tr], y_train[tr, g]).predict(xe[ev])
    return pred


def _cosine_rows(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Cosine of each row of ``a`` and ``b`` over the entries ``keep``; NaN where a norm is zero."""
    a, b = np.where(keep, a, 0.0), np.where(keep, b, 0.0)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, (a * b).sum(1) / norm, np.nan)


def _pooled_cosine(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float:
    """Cosine of ``a`` and ``b`` taken as one long vector over the entries ``keep``."""
    a, b = a[keep], b[keep]
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 0 else float("nan")


def probe_expression(
    train: ProbeBatchFeatures,
    test: ProbeBatchFeatures,
    genes: list[str],
    *,
    gene_groups: dict[str, list[str]] | None = None,
    alpha: float = 1.0,
    min_entries: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the whole gene vector from each representation; score the hidden entries by cosine.

    Parameters
    ----------
    train, test
        Collected with a masked pass and ``gene_index`` covering ``genes``; probes are fit on
        ``train`` and scored on ``test``.
    genes
        The gene vector, in ``var`` order.
    gene_groups
        ``{group: [genes]}`` for the per-group breakdown, e.g. ``var['program']``.
    alpha
        Ridge penalty, on standardised features.
    min_entries
        Cells with fewer hidden entries are left out of the per-cell mean.

    Returns
    -------
    (summary, per_gene)
        ``summary``: one row per ``(feature_set, scope, metric)`` -- scope ``all`` holds the per-cell
        means (``cosine_raw``, ``cosine_centred``), each group its pooled ``cosine_centred``.
        ``per_gene``: pooled ``cosine_centred`` of every gene, rows genes, columns feature sets.
    """
    y_tr = np.stack([train.continuous[g] for g in genes], axis=1).astype(np.float64)
    y_te = np.stack([test.continuous[g] for g in genes], axis=1).astype(np.float64)
    m_tr = np.stack([train.gene_masked[g] for g in genes], axis=1).astype(bool)
    m_te = np.stack([test.gene_masked[g] for g in genes], axis=1).astype(bool)
    # The training mean of each gene over its hidden entries: the floor and the centring point.
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(m_tr, y_tr, np.nan), axis=0)
    mean = np.nan_to_num(mean)
    enough = m_te.sum(1) >= min_entries
    position = {g: i for i, g in enumerate(genes)}
    groups = {k: [position[g] for g in v if g in position] for k, v in (gene_groups or {}).items()}

    sets = feature_sets(train)
    test_sets = feature_sets(test)
    rows, per_gene = [], {}
    predictions = {name: _ridge_predictions(x, y_tr, m_tr, test_sets[name], m_te, alpha) for name, x in sets.items()}
    predictions["mean"] = np.where(m_te, mean, np.nan)

    for name, pred in predictions.items():
        pred = np.nan_to_num(pred)
        raw = _cosine_rows(pred, y_te, m_te)[enough]
        centred = _cosine_rows(pred - mean, y_te - mean, m_te)[enough]
        rows.append(
            {
                "feature_set": name,
                "scope": "all",
                "metric": "cosine_raw",
                "value": float(np.nanmean(raw)),
                "n": int(enough.sum()),
            }
        )
        rows.append(
            {
                "feature_set": name,
                "scope": "all",
                "metric": "cosine_centred",
                "value": float(np.nanmean(centred)),
                "n": int(enough.sum()),
            }
        )
        for group, cols in groups.items():
            keep = np.zeros_like(m_te)
            keep[:, cols] = m_te[:, cols]
            rows.append(
                {
                    "feature_set": name,
                    "scope": group,
                    "metric": "cosine_centred",
                    "value": _pooled_cosine(pred - mean, y_te - mean, keep),
                    "n": int(keep.sum()),
                }
            )
        per_gene[name] = [
            _pooled_cosine((pred - mean)[:, [j]], (y_te - mean)[:, [j]], m_te[:, [j]]) for j in range(len(genes))
        ]
    return pd.DataFrame(rows), pd.DataFrame(per_gene, index=pd.Index(genes, name="gene"))


def probe_labels(
    train: ProbeBatchFeatures,
    test: ProbeBatchFeatures,
    targets: list[str],
    *,
    hidden_only: bool = True,
    max_iter: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Read each categorical target out of each representation with a balanced logistic regression.

    With ``hidden_only`` (collected from a ``"cells"`` pass) only hidden cells are used, so the label
    has to come from the context: a label the cell's own expression encodes -- a response gene that
    marks it -- is otherwise read straight off its input.

    Returns
    -------
    pandas.DataFrame
        One row per ``(target, feature_set, metric)``: ``balanced_accuracy`` and ``f1_macro``, with
        a ``majority`` row as the floor.
    """
    tr = train.cell_masked.astype(bool) if hidden_only else np.ones(train.n_cells(), bool)
    te = test.cell_masked.astype(bool) if hidden_only else np.ones(test.n_cells(), bool)
    sets, test_sets = feature_sets(train), feature_sets(test)
    rows = []
    for target in targets:
        y_tr, y_te = train.categorical[target][tr], test.categorical[target][te]
        classes = np.unique(np.concatenate([y_tr, y_te]))
        majority = np.full_like(y_te, np.bincount(y_tr).argmax())
        scored = {"majority": majority}
        if len(np.unique(y_tr)) > 1:
            for name, x in sets.items():
                scaler = StandardScaler().fit(x[tr])
                clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
                clf.fit(scaler.transform(x[tr]), y_tr)
                scored[name] = clf.predict(scaler.transform(test_sets[name][te]))
        for name, pred in scored.items():
            rows.append(
                {
                    "target": target,
                    "feature_set": name,
                    "metric": "balanced_accuracy",
                    "value": balanced_accuracy_score(y_te, pred),
                    "n": int(te.sum()),
                }
            )
            rows.append(
                {
                    "target": target,
                    "feature_set": name,
                    "metric": "f1_macro",
                    "value": f1_score(y_te, pred, average="macro", labels=classes, zero_division=0),
                    "n": int(te.sum()),
                }
            )
    return pd.DataFrame(rows)


def probe_covariates(
    train: ProbeBatchFeatures,
    test: ProbeBatchFeatures,
    names: list[str],
    *,
    hidden_only: bool = True,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Predict a vector of numeric covariates (e.g. distances to every source); score by cosine.

    Each covariate is z-scored with the training statistics first, so every one weighs the same in
    the cosine and the training mean sits at 0: ``cosine_centred`` is then the cosine of the
    predicted and true z-vectors, per cell, averaged. ``mean`` is the floor (undefined, NaN).

    Returns
    -------
    pandas.DataFrame
        One row per ``feature_set`` with the mean per-cell ``cosine_centred``.
    """
    tr = train.cell_masked.astype(bool) if hidden_only else np.ones(train.n_cells(), bool)
    te = test.cell_masked.astype(bool) if hidden_only else np.ones(test.n_cells(), bool)
    y_tr = np.stack([train.continuous[c] for c in names], axis=1)[tr].astype(np.float64)
    y_te = np.stack([test.continuous[c] for c in names], axis=1)[te].astype(np.float64)
    mu, sd = y_tr.mean(0), y_tr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z_tr, z_te = (y_tr - mu) / sd, (y_te - mu) / sd
    sets, test_sets = feature_sets(train), feature_sets(test)
    keep = np.ones_like(z_te, dtype=bool)
    rows = []
    for name, x in sets.items():
        scaler = StandardScaler().fit(x[tr])
        pred = Ridge(alpha=alpha).fit(scaler.transform(x[tr]), z_tr).predict(scaler.transform(test_sets[name][te]))
        rows.append(
            {
                "feature_set": name,
                "metric": "cosine_centred",
                "value": float(np.nanmean(_cosine_rows(pred, z_te, keep))),
                "n": int(te.sum()),
            }
        )
    return pd.DataFrame(rows)
