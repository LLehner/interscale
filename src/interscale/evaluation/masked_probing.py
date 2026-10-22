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
* ``"blocks"`` -- whole cells hidden in discs (:func:`~interscale.tl.masking.sample_block_mask`).
  Deep inside a disc wider than the local component's reach, the context a cell's embedding knows
  is only what came from beyond the disc.

Besides the probes, :func:`collect_head_predictions` and :func:`score_heads` score the model's own
decoder heads on the hidden entries of such a pass -- per gene, per gene group, and by how deep
inside the hidden area a cell lies. That is where a dual decoder's local and global heads can be
compared fairly: a pass without masking (``get_model_output``) hands each head the values it is
scored on. Next to them, :func:`reference_predictions` gives the bar any head must clear: averages
of the visible entries of the same graph (slide mean, nearest cells, Gaussian kernels), which use
no model at all.

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
``local``, ``global``, ``pe``, the input baselines and :func:`smoother_features` -- plus
``local+global``, and ``smoother+<embedding>`` for each model embedding when the smoother is there.
The smoother is the bar for context: what averages of the visible expression around a hidden cell
say about it. An embedding knows more than such a smoother only where ``smoother+<embedding>``
beats ``smoother``. With ``depth_bins_um`` the label and covariate probes are also scored by how deep
each hidden cell lies (distance to the nearest visible cell): most hidden cells of a disc lie near
its rim, inside the local component's reach, so a pooled score mostly measures the rim.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from interscale.evaluation.online_probes import ProbeBatchFeatures, _rng_untouched
from interscale.evaluation.online_probes import hidden_depth as _hidden_depth
from interscale.tl.geome_utils import label_codes
from interscale.tl.masking import sample_block_mask, sample_gene_set_mask

MASK_KINDS = ("entries", "cells", "blocks", "gene_set")


def draw_masks(data_list, kind: str, rate: float, seed: int, radius: float | None = None, columns=None) -> None:
    """Hide entries or whole cells of every graph in ``data_list``, in place, from a private generator.

    Writes ``data.gene_mask`` ``[N, G]`` and ``data.mask`` ``[N]`` (its row-wise OR), the attributes
    the gene-masking pipeline reads; under ``mask_strategy="node"`` the model reads ``data.mask``,
    which for ``"cells"`` and ``"blocks"`` hides the same cells. ``"cells"`` hides every gene of a
    hidden cell. ``"blocks"`` hides every gene of the cells in discs of ``radius`` (in the units of
    ``data.pos``) until ``rate`` of each graph's cells lie in one. ``"gene_set"`` hides the gene
    ``columns`` -- and nothing else -- in each cell with probability ``rate`` (1.0: every cell), as
    ``dataset.mask_genes`` does in training; read it with ``mask_strategy="gene"``. The global torch
    generator is not touched.
    """
    if kind not in MASK_KINDS:
        raise ValueError(f"kind must be one of {MASK_KINDS}, got {kind!r}.")
    if kind == "blocks" and not (radius is not None and radius > 0 and 0 < rate < 1):
        raise ValueError(f"'blocks' needs a radius > 0 and a rate in (0, 1), got radius {radius} and rate {rate}.")
    if kind == "gene_set" and (columns is None or not len(columns) or not 0 < rate <= 1):
        raise ValueError(f"'gene_set' needs columns and a cell rate in (0, 1], got {columns} and {rate}.")
    gen = torch.Generator().manual_seed(int(seed))
    for data in data_list:
        n, g = data.x.shape
        if kind == "entries":
            gene_mask = torch.rand(n, g, generator=gen) < rate
        elif kind == "cells":
            gene_mask = (torch.rand(n, generator=gen) < rate)[:, None].expand(n, g).clone()
        elif kind == "gene_set":
            gene_mask = sample_gene_set_mask(n, g, columns, rate, generator=gen)
        else:
            if getattr(data, "pos", None) is None:
                raise ValueError("'blocks' are drawn over cell coordinates, but a graph has no `pos`.")
            gene_mask = sample_block_mask(data.pos, radius, rate, generator=gen)[:, None].expand(n, g).clone()
        data.gene_mask = gene_mask
        data.mask = gene_mask.any(dim=1)


#: The feature set :func:`smoother_features` adds.
SMOOTHER = "smoother"
#: The feature set :func:`position_features` adds.
POSITION = "position"
#: The model embeddings that get a ``smoother+<name>`` set when the smoother was collected.
MODEL_EMBEDDINGS = ("local", "global", "local+global")


def feature_sets(features: ProbeBatchFeatures, *, joint_embeddings: bool = True) -> dict[str, np.ndarray]:
    """The collected representations, plus the combinations the probes compare.

    ``local+global``; ``smoother+<embedding>`` with the smoother; with the position as well, the
    model-free bar ``smoother+position`` and ``smoother+position+<embedding>``; with the position
    alone, ``position+<embedding>``. An embedding knows more than the bar where bar+embedding beats it.
    ``joint_embeddings=False`` leaves out ``local+global`` and every set built on it (three of the twelve
    sets of a dual model with a smoother and a position, two of them the widest): the summaries of a sweep
    read ``local`` and ``global`` each against the bars, never the two together.
    """
    sets = dict(features.embeddings)
    if joint_embeddings and "local" in sets and "global" in sets:
        sets["local+global"] = np.concatenate([sets["local"], sets["global"]], axis=1)
    models = [name for name in MODEL_EMBEDDINGS if name in sets]
    if SMOOTHER in sets:
        for name in models:
            sets[f"{SMOOTHER}+{name}"] = np.concatenate([sets[SMOOTHER], sets[name]], axis=1)
    if POSITION in sets:
        bar = f"{SMOOTHER}+{POSITION}" if SMOOTHER in sets else POSITION
        if SMOOTHER in sets:
            sets[bar] = np.concatenate([sets[SMOOTHER], sets[POSITION]], axis=1)
        for name in models:
            sets[f"{bar}+{name}"] = np.concatenate([sets[bar], sets[name]], axis=1)
    return sets


def combined_features(*functions):
    """One ``extra_features`` function for :func:`~interscale.evaluation.online_probes.collect_features` from several."""

    def features(batch, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for function in functions:
            out.update(function(batch, hidden))
        return out

    return features


def position_features(*, spatial_unit_um: float, patch_um: tuple[float, ...], max_offset_um: float):
    """A model-free feature set for the probes: where a cell lies on its slide, and nothing else.

    Returns a function for :func:`~interscale.evaluation.online_probes.collect_features`'s
    ``extra_features``. Coordinates are taken relative to the centroid of the cell's graph, so slides
    placed anywhere share one frame. Each cell gets, as :data:`POSITION`:

    * its offset from the centroid in µm (x, y) and its distance from it;
    * per size ``p`` in ``patch_um``, a one-hot of the square patch of side ``p`` it falls in, on a
      grid centred on the centroid that covers ``max_offset_um`` in every direction (cells beyond
      fall into the border patches).

    A linear probe on the patches is a lookup of the average label per area, learned on the
    training slides: what position alone says about a hidden cell, at several scales. The
    positional encoding hands a model at least this much without any learning, so an embedding
    knows more than position only where ``smoother+position+<embedding>`` beats ``smoother+position``.
    No expression enters.

    Parameters
    ----------
    spatial_unit_um
        µm per unit of ``pos``.
    patch_um
        Patch sides in µm, e.g. ``(500, 1000, 2500)``.
    max_offset_um
        The largest distance of a cell from its graph's centroid along either axis, over every
        graph the features are collected for, in µm. Fixes the number of patches, so every graph
        and split gets the same columns.
    """
    patches = [float(p) for p in patch_um]
    if any(p <= 0 for p in patches):
        raise ValueError(f"patch sides must be positive, got {patch_um}.")
    if not max_offset_um > 0:
        raise ValueError(f"max_offset_um must be positive, got {max_offset_um}.")
    half = [int(np.ceil(float(max_offset_um) / p)) for p in patches]

    def features(batch, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        pos = getattr(batch, "pos", None)
        if pos is None:
            raise ValueError("the position features need coordinates, but the graphs carry no `pos`.")
        pos = pos[:, :2].double() * float(spatial_unit_um)
        graph = batch.batch
        n_graphs = int(graph.max()) + 1
        sums = torch.zeros(n_graphs, 2, dtype=pos.dtype, device=pos.device).index_add_(0, graph, pos)
        counts = torch.bincount(graph, minlength=n_graphs).to(pos.dtype)
        offset = pos - (sums / counts[:, None])[graph]
        parts = [offset, offset.norm(dim=1, keepdim=True)]
        for p, h in zip(patches, half, strict=True):
            n = 2 * h
            cell = (torch.floor(offset / p).long() + h).clamp(0, n - 1)
            one_hot = torch.zeros(len(pos), n * n, dtype=pos.dtype, device=pos.device)
            one_hot[torch.arange(len(pos), device=pos.device), cell[:, 0] * n + cell[:, 1]] = 1.0
            parts.append(one_hot)
        return {POSITION: torch.cat(parts, dim=1).float()}

    return features


def smoother_features(*, spatial_unit_um: float, kernels_um: tuple[float, ...] = (), k_nearest: tuple[int, ...] = ()):
    """A model-free feature set for the probes: averages of the visible expression around each cell.

    Returns a function for :func:`~interscale.evaluation.online_probes.collect_features`'s
    ``extra_features``. It gives each cell its :func:`reference_predictions` -- the slide mean, one
    Gaussian kernel per bandwidth in ``kernels_um``, the ``k`` nearest cells per ``k_nearest`` --
    side by side as :data:`SMOOTHER`, ``[N, G * (1 + kernels + k)]``, from the entries the model was
    shown. No label mean: a label probe would read its own target from it.
    """

    def features(batch, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if getattr(batch, "pos", None) is None:
            raise ValueError("the smoother averages over neighbours in µm, but the graphs carry no `pos`.")
        refs = reference_predictions(
            batch.x,
            hidden,
            batch.pos,
            batch.batch,
            spatial_unit_um=float(spatial_unit_um),
            kernels_um=tuple(kernels_um),
            k_nearest=tuple(k_nearest),
        )
        return {SMOOTHER: torch.cat(list(refs.values()), dim=1).float()}

    return features


def _flag(column: pd.Series, name: str) -> np.ndarray | None:
    """``column`` as a 0/1 float vector when it is boolean (also as the strings "True" / "False"), else None."""
    if column.dtype == bool:
        return column.to_numpy(dtype=np.float64)
    if isinstance(column.dtype, pd.CategoricalDtype) or column.dtype == object:
        levels = {str(v).lower() for v in column.dropna().unique()}
        if levels <= {"true", "false", "1", "0"}:
            return column.astype(str).str.lower().isin(["true", "1"]).to_numpy(dtype=np.float64)
        return None
    values = column.dropna().unique()
    if len(values) <= 2 and set(values.tolist()) <= {0, 1}:
        return column.fillna(0.0).to_numpy(dtype=np.float64)
    return None


def _driver_block(column: pd.Series, name: str, n_basis: int) -> np.ndarray:
    """``[n, D]`` float64 features of one driver column.

    A flag is itself; another categorical is one-hot; a number becomes ``n_basis`` Gaussian bumps (centres at
    quantiles of its finite values, each as wide as the spacing of its neighbours, so a ridge can draw any smooth
    function of it, steeper where the values are dense) and, where it has missing values (a driver that exists on
    some slides only: 0 in the bumps), an indicator of them.
    """
    flag = _flag(column, name)
    if flag is not None:
        return flag[:, None]
    if isinstance(column.dtype, pd.CategoricalDtype) or column.dtype == object:
        return pd.get_dummies(column.astype(str)).to_numpy(dtype=np.float64)
    x = column.to_numpy(dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError(f"driver {name!r} has no finite value.")
    centres = np.unique(np.quantile(x[finite], np.linspace(0.0, 1.0, int(n_basis))))
    gaps = np.diff(centres)
    if len(centres) == 1:
        width = np.ones(1)
    else:
        left, right = np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]]
        width = np.maximum((left + right) / 2.0, 1e-12)
    bumps = np.exp(-0.5 * ((np.where(finite, x, 0.0)[:, None] - centres[None]) / width[None]) ** 2)
    bumps[~finite] = 0.0
    if (~finite).any():
        bumps = np.concatenate([bumps, (~finite).astype(np.float64)[:, None]], axis=1)
    return bumps


def oracle_features(
    obs: pd.DataFrame, keys, *, gates: dict[str, str] | None = None, n_basis: int = 24, offset: int = 1
):
    """The true drivers of a synthetic dataset as a ``drivers(batch, hidden) -> [N, D]`` function for :class:`FittedReferences`.

    ``keys`` name columns of ``obs`` that drive the planted signals: distances to the sources, flags (a cell is
    exposed to a sender, lies in a stripe), receiver flags. Each becomes features by :func:`_driver_block`. A key
    in ``gates`` (``{key: gate column}``, the gate a boolean column) also gets its features multiplied by the gate,
    for a response that exists only where the gate is on (a sparse receiver at a distance from its source; a cone).
    The features are the whole of ``obs``, computed once; a batch picks its cells' rows by ``batch.obs_names`` (a cell's
    position in ``obs`` plus ``offset``, as ``prepare_geome_dataset`` names them). Nothing here reads expression.

    This is a bar, not a model: a ridge on these features (``ref_oracle``) says what is learnable from the planted
    drivers by a smooth, additive-in-basis function. Where it stays far below 1 on a noiseless programme the basis
    is too coarse (raise ``n_basis``), not the task impossible.
    """
    gates = dict(gates or {})
    keys = list(keys)
    missing = [c for c in [*keys, *gates.values()] if c not in obs.columns]
    if missing:
        raise ValueError(f"obs lacks the driver columns {missing}.")
    stray = sorted(set(gates) - set(keys))
    if stray:
        raise ValueError(f"gates name keys that are not drivers: {stray}.")
    blocks = []
    for key in keys:
        block = _driver_block(obs[key], key, n_basis)
        blocks.append(block)
        if key in gates:
            gate = _flag(obs[gates[key]], gates[key])
            if gate is None:
                raise ValueError(f"the gate {gates[key]!r} of {key!r} is not a boolean column.")
            blocks.append(block * gate[:, None])
    matrix = np.concatenate(blocks, axis=1).astype(np.float32)

    def drivers(batch, hidden: torch.Tensor) -> torch.Tensor:
        names = batch.obs_names
        names = names.detach().cpu().numpy() if isinstance(names, torch.Tensor) else np.asarray(names)
        rows = np.rint(names.astype(np.float64)).astype(np.int64) - int(offset)
        if rows.min() < 0 or rows.max() >= len(matrix):
            raise ValueError(f"the graph's cells point outside the {len(matrix)} rows of obs; wrong offset or data?")
        return torch.as_tensor(matrix[rows], dtype=torch.float64, device=batch.x.device)

    drivers.n_features = matrix.shape[1]
    return drivers


def _fit_rows(rows: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    """``rows`` (boolean over the training cells), cut to a seeded random subset of at most ``max_rows`` of them.

    A probe's cost grows with the cells it is fitted on, and a probe is a read-out, not an estimate that wants all of
    them. One draw per call, so every feature set and every target is fitted on the same cells and the sets stay
    paired; the test cells are never cut. ``None``: every row, as before.
    """
    if max_rows is None or int(rows.sum()) <= max_rows:
        return rows
    if max_rows < 2:
        raise ValueError(f"max_train_rows must be at least 2 (or None for every row), got {max_rows}.")
    chosen = np.random.default_rng(seed).choice(np.flatnonzero(rows), size=int(max_rows), replace=False)
    keep = np.zeros_like(rows)
    keep[chosen] = True
    return keep


def _depth_rows(depth_um: np.ndarray | None, rows: np.ndarray, depth_bins_um) -> list[tuple[str, np.ndarray]]:
    """``("all", every row)``, then one ``(bin, rows)`` per non-empty depth bin of the wholly hidden rows.

    ``rows`` selects the cells a probe is scored on; the masks returned are over that selection.
    """
    out = [("all", np.ones(int(rows.sum()), dtype=bool))]
    if depth_bins_um is None:
        return out
    if depth_um is None:
        raise ValueError("depth_bins_um needs each cell's depth; collect the features with spatial_unit_um.")
    depth = np.asarray(depth_um, dtype=float)[rows]
    cut = pd.cut(pd.Series(depth), bins=[float(b) for b in depth_bins_um], right=True)
    for interval in cut.cat.categories:
        in_bin = (depth > 0) & (cut == interval).to_numpy()
        if in_bin.any():
            out.append((str(interval), in_bin))
    return out


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
    depth_bins_um: list[float] | None = None,
    max_train_rows: int | None = None,
    joint_embeddings: bool = True,
) -> pd.DataFrame:
    """Read each categorical target out of each representation with a balanced logistic regression.

    With ``hidden_only`` (collected from a ``"cells"`` pass) only hidden cells are used, so the label
    has to come from the context: a label the cell's own expression encodes -- a response gene that
    marks it -- is otherwise read straight off its input.

    The cost is the logistic regressions: one per target and feature set, each on every training row and up to
    ``max_iter`` iterations, the sets holding the smoother the widest. ``max_train_rows`` fits each on one seeded
    random subset of that many training rows (the same rows for every set and target; the test rows are all scored),
    and ``joint_embeddings=False`` drops the sets that hold the local and the global embedding together
    (:func:`feature_sets`). The defaults fit everything, as before.

    Returns
    -------
    pandas.DataFrame
        One row per ``(target, feature_set, metric)``: ``balanced_accuracy`` and ``f1_macro``, with
        a ``majority`` row as the floor. With ``depth_bins_um`` (needs ``depth_um``) every row also
        carries ``depth_um``: ``"all"`` for every test cell, else the depth bin -- the same probe, fit
        on every training cell, scored on the wholly hidden test cells in that bin. A bin holding a
        single class scores NaN: there is nothing to tell apart (balanced accuracy would read 1 for
        predicting that class everywhere).
    """
    tr = train.cell_masked.astype(bool) if hidden_only else np.ones(train.n_cells(), bool)
    te = test.cell_masked.astype(bool) if hidden_only else np.ones(test.n_cells(), bool)
    tr = _fit_rows(tr, max_train_rows, seed)
    groups = _depth_rows(test.depth_um, te, depth_bins_um)
    sets = feature_sets(train, joint_embeddings=joint_embeddings)
    test_sets = feature_sets(test, joint_embeddings=joint_embeddings)
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
            for depth, keep in groups:
                if len(np.unique(y_te[keep])) < 2:
                    accuracy = f1 = float("nan")
                else:
                    with warnings.catch_warnings():
                        # A depth bin can lack a class the probe predicts; it is scored over the classes present.
                        warnings.simplefilter("ignore", UserWarning)
                        accuracy = balanced_accuracy_score(y_te[keep], pred[keep])
                    f1 = f1_score(y_te[keep], pred[keep], average="macro", labels=classes, zero_division=0)
                extra = {} if depth_bins_um is None else {"depth_um": depth}
                rows.append(
                    {"target": target, "feature_set": name, "metric": "balanced_accuracy", "value": accuracy}
                    | extra
                    | {"n": int(keep.sum())}
                )
                rows.append(
                    {"target": target, "feature_set": name, "metric": "f1_macro", "value": f1}
                    | extra
                    | {"n": int(keep.sum())}
                )
    return pd.DataFrame(rows)


def probe_covariates(
    train: ProbeBatchFeatures,
    test: ProbeBatchFeatures,
    names: list[str],
    *,
    hidden_only: bool = True,
    alpha: float = 1.0,
    depth_bins_um: list[float] | None = None,
    max_train_rows: int | None = None,
    joint_embeddings: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Predict a vector of numeric covariates (e.g. distances to every source); score by cosine.

    ``max_train_rows`` and ``joint_embeddings`` cut the cost as in :func:`probe_labels` (``seed`` draws the rows).

    Each covariate is z-scored with the training statistics first, so every one weighs the same in
    the cosine and the training mean sits at 0: ``cosine_centred`` is then the cosine of the
    predicted and true z-vectors, per cell, averaged. ``mean`` is the floor (undefined, NaN).

    Returns
    -------
    pandas.DataFrame
        One row per ``feature_set`` with the mean per-cell ``cosine_centred``. With ``depth_bins_um``
        (needs ``depth_um``) also one per depth bin, each row carrying ``depth_um`` (``"all"`` for
        every test cell), as in :func:`probe_labels`.
    """
    tr = train.cell_masked.astype(bool) if hidden_only else np.ones(train.n_cells(), bool)
    te = test.cell_masked.astype(bool) if hidden_only else np.ones(test.n_cells(), bool)
    tr = _fit_rows(tr, max_train_rows, seed)
    groups = _depth_rows(test.depth_um, te, depth_bins_um)
    y_tr = np.stack([train.continuous[c] for c in names], axis=1)[tr].astype(np.float64)
    y_te = np.stack([test.continuous[c] for c in names], axis=1)[te].astype(np.float64)
    mu, sd = y_tr.mean(0), y_tr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z_tr, z_te = (y_tr - mu) / sd, (y_te - mu) / sd
    sets = feature_sets(train, joint_embeddings=joint_embeddings)
    test_sets = feature_sets(test, joint_embeddings=joint_embeddings)
    keep_all = np.ones_like(z_te, dtype=bool)
    rows = []
    for name, x in sets.items():
        scaler = StandardScaler().fit(x[tr])
        pred = Ridge(alpha=alpha).fit(scaler.transform(x[tr]), z_tr).predict(scaler.transform(test_sets[name][te]))
        cosine = _cosine_rows(pred, z_te, keep_all)
        for depth, keep in groups:
            extra = {} if depth_bins_um is None else {"depth_um": depth}
            rows.append(
                {"feature_set": name, "metric": "cosine_centred", "value": float(np.nanmean(cosine[keep]))}
                | extra
                | {"n": int(keep.sum())}
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the decoder heads


#: Name prefix of the reference predictors among the heads of a :class:`HeadPredictions`.
REFERENCE_PREFIX = "ref_"
#: Name prefix of the blends of a model head with a reference (:func:`blend_heads`).
BLEND_PREFIX = "blend_"


@dataclass
class HeadPredictions:
    """The model's decoder heads, and any reference predictors, over one masked pass.

    Every array is in the same cell order.

    Attributes
    ----------
    heads
        ``{name: [N, G]}``. The model's own heads are ``local`` and/or ``global``: both for a dual
        decoder (the global one as the model reports it, i.e. local plus residual under
        ``model.decoder.global_residual``, or the sum of both decoders under ``model.decoder.additive``, which adds
        ``global_part``, the global decoder's output alone, while ``local`` is the local decoder's alone), ``global``
        for a ``GlobalModel`` or a single-decoder ``CombinedModel``, ``local`` for a ``LocalModel``. Reference predictors
        (:func:`reference_predictions`) are added under names starting with
        :data:`REFERENCE_PREFIX`.
    true
        ``[N, G]`` uncorrupted expression.
    hidden
        ``[N, G]`` boolean, True where the entry was hidden from the model.
    depth_um
        ``[N]`` distance from each wholly hidden cell to the nearest cell of its graph with any
        visible entry, in µm; 0 for every other cell and ``inf`` in a graph hidden completely.
        None when the graphs carry no ``pos``.
    cells
        ``[N]`` the name of each row's cell as its graph carries it (``obs_names``: the cell's
        position in the AnnData plus the dataset's offset), to find a row again in ``adata.obs``
        (:func:`group_receivers`). None when the graphs carry none.
    parts
        ``{head: {part: [N, G]}}`` for a head whose prediction is the sum of named parts (``StepOutput.parts``: the own-cell
        skip and the transformer's read-out; a residual dual decoder's local and global), rows as in ``heads``. What
        :func:`attribution_shares` reads. None where no head has more than one part.
    """

    heads: dict[str, np.ndarray]
    true: np.ndarray
    hidden: np.ndarray
    depth_um: np.ndarray | None = None
    cells: np.ndarray | None = None
    parts: dict[str, dict[str, np.ndarray]] | None = None


def _weighted_mean(w: torch.Tensor, xv: torch.Tensor, vis: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """``(w @ xv) / (w @ vis)`` per entry: the ``w``-weighted mean of the visible values, else ``fallback``."""
    num, den = w @ xv, w @ vis
    return torch.where(den > 1e-12, num / den.clamp_min(1e-12), fallback)


def reference_predictions(
    x: torch.Tensor,
    hidden: torch.Tensor,
    pos: torch.Tensor,
    graph: torch.Tensor,
    *,
    spatial_unit_um: float,
    kernels_um: tuple[float, ...] = (),
    k_nearest: tuple[int, ...] = (),
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Predictions that use no model: averages of the visible entries of the same graph.

    The bar a model head must clear on hidden cells. Each predicts entry ``(i, g)`` as a weighted
    mean of gene ``g`` over the other cells of ``i``'s graph, counting only entries that are
    visible: ``sum_j w_ij v_jg x_jg / sum_j w_ij v_jg``. A hidden entry (``v = 0``) never enters any
    average, so no reference can read a value the model could not see, under cell, entry or block
    masking alike, and a cell never averages itself.

    * ``ref_slide_mean`` -- every cell weighted equally.
    * ``ref_label_mean`` -- with ``labels``: the cells sharing ``i``'s label (an oracle when the
      label is ground truth the model is not given, e.g. a region).
    * ``ref_kernel_<s>um`` -- a Gaussian of the distance, bandwidth ``s`` µm.
    * ``ref_nearest_<k>`` -- the ``k`` nearest cells that have any visible entry, equally.

    Where a weighted mean has nothing to average (a kernel underflowing far from every visible
    cell), the slide mean stands in; a gene with no visible entry in the graph is predicted as 0.

    Parameters
    ----------
    x
        ``[N, G]`` true values, in batch node order.
    hidden
        ``[N, G]`` boolean, True where the entry is hidden.
    pos
        ``[N, D]`` coordinates, in the units of ``obsm[spatial_key]``.
    graph
        ``[N]`` graph index of each cell (``batch.batch``); graphs never mix.
    spatial_unit_um
        µm per unit of ``pos``.
    kernels_um, k_nearest
        Bandwidths in µm and neighbour counts, one reference each.
    labels
        ``[N]`` integer labels for ``ref_label_mean``, or None to leave it out.

    Returns
    -------
    dict
        ``{name: [N, G]}`` float64 tensors, in batch node order.
    """
    names = ["ref_slide_mean"] + (["ref_label_mean"] if labels is not None else [])
    names += [f"ref_kernel_{s:g}um" for s in kernels_um] + [f"ref_nearest_{k}" for k in k_nearest]
    out = {name: torch.zeros(x.shape, dtype=torch.float64, device=x.device) for name in names}
    for g in torch.unique(graph):
        rows = torch.where(graph == g)[0]
        vis = (~hidden[rows]).double()
        xv = x[rows].double() * vis
        slide = torch.nan_to_num(xv.sum(0) / vis.sum(0), nan=0.0)
        n = len(rows)
        out["ref_slide_mean"][rows] = slide.expand(n, -1)

        fallback = slide.expand(n, -1)

        if labels is not None:
            lab = labels[rows]
            same = (lab[:, None] == lab[None, :]).double()
            same.fill_diagonal_(0.0)
            out["ref_label_mean"][rows] = _weighted_mean(same, xv, vis, fallback)
        if not kernels_um and not k_nearest:
            continue
        # Candidates: other cells with any visible entry. A wholly hidden cell would add nothing to
        # either sum, and leaving it out keeps the kernel's exponent <= 0 (no overflow).
        open_cell = ~hidden[rows].all(dim=1)
        d = torch.cdist(pos[rows].double(), pos[rows].double()) * float(spatial_unit_um)
        d.fill_diagonal_(float("inf"))
        d = d.masked_fill(~open_cell[None, :], float("inf"))
        candidate = torch.isfinite(d)
        d_min = d.min(dim=1, keepdim=True).values
        for s in kernels_um:
            exponent = -(d.pow(2) - d_min.pow(2)) / (2.0 * float(s) ** 2)
            w = torch.where(candidate, torch.exp(exponent.masked_fill(~candidate, 0.0)), torch.zeros_like(d))
            out[f"ref_kernel_{s:g}um"][rows] = _weighted_mean(w, xv, vis, fallback)
        for k in k_nearest:
            kk = min(int(k), int(open_cell.sum()))
            w = torch.zeros_like(d)
            if kk > 0:
                nearest = torch.topk(d, kk, dim=1, largest=False).indices
                w.scatter_(1, nearest, 1.0)
                w = w * candidate
            out[f"ref_nearest_{k}"][rows] = _weighted_mean(w, xv, vis, fallback)
    return out


def _reference_parts(
    batch, hidden: torch.Tensor, *, spatial_unit_um: float, kernels_um, position, drivers=None
) -> dict:
    """Per cell, model-free inputs of the fitted references, in batch node order (float64).

    ``own``: the cell's visible values (a hidden entry reads 0); ``context``: Gaussian-kernel means
    of the visible values of the other cells of its graph, one block of genes per bandwidth;
    ``position``: :func:`position_features` (or absent); ``driver``: :func:`oracle_features` (or absent). No
    hidden value enters any of them.
    """
    x = batch.x.double()
    parts = {"own": torch.where(hidden, torch.zeros_like(x), x)}
    if kernels_um:
        refs = reference_predictions(
            batch.x, hidden, batch.pos, batch.batch, spatial_unit_um=spatial_unit_um, kernels_um=tuple(kernels_um)
        )
        parts["context"] = torch.cat([refs[f"ref_kernel_{s:g}um"] for s in kernels_um], dim=1)
    if position is not None:
        parts["position"] = position(batch, hidden)[POSITION].double()
    if drivers is not None:
        parts["driver"] = drivers(batch, hidden).double()
    return parts


#: The fitted references and the input parts each one reads.
FITTED_REFERENCES = {
    "ref_own": ("own",),
    "ref_own+context": ("own", "context"),
    "ref_own+context+position": ("own", "context", "position"),
    "ref_oracle": ("own", "driver"),
}
#: The reference that reads the planted drivers (:func:`oracle_features`): a bar for the task, not for a model. The
#: report leaves it out of "the best reference".
ORACLE_REFERENCE = "ref_oracle"


class FittedReferences:
    """Ridge references fitted on training graphs: what a hidden entry's own cell, surroundings and place say.

    The visible-average references (:func:`reference_predictions`) need the hidden gene to be
    visible somewhere nearby. When a gene set is hidden in every cell -- a signalling programme's
    responses, say -- nothing nearby shows it, and the bar for a model becomes what the visible
    *other* genes predict. Three ridge regressions per hidden gene, fit on training graphs carrying
    their masks and applied to any graph with the same masking:

    * ``ref_own`` -- the cell's own visible genes: within-cell co-expression, no space (a receptor
      marks a responder; an identity gene tracks a pathway readout).
    * ``ref_own+context`` -- plus Gaussian-kernel means of the visible genes of the surrounding
      cells at each bandwidth: a fixed multi-scale linear view of the drivers around the cell.
    * ``ref_own+context+position`` -- plus :func:`position_features`.
    * ``ref_oracle`` (with ``drivers``) -- the cell's own visible genes plus the true drivers of the planted signals
      (:func:`oracle_features`; synthetic data only). What a ridge on the truth reaches: where a model falls short of
      it the model is the limit, where it falls short too the task cannot be solved from these inputs (or the basis
      is too coarse).

    A model knows more than fixed kernels of its inputs only where it beats ``ref_own+context``.
    Call the instance as ``(batch, hidden) -> {name: [N, G]}`` (``collect_head_predictions``'s
    ``fitted_references``); genes never hidden in training are predicted as their training mean.

    Parameters
    ----------
    train_data
        Training graphs carrying ``pos`` and the masks to fit under (``gene_mask``).
    spatial_unit_um
        µm per unit of ``pos``.
    kernels_um
        Bandwidths of the context, µm.
    position
        A :func:`position_features` function, or None to leave the position reference out.
    drivers
        An :func:`oracle_features` function, or None to leave the oracle out.
    alphas
        Candidate ridge penalties (on standardised features). Each reference takes the one with the
        lowest squared error on held-out training graphs (``folds`` groups of whole graphs) and is
        then refit on every training graph, so a wide feature set cannot lose to a narrower one by
        overfitting. The choice is in ``alpha_``.
    folds
        Groups of training graphs for that choice (at most one per graph).
    device
        Where the features are computed.
    """

    def __init__(
        self,
        train_data,
        *,
        spatial_unit_um: float,
        kernels_um=(),
        position=None,
        drivers=None,
        alphas=(0.1, 1.0, 10.0, 100.0, 1e3, 1e4),
        folds: int = 3,
        device="cpu",
    ):
        self.spatial_unit_um, self.kernels_um, self.position = float(spatial_unit_um), tuple(kernels_um), position
        self.drivers = drivers
        names = [n for n, parts in FITTED_REFERENCES.items() if position is not None or "position" not in parts]
        names = [n for n in names if kernels_um or "context" not in FITTED_REFERENCES[n]]
        names = [n for n in names if drivers is not None or "driver" not in FITTED_REFERENCES[n]]
        chunks, targets, hidden_rows, graph_of = {}, [], [], []
        graphs = list(train_data)
        loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
        with torch.no_grad():
            for g, batch in enumerate(loader):
                batch = batch.to(device)
                hidden = self._hidden(batch)
                for part, value in self._parts(batch, hidden).items():
                    chunks.setdefault(part, []).append(value.cpu().numpy().astype(np.float32))
                targets.append(batch.x.double().cpu().numpy())
                hidden_rows.append(hidden.cpu().numpy())
                graph_of.append(np.full(batch.num_nodes, g))
        parts = {k: np.concatenate(v) for k, v in chunks.items()}
        y, rows, graph_of = np.concatenate(targets), np.concatenate(hidden_rows), np.concatenate(graph_of)
        self.fill = y.mean(0)
        self.target_columns = np.flatnonzero(rows.any(0))
        fold_of = (np.arange(len(graphs)) % max(1, min(int(folds), len(graphs))))[graph_of]
        self.models, self.alpha_ = {}, {}
        for name in names:
            x = np.concatenate([parts[p] for p in FITTED_REFERENCES[name]], axis=1)
            self.alpha_[name] = self._choose_alpha(x, y, rows, fold_of, alphas)
            self.models[name] = self._fit(x, y, rows, self.alpha_[name])

    @staticmethod
    def _hidden(batch) -> torch.Tensor:
        gene_mask = getattr(batch, "gene_mask", None)
        if gene_mask is not None:
            return gene_mask.bool()
        return batch.mask.bool()[:, None].expand_as(batch.x)

    def _parts(self, batch, hidden):
        return _reference_parts(
            batch,
            hidden,
            spatial_unit_um=self.spatial_unit_um,
            kernels_um=self.kernels_um,
            position=self.position,
            drivers=self.drivers,
        )

    def _choose_alpha(self, x, y, rows, fold, alphas) -> float:
        """The penalty with the lowest squared error on held-out folds of whole training graphs."""
        alphas = [float(a) for a in alphas]
        if len(alphas) == 1 or len(np.unique(fold)) < 2:
            return alphas[0]
        errors = np.zeros(len(alphas))
        for f in np.unique(fold):
            held = fold == f
            for k, alpha in enumerate(alphas):
                for columns, scaler, ridge in self._fit(x[~held], y[~held], rows[~held], alpha):
                    keep = held & rows[:, columns[0]]
                    if keep.any():
                        pred = ridge.predict(scaler.transform(x[keep])).reshape(int(keep.sum()), -1)
                        errors[k] += float(((pred - y[keep][:, columns]) ** 2).sum())
        return alphas[int(np.argmin(errors))]

    def _fit(self, x, y, rows, alpha):
        """``[(columns, scaler, ridge)]``: one multi-output fit per set of genes hidden in the same cells."""
        fitted, done = [], set()
        for g in self.target_columns:
            if g in done:
                continue
            same = [c for c in self.target_columns if c not in done and np.array_equal(rows[:, c], rows[:, g])]
            done.update(same)
            keep = rows[:, g]
            if keep.sum() < 2:
                continue
            scaler = StandardScaler().fit(x[keep])
            ridge = Ridge(alpha=alpha).fit(scaler.transform(x[keep]), y[keep][:, same])
            fitted.append((np.asarray(same), scaler, ridge))
        return fitted

    #: What a fitted instance holds besides the ``position`` function (a closure, which a file cannot keep).
    STATE = ("spatial_unit_um", "kernels_um", "fill", "target_columns", "models", "alpha_")

    def state(self) -> dict:
        """The fit as a plain dict of arrays and sklearn objects, for :func:`pickle` (see :meth:`from_state`)."""
        return {name: getattr(self, name) for name in self.STATE}

    @classmethod
    def from_state(cls, state: dict, position=None, drivers=None) -> FittedReferences:
        """An instance with the fit of :meth:`state`; ``position`` and ``drivers`` are the same functions the fit used.

        Lets jobs that evaluate different models on the same training slides share one fit: the ridge
        regressions depend on the graphs, their masks and the settings only, never on a model.
        """
        missing = [name for name in cls.STATE if name not in state]
        if missing:
            raise ValueError(f"the saved fit lacks {missing}")
        needs = [n for n in state["models"] if "driver" in FITTED_REFERENCES.get(n, ())]
        if needs and drivers is None:
            raise ValueError(f"the saved fit holds {needs}, which read the true drivers: pass `drivers`.")
        if drivers is not None and not needs:
            raise ValueError("the saved fit holds no oracle, but `drivers` were given: it was fitted without them.")
        obj = cls.__new__(cls)
        for name in cls.STATE:
            setattr(obj, name, state[name])
        obj.position, obj.drivers = position, drivers
        return obj

    def __call__(self, batch, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = {k: v.cpu().numpy() for k, v in self._parts(batch, hidden).items()}
        out = {}
        for name, fitted in self.models.items():
            x = np.concatenate([parts[p] for p in FITTED_REFERENCES[name]], axis=1)
            pred = np.tile(self.fill, (len(x), 1))
            for columns, scaler, ridge in fitted:
                pred[:, columns] = ridge.predict(scaler.transform(x)).reshape(len(x), -1)
            out[name] = torch.as_tensor(pred, dtype=torch.float64, device=batch.x.device)
        return out


def _model_heads(module, out) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """The decoder heads in one step's output, by name, and the half of ``y_true``/``entry_mask`` they share."""
    from interscale.module.combined_module import DualDecoderCombinedModule

    if isinstance(module, DualDecoderCombinedModule):
        n = len(out.y_pred) // 2
        heads = {"local": out.y_pred[:n], "global": out.y_pred[n:]}
        if module.additive:
            # The global half is the sum; the global decoder's output alone is the other part to score.
            heads["global_part"] = out.parts["global"]["global"]
        return heads, out.y_true[:n], out.entry_mask[:n]
    name = "global" if getattr(module, "registered_global_component", False) else "local"
    return {name: out.y_pred}, out.y_true, out.entry_mask


def collect_head_predictions(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    mask_strategy: str,
    batch_size: int,
    device,
    spatial_unit_um: float | None = None,
    references: dict | None = None,
    fitted_references=None,
) -> HeadPredictions:
    """Run ``module`` over masked graphs and gather its decoder heads' predictions.

    The graphs keep the masks they carry (e.g. from :func:`draw_masks`), and the module corrupts
    its input with them exactly as in training, so every hidden entry is predicted from what the
    model could see. Row ``i`` of every output array is the same cell: a dual decoder's two halves
    are already gathered by one index, and the depth and the references are gathered by it too.

    Parameters
    ----------
    module
        Any module of a ``CombinedModel`` (dual or single decoder), ``GlobalModel`` or
        ``LocalModel``; the heads it has are collected (see :class:`HeadPredictions`).
    data_list
        PyG ``Data`` objects carrying ``mask`` and, for ``mask_strategy="gene"``, ``gene_mask``.
    prediction_task, prediction_level
        Must be ``"regression"`` and ``"node"``: the heads reconstruct expression per cell.
    mask_strategy
        ``cfg.dataset.mask_strategy``; decides which attribute says what was hidden, as in
        :func:`~interscale.evaluation.online_probes.collect_features`.
    batch_size
        Graphs per forward pass.
    device
        Device to move each batch to.
    spatial_unit_um
        ``dataset.spatial_unit_um``. When given and the graphs carry ``pos``, ``depth_um`` is filled.
    references
        Keyword arguments of :func:`reference_predictions` -- ``kernels_um``, ``k_nearest``, and
        ``label_key``, the name of an annotation on the graphs for ``ref_label_mean`` -- to add the
        references as heads, or None for none. Needs ``spatial_unit_um`` and ``pos``.
    fitted_references
        A :class:`FittedReferences`, whose predictions join the heads, or None.

    Returns
    -------
    HeadPredictions
    """
    if "regression" not in prediction_task or prediction_level != "node":
        raise ValueError("The heads are scored on reconstructed expression; it needs regression at node level.")
    if references is not None and spatial_unit_um is None:
        raise ValueError("references average over neighbours in µm; pass spatial_unit_um.")

    was_training = module.training
    module.eval()
    heads: dict[str, list[np.ndarray]] = {}
    part_chunks: dict[str, dict[str, list[np.ndarray]]] = {}
    chunks = {"true": [], "hidden": [], "depth_um": [], "cells": []}
    loader = DataLoader(dataset=data_list, shuffle=False, batch_size=batch_size, num_workers=0)
    try:
        with _rng_untouched(device), torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = module._common_step(batch, prediction_task, prediction_level)
                found, y_true, entry_mask = _model_heads(module, out)
                idx = out.view.padded_node_idx
                if idx is None:  # a local module: every cell, in batch order
                    idx = torch.arange(batch.num_nodes, device=batch.x.device)
                for name, value in found.items():
                    heads.setdefault(name, []).append(value.cpu().numpy())
                for head, named in (out.parts or {}).items():
                    for part, value in named.items():
                        part_chunks.setdefault(head, {}).setdefault(part, []).append(value.detach().cpu().numpy())
                chunks["true"].append(y_true.cpu().numpy())
                chunks["hidden"].append(entry_mask.bool().cpu().numpy())
                names = getattr(batch, "obs_names", None)
                if isinstance(names, torch.Tensor):
                    chunks["cells"].append(names[idx].double().cpu().numpy())

                gene_mask = getattr(batch, "gene_mask", None) if mask_strategy == "gene" else None
                hidden = batch.mask.bool()[:, None].expand_as(batch.x) if gene_mask is None else gene_mask.bool()
                pos = getattr(batch, "pos", None)
                if spatial_unit_um is not None and pos is not None:
                    depth = _hidden_depth(pos, hidden.all(dim=1), batch.batch) * float(spatial_unit_um)
                    chunks["depth_um"].append(depth[idx].cpu().numpy())
                if references is not None:
                    if pos is None:
                        raise ValueError("references average over neighbours, but the graphs carry no `pos`.")
                    if not torch.equal(hidden[idx], entry_mask.bool()):
                        raise RuntimeError("the hidden entries the references see differ from the model's.")
                    options = dict(references)
                    label_key = options.pop("label_key", None)
                    labels = label_codes(batch, label_key) if label_key else None
                    refs = reference_predictions(
                        batch.x,
                        hidden,
                        pos,
                        batch.batch,
                        spatial_unit_um=float(spatial_unit_um),
                        labels=labels,
                        **options,
                    )
                    for name, value in refs.items():
                        heads.setdefault(name, []).append(value[idx].cpu().numpy())
                if fitted_references is not None:
                    for name, value in fitted_references(batch, hidden).items():
                        heads.setdefault(name, []).append(value[idx].cpu().numpy())
    finally:
        module.train(was_training)

    def cat(name):
        return np.concatenate(chunks[name]) if chunks[name] else None

    return HeadPredictions(
        heads={name: np.concatenate(parts) for name, parts in heads.items()},
        true=cat("true"),
        hidden=cat("hidden"),
        depth_um=cat("depth_um"),
        cells=cat("cells"),
        parts={
            head: {part: np.concatenate(values) for part, values in named.items()}
            for head, named in part_chunks.items()
        }
        or None,
    )


def blend_heads(pred: HeadPredictions, reference: str, *, weight: float = 0.5) -> HeadPredictions:
    """``pred`` with each model head also blended with a reference: ``blend_<head>``.

    ``blend_<head> = (1 - weight) * head + weight * reference``, entry by entry. A blend that scores
    above both the head and the reference means the head holds something the reference lacks, and
    the other way round; a blend between the two means one only adds noise to the other. A plain
    average: under the centred cosine the one whose deviations from the centre are larger counts
    for more.

    Parameters
    ----------
    pred
        From :func:`collect_head_predictions`, with ``reference`` among its heads.
    reference
        A reference's head name, e.g. ``"ref_kernel_400um"``.
    weight
        The reference's share, in [0, 1].
    """
    if reference not in pred.heads:
        raise ValueError(f"no head {reference!r} to blend with; the heads are {sorted(pred.heads)}.")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight}.")
    models = [h for h in pred.heads if not h.startswith((REFERENCE_PREFIX, BLEND_PREFIX))]
    ref = pred.heads[reference]
    blends = {f"{BLEND_PREFIX}{h}": (1.0 - weight) * pred.heads[h] + weight * ref for h in models}
    return HeadPredictions(
        heads=pred.heads | blends,
        true=pred.true,
        hidden=pred.hidden,
        depth_um=pred.depth_um,
        cells=pred.cells,
        parts=pred.parts,
    )


#: The columns of :func:`attribution_shares`.
SHARE_COLUMNS = ["head", "part", "group", "depth_um", "n_hidden", "norm_ratio", "cov_share"]


def _centred_parts(parts: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """Each part with its mean over the kept entries of every gene taken out, and everything outside ``keep`` zeroed."""
    count = np.maximum(keep.sum(axis=0), 1)
    out = {}
    for name, part in parts.items():
        mean = (part * keep).sum(axis=0) / count
        out[name] = (part - mean) * keep
    return out


def attribution_shares(
    pred: HeadPredictions,
    genes: list[str],
    *,
    gene_groups: dict[str, list[str]] | None = None,
    depth_bins_um: list[float] | None = None,
) -> pd.DataFrame:
    """How much of a head's variation across the hidden cells each of its parts supplies.

    A head whose prediction is the sum of named parts (``pred.parts``: a global-only model's transformer read-out and its
    own-cell skip; the local and the global part of a residual dual decoder; later the two parts of an additive decoder)
    is taken apart. Over the hidden entries of a group of genes and a set of cells, each part is centred by its mean per
    gene over those entries, so the centred parts add up to the centred prediction ``w``. Then, per part ``k``:

    * ``norm_ratio`` -- ``||c_k|| / ||w||``: the size of the part against the whole. Parts that cancel make the ratios
      sum to more than 1.
    * ``cov_share`` -- ``<c_k, w> / ||w||^2``: the part's share of the whole's variance. It sums to 1 over the parts, and is
      negative for a part that works against the others.

    Groups: ``all`` (every gene), then each of ``gene_groups`` (e.g. the planted programmes); cells: ``all`` (every cell with a
    hidden entry), then each depth bin of the wholly hidden cells (``depth_bins_um``; needs ``pred.depth_um``). A separated
    model would give the global part the deep field and the long-range programmes, the local part the rim.

    Returns
    -------
    pandas.DataFrame
        :data:`SHARE_COLUMNS`, one row per head with parts, part, group and depth; empty when no head has parts.
    """
    if not pred.parts:
        return pd.DataFrame(columns=SHARE_COLUMNS)
    hidden = pred.hidden.astype(bool)
    scored = hidden.any(axis=1)
    position = {g: i for i, g in enumerate(genes)}
    groups = {"all": list(range(hidden.shape[1]))}
    for group, members in (gene_groups or {}).items():
        if cols := [position[g] for g in members if g in position]:
            groups[group] = cols
    cell_sets = [("all", scored)]
    if depth_bins_um is not None:
        if pred.depth_um is None:
            raise ValueError("depth_bins_um needs the depth of each cell; collect with spatial_unit_um and `pos`.")
        bins = pd.cut(pd.Series(pred.depth_um), bins=list(depth_bins_um), right=True)
        for interval in bins.cat.categories:
            in_bin = scored & (pred.depth_um > 0) & (bins == interval).to_numpy()
            if in_bin.any():
                cell_sets.append((str(interval), in_bin))
    rows = []
    for head, named in pred.parts.items():
        parts = {name: value.astype(np.float64) for name, value in named.items()}
        for group, cols in groups.items():
            for depth, cells in cell_sets:
                keep = np.zeros_like(hidden)
                keep[:, cols] = hidden[:, cols]
                keep[~cells] = False
                if not keep.any():
                    continue
                centred = _centred_parts(parts, keep)
                whole = sum(centred.values())
                size = float((whole**2).sum())
                if size <= 0:
                    continue
                for name, c in centred.items():
                    rows.append(
                        {
                            "head": head,
                            "part": name,
                            "group": group,
                            "depth_um": depth,
                            "n_hidden": int(keep.sum()),
                            "norm_ratio": float(np.sqrt((c**2).sum() / size)),
                            "cov_share": float((c * whole).sum() / size),
                        }
                    )
    return pd.DataFrame(rows, columns=SHARE_COLUMNS)


def group_receivers(
    pred: HeadPredictions,
    edges: pd.DataFrame,
    edge_genes,
    gene_groups: dict[str, list[str]],
    genes: list[str],
    *,
    offset: int = 1,
) -> dict[str, np.ndarray]:
    """``{group: [N] bool}``: which rows of ``pred`` are receivers of the programme a gene group holds.

    A group stands for a programme when the genes of it that are hidden (a group also holds the
    programme's visible source and receptor genes) are exactly the response genes of at least one
    planted programme (``edge_genes(program)``, the columns, as for
    :func:`~interscale.evaluation.receptive_field.response_influence`); its receivers are the receivers
    of all such programmes. A group of other genes -- a role pooled over programmes, noise -- is left
    out, as is one with no planted receiver among the rows. The rows are found again through
    ``pred.cells``.

    Parameters
    ----------
    pred
        From :func:`collect_head_predictions`, which fills ``cells``.
    edges
        Planted ``receiver`` and ``program`` pairs by position in the AnnData the graphs were built from.
    edge_genes
        ``program -> [columns]``.
    gene_groups, genes
        As for :func:`score_heads`.
    offset
        What the graphs add to a cell's position to name it (``tl.geome_utils.prepare_geome_dataset``).
    """
    if pred.cells is None:
        raise ValueError("the predictions carry no cell names; collect them from graphs with `obs_names`.")
    position = np.rint(pred.cells).astype(np.int64) - int(offset)
    index = {g: k for k, g in enumerate(genes)}
    hidden = set(np.flatnonzero(pred.hidden.any(axis=0)).tolist())
    programmes = {str(p): sorted(int(c) for c in edge_genes(str(p))) for p in edges["program"].unique()}
    out = {}
    for group, members in gene_groups.items():
        columns = sorted(index[g] for g in members if g in index and index[g] in hidden)
        found = [p for p, response in programmes.items() if response and response == columns]
        if not found:
            continue
        receivers = edges.loc[edges["program"].astype(str).isin(found), "receiver"].to_numpy().astype(np.int64)
        rows = np.isin(position, receivers)
        if rows.any():
            out[group] = rows
    return out


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    """``1 - SSE / SST`` around the mean of ``true``; NaN with fewer than two values or no spread."""
    if len(true) < 2:
        return float("nan")
    sst = float(((true - true.mean()) ** 2).sum())
    return 1.0 - float(((pred - true) ** 2).sum()) / sst if sst > 0 else float("nan")


def _pearson(pred: np.ndarray, true: np.ndarray) -> float:
    if len(true) < 2 or pred.std() == 0 or true.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


#: What :func:`score_heads` can score per gene.
GENE_METRICS = ("r2", "pearson", "cosine")


def score_heads(
    pred: HeadPredictions,
    genes: list[str],
    *,
    center: np.ndarray | None = None,
    gene_groups: dict[str, list[str]] | None = None,
    depth_bins_um: list[float] | None = None,
    gene_metrics: tuple[str, ...] = ("r2", "pearson"),
    group_cells: dict[str, np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    """Score every head of one pass -- the model's and the references -- on its hidden entries.

    Parameters
    ----------
    pred
        From :func:`collect_head_predictions`.
    genes
        Gene names, in the column order of ``pred``.
    center
        ``[G]`` the training mean of each gene (``train._utils.train_gene_means``). The centred
        cosine is the cosine of the deviations from it: predicting the mean scores 0. Without it,
        the centred columns are left out.
    gene_groups
        ``{group: [genes]}``, e.g. ``var['program']``, for the per-group table.
    depth_bins_um
        Bin edges for the depth table, in µm, e.g. ``[0, 100, 300, 500, inf]``. Needs
        ``pred.depth_um``; only wholly hidden cells (depth > 0) are binned.
    gene_metrics
        What the per-gene table holds, from :data:`GENE_METRICS`: ``r2``, ``pearson`` (Pearson r),
        and ``cosine``, the centred cosine pooled over the gene's hidden entries (needs ``center``).
    group_cells
        ``{group: [N] bool}`` the rows (cells) a group is also scored on alone, e.g. the receivers
        of a programme (:func:`group_receivers`), for the ``"by_group_cells"`` table. A group not
        in ``gene_groups`` is an error.

    Returns
    -------
    dict
        One column per head and metric, named ``<head>_<metric>``. ``"cells"``: over the cells
        with a hidden entry, the mean SCE (1 − cosine of the hidden entries, as
        ``val_masked_scaled_cosine_error`` at gamma 1) and the mean centred cosine. ``"per_gene"``:
        per gene, its hidden entries' count and each head's ``gene_metrics``. ``"by_group"`` (with
        ``gene_groups``): per group, the cosine pooled over all its hidden entries, and with ``r2``
        among the ``gene_metrics`` the mean of its genes' R². ``"by_depth"`` (with
        ``depth_bins_um``): the ``"cells"`` columns per depth bin. ``"by_group_depth"`` (with both):
        per group and depth (``"all"``, then each bin of wholly hidden cells), the hidden entries'
        count and each head's pooled centred cosine (``<head>_cosine``). ``"by_group_cells"`` (with
        ``group_cells``): per group, the pooled cosine over its hidden entries in the flagged cells
        alone, with their count (``n_cells``): the response where there is one to predict, apart from
        telling which cells respond (``"by_group"`` scores both at once). With both a ``local`` and a
        ``global`` head, each table but ``"by_group_depth"`` also carries ``global - local`` columns:
        positive is the global head better for R², r and cosine, negative for SCE.
    """
    unknown = sorted(set(gene_metrics) - set(GENE_METRICS))
    if unknown:
        raise ValueError(f"gene_metrics must come from {GENE_METRICS}, got {unknown}.")
    if "cosine" in gene_metrics and center is None:
        raise ValueError("the per-gene cosine is centred on the training mean; pass center.")
    heads = {name: value.astype(np.float64) for name, value in pred.heads.items()}
    both = "local" in heads and "global" in heads
    true, hidden = pred.true.astype(np.float64), pred.hidden.astype(bool)
    scored = hidden.any(axis=1)
    tables = {}

    def cell_columns(rows: np.ndarray) -> dict[str, float]:
        out = {"n_cells": int(rows.sum())}
        for name, p in heads.items():
            out[f"{name}_sce"] = float(np.nanmean(1.0 - _cosine_rows(p[rows], true[rows], hidden[rows])))
            if center is not None:
                cos = _cosine_rows(p[rows] - center, true[rows] - center, hidden[rows])
                out[f"{name}_cosine_centred"] = float(np.nanmean(cos))
        if both:
            out["global - local_sce"] = out["global_sce"] - out["local_sce"]
            if center is not None:
                out["global - local_cosine_centred"] = out["global_cosine_centred"] - out["local_cosine_centred"]
        return out

    tables["cells"] = pd.DataFrame([cell_columns(scored)])

    rows = []
    for j, gene in enumerate(genes):
        keep = hidden[:, j]
        row = {"gene": gene, "n_hidden": int(keep.sum())}
        for name, p in heads.items():
            if "r2" in gene_metrics:
                row[f"{name}_r2"] = _r2(p[keep, j], true[keep, j])
            if "pearson" in gene_metrics:
                row[f"{name}_pearson"] = _pearson(p[keep, j], true[keep, j])
            if "cosine" in gene_metrics:
                row[f"{name}_cosine"] = _pooled_cosine(p[:, [j]] - center[j], true[:, [j]] - center[j], hidden[:, [j]])
        if both:
            for metric in gene_metrics:
                row[f"global - local_{metric}"] = row[f"global_{metric}"] - row[f"local_{metric}"]
        rows.append(row)
    per_gene = pd.DataFrame(rows)
    tables["per_gene"] = per_gene

    group_columns = {}
    if gene_groups:
        position = {g: i for i, g in enumerate(genes)}
        group_columns = {
            group: cols
            for group, members in gene_groups.items()
            if (cols := [position[g] for g in members if g in position])
        }
    offset = center if center is not None else 0.0
    if group_columns:
        rows = []
        for group, cols in group_columns.items():
            keep = np.zeros_like(hidden)
            keep[:, cols] = hidden[:, cols]
            in_group = per_gene["gene"].isin([genes[c] for c in cols])
            row = {"group": group, "n_genes": len(cols), "n_hidden": int(keep.sum())}
            for name, p in heads.items():
                if "r2" in gene_metrics:
                    row[f"{name}_r2"] = float(per_gene.loc[in_group, f"{name}_r2"].mean())
                row[f"{name}_cosine"] = _pooled_cosine(p - offset, true - offset, keep)
            if both:
                if "r2" in gene_metrics:
                    row["global - local_r2"] = row["global_r2"] - row["local_r2"]
                row["global - local_cosine"] = row["global_cosine"] - row["local_cosine"]
            rows.append(row)
        tables["by_group"] = pd.DataFrame(rows)

    if group_cells:
        unknown = sorted(set(group_cells) - set(group_columns))
        if unknown:
            raise ValueError(f"group_cells names groups that gene_groups lacks: {unknown}.")
        rows = []
        for group, cells in group_cells.items():
            cells = np.asarray(cells, dtype=bool)
            if cells.shape != (len(true),):
                raise ValueError(
                    f"group_cells[{group!r}] has shape {cells.shape}; the predictions have {len(true)} cells."
                )
            keep = np.zeros_like(hidden)
            keep[:, group_columns[group]] = hidden[:, group_columns[group]]
            keep[~cells] = False
            if not keep.any():
                continue
            row = {"group": group, "n_cells": int(keep.any(axis=1).sum()), "n_hidden": int(keep.sum())}
            for name, p in heads.items():
                row[f"{name}_cosine"] = _pooled_cosine(p - offset, true - offset, keep)
            if both:
                row["global - local_cosine"] = row["global_cosine"] - row["local_cosine"]
            rows.append(row)
        tables["by_group_cells"] = pd.DataFrame(rows)

    if depth_bins_um is not None:
        if pred.depth_um is None:
            raise ValueError("depth_bins_um needs the depth of each cell; collect with spatial_unit_um and `pos`.")
        deep = scored & (pred.depth_um > 0)
        bins = pd.cut(pd.Series(pred.depth_um), bins=list(depth_bins_um), right=True)
        rows = []
        depth_rows = [("all", np.ones(len(true), bool))]
        for interval in bins.cat.categories:
            in_bin = deep & (bins == interval).to_numpy()
            if in_bin.any():
                rows.append({"depth_um": str(interval), **cell_columns(in_bin)})
                depth_rows.append((str(interval), in_bin))
        tables["by_depth"] = pd.DataFrame(rows)

        if group_columns:
            # The groups' pooled cosine per depth bin: whether the genes that carry the context
            # (receivers, responses) are filled in deep inside the hidden area, apart from noise.
            rows = []
            for group, cols in group_columns.items():
                for depth, cells in depth_rows:
                    keep = np.zeros_like(hidden)
                    keep[:, cols] = hidden[:, cols]
                    keep[~cells] = False
                    if not keep.any():
                        continue
                    row = {"group": group, "depth_um": depth, "n_genes": len(cols), "n_hidden": int(keep.sum())}
                    for name, p in heads.items():
                        row[f"{name}_cosine"] = _pooled_cosine(p - offset, true - offset, keep)
                    rows.append(row)
            tables["by_group_depth"] = pd.DataFrame(rows)
    return tables
