"""What the decoder reads off each embedding dimension (roadmap task 2, ED0-ED2; the global-only form designed 2026-10-31).

Reconstruction scores say how well a model predicts, not what it learned. Here each dimension of the embedding a linear decoder
reads is taken apart -- for a GlobalModel (also hierarchical) its global tokens, for a LocalModel its local embedding -- read on a
**masked pass, in the hidden cells**: what the decoder was trained on, where a cell's embedding knows only its context.

* **ED0** :func:`collect_dimensions` -- the embedding ``Z`` of every cell the pass scored, the truth, the hidden entries, the
  decoder's ``W`` and ``b``, graph and position. With ``folds`` every cell is hidden once (each fold in turn, the graphs' own
  masks put back), so every dimension has a value everywhere -- the discs' hidden cells alone are patches.
* **ED1** :func:`dimension_importance` -- per dimension, three readings that should agree on the top: the standardised loading
  ``||W[:, k]|| std(z_k) / std(x)``, the share of the prediction's variance (``<c_k, c> / ||c||^2``, ``c_k = W[:, k] z_k`` centred
  per gene over the hidden entries), and the loss in centred cosine when ``z_k`` is set to its mean. :func:`effective_directions`:
  the SVD of ``W Σ_z^{1/2}`` -- a linear decoder fixes its dimensions only up to a rotation, the singular directions not (with
  uncorrelated dimensions this is ``W diag(std z)``).
* **ED2** :func:`dimension_genes` -- per dimension, the genes it **writes** (top standardised loadings, both signs) and the genes
  it **tracks** (Spearman over the hidden cells between ``z_k`` and the *true* expression no input of those cells held), and
  their overlap; :func:`track_summary` puts that next to an untrained twin's (a dimension that tracks genes as well untrained only
  copies the input).

Everything after the collector is a function of arrays, dataset-agnostic: gene names, the centre and the twin come from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation.gene_scales import _hide_fold, _masks_kept, cell_folds
from interscale.evaluation.online_probes import _rng_untouched
from interscale.module.base._step_output import gather_tokens


@dataclass
class DimensionReadout:
    """One pass's embedding and decoder, by cell (rows) and gene (columns).

    ``z`` ``[N, E]`` the embedding the decoder reads; ``true`` ``[N, G]``; ``hidden`` ``[N, G]`` the entries the loss scored;
    ``weight`` ``[G, E]`` and ``bias`` ``[G]`` the linear decoder; ``graph`` ``[N]`` each row's graph in the list; ``pos`` ``[N, 2]``
    (the graphs' units) or None; ``cells`` ``[N]`` the graphs' ``obs_names`` or None.
    """

    z: np.ndarray
    true: np.ndarray
    hidden: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    graph: np.ndarray
    pos: np.ndarray | None = None
    cells: np.ndarray | None = None


def decoder_weights(module) -> tuple[np.ndarray, np.ndarray]:
    """``(W [G, E], b [G])`` of a module's linear decoder; a nonlinear (or LSE) decoder has no dimension-wise reading."""
    from interscale.nn import LinearDecoder

    decoder = getattr(module, "decoder", None)
    if not isinstance(decoder, LinearDecoder):
        raise ValueError(
            f"the dimensions are read through a linear decoder; this module's is {type(decoder).__name__}. Use "
            "model.decoder.type linear (a dual model: pass the component module whose decoder you read)."
        )
    layer = decoder.decoder
    return layer.weight.detach().cpu().double().numpy(), layer.bias.detach().cpu().double().numpy()


def _embedding_rows(module, out) -> torch.Tensor:
    """The rows the decoder read in this step, in ``y_pred`` order: a global module's tokens, a local module's embedding."""
    view = out.view
    if view.global_embedding is not None and getattr(module, "registered_global_component", False):
        return gather_tokens(view.global_embedding, view.src_padding_mask)
    if view.local_embedding is not None:
        return view.local_embedding
    raise ValueError("this step carries no embedding the decoder reads.")


def _one_pass(module, data_list, *, batch_size, device, offset_graph=0):
    chunks = {k: [] for k in ("z", "true", "hidden", "graph", "pos", "cells")}
    loader = DataLoader(dataset=data_list, shuffle=False, batch_size=batch_size, num_workers=0)
    start = 0
    for batch in loader:
        batch = batch.to(device)
        out = module._common_step(batch, "regression", "node")
        z = _embedding_rows(module, out)
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(batch.num_nodes, device=batch.x.device)
        if len(z) != len(out.y_pred):
            raise RuntimeError("the embedding rows and the predictions differ in number.")
        chunks["z"].append(z.detach().cpu().double().numpy())
        chunks["true"].append(out.y_true.detach().cpu().double().numpy())
        chunks["hidden"].append(out.entry_mask.bool().cpu().numpy())
        chunks["graph"].append((batch.batch[idx] + start + offset_graph).cpu().numpy())
        pos = getattr(batch, "pos", None)
        chunks["pos"].append(None if pos is None else pos[idx, :2].detach().cpu().double().numpy())
        names = getattr(batch, "obs_names", None)
        chunks["cells"].append(names[idx].double().cpu().numpy() if isinstance(names, torch.Tensor) else None)
        start += int(batch.num_graphs)
    return chunks


def collect_dimensions(module, data_list, *, batch_size: int, device, folds: int | None = None, seed: int = 0):
    """ED0: :class:`DimensionReadout` of ``module`` over ``data_list``, read on masked passes in eval mode.

    Without ``folds`` the graphs keep the masks they carry (e.g. the evaluation's discs, :func:`masked_probing.draw_masks`), and
    every row the step scored is kept. With ``folds`` (>= 2) every cell is hidden whole once, fold by fold
    (:func:`gene_scales.cell_folds`), and only the rows of each fold's hidden cells are kept; the graphs' own masks are put back.
    Draws nothing from the global generators.
    """
    weight, bias = decoder_weights(module)
    was_training = module.training
    module.eval()
    passes = []
    try:
        with _rng_untouched(device), torch.no_grad():
            if folds is None:
                passes.append((_one_pass(module, data_list, batch_size=batch_size, device=device), None))
            else:
                assigned = cell_folds(data_list, int(folds), seed)
                with _masks_kept(data_list):
                    for fold in range(int(folds)):
                        _hide_fold(data_list, assigned, fold)
                        passes.append((_one_pass(module, data_list, batch_size=batch_size, device=device), fold))
    finally:
        module.train(was_training)

    rows = {k: [] for k in ("z", "true", "hidden", "graph", "pos", "cells")}
    for chunks, fold in passes:
        for i in range(len(chunks["z"])):
            keep = chunks["hidden"][i].all(axis=1) if fold is not None else np.ones(len(chunks["z"][i]), bool)
            for key in rows:
                value = chunks[key][i]
                rows[key].append(None if value is None else value[keep])

    def cat(key):
        parts = rows[key]
        return None if any(p is None for p in parts) else np.concatenate(parts)

    return DimensionReadout(
        z=cat("z"),
        true=cat("true"),
        hidden=cat("hidden"),
        weight=weight,
        bias=bias,
        graph=cat("graph"),
        pos=cat("pos"),
        cells=cat("cells"),
    )


# --------------------------------------------------------------------------- ED1


def _kept_rows(r: DimensionReadout) -> np.ndarray:
    rows = r.hidden.any(axis=1)
    if not rows.any():
        raise ValueError("no hidden entry: the dimensions are read on the cells the decoder was trained on.")
    return rows


def standardised_loadings(r: DimensionReadout) -> np.ndarray:
    """``[G, E]``: ``S[g, k] = W[g, k] std(z_k) / std(x_g)``, the spreads over the hidden cells (a gene with no spread: 0)."""
    rows = _kept_rows(r)
    std_z = r.z[rows].std(axis=0)
    x = np.where(r.hidden[rows], r.true[rows], np.nan)
    std_x = np.nanstd(x, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(std_x > 0, 1.0 / std_x, 0.0)
    return r.weight * std_z[None, :] * np.nan_to_num(scale)[:, None]


def _centred_cosine(pred: np.ndarray, true: np.ndarray, keep: np.ndarray, center: np.ndarray) -> float:
    a, b = np.where(keep, pred - center, 0.0), np.where(keep, true - center, 0.0)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(norm > 0, (a * b).sum(1) / norm, np.nan)))


def dimension_importance(r: DimensionReadout, *, center: np.ndarray) -> pd.DataFrame:
    """ED1: per dimension, its standardised loading, its share of the prediction's variance, and the cost of ablating it.

    Over the cells with a hidden entry and their hidden entries: ``loading`` ``||S[:, k]||``; ``share`` ``<c_k, c> / ||c||^2``, the
    parts ``c_k = W[:, k] z_k`` centred per gene over the hidden entries and ``c`` their sum (sums to 1; negative for a dimension
    that works against the others); ``ablation`` the mean centred cosine (around ``center``, the training gene means) of the
    decoder's prediction minus the same with ``z_k`` set to its mean -- positive: the prediction needs the dimension. Sorted by
    share, with ``cumulative_share`` for an elbow.
    """
    rows = _kept_rows(r)
    z, keep, true = r.z[rows], r.hidden[rows], r.true[rows]
    center = np.asarray(center, dtype=np.float64)
    count = np.maximum(keep.sum(axis=0), 1)
    parts = []
    for k in range(z.shape[1]):
        part = np.outer(z[:, k], r.weight[:, k])
        part = (part - (part * keep).sum(0) / count) * keep
        parts.append(part)
    whole = sum(parts)
    size = float((whole**2).sum())
    loadings = np.linalg.norm(standardised_loadings(r), axis=0)
    full = _centred_cosine(z @ r.weight.T + r.bias, true, keep, center)
    out = []
    for k in range(z.shape[1]):
        ablated = z.copy()
        ablated[:, k] = z[:, k].mean()
        out.append(
            {
                "dim": k,
                "loading": float(loadings[k]),
                "share": float((parts[k] * whole).sum() / size) if size > 0 else float("nan"),
                "ablation": full - _centred_cosine(ablated @ r.weight.T + r.bias, true, keep, center),
            }
        )
    table = pd.DataFrame(out).sort_values("share", ascending=False).reset_index(drop=True)
    table["cumulative_share"] = table["share"].cumsum()
    return table


def effective_directions(r: DimensionReadout, genes: list[str], *, top: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rotation-free view of ED1/ED2: the SVD of ``W Σ_z^{1/2}``, ``Σ_z`` the covariance of ``z`` over the hidden cells.

    Returns ``(directions, genes)``: per singular direction its singular value and share of the squared sum; per direction the
    ``top`` genes by the absolute loading, with their signed loading. Rotating ``z`` by any orthogonal ``R`` (and ``W`` by ``R^T``)
    changes the single dimensions but none of this.
    """
    u, s = _effective_basis(r)
    return _direction_tables(u, s, genes, top=top)


def _effective_basis(r: DimensionReadout) -> tuple[np.ndarray, np.ndarray]:
    """The sign-fixed left singular vectors ``[G, r]`` and singular values of ``W Σ_z^{1/2}``."""
    rows = _kept_rows(r)
    centred = r.z[rows] - r.z[rows].mean(axis=0)
    # The covariance's square root (not diag(std z)) keeps the result rotation-free when the dimensions are correlated.
    cov = centred.T @ centred / max(len(centred) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    root = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T
    u, s, _ = np.linalg.svd(r.weight @ root, full_matrices=False)
    return _signed(u), s


def _signed(u: np.ndarray) -> np.ndarray:
    """``u`` with each column's sign fixed so its largest entry by size is positive (an SVD's signs are arbitrary; this makes
    two runs, or a model and a reference, comparable sign for sign)."""
    if u.size == 0:
        return u
    peak = np.argmax(np.abs(u), axis=0)
    sign = np.sign(u[peak, np.arange(u.shape[1])])
    return u * np.where(sign == 0, 1.0, sign)


def _direction_tables(u: np.ndarray, s: np.ndarray, genes: list[str], *, top: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    directions = pd.DataFrame({"direction": np.arange(len(s)), "singular_value": s, "share": s**2 / max((s**2).sum(), 1e-300)})
    rows_out = []
    for d in range(len(s)):
        order = np.argsort(-np.abs(u[:, d]), kind="stable")[: int(top)]
        for rank, g in enumerate(order):
            rows_out.append({"direction": d, "rank": rank, "gene": genes[g], "loading": float(u[g, d] * s[d])})
    return directions, pd.DataFrame(rows_out, columns=["direction", "rank", "gene", "loading"])


def _spread(x: np.ndarray) -> np.ndarray:
    """Per column, the standard deviation (ddof 1) of the finite entries; 0 where fewer than two."""
    finite = np.isfinite(x)
    count = finite.sum(axis=0)
    filled = np.where(finite, x, 0.0)
    mean = filled.sum(axis=0) / np.maximum(count, 1)
    squares = (np.where(finite, x - mean, 0.0) ** 2).sum(axis=0)
    return np.where(count > 1, np.sqrt(squares / np.maximum(count - 1, 1)), 0.0)


def _priority_table(variance: np.ndarray, std_true: np.ndarray, genes: list[str]) -> pd.DataFrame:
    """Every gene by how much a prediction moves it: ``share`` of the summed variance, ``std_ratio`` its spread over the truth's."""
    variance = np.clip(np.asarray(variance, dtype=np.float64), 0.0, None)
    total = float(variance.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std_true > 0, np.sqrt(variance) / np.where(std_true > 0, std_true, 1.0), np.nan)
    table = pd.DataFrame(
        {
            "gene": list(genes),
            "share": variance / total if total > 0 else np.full(len(variance), np.nan),
            "variance": variance,
            "std_ratio": ratio,
        }
    )
    table = table.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.insert(3, "cumulative_share", table["share"].cumsum())
    table["rank_std_ratio"] = table["std_ratio"].rank(ascending=False, method="min")
    return table


def gene_priority(r: DimensionReadout, genes: list[str]) -> pd.DataFrame:
    """ED2, every gene: how much the decoder's prediction moves it over the hidden cells.

    ``variance`` is the gene's squared row norm of ``W Σ_z^{1/2}`` -- its weight summed over *every* singular direction of
    :func:`effective_directions`, not only the listed top -- which is the variance of the context part ``W z`` of its prediction
    over the rows :func:`dimension_importance` reads; ``share`` its part of the sum over genes (sorted by it, ``rank`` 1 first,
    ``cumulative_share`` for an elbow). In expression units, so abundant genes lead; ``std_ratio`` is the gene's prediction
    spread over the truth's spread on its hidden entries (``rank_std_ratio``: the ranking per gene's own scale; NaN for a gene
    without spread there).
    """
    rows = _kept_rows(r)
    centred = r.z[rows] - r.z[rows].mean(axis=0)
    cov = centred.T @ centred / max(len(centred) - 1, 1)
    variance = np.einsum("gk,kl,gl->g", r.weight, cov, r.weight)
    truth = np.where(r.hidden[rows], r.true[rows], np.nan)
    return _priority_table(variance, _spread(truth), genes)


# --------------------------------------------------------------------------- the same readout on any prediction


def _centre_rows(values: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Each column minus its mean over the finite entries, within each group (all rows one group when ``groups`` is None)."""
    out = np.array(values, dtype=np.float64)
    labels = np.zeros(len(out), dtype=np.int64) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
    for label in np.unique(labels):
        here = labels == label
        block = out[here]
        finite = np.isfinite(block)
        mean = np.where(finite, block, 0.0).sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
        out[here] = block - mean
    return out


def prediction_readout(
    pred: np.ndarray,
    hidden: np.ndarray,
    genes: list[str],
    *,
    true: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    top: int = 10,
) -> dict:
    """:func:`effective_directions` and :func:`gene_priority` for any ``[N, G]`` prediction: a model head (whatever its decoder)
    or a reference (``ref_nearest_<k>``), so a model's directions can be read against a smoother's.

    Over the rows with a hidden entry, each gene centred (within each of ``groups``, e.g. the section: then the readout is
    within sections, as ``masked_probing.within_section`` scores); the SVD of the centred rows over ``sqrt(n - 1)``. For a linear
    decoder's ``W z + b`` without groups this equals :func:`effective_directions` and :func:`gene_priority` on its readout
    (``cov(W z) = W Σ_z W^T``). ``true`` gives ``std_ratio`` (the truth's spread on the hidden entries, centred the same way).

    Returns ``{"directions", "genes", "priority", "basis", "singular_values"}``: the two tables of :func:`effective_directions`,
    the table of :func:`gene_priority`, and the ``[G, r]`` sign-fixed directions with their singular values.
    """
    pred, hidden = np.asarray(pred, dtype=np.float64), np.asarray(hidden, dtype=bool)
    rows = hidden.any(axis=1)
    if not rows.any():
        raise ValueError("no hidden entry: the readout is on the cells the model had to predict.")
    group_rows = None if groups is None else np.asarray(groups)[rows]
    centred = _centre_rows(pred[rows], group_rows)
    n = len(centred)
    _, s, vt = np.linalg.svd(centred / np.sqrt(max(n - 1, 1)), full_matrices=False)
    u = _signed(vt.T)
    directions, gene_rows = _direction_tables(u, s, genes, top=top)
    if true is None:
        std_true = np.zeros(len(genes))
    else:
        truth = np.where(hidden[rows], np.asarray(true, dtype=np.float64)[rows], np.nan)
        std_true = _spread(_centre_rows(truth, group_rows))
    priority = _priority_table((centred**2).sum(axis=0) / max(n - 1, 1), std_true, genes)
    return {"directions": directions, "genes": gene_rows, "priority": priority, "basis": u, "singular_values": s}


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """The mean squared cosine of the principal angles between the first ``k`` columns of two orthonormal bases: 1 the same
    subspace, 0 orthogonal (``k`` is capped at the smaller basis)."""
    k = int(min(k, a.shape[1], b.shape[1]))
    if k < 1:
        return float("nan")
    return float((np.linalg.norm(a[:, :k].T @ b[:, :k]) ** 2) / k)


def varimax(loadings: np.ndarray, *, kaiser: bool = False, max_iter: int = 1000, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Kaiser's varimax: the orthogonal ``R`` ``[k, k]`` that makes the columns of ``loadings @ R`` as sparse as it can (the
    variance of the squared loadings per column, summed, at its maximum). Returns ``(loadings @ R, R)``.

    The rotated columns span the same subspace and keep the summed squares (each gene's priority over the ``k`` columns);
    only how that is split into columns changes. ``kaiser`` scales every gene to unit length first (R's ``varimax`` default):
    off here, because then a gene the prediction hardly moves -- a negative probe -- weighs as much as a marker.
    """
    loadings = np.asarray(loadings, dtype=np.float64)
    p, k = loadings.shape
    if k < 2:
        return loadings.copy(), np.eye(k)
    norms = np.sqrt((loadings**2).sum(axis=1, keepdims=True)) if kaiser else np.ones((p, 1))
    x = loadings / np.where(norms > 0, norms, 1.0)
    rotation = np.eye(k)
    criterion = 0.0
    for _ in range(int(max_iter)):
        lam = x @ rotation
        u, s, vt = np.linalg.svd(x.T @ (lam**3 - lam @ np.diag((lam**2).sum(axis=0)) / p))
        rotation = u @ vt
        previous, criterion = criterion, float(s.sum())
        if previous > 0 and criterion <= previous * (1.0 + tol):
            break
    return loadings @ rotation, rotation


def rotated_tables(
    basis: np.ndarray, singular_values: np.ndarray, genes: list[str], *, k: int, top: int = 10, kaiser: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The first ``k`` directions (loadings ``u * s``) after :func:`varimax`, as ``(components, genes)``.

    ``components``: per rotated component its ``variance`` (the summed squared loadings), its ``share`` of the whole prediction's
    variance and of the ``k`` directions' (``share_of_rotated``); sorted by variance, each sign-fixed (largest gene positive).
    ``genes``: per component the ``top`` genes by absolute loading. Read a component as a gene program; the split depends on
    ``k`` (rotate the directions that carry the signal, e.g. those up to an elbow of ``share``), the subspace does not.
    """
    k = int(min(k, basis.shape[1]))
    total = float((singular_values**2).sum())
    rotated, _ = varimax(basis[:, :k] * singular_values[:k], kaiser=kaiser)
    variance = (rotated**2).sum(axis=0)
    order = np.argsort(-variance, kind="stable")
    rotated, variance = _signed(rotated[:, order]), variance[order]
    components = pd.DataFrame(
        {
            "component": np.arange(k),
            "variance": variance,
            "share": variance / total if total > 0 else np.nan,
            "share_of_rotated": variance / max(float(variance.sum()), 1e-300),
        }
    )
    rows_out = []
    for c in range(k):
        for rank, g in enumerate(np.argsort(-np.abs(rotated[:, c]), kind="stable")[: int(top)]):
            rows_out.append({"component": c, "rank": rank, "gene": genes[g], "loading": float(rotated[g, c])})
    return components, pd.DataFrame(rows_out, columns=["component", "rank", "gene", "loading"])


def prediction_tables(
    heads: dict[str, np.ndarray],
    true: np.ndarray,
    hidden: np.ndarray,
    genes: list[str],
    *,
    groups: np.ndarray | None = None,
    include_truth: bool = True,
    top: int = 10,
    n_directions: int = 8,
    overlap_k: tuple[int, ...] = (1, 3, 8),
    rotation: str = "none",
    rotate_k: int = 8,
    kaiser: bool = False,
) -> dict[str, pd.DataFrame]:
    """:func:`prediction_readout` of every prediction in ``heads`` (and of the truth, ``include_truth``: the directions the data
    itself varies along on the same cells) as report tables:

    * ``pred_directions`` -- per head the first ``n_directions`` singular values, shares, cumulative shares, and how many
      directions carry 99% (``directions_for_99``);
    * ``pred_direction_genes`` -- per head and direction the ``top`` genes with their signed loadings;
    * ``pred_gene_priority`` -- every gene (rows), per head its ``<head>_share`` and ``<head>_std_ratio``, sorted by the first
      head's share;
    * ``pred_overlap`` -- per pair of heads and ``k`` in ``overlap_k``, :func:`subspace_overlap` of their first ``k`` directions:
      does a model move along the axes a smoother (or the data) does?
    * with ``rotation="varimax"``: ``pred_rotated`` and ``pred_rotated_genes``, each head's first ``rotate_k`` directions
      rotated to sparse gene programs (:func:`rotated_tables`; ``kaiser`` as for :func:`varimax`). Each head is rotated on its
      own; ``pred_overlap`` does not change (a rotation keeps the subspace).
    """
    if rotation not in ("none", "varimax"):
        raise ValueError(f"rotation is 'none' or 'varimax', not {rotation!r}.")
    items = list(heads.items()) + ([("truth", true)] if include_truth else [])
    if not items:
        return {}
    direction_tables, gene_tables, bases = [], [], {}
    rotated_components, rotated_genes = [], []
    wide = None
    for name, value in items:
        read = prediction_readout(value, hidden, genes, true=true, groups=groups, top=top)
        d = read["directions"]
        cumulative = d["share"].cumsum().to_numpy()
        d = d.assign(cumulative_share=cumulative, directions_for_99=int(np.searchsorted(cumulative, 0.99) + 1))
        d = d.head(int(n_directions))
        d.insert(0, "head", name)
        direction_tables.append(d)
        g = read["genes"][read["genes"]["direction"] < int(n_directions)].copy()
        g.insert(0, "head", name)
        gene_tables.append(g)
        p = read["priority"].set_index("gene")[["share", "std_ratio"]].add_prefix(f"{name}_")
        wide = p if wide is None else wide.join(p)
        bases[name] = read["basis"]
        if rotation == "varimax":
            comp, comp_genes = rotated_tables(
                read["basis"], read["singular_values"], genes, k=rotate_k, top=top, kaiser=kaiser
            )
            comp.insert(0, "head", name)
            comp_genes.insert(0, "head", name)
            rotated_components.append(comp)
            rotated_genes.append(comp_genes)
    names = list(bases)
    overlap = [
        {"head": a, "against": b, "k": int(k), "overlap": subspace_overlap(bases[a], bases[b], k)}
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        for k in overlap_k
    ]
    first = f"{items[0][0]}_share"
    rotated = {}
    if rotation == "varimax":
        rotated = {
            "pred_rotated": pd.concat(rotated_components, ignore_index=True),
            "pred_rotated_genes": pd.concat(rotated_genes, ignore_index=True),
        }
    return {
        **rotated,
        "pred_directions": pd.concat(direction_tables, ignore_index=True),
        "pred_direction_genes": pd.concat(gene_tables, ignore_index=True),
        "pred_gene_priority": wide.sort_values(first, ascending=False, kind="stable").reset_index(),
        "pred_overlap": pd.DataFrame(overlap, columns=["head", "against", "k", "overlap"]),
    }


# --------------------------------------------------------------------------- ED2


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks along axis 0 (ties share their mean rank)."""
    return np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0, values)


def track_correlations(r: DimensionReadout) -> np.ndarray:
    """``[E, G]``: Spearman's rho of each dimension with each gene's *true* expression over the cells where the gene was hidden.

    NaN where a gene has fewer than 3 hidden cells or no spread there.
    """
    rows = _kept_rows(r)
    z, true, keep = r.z[rows], r.true[rows], r.hidden[rows]
    rho = np.full((z.shape[1], true.shape[1]), np.nan)
    groups: dict[bytes, list[int]] = {}
    for g in range(true.shape[1]):
        groups.setdefault(np.packbits(keep[:, g]).tobytes(), []).append(g)
    for genes in groups.values():
        cells = keep[:, genes[0]]
        if cells.sum() < 3:
            continue
        rz, rx = _ranks(z[cells]), _ranks(true[cells][:, genes])
        rz, rx = rz - rz.mean(0), rx - rx.mean(0)
        denom = np.outer(np.linalg.norm(rz, axis=0), np.linalg.norm(rx, axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho[:, genes] = np.where(denom > 0, rz.T @ rx / denom, np.nan)
    return rho


def dimension_genes(r: DimensionReadout, genes: list[str], *, top: int = 20, dims=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ED2: per dimension the genes it writes and the genes it tracks, and their overlap.

    Returns ``(genes, overlap)``. ``genes``: rows ``dim, kind ("write" | "track"), rank, gene, value`` -- ``write`` the ``top`` genes
    by ``|S[:, k]|`` (:func:`standardised_loadings`, signed), ``track`` the ``top`` by ``|rho|`` (:func:`track_correlations`, signed).
    ``overlap``: per dimension how many genes the two lists share, and their Jaccard index. A dimension that writes the genes it
    tracks is read the way it is written; one that tracks genes it does not write carries them for another reason (a cell type, a
    region). ``dims``: the dimensions to list (default all, e.g. the top of :func:`dimension_importance`).
    """
    loadings, rho = standardised_loadings(r), track_correlations(r)
    out, overlap = [], []
    for k in range(r.z.shape[1]) if dims is None else dims:
        write = np.argsort(-np.abs(loadings[:, k]))[: int(top)]
        track = np.argsort(-np.nan_to_num(np.abs(rho[k]), nan=-1.0))[: int(top)]
        track = [g for g in track if np.isfinite(rho[k, g])]
        for rank, g in enumerate(write):
            out.append({"dim": k, "kind": "write", "rank": rank, "gene": genes[g], "value": float(loadings[g, k])})
        for rank, g in enumerate(track):
            out.append({"dim": k, "kind": "track", "rank": rank, "gene": genes[g], "value": float(rho[k, g])})
        shared = len(set(write.tolist()) & set(track))
        union = len(set(write.tolist()) | set(track))
        overlap.append({"dim": k, "shared": shared, "jaccard": shared / union if union else float("nan")})
    return pd.DataFrame(out), pd.DataFrame(overlap)


def track_summary(r: DimensionReadout, *, twin: DimensionReadout | None = None, top: int = 5) -> pd.DataFrame:
    """Per dimension, the mean ``|rho|`` of its ``top`` tracked genes; with ``twin`` (the same architecture untrained, read the
    same way), the twin's mean of that over its dimensions as ``twin_mean`` and the dimension's ``lift`` over it."""

    def strength(readout):
        rho = np.abs(track_correlations(readout))
        rho = np.where(np.isfinite(rho), rho, -np.inf)
        best = -np.sort(-rho, axis=1)[:, : int(top)]
        return np.where(np.isfinite(best), best, np.nan).mean(axis=1)

    table = pd.DataFrame({"dim": np.arange(r.z.shape[1]), "track_strength": strength(r)})
    if twin is not None:
        base = float(np.nanmean(strength(twin)))
        table["twin_mean"] = base
        table["lift"] = table["track_strength"] - base
    return table


def dimension_tables(
    r: DimensionReadout,
    genes: list[str],
    *,
    center: np.ndarray,
    twin: DimensionReadout | None = None,
    top: int = 20,
    n_dims: int = 8,
    rotation: str = "none",
    rotate_k: int = 8,
    kaiser: bool = False,
) -> dict[str, pd.DataFrame]:
    """ED1 and ED2 as report tables: ``dim_importance``, ``dim_directions``, ``dim_direction_genes``, ``dim_genes`` and
    ``dim_overlap`` (the ``n_dims`` most important dimensions), ``dim_tracks`` (against ``twin`` when given), and
    ``dim_gene_priority`` (every gene, :func:`gene_priority`); with ``rotation="varimax"`` also ``dim_rotated`` and
    ``dim_rotated_genes`` (the first ``rotate_k`` directions of :func:`effective_directions` rotated, :func:`rotated_tables`)."""
    if rotation not in ("none", "varimax"):
        raise ValueError(f"rotation is 'none' or 'varimax', not {rotation!r}.")
    importance = dimension_importance(r, center=center)
    directions, direction_genes = effective_directions(r, genes, top=min(top, 10))
    chosen = importance["dim"].head(int(n_dims)).tolist()
    gene_rows, overlap = dimension_genes(r, genes, top=top, dims=chosen)
    return {
        "dim_importance": importance,
        "dim_directions": directions,
        "dim_direction_genes": direction_genes[direction_genes["direction"] < int(n_dims)],
        "dim_genes": gene_rows,
        "dim_overlap": overlap,
        "dim_tracks": track_summary(r, twin=twin),
        "dim_gene_priority": gene_priority(r, genes),
        **_dimension_rotation(r, genes, rotation=rotation, k=rotate_k, top=min(top, 10), kaiser=kaiser),
    }


def _dimension_rotation(r, genes, *, rotation, k, top, kaiser) -> dict[str, pd.DataFrame]:
    if rotation != "varimax":
        return {}
    u, s = _effective_basis(r)
    components, gene_rows = rotated_tables(u, s, genes, k=k, top=top, kaiser=kaiser)
    return {"dim_rotated": components, "dim_rotated_genes": gene_rows}
