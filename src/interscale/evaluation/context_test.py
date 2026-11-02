"""V2: a per-gene test of whether a model uses the context beyond (or within) a radius (``.claude/yardsticks_plan.md``).

A conditional randomisation test (the holdout randomisation test of Tansey et al. 2022) with the model as a black box: is gene
g's prediction at a hidden cell better with the true context beyond ``r`` than with a context of the same tissue that has no
relation to the cell? Works for any model -- global-only, hierarchical, dual (the reported sum) -- because it changes only the
input.

**Surrogate** (:func:`shift_sources`). Per graph a random translation (at least ``min_shift`` of the graph's extent along each
axis) with wrap-around in the bounding box; every cell takes the *whole row* of the cell nearest to its shifted position: the
``fields`` (``x``, ``mask``, ``gene_mask``, and any other expression-derived per-cell attribute the caller names), so hidden
entries stay hidden wherever they go and a receiver's own hidden values never appear elsewhere. Positions, edges, names and
positional encodings stay where they are. Composition and spatial autocorrelation are kept; the link to the receivers is broken.

**Receivers** (:func:`choose_receivers`). Cells with a hidden entry, in batches: within a batch any two lie more than ``2 r``
apart and the union of their ``r``-zones (the receiver and every cell within ``r``) covers at most ``max_cover`` of the graph.
``region="far"``: the zones keep their true rows, every other cell takes its surrogate row (the other receivers' zones are true
far context for a receiver: at most ``max_cover`` of it, which costs power, not validity). ``region="near"``: the cells within
``r`` of a receiver, the receiver left out, take surrogate rows -- **one receiver per graph and pass** by default
(``per_batch``): another receiver's replaced zone is far context for this one, so with several a gene that reads only far
context would show a near effect (a false positive, not a loss of power).

**Statistic** (:func:`context_test`). One true pass (the graphs as they are) and per receiver batch ``n_shifts`` surrogate
passes. Per gene the mean squared error over the receivers' hidden entries: ``loss_true`` and ``loss_k`` for shift ``k`` (every
batch's ``k``-th surrogate pooled). ``p_rank = (1 + #{loss_k <= loss_true}) / (n_shifts + 1)`` is exact but never below
``1 / (n_shifts + 1)``, which no multiple-testing correction over hundreds of genes can pass; so ``p`` (and ``q``,
Benjamini-Hochberg over genes) is one-sided from ``z = (mean_k loss_k - loss_true) / sd_k``, the surrogate losses taken as
normal (each is a mean over many receivers). A tie (the replaced context does not reach the gene) gives ``p = 1``. Effect
``delta_r2 = (mean_k loss_k - loss_true) / var``, ``var`` the gene's variance within each graph (all its cells), weighted by the graph's scored entries.
:func:`context_labels` puts a far and a near test together: ``global``, ``local``, ``both``, ``none``.

**Caveats.** Shifts are liberal at a graph's edge (the nearest-cell mapping repeats edge cells). A graph the model subsamples
(more cells than ``max_seq_len``) is refused: every receiver must be predicted in every pass. Cost ``1 + n_batches * n_shifts``
passes. No global random generator is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import norm
from torch_geometric.loader import DataLoader

from interscale.evaluation.online_probes import _rng_untouched

#: Which context a test replaces.
REGIONS = ("far", "near")
#: The per-cell attributes that move with a cell's row by default.
DEFAULT_FIELDS = ("x", "mask", "gene_mask")
#: Per-cell attributes that stay at their position: positions and what is computed from positions alone, names, graph-level
#: labels. Anything else per cell must be in ``fields`` (it moves) or ``stay`` (the caller says it is positional).
POSITIONAL = frozenset(
    {
        "pos",
        "obs_names",
        "lap_pe",
        "rw_pe",
        "spectral_pe",
        "spectral_eigval",
        "density_radius",
        "slide",
        "group",
        "condition",
    }
)
#: Relative difference below which a surrogate's loss counts as equal to the true one (float sums in another order).
TIE_TOLERANCE = 1e-9
#: :func:`context_labels`' classes.
LABELS = ("none", "local", "global", "both")


# ------------------------------------------------------------------------------- surrogates and receivers


def shift_sources(pos, *, rng: np.random.Generator, min_shift: float = 0.25) -> np.ndarray:
    """``[n]`` for each cell of one graph the cell whose row it takes: the nearest cell to its position shifted with wrap-around.

    The shift is drawn per axis in ``[min_shift, 1 - min_shift]`` of the period (the extent plus the median nearest-neighbour
    distance, so a lattice wraps onto itself).
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    if not 0.0 <= float(min_shift) < 0.5:
        raise ValueError(f"min_shift must be in [0, 0.5), got {min_shift}.")
    n = len(pos)
    if n < 2:
        return np.arange(n)
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=2)
    spacing = float(np.median(d[:, 1]))
    lo = pos.min(axis=0)
    period = pos.max(axis=0) - lo + spacing
    delta = rng.uniform(float(min_shift), 1.0 - float(min_shift), size=2) * period
    # The nearest cell on the torus, so a point shifted into the gap past the last row finds the first one.
    local = np.mod(pos - lo, period)
    periodic = cKDTree(local, boxsize=period)
    _, source = periodic.query(np.mod(local + delta, period), k=1)
    return np.asarray(source, dtype=np.int64)


def zones(pos, receivers, radius: float) -> list[np.ndarray]:
    """For each receiver the cells within ``radius`` of it (itself included)."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    tree = cKDTree(pos)
    return [np.asarray(sorted(tree.query_ball_point(pos[i], r=float(radius))), dtype=np.int64) for i in receivers]


def choose_receivers(
    pos,
    candidates,
    radius: float,
    *,
    rng: np.random.Generator,
    max_cover: float = 0.1,
    n_batches: int = 1,
    per_batch: int | None = None,
) -> list[np.ndarray]:
    """Up to ``n_batches`` batches of receivers of one graph (no receiver in two batches).

    Each batch is drawn greedily from the unused candidates in a random order: a candidate joins when it lies more than
    ``2 * radius`` from every receiver of the batch, the batch's zones then cover at most ``max_cover`` of the cells (the
    first receiver of a batch always joins) and the batch holds fewer than ``per_batch`` (None: no limit). An empty list when
    there is no candidate.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    candidates = np.flatnonzero(np.asarray(candidates, dtype=bool))
    if not 0.0 < float(max_cover) <= 1.0:
        raise ValueError(f"max_cover must be in (0, 1], got {max_cover}.")
    n = len(pos)
    tree = cKDTree(pos)
    unused = list(rng.permutation(candidates))
    batches = []
    for _ in range(int(n_batches)):
        if not unused:
            break
        chosen, covered, left = [], np.zeros(n, bool), []
        for c in unused:
            zone = np.asarray(tree.query_ball_point(pos[c], r=float(radius)), dtype=np.int64)
            far_enough = all(np.hypot(*(pos[c] - pos[o])) > 2.0 * float(radius) for o in chosen)
            full = per_batch is not None and len(chosen) >= int(per_batch)
            if chosen and (full or not far_enough or (covered.sum() + len(zone)) > float(max_cover) * n):
                left.append(c)
                continue
            chosen.append(c)
            covered[zone] = True
        batches.append(np.asarray(sorted(chosen), dtype=np.int64))
        unused = left
    return batches


def surrogate_sources(
    pos, receivers, radius: float, region: str, *, rng: np.random.Generator, min_shift: float = 0.25
) -> np.ndarray:
    """``[n]`` the row each cell of one graph takes in one surrogate: the identity where the true context is kept."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}.")
    n = len(pos)
    shifted = shift_sources(pos, rng=rng, min_shift=min_shift)
    source = np.arange(n)
    in_zone = np.zeros(n, bool)
    for zone in zones(pos, receivers, radius):
        in_zone[zone] = True
    if region == "far":
        replace = ~in_zone
    else:
        replace = in_zone.copy()
        replace[np.asarray(receivers, dtype=np.int64)] = False
    source[replace] = shifted[replace]
    return source


def _per_cell_attributes(data) -> list[str]:
    n = int(data.num_nodes)
    found = []
    for key in data.keys():
        if key in ("edge_index", "batch", "ptr"):
            continue
        value = data[key]
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n:
            found.append(key)
    return found


def _check_fields(data_list, fields, stay) -> None:
    allowed = set(fields) | set(stay) | POSITIONAL
    for i, data in enumerate(data_list):
        unknown = [k for k in _per_cell_attributes(data) if k not in allowed]
        if unknown:
            raise ValueError(
                f"graph {i} carries per-cell attributes {unknown} that are neither moved with the rows (`fields`) nor declared "
                "positional (`stay`). An expression-derived one (embeddings, labels) must move, or the surrogate keeps the "
                "receivers' true context through it."
            )


def apply_sources(data, source: np.ndarray, fields) -> object:
    """A copy of ``data`` whose ``fields`` (those it has) are taken row by row from ``source``."""
    out = data.clone()
    index = torch.as_tensor(source, dtype=torch.long)
    for key in fields:
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key][index.to(out[key].device)]
    return out


# ------------------------------------------------------------------------------- passes


def _predict(module, data_list, *, head: str, batch_size: int, device) -> list[np.ndarray]:
    """Per graph ``[n, G]`` the head's prediction at every scored cell (NaN elsewhere)."""
    from interscale.evaluation.masked_probing import _model_heads

    preds = [np.full((int(d.num_nodes), int(d.x.shape[1])), np.nan) for d in data_list]
    loader = DataLoader(dataset=data_list, shuffle=False, batch_size=batch_size, num_workers=0)
    start = 0
    for batch in loader:
        batch = batch.to(device)
        out = module._common_step(batch, "regression", "node")
        heads, _, _ = _model_heads(module, out)
        if head not in heads:
            raise ValueError(f"no head {head!r}; this module has {sorted(heads)}.")
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(batch.num_nodes, device=batch.x.device)
        graph = batch.batch[idx]
        local = idx - batch.ptr[graph]
        value = heads[head].detach().double().cpu().numpy()
        for g, i, row in zip(graph.cpu().numpy(), local.cpu().numpy(), value, strict=True):
            preds[start + int(g)][int(i)] = row
        start += int(batch.num_graphs)
    return preds


def _hidden_entries(data, mask_strategy: str) -> np.ndarray:
    n, g = int(data.num_nodes), int(data.x.shape[1])
    gene_mask = getattr(data, "gene_mask", None) if mask_strategy == "gene" else None
    if gene_mask is not None:
        return gene_mask.bool().cpu().numpy()
    mask = getattr(data, "mask", None)
    if mask is None:
        return np.zeros((n, g), bool)
    return np.repeat(mask.bool().cpu().numpy()[:, None], g, axis=1)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order] * len(v) / np.arange(1, len(v) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(v)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


@dataclass
class ContextTest:
    """Per gene the test's result; ``losses`` ``[n_shifts, G]`` the surrogate losses; ``settings`` the arguments and counts."""

    table: pd.DataFrame
    losses: np.ndarray
    settings: dict


def context_test(
    module,
    data_list,
    genes,
    *,
    radius: float,
    region: str = "far",
    mask_strategy: str,
    head: str = "global",
    n_shifts: int = 50,
    n_batches: int = 1,
    max_cover: float = 0.1,
    per_batch: int | None | str = "auto",
    min_shift: float = 0.25,
    fields=DEFAULT_FIELDS,
    stay=(),
    batch_size: int = 1,
    device="cpu",
    seed: int = 0,
) -> ContextTest:
    """The per-gene context test of ``module`` on ``data_list`` (graphs carrying the masks to score, e.g. one fold hidden).

    Parameters
    ----------
    module
        Any module with a ``head`` (``global``: a GlobalModel, a dual model's reported sum).
    data_list
        PyG graphs with ``x``, ``pos`` and their masks; not changed.
    genes
        The ``G`` gene names.
    radius
        ``r`` in the units of ``pos``.
    region
        :data:`REGIONS`: which context the surrogates replace.
    mask_strategy
        ``cfg.dataset.mask_strategy``: which attribute says what is hidden.
    n_shifts, n_batches, max_cover, min_shift
        Surrogates per batch, receiver batches per graph, the zones' cover, the smallest shift.
    per_batch
        Receivers per graph and batch; ``"auto"``: 1 for ``near`` (anything more makes the near test read far context), no
        limit for ``far``.
    fields, stay
        Per-cell attributes that move with a row / stay at their position (beyond :data:`POSITIONAL`).
    """
    genes = [str(g) for g in genes]
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}.")
    if int(n_shifts) < 1:
        raise ValueError(f"n_shifts must be at least 1, got {n_shifts}.")
    if any(getattr(d, "pos", None) is None for d in data_list):
        raise ValueError("the test places zones and shifts by position; every graph needs `pos`.")
    _check_fields(data_list, fields, stay)
    if per_batch == "auto":
        per_batch = 1 if region == "near" else None
    rng = np.random.default_rng(seed)
    hidden = [_hidden_entries(d, mask_strategy) for d in data_list]
    truth = [d.x.double().cpu().numpy() for d in data_list]
    pos = [d.pos.double().cpu().numpy() for d in data_list]
    batches = [
        choose_receivers(
            p, h.any(axis=1), radius, rng=rng, max_cover=max_cover, n_batches=n_batches, per_batch=per_batch
        )
        for p, h in zip(pos, hidden, strict=True)
    ]
    receivers = [np.concatenate(b) if b else np.zeros(0, np.int64) for b in batches]
    if not any(len(r) for r in receivers):
        raise ValueError("no receiver: no graph has a cell with a hidden entry.")

    n_genes = len(genes)
    count = np.zeros(n_genes)
    loss_true = np.zeros(n_genes)
    loss_shift = np.zeros((int(n_shifts), n_genes))
    weighted_var = np.zeros(n_genes)

    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad():
            base = _predict(module, data_list, head=head, batch_size=batch_size, device=device)
            for g, r in enumerate(receivers):
                if not len(r):
                    continue
                h = hidden[g][r]
                if np.isnan(base[g][r][h]).any():
                    raise ValueError(
                        f"graph {g}: a receiver was not predicted (a graph longer than max_seq_len is subsampled)."
                    )
                err = np.where(h, (base[g][r] - truth[g][r]) ** 2, 0.0)
                loss_true += err.sum(axis=0)
                count += h.sum(axis=0)
                # The normaliser: the gene's variance over all cells of the graph, weighted by the graph's scored entries
                # (over the receivers alone it is 0 with one receiver per graph, as the near test has).
                graph_var = truth[g].var(axis=0, ddof=1) if len(truth[g]) > 1 else np.zeros(n_genes)
                weighted_var += h.sum(axis=0) * graph_var
            for b in range(max((len(x) for x in batches), default=0)):
                members = [g for g, x in enumerate(batches) if len(x) > b]
                for k in range(int(n_shifts)):
                    surrogates = []
                    for g in members:
                        source = surrogate_sources(pos[g], batches[g][b], radius, region, rng=rng, min_shift=min_shift)
                        surrogates.append(apply_sources(data_list[g], source, fields))
                    preds = _predict(module, surrogates, head=head, batch_size=batch_size, device=device)
                    for pred, g in zip(preds, members, strict=True):
                        r = batches[g][b]
                        h = hidden[g][r]
                        if np.isnan(pred[r][h]).any():
                            raise ValueError(f"graph {g}: a receiver was not predicted in a surrogate pass.")
                        loss_shift[k] += np.where(h, (pred[r] - truth[g][r]) ** 2, 0.0).sum(axis=0)
    finally:
        module.train(was_training)

    with np.errstate(invalid="ignore", divide="ignore"):
        loss_true_mean = np.where(count > 0, loss_true / count, np.nan)
        loss_shift_mean = np.where(count > 0, loss_shift / count, np.nan)
        variance = np.where(count > 0, weighted_var / np.maximum(count, 1), np.nan)
        delta = loss_shift_mean.mean(axis=0) - loss_true_mean
        delta_r2 = np.where(variance > 0, delta / variance, np.nan)
        # A surrogate that leaves a gene's prediction as it was gives the same loss up to the order of the float sums: a tie,
        # which must count as "not worse" (p = 1 for a gene the replaced context does not reach).
        tie = TIE_TOLERANCE * np.maximum(np.abs(loss_true_mean), 1e-300)
        p_rank = np.where(
            count > 0, (1.0 + (loss_shift_mean <= loss_true_mean + tie).sum(axis=0)) / (int(n_shifts) + 1.0), np.nan
        )
        sd = loss_shift_mean.std(axis=0, ddof=1) if int(n_shifts) > 1 else np.full(n_genes, np.nan)
        z = np.where(sd > tie, delta / np.where(sd > tie, sd, 1.0), np.where(delta > tie, np.inf, -np.inf))
        p_gauss = np.where(count > 0, norm.sf(z), np.nan)
    table = pd.DataFrame(
        {
            "n_entries": count.astype(int),
            "variance": variance,
            "loss_true": loss_true_mean,
            "loss_shift": loss_shift_mean.mean(axis=0),
            "loss_shift_sd": sd,
            "delta": delta,
            "delta_r2": delta_r2,
            "z": z,
            "p": p_gauss,
            "q": _bh(p_gauss),
            "p_rank": p_rank,
        },
        index=pd.Index(genes, name="gene"),
    )
    settings = {
        "radius": float(radius),
        "region": region,
        "head": head,
        "n_shifts": int(n_shifts),
        "n_batches": int(n_batches),
        "max_cover": float(max_cover),
        "per_batch": per_batch,
        "min_shift": float(min_shift),
        "seed": int(seed),
        "n_receivers": int(sum(len(r) for r in receivers)),
        "n_graphs": len(data_list),
        "passes": 1 + max((len(x) for x in batches), default=0) * int(n_shifts),
    }
    return ContextTest(table=table, losses=loss_shift_mean, settings=settings)


def context_labels(far: ContextTest, near: ContextTest, *, q: float = 0.05, min_delta_r2: float = 0.0) -> pd.DataFrame:
    """Per gene ``global`` (far context used), ``local`` (near context used), ``both``, ``none``; significant at ``q`` and with
    ``delta_r2`` above ``min_delta_r2``.
    """
    a, b = far.table, near.table.reindex(far.table.index)
    uses_far = (a["q"] <= q) & (a["delta_r2"] > min_delta_r2)
    uses_near = (b["q"] <= q) & (b["delta_r2"] > min_delta_r2)
    label = np.select([uses_far & uses_near, uses_far, uses_near], ["both", "global", "local"], default="none")
    return pd.DataFrame(
        {
            "label": label,
            "far_delta_r2": a["delta_r2"],
            "far_q": a["q"],
            "near_delta_r2": b["delta_r2"],
            "near_q": b["q"],
        },
        index=a.index,
    )
