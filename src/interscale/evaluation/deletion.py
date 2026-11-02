"""R2: sender genes by in-silico deletion (``.claude/ranking_plan.md``).

Which input genes does a model read to predict each hidden gene -- and, with ``region="context"``, from *other* cells? A sender
gene (or a group of genes) is replaced by a draw from its distribution given the cell's other genes, in the cells of a region, and
the change of every target gene's squared error at the receivers is read: a sender x target matrix of ΔR². Geneformer ranks genes
by the shift a deletion causes (Theodoris et al. 2023); the holdout randomisation test (Tansey et al. 2022) says to replace a
feature by a draw from its conditional distribution, not by zero, which lies off the data and measures how the model handles
nonsense.

**Replacement** (:func:`conditional_parts`, ``method``). Per graph over its fully visible cells (the rows showing every column of
the common visible set): the covariance of the visible genes, regularised by ``ridge`` times its mean variance, and its inverse Θ.
For a block B of genes, the conditional mean given the other genes is ``x_B - (X_c Θ)_B Θ_BB^{-1}`` (``X_c`` centred); all blocks
from one inverse. ``conditional``: that mean plus the residual rows of other visible cells (drawn with replacement, jointly for a
block); ``permute``: the block's own rows drawn from other cells (which also breaks its link to the cell's other genes);
``mean``: the graph's visible mean.

**Regions.** ``all``: the sender replaced in every visible cell, the receivers' own rows included (own and context). ``context``:
everywhere but the receivers (sparse receivers, at most ``max_cover`` of a graph; the other receivers' rows stay true, which costs
power, not validity). ``far``: only beyond ``radius`` of a receiver (``context_test``'s zones). Under cell masking a receiver is hidden
whole, so ``all`` and ``context`` agree there and every hidden cell is a receiver (``all``).

**Statistic.** One true pass, one pass per sender and draw. Per receiver and target the difference of squared errors on its hidden
entries, averaged over the draws; per (sender, target) its mean over receivers is ``delta``, over the graph-weighted variance of the
target (``context_test``'s normaliser) ``delta_r2``, and ``z`` its mean over its standard error across receivers (neighbouring
receivers are not independent: read z as a guide, rank by ``delta_r2``). Per sender ``importance`` = the sum over targets of the
positive ΔR², ``n_targets`` = targets with ΔR² above ``min_effect`` and z above 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from interscale.evaluation.context_test import _hidden_entries, _predict, choose_receivers, zones
from interscale.evaluation.online_probes import _rng_untouched

REGIONS = ("all", "context", "far")
METHODS = ("conditional", "permute", "mean")


# ------------------------------------------------------------------------------- replacement


def conditional_parts(x: np.ndarray, *, ridge: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """``(mean [G], precision [G, G])`` of the rows ``x`` ``[n, G]``: their mean and the inverse of their covariance plus ``ridge``
    times its mean variance (a gene without variance gets the ridge alone).
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    centred = x - mean
    cov = centred.T @ centred / max(len(x) - 1, 1)
    scale = float(np.mean(np.diag(cov))) or 1.0
    precision = np.linalg.inv(cov + float(ridge) * scale * np.eye(len(cov)))
    return mean, precision


def conditional_mean(x: np.ndarray, block, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """``[n, |B|]`` the Gaussian conditional mean of the columns ``block`` given the other columns, row by row."""
    block = np.asarray(block, dtype=np.int64)
    centred = np.asarray(x, dtype=np.float64) - mean
    r = (centred @ precision[:, block]) @ np.linalg.inv(precision[np.ix_(block, block)])
    return np.asarray(x, dtype=np.float64)[:, block] - r


@dataclass
class _GraphModel:
    rows: np.ndarray  # the fully visible rows the conditional is fitted on
    cols: np.ndarray  # the columns visible in all of them
    mean: np.ndarray | None
    precision: np.ndarray | None


def _graph_model(x: np.ndarray, visible: np.ndarray, *, method: str, ridge: float) -> _GraphModel:
    counts = visible.sum(axis=1)
    rows = np.flatnonzero(counts == counts.max()) if len(counts) else np.zeros(0, np.int64)
    cols = np.flatnonzero(visible[rows].all(axis=0)) if len(rows) else np.zeros(0, np.int64)
    if method != "conditional" or len(rows) < 3:
        return _GraphModel(rows, cols, None, None)
    mean, precision = conditional_parts(x[np.ix_(rows, cols)], ridge=ridge)
    return _GraphModel(rows, cols, mean, precision)


def _replaced_values(
    x, model: _GraphModel, block, targets_rows, *, method: str, rng: np.random.Generator
) -> np.ndarray:
    """``[len(targets_rows), |B|]`` the replacement values of the sender columns ``block`` at the rows ``targets_rows``."""
    pos = {c: k for k, c in enumerate(model.cols)}
    local = [pos[c] for c in block]
    donors = rng.integers(0, len(model.rows), size=len(targets_rows))
    if method == "mean":
        return np.broadcast_to(x[np.ix_(model.rows, block)].mean(axis=0), (len(targets_rows), len(block))).copy()
    if method == "permute" or model.precision is None:
        return x[np.ix_(model.rows[donors], block)]
    sub = x[np.ix_(targets_rows, model.cols)]
    fitted = conditional_mean(sub, local, model.mean, model.precision)
    donor_x = x[np.ix_(model.rows, model.cols)]
    residual = donor_x[:, local] - conditional_mean(donor_x, local, model.mean, model.precision)
    return fitted + residual[donors]


# ------------------------------------------------------------------------------- the deletion


@dataclass
class Deletion:
    """``delta_r2``, ``z``, ``delta`` (senders x targets); ``senders`` per sender ``importance``, ``n_targets``, ``top_target``;
    ``targets`` per target ``loss_true``, ``variance``, ``n_entries``; ``settings``.
    """

    delta_r2: pd.DataFrame
    z: pd.DataFrame
    delta: pd.DataFrame
    senders: pd.DataFrame
    targets: pd.DataFrame
    settings: dict


def _sender_blocks(senders, genes: list[str]) -> dict[str, list[int]]:
    position = {g: k for k, g in enumerate(genes)}
    if isinstance(senders, dict):
        items = {str(k): list(v) for k, v in senders.items()}
    else:
        items = {str(s): [s] for s in senders}
    out = {}
    for name, members in items.items():
        missing = [m for m in members if str(m) not in position]
        if missing or not members:
            raise ValueError(f"sender {name!r}: genes {missing or members} are not among the genes.")
        out[name] = [position[str(m)] for m in members]
    return out


def sender_deletion(
    module,
    data_list,
    genes,
    *,
    senders,
    mask_strategy: str,
    region: str = "context",
    radius: float = 0.0,
    method: str = "conditional",
    n_draws: int = 3,
    max_cover: float = 0.1,
    n_batches: int = 1,
    ridge: float = 1e-2,
    min_effect: float = 0.01,
    head: str = "global",
    batch_size: int = 1,
    device="cpu",
    seed: int = 0,
) -> Deletion:
    """The in-silico deletion of each sender in ``data_list`` (graphs carrying the masks to score).

    Parameters
    ----------
    senders
        Gene names (each deleted alone) or ``{name: [genes]}`` (each group deleted together). A sender must be visible in the
        graphs' fully visible cells (a hidden gene shows the model nothing to delete).
    region, radius
        :data:`REGIONS`; ``radius`` (units of ``pos``) for ``far``.
    method, n_draws, ridge
        :data:`METHODS`; draws per sender; the conditional's ridge (times the mean variance).
    max_cover, n_batches
        Receivers for ``context`` / ``far`` (as ``context_test.choose_receivers``).
    min_effect
        ΔR² a target must exceed (with z > 3) to count for a sender's ``n_targets``.
    """
    genes = [str(g) for g in genes]
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}.")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    if int(n_draws) < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}.")
    if any(getattr(d, "embeddings", None) is not None for d in data_list):
        raise ValueError("the graphs carry precomputed embeddings: a deletion of x would not reach them.")
    blocks = _sender_blocks(senders, genes)
    rng = np.random.default_rng(seed)
    hidden = [_hidden_entries(d, mask_strategy) for d in data_list]
    truth = [d.x.double().cpu().numpy() for d in data_list]
    visible = [~h for h in hidden]
    models = [_graph_model(x, v, method=method, ridge=ridge) for x, v in zip(truth, visible, strict=True)]
    for name, block in blocks.items():
        for g, m in enumerate(models):
            if not set(block) <= set(m.cols.tolist()):
                raise ValueError(f"sender {name!r} is hidden in graph {g}'s visible cells: nothing to delete.")

    # receivers and the cells whose sender values are replaced, per graph
    receivers, replaced = [], []
    for g, d in enumerate(data_list):
        candidates = hidden[g].any(axis=1)
        rows_ok = np.zeros(len(candidates), bool)
        rows_ok[models[g].rows] = True
        if region == "all":
            r = np.flatnonzero(candidates)
            rep = rows_ok.copy()
        else:
            if getattr(d, "pos", None) is None:
                raise ValueError(f"region {region!r} places receivers by position; graph {g} has no `pos`.")
            pos = d.pos.double().cpu().numpy()
            rad = float(radius) if region == "far" else 0.0
            batches = choose_receivers(pos, candidates, rad, rng=rng, max_cover=max_cover, n_batches=n_batches)
            r = np.concatenate(batches) if batches else np.zeros(0, np.int64)
            keep = np.zeros(len(candidates), bool)
            for zone in zones(pos, r, rad) if len(r) else []:
                keep[zone] = True
            keep[r] = True
            rep = rows_ok & ~keep
        receivers.append(np.asarray(r, dtype=np.int64))
        replaced.append(np.flatnonzero(rep))
    if not any(len(r) for r in receivers):
        raise ValueError("no receiver: no graph has a cell with a hidden entry.")

    n_genes = len(genes)
    count = np.zeros(n_genes)
    weighted_var = np.zeros(n_genes)
    loss_true = np.zeros(n_genes)
    names = list(blocks)
    total = np.zeros((len(names), n_genes))
    total_sq = np.zeros((len(names), n_genes))
    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad():
            base = _predict(module, data_list, head=head, batch_size=batch_size, device=device)
            err_true = []
            for g, r in enumerate(receivers):
                h = hidden[g][r]
                if len(r) and np.isnan(base[g][r][h]).any():
                    raise ValueError(
                        f"graph {g}: a receiver was not predicted (a graph longer than max_seq_len is subsampled)."
                    )
                e = np.where(h, (base[g][r] - truth[g][r]) ** 2, 0.0) if len(r) else np.zeros((0, n_genes))
                err_true.append(e)
                loss_true += e.sum(axis=0)
                count += h.sum(axis=0)
                graph_var = truth[g].var(axis=0, ddof=1) if len(truth[g]) > 1 else np.zeros(n_genes)
                weighted_var += h.sum(axis=0) * graph_var
            for s, name in enumerate(names):
                block = blocks[name]
                diff = [np.zeros((len(r), n_genes)) for r in receivers]
                for _ in range(int(n_draws)):
                    modified = []
                    for g, d in enumerate(data_list):
                        c = d.clone()
                        rows = replaced[g]
                        if len(rows):
                            x = c.x.clone()
                            values = _replaced_values(truth[g], models[g], block, rows, method=method, rng=rng)
                            x[torch.as_tensor(rows)[:, None], torch.as_tensor(block)[None, :]] = torch.as_tensor(
                                values, dtype=x.dtype
                            )
                            c.x = x
                        modified.append(c)
                    preds = _predict(module, modified, head=head, batch_size=batch_size, device=device)
                    for g, r in enumerate(receivers):
                        if not len(r):
                            continue
                        h = hidden[g][r]
                        diff[g] += np.where(h, (preds[g][r] - truth[g][r]) ** 2, 0.0) - err_true[g]
                for g, r in enumerate(receivers):
                    if not len(r):
                        continue
                    mean_diff = diff[g] / int(n_draws)
                    h = hidden[g][r]
                    total[s] += np.where(h, mean_diff, 0.0).sum(axis=0)
                    total_sq[s] += np.where(h, mean_diff**2, 0.0).sum(axis=0)
    finally:
        module.train(was_training)

    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.maximum(count, 1)
        delta = np.where(count > 0, total / n, np.nan)
        variance = np.where(count > 0, weighted_var / n, np.nan)
        delta_r2 = np.where(variance > 0, delta / variance, np.nan)
        sd = np.sqrt(np.maximum(total_sq / n - delta**2, 0.0) * n / np.maximum(n - 1, 1))
        z = np.where(sd > 0, delta / (sd / np.sqrt(n)), np.where(delta > 0, np.inf, 0.0))
    scored = count > 0
    target_names = [g for g, s in zip(genes, scored, strict=True) if s]
    dr2 = pd.DataFrame(delta_r2[:, scored], index=pd.Index(names, name="sender"), columns=target_names)
    zz = pd.DataFrame(z[:, scored], index=dr2.index, columns=target_names)
    dd = pd.DataFrame(delta[:, scored], index=dr2.index, columns=target_names)
    strong = (dr2 > float(min_effect)) & (zz > 3.0)
    senders_table = pd.DataFrame(
        {
            "importance": dr2.clip(lower=0).sum(axis=1),
            "n_targets": strong.sum(axis=1),
            "top_target": dr2.idxmax(axis=1) if dr2.shape[1] else None,
            "top_delta_r2": dr2.max(axis=1) if dr2.shape[1] else np.nan,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")
    targets_table = pd.DataFrame(
        {
            "loss_true": np.where(scored, loss_true / np.maximum(count, 1), np.nan)[scored],
            "variance": variance[scored],
            "n_entries": count[scored].astype(int),
        },
        index=pd.Index(target_names, name="target"),
    )
    settings = {
        "region": region,
        "radius": float(radius),
        "method": method,
        "n_draws": int(n_draws),
        "max_cover": float(max_cover),
        "ridge": float(ridge),
        "seed": int(seed),
        "head": head,
        "n_senders": len(names),
        "n_receivers": int(sum(len(r) for r in receivers)),
        "passes": 1 + len(names) * int(n_draws),
    }
    return Deletion(delta_r2=dr2, z=zz, delta=dd, senders=senders_table, targets=targets_table, settings=settings)


def planted_recovery(deletion: Deletion, edges, *, null: Deletion | None = None) -> pd.DataFrame:
    """Per target with planted senders: the best rank of a planted sender among all senders by ΔR² (1 = top), its reciprocal, and
    whether it is first; ``edges`` ``[(sender, target), ...]``. With ``null`` (e.g. the untrained twin) its ranks beside.
    """

    def ranks(d: Deletion) -> dict:
        out = {}
        by_target: dict = {}
        for s, t in edges:
            by_target.setdefault(str(t), set()).add(str(s))
        for target, planted in by_target.items():
            if target not in d.delta_r2.columns:
                continue
            column = d.delta_r2[target].rank(ascending=False, method="min")
            present = [s for s in planted if s in column.index]
            if present:
                out[target] = float(column[present].min())
        return out

    own = ranks(deletion)
    rows = [{"target": t, "rank": r, "reciprocal_rank": 1.0 / r, "first": r == 1.0} for t, r in own.items()]
    table = (
        pd.DataFrame(rows).set_index("target") if rows else pd.DataFrame(columns=["rank", "reciprocal_rank", "first"])
    )
    if null is not None and len(table):
        other = ranks(null)
        table["null_rank"] = [other.get(t, np.nan) for t in table.index]
    return table
