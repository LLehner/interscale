"""From a trained model to biology: who affects whom, through which genes, and what the embedding programmes are.

Four readouts of one trained model, each read against a null, none needing a gradient, and dataset-agnostic: the labels, gene sets,
ligand-receptor pairs and conditions come from the caller (``.claude/interaction_readout_plan.md``, "Design 2026-11-06").

**Context use and cell-type flows, by intervention** (:func:`type_flows`). Receivers keep their own row; the visible expression of
other cells is replaced, and the change in the receivers' error on their hidden genes is the flow. Scrambling instead of reading attention
makes it a statement about what the model *uses*, and it is directional by construction (the sender's cells change, the receiver's are scored).

* ``identity``: the sender type's cells take the profile of a random cell of another type. What the model takes from that type being there.
* ``identity_null``: the same replacement on the cells nearest to the sender's, of other types. Same place, same number of cells. ``identity``
  minus this is what is specific to the type and not to the place (co-location is the confound of every label-based flow).
* ``state``: the sender type's cells swap profiles among themselves. Type and composition stay at every position, only which state sits where
  changes: the closest this gets to communication, and it does not break the link between a label and its expression.
* ``all_state``, ``all_arrangement``: every other cell swapped within its type, or among all cells: how much context a receiver type uses at all.

Regions: ``context`` keeps only the receivers themselves, ``far`` also every cell within ``far_radius`` of a receiver, so only distant cells change.

**Gene flows** (:func:`gene_flows`). One sender gene (a ligand, say) permuted in the context, within type by default: per receiver type and hidden
target the same statistic. :func:`lr_support` asks whether known ligand-receptor pairs rank above other targets of the same ligand, against
expression-matched permutations; :func:`enrichment` reads gene lists against gene sets with the panel as the background.

**Attention, read biologically** (:func:`attention_biology`, :func:`attention_pairs`). Which cell types and which genes sit in the sources and sinks of
the net attention flow, within type and section; the type-by-type table of the top sources against labels shuffled locally.

**Programmes** (:func:`programme_cards`). The prediction's main directions, rotated to sparse gene programmes, each read for genes, pathways, cell type,
spatial scale within type, neighbourhood and condition, and sorted into a class by stated thresholds.

Nothing here says what a model reads from a reconstruction score. A number is read against the untrained twin of the same run; a condition difference
per donor pair, so across folds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import hypergeom

from interscale.evaluation.context_test import _hidden_entries, _predict
from interscale.evaluation.dimensions import DimensionReadout, _effective_basis, _kept_rows, _signed, varimax
from interscale.evaluation.online_probes import _rng_untouched

TYPE_KINDS = ("identity", "identity_null", "state", "all_state", "all_arrangement")
#: A sender name for the perturbations of every other cell.
EVERYONE = "*"
#: The class thresholds of :func:`programme_class`.
CLASS_THRESHOLDS = {
    "handling_share": 0.35,
    "identity_eta2": 0.4,
    "moran": 0.1,
    "neighbour_r2": 0.02,
    "weak_share": 0.01,
}


# ----------------------------------------------------------------------------------------------- biology resources


def enrichment(hits, background, gene_sets: dict, *, min_overlap: int = 2) -> pd.DataFrame:
    """Over-representation of ``hits`` in ``gene_sets`` ``{library: {name: genes}}``: a hypergeometric test with ``background`` as the universe.

    Sets are cut to the background first, so a panel is its own universe (a panel chosen for its biology would otherwise enrich everything).
    ``q`` is Benjamini-Hochberg within a library. Sorted by ``p``; sets with fewer than ``min_overlap`` hits are left out of the table but counted
    in the correction.
    """
    universe = {str(g) for g in background}
    hits = {str(g) for g in hits} & universe
    columns = ["library", "set", "overlap", "set_size", "hits", "p", "q", "genes"]
    rows = []
    for library, sets in gene_sets.items():
        tested = []
        for name, members in sets.items():
            members = {str(g) for g in members} & universe
            if not members:
                continue
            found = hits & members
            p = float(hypergeom.sf(len(found) - 1, len(universe), len(members), len(hits))) if found else 1.0
            tested.append((name, len(found), len(members), p, sorted(found)))
        if not tested:
            continue
        ps = np.array([t[3] for t in tested])
        order = np.argsort(ps, kind="stable")
        adjusted = ps[order] * len(ps) / np.arange(1, len(ps) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q = np.empty(len(ps))
        q[order] = np.minimum(adjusted, 1.0)
        for (name, n, size, p, found), value in zip(tested, q, strict=True):
            if n >= int(min_overlap):
                rows.append(
                    {
                        "library": library,
                        "set": name,
                        "overlap": n,
                        "set_size": size,
                        "hits": len(hits),
                        "p": p,
                        "q": float(value),
                        "genes": ",".join(found),
                    }
                )
    return pd.DataFrame(rows, columns=columns).sort_values("p", kind="stable").reset_index(drop=True)


def lr_support(
    score: pd.DataFrame,
    pairs,
    *,
    expression: pd.Series | None = None,
    n_bins: int = 5,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Do known receptors rank above the other targets of the same ligand? ``score`` senders x targets (NaN: not scored); ``pairs`` ``(ligand, receptor)``.

    Per pair the rank of the receptor among the ligand's scored targets, as an AUC (0.5: chance). The null keeps each ligand and how many receptors it has,
    and draws the receptors from targets of similar mean ``expression`` (``n_bins`` quantile bins; expressed genes are easier to predict, and to move). It
    returns ``{"n_pairs", "auc", "null_mean", "null_sd", "p", "pairs", "null_draws"}``; ``pairs`` holds each scored pair's score and rank. Support for the pairs, not proof
    of signalling: a receptor's level is only one thing a ligand could change.
    """
    rng = np.random.default_rng(seed)
    empty = {"n_pairs": 0, "auc": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p": np.nan, "null_draws": np.zeros(0)}
    pair_columns = ["ligand", "receptor", "score", "auc"]
    targets = [str(t) for t in score.columns]
    expr = None if expression is None else expression.reindex(targets).to_numpy(dtype=float)
    bins = None
    if expr is not None and np.isfinite(expr).any():
        filled = np.where(np.isfinite(expr), expr, np.nanmedian(expr))
        edges = np.quantile(filled, np.linspace(0, 1, int(n_bins) + 1)[1:-1])
        bins = np.searchsorted(edges, filled)
    by_ligand: dict[str, list[int]] = {}
    position = {t: k for k, t in enumerate(targets)}
    for ligand, receptor in pairs:
        ligand, receptor = str(ligand), str(receptor)
        if ligand in score.index and receptor in position and ligand != receptor:
            by_ligand.setdefault(ligand, []).append(position[receptor])
    rows, ligand_rows = [], []
    for ligand, chosen in by_ligand.items():
        values = score.loc[ligand].to_numpy(dtype=float)
        scored = np.flatnonzero(np.isfinite(values))
        chosen = sorted({c for c in chosen if np.isfinite(values[c])})
        if not chosen or len(scored) < 2:
            continue
        ranks = pd.Series(values[scored]).rank(method="average").to_numpy()
        auc = np.empty(len(values))
        auc[:] = np.nan
        auc[scored] = (ranks - 1) / max(len(scored) - 1, 1)
        ligand_rows.append((scored, auc, chosen))
        rows += [
            {"ligand": ligand, "receptor": targets[c], "score": float(values[c]), "auc": float(auc[c])} for c in chosen
        ]
    if not rows:
        return empty | {"pairs": pd.DataFrame(columns=pair_columns)}
    table = (
        pd.DataFrame(rows, columns=pair_columns)
        .sort_values("score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    observed = float(table["auc"].mean())
    null = np.zeros(int(n_perm))
    for b in range(int(n_perm)):
        total, count = 0.0, 0
        for scored, auc, chosen in ligand_rows:
            for c in chosen:
                pool = scored if bins is None else scored[bins[scored] == bins[c]]
                total += float(auc[pool[rng.integers(len(pool))] if len(pool) else c])
                count += 1
        null[b] = total / count
    p = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return {
        "n_pairs": len(table),
        "auc": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p": p,
        "pairs": table,
        "null_draws": null,
    }


def lr_support_by_type(long: pd.DataFrame, pairs, *, n_bins: int = 5, n_perm: int = 2000, seed: int = 0) -> dict:
    """:func:`lr_support` within each receiver type, pooled: do known receptors rank above the other targets of the same ligand *in the same receiver type*?

    ``long`` is :func:`gene_flow_long`'s table (``sender``, ``receiver``, ``target``, ``delta_cos``, ``expressed``). Ranking a ligand's targets within one type, with
    the null drawn from targets of similar detection *in that type*, removes the bias of a score maximised over the types expressing a target: a type-specific
    gene (a receptor, often) is maximised over fewer types and ranks lower whatever the model does (the untrained twin's AUC 0.43 against a null of 0.485,
    notes.md 2026-11-07). The observed AUC is the mean over every (pair, type) read, the null the same mean over each type's null draws (one shared draw count,
    a seed per type). Returns the keys of :func:`lr_support` (``pairs`` with a ``receiver`` column) and ``n_types``.
    """
    columns = ["ligand", "receptor", "receiver", "score", "auc"]
    empty = {
        "n_pairs": 0, "auc": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p": np.nan, "p_independent": np.nan, "cluster_sd": np.nan, "z": np.nan,
        "n_types": 0, "pairs": pd.DataFrame(columns=columns),
    }  # fmt: skip
    if not len(long):
        return empty
    tables, nulls, weights = [], [], []
    for k, (receiver, part) in enumerate(sorted(long.groupby("receiver", observed=True), key=lambda item: str(item[0]))):
        score = part.pivot_table(index="sender", columns="target", values="delta_cos", aggfunc="mean")
        expression = part.groupby("target", observed=True)["expressed"].first()
        found = lr_support(score, pairs, expression=expression, n_bins=n_bins, n_perm=n_perm, seed=seed + k)
        if not found["n_pairs"]:
            continue
        tables.append(found["pairs"].assign(receiver=str(receiver)))
        nulls.append(found["null_draws"])
        weights.append(found["n_pairs"])
    if not tables:
        return empty
    weights = np.asarray(weights, dtype=float)
    table = pd.concat(tables, ignore_index=True)[columns].sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    observed = float(table["auc"].mean())
    null = (np.vstack(nulls) * weights[:, None]).sum(axis=0) / weights.sum()
    clustered = lr_cluster_p(table, float(null.mean()), float(null.std()), seed=seed)
    return {
        "n_pairs": len(table),
        "auc": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p": clustered["p"],
        "p_independent": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "cluster_sd": clustered["cluster_sd"],
        "z": clustered["z"],
        "n_types": len(tables),
        "pairs": table,
    }


def lr_cluster_p(pairs: pd.DataFrame, null_mean: float, null_sd: float, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """The known-pair test's ``p`` with the pairs' shared genes allowed for: ``{"cluster_sd", "z", "p"}``. ``pairs`` holds ``ligand``, ``receptor``, ``auc``.

    The pairs are not independent: on the CosMx panel 1,245 (pair, type) reads came from 124 receptors and 165 ligands (ITGB1 alone 144), and a gene whose
    targets move under any permutation carries all of its pairs with it. The null of :func:`lr_support` draws a separate target for every pair, so its
    spread (0.008) understated the observed AUC's (0.022), and the untrained twin's 0.470 against 0.501 looked like a bias (notes.md 2026-11-08). Here the
    observed mean AUC is bootstrapped over ligand and receptor genes at once (each gene list resampled, a pair weighted by the product of its two genes'
    counts), and ``p`` is one-sided from ``z = (auc - null_mean) / sqrt(cluster_sd² + null_sd²)``.
    """
    auc = pd.to_numeric(pairs["auc"], errors="coerce").to_numpy(dtype=float) if len(pairs) else np.zeros(0)
    keep = np.isfinite(auc)
    if keep.sum() < 2 or not np.isfinite(null_mean):
        return {"cluster_sd": np.nan, "z": np.nan, "p": np.nan}
    auc = auc[keep]
    ligand = pd.factorize(pairs["ligand"].astype(str).to_numpy()[keep])[0]
    receptor = pd.factorize(pairs["receptor"].astype(str).to_numpy()[keep])[0]
    rng = np.random.default_rng(seed)
    n_l, n_r = ligand.max() + 1, receptor.max() + 1
    w = rng.multinomial(n_l, np.full(n_l, 1.0 / n_l), size=int(n_boot))[:, ligand] * rng.multinomial(n_r, np.full(n_r, 1.0 / n_r), size=int(n_boot))[:, receptor]
    total = w.sum(axis=1)
    means = (w @ auc)[total > 0] / total[total > 0]
    cluster_sd = float(means.std()) if len(means) > 1 else np.nan
    spread = float(np.hypot(cluster_sd, null_sd if np.isfinite(null_sd) else 0.0))
    z = (float(auc.mean()) - float(null_mean)) / spread if spread > 0 else np.nan
    return {"cluster_sd": cluster_sd, "z": float(z), "p": float(0.5 * math.erfc(z / math.sqrt(2.0))) if np.isfinite(z) else np.nan}


# ----------------------------------------------------------------------------------------------- interventions: choosing cells


def pick_receivers(
    pos,
    labels,
    candidates,
    radius: float,
    *,
    rng: np.random.Generator,
    per_type: int,
    max_share: float = 0.5,
    max_cover: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """``(receivers, keep [N])``: up to ``per_type`` receivers of each type, at most ``max_share`` of that type, and the cells that stay untouched.

    Rare types choose first, so a rare type keeps its receivers when the cells within ``radius`` of receivers reach ``max_cover`` of the section. ``keep`` holds
    the receivers and those cells: a receiver's own row is never changed, and in the ``far`` region neither is its neighbourhood.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    labels = np.asarray(labels).astype(str)
    candidates = np.asarray(candidates, dtype=bool)
    n = len(labels)
    tree = cKDTree(pos) if float(radius) > 0 else None
    keep = np.zeros(n, bool)
    chosen = []
    names, counts = np.unique(labels[candidates], return_counts=True)
    for name in names[np.argsort(counts, kind="stable")]:
        pool = rng.permutation(np.flatnonzero(candidates & (labels == name)))
        wanted = min(int(per_type), max(1, int(np.ceil(float(max_share) * len(pool)))))
        taken = 0
        for cell in pool:
            if taken >= wanted:
                break
            if tree is None:
                keep[cell] = True
            else:
                zone = np.asarray(tree.query_ball_point(pos[cell], r=float(radius)), dtype=np.int64)
                if taken and (keep.sum() + (~keep[zone]).sum()) > float(max_cover) * n:
                    break
                keep[zone] = True
            chosen.append(int(cell))
            taken += 1
    return np.asarray(sorted(chosen), dtype=np.int64), keep


def matched_cells(pos, rows, pool, *, rng: np.random.Generator, k: int = 32) -> np.ndarray:
    """For each cell of ``rows`` the nearest unused cell of ``pool``: the same place, other cells. Fewer than ``len(rows)`` when the pool runs out nearby."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    rows, pool = np.asarray(rows, dtype=np.int64), np.asarray(pool, dtype=np.int64)
    if not len(rows) or not len(pool):
        return np.zeros(0, np.int64)
    k = min(int(k), len(pool))
    _, near = cKDTree(pos[pool]).query(pos[rows], k=k)
    near = near.reshape(len(rows), k)
    used = np.zeros(len(pool), bool)
    out = []
    for i in rng.permutation(len(rows)):
        free = near[i][~used[near[i]]]
        if len(free):
            used[free[0]] = True
            out.append(pool[free[0]])
    return np.asarray(sorted(out), dtype=np.int64)


def _type_perturbations(labels, keep, pos, rng):
    """Yields ``(kind, sender, rows, donors)``: the cells changed and whose visible profile each takes."""
    labels = np.asarray(labels).astype(str)
    free = np.flatnonzero(~keep)
    everyone = np.arange(len(labels))
    for sender in sorted(set(labels.tolist())):
        rows = free[labels[free] == sender]
        if len(rows) < 2:
            continue
        others = everyone[labels != sender]
        if not len(others):
            continue
        yield "identity", sender, rows, rng.choice(others, size=len(rows))
        null = matched_cells(pos, rows, free[labels[free] != sender], rng=rng)
        if len(null):
            yield (
                "identity_null",
                sender,
                null,
                rng.choice(np.setdiff1d(everyone, null, assume_unique=False), size=len(null)),
            )
        yield "state", sender, rows, rng.permutation(rows)
    within = []
    for name in sorted(set(labels.tolist())):
        members = free[labels[free] == name]
        within.append((members, rng.permutation(members)))
    if within:
        yield "all_state", EVERYONE, np.concatenate([m for m, _ in within]), np.concatenate([d for _, d in within])
    if len(free) >= 2:
        yield "all_arrangement", EVERYONE, free, rng.permutation(free)


def _changed(graph, visible: torch.Tensor, rows, donors):
    """A copy of ``graph`` whose cells ``rows`` show the visible entries of ``donors``. Hidden entries stay: they were never shown to the model."""
    copy = graph.clone()
    x = graph.x.clone()
    rows_t, donors_t = torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(donors, dtype=torch.long)
    x[rows_t] = torch.where(visible[rows_t], graph.x[donors_t], graph.x[rows_t])
    copy.x = x
    return copy


def column_means(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Each column's mean over its ``valid`` rows (NaN where none is)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, values, 0.0).sum(axis=0) / valid.sum(axis=0)


def _centred(values: np.ndarray, valid: np.ndarray, centre: np.ndarray | None) -> np.ndarray:
    """``values`` minus ``centre`` (default: each column's mean over its ``valid`` rows); 0 outside ``valid``."""
    centre = column_means(values, valid) if centre is None else centre
    return np.where(valid, values - np.nan_to_num(centre), 0.0)


def centred_cosine(
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    axis: int,
    centre_pred: np.ndarray | None = None,
    centre_truth: np.ndarray | None = None,
) -> np.ndarray:
    """The cosine of prediction and truth over the ``valid`` entries, each centred per column (gene) first.

    ``axis=1``: one value per row (a cell, over its valid genes); ``axis=0``: one per column (a gene, over the cells where it is valid). NaN with fewer than 2
    (rows) or 3 (columns) entries or a zero norm. The centres default to each one's own mean over the rows given; the flows pass fixed ones, the receiver type's
    means over the whole section (truth, and the *unchanged* model's prediction), so a score does not depend on how many receivers were drawn and a changed
    prediction is read against the same point. Within type, a level or scale error of the model cancels, which the squared error does not (the CosMx models'
    R² is far below 0, notes.md 2026-11-07).
    """
    a, b = _centred(pred, valid, centre_pred), _centred(truth, valid, centre_truth)
    n = valid.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a * b).sum(axis=axis) / np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    return np.where((n >= (2 if axis == 1 else 3)) & np.isfinite(out), out, np.nan)


def centred_sums(pred, truth, valid, centre_pred, centre_truth) -> np.ndarray:
    """``[3, G]``: per column the sums of ``a * b``, ``a * a``, ``b * b`` over the ``valid`` rows, ``a``, ``b`` prediction and truth minus the fixed centres. Sums over
    draws and sections add; :func:`cosine_from_sums` turns them into one centred cosine per gene.
    """
    a, b = _centred(pred, valid, centre_pred), _centred(truth, valid, centre_truth)
    return np.stack([(a * b).sum(axis=0), (a * a).sum(axis=0), (b * b).sum(axis=0)])


def cosine_from_sums(sums: np.ndarray, count: np.ndarray, *, min_count: int = 3) -> np.ndarray:
    """The centred cosine per gene from :func:`centred_sums` added up; NaN with fewer than ``min_count`` entries or a zero norm."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums[0] / np.sqrt(sums[1] * sums[2])
    return np.where((count >= min_count) & np.isfinite(out), out, np.nan)


def _type_centres(base, truth, hidden, labels, types) -> dict:
    """Per type ``(prediction centre, truth centre)``: each gene's mean over the type's cells where it is hidden, the prediction the unchanged model's."""
    out = {}
    for t in types:
        members = labels == t
        valid = hidden[members] & np.isfinite(base[members])
        out[t] = (column_means(base[members], valid), column_means(truth[members], valid))
    return out


def _type_variance(truth, labels, types) -> dict[str, np.ndarray]:
    return {
        t: truth[labels == t].var(axis=0, ddof=1) if (labels == t).sum() > 1 else np.zeros(truth.shape[1])
        for t in types
    }


# ----------------------------------------------------------------------------------------------- interventions: type flows


def type_flows(
    module,
    graph,
    labels,
    genes,
    *,
    head: str = "global",
    far_radius: float = 60.0,
    n_draws: int = 2,
    per_type: int = 1500,
    far_per_type: int = 60,
    max_cover: float = 0.4,
    top_genes: int = 20,
    seed: int = 0,
    device="cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One section's flows: ``(flows, target_genes)``. See the module docstring for the perturbations.

    The score is the centred cosine within the receiver type (:func:`centred_cosine`): per receiver cell over its hidden genes, prediction and truth each
    centred on their own mean over that type's receivers. A flow is how much it drops, so a level or scale change of the prediction does not count.

    ``flows``, one row per region, draw, perturbation and receiver type: ``sum_delta`` (the drop in the cosine, summed over receiver cells; positive: the
    prediction got worse), ``n_scored`` (the cells scored), ``delta_cos`` their ratio, ``receivers``, ``changed`` (cells changed). Sum over rows before taking a
    ratio. ``target_genes``: per region, perturbation and receiver type the targets with the largest effect, over draws: ``delta_cos``, the drop of the target's
    centred cosine across the type's receivers (a correlation within type), and ``shift``, the signed mean change of the prediction.
    """
    genes = [str(g) for g in genes]
    labels = np.asarray(labels).astype(str)
    types = sorted(set(labels.tolist()))
    truth = graph.x.double().cpu().numpy()
    hidden = _hidden_entries(graph, "gene")
    visible = torch.as_tensor(~hidden)
    pos = graph.pos[:, :2].double().cpu().numpy()
    candidates = hidden.any(axis=1)
    variance = _type_variance(truth, labels, types)
    rng = np.random.default_rng(seed)
    rows_out, sums = [], {}
    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad():
            base = _predict(module, [graph], head=head, batch_size=1, device=device)[0]
            centres = _type_centres(base, truth, hidden, labels, types)
            regions = (("context", 0.0, int(per_type)), ("far", float(far_radius), int(far_per_type)))
            for region, radius, wanted in regions:
                if radius > 0 and wanted <= 0:
                    continue
                for draw in range(int(n_draws)):
                    receivers, keep = pick_receivers(
                        pos, labels, candidates, radius, rng=rng, per_type=wanted, max_cover=max_cover
                    )
                    if not len(receivers):
                        continue
                    groups = {t: receivers[labels[receivers] == t] for t in types}
                    # The unchanged model's scores, once per receiver type: per cell (the flow) and per gene (its targets).
                    reference = {}
                    for receiver, cells in groups.items():
                        if len(cells):
                            valid = hidden[cells] & np.isfinite(base[cells])
                            c = centres[receiver]
                            reference[receiver] = centred_cosine(
                                base[cells], truth[cells], valid, axis=1, centre_pred=c[0], centre_truth=c[1]
                            )
                    for kind, sender, rows, donors in _type_perturbations(labels, keep, pos, rng):
                        pred = _predict(
                            module, [_changed(graph, visible, rows, donors)], head=head, batch_size=1, device=device
                        )[0]
                        for receiver, cells in groups.items():
                            if not len(cells):
                                continue
                            valid = hidden[cells] & np.isfinite(pred[cells]) & np.isfinite(base[cells])
                            c = centres[receiver]
                            drop = reference[receiver] - centred_cosine(
                                pred[cells], truth[cells], valid, axis=1, centre_pred=c[0], centre_truth=c[1]
                            )
                            scored = np.isfinite(drop)
                            shift = np.where(valid, pred[cells] - base[cells], 0.0).sum(axis=0)
                            n_scored = int(scored.sum())
                            total = float(drop[scored].sum())
                            rows_out.append(
                                {
                                    "region": region,
                                    "draw": draw,
                                    "kind": kind,
                                    "sender": sender,
                                    "receiver": receiver,
                                    "sum_delta": total,
                                    "n_scored": n_scored,
                                    "delta_cos": total / n_scored if n_scored else np.nan,
                                    "receivers": len(cells),
                                    "changed": len(rows),
                                }
                            )
                            key = (region, kind, sender, receiver)
                            held = sums.setdefault(
                                key, [np.zeros((3, len(genes))), np.zeros((3, len(genes))), np.zeros(len(genes)), np.zeros(len(genes))]
                            )
                            held[0] += centred_sums(base[cells], truth[cells], valid, c[0], c[1])
                            held[1] += centred_sums(pred[cells], truth[cells], valid, c[0], c[1])
                            held[2] += shift
                            held[3] += valid.sum(axis=0)
    finally:
        module.train(was_training)
    target_rows = []
    for (region, kind, sender, receiver), (base_sums, pred_sums, shift, count) in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            # A near-constant target has no pattern within the type to lose.
            floor = max(1e-3, float(np.nanmedian(variance[receiver])) * 0.1)
            drop = cosine_from_sums(base_sums, count) - cosine_from_sums(pred_sums, count)
            score = np.where(variance[receiver] >= floor, drop, np.nan)
            moved = np.where(count > 0, shift / count, np.nan)
        if not np.isfinite(score).any():
            continue
        for g in np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")[: int(top_genes)]:
            if np.isfinite(score[g]) and score[g] > 0:  # a target that did not get worse is no target of this flow
                target_rows.append(
                    {
                        "region": region,
                        "kind": kind,
                        "sender": sender,
                        "receiver": receiver,
                        "gene": genes[g],
                        "delta_cos": float(score[g]),
                        "shift": float(moved[g]),
                    }
                )
    columns = [
        "region",
        "draw",
        "kind",
        "sender",
        "receiver",
        "sum_delta",
        "n_scored",
        "delta_cos",
        "receivers",
        "changed",
    ]
    return pd.DataFrame(rows_out, columns=columns), pd.DataFrame(
        target_rows, columns=["region", "kind", "sender", "receiver", "gene", "delta_cos", "shift"]
    )


def flow_matrix(
    flows: pd.DataFrame, *, kind: str = "identity", region: str = "context", minus_null: bool = True
) -> pd.DataFrame:
    """Senders x receivers, the drop in centred cosine from :func:`type_flows`' rows (of one or more sections): sums first, then the ratio. ``identity`` has its
    location-matched null taken off.
    """

    def ratio(which: str) -> pd.DataFrame:
        part = flows[(flows["kind"] == which) & (flows["region"] == region)]
        if not len(part):
            return pd.DataFrame()
        summed = part.groupby(["sender", "receiver"], observed=True)[["sum_delta", "n_scored"]].sum()
        values = summed["sum_delta"] / summed["n_scored"].where(summed["n_scored"] > 0)
        return values.unstack("receiver")

    table = ratio(kind)
    if minus_null and kind == "identity" and len(table):
        null = ratio("identity_null")
        if len(null):
            table = table - null.reindex(index=table.index, columns=table.columns).fillna(0.0)
    return table


def context_use(flows: pd.DataFrame) -> pd.DataFrame:
    """Per region and receiver type, the drop in centred cosine when every other cell is swapped within its type (``state``) or among all cells
    (``arrangement``), and the spread of the draws.
    """
    rows = []
    part = flows[flows["kind"].isin(["all_state", "all_arrangement"])]
    for (region, receiver), sub in part.groupby(["region", "receiver"], observed=True):
        row = {
            "region": region,
            "receiver": receiver,
            "receivers": int(sub[sub["kind"] == "all_state"]["receivers"].sum()),
        }
        for kind, name in (("all_state", "state"), ("all_arrangement", "arrangement")):
            one = sub[sub["kind"] == kind]
            by_draw = one.groupby("draw")[["sum_delta", "n_scored"]].sum()
            ratios = by_draw["sum_delta"] / by_draw["n_scored"].where(by_draw["n_scored"] > 0)
            row[name] = float(one["sum_delta"].sum() / one["n_scored"].sum()) if one["n_scored"].sum() > 0 else np.nan
            row[f"{name}_draw_spread"] = float(ratios.max() - ratios.min()) if len(ratios) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- interventions: gene flows


@dataclass
class GeneFlows:
    """``drop`` (a target's centred cosine within the receiver type, unchanged minus changed, from sums pooled over graphs; NaN: not read) and ``sum_shift``
    ``[S, R, G]``; ``count``, ``expressed`` ``[R, G]``; the names of each axis.
    """

    senders: list[str]
    receivers: list[str]
    genes: list[str]
    drop: np.ndarray
    sum_shift: np.ndarray
    count: np.ndarray
    expressed: np.ndarray

    def delta_cos(self) -> np.ndarray:
        """``[S, R, G]``: the drop in the target's centred cosine within the receiver type (NaN: not read)."""
        return self.drop

    def shift(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.sum_shift / self.count, np.nan)


def gene_flows(
    module,
    graphs,
    labels,
    genes,
    senders,
    *,
    within_labels: bool = True,
    far_radius: float = 0.0,
    head: str = "global",
    per_type: int = 1000,
    seed: int = 0,
    device="cpu",
) -> GeneFlows:
    """Each gene of ``senders`` (those visible in every graph) permuted in the context of the receivers, one pass per sender and graph.

    ``within_labels``: among cells of one type, so a marker's cell type stays where it is and only which cell has how much changes. ``labels`` is one array per graph.
    ``far_radius`` (units of ``pos``): above 0, the cells within it of a receiver keep their values too, so only distant cells change (the ``far`` region).
    """
    genes = [str(g) for g in genes]
    position = {g: k for k, g in enumerate(genes)}
    labels = [np.asarray(v).astype(str) for v in labels]
    types = sorted(set().union(*[set(v.tolist()) for v in labels]))
    hidden = [_hidden_entries(d, "gene") for d in graphs]
    usable = [str(s) for s in dict.fromkeys(senders) if str(s) in position]
    usable = [s for s in usable if all(not h[:, position[s]].all() for h in hidden)]
    if not usable:
        raise ValueError("no sender is visible in every graph.")
    shape = (len(usable), len(types), len(genes))
    pred_sums, sum_shift, scored_count = np.zeros((3, *shape)), np.zeros(shape), np.zeros(shape)
    base_sums = np.zeros((3, len(types), len(genes)))
    count, detected, total = (np.zeros((len(types), len(genes))) for _ in range(3))
    readable = np.ones((len(types), len(genes)), dtype=bool)
    rng = np.random.default_rng(seed)
    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad():
            for graph, label, h in zip(graphs, labels, hidden, strict=True):
                truth = graph.x.double().cpu().numpy()
                visible = torch.as_tensor(~h)
                pos = graph.pos[:, :2].double().cpu().numpy()
                receivers, keep = pick_receivers(pos, label, h.any(axis=1), float(far_radius), rng=rng, per_type=int(per_type))
                base = _predict(module, [graph], head=head, batch_size=1, device=device)[0]
                variance = _type_variance(truth, label, types)
                free = np.flatnonzero(~keep)
                groups = [free[label[free] == t] for t in types] if within_labels else [free]
                cells = {r: receivers[label[receivers] == t] for r, t in enumerate(types)}
                centres = _type_centres(base, truth, h, label, types)
                for r, t in enumerate(types):
                    if len(cells[r]):
                        count[r] += h[cells[r]].sum(axis=0)
                        valid = h[cells[r]] & np.isfinite(base[cells[r]])
                        base_sums[:, r] += centred_sums(base[cells[r]], truth[cells[r]], valid, *centres[t])
                        # A near-constant target has no pattern within the type to lose (as in type_flows).
                        readable[r] &= variance[t] >= max(1e-3, float(np.nanmedian(variance[t])) * 0.1)
                    members = label == t
                    detected[r] += (truth[members] > 0).sum(axis=0)
                    total[r] += members.sum()
                for s, sender in enumerate(usable):
                    column = position[sender]
                    x = graph.x.clone()
                    for group in groups:
                        if len(group) > 1:
                            shuffled = rng.permutation(group)
                            x[group, column] = torch.where(
                                visible[group, column], graph.x[shuffled, column], graph.x[group, column]
                            )
                    copy = graph.clone()
                    copy.x = x
                    pred = _predict(module, [copy], head=head, batch_size=1, device=device)[0]
                    for r in range(len(types)):
                        rows = cells[r]
                        if len(rows):
                            scored = h[rows] & np.isfinite(pred[rows]) & np.isfinite(base[rows])
                            pred_sums[:, s, r] += centred_sums(pred[rows], truth[rows], scored, *centres[types[r]])
                            scored_count[s, r] += scored.sum(axis=0)
                            sum_shift[s, r] += np.where(scored, pred[rows] - base[rows], 0.0).sum(axis=0)
    finally:
        module.train(was_training)
    base_cos = cosine_from_sums(base_sums, count)
    drop = np.where(readable[None], base_cos[None] - cosine_from_sums(pred_sums, scored_count), np.nan)
    return GeneFlows(usable, types, genes, drop, sum_shift, count, detected / np.maximum(total, 1))


def gene_flow_tables(
    flows: GeneFlows, *, min_expressed: float = 0.1, top: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(score, top)``: senders x targets, the largest drop in centred cosine (:meth:`GeneFlows.delta_cos`) over the receiver types expressing the target in at least ``min_expressed`` of their cells (NaN: not
    scored); and the ``top`` (sender, target, receiver type) triples with the signed ``shift`` of the prediction.
    """
    dr2 = flows.delta_cos()
    expressed = flows.expressed >= float(min_expressed)
    masked = np.where(expressed[None, :, :], dr2, np.nan)
    # The largest over the types expressing a target; NaN where no type does (np.nanmax warns on an all-NaN slice).
    best = np.where(np.isfinite(masked).any(axis=1), np.nan_to_num(masked, nan=-np.inf).max(axis=1), np.nan)
    score = pd.DataFrame(best, index=pd.Index(flows.senders, name="sender"), columns=flows.genes)
    score = score.loc[:, score.notna().any(axis=0)]
    shift = flows.shift()
    finite = np.nan_to_num(masked, nan=-np.inf)
    flat = np.argsort(-finite, axis=None, kind="stable")[: int(top)]
    s, r, g = np.unravel_index(flat, finite.shape)
    rows = [
        {
            "sender": flows.senders[a],
            "target": flows.genes[c],
            "receiver": flows.receivers[b],
            "delta_cos": float(dr2[a, b, c]),
            "shift": float(shift[a, b, c]),
        }
        for a, b, c in zip(s, r, g, strict=True)
        if np.isfinite(masked[a, b, c])
    ]
    return score, pd.DataFrame(rows, columns=["sender", "target", "receiver", "delta_cos", "shift"])


def gene_flow_long(flows: GeneFlows, *, min_expressed: float = 0.1) -> pd.DataFrame:
    """Every read (sender, receiver type, target) with its ``delta_cos`` and ``expressed`` (the target's detection rate in that type), for targets detected in at
    least ``min_expressed`` of the type's cells: the input of :func:`lr_support_by_type`.
    """
    dr2 = flows.delta_cos()
    s, r, g = np.nonzero(np.isfinite(dr2) & (flows.expressed >= float(min_expressed))[None, :, :])
    return pd.DataFrame(
        {
            "sender": np.asarray(flows.senders, dtype=object)[s],
            "receiver": np.asarray(flows.receivers, dtype=object)[r],
            "target": np.asarray(flows.genes, dtype=object)[g],
            "delta_cos": dr2[s, r, g],
            "expressed": flows.expressed[r, g],
        }
    )


def gene_influence(flows: GeneFlows, *, min_expressed: float = 0.1) -> pd.DataFrame:
    """Per sender gene of :func:`gene_flows`: ``influence``, the mean drop in centred cosine over every (receiver type, target) read (targets detected in at least
    ``min_expressed`` of the type's cells), and ``top``, the largest. The mean is unbiased where a sender moves nothing (drops of either sign cancel), so it can
    be set against the untrained twin's; ``top`` finds a sender that moves one target strongly. ``reads``: how many (type, target) it was read on.
    """
    drop = np.where((flows.expressed >= float(min_expressed))[None], flows.delta_cos(), np.nan)
    flat = drop.reshape(len(flows.senders), -1)
    reads = np.isfinite(flat).sum(axis=1)
    with np.errstate(invalid="ignore"):
        influence = np.where(reads > 0, np.nansum(flat, axis=1) / np.maximum(reads, 1), np.nan)
        top = np.where(reads > 0, np.nan_to_num(flat, nan=-np.inf).max(axis=1), np.nan)
    return pd.DataFrame({"influence": influence, "top": top, "reads": reads}, index=pd.Index(flows.senders, name="gene"))


def moran_genes(values, pos, sections, *, k: int, max_cells: int = 20000, chunk: int = 2000, seed: int = 0) -> np.ndarray:
    """``[G]``: :func:`moran` of every column of ``values`` at once (``k`` nearest cells within each section, a seeded sample of ``max_cells`` centres per
    section), each column centred within its section.
    """
    values = np.asarray(values, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    sections = np.asarray(sections).astype(str)
    rng = np.random.default_rng(seed)
    top, bottom = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    for section in sorted(set(sections.tolist())):
        where = np.flatnonzero(sections == section)
        if len(where) <= int(k):
            continue
        z = values[where] - values[where].mean(axis=0)
        chosen = rng.choice(len(where), size=min(int(max_cells), len(where)), replace=False)
        _, near = cKDTree(pos[where]).query(pos[where][chosen], k=int(k) + 1)
        for start in range(0, len(chosen), int(chunk)):
            rows, neighbours = chosen[start : start + int(chunk)], near[start : start + int(chunk), 1:]
            top += (z[rows] * z[neighbours].mean(axis=1)).sum(axis=0)
            bottom += (z[rows] ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(bottom > 0, top / bottom, np.nan)


def gene_spatial(x, genes, labels, sections, pos, *, k_local: int = 8, n_centres: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per gene, how it varies in space on the cells read: ``mean``, ``detected``; ``moran_local`` (:func:`moran_genes` at ``k_local`` neighbours) and
    ``moran_global`` (``gene_scales.global_moran``: inverse-distance weights over the whole section), each on the expression with every (section, type)'s mean
    taken out, so a gene that only marks a cell type is not spatially variable here; ``moran_local_raw``, the same at ``k_local`` with the types left in (the usual
    SVG statistic).
    """
    from interscale.evaluation.gene_scales import global_moran

    x = np.asarray(x, dtype=np.float64)
    labels, sections = np.asarray(labels).astype(str), np.asarray(sections).astype(str)
    groups = pd.Series(sections).str.cat(pd.Series(labels), sep="|").to_numpy()
    residual = x - pd.DataFrame(x).groupby(groups).transform("mean").to_numpy()
    codes = pd.factorize(sections)[0]
    return pd.DataFrame(
        {
            "mean": x.mean(axis=0),
            "detected": (x > 0).mean(axis=0),
            "moran_local": moran_genes(residual, pos, sections, k=k_local, seed=seed),
            "moran_global": global_moran(residual, codes, np.asarray(pos, dtype=np.float64)[:, :2], n_centres=n_centres, seed=seed),
            "moran_local_raw": moran_genes(x, pos, sections, k=k_local, seed=seed),
        },
        index=pd.Index([str(g) for g in genes], name="gene"),
    )


def edge_recovery(score: pd.DataFrame, edges) -> pd.DataFrame:
    """Where the truth is planted: per target, the best rank of a planted sender among the senders, against the chance of drawing one (a check of the readout, not of a model)."""
    planted: dict[str, set] = {}
    for sender, target in edges:
        planted.setdefault(str(target), set()).add(str(sender))
    rows = []
    for target, senders in planted.items():
        if target not in score.columns:
            continue
        column = score[target].dropna()
        present = [s for s in senders if s in column.index]
        if not present or len(column) < 2:
            continue
        ranks = column.rank(ascending=False, method="min")
        rows.append(
            {
                "target": target,
                "best_rank": float(ranks[present].min()),
                "senders_scored": len(column),
                "planted": len(present),
                "chance_top1": len(present) / len(column),
            }
        )
    return pd.DataFrame(rows, columns=["target", "best_rank", "senders_scored", "planted", "chance_top1"])


# ----------------------------------------------------------------------------------------------- attention, read biologically


def neighbour_mean(values, pos, *, k: int = 16) -> np.ndarray:
    """Each cell's mean of ``values`` (``[N]`` or ``[N, D]``) over itself and its ``k`` nearest cells."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    values = np.asarray(values, dtype=np.float64)
    k = min(int(k) + 1, len(pos))
    _, near = cKDTree(pos).query(pos, k=k)
    near = near.reshape(len(pos), k)
    return values[near].mean(axis=1)


def _centre_within(values, groups) -> np.ndarray:
    """``values`` minus the mean of each group (``[N]`` or ``[N, D]``)."""
    values = np.asarray(values, dtype=np.float64)
    codes, uniques = pd.factorize(pd.Series(groups).astype(str))
    squeeze = values.ndim == 1
    v = values[:, None] if squeeze else values
    totals = np.zeros((len(uniques), v.shape[1]))
    np.add.at(totals, codes, v)
    means = totals / np.bincount(codes, minlength=len(uniques))[:, None]
    out = v - means[codes]
    return out[:, 0] if squeeze else out


def attention_biology(
    flow_out, flow_in, x, genes, labels, sections, pos, *, k: int = 16, n_top: int = 40, min_cells: int = 30
) -> dict[str, pd.DataFrame]:
    """What sits in the sources and sinks of the net attention. ``flow_out - flow_in`` per cell (positive: it is read more than it reads), smoothed over ``k`` neighbours.

    * ``by_type``: per section and type the mean smoothed net flow, its standard deviation, the share of the type's cells in the top and bottom decile.
    * ``genes``: per gene the correlation of a cell's expression with the smoothed net flow within section and type: identity and section level are out. An effect size,
      not a test: neighbouring cells are not independent; read it against the twin.
    * ``composition``: per type, the correlation of the fraction of that type among the neighbours with the net flow, within section and type.
    """
    genes = [str(g) for g in genes]
    labels, sections = np.asarray(labels).astype(str), np.asarray(sections).astype(str)
    net = np.asarray(flow_out, dtype=np.float64) - np.asarray(flow_in, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    smooth = np.zeros(len(net))
    types = sorted(set(labels.tolist()))
    onehot = np.stack([labels == t for t in types], axis=1).astype(np.float64)
    fraction = np.zeros_like(onehot)
    rows = []
    for section in sorted(set(sections.tolist())):
        where = np.flatnonzero(sections == section)
        smooth[where] = neighbour_mean(net[where], pos[where], k=k)
        fraction[where] = neighbour_mean(onehot[where], pos[where], k=32)
        low, high = np.quantile(smooth[where], [0.1, 0.9])
        for t in types:
            cells = where[labels[where] == t]
            if len(cells) >= int(min_cells):
                rows.append(
                    {
                        "section": section,
                        "cell_type": t,
                        "cells": len(cells),
                        "mean_net": float(smooth[cells].mean()),
                        "sd_net": float(smooth[cells].std()),
                        "share_top_decile": float((smooth[cells] >= high).mean()),
                        "share_bottom_decile": float((smooth[cells] <= low).mean()),
                    }
                )
    groups = pd.Series(sections).str.cat(pd.Series(labels), sep="|").to_numpy()
    resid_net = _centre_within(smooth, groups)
    resid_x = _centre_within(x, groups)
    denom = np.sqrt((resid_x**2).sum(axis=0) * (resid_net**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, resid_x.T @ resid_net / denom, np.nan)
    table = (
        pd.DataFrame({"gene": genes, "r": r, "mean_expression": x.mean(axis=0)})
        .sort_values("r", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    resid_fraction = _centre_within(fraction, groups)
    denom = np.sqrt((resid_fraction**2).sum(axis=0) * (resid_net**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.where(denom > 0, resid_fraction.T @ resid_net / denom, np.nan)
    return {
        "by_type": pd.DataFrame(rows),
        "genes": table,
        "composition": pd.DataFrame({"cell_type": types, "r_with_neighbour_fraction": rc}),
        "top_up": table.head(int(n_top)),
        "top_down": table.tail(int(n_top)).iloc[::-1],
    }


def attention_pairs(
    receiver, source, flow, labels, pos, *, grid_um: float = 100.0, n_perm: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Source type x receiver type of the sampled receivers' top sources (``flow_maps.sample_pairs``), against labels shuffled within grid squares of ``grid_um``.

    Local shuffling keeps which types live where, so what is left is more than co-location explains. Rows: ``source``, ``receiver``, ``flow`` (summed net flow), ``expected``
    (over the shuffles), ``log2_ratio``, ``z``, ``median_distance_um``.
    """
    receiver, source, flow = np.asarray(receiver), np.asarray(source), np.asarray(flow, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    columns = ["source", "receiver", "flow", "expected", "log2_ratio", "z", "median_distance_um"]
    if not len(receiver):
        return pd.DataFrame(columns=columns)
    rec = np.repeat(receiver[:, None], source.shape[1], axis=1).ravel()
    src, weight = source.ravel(), np.nan_to_num(flow.ravel(), nan=0.0)
    keep = (src >= 0) & (weight > 0)
    rec, src, weight = rec[keep], src[keep], weight[keep]
    if not len(rec):
        return pd.DataFrame(columns=columns)
    types, codes = np.unique(labels, return_inverse=True)
    n_types = len(types)

    def table(code):
        out = np.zeros((n_types, n_types))
        np.add.at(out, (code[src], code[rec]), weight)
        return out

    observed = table(codes)
    cell = np.floor(pos / float(grid_um)).astype(np.int64)
    square = pd.factorize(pd.Series(cell[:, 0]).astype(str) + "_" + pd.Series(cell[:, 1]).astype(str))[0]
    members = pd.Series(np.arange(len(codes))).groupby(square).indices
    rng = np.random.default_rng(seed)
    nulls = np.zeros((int(n_perm), n_types, n_types))
    for b in range(int(n_perm)):
        shuffled = codes.copy()
        for rows in members.values():
            shuffled[rows] = codes[rng.permutation(rows)]
        nulls[b] = table(shuffled)
    mean, sd = nulls.mean(axis=0), nulls.std(axis=0)
    distance = np.hypot(*(pos[src] - pos[rec]).T)
    rows = []
    for a in range(n_types):
        for b in range(n_types):
            mask = (codes[src] == a) & (codes[rec] == b)
            rows.append(
                {
                    "source": types[a],
                    "receiver": types[b],
                    "flow": float(observed[a, b]),
                    "expected": float(mean[a, b]),
                    "log2_ratio": float(np.log2((observed[a, b] + 1e-9) / (mean[a, b] + 1e-9)))
                    if mean[a, b] > 0
                    else np.nan,
                    "z": float((observed[a, b] - mean[a, b]) / sd[a, b]) if sd[a, b] > 0 else np.nan,
                    "median_distance_um": float(np.median(distance[mask])) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------------------------- programmes


def programmes(r: DimensionReadout, *, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(loadings [G, k], share [k], scores [N, k])``: the prediction's first ``k`` directions rotated by varimax, on the rows the decoder was scored on.

    ``share`` is each programme's share of the prediction's variance (largest first), ``scores`` each cell's position along it (the centred predicted profile projected on its loadings).
    """
    u, s = _effective_basis(r)
    k = int(min(k, u.shape[1]))
    rotated, _ = varimax(u[:, :k] * s[:k])
    variance = (rotated**2).sum(axis=0)
    order = np.argsort(-variance, kind="stable")
    rotated, variance = _signed(rotated[:, order]), variance[order]
    total = float((s**2).sum())
    rows = _kept_rows(r)
    centred = r.z[rows] - r.z[rows].mean(axis=0)
    # z W^T L / |L|^2, without building the [N, G] prediction
    scores = centred @ (r.weight.T @ rotated) / np.maximum((rotated**2).sum(axis=0), 1e-30)
    return rotated, variance / max(total, 1e-300), scores


def moran(values, pos, sections, *, k: int, max_cells: int = 20000, seed: int = 0) -> float:
    """Moran's I of ``values`` (already centred as wanted) on the ``k`` nearest cells within each section, the receivers of the sum a seeded sample of ``max_cells`` per section."""
    values = np.asarray(values, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    sections = np.asarray(sections).astype(str)
    rng = np.random.default_rng(seed)
    top, bottom = 0.0, 0.0
    for section in sorted(set(sections.tolist())):
        where = np.flatnonzero(sections == section)
        if len(where) <= int(k):
            continue
        z = values[where] - values[where].mean()
        chosen = rng.choice(len(where), size=min(int(max_cells), len(where)), replace=False)
        _, near = cKDTree(pos[where]).query(pos[where][chosen], k=int(k) + 1)
        top += float((z[chosen] * z[near[:, 1:]].mean(axis=1)).sum())
        bottom += float((z[chosen] ** 2).sum())
    return top / bottom if bottom > 0 else float("nan")


def eta_squared(values, labels, sections) -> float:
    """The share of ``values``' variance within sections that the labels explain."""
    within_section = _centre_within(values, sections)
    groups = (
        pd.Series(np.asarray(sections).astype(str))
        .str.cat(pd.Series(np.asarray(labels).astype(str)), sep="|")
        .to_numpy()
    )
    within_type = _centre_within(values, groups)
    total = float((within_section**2).sum())
    return 1.0 - float((within_type**2).sum()) / total if total > 0 else float("nan")


def neighbour_r2(residual, labels, pos, sections, *, k: int = 32, ridge: float = 1e-2) -> float:
    """How much of a within-type residual the type composition around a cell explains, fitted on one half of each section and scored on the other (split by position)."""
    labels, sections = np.asarray(labels).astype(str), np.asarray(sections).astype(str)
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    residual = np.asarray(residual, dtype=np.float64)
    types = sorted(set(labels.tolist()))
    onehot = np.stack([labels == t for t in types], axis=1).astype(np.float64)
    design = np.zeros_like(onehot)
    half = np.zeros(len(labels), bool)
    for section in sorted(set(sections.tolist())):
        where = np.flatnonzero(sections == section)
        design[where] = neighbour_mean(onehot[where], pos[where], k=k)
        half[where] = pos[where, 0] < np.median(pos[where, 0])
    design = design - design.mean(axis=0)
    scores = []
    for fit in (half, ~half):
        test = ~fit
        if fit.sum() < 2 or test.sum() < 2:
            continue
        a = design[fit]
        coef = np.linalg.solve(a.T @ a + float(ridge) * len(a) * np.eye(a.shape[1]), a.T @ residual[fit])
        rss = float(((residual[test] - design[test] @ coef) ** 2).sum())
        tss = float(((residual[test] - residual[test].mean()) ** 2).sum())
        if tss > 0:
            scores.append(1.0 - rss / tss)
    return float(np.mean(scores)) if scores else float("nan")


def programme_class(share, eta2, moran_within, neighbour, handling_share, *, thresholds: dict | None = None) -> str:
    """One word for a programme, by stated thresholds (:data:`CLASS_THRESHOLDS`). A sorting aid, in this order: ``weak``, ``handling``, ``niche``, ``spatial state``, ``identity``, ``other``.

    ``niche`` and ``spatial state`` come before ``identity``: a programme can be mostly a cell type and still vary within it, and the within-type part is the interesting one.
    """
    t = CLASS_THRESHOLDS | (thresholds or {})
    if not share >= t["weak_share"]:
        return "weak"
    if handling_share >= t["handling_share"]:
        return "handling"
    spatial = moran_within >= t["moran"]
    if spatial and neighbour >= t["neighbour_r2"]:
        return "niche"
    if spatial:
        return "spatial state"
    if eta2 >= t["identity_eta2"]:
        return "identity"
    return "other"


def programme_cards(
    r: DimensionReadout,
    genes,
    labels,
    sections,
    *,
    conditions=None,
    gene_sets: dict | None = None,
    handling_genes=(),
    k: int = 12,
    top: int = 30,
    min_cells: int = 100,
    thresholds: dict | None = None,
    seed: int = 0,
) -> dict:
    """Per programme of :func:`programmes`, on the same rows as ``labels``, ``sections`` and ``conditions`` (arrays over the readout's kept rows):

    ``cards``: ``programme``, ``share``, ``class``, ``top_genes``, ``top_negative``, ``pathway`` (the best set at q < 0.05), ``top_type``, ``eta2``, ``moran_within_near`` (k 8), ``moran_within_far`` (k 128),
    ``neighbour_r2``, ``handling_share``, and with conditions ``condition_effect`` / ``condition_type`` (the largest standardised difference of the second condition minus the first, within a type).
    ``by_type``: mean score per programme and type, in within-section SD. ``genes``: loadings of the top genes. ``enrichment``. ``condition``: every type's difference. ``loadings``, ``scores``: arrays.
    """
    genes = [str(g) for g in genes]
    rows = _kept_rows(r)
    pos = r.pos[rows]
    labels, sections = np.asarray(labels).astype(str), np.asarray(sections).astype(str)
    if len(labels) != len(pos) or len(sections) != len(pos):
        raise ValueError(f"labels and sections must hold one entry per kept row of the readout ({len(pos)}).")
    loadings, share, scores = programmes(r, k=k)
    handling = {str(g) for g in handling_genes}
    groups = pd.Series(sections).str.cat(pd.Series(labels), sep="|").to_numpy()
    cards, by_type, gene_rows, enriched, condition_rows = [], [], [], [], []
    conditions = None if conditions is None else np.asarray(conditions).astype(str)
    condition_names = [] if conditions is None else sorted(set(conditions.tolist()))
    for c in range(loadings.shape[1]):
        score = scores[:, c]
        order = np.argsort(-loadings[:, c], kind="stable")
        up, down = order[: int(top)], order[::-1][: int(top)]
        down = [g for g in down if loadings[g, c] < 0]
        up = [g for g in up if loadings[g, c] > 0]
        strongest = np.argsort(-np.abs(loadings[:, c]), kind="stable")[: int(top)]
        for rank, g in enumerate(strongest):
            gene_rows.append({"programme": c, "rank": rank, "gene": genes[g], "loading": float(loadings[g, c])})
        weight = loadings[strongest, c] ** 2
        hand = (
            float(weight[[genes[g] in handling for g in strongest]].sum() / weight.sum()) if weight.sum() > 0 else 0.0
        )
        pathway = ""
        if gene_sets:
            for side, members in (("+", up), ("-", down)):
                found = enrichment([genes[g] for g in members], genes, gene_sets)
                if len(found):
                    enriched.append(found.assign(programme=c, side=side))
            joined = [e for e in enriched if int(e["programme"].iloc[0]) == c]
            if joined:
                best = pd.concat(joined).sort_values("q", kind="stable").iloc[0]
                if best["q"] < 0.05:
                    pathway = f"{best['set']} ({best['side']}, q={best['q']:.1e})"
        within_section = _centre_within(score, sections)
        spread = float(within_section.std()) or 1.0
        means = pd.Series(within_section).groupby(labels).mean() / spread
        counts = pd.Series(labels).value_counts()
        by_type += [{"programme": c, "cell_type": t, "z": float(v), "cells": int(counts[t])} for t, v in means.items()]
        eligible = means[[counts[t] >= int(min_cells) for t in means.index]]
        residual = _centre_within(score, groups)
        near = moran(residual, pos, sections, k=8, seed=seed)
        far = moran(residual, pos, sections, k=128, seed=seed)
        eta2 = eta_squared(score, labels, sections)
        neighbour = neighbour_r2(residual, labels, pos, sections)
        card = {
            "programme": c,
            "share": float(share[c]),
            "class": programme_class(float(share[c]), eta2, near, neighbour, hand, thresholds=thresholds),
            "top_genes": " ".join(genes[g] for g in up[:8]),
            "top_negative": " ".join(genes[g] for g in down[:5]),
            "pathway": pathway,
            "top_type": str(eligible.abs().idxmax()) if len(eligible) else "",
            "top_type_z": float(eligible[eligible.abs().idxmax()]) if len(eligible) else np.nan,
            "eta2": eta2,
            "moran_within_near": near,
            "moran_within_far": far,
            "neighbour_r2": neighbour,
            "handling_share": hand,
        }
        if len(condition_names) == 2:
            first, second = condition_names
            best_effect, best_type = 0.0, ""
            for t in sorted(set(labels.tolist())):
                a, b = score[(labels == t) & (conditions == first)], score[(labels == t) & (conditions == second)]
                if min(len(a), len(b)) < int(min_cells):
                    continue
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                effect = float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0
                condition_rows.append(
                    {
                        "programme": c,
                        "cell_type": t,
                        "effect": effect,
                        f"cells_{first}": len(a),
                        f"cells_{second}": len(b),
                    }
                )
                if abs(effect) > abs(best_effect):
                    best_effect, best_type = effect, t
            card |= {"condition_effect": best_effect, "condition_type": best_type}
        cards.append(card)
    return {
        "cards": pd.DataFrame(cards),
        "by_type": pd.DataFrame(by_type),
        "genes": pd.DataFrame(gene_rows),
        "enrichment": pd.concat(enriched, ignore_index=True) if enriched else pd.DataFrame(),
        "condition": pd.DataFrame(condition_rows),
        "loadings": loadings,
        "scores": scores,
    }


def match_programmes(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """Programmes of two models on the same genes, matched one to one by the absolute cosine of their loadings. ``sign``: -1 where ``b``'s points the other way."""
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    cosine = (a.T @ b) / np.maximum(np.outer(na, nb), 1e-30)
    rows, cols = linear_sum_assignment(-np.abs(cosine))
    return (
        pd.DataFrame(
            {
                "a": rows,
                "b": cols,
                "cosine": np.abs(cosine[rows, cols]),
                "sign": np.sign(cosine[rows, cols]).astype(int),
            }
        )
        .sort_values("a", kind="stable")
        .reset_index(drop=True)
    )
