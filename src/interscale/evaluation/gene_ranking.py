"""Gene rankings of cell-cell interaction (``.claude/ccc_model_plan.md``, E7, 2026-11-22).

The question is not how well a model reconstructs expression but which genes it treats as interacting: which genes **respond** to other cells, which
genes in other cells **drive** them, and which **sender gene -> responder gene** pairs carry it. Each ranking is read off a trained model by
intervention, per receiver group (a cell type, or every cell where there are no types), and scored against planted truth where it exists. The
interventions are :mod:`interscale.evaluation.interpretation`'s: receivers keep their own rows, only other cells change, and the score is a target's
centred cosine within the receiver group (pooled over graphs), so a level or scale error of the model cancels.

* **Responders** (:func:`context_flows`, kind ``all_state``): every other cell's visible row swapped with another cell's of the same label. What a
  target's prediction loses is what the model took from the arrangement of the other cells' states, beyond their composition. A target scores only
  if it varies within the group *and* the model explains that variation from other cells; one the model cannot predict, or one constant within the
  group, scores nothing. ``all_arrangement`` (swapped among all cells) is reported beside it: it also removes composition.
* **Drivers and pairs** (:func:`interpretation.gene_flows`): one sender gene permuted within label among the other cells; each target's drop is the
  pair's score, the mean over targets (the sender itself left out) the sender's.
* **The same gene in other cells** (the pair ``g -> g``): where a target is visible in the other cells, the part of its responder score that the same
  gene there carries -- spatial smoothness, shared regional states, segmentation spillover: not an interaction between genes. ``responder_cross``
  takes it off. With the target hidden in every cell (programme or channel masks, :func:`gene_folds`) the pair is not read and nothing is taken off.

**Sender types** (:func:`context_flows`, ``identity`` minus ``identity_null``): a type's cells take random other-type profiles, against the same change
on as many nearby cells of other types (co-location). Per receiver group and target: which sender type it depends on.

**Bars**, model-free, on the same cells: within-group spatial autocorrelation (:func:`within_group_moran`: spatially variable genes), within-group
variance (:func:`within_group_variance`: variable genes), and a ridge of each target on the neighbours' within-group residuals and label shares with the
target's own neighbour feature left out (:func:`neighbour_ridge`: a linear cross-gene neighbourhood model, NCEM-like). Truth: :func:`roles_truth`
(gene roles, synth_spot), :func:`programme_truth` (programmes planted on real cells, ``semi_synthetic``); :func:`ranking_metrics` scores a ranking.

Nothing here reads attention or a reconstruction score as such; every number is a change of the model's prediction under a stated intervention, and is
read against the untrained twin and the bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from interscale.evaluation.context_test import _hidden_entries, _predict
from interscale.evaluation.interaction_readout import _auroc, _average_precision
from interscale.evaluation.interpretation import (
    EVERYONE,
    GeneFlows,
    _changed,
    _rng_untouched,
    _scoring_every_cell,
    _type_centres,
    _type_perturbations,
    _type_variance,
    centred_sums,
    cosine_from_sums,
    moran_genes,
    pick_receivers,
    scored_entries,
)

CONTEXT_KINDS = ("identity", "identity_null", "state", "all_state", "all_arrangement")
#: synth_spot's gene roles (``var['role']``): who responds, who drives, and the genes that vary in space or by sample without any sender.
RESPONDER_ROLES = ("receiver", "response")
DRIVER_ROLES = ("sender", "source")
HARD_ROLES = ("spatial", "marker", "batch", "condition")


# ----------------------------------------------------------------------------------------------- context flows: responders, sender types


@dataclass
class ContextFlows:
    """``drop`` ``[K, S, R, G]``: a target's centred cosine within the receiver group, unchanged minus changed, per perturbation kind and sender label
    (:data:`interpretation.EVERYONE` for the ``all_*`` kinds); NaN where not read. ``base``, ``count``, ``expressed`` ``[R, G]``."""

    kinds: list[str]
    senders: list[str]
    receivers: list[str]
    genes: list[str]
    drop: np.ndarray
    base: np.ndarray
    count: np.ndarray
    expressed: np.ndarray

    def get(self, kind: str, sender: str = EVERYONE) -> np.ndarray:
        """``[R, G]`` of one kind and sender label (all NaN when it was not run)."""
        if kind not in self.kinds or sender not in self.senders:
            return np.full((len(self.receivers), len(self.genes)), np.nan)
        return self.drop[self.kinds.index(kind), self.senders.index(sender)]

    def responder(self) -> np.ndarray:
        """``[R, G]``: the drop when every other cell's state is swapped within its label (``all_state``)."""
        return self.get("all_state")

    def sender_types(self, *, minus_null: bool = True) -> np.ndarray:
        """``[S, R, G]`` over the real sender labels (``self.senders`` without EVERYONE): ``identity``, minus ``identity_null`` where read."""
        labels = [s for s in self.senders if s != EVERYONE]
        out = np.stack([self.get("identity", s) for s in labels]) if labels else np.zeros((0, len(self.receivers), len(self.genes)))
        if minus_null and labels:
            null = np.stack([self.get("identity_null", s) for s in labels])
            out = out - np.nan_to_num(null)
        return out


def context_flows(
    module,
    graphs,
    labels,
    genes,
    *,
    targets: str = "hidden",
    kinds=CONTEXT_KINDS,
    head: str = "global",
    per_type: int = 1000,
    seed: int = 0,
    device="cpu",
) -> ContextFlows:
    """The perturbations of :func:`interpretation.type_flows` over every graph, pooled, with every target kept (not only the top ones).

    ``labels``: one array per graph (a constant array: one group). ``targets``: ``"hidden"`` (the graphs' hidden entries) or ``"all"`` (every entry of
    the receivers; a model blind to its own cell, :func:`interpretation.scored_entries`). ``kinds``: a subset of :data:`CONTEXT_KINDS` (a kind left
    out costs no prediction). Receivers keep their rows; per graph up to ``per_type`` of them per label (``pick_receivers``).
    """
    genes = [str(g) for g in genes]
    labels = [np.asarray(v).astype(str) for v in labels]
    types = sorted(set().union(*[set(v.tolist()) for v in labels]))
    senders = types + [EVERYONE]
    kinds = [k for k in CONTEXT_KINDS if k in set(kinds)]
    n_k, n_s, n_r, n_g = len(kinds), len(senders), len(types), len(genes)
    pred_sums, pcount = np.zeros((n_k, n_s, n_r, 3, n_g)), np.zeros((n_k, n_s, n_r, n_g))
    base_sums, count = np.zeros((n_r, 3, n_g)), np.zeros((n_r, n_g))
    detected, total = np.zeros((n_r, n_g)), np.zeros((n_r, 1))
    readable = np.ones((n_r, n_g), dtype=bool)
    rng = np.random.default_rng(seed)
    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad(), _scoring_every_cell(module, targets == "all"):
            for graph, label in zip(graphs, labels, strict=True):
                truth = graph.x.double().cpu().numpy()
                hidden = _hidden_entries(graph, "gene")
                target = scored_entries(graph, module, targets)
                visible = torch.as_tensor(~hidden)
                pos = graph.pos[:, :2].double().cpu().numpy()
                receivers, keep = pick_receivers(pos, label, target.any(axis=1), 0.0, rng=rng, per_type=int(per_type))
                base = _predict(module, [graph], head=head, batch_size=1, device=device)[0]
                variance = _type_variance(truth, label, types)
                cells = {r: receivers[label[receivers] == t] for r, t in enumerate(types)}
                centres = _type_centres(base, truth, target, label, types)
                for r, t in enumerate(types):
                    members = label == t
                    detected[r] += (truth[members] > 0).sum(axis=0)
                    total[r] += members.sum()
                    if len(cells[r]):
                        valid = target[cells[r]] & np.isfinite(base[cells[r]])
                        count[r] += valid.sum(axis=0)
                        base_sums[r] += centred_sums(base[cells[r]], truth[cells[r]], valid, *centres[t])
                        readable[r] &= variance[t] >= max(1e-3, float(np.nanmedian(variance[t])) * 0.1)
                for kind, sender, rows, donors in _type_perturbations(label, keep, pos, rng):
                    if kind not in kinds:
                        continue
                    pred = _predict(module, [_changed(graph, visible, rows, donors)], head=head, batch_size=1, device=device)[0]
                    k, s = kinds.index(kind), senders.index(sender)
                    for r, t in enumerate(types):
                        if len(cells[r]):
                            rows_r = cells[r]
                            scored = target[rows_r] & np.isfinite(pred[rows_r]) & np.isfinite(base[rows_r])
                            pred_sums[k, s, r] += centred_sums(pred[rows_r], truth[rows_r], scored, *centres[t])
                            pcount[k, s, r] += scored.sum(axis=0)
    finally:
        module.train(was_training)
    base_cos = np.stack([cosine_from_sums(base_sums[r], count[r]) for r in range(n_r)]) if n_r else np.zeros((0, n_g))
    drop = np.full((n_k, n_s, n_r, n_g), np.nan)
    for k in range(n_k):
        for s in range(n_s):
            for r in range(n_r):
                if pcount[k, s, r].any():
                    changed = cosine_from_sums(pred_sums[k, s, r], pcount[k, s, r])
                    drop[k, s, r] = np.where(readable[r], base_cos[r] - changed, np.nan)
    expressed = detected / np.maximum(total, 1)
    return ContextFlows(kinds, senders, types, genes, drop, np.where(readable, base_cos, np.nan), count, expressed)


# ----------------------------------------------------------------------------------------------- folds: every gene hidden in every cell once


def gene_folds(n_genes: int, fraction: float, *, partitions: int = 2, seed: int = 0) -> list[np.ndarray]:
    """Gene index folds of about ``fraction`` of the genes each (``round(1 / fraction)`` per partition), ``partitions`` independent shuffles.

    For a model trained with channel masks (a random share of the genes hidden in every cell): each fold hidden in every cell is a training-like input,
    every gene is a target once per partition, and two genes share a fold in every partition only by chance, so (almost) every pair is read.
    """
    if not 0 < float(fraction) < 1:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}.")
    rng = np.random.default_rng(seed)
    n_folds = max(2, int(round(1.0 / float(fraction))))
    out = []
    for _ in range(int(partitions)):
        out += [np.sort(fold) for fold in np.array_split(rng.permutation(int(n_genes)), n_folds) if len(fold)]
    return out


def hide_genes(graphs, columns) -> list:
    """Copies of ``graphs`` with the gene columns ``columns`` hidden in every cell (``gene_mask``; no cell hidden whole); the originals untouched.

    ``mask`` is set as the gene-masking dataloader sets it (the cells with a hidden entry), which ``tl.masking.apply_mask`` requires to exist."""
    out = []
    for graph in graphs:
        copy = graph.clone()
        mask = torch.zeros(int(graph.num_nodes), int(graph.x.shape[1]), dtype=torch.bool)
        mask[:, torch.as_tensor(np.asarray(columns, dtype=np.int64))] = True
        copy.gene_mask = mask
        copy.mask = mask.any(dim=1)
        out.append(copy)
    return out


def show_everything(graphs) -> list:
    """Copies of ``graphs`` with nothing hidden (no ``gene_mask``, an all-False ``mask``): the clean pass for ``targets='all'``."""
    out = []
    for graph in graphs:
        copy = graph.clone()
        if getattr(copy, "gene_mask", None) is not None:
            del copy.gene_mask
        copy.mask = torch.zeros(int(graph.num_nodes), dtype=torch.bool)
        out.append(copy)
    return out


def _nanmean_stack(arrays) -> np.ndarray:
    stack = np.stack(arrays)
    finite = np.isfinite(stack)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(finite.any(axis=0), np.where(finite, stack, 0.0).sum(axis=0) / finite.sum(axis=0), np.nan)


def merge_gene_flows(parts: list[GeneFlows]) -> GeneFlows:
    """One :class:`GeneFlows` from several runs on the same receivers and genes (the folds): each read averaged where it was read, counts added."""
    if not parts:
        raise ValueError("nothing to merge.")
    receivers, genes = parts[0].receivers, parts[0].genes
    if any(p.receivers != receivers or p.genes != genes for p in parts):
        raise ValueError("the parts read different receiver groups or genes.")
    senders = list(dict.fromkeys(s for p in parts for s in p.senders))
    shape = (len(senders), len(receivers), len(genes))
    drops, shift = [], np.zeros(shape)
    for p in parts:
        full = np.full(shape, np.nan)
        rows = [senders.index(s) for s in p.senders]
        full[rows] = p.drop
        drops.append(full)
        shift[rows] += p.sum_shift
    base = _nanmean_stack([p.base if p.base is not None else np.full((len(receivers), len(genes)), np.nan) for p in parts])
    count = np.sum([p.count for p in parts], axis=0)
    return GeneFlows(senders, receivers, genes, _nanmean_stack(drops), shift, count, parts[0].expressed, base)


def merge_context_flows(parts: list[ContextFlows]) -> ContextFlows:
    """One :class:`ContextFlows` from several runs on the same kinds, labels and genes (the folds): reads averaged where read, counts added."""
    if not parts:
        raise ValueError("nothing to merge.")
    first = parts[0]
    if any((p.kinds, p.senders, p.receivers, p.genes) != (first.kinds, first.senders, first.receivers, first.genes) for p in parts):
        raise ValueError("the parts read different kinds, labels or genes.")
    return ContextFlows(
        first.kinds, first.senders, first.receivers, first.genes,
        _nanmean_stack([p.drop for p in parts]), _nanmean_stack([p.base for p in parts]), np.sum([p.count for p in parts], axis=0), first.expressed,
    )  # fmt: skip


# ----------------------------------------------------------------------------------------------- the three rankings


def responder_table(context: ContextFlows, flows: GeneFlows | None = None) -> pd.DataFrame:
    """One row per (receiver group, target): ``base`` (the unchanged model's centred cosine within the group), ``responder`` (``all_state`` drop),
    ``arrangement`` (``all_arrangement`` drop), ``self`` (the drop when the target's own gene is permuted in the other cells; NaN where not read),
    ``responder_cross`` (``responder - self`` where ``self`` was read, else ``responder``: the ranking score), ``expressed``, ``count``."""
    resp, arr = context.responder(), context.get("all_arrangement")
    self_drop = np.full_like(resp, np.nan)
    if flows is not None:
        if flows.receivers != context.receivers or flows.genes != context.genes:
            raise ValueError("the gene flows read other receiver groups or genes than the context flows.")
        for g, gene in enumerate(context.genes):
            if gene in flows.senders:
                self_drop[:, g] = flows.drop[flows.senders.index(gene), :, g]
    cross = np.where(np.isfinite(self_drop), resp - self_drop, resp)
    rows = []
    for r, receiver in enumerate(context.receivers):
        rows.append(
            pd.DataFrame(
                {
                    "receiver": receiver,
                    "gene": context.genes,
                    "base": context.base[r],
                    "responder": resp[r],
                    "arrangement": arr[r],
                    "self": self_drop[r],
                    "responder_cross": cross[r],
                    "expressed": context.expressed[r],
                    "count": context.count[r],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _off_diagonal(flows: GeneFlows, min_expressed: float) -> np.ndarray:
    """``[S, R, G]`` drops with the sender's own gene as target and targets expressed below ``min_expressed`` in a group set to NaN."""
    drop = np.where((flows.expressed >= float(min_expressed))[None], flows.drop, np.nan).copy()
    position = {g: k for k, g in enumerate(flows.genes)}
    for s, sender in enumerate(flows.senders):
        if sender in position:
            drop[s, :, position[sender]] = np.nan
    return drop


def driver_table(flows: GeneFlows, *, min_expressed: float = 0.1) -> pd.DataFrame:
    """One row per (receiver group, sender gene), and per sender pooled over groups (receiver :data:`EVERYONE`): ``influence`` (the mean drop over the
    targets read, the sender's own gene left out: drops of either sign cancel where it moves nothing), ``positive`` (the summed positive drops), ``top``
    (the largest), ``reads``."""
    drop = _off_diagonal(flows, min_expressed)
    rows = []

    def summary(values: np.ndarray) -> dict:
        reads = np.isfinite(values).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return {
                "influence": np.where(reads > 0, np.nansum(values, axis=-1) / np.maximum(reads, 1), np.nan),
                "positive": np.nansum(np.clip(values, 0, None), axis=-1),
                "top": np.where(reads > 0, np.nan_to_num(values, nan=-np.inf).max(axis=-1), np.nan),
                "reads": reads,
            }

    for r, receiver in enumerate(flows.receivers):
        rows.append(pd.DataFrame({"receiver": receiver, "sender": flows.senders} | summary(drop[:, r, :])))
    pooled = drop.reshape(len(flows.senders), -1)
    rows.append(pd.DataFrame({"receiver": EVERYONE, "sender": flows.senders} | summary(pooled)))
    return pd.concat(rows, ignore_index=True)


def pair_table(flows: GeneFlows, *, min_expressed: float = 0.1, top: int | None = None, include=()) -> pd.DataFrame:
    """Every read (receiver group, sender gene, target gene) with ``sender != target``, by ``drop`` descending, with its ``rank`` within the group;
    ``top`` keeps the first ``top`` per group plus every (sender, target) in ``include`` (the planted pairs)."""
    drop = _off_diagonal(flows, min_expressed)
    include = {(str(a), str(b)) for a, b in include}
    out = []
    for r, receiver in enumerate(flows.receivers):
        s, g = np.nonzero(np.isfinite(drop[:, r, :]))
        part = pd.DataFrame(
            {
                "receiver": receiver,
                "sender": np.asarray(flows.senders, dtype=object)[s],
                "target": np.asarray(flows.genes, dtype=object)[g],
                "drop": drop[s, r, g],
            }
        )
        part = part.sort_values("drop", ascending=False, kind="stable").reset_index(drop=True)
        part["rank"] = np.arange(1, len(part) + 1)
        if top is not None:
            keep = part["rank"] <= int(top)
            if include:
                keep |= [(a, b) in include for a, b in zip(part["sender"], part["target"], strict=True)]
            part = part[keep]
        out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["receiver", "sender", "target", "drop", "rank"])


def pooled_pair_scores(flows: GeneFlows, *, min_expressed: float = 0.1) -> pd.Series:
    """Per (sender, target) the largest drop over the receiver groups (index: tuples), the sender's own gene left out: one pair ranking for data
    whose truth names pairs but not the group (synth_spot)."""
    drop = _off_diagonal(flows, min_expressed)
    finite = np.isfinite(drop).any(axis=1)
    best = np.where(finite, np.nan_to_num(drop, nan=-np.inf).max(axis=1), np.nan)
    s, g = np.nonzero(np.isfinite(best))
    index = pd.MultiIndex.from_arrays([np.asarray(flows.senders, dtype=object)[s], np.asarray(flows.genes, dtype=object)[g]], names=["sender", "target"])
    return pd.Series(best[s, g], index=index)


# ----------------------------------------------------------------------------------------------- truth and metrics


def roles_truth(
    var: pd.DataFrame,
    *,
    role: str = "role",
    interaction: str = "interaction",
    responder_roles=RESPONDER_ROLES,
    driver_roles=DRIVER_ROLES,
    hard_roles=HARD_ROLES,
) -> dict:
    """From gene roles (synth_spot's ``var``): ``responders``, ``drivers``, ``hard_negatives`` (genes that vary in space or by sample without a sender) and
    ``pairs`` (every driver -> responder of the same ``interaction`` programme), as sets of gene names (pairs: tuples)."""
    names = np.asarray(var.index.astype(str))
    roles = var[role].astype(str).to_numpy()
    programme = var[interaction].astype(str).to_numpy() if interaction in var else np.full(len(names), "")
    responders = set(names[np.isin(roles, list(responder_roles))])
    drivers = set(names[np.isin(roles, list(driver_roles))])
    pairs = set()
    for name in sorted(set(programme.tolist()) - {"", "nan"}):
        members = programme == name
        pairs |= {(d, r) for d in names[members & np.isin(roles, list(driver_roles))] for r in names[members & np.isin(roles, list(responder_roles))]}
    return {"responders": responders, "drivers": drivers, "pairs": pairs, "hard_negatives": set(names[np.isin(roles, list(hard_roles))])}


def programme_truth(programmes: pd.DataFrame) -> dict:
    """From the programme table of :func:`semi_synthetic.spike_in` (``program``, ``sender``, ``receiver``, ``genes`` as ``"a, b"``, ``spike``):
    ``responders`` per receiver label (spiked programmes only), ``controls`` (the genes of the programmes that were not spiked: chosen by the same rule,
    never changed), ``sender_of`` ``{(receiver, gene): sender}``, ``programmes`` (one dict per programme with its gene list)."""
    responders, controls, sender_of, rows = {}, set(), {}, []
    for row in programmes.to_dict("records"):
        genes = [g.strip() for g in str(row["genes"]).split(",") if g.strip()]
        spiked = bool(row.get("spike", True))
        rows.append({"program": str(row["program"]), "sender": str(row["sender"]), "receiver": str(row["receiver"]), "genes": genes, "spike": spiked})
        if spiked:
            responders.setdefault(str(row["receiver"]), set()).update(genes)
            sender_of.update({(str(row["receiver"]), g): str(row["sender"]) for g in genes})
        else:
            controls.update(genes)
    return {"responders": responders, "controls": controls, "sender_of": sender_of, "programmes": rows}


def ranking_metrics(scores, positives, *, negatives=None, k: int | None = None) -> dict:
    """How well ``scores`` (a Series; higher = more interacting; NaN = not scored, left out) put ``positives`` first.

    ``auprc`` (average precision; ``base_rate`` is its chance level), ``auroc``, ``auroc_hard`` (positives against ``negatives`` alone, e.g. genes that
    vary in space without a sender), ``precision_at_k`` (``k``: the number of positives scored by default), ``median_rank`` of the positives (1 = top),
    ``n_pos``, ``n_scored``. NaN where a metric is undefined."""
    s = pd.Series(scores, dtype=float).dropna()
    positives = {p for p in positives if p in s.index}
    out = {"n_pos": len(positives), "n_scored": len(s), "base_rate": len(positives) / len(s) if len(s) else np.nan}
    nan = dict.fromkeys(("auprc", "auroc", "auroc_hard", "precision_at_k", "k", "median_rank"), np.nan)
    if not positives or len(positives) == len(s):
        return out | nan
    labels = np.array([i in positives for i in s.index])
    values = s.to_numpy()
    out["auprc"] = float(_average_precision(values, labels))
    out["auroc"] = float(_auroc(values, labels))
    if negatives is not None:
        neg = np.array([i in set(negatives) for i in s.index]) & ~labels
        out["auroc_hard"] = float(_auroc(values[labels | neg], labels[labels | neg])) if neg.any() else np.nan
    else:
        out["auroc_hard"] = np.nan
    kk = int(k) if k is not None else len(positives)
    order = np.argsort(-values, kind="stable")
    out["k"] = kk
    out["precision_at_k"] = float(labels[order[:kk]].mean()) if kk > 0 else np.nan
    ranks = pd.Series(values).rank(ascending=False, method="average").to_numpy()
    out["median_rank"] = float(np.median(ranks[labels]))
    return out


# ----------------------------------------------------------------------------------------------- bars, model-free


def within_group_residual(x, labels, graphs) -> np.ndarray:
    """``x`` minus each (graph, label)'s mean: the within-type, within-section variation the rankings are about."""
    x = np.asarray(x, dtype=np.float64)
    key = pd.Series(np.asarray(graphs).astype(str)).str.cat(pd.Series(np.asarray(labels).astype(str)), sep="|").to_numpy()
    return x - pd.DataFrame(x).groupby(key).transform("mean").to_numpy()


def within_group_moran(x, pos, labels, graphs, *, k: int = 8, max_cells: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Labels x genes: Moran's I of the within-group residual among the group's own cells (``k`` nearest within each graph): which genes vary smoothly in
    space within a type -- spatially variable, whether or not anything drives them."""
    labels = np.asarray(labels).astype(str)
    resid = within_group_residual(x, labels, graphs)
    pos, graphs = np.asarray(pos, dtype=np.float64)[:, :2], np.asarray(graphs).astype(str)
    types = sorted(set(labels.tolist()))
    rows = [moran_genes(resid[labels == t], pos[labels == t], graphs[labels == t], k=int(k), max_cells=int(max_cells), seed=seed) for t in types]
    return pd.DataFrame(np.vstack(rows), index=pd.Index(types, name="receiver"))


def within_group_variance(x, labels, graphs) -> pd.DataFrame:
    """Labels x genes: the variance of the within-group residual (the variable genes, whatever their cause)."""
    labels = np.asarray(labels).astype(str)
    resid = within_group_residual(x, labels, graphs)
    types = sorted(set(labels.tolist()))
    return pd.DataFrame(np.vstack([resid[labels == t].var(axis=0) for t in types]), index=pd.Index(types, name="receiver"))


def spillover_flags(x, labels, *, fold: float = 3.0, quantile: float = 0.75) -> pd.DataFrame:
    """Labels x genes, True where the gene is a marker of another label: its mean in some other label is at least ``fold`` times the receiver label's and
    at least the ``quantile`` of all positive label means. A responder flagged here may be segmentation spillover from a neighbour of that label."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    types = sorted(set(labels.tolist()))
    means = np.vstack([x[labels == t].mean(axis=0) for t in types])
    positive = means[means > 0]
    floor = float(np.quantile(positive, float(quantile))) if len(positive) else np.inf
    flags = np.zeros_like(means, dtype=bool)
    for r in range(len(types)):
        if len(types) < 2:
            break
        other = np.delete(means, r, axis=0).max(axis=0)
        flags[r] = (other >= float(fold) * means[r]) & (other >= floor)
    return pd.DataFrame(flags, index=pd.Index(types, name="receiver"))


def neighbour_features(x, pos, labels, graphs, types, *, k: int) -> np.ndarray:
    """``[N, G + T]``: per cell the mean within-group residual of its ``k`` nearest other cells of the same graph, and their label shares (``types``)."""
    x = np.asarray(x, dtype=np.float64)
    labels, graphs = np.asarray(labels).astype(str), np.asarray(graphs).astype(str)
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    resid = within_group_residual(x, labels, graphs)
    code = np.searchsorted(np.asarray(types), labels)
    out = np.zeros((len(x), x.shape[1] + len(types)))
    for name in sorted(set(graphs.tolist())):
        where = np.flatnonzero(graphs == name)
        kk = min(int(k), len(where) - 1)
        if kk < 1:
            continue
        _, near = cKDTree(pos[where]).query(pos[where], k=kk + 1)
        near = where[near[:, 1:]]
        out[where, : x.shape[1]] = resid[near].mean(axis=1)
        shares = np.zeros((len(where), len(types)))
        for j in range(kk):
            shares[np.arange(len(where)), code[near[:, j]]] += 1.0 / kk
        out[where, x.shape[1] :] = shares
    return out


@dataclass
class RidgeRanking:
    """The neighbour ridge per receiver label: ``responder`` ``[R, G]`` (the centred cosine within (graph, label) of the leave-own-out prediction on the
    test cells), ``pairs`` ``[R, G sender, G target]`` (|coefficient| per standard deviation of the sender feature over that of the target, 0 on the
    diagonal), ``alpha`` per label (NaN: too few cells)."""

    receivers: list[str]
    genes: list[str]
    responder: np.ndarray
    pairs: np.ndarray
    alpha: dict

    def drivers(self) -> np.ndarray:
        """``[R, G]``: per sender gene the mean of its pair scores over the other targets."""
        g = len(self.genes)
        with np.errstate(invalid="ignore"):
            return self.pairs.sum(axis=2) / max(g - 1, 1)


def neighbour_ridge(
    train: dict,
    test: dict,
    genes,
    *,
    k: int = 16,
    alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
    val_share: float = 0.2,
    min_cells: int = 50,
    seed: int = 0,
) -> RidgeRanking:
    """A linear, cross-gene neighbourhood model per receiver label, fitted on ``train`` and scored on ``test`` (dicts of ``x`` ``[N, G]``, ``pos``, ``labels``,
    ``graphs``): each target's within-group residual on its neighbours' mean residuals and label shares (:func:`neighbour_features`), standardised, ridge with
    the penalty (``alphas`` x rows) chosen on a random ``val_share`` of the training cells. The target's own neighbour feature is left out exactly (the
    one-column downdate of the inverse), so a gene that is only smooth in space gets nothing from its own neighbours.

    A weak bar on purpose: the neighbour means do not tell the sender's type, so a gene that acts from one type only is diluted by the others (on the
    test grid A's g3 reaches 0.2-0.3 where the rule is exact), and spatially smooth targets pick up chance correlations with smooth features (~0.1).
    Per-type means would multiply the features by the number of types (~10^4 on a CosMx panel)."""
    genes = [str(g) for g in genes]
    n_g = len(genes)
    types = sorted(set(np.asarray(train["labels"]).astype(str).tolist()) | set(np.asarray(test["labels"]).astype(str).tolist()))
    f_tr = neighbour_features(train["x"], train["pos"], train["labels"], train["graphs"], types, k=k)
    f_te = neighbour_features(test["x"], test["pos"], test["labels"], test["graphs"], types, k=k)
    y_tr = within_group_residual(train["x"], train["labels"], train["graphs"])
    y_te = within_group_residual(test["x"], test["labels"], test["graphs"])
    lab_tr, lab_te = np.asarray(train["labels"]).astype(str), np.asarray(test["labels"]).astype(str)
    graphs_te = np.asarray(test["graphs"]).astype(str)
    rng = np.random.default_rng(seed)
    responder = np.full((len(types), n_g), np.nan)
    pairs = np.zeros((len(types), n_g, n_g))
    chosen = {}
    for r, t in enumerate(types):
        rows = np.flatnonzero(lab_tr == t)
        test_rows = np.flatnonzero(lab_te == t)
        if len(rows) < int(min_cells) or len(test_rows) < 3:
            chosen[t] = np.nan
            continue
        x_all, y_all = f_tr[rows], y_tr[rows]
        mean_x, mean_y = x_all.mean(axis=0), y_all.mean(axis=0)
        sd_x = x_all.std(axis=0)
        sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
        z, yc = (x_all - mean_x) / sd_x, y_all - mean_y
        order = rng.permutation(len(rows))
        n_val = max(1, int(round(float(val_share) * len(rows))))
        val, fit = order[:n_val], order[n_val:]
        gram_fit, cross_fit = z[fit].T @ z[fit], z[fit].T @ yc[fit]
        best, best_err = None, np.inf
        for a in alphas:
            coef = np.linalg.solve(gram_fit + float(a) * len(fit) * np.eye(z.shape[1]), cross_fit)
            err = float(((yc[val] - z[val] @ coef) ** 2).sum())
            if err < best_err:
                best, best_err = float(a), err
        chosen[t] = best
        inverse = np.linalg.inv(z.T @ z + best * len(rows) * np.eye(z.shape[1]))
        coef = inverse @ (z.T @ yc)
        # Leave the target's own neighbour feature out: coef[:, g] - Q[:, g] coef[g, g] / Q[g, g] for every target g (columns 0..G-1 align with genes).
        diag_q = np.diag(inverse)[:n_g]
        own = coef[np.arange(n_g), np.arange(n_g)]
        lso = coef - inverse[:, :n_g] * (own / diag_q)[None, :]
        lso[np.arange(n_g), np.arange(n_g)] = 0.0
        pred = ((f_te[test_rows] - mean_x) / sd_x) @ lso + mean_y
        truth = y_te[test_rows]
        pred_c = within_group_residual(pred, lab_te[test_rows], graphs_te[test_rows])
        with np.errstate(invalid="ignore", divide="ignore"):
            num = (pred_c * truth).sum(axis=0)
            den = np.sqrt((pred_c**2).sum(axis=0) * (truth**2).sum(axis=0))
            responder[r] = np.where(den > 0, num / den, np.nan)
        sd_y = y_all.std(axis=0)
        pairs[r] = np.abs(lso[:n_g]) / np.where(sd_y > 1e-12, sd_y, np.inf)[None, :]
    return RidgeRanking(types, genes, responder, pairs, chosen)
