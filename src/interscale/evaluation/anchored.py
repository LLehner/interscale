"""Readouts of the flipped anchored-SVG task (``.claude/svg_targets_plan.md``, round 3; ``dataset.anchor_rings``).

A model trained to predict the SVG targets in the rings around one grid disc from the disc's cells alone (the only keys) is read,
on every grid disc of the test graphs, by intervention on the disc:

* **base** -- per target and ring, the Pearson correlation of prediction and truth over the ring cells of every disc read (pooled,
  centred within the ring: a disc that carries a response raises its rings, which is the signal, so it is not centred away);
* **senders** -- one disc gene replaced, in the disc's cells, by its values in cells drawn from *outside* the disc (of the same label
  when labels are given; a permutation within the disc would keep the disc's level); per (sender, target, ring) the drop of the
  correlation and the mean absolute shift of the prediction. The target's own gene in the disc is never touched for its own score
  (self-pairs are left out), so every sender score is beyond the target's own level in the disc;
* **responders (beyond self)** -- every disc gene but the target replaced the same way (whole donor rows, the target's column kept):
  the drop of the target's correlation is what the disc's other genes carry;
* **curves** -- per disc and target the prediction and the truth averaged in distance bins from the disc's edge (the bridge's grid:
  ``n_bins + 1`` points over ``[0, r_max]``), to compare with diffsvg's fitted curve around the same disc and with the average curve;
* **sender groups** (``groups``: genes that co-vary across discs, e.g. a source and its responses) -- the whole group replaced at once
  (the target's own gene kept), and the group with one member kept: a member whose keeping leaves the drop near zero carries the
  group's effect alone (``sufficiency``), which one-at-a-time replacement cannot tell among collinear genes.

**Donor schemes** (``donor_labels``): every intervention is run once per scheme. ``any`` draws donors anywhere outside the disc;
a label scheme (e.g. ``region``) draws each disc cell's donor from outside cells of its own label, so what the label carries (a
region's level) survives the replacement and only what lies beyond it is scored: the "beyond region" control.

Positions alone choose discs and rings (``tl.masking.anchor_grid_discs``, the bridge's rule). The receivers' rows never change. Every
random draw uses a private numpy generator; the global generators are restored (``_rng_untouched``). ``batch_size`` sends several
perturbed copies of one view through the model together (the same predictions as one at a time, tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from interscale.evaluation.context_test import _predict
from interscale.evaluation.online_probes import _rng_untouched
from interscale.tl.masking import anchor_grid_discs, anchor_ring_masks


def disc_views(graph, gene_columns, *, grid: int, r_disc: float, r_max: float, n_rings: int = 3):
    """Per grid disc of ``graph``: ``(disc_id, view, disc, ring, edge)``. ``view`` is a clone with the training draw's masks for that
    disc (``gene_columns`` hidden in the ring cells, ``key_mask`` = the disc), ``ring`` each cell's ring (-1 outside), ``edge`` its
    distance from the disc's edge (units of ``pos``).
    """
    out = []
    cols = torch.as_tensor(list(gene_columns), dtype=torch.long)
    for disc_id, centre in anchor_grid_discs(graph.pos, grid):
        disc, rings, ring = anchor_ring_masks(graph.pos, centre, r_disc, r_max, n_rings)
        view = graph.clone()
        gene_mask = torch.zeros(graph.num_nodes, graph.x.shape[1], dtype=torch.bool)
        gene_mask[rings.nonzero(as_tuple=True)[0][:, None], cols[None, :]] = True
        view.gene_mask, view.mask, view.key_mask = gene_mask, rings.clone(), disc.clone()
        view.anchor_disc = torch.tensor([disc_id])
        edge = (graph.pos[:, :2].double() - graph.pos[centre, :2].double()).norm(dim=1) - r_disc
        out.append((disc_id, view, disc.numpy(), ring.numpy(), edge.numpy()))
    return out


class _Pearson:
    """Running sums for Pearson correlations over pooled cells, any leading shape."""

    def __init__(self, shape):
        self.s = np.zeros((6, *shape))

    def add(self, index, pred, truth):
        ok = np.isfinite(pred) & np.isfinite(truth)
        p, t = np.where(ok, pred, 0.0), np.where(ok, truth, 0.0)
        for k, v in enumerate((ok.sum(0), p.sum(0), t.sum(0), (p * p).sum(0), (t * t).sum(0), (p * t).sum(0))):
            self.s[k][index] += v

    def corr(self):
        n, sp, st, spp, stt, spt = self.s
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = spt - sp * st / n
            vp, vt = spp - sp * sp / n, stt - st * st / n
            r = cov / np.sqrt(vp * vt)
        return np.where((n >= 3) & (vp > 0) & (vt > 0), r, np.nan)


@dataclass
class AnchoredRead:
    targets: list
    senders: list
    n_rings: int
    base: np.ndarray  # [targets, rings] correlation
    sender_corr: np.ndarray  # [schemes, senders, targets, rings]
    sender_shift: np.ndarray  # [schemes, senders, targets, rings] mean |prediction change|
    responder_corr: np.ndarray  # [schemes, targets, rings], every disc gene but the target replaced
    curves: list = field(default_factory=list)  # rows: name, disc, target, bin, pred, truth, cells
    schemes: list = field(default_factory=lambda: ["any"])
    groups: list = field(default_factory=list)  # multi-gene sender groups (lists of gene names)
    set_corr: dict = field(default_factory=dict)  # frozenset of replaced genes -> [schemes, targets, rings]

    def tables(self) -> dict[str, pd.DataFrame]:
        rings = range(self.n_rings)
        base = pd.DataFrame(
            [(t, r, k, self.base[i, r], self.responder_corr[s, i, r]) for s, k in enumerate(self.schemes)
             for i, t in enumerate(self.targets) for r in rings],
            columns=["target", "ring", "scheme", "corr", "corr_others_replaced"],
        )  # fmt: skip
        base["responder_beyond_self"] = base["corr"] - base["corr_others_replaced"]
        rows = []
        for k, scheme in enumerate(self.schemes):
            for s, sender in enumerate(self.senders):
                for i, t in enumerate(self.targets):
                    if sender == t:
                        continue
                    for r in rings:
                        rows.append((sender, t, r, scheme, self.base[i, r] - self.sender_corr[k, s, i, r], self.sender_shift[k, s, i, r]))
        pairs = pd.DataFrame(rows, columns=["sender", "target", "ring", "scheme", "drop", "shift"])
        rows = []
        for gid, members in enumerate(self.groups):
            for i, t in enumerate(self.targets):
                for kept in [None, *[m for m in members if m != t]]:
                    replaced = frozenset(members) - {t} - ({kept} if kept else set())
                    if not replaced or replaced not in self.set_corr:
                        continue
                    corr = self.set_corr[replaced]
                    for k, scheme in enumerate(self.schemes):
                        for r in rings:
                            rows.append((gid, " ".join(members), t, t in members, kept or "", r, scheme, self.base[i, r] - corr[k, i, r]))
        groups = pd.DataFrame(rows, columns=["group", "members", "target", "target_in_group", "kept", "ring", "scheme", "drop"])
        curves = pd.DataFrame(self.curves, columns=["graph", "disc", "target", "bin", "pred", "truth", "cells"])
        return {"anch_base": base, "anch_pairs": pairs, "anch_groups": groups, "anch_curves": curves}


def _donor_rows(rng, disc_cells, labels, n_cells):
    """For every disc cell a donor cell outside the disc (same label when ``labels``), drawn with replacement."""
    outside = np.flatnonzero(~disc_cells)
    rows = np.flatnonzero(disc_cells)
    donors = np.empty(len(rows), dtype=int)
    if labels is None:
        donors[:] = rng.choice(outside, len(rows))
    else:
        for lab in np.unique(labels[rows]):
            pool = outside[labels[outside] == lab]
            pool = pool if len(pool) else outside
            sel = labels[rows] == lab
            donors[sel] = rng.choice(pool, int(sel.sum()))
    return rows, donors


def _group_sets(groups, targets) -> list[frozenset]:
    """Every replaced set the group readout needs: per group G and target t, ``G - {t}`` and, per kept member m, ``G - {t, m}``."""
    out = []
    for members in groups:
        G = frozenset(members)
        for t in [None, *[t for t in targets if t in G]]:
            whole = G - {t}
            out.append(whole)
            out += [whole - {m} for m in whole]
    seen, unique = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def read_discs(
    module,
    graphs,
    genes,
    targets,
    *,
    grid: int,
    r_disc: float,
    r_max: float,
    n_rings: int = 3,
    senders=None,
    labels=None,
    donor_labels: dict | None = None,
    groups=None,
    names=None,
    n_bins: int = 20,
    responders: bool = True,
    head: str = "global",
    device="cpu",
    seed: int = 0,
    batch_size: int = 1,
) -> AnchoredRead:
    """The readouts of the module docstring over every grid disc of ``graphs`` (clean test graphs; units of ``pos`` for the radii).

    ``targets``: the hidden gene set (names in ``genes``); ``senders``: the disc genes to replace one at a time (default: every gene).
    ``labels``: one array per graph (donors of the same label), or None -- the single scheme ``any`` (kept for older callers);
    ``donor_labels``: ``{scheme: one array per graph, or None}`` instead (the first scheme draws exactly as ``labels`` did).
    ``groups``: sender groups (lists of gene names; singletons are the per-sender passes and are skipped). ``names``: one name per
    graph for the curve rows. ``batch_size``: perturbed copies per forward pass.
    """
    genes = [str(g) for g in genes]
    position = {g: k for k, g in enumerate(genes)}
    targets = [str(t) for t in targets]
    tcols = np.array([position[t] for t in targets])
    senders = genes if senders is None else [str(s) for s in senders if str(s) in position]
    scols = np.array([position[s] for s in senders])
    schemes = {"any": labels} if donor_labels is None else dict(donor_labels)
    if not schemes:
        raise ValueError("donor_labels names no scheme.")
    K = len(schemes)
    groups = [[str(m) for m in G if str(m) in position] for G in (groups or [])]
    groups = [G for G in groups if len(G) > 1]
    sets = _group_sets(groups, targets)
    set_cols = {s: np.array(sorted(position[m] for m in s)) for s in sets}
    base = _Pearson((len(targets), n_rings))
    by_sender = _Pearson((K, len(senders), len(targets), n_rings))
    shift_sum = np.zeros((K, len(senders), len(targets), n_rings))
    shift_n = np.zeros((K, len(senders), len(targets), n_rings))
    others = _Pearson((K, len(targets), n_rings))
    by_set = {s: _Pearson((K, len(targets), n_rings)) for s in sets}
    curves = []
    # scheme 0 draws from default_rng(seed), as the single-scheme readout did; the others from their own streams
    rngs = [np.random.default_rng(seed) if k == 0 else np.random.default_rng([seed, k]) for k in range(K)]
    all_cols = np.arange(len(genes))
    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad():
            for g, graph in enumerate(graphs):
                name = str(g) if names is None else str(names[g])
                truth_all = graph.x.double().cpu().numpy()
                for disc_id, view, disc, ring, edge in disc_views(graph, tcols, grid=grid, r_disc=r_disc, r_max=r_max, n_rings=n_rings):
                    if not disc.any() or not (ring >= 0).any():
                        continue
                    pred = _predict(module, [view], head=head, batch_size=1, device=device)[0][:, tcols]
                    truth = truth_all[:, tcols]
                    cells_of = [ring == r for r in range(n_rings)]
                    for r, cells in enumerate(cells_of):
                        base.add((slice(None), r), pred[cells], truth[cells])
                    # curves: distance bins from the disc's edge on the bridge's grid
                    in_rings = ring >= 0
                    b = np.rint(edge[in_rings] / r_max * n_bins).astype(int)
                    for i, t in enumerate(targets):
                        p, y = pred[in_rings, i], truth[in_rings, i]
                        for k in np.unique(b):
                            sel = b == k
                            curves.append((name, disc_id, t, int(k), float(np.nanmean(p[sel])), float(np.nanmean(y[sel])), int(sel.sum())))
                    for k, lab_list in enumerate(schemes.values()):
                        lab = None if lab_list is None else np.asarray(lab_list[g]).astype(str)
                        rows, donors = _donor_rows(rngs[k], disc, lab, graph.num_nodes)
                        # jobs: (kind, index or key, replaced columns)
                        jobs = [("sender", s, np.array([col])) for s, col in enumerate(scols)]
                        if responders:
                            jobs += [("responder", i, np.setdiff1d(all_cols, [col])) for i, col in enumerate(tcols)]
                        jobs += [("set", key, set_cols[key]) for key in sets]
                        for start in range(0, len(jobs), max(1, int(batch_size))):
                            chunk = jobs[start : start + max(1, int(batch_size))]
                            copies = []
                            for _, _, cols in chunk:
                                x = graph.x.clone()
                                c = torch.as_tensor(cols, dtype=torch.long)
                                x[torch.as_tensor(rows)[:, None], c[None, :]] = graph.x[torch.as_tensor(donors)[:, None], c[None, :]]
                                copy = view.clone()
                                copy.x = x
                                copies.append(copy)
                            preds = _predict(module, copies, head=head, batch_size=len(copies), device=device)
                            for (kind, idx, _), full in zip(chunk, preds, strict=True):
                                p = full[:, tcols]
                                for r, cells in enumerate(cells_of):
                                    if kind == "sender":
                                        by_sender.add((k, idx, slice(None), r), p[cells], truth[cells])
                                        delta = np.abs(p[cells] - pred[cells])
                                        shift_sum[k, idx, :, r] += np.nansum(delta, axis=0)
                                        shift_n[k, idx, :, r] += np.isfinite(delta).sum(axis=0)
                                    elif kind == "responder":
                                        others.add((k, idx, r), p[cells, idx], truth[cells, idx])
                                    else:
                                        by_set[idx].add((k, slice(None), r), p[cells], truth[cells])
    finally:
        module.train(was_training)
    with np.errstate(invalid="ignore"):
        shift = shift_sum / shift_n
    return AnchoredRead(
        targets, senders, n_rings, base.corr(), by_sender.corr(), shift,
        others.corr() if responders else np.full((K, len(targets), n_rings), np.nan), curves,
        schemes=list(schemes), groups=groups, set_corr={s: acc.corr() for s, acc in by_set.items()},
    )  # fmt: skip


def rank_tables(tables: dict, local_rings, *, scheme: str = "any", min_drop: float = 0.01) -> dict[str, pd.DataFrame]:
    """The model's gene rankings from one read's tables (``AnchoredRead.tables``), for one donor scheme.

    ``local_rings``: the rings counted as local (the rest are global). Every score is a correlation drop averaged over the rings of
    its part (``local``, ``global``, ``total`` = all rings); sums over targets clip negative drops at 0.

    * ``rank_pairs`` -- sender gene -> target: drops per part, the sender's rank among the target's senders;
    * ``rank_senders`` -- per sender: summed drops over targets per part, local share, targets with a total drop >= ``min_drop``,
      its three strongest targets;
    * ``rank_responders`` -- per target: what the disc's other genes carry (beyond self) per part, and the ring correlations;
    * ``rank_groups`` -- per (sender group, target): the whole group's drop per part (singletons: the sender's own), and for a
      multi-gene group each kept member's ``sufficiency`` = 1 - drop(member kept) / drop(group): 1 when the member alone carries it.
    """
    local_rings = sorted(int(r) for r in local_rings)

    def parts(frame, keys, value="drop"):
        frame = frame.assign(part=np.where(frame["ring"].isin(local_rings), "local", "global"))
        by = frame.groupby([*keys, "part"])[value].mean().unstack("part").reindex(columns=["local", "global"])
        by["total"] = frame.groupby(keys)[value].mean()
        return by.reset_index()

    pairs = tables["anch_pairs"]
    pairs = pairs[pairs["scheme"] == scheme]
    rp = parts(pairs, ["sender", "target"])
    rp["rank_in_target"] = rp.groupby("target")["total"].rank(ascending=False, method="min").astype(int)
    rp = rp.sort_values(["target", "rank_in_target"]).reset_index(drop=True)

    clip = rp.assign(**{c: rp[c].clip(lower=0) for c in ("local", "global", "total")})
    rs = clip.groupby("sender")[["local", "global", "total"]].sum()
    rs["local_share"] = rs["local"] / (rs["local"] + rs["global"]).replace(0, np.nan)
    rs["n_targets"] = clip.groupby("sender")["total"].apply(lambda v: int((v >= min_drop).sum()))
    rs["top_targets"] = clip.sort_values("total", ascending=False).groupby("sender")["target"].apply(lambda v: " ".join(v.head(3)))
    rs = rs.sort_values("total", ascending=False)
    rs.insert(0, "rank", np.arange(1, len(rs) + 1))
    rs = rs.reset_index()

    base = tables["anch_base"]
    base = base[base["scheme"] == scheme]
    rr = parts(base, ["target"], "responder_beyond_self")
    corr = parts(base, ["target"], "corr").rename(columns={"local": "corr_local", "global": "corr_global", "total": "corr_total"})
    rr = rr.merge(corr, on="target").sort_values("total", ascending=False)
    rr.insert(0, "rank", np.arange(1, len(rr) + 1))
    rr = rr.reset_index(drop=True)

    groups = tables.get("anch_groups")
    rows = []
    if groups is not None and len(groups):
        gr = groups[groups["scheme"] == scheme]
        whole = parts(gr[gr["kept"] == ""], ["group", "members", "target", "target_in_group"])
        kept = parts(gr[gr["kept"] != ""], ["group", "target", "kept"])
        for _, w in whole.iterrows():
            ks = kept[(kept["group"] == w["group"]) & (kept["target"] == w["target"])]
            suff = {
                k["kept"]: (1.0 - k["total"] / w["total"]) if w["total"] > 0 else np.nan for _, k in ks.iterrows()
            }
            finite = {m: v for m, v in suff.items() if np.isfinite(v)}
            best = max(finite, key=finite.get) if finite else ""  # no drop to share: no member named
            rows.append({"group": f"G{int(w['group'])}", "members": w["members"], "target": w["target"],
                         "target_in_group": bool(w["target_in_group"]), "local": w["local"], "global": w["global"],
                         "total": w["total"], "most_sufficient": best,
                         "sufficiency": " ".join(f"{m}:{v:.2f}" for m, v in sorted(suff.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-9)))})  # fmt: skip
    grouped = {m for G in (groups["members"].unique() if groups is not None and len(groups) else []) for m in G.split()}
    for _, p in rp[~rp["sender"].isin(grouped)].iterrows():
        rows.append({"group": p["sender"], "members": p["sender"], "target": p["target"], "target_in_group": False,
                     "local": p["local"], "global": p["global"], "total": p["total"], "most_sufficient": p["sender"],
                     "sufficiency": ""})  # fmt: skip
    rg = pd.DataFrame(rows, columns=["group", "members", "target", "target_in_group", "local", "global", "total", "most_sufficient", "sufficiency"])
    if len(rg):
        rg["rank_in_target"] = rg.groupby("target")["total"].rank(ascending=False, method="min").astype(int)
        rg = rg.sort_values(["target", "rank_in_target"]).reset_index(drop=True)
    return {"rank_pairs": rp, "rank_senders": rs, "rank_responders": rr, "rank_groups": rg}


def curve_agreement(curves: pd.DataFrame, observed: dict, average: dict | None = None) -> pd.DataFrame:
    """Per (graph, disc, target): correlation over bins of the model's curve with diffsvg's (``observed[(graph, disc)][target]``, an
    array over the bins) and, with ``average`` (per target the mean curve over the training discs), the same for the average curve:
    beating it means the model reads the disc's content, not only the distance.
    """
    rows = []
    for (graph, disc, target), part in curves.groupby(["graph", "disc", "target"]):
        obs = observed.get((str(graph), int(disc)), {}).get(target)
        if obs is None:
            continue
        part = part.sort_values("bin")
        k = part["bin"].to_numpy()
        ok = (k >= 0) & (k < len(obs)) & np.isfinite(part["pred"].to_numpy())
        if ok.sum() < 3:
            continue
        o, p = np.asarray(obs)[k[ok]], part["pred"].to_numpy()[ok]
        r_model = np.corrcoef(p, o)[0, 1] if np.std(p) > 0 and np.std(o) > 0 else np.nan
        r_avg = np.nan
        if average is not None and target in average:
            a = np.asarray(average[target])[k[ok]]
            r_avg = np.corrcoef(a, o)[0, 1] if np.std(a) > 0 and np.std(o) > 0 else np.nan
        rows.append((graph, disc, target, r_model, r_avg))
    return pd.DataFrame(rows, columns=["graph", "disc", "target", "corr_model", "corr_average"])
