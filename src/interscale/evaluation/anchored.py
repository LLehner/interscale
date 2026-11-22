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
  ``n_bins + 1`` points over ``[0, r_max]``), to compare with diffsvg's fitted curve around the same disc and with the average curve.

Positions alone choose discs and rings (``tl.masking.anchor_grid_discs``, the bridge's rule). The receivers' rows never change. Every
random draw uses a private numpy generator; the global generators are restored (``_rng_untouched``).
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
    sender_corr: np.ndarray  # [senders, targets, rings]
    sender_shift: np.ndarray  # [senders, targets, rings] mean |prediction change|
    responder_corr: np.ndarray  # [targets, rings], every disc gene but the target replaced
    curves: list = field(default_factory=list)  # rows: name, disc, target, bin, pred, truth, cells

    def tables(self) -> dict[str, pd.DataFrame]:
        rings = range(self.n_rings)
        base = pd.DataFrame(
            [(t, r, self.base[i, r], self.responder_corr[i, r]) for i, t in enumerate(self.targets) for r in rings],
            columns=["target", "ring", "corr", "corr_others_replaced"],
        )
        base["responder_beyond_self"] = base["corr"] - base["corr_others_replaced"]
        rows = []
        for s, sender in enumerate(self.senders):
            for i, t in enumerate(self.targets):
                if sender == t:
                    continue
                for r in rings:
                    rows.append((sender, t, r, self.base[i, r] - self.sender_corr[s, i, r], self.sender_shift[s, i, r]))
        pairs = pd.DataFrame(rows, columns=["sender", "target", "ring", "drop", "shift"])
        curves = pd.DataFrame(self.curves, columns=["graph", "disc", "target", "bin", "pred", "truth", "cells"])
        return {"anch_base": base, "anch_pairs": pairs, "anch_curves": curves}


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
    names=None,
    n_bins: int = 20,
    responders: bool = True,
    head: str = "global",
    device="cpu",
    seed: int = 0,
) -> AnchoredRead:
    """The readouts of the module docstring over every grid disc of ``graphs`` (clean test graphs; units of ``pos`` for the radii).

    ``targets``: the hidden gene set (names in ``genes``); ``senders``: the disc genes to replace one at a time (default: every gene).
    ``labels``: one array per graph (donors of the same label), or None. ``names``: one name per graph for the curve rows.
    """
    genes = [str(g) for g in genes]
    position = {g: k for k, g in enumerate(genes)}
    targets = [str(t) for t in targets]
    tcols = np.array([position[t] for t in targets])
    senders = genes if senders is None else [str(s) for s in senders if str(s) in position]
    scols = np.array([position[s] for s in senders])
    base = _Pearson((len(targets), n_rings))
    by_sender = _Pearson((len(senders), len(targets), n_rings))
    shift_sum = np.zeros((len(senders), len(targets), n_rings))
    shift_n = np.zeros((len(senders), len(targets), n_rings))
    others = _Pearson((len(targets), n_rings))
    curves = []
    rng = np.random.default_rng(seed)
    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device), torch.no_grad():
            for g, graph in enumerate(graphs):
                lab = None if labels is None else np.asarray(labels[g]).astype(str)
                name = str(g) if names is None else str(names[g])
                truth_all = graph.x.double().cpu().numpy()
                for disc_id, view, disc, ring, edge in disc_views(graph, tcols, grid=grid, r_disc=r_disc, r_max=r_max, n_rings=n_rings):
                    if not disc.any() or not (ring >= 0).any():
                        continue
                    pred = _predict(module, [view], head=head, batch_size=1, device=device)[0][:, tcols]
                    truth = truth_all[:, tcols]
                    for r in range(n_rings):
                        cells = ring == r
                        base.add((slice(None), r), pred[cells], truth[cells])
                    # curves: distance bins from the disc's edge on the bridge's grid
                    in_rings = ring >= 0
                    b = np.rint(edge[in_rings] / r_max * n_bins).astype(int)
                    for i, t in enumerate(targets):
                        p, y = pred[in_rings, i], truth[in_rings, i]
                        for k in np.unique(b):
                            sel = b == k
                            curves.append((name, disc_id, t, int(k), float(np.nanmean(p[sel])), float(np.nanmean(y[sel])), int(sel.sum())))
                    rows, donors = _donor_rows(rng, disc, lab, graph.num_nodes)
                    for s, col in enumerate(scols):
                        x = graph.x.clone()
                        x[rows, col] = graph.x[donors, col]
                        copy = view.clone()
                        copy.x = x
                        pred_s = _predict(module, [copy], head=head, batch_size=1, device=device)[0][:, tcols]
                        for r in range(n_rings):
                            cells = ring == r
                            by_sender.add((s, slice(None), r), pred_s[cells], truth[cells])
                            delta = np.abs(pred_s[cells] - pred[cells])
                            shift_sum[s, :, r] += np.nansum(delta, axis=0)
                            shift_n[s, :, r] += np.isfinite(delta).sum(axis=0)
                    if responders:
                        for i, col in enumerate(tcols):
                            x = graph.x.clone()
                            keep = x[rows, col].clone()
                            x[rows] = graph.x[donors]
                            x[rows, col] = keep
                            copy = view.clone()
                            copy.x = x
                            pred_o = _predict(module, [copy], head=head, batch_size=1, device=device)[0][:, col]
                            for r in range(n_rings):
                                cells = ring == r
                                others.add((i, r), pred_o[cells], truth_all[cells, col])
    finally:
        module.train(was_training)
    with np.errstate(invalid="ignore"):
        shift = shift_sum / shift_n
    return AnchoredRead(
        targets, senders, n_rings, base.corr(), by_sender.corr(), shift,
        others.corr() if responders else np.full((len(targets), n_rings), np.nan), curves,
    )


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
