"""Each receiver's attention over the cells of its graph, for every model kind, and what the rows look like (``.claude/comparison.md``,
2.1 masking stability and 2.2 heads, 2026-11-24).

:func:`attention_rows` returns ``[Q, N]`` head-mean weights in graph node order, CLS and the null key left out (a row sums to its cell mass,
at most 1): the read-out attention of an interaction read-out (``interaction_readout.readout_pass``) or the explicit map of a dense model, a
global-only model's or the prototype's transformer (``recording_attention``). :func:`row_summary` gives per receiver the cell mass, the entropy
over the cells relative to uniform, the largest share, the effective number of keys and the mean distance read; :func:`row_agreement` compares
two models' rows for the same receivers (Spearman over the union of their top cells, Jensen-Shannon divergence): seed stability of attention.

These are routing weights, not effects (``evaluation.attribution`` measures effects); they answer how concentrated and how stable the
attention is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from interscale.evaluation.interpretation import _rng_untouched


def _readout_rows(module, graph, receivers, layer, eval_step: bool) -> np.ndarray:
    from interscale.evaluation.interaction_readout import _block_weights, readout_layers, readout_pass

    rp = readout_pass(module, graph, eval_step=eval_step)
    n_layers = len(readout_layers(module))
    nodes = np.asarray(rp.index_nodes)
    cells_pos = np.arange(rp.offset, rp.offset + len(nodes))
    out = np.zeros((len(receivers), int(graph.num_nodes)), dtype=np.float32)
    positions = rp.position_of(receivers)
    with torch.no_grad():
        for k, p in enumerate(positions):
            if p < 0:
                out[k] = np.nan
                continue
            cells, _ = _block_weights(module, rp, int(p), int(p) + 1)  # [1, layers * heads, L]
            per_layer = cells[0].view(n_layers, -1, cells.shape[-1])  # [layers, heads, L]
            chosen = per_layer.mean(0) if layer == "mean" else per_layer[int(layer)]
            out[k, nodes] = chosen.mean(0)[cells_pos].double().cpu().numpy()
    return out


def _dense_rows(module, graph, receivers, layer, eval_step: bool) -> np.ndarray:
    from interscale.evaluation.gene_ranking import show_everything

    transformer = getattr(module, "global_module", module)
    if not hasattr(transformer, "recording_attention"):
        raise ValueError("attention_rows: this module has no transformer.")
    g = show_everything([graph])[0] if eval_step else graph
    if eval_step and getattr(module, "mask_strategy", "node") == "gene":  # a gene-strategy step reads gene_mask: nothing hidden
        g.gene_mask = torch.zeros(g.x.shape, dtype=torch.bool)
    if getattr(g, "batch", None) is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=g.x.device)
    device = next(module.parameters()).device
    g = g.to(device)
    with torch.no_grad(), _rng_untouched(device), transformer.recording_attention() as record:
        module._common_step(g, "regression", "node")
        layers = list(transformer.transformer_encoder.layers)
        maps = [lay.get_attn_output_weights() for lay in layers]
    if any(m is None for m in maps) or not record:
        raise ValueError("attention_rows: the pass kept no attention map (pooled, hierarchical or kernel backends keep none).")
    entry = record[-1]
    kept = np.asarray(entry["index_nodes"][0], dtype=np.int64)
    chosen = torch.stack([m.float() for m in maps]).mean(0) if layer == "mean" else maps[int(layer)].float()
    seq = int(chosen.shape[-1]) - 1  # the map's keys: the cells (left padded), CLS last
    offset = seq - len(kept)
    heads = chosen.shape[0]  # one graph: [heads, S + 1, S + 1]
    a = chosen.view(heads, chosen.shape[-2], chosen.shape[-1]).mean(0)[:seq, :seq]  # cell queries over cell keys
    position = np.full(int(graph.num_nodes), -1, dtype=np.int64)
    position[kept] = offset + np.arange(len(kept))
    out = np.full((len(receivers), int(graph.num_nodes)), np.nan, dtype=np.float32)
    cols = position[kept]
    for k, r in enumerate(np.asarray(receivers, dtype=np.int64)):
        if position[r] < 0:
            continue
        out[k] = 0.0
        out[k, kept] = a[position[r], torch.as_tensor(cols, device=a.device)].double().cpu().numpy()
    return out


def attention_rows(module, graph, receivers, *, layer=-1, eval_step: bool = True) -> np.ndarray:
    """``[Q, N]``: each receiver's head-mean attention over the cells of the single graph ``graph`` (graph node order; CLS and the null key left out;
    NaN rows for receivers the pass did not keep). ``layer``: an index (-1 the last) or ``"mean"`` over layers. ``eval_step``: the clean pass
    (every cell visible) or the graph's own masks. Read-out models through their recorded read-out weights, dense models through the explicit
    maps; other backends refuse."""
    from interscale.evaluation.interaction_readout import readout_layers

    receivers = np.asarray(receivers, dtype=np.int64)
    was_training = module.training
    module.eval()
    try:
        if readout_layers(module) and getattr(module, "context_readout", "stream") == "additive":
            return _readout_rows(module, graph, receivers, layer, eval_step)
        return _dense_rows(module, graph, receivers, layer, eval_step)
    finally:
        module.train(was_training)


def row_summary(rows: np.ndarray, pos_um: np.ndarray, receivers) -> pd.DataFrame:
    """Per receiver: ``mass`` (the row's sum over cells), ``entropy`` (of the row normalised over the cells, divided by ``log`` of the number of
    other cells: 1 = uniform), ``max_share``, ``eff_keys`` (``1 / sum p^2``), ``mean_dist_um`` (the distance read, weighted by ``p``)."""
    receivers = np.asarray(receivers, dtype=np.int64)
    out = []
    for k, r in enumerate(receivers):
        row = np.nan_to_num(np.asarray(rows[k], dtype=np.float64))
        row[r] = 0.0
        mass = row.sum()
        if mass <= 0 or not np.isfinite(rows[k]).any():
            out.append({"receiver": int(r), "mass": np.nan, "entropy": np.nan, "max_share": np.nan, "eff_keys": np.nan, "mean_dist_um": np.nan})
            continue
        p = row / mass
        nz = p[p > 0]
        d = np.linalg.norm(pos_um - pos_um[r], axis=1)
        out.append({"receiver": int(r), "mass": float(mass), "entropy": float(-(nz * np.log(nz)).sum() / np.log(max(len(p) - 1, 2))),
                    "max_share": float(p.max()), "eff_keys": float(1.0 / (p**2).sum()), "mean_dist_um": float((p * d).sum())})  # fmt: skip
    return pd.DataFrame(out)


def row_agreement(a: np.ndarray, b: np.ndarray, receivers, *, top: int = 32) -> pd.DataFrame:
    """Per receiver, two models' rows for the same receivers: ``spearman`` over the union of both rows' ``top`` cells, ``top_overlap`` (shared
    share of the two top sets), ``js`` (Jensen-Shannon divergence of the rows normalised over the cells, natural log; 0 = identical)."""
    receivers = np.asarray(receivers, dtype=np.int64)
    out = []
    for k, r in enumerate(receivers):
        x, y = np.nan_to_num(np.asarray(a[k], dtype=np.float64)), np.nan_to_num(np.asarray(b[k], dtype=np.float64))
        x[r], y[r] = 0.0, 0.0
        if x.sum() <= 0 or y.sum() <= 0:
            out.append({"receiver": int(r), "spearman": np.nan, "top_overlap": np.nan, "js": np.nan})
            continue
        tx, ty = np.argsort(-x)[:top], np.argsort(-y)[:top]
        union = np.union1d(tx, ty)
        rho = spearmanr(x[union], y[union]).correlation if len(union) > 2 else np.nan
        p, q = x / x.sum(), y / y.sum()
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = lambda u: float(np.nansum(np.where(u > 0, u * np.log(u / m), 0.0)))  # noqa: E731
        out.append({"receiver": int(r), "spearman": float(rho), "top_overlap": len(np.intersect1d(tx, ty)) / float(top), "js": 0.5 * (kl(p) + kl(q))})
    return pd.DataFrame(out)
