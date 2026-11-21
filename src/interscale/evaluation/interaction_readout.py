"""Readouts of the interaction read-out (``.claude/ccc_model_plan.md``, IR; E3-E5): exact influence, planted truth, types, maps.

A model with ``context_only: query``, ``context_readout: additive``, the read-out attention (``attention.kind: readout``) and a
linear decoder predicts each cell's context part as a sum over the cells it reads:

    y_ctx(i) = W_dec sum_l sum_h sum_j a^l_hij W^l_O,h v^l_hj  +  intercept,

with ``a^l_hij`` the attention of layer ``l``, head ``h`` (the null key's column carries no value) and ``v^l_hj`` cell ``j``'s value.
So ``c_ij = W_dec sum_lh a^l_hij Y^l_hj`` (``Y = W_O v``) is cell ``j``'s exact contribution to cell ``i``'s prediction -- GITIII's
influence tensor, gene-resolved. It is a decomposition, not an intervention: removing ``j`` renormalises the others' weights, which
:func:`type_tests` measures directly (``ablation``) next to the decomposition (``decomposition``).

Everything runs one graph at a time and a block of query rows at a time (:func:`influence`), so whole sections fit: nothing
``N x N`` is kept. What it gives:

* :func:`readout_pass` -- one recorded pass (projected queries, keys, values and masks per layer);
* :func:`influence` -- per receiver: null mass per layer and head, total influence, the top-k senders by magnitude with their
  gene vectors and distances, and optionally every sender's score on a target gene set;
* :func:`planted_scores` -- synth_spot's planted truth (E3): per programme the planted sender in the top 5, sender / receiver /
  gene average precision, the range the influence reads at, null mass of receivers against non-receivers;
* :func:`head_windows` -- each read head's distance window (scale, strength, the distance at which its prior falls by ln 10), its
  null logit, and the content-rate factors by label;
* :func:`type_tests` -- per sender type x receiver type x gene: the correlation of the type's contribution (decomposition, and
  the change when its cells are removed as keys: ablation) with the receivers' observed residual, Wald z, BH (AMICI / GITIII),
  against the same with the sender labels shuffled within the graph (composition control);
* :func:`segmentation_flags` -- AMICI's spillover test, model-free;
* :func:`cell_maps` -- per cell its interaction mass (1 - null mass), the influence it receives and the influence it sends.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------- the recorded pass


def readout_layers(module) -> list:
    """The read-out attention of every layer of ``module``'s transformer, in order (empty when it has none)."""
    encoder = getattr(module, "transformer_encoder", None)
    if encoder is None:
        return []
    return [layer.self_attn for layer in encoder.layers if hasattr(layer.self_attn, "weights_rows")]


def check_decomposable(module) -> None:
    """Raise unless ``module``'s context part is a sum over senders: additive read-out, read-out attention, linear decoder."""
    if getattr(module, "context_only", "none") != "query" or getattr(module, "context_readout", "stream") != "additive":
        raise ValueError("the exact influence needs context_only query with context_readout additive.")
    if not readout_layers(module) or len(readout_layers(module)) != len(module.transformer_encoder.layers):
        raise ValueError("the exact influence needs attention.kind readout in every layer.")
    decoder = getattr(module, "decoder", None)
    if decoder is None or not hasattr(decoder, "decoder") or not isinstance(decoder.decoder, torch.nn.Linear):
        raise ValueError("the exact influence needs a linear decoder (model.decoder.type linear).")


@contextmanager
def recording(module):
    """Inside, every read-out layer records each call (``ReadoutAttention.recording``); yields the per-layer lists."""
    layers = readout_layers(module)
    saved = [layer.recording for layer in layers]
    for layer in layers:
        layer.recording = []
    try:
        yield [layer.recording for layer in layers]
    finally:
        for layer, old in zip(layers, saved):
            layer.recording = old


@dataclass
class ReadoutPass:
    """One recorded pass over one graph: per layer its record (``ReadoutAttention.recording``), the cells in sequence order, and
    the prediction's context part per cell (``[N, G]`` in the graph's node order, NaN for a cell the sequence left out)."""

    records: list
    index_nodes: list
    seq_len: int
    pos_um: np.ndarray
    context: np.ndarray
    hidden_keys: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def offset(self) -> int:
        """The sequence position of the first cell (left padding)."""
        return self.seq_len - 1 - len(self.index_nodes)

    def node_of(self, positions) -> np.ndarray:
        """Graph node index of sequence positions (cells only)."""
        return np.asarray(self.index_nodes)[np.asarray(positions) - self.offset]

    def position_of(self, nodes) -> np.ndarray:
        """Sequence position of graph nodes (all kept, as with max_seq_len at the largest graph)."""
        lookup = np.full(max(self.index_nodes) + 1, -1, dtype=np.int64)
        lookup[np.asarray(self.index_nodes)] = np.arange(len(self.index_nodes)) + self.offset
        return lookup[np.asarray(nodes)]


def readout_pass(module, batch, *, eval_step: bool, hide_nodes: np.ndarray | None = None, spatial_unit_um: float = 1.0) -> ReadoutPass:
    """One recorded, gradient-free pass of ``module`` (eval mode) over the single graph ``batch``.

    ``batch`` as the model should read it: the masked batch of the evaluation draw for a task that hides genes (synth_spot's
    programmes; ``eval_step=False`` so its masks apply), or the clean batch (``eval_step=True``: every cell visible, every cell a key
    -- the every-cell-a-target models' reading). ``hide_nodes`` (``[N]`` bool) removes those cells as keys on top (the ablation).
    """
    check_decomposable(module)
    if int(getattr(batch, "num_graphs", 1)) != 1:
        raise ValueError("readout_pass reads one graph at a time.")
    if getattr(batch, "batch", None) is None:
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad(), recording(module) as records:
            tokens = module.gex_embedding(batch)
            given = module.common_step_local_to_global(batch, tokens, eval_step=eval_step)
            mask = given.attention_mask
            if hide_nodes is not None:
                from interscale.tl.padding import pad_like

                seq = given.src_padding_mask.shape[1]
                hide = torch.as_tensor(np.asarray(hide_nodes, dtype=np.float32), device=tokens.device)[:, None]
                keys = pad_like(hide, batch.batch.to(tokens.device), given.index_nodes, seq)[..., 0] > 0.5
                keys = torch.nn.functional.pad(keys, (0, 1), value=False)
                mask = mask._replace(hidden_keys=keys if mask.hidden_keys is None else mask.hidden_keys | keys)
            out, padding, _ = module.forward(
                given.padded_emb, given.src_padding_mask, mask, rotary=given.rotary, **module._query_kwargs(given)
            )
            from interscale.module.base._step_output import gather_tokens

            rows = module.decoder(gather_tokens(out, padding)).float().cpu().numpy()
    finally:
        module.train(was_training)
    index = list(given.index_nodes[0])
    context = np.full((batch.num_nodes, rows.shape[1]), np.nan, dtype=np.float32)
    context[np.asarray(index)] = rows
    pos = batch.pos[:, :2].detach().cpu().numpy().astype(np.float64) * float(spatial_unit_um)
    hidden = None
    final = records[0][0]["context"].hidden_keys
    if final is not None:
        hidden = final[0].detach().cpu().numpy()
    return ReadoutPass([layer[0] for layer in records], index, int(out.shape[0]), pos, context, hidden)


# --------------------------------------------------------------------------- exact influence


def _values(module, rp: ReadoutPass) -> torch.Tensor:
    """``[layers * heads, L, E]``: every layer and head's projected values ``Y^l_hj = W^l_O,h v^l_hj`` (graph 0)."""
    blocks = []
    for layer, record in zip(readout_layers(module), rp.records):
        v = record["v"][0]  # [H, L, hd]
        heads, _, head_dim = v.shape
        w_o = layer.out_proj.weight.detach().to(v.dtype).view(-1, heads, head_dim)  # [E, H, hd]
        blocks.append(torch.einsum("ehd,hld->hle", w_o, v))
    return torch.cat(blocks, dim=0)


def _block_weights(module, rp: ReadoutPass, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(cells [rows, layers * heads, L], null [rows, layers * heads] or None)`` for query rows ``start..stop``."""
    cells, nulls = [], []
    seq_len = rp.seq_len
    for layer, record in zip(readout_layers(module), rp.records):
        weights = layer.weights_rows(record, start, stop)[0]  # [H, r, L(+1)]
        cells.append(weights[..., :seq_len].transpose(0, 1))
        if weights.shape[-1] > seq_len:
            nulls.append(weights[..., seq_len].transpose(0, 1))
    null = torch.cat(nulls, dim=1) if nulls else None
    return torch.cat(cells, dim=1), null


@torch.no_grad()
def influence(
    module,
    rp: ReadoutPass,
    receivers: np.ndarray,
    *,
    targets: torch.Tensor | None = None,
    receiver_target: np.ndarray | None = None,
    top_k: int = 16,
    gene_top: int = 4,
    type_codes: np.ndarray | None = None,
    n_types: int = 0,
    chunk_pairs: int = 2**25,
) -> dict:
    """The exact influence on each of ``receivers`` (graph node indices), a block of query rows at a time.

    Returns per receiver (row order of ``receivers``): ``null`` ``[R, layers * heads]`` (the null key's mass; zeros without one),
    ``total`` ``[R]`` (sum over senders of ``||c_ij||``), ``top_nodes`` / ``top_mag`` / ``top_dist`` ``[R, top_k]`` (senders by
    ``||c_ij||``, graph node index, magnitude, distance um), ``top_genes`` ``[R, gene_top, G]`` (``c_ij`` of the first ``gene_top``).
    With ``targets`` ``[T, G]`` (gene weights) and ``receiver_target`` ``[R]`` (which row of ``targets`` each receiver is scored on,
    -1 for none): ``scores`` ``[R, N]``, every sender's contribution to the receiver's target, ``t . c_ij`` (graph node order; NaN for
    cells outside the sequence). With ``type_codes`` ``[N]`` (graph node order) and ``n_types``: ``by_type`` ``[R, n_types, G]``, the
    summed contribution of each sender type (GITIII's aggregated influence tensor).
    """
    check_decomposable(module)
    w_dec = module.decoder.decoder.weight.detach().float()  # [G, E]
    device = w_dec.device
    y = _values(module, rp).float().to(device)  # [X, L, E]
    n_x, seq_len, _ = y.shape
    gram = w_dec.T @ w_dec  # [E, E]
    metric = torch.einsum("xle,ef,ylf->lxy", y, gram, y)  # [L, X, X]
    receivers = np.asarray(receivers, dtype=np.int64)
    positions = rp.position_of(receivers)
    if (positions < 0).any():
        raise ValueError("a receiver is not in the sequence (pad_batch subsampled the graph: raise max_seq_len).")
    order = np.argsort(positions)
    pos_t = torch.as_tensor(rp.pos_um, dtype=torch.float32, device=device)
    cells_nodes = torch.as_tensor(np.asarray(rp.index_nodes), dtype=torch.long, device=device)
    cell_slots = torch.arange(rp.offset, seq_len - 1, device=device)

    n_rec, n_genes = len(receivers), w_dec.shape[0]
    out = {
        "null": np.zeros((n_rec, n_x), dtype=np.float32),
        "total": np.zeros(n_rec, dtype=np.float32),
        "top_nodes": np.full((n_rec, top_k), -1, dtype=np.int64),
        "top_mag": np.zeros((n_rec, top_k), dtype=np.float32),
        "top_dist": np.full((n_rec, top_k), np.nan, dtype=np.float32),
        "top_genes": np.zeros((n_rec, gene_top, n_genes), dtype=np.float32),
    }
    z = None
    if targets is not None:
        targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
        z = torch.einsum("tg,ge,xle->txl", targets, w_dec, y)  # [T, X, L]
        out["scores"] = np.full((n_rec, rp.context.shape[0]), np.nan, dtype=np.float32)
    onehot = None
    if type_codes is not None and n_types > 0:
        codes = torch.as_tensor(np.asarray(type_codes, dtype=np.int64), device=device)
        slot_codes = torch.full((seq_len,), -1, dtype=torch.long, device=device)
        slot_codes[cell_slots] = codes[cells_nodes]
        onehot = torch.zeros(seq_len, n_types, device=device)
        valid = slot_codes >= 0
        onehot[valid, slot_codes[valid]] = 1.0
        out["by_type"] = np.zeros((n_rec, n_types, n_genes), dtype=np.float32)

    rows_per_block = max(1, int(chunk_pairs) // max(1, n_x * seq_len))
    sorted_pos = positions[order]
    i = 0
    while i < len(order):
        start = int(sorted_pos[i])
        stop = min(start + rows_per_block, seq_len)
        j = i
        while j < len(order) and sorted_pos[j] < stop:
            j += 1
        idx = order[i:j]
        local = torch.as_tensor(sorted_pos[i:j] - start, dtype=torch.long, device=device)
        weights, null = _block_weights(module, rp, start, stop)
        weights = weights[local].float()  # [r, X, L]
        if null is not None:
            out["null"][idx] = null[local].float().cpu().numpy()
        mag = torch.einsum("rxl,lxy,ryl->rl", weights, metric, weights).clamp_min(0).sqrt()  # [r, L]
        out["total"][idx] = mag.sum(1).cpu().numpy()
        k = min(top_k, seq_len)
        top_mag, top_slot = mag.topk(k, dim=1)
        is_cell = (top_slot >= rp.offset) & (top_slot < seq_len - 1)
        top_node = torch.where(is_cell, cells_nodes[(top_slot - rp.offset).clamp(0, len(rp.index_nodes) - 1)], torch.full_like(top_slot, -1))
        receiver_node = torch.as_tensor(receivers[idx], dtype=torch.long, device=device)
        dist = torch.linalg.norm(pos_t[top_node.clamp_min(0)] - pos_t[receiver_node][:, None, :], dim=-1)
        out["top_nodes"][idx, :k] = top_node.cpu().numpy()
        out["top_mag"][idx, :k] = (top_mag * is_cell).cpu().numpy()
        out["top_dist"][idx, :k] = torch.where(is_cell, dist, torch.full_like(dist, float("nan"))).cpu().numpy()
        g = min(gene_top, k)
        sel = top_slot[:, :g]  # [r, g]
        w_sel = torch.gather(weights, 2, sel[:, None, :].expand(-1, n_x, -1))  # [r, X, g]
        y_sel = y[:, sel, :].permute(1, 0, 2, 3)  # [r, X, g, E]
        vec = torch.einsum("rxg,rxge->rge", w_sel, y_sel) @ w_dec.T  # [r, g, G]
        out["top_genes"][idx, :g] = vec.cpu().numpy()
        if z is not None and receiver_target is not None:
            tgt = np.asarray(receiver_target)[idx]
            have = tgt >= 0
            if have.any():
                zt = z[torch.as_tensor(tgt[have], dtype=torch.long, device=device)]  # [r', X, L]
                s = torch.einsum("rxl,rxl->rl", weights[torch.as_tensor(np.flatnonzero(have), device=device)], zt)
                full = np.full((int(have.sum()), rp.context.shape[0]), np.nan, dtype=np.float32)
                full[:, np.asarray(rp.index_nodes)] = s[:, rp.offset : seq_len - 1].cpu().numpy()
                out["scores"][idx[have]] = full
        if onehot is not None:
            agg = torch.einsum("rxl,xle,lt->rte", weights, y, onehot) @ w_dec.T  # [r, T, G]
            out["by_type"][idx] = agg.cpu().numpy()
        i = j
    return out


# --------------------------------------------------------------------------- planted truth (synth_spot)


def programme_targets(var: pd.DataFrame, programmes: list[str], *, column: str = "interaction", roles=("response", "receiver")):
    """``(targets [P, G] float, programmes kept)``: per planted programme the indicator of its response genes (normalised)."""
    rows, kept = [], []
    for name in programmes:
        genes = (var[column].astype(str) == str(name)) & var["role"].astype(str).isin(roles)
        if genes.any():
            weights = genes.to_numpy().astype(np.float32)
            rows.append(weights / weights.sum())
            kept.append(str(name))
    return (np.stack(rows) if rows else np.zeros((0, len(var)), dtype=np.float32)), kept


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    keep = np.isfinite(scores)
    labels = labels[keep].astype(bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, scores[keep]))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    keep = np.isfinite(scores)
    labels = labels[keep].astype(bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores[keep]))


def planted_scores(
    module,
    rp: ReadoutPass,
    edges: pd.DataFrame,
    var: pd.DataFrame,
    *,
    receivers_per_programme: int = 48,
    seed: int = 0,
    top: int = 5,
    knn: int = 49,
    negatives: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """synth_spot's planted truth read off the exact influence of one graph (E3).

    ``edges``: the planted pairs inside this graph as node indices (``_spatial_weights.graph_edges``), with ``program`` and
    ``range_class``. Per programme: up to ``receivers_per_programme`` receivers and as many cells that receive nothing from it
    (seeded). Columns: ``top5`` (share of receivers with a planted sender among the ``top`` senders by ``|t . c_ij|``),
    ``sender_ap`` (mean over receivers of the average precision of ``|t . c_ij|`` over every cell, planted senders positive;
    ``sender_ap_knn<knn>`` the same over the receiver's ``knn`` nearest cells, GITIII's neighbourhood),
    ``receiver_ap`` / ``receiver_auroc`` (the receivers against the others by the influence on the programme's genes, ``sum_j |t .
    c_ij|``), ``mass_auroc`` (the same by interaction mass ``1 - null``), ``gene_ap`` (per receiver, its planted senders'
    summed gene vector ranks the programme's genes among all; from the top senders' gene vectors), ``range_um`` (median over
    receivers of the influence-weighted mean distance), ``planted_dist_um`` (median distance of the planted senders).

    ``negatives``: per programme, ``[n_nodes]`` bool, the cells that may serve as non-receivers (default: every cell in the sequence).
    On data with cell types pass the receiver type's cells: a model whose query holds the type would otherwise score the type, not
    the interaction (GITIII's readout compares within the receiver type too). A programme without eligible negatives gets NaN
    receiver scores.
    """
    rng = np.random.default_rng(seed)
    programmes = sorted(edges["program"].astype(str).unique())
    targets, kept = programme_targets(var, programmes)
    if not kept:
        return pd.DataFrame()
    n_nodes = rp.context.shape[0]
    in_seq = np.zeros(n_nodes, dtype=bool)
    in_seq[np.asarray(rp.index_nodes)] = True
    chosen, target_of, groups = [], [], []
    for p, name in enumerate(kept):
        pairs = edges[edges["program"].astype(str) == name]
        recv = np.unique(pairs["receiver"].to_numpy())
        recv = recv[in_seq[recv]]
        if not len(recv):
            continue
        recv = rng.choice(recv, size=min(len(recv), receivers_per_programme), replace=False)
        pool = in_seq if negatives is None or name not in negatives else in_seq & np.asarray(negatives[name], dtype=bool)
        others = np.setdiff1d(np.flatnonzero(pool), np.unique(pairs["receiver"].to_numpy()))
        others = rng.choice(others, size=min(len(others), len(recv)), replace=False)
        for node, is_recv in [(r, True) for r in recv] + [(o, False) for o in others]:
            chosen.append(node)
            target_of.append(p)
            groups.append((name, is_recv))
    if not chosen:
        return pd.DataFrame()
    chosen = np.asarray(chosen)
    inf = influence(
        module, rp, chosen, targets=torch.as_tensor(targets), receiver_target=np.asarray(target_of), top_k=max(top, 16), gene_top=8
    )
    rows = []
    for p, name in enumerate(kept):
        sel = np.array([g[0] == name for g in groups])
        if not sel.any():
            continue
        is_recv = np.array([g[1] for g in groups])[sel]
        idx = np.flatnonzero(sel)
        pairs = edges[edges["program"].astype(str) == name]
        senders_of = pairs.groupby("receiver")["sender"].apply(lambda s: set(s.to_numpy().tolist())).to_dict()
        genes = targets[p] > 0
        top_hits, sender_ap, sender_ap_knn, gene_ap, ranges, planted_d = [], [], [], [], [], []
        for k in idx[is_recv]:
            receiver = int(chosen[k])
            scores = np.abs(inf["scores"][k])
            labels = np.zeros(n_nodes, dtype=bool)
            labels[list(senders_of.get(receiver, set()))] = True
            labels[receiver] = False
            scores[receiver] = np.nan
            order = np.argsort(-np.nan_to_num(scores, nan=-1.0))[:top]
            top_hits.append(bool(labels[order].any()))
            sender_ap.append(_average_precision(scores, labels))
            finite = np.isfinite(scores)
            d = np.linalg.norm(rp.pos_um - rp.pos_um[receiver], axis=1)
            # The same over the receiver's knn nearest cells only: what a method that reads a fixed neighbourhood (GITIII: 49) is
            # scored on.
            near = np.argsort(np.where(np.arange(n_nodes) == receiver, np.inf, d))[:knn]
            sender_ap_knn.append(_average_precision(scores[near], labels[near]))
            if scores[finite].sum() > 0:
                ranges.append(float((scores[finite] * d[finite]).sum() / scores[finite].sum()))
            planted = list(senders_of.get(receiver, set()))
            if planted:
                planted_d.append(float(np.median(d[planted])))
            # The planted senders among the receiver's top senders: their gene vectors summed.
            tops = inf["top_nodes"][k, : inf["top_genes"].shape[1]]
            hit = np.isin(tops, planted)
            if hit.any():
                vec = np.abs(inf["top_genes"][k][hit].sum(0))
                gene_ap.append(_average_precision(vec, genes))
        target_influence = np.nansum(np.abs(inf["scores"][idx]), axis=1)
        mass = 1.0 - inf["null"][idx].mean(1)
        range_class = str(pairs["range_class"].astype(str).mode().iloc[0]) if len(pairs) else ""
        rows.append(
            {
                "program": name,
                "range_class": range_class,
                "receivers": int(is_recv.sum()),
                "negatives": int((~is_recv).sum()),
                f"top{top}": float(np.mean(top_hits)) if top_hits else np.nan,
                "sender_ap": float(np.nanmean(sender_ap)) if sender_ap else np.nan,
                f"sender_ap_knn{knn}": float(np.nanmean(sender_ap_knn)) if sender_ap_knn else np.nan,
                "receiver_ap": _average_precision(target_influence, is_recv) if (~is_recv).any() else np.nan,
                "receiver_auroc": _auroc(target_influence, is_recv),
                "mass_auroc": _auroc(mass, is_recv),
                "gene_ap": float(np.nanmean(gene_ap)) if gene_ap else np.nan,
                "gene_ap_receivers": len(gene_ap),
                "range_um": float(np.median(ranges)) if ranges else np.nan,
                "planted_dist_um": float(np.median(planted_d)) if planted_d else np.nan,
                "null_receivers": float(inf["null"][idx[is_recv]].mean()) if is_recv.any() else np.nan,
                "null_others": float(inf["null"][idx[~is_recv]].mean()) if (~is_recv).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- windows and length scales


def head_windows(module) -> pd.DataFrame:
    """Per read layer and head: the distance prior's scale and strength, the distance ``d10`` at which the prior falls by ``ln 10``
    (exponential: ``l ln10 / s``; gaussian: ``l sqrt(2 ln10 / s)``), the null logit, and -- with the content rate -- the receiver
    factor ``exp(u_h[t])`` per label code (columns ``rate_t<code>``)."""
    rows = []
    bias = getattr(module, "attention_bias", None)
    encoder = None
    if bias is not None and "distance" in getattr(bias, "encoders", {}):
        encoder = bias.encoders["distance"]
    for l, layer in enumerate(readout_layers(module)):
        for h in range(layer.num_heads):
            row = {"layer": l, "head": h}
            if encoder is not None and hasattr(encoder, "log_scale"):
                served = list(getattr(encoder, "served_heads", range(layer.num_heads)))
                if h in served:
                    e = served.index(h)
                    scale = float(encoder.log_scale[e].detach().exp())
                    strength = float(encoder.strength[e].detach())
                    row["scale_um"], row["strength"] = scale, strength
                    if strength > 0:
                        row["d10_um"] = (
                            scale * math.log(10.0) / strength
                            if encoder.shape == "exponential"
                            else scale * math.sqrt(2.0 * math.log(10.0) / strength)
                        )
            if layer.null_key:
                row["null_logit"] = float(layer.null_logit[h].detach())
            if layer.content_rate:
                factors = layer.rate_type[:, h].detach().clamp(-8, 8).exp().cpu().numpy()
                for code, value in enumerate(factors):
                    row[f"rate_t{code}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- sender type x receiver type (real data)


class TypeTestAccumulator:
    """Sufficient statistics of the correlation, per sender type x receiver type x gene, between a contribution and the receivers'
    observed residual, summed over graphs (so the test pools sections, as Stouffer would pool their z)."""

    def __init__(self, n_types: int, n_genes: int):
        shape = (n_types, n_types, n_genes)
        self.n = np.zeros(shape[:2])
        self.sx, self.sy, self.sxx, self.syy, self.sxy = (np.zeros(shape) for _ in range(5))

    def add(self, contribution: np.ndarray, residual: np.ndarray, receiver_codes: np.ndarray) -> None:
        """``contribution`` ``[R, S, G]`` (per receiver, sender type), ``residual`` ``[R, G]``, ``receiver_codes`` ``[R]``."""
        for r in np.unique(receiver_codes):
            rows = receiver_codes == r
            x, y = contribution[rows], residual[rows][:, None, :]
            self.n[:, r] += rows.sum()
            self.sx[:, r] += x.sum(0)
            self.sy[:, r] += np.broadcast_to(y, x.shape).sum(0)
            self.sxx[:, r] += (x * x).sum(0)
            self.syy[:, r] += np.broadcast_to(y * y, x.shape).sum(0)
            self.sxy[:, r] += (x * y).sum(0)

    def table(self, type_names: list[str], gene_names: list[str], alpha: float = 0.05) -> pd.DataFrame:
        """Per sender type x receiver type: the number of genes with BH-significant correlation (Wald z of Pearson r), and the
        strongest; long form per (sender, receiver)."""
        from scipy.stats import norm

        n = self.n[..., None]
        cov = self.sxy / np.maximum(n, 1) - (self.sx / np.maximum(n, 1)) * (self.sy / np.maximum(n, 1))
        vx = self.sxx / np.maximum(n, 1) - (self.sx / np.maximum(n, 1)) ** 2
        vy = self.syy / np.maximum(n, 1) - (self.sy / np.maximum(n, 1)) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / np.sqrt(vx * vy)
            z = r * np.sqrt(np.maximum(n - 2, 0)) / np.sqrt(np.clip(1 - r * r, 1e-12, None))
        p = 2 * norm.sf(np.abs(np.nan_to_num(z)))
        p[~np.isfinite(z)] = 1.0
        q = _bh(p.ravel()).reshape(p.shape)
        rows = []
        for s, sender in enumerate(type_names):
            for rr, receiver in enumerate(type_names):
                if self.n[s, rr] < 3:
                    continue
                sig = (q[s, rr] < alpha) & (r[s, rr] > 0)
                sig_neg = (q[s, rr] < alpha) & (r[s, rr] < 0)
                best = int(np.nanargmax(np.nan_to_num(z[s, rr], nan=-np.inf)))
                rows.append(
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "receivers": int(self.n[s, rr]),
                        "genes_up": int(sig.sum()),
                        "genes_down": int(sig_neg.sum()),
                        "top_gene": gene_names[best],
                        "top_r": float(r[s, rr, best]),
                        "top_z": float(z[s, rr, best]),
                        "mean_abs_contribution": float(np.sqrt(np.maximum(self.sxx[s, rr] / max(self.n[s, rr], 1), 0)).mean()),
                    }
                )
        return pd.DataFrame(rows)


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


@torch.no_grad()
def type_contributions(
    module,
    batch,
    rp: ReadoutPass,
    codes: np.ndarray,
    n_types: int,
    *,
    eval_step: bool,
    shuffle_seed: int | None = None,
    spatial_unit_um: float = 1.0,
) -> dict:
    """Per cell of the graph and sender type: ``decomposition`` ``[N, T, G]`` (the summed exact contributions) and ``ablation``
    ``[N, T, G]`` (the context part's change when every cell of the type is removed as a key: full minus ablated).

    ``shuffle_seed``: the sender labels are shuffled among the graph's cells first (the same counts: composition kept, which cells
    are of a type random) -- the control both columns are read against.
    """
    codes = np.asarray(codes, dtype=np.int64)
    senders = codes.copy()
    if shuffle_seed is not None:
        senders = np.random.default_rng(shuffle_seed).permutation(senders)
    nodes = np.asarray(rp.index_nodes)
    inf = influence(module, rp, nodes, type_codes=senders, n_types=n_types, top_k=1, gene_top=1)
    decomposition = np.full((len(codes), n_types, rp.context.shape[1]), np.nan, dtype=np.float32)
    decomposition[nodes] = inf["by_type"]
    ablation = np.full_like(decomposition, np.nan)
    for t in range(n_types):
        hide = senders == t
        if not hide.any():
            continue
        ablated = readout_pass(module, batch, eval_step=eval_step, hide_nodes=hide, spatial_unit_um=spatial_unit_um)
        ablation[:, t] = rp.context - ablated.context
    return {"decomposition": decomposition, "ablation": ablation, "senders": senders}


# --------------------------------------------------------------------------- spillover (model-free)


def segmentation_flags(
    x: np.ndarray,
    codes: np.ndarray,
    pos_um: np.ndarray,
    graph: np.ndarray,
    type_names: list[str],
    gene_names: list[str],
    *,
    near_um: float = 20.0,
    far_um: float = 100.0,
    pairs: list[tuple[int, int]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """AMICI's segmentation test, per sender type S, receiver type R and gene: is the gene higher in R cells within ``near_um`` of an
    S cell than in S cells at least ``far_um`` from any R cell (one-sided Mann-Whitney)? If not, a receiver's apparent response may be
    the sender's transcripts read into the receiver's segmentation: ``flagged``. Pools graphs (neighbours never across graphs).
    Only meaningful for genes that are not expressed by both types (AMICI's caveat)."""
    from scipy.spatial import cKDTree
    from scipy.stats import mannwhitneyu

    n_types = len(type_names)
    pairs = pairs or [(s, r) for s in range(n_types) for r in range(n_types) if s != r]
    near_recv = {pair: [] for pair in pairs}
    far_send = {pair: [] for pair in pairs}
    for g in np.unique(graph):
        rows = np.flatnonzero(graph == g)
        tree_of = {}
        for t in range(n_types):
            members = rows[codes[rows] == t]
            tree_of[t] = (members, cKDTree(pos_um[members]) if len(members) else None)
        for s, r in pairs:
            send, s_tree = tree_of[s]
            recv, r_tree = tree_of[r]
            if s_tree is None or r_tree is None:
                continue
            d_recv, _ = s_tree.query(pos_um[recv], k=1)
            near_recv[(s, r)].append(recv[d_recv <= near_um])
            d_send, _ = r_tree.query(pos_um[send], k=1)
            far_send[(s, r)].append(send[d_send >= far_um])
    out = []
    for s, r in pairs:
        a = np.concatenate(near_recv[(s, r)]) if near_recv[(s, r)] else np.zeros(0, dtype=np.int64)
        b = np.concatenate(far_send[(s, r)]) if far_send[(s, r)] else np.zeros(0, dtype=np.int64)
        if len(a) < 5 or len(b) < 5:
            continue
        stat = mannwhitneyu(x[a], x[b], alternative="greater", axis=0)
        for k, gene in enumerate(gene_names):
            out.append(
                {
                    "sender": type_names[s],
                    "receiver": type_names[r],
                    "gene": gene,
                    "near_receivers": len(a),
                    "far_senders": len(b),
                    "p": float(stat.pvalue[k]),
                }
            )
    table = pd.DataFrame(out)
    if len(table):
        table["q"] = _bh(table["p"].to_numpy())
        table["flagged"] = table["q"] >= alpha
    return table


# --------------------------------------------------------------------------- maps


@torch.no_grad()
def cell_maps(module, rp: ReadoutPass, *, top_k: int = 16) -> dict:
    """Per cell of the graph (graph node order, NaN outside the sequence): ``mass`` (``1 - null`` mass, mean over layers and heads),
    ``received`` (sum over senders of ``||c_ij||``), ``sent`` (approximately the sum over receivers of ``||c_ij||``, from each
    receiver's ``top_k`` senders), ``reach_um`` (influence-weighted mean distance of the top senders)."""
    nodes = np.asarray(rp.index_nodes)
    inf = influence(module, rp, nodes, top_k=top_k, gene_top=1)
    n = rp.context.shape[0]
    mass = np.full(n, np.nan, dtype=np.float32)
    received = np.full(n, np.nan, dtype=np.float32)
    reach = np.full(n, np.nan, dtype=np.float32)
    sent = np.zeros(n, dtype=np.float32)
    mass[nodes] = 1.0 - inf["null"].mean(1)
    received[nodes] = inf["total"]
    weights = inf["top_mag"]
    with np.errstate(invalid="ignore", divide="ignore"):
        reach[nodes] = np.nansum(weights * inf["top_dist"], axis=1) / weights.sum(1)
    valid = inf["top_nodes"] >= 0
    np.add.at(sent, inf["top_nodes"][valid], weights[valid])
    return {"mass": mass, "received": received, "sent": sent, "reach_um": reach}
