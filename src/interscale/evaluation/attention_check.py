"""Is the attention computed as the code says, and what does each way of reading it say? (roadmap.md, T4)

**Is it computed correctly?** :func:`check_attention` runs a masked pass and recomputes every layer's
attention from the inputs that layer actually received:

* ``reference`` -- a plain implementation of the definition, ``softmax(q kᵀ / √d + mask) v``, with the
  rotation (RoPE), the dense mask (blocked pairs, merged biases) and the key padding as the layer got them;
  compared are the map the layer stores (what every readout reads) and the layer's output (what the
  model uses). A sparse normaliser (entmax, top-k) is the layer's own, so for those only the plumbing
  around it is checked;
* ``torch`` -- torch's stock ``nn.MultiheadAttention`` with the same weights, where it can take the same
  inputs (torch's vendored attention, no rotation);
* ``torch_layer`` -- the whole encoder layer against torch's stock ``nn.TransformerEncoderLayer`` with the
  same weights (no rotation, no centring of queries and keys); torch's layer takes its fused attention
  path (``need_weights=False``), so this also compares the explicit and the fused computation.

The vendored attention (``transformer_utils.multi_head_attention_forward_with_gradients``) is a copy of
torch's, edited so the map stays in the autograd graph and comes back per head; these checks are what
says the copy and its edits compute what torch computes, on the real model and inputs. Kernel backends
(Performer, linear) and Exphormer are skipped: they estimate or restrict softmax attention by design
(``attention_diagnostics``' ``tv_softmax`` measures that).

**What does each readout say?** :func:`attention_readouts` scores, for sampled query cells of a masked
pass, every other cell by:

* ``attn_last`` -- the last layer's map, the mean over heads; ``attn_mean`` -- the mean over layers too;
* ``rollout`` -- attention rollout with the residual path (Abnar & Zuidema 2020), ``Π_l (Ā_l + I) / 2``;
* ``grad_attn_last`` -- Chefer's rule 5 on the last layer: the mean over heads of ``(∂t/∂A ⊙ A)⁺``;
* ``relevance`` -- Chefer's rules 5 and 6 over every layer (``SelfAttentionRelevance``), from the query's
  own target ``t``;
* ``relevance_sum`` -- the same rules from the summed encoder output, the target ``evaluate()`` and the
  manuscript's CLS and flow analyses used; after the encoder's affine LayerNorm it is degenerate
  (``.claude/linear_probing_docs.md``), shown so the difference can be seen;
* ``grad_x_token`` -- ``|(x_j - x̄) · ∂t/∂x_j|`` at the tokens entering the transformer: gradient times
  input from the mean token, which reads no attention map.

Query ``i``'s target is ``½ ‖ŷ_i − μ‖²`` over its scored entries, ``μ`` the training gene means: how far
its prediction moves from the mean. Its gradient has the same sign meaning for every query, which Chefer's
``⁺`` needs (a random projection would flip it with its sign).

Read against the planted pairs: the sender's rank in its receiver's scores among cells as far away
(``matched_ranks``: 0.5 is blind to the sender, a distance kernel for one; 1 the top); between readouts,
the rank correlation of their scores per query; and each readout's span. Read every number against the
untrained twin: a readout that ranks senders as well untrained reads the architecture, not what was learned.

Both functions consume no RNG and leave the module as they found it (training flag, gradients).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from interscale.evaluation._spatial_weights import graph_edges, hidden_cells, matched_ranks
from interscale.evaluation.attention_diagnostics import _refuse_pooled
from interscale.evaluation.online_probes import _rng_untouched

#: The readouts :func:`attention_readouts` compares, in the order the tables list them.
READOUTS = ("attn_last", "attn_mean", "rollout", "grad_attn_last", "relevance", "relevance_sum", "grad_x_token")
#: The quantities :func:`check_attention` compares, per layer.
CHECKS = (
    ("reference", "weights"),
    ("reference", "output"),
    ("torch", "weights"),
    ("torch", "output"),
    ("torch_layer", "output"),
)


def _additive(mask: torch.Tensor | None, like: torch.Tensor):
    """A boolean (True: blocked) or float mask as the float added to the logits, ``-inf`` where blocked."""
    if mask is None:
        return 0.0
    if mask.dtype == torch.bool:
        return torch.zeros(mask.shape, dtype=like.dtype, device=like.device).masked_fill(mask, float("-inf"))
    return mask.to(like.dtype)


def reference_attention(attn, query, key, value, attn_mask=None, key_padding_mask=None, rotary=None, normalise=None):
    """``(output [L, B, E], weights [B * heads, L, S])`` of ``attn``'s weights by the definition, written out.

    ``attn`` is torch's vendored attention or a :class:`~interscale.module.global_modules.attention.DenseAttention`
    (``in_proj_weight``, ``in_proj_bias``, ``out_proj``, ``num_heads``, ``head_dim``). ``attn_mask`` is ``[L, S]``
    or ``[B * heads, L, S]``, ``key_padding_mask`` ``[B, S]``, each boolean (True: blocked) or float (added).
    ``rotary`` is ``(cos, sin)`` as the layer applies it; ``normalise`` replaces the softmax over the last axis.
    """
    from interscale.module.global_modules.transformer_utils import apply_rotary

    seq_len, n_graphs, _ = query.shape
    heads, width = int(attn.num_heads), int(attn.head_dim)
    inner = heads * width
    w_q, w_k, w_v = attn.in_proj_weight.split(inner)
    b_q, b_k, b_v = (None, None, None) if attn.in_proj_bias is None else attn.in_proj_bias.split(inner)

    def split(x, w, b):
        return F.linear(x, w, b).reshape(x.shape[0], n_graphs * heads, width).transpose(0, 1)

    q, k, v = split(query, w_q, b_q), split(key, w_k, b_k), split(value, w_v, b_v)
    if rotary is not None:
        q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
    scores = q @ k.transpose(1, 2) / math.sqrt(width)
    scores = scores + _additive(attn_mask, scores)
    if key_padding_mask is not None:
        pad = _additive(key_padding_mask, scores)
        scores = scores + pad.view(n_graphs, 1, 1, -1).expand(-1, heads, 1, -1).reshape(n_graphs * heads, 1, -1)
    weights = torch.softmax(scores, dim=-1) if normalise is None else normalise(scores)
    out = (weights @ v).transpose(0, 1).reshape(seq_len, n_graphs, inner)
    return attn.out_proj(out), weights


def _checkable(attn) -> bool:
    """Whether :func:`reference_attention` computes what ``attn`` computes: softmax-style dense attention."""
    from interscale.module.global_modules.attention import DenseAttention

    return not getattr(attn, "is_backend", False) or isinstance(attn, DenseAttention)


def _is_torchs(attn) -> bool:
    from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits

    return type(attn) is MultiHeadAttentionWithEdits


def _stock_attention(attn):
    stock = torch.nn.MultiheadAttention(attn.embed_dim, attn.num_heads, dropout=0.0, bias=attn.in_proj_bias is not None)
    stock.load_state_dict(attn.state_dict())
    return stock.to(attn.in_proj_weight.device).eval()


def _stock_layer(layer):
    stock = torch.nn.TransformerEncoderLayer(
        layer.self_attn.embed_dim,
        layer.self_attn.num_heads,
        dim_feedforward=layer.linear1.out_features,
        dropout=0.0,
        activation=layer.activation,
        layer_norm_eps=layer.norm1.eps,
        norm_first=layer.norm_first,
        bias=layer.linear1.bias is not None,
    )
    stock.load_state_dict(layer.state_dict())
    return stock.to(layer.linear1.weight.device).eval()


def _difference(found: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """The largest absolute difference over finite entries, and the largest absolute value (its scale)."""
    found, expected = found.detach().double(), expected.detach().double()
    if found.shape != expected.shape:
        return float("inf"), float("nan")
    both = torch.isfinite(found) & torch.isfinite(expected)
    if not torch.equal(torch.isfinite(found), torch.isfinite(expected)):
        return float("inf"), float("nan")
    if not both.any():
        return 0.0, 0.0
    return float((found - expected)[both].abs().max()), float(expected[both].abs().max())


def check_attention(
    module, data_list, *, prediction_task: str, prediction_level: str, device, max_graphs: int = 1
) -> pd.DataFrame:
    """Every layer's attention of a masked pass, recomputed from its own inputs; see the module docstring.

    The graphs keep the masks they carry, as in the other masked-pass readouts.

    Returns
    -------
    pd.DataFrame
        ``layer, check, quantity, max_abs_diff, max_abs, n_graphs, backend``: per layer and check (``reference``,
        ``torch``, ``torch_layer``) and quantity (``weights``, ``output``), the largest absolute difference over
        the graphs and the largest absolute value of the expected tensor. A blocked entry that is finite in one
        and not the other counts as ``inf``. Checks that do not apply to a layer (rotation, centring, another
        backend) are left out.
    """
    transformer = getattr(module, "global_module", module)
    if not hasattr(transformer, "recording_attention"):
        warnings.warn(
            "check_attention: this module has no transformer; nothing to check.", RuntimeWarning, stacklevel=2
        )
        return pd.DataFrame()
    _refuse_pooled(transformer, "check_attention")
    layers = list(transformer.transformer_encoder.layers)
    found: dict[tuple, list] = defaultdict(lambda: [0.0, 0.0, 0])

    def add(layer: int, check: str, quantity: str, diff: tuple[float, float]) -> None:
        acc = found[(layer, check, quantity)]
        acc[0] = max(acc[0], diff[0])
        if np.isfinite(diff[1]):
            acc[1] = max(acc[1], diff[1])
        acc[2] += 1

    def attn_hook(index):
        def hook(attn, args, kwargs, output):
            if not _checkable(attn):
                return
            query, key, value = args[:3]
            out, weights = output
            masks = {"attn_mask": kwargs.get("attn_mask"), "key_padding_mask": kwargs.get("key_padding_mask")}
            rotary = kwargs.get("rotary")
            ref_out, ref_weights = reference_attention(
                attn, query, key, value, rotary=rotary, normalise=getattr(attn, "_normalise", None), **masks
            )
            add(index, "reference", "output", _difference(out, ref_out))
            if weights is not None:
                add(index, "reference", "weights", _difference(weights, ref_weights))
            if _is_torchs(attn) and rotary is None:
                stock_out, stock_weights = _stock_attention(attn)(
                    query, key, value, need_weights=True, average_attn_weights=False, **masks
                )
                add(index, "torch", "output", _difference(out, stock_out))
                if weights is not None:
                    add(index, "torch", "weights", _difference(weights, stock_weights.flatten(0, 1)))

        return hook

    def layer_hook(index):
        def hook(layer, args, kwargs, output):
            # The rotation and centring are set on the layer for the whole encoder pass, so they are readable here.
            if not _is_torchs(layer.self_attn) or layer.rotary is not None or layer.center_qk:
                return
            stock = _stock_layer(layer)(
                args[0],
                src_mask=kwargs.get("src_mask"),
                src_key_padding_mask=kwargs.get("src_key_padding_mask"),
                is_causal=bool(kwargs.get("is_causal", False)),
            )
            add(index, "torch_layer", "output", _difference(output, stock))

        return hook

    graphs = list(data_list)[: max(int(max_graphs), 1)]
    was_training = module.training
    module.eval()
    handles = []
    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(attn_hook(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(layer_hook(index), with_kwargs=True))
    try:
        loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
        with torch.no_grad(), _rng_untouched(device), transformer.recording_attention():
            for batch in loader:
                module._common_step(batch.to(device), prediction_task, prediction_level)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    backends = {index: type(layer.self_attn).__name__ for index, layer in enumerate(layers)}
    rows = [
        {
            "layer": layer,
            "check": check,
            "quantity": quantity,
            "max_abs_diff": acc[0],
            "max_abs": acc[1],
            "n_graphs": acc[2],
            "backend": backends[layer],
        }
        for (layer, check, quantity), acc in sorted(found.items(), key=lambda kv: (kv[0][0], CHECKS.index(kv[0][1:])))
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- readouts


def _queries(scored: np.ndarray, pairs: pd.DataFrame, n_random: int, n_edge: int, rng) -> np.ndarray:
    """Up to ``n_edge`` planted receivers per range class, plus ``n_random`` other scored cells, sorted."""
    chosen: set[int] = set()
    for _, group in pairs.groupby("range_class", sort=True):
        receivers = np.unique(group["receiver"].to_numpy())
        receivers = receivers[scored[receivers]]
        if len(receivers):
            chosen.update(rng.choice(receivers, size=min(n_edge, len(receivers)), replace=False).tolist())
    others = np.setdiff1d(np.flatnonzero(scored), np.fromiter(chosen, dtype=np.int64, count=len(chosen)))
    if len(others) and n_random > 0:
        chosen.update(rng.choice(others, size=min(n_random, len(others)), replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def _rollout(maps: torch.Tensor) -> torch.Tensor:
    """``[T, T]``: ``Π_l (Ā_l + I) / 2`` of ``[layers, heads, T, T]`` maps, last layer leftmost."""
    eye = torch.eye(maps.shape[-1], dtype=torch.float64, device=maps.device)
    flow = eye
    for layer in range(maps.shape[0]):
        flow = (0.5 * maps[layer].mean(0).double() + 0.5 * eye) @ flow
    return flow


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def attention_readouts(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    spatial_unit_um: float,
    edges: pd.DataFrame | None = None,
    tolerance_um: float = 10.0,
    queries: int = 8,
    edge_queries: int = 8,
    gene_means=None,
    radii_um: Sequence[float] = (),
    seed: int = 0,
    max_graphs: int = 0,
) -> dict[str, pd.DataFrame]:
    """The readouts of :data:`READOUTS` for sampled queries of a masked pass, against planted pairs and each other.

    The graphs keep the masks they carry. Queries are cells with a scored entry (hidden under the masks):
    up to ``edge_queries`` planted receivers per range class and ``queries`` others per graph, drawn from a
    generator seeded by ``seed``. A query's scores cover the other cells the pass kept; comparison cells are
    those not hidden whole (a wholly hidden cell's token holds none of its values), as are the senders.

    Parameters
    ----------
    edges
        Planted ``sender``, ``receiver``, ``range_class`` pairs by position in the AnnData the graphs were
        built from (``_spatial_weights.graph_edges``); None leaves out the ``edges`` table.
    gene_means
        ``[G]``: the training gene means, the point each query's target measures its prediction from (None: 0).
    radii_um
        Each readout's share of its score mass within these distances of the query, next to its span.

    Returns
    -------
    dict
        ``"edges"``: ``readout, range_class, matched_rank, n_pairs`` (range class ``any``: every pair);
        ``"agreement"``: ``readout_a, readout_b, spearman, n_queries`` -- the mean rank correlation of two
        readouts' scores over a query's comparison cells; ``"span"``: ``readout, metric, value, n_queries`` --
        the score-weighted distance to the query (``span_um``) and the share within each radius.
    """
    from interscale.evaluation.attention_flow import _model_head

    transformer = getattr(module, "global_module", module)
    if not hasattr(transformer, "recording_attention"):
        warnings.warn(
            "attention_readouts: this module has no transformer; nothing to read.", RuntimeWarning, stacklevel=2
        )
        return {}
    _refuse_pooled(transformer, "attention_readouts")
    layers = list(transformer.transformer_encoder.layers)
    if not all(_checkable(layer.self_attn) for layer in layers):
        warnings.warn(
            "attention_readouts: the gradient readouts need dense maps in the autograd graph; this backend "
            f"({type(layers[0].self_attn).__name__}) does not form them. Nothing read.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    graphs = list(data_list)
    if max_graphs > 0:
        graphs = graphs[:max_graphs]
    unit = float(spatial_unit_um)
    rng = np.random.default_rng(int(seed))
    ranks: dict[tuple[str, str], list[float]] = defaultdict(list)
    agree: dict[tuple[str, str], list[float]] = defaultdict(list)
    spans: dict[tuple[str, str], list[float]] = defaultdict(list)
    radii = [float(r) for r in radii_um]

    was_training = module.training
    module.eval()
    loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
    try:
        with _rng_untouched(device):
            for batch in loader:
                batch = batch.to(device)
                leaf: dict[str, torch.Tensor] = {}

                def tokens_in(norm, args):
                    # The first call normalises the flat [N, E] cell tokens (common_step_local_to_global); the
                    # second, inside forward, the padded sequence. The tokens become a leaf to take a gradient at.
                    if args[0].dim() != 2 or "x" in leaf:
                        return None
                    leaf["x"] = args[0].detach().requires_grad_(True)
                    return (leaf["x"],)

                handle = transformer.norm_input.register_forward_pre_hook(tokens_in)
                try:
                    with transformer.hooking_attention(), transformer.recording_attention() as record:
                        out = module._common_step(batch, prediction_task, prediction_level)
                finally:
                    handle.remove()
                _read_graph(
                    module, transformer, layers, batch, out, record[0], leaf["x"], _model_head(module, out, "global"),
                    gene_means, edges, rng, queries, edge_queries, unit, tolerance_um, radii, ranks, agree, spans,
                )
                module.zero_grad(set_to_none=True)
                del out, record, leaf
    finally:
        module.zero_grad(set_to_none=True)
        module.train(was_training)

    edge_rows = [
        {"readout": r, "range_class": rc, "matched_rank": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
         "n_pairs": int(np.isfinite(v).sum())}
        for (r, rc), v in ((k, np.asarray(v, dtype=float)) for k, v in sorted(ranks.items(), key=_readout_order))
    ]
    agree_rows = [
        {"readout_a": a, "readout_b": b, "spearman": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
         "n_queries": int(np.isfinite(v).sum())}
        for (a, b), v in ((k, np.asarray(v, dtype=float)) for k, v in sorted(agree.items(), key=_pair_order))
    ]
    span_rows = [
        {"readout": r, "metric": m, "value": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
         "n_queries": int(np.isfinite(v).sum())}
        for (r, m), v in ((k, np.asarray(v, dtype=float)) for k, v in sorted(spans.items(), key=_readout_order))
    ]
    tables = {"agreement": pd.DataFrame(agree_rows), "span": pd.DataFrame(span_rows)}
    if edges is not None:
        tables = {"edges": pd.DataFrame(edge_rows), **tables}
    return tables


def _readout_order(item):
    (readout, second), _ = item
    return READOUTS.index(readout), str(second) != "any", str(second)


def _pair_order(item):
    (a, b), _ = item
    return READOUTS.index(a), READOUTS.index(b)


def _read_graph(
    module, transformer, layers, batch, out, entry, tokens, prediction, gene_means, edges, rng,
    n_queries, n_edge, unit, tolerance_um, radii, ranks, agree, spans,
) -> None:
    """One graph's queries into the running ``ranks``, ``agree`` and ``spans``."""
    from interscale.evaluation.masked_probing import _model_heads

    dev = prediction.device
    padding = entry["src_padding_mask"][0].bool().to(dev)
    spots = (~padding).nonzero().squeeze(1)  # each kept cell's token position, in index_nodes order
    nodes = torch.as_tensor(entry["index_nodes"][0], dtype=torch.long, device=dev)
    n_kept = len(nodes)
    _, _, entry_mask = _model_heads(module, out)
    scored_rows = entry_mask.bool().any(dim=1).cpu().numpy()  # per row = per kept cell, nodes' order
    hidden = hidden_cells(module, batch)[nodes].cpu().numpy()
    pos = batch.pos.double().to(dev)[nodes] * unit
    dist = torch.cdist(pos, pos)  # [kept, kept], rows and columns in nodes' order

    pairs = pd.DataFrame(columns=["sender", "receiver", "range_class"])
    if edges is not None:
        found = graph_edges(edges, batch.obs_names)
        row_of = np.full(batch.num_nodes, -1, dtype=np.int64)
        row_of[nodes.cpu().numpy()] = np.arange(n_kept)
        sender, receiver = row_of[found["sender"].to_numpy()], row_of[found["receiver"].to_numpy()]
        keep = (sender >= 0) & (receiver >= 0) & (sender != receiver)
        keep[keep] &= ~hidden[sender[keep]]
        pairs = pd.DataFrame(
            {"sender": sender[keep], "receiver": receiver[keep], "range_class": found["range_class"].to_numpy()[keep]}
        )
    chosen = _queries(scored_rows, pairs, n_queries, n_edge, rng)
    if not len(chosen):
        return

    # Readouts from the maps alone.
    maps = entry["maps"][:, :, :, :].to(dev)  # [layers, heads, T, T] (one graph)
    q_tok = spots[torch.as_tensor(chosen, device=dev)]
    scores = {
        "attn_last": maps[-1].mean(0)[q_tok][:, spots].double(),
        "attn_mean": maps.mean((0, 1))[q_tok][:, spots].double(),
        "rollout": _rollout(maps)[q_tok][:, spots],
    }
    relevance = transformer.self_attn_relevance
    embedding = out.view.global_embedding
    scores["relevance_sum"] = relevance.generate_relevance(embedding).detach()[q_tok][:, spots].double()

    # Readouts from each query's own target.
    if gene_means is None:
        means = torch.zeros(prediction.shape[1], device=dev)
    else:
        means = torch.as_tensor(gene_means, device=dev)
    rows = {name: [] for name in ("grad_attn_last", "relevance", "grad_x_token")}
    centred = tokens.detach() - tokens.detach().mean(0)
    for k, row in enumerate(chosen):
        diff = (prediction[row] - means.to(prediction.dtype)) * entry_mask[row].to(prediction.dtype)
        target = 0.5 * (diff * diff).sum()
        tokens.grad = None
        full = relevance.generate_relevance(embedding, target=target)
        rows["relevance"].append(full.detach()[q_tok[k]][spots].double())
        last = layers[-1]
        cam = (last.get_attn_gradients() * last.get_attn_output_weights().detach()).clamp(min=0).mean(0)
        rows["grad_attn_last"].append(cam[q_tok[k]][spots].double())
        grad = tokens.grad if tokens.grad is not None else torch.zeros_like(tokens)
        rows["grad_x_token"].append((centred * grad).sum(1).abs()[nodes].double())
    scores.update({name: torch.stack(values) for name, values in rows.items()})

    # Comparison cells: every kept cell not hidden whole, but the query.
    chosen_t = torch.as_tensor(chosen, device=dev)
    compare = torch.as_tensor(~hidden, device=dev).expand(len(chosen), -1).clone()
    compare[torch.arange(len(chosen), device=dev), chosen_t] = False
    d_q = dist[chosen_t]

    if len(pairs):
        query_of = {int(r): k for k, r in enumerate(chosen)}
        mine = pairs[pairs["receiver"].isin(query_of)]
        if len(mine):
            receivers = mine["receiver"].to_numpy()
            rows_t = torch.as_tensor([query_of[int(r)] for r in receivers], device=dev)
            senders_t = torch.as_tensor(mine["sender"].to_numpy(), device=dev)
            exclude = torch.zeros(len(mine), n_kept, dtype=torch.bool, device=dev)
            for p, r in enumerate(receivers):  # the receiver's other senders are no comparison
                exclude[p, torch.as_tensor(pairs.loc[pairs["receiver"] == r, "sender"].to_numpy(), device=dev)] = True
            classes = mine["range_class"].astype(str).to_numpy()
            for name in READOUTS:
                rank = matched_ranks(scores[name], d_q, compare, rows_t, senders_t, exclude, tolerance_um)
                rank = rank.cpu().numpy()
                ranks[(name, "any")].extend(rank.tolist())
                for rc in np.unique(classes):
                    ranks[(name, rc)].extend(rank[classes == rc].tolist())

    for k in range(len(chosen)):
        keep = compare[k]
        d = d_q[k][keep].cpu().numpy()
        values = {name: scores[name][k][keep].abs().cpu().numpy() for name in READOUTS}
        for i, a in enumerate(READOUTS):
            for b in READOUTS[i + 1 :]:
                agree[(a, b)].append(_spearman(values[a], values[b]))
            mass = values[a].sum()
            if not np.isfinite(mass) or mass <= 0:
                spans[(a, "span_um")].append(np.nan)
                continue
            spans[(a, "span_um")].append(float((values[a] * d).sum() / mass))
            for r in radii:
                spans[(a, f"within_{r:g}um")].append(float(values[a][d <= r].sum() / mass))
