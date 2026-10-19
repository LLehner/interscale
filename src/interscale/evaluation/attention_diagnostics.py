"""Attention diagnostics: how far each head reaches, how much it sinks into CLS, how peaked it is.

Stage 1 of ``.claude/attention_plan.md``. Per layer and head, read from the explicit maps of a
clean pass (nothing masked, dropout off) over a fixed set of graphs:

* **span** -- ``sum_j A_ij |p_i - p_j| / sum_j A_ij`` over the cell keys, in µm: how far a cell's
  attention reaches, LRA's attention span in physical units. Mean over cells, and quartiles.
* **mass beyond a radius** ``r`` -- the share of a cell's attention on cells farther than ``r``.
* **CLS mass** -- ``A_i,CLS``, the attention sink.
* **effective keys** -- ``exp`` of the entropy of the row over the cell keys: 1 for a head that
  looks at one cell, ``n`` for one spread evenly over ``n``.
* **TV to softmax**, for a backend that estimates softmax (the Performer) -- ``1/2 sum_j |Â_ij - A_ij|``
  between its implied map and the exact softmax on the same queries, keys and biases.
* **token cosine**, per layer -- the mean cosine between two cell tokens entering attention,
  before any centring. Near 1 the tokens are nearly identical, and attention computed from their
  content cannot tell keys apart (see ``attention.center_qk``).
* **against uniform** -- ``span_rel``, the span over the mean distance to the keys the query can
  attend to (what uniform attention reaches; 2.6 mm on a 5 mm slide), and ``effective_keys_rel``,
  effective keys over the number of those keys: 1 for uniform attention.

:func:`attention_by_query` reads a **masked** pass instead -- the masks the graphs carry, as in
training -- by how deep each query lies in the hidden area, with visible-versus-hidden key mass,
distance profiles and planted edges against distance-matched cells. That is where a global
component has to work; :func:`attention_diagnostics` can take the masked pass too (``masked``).

Every number is over **cell queries**: CLS and padding rows are left out. Padding is never a key.
Span, radii and effective keys renormalise each row over its cell keys -- CLS is the sink and is
read on its own -- and skip a row whose cell mass is below :data:`MIN_CELL_MASS`.

Logged by the probe callback when ``probe.attention_diagnostics`` is set, as
``probe/attn_<metric>_l<layer>_h<head>``, with ``..._l<layer>_mean`` (over that layer's heads) and
``probe/attn_<metric>_mean`` (over every layer and head) beside them; the callback's charts then
put a layer's heads on one plot. Callable post hoc on any trained module. Dataset-agnostic: the
coordinates are the graphs' ``pos``, scaled by ``dataset.spatial_unit_um``, and the radii come
from config.

The pass keeps every layer's ``S x S`` maps, so it reads a few graphs, not a split. It consumes
no RNG (:func:`~interscale.evaluation.online_probes._rng_untouched`) and changes nothing the model
computes (``TransformerNodeEncoderHook.recording_attention``).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation._spatial_weights import (
    VISIBLE,
    add_finite,
    bin_labels,
    distance_profile,
    graph_edges,
    hidden_cells,
    matched_ranks,
    pair_exclusions,
    query_groups,
    radius_metric,
    row_statistics,
)
from interscale.evaluation.online_probes import PROBE_PREFIX, _rng_untouched, hidden_depth
from interscale.tl.padding import pad_like

#: A row whose attention on cells sums to less than this is all sink: its span, radii and effective
#: keys are undefined (renormalising it would amplify rounding), so it is skipped for those.
MIN_CELL_MASS = 1e-6
#: Percentiles of the per-cell span logged beside its mean.
SPAN_QUANTILES = (25, 50, 75)
#: Read on every cell query, whatever its cell mass; every other statistic skips the all-sink rows.
_EVERY_ROW = ("cls_mass", "tv_softmax")


def attention_metric_name(metric: str, layer: int | None = None, head: int | None = None) -> str:
    """The scalar name of a diagnostic: per head, the mean over a layer's heads, or the mean over all."""
    if layer is None:
        return f"{PROBE_PREFIX}/attn_{metric}_mean"
    if head is None:
        return f"{PROBE_PREFIX}/attn_{metric}_l{layer}_mean"
    return f"{PROBE_PREFIX}/attn_{metric}_l{layer}_h{head}"


def head_statistics(
    maps: torch.Tensor,
    padding: torch.Tensor,
    *,
    n_heads: int,
    pos_um: torch.Tensor | None = None,
    radii_um: Sequence[float] = (),
    reference: torch.Tensor | None = None,
    allowed: torch.Tensor | None = None,
) -> dict[tuple[str, int, int], list[np.ndarray]]:
    """Per-cell statistics of one pass's maps: ``{(metric, layer, head): [values per cell query]}``.

    Parameters
    ----------
    maps
        ``[layers, B * heads, S + 1, S + 1]``: every layer's attention, rows graph-major (row
        ``b * heads + h`` is graph ``b``, head ``h``), tokens left-padded, CLS last.
    padding
        ``[B, S]``, True at padding.
    n_heads
        Heads per layer.
    pos_um
        ``[B, S, 2]`` coordinates in µm, laid out like the tokens
        (:func:`~interscale.tl.padding.pad_like`); None skips span and radii.
    radii_um
        Radii for the mass beyond them.
    reference
        Like ``maps``: the exact attention a backend estimates, for the TV distance; or None.
    allowed
        ``[B, S + 1, S + 1]``, True where a query may attend to a key (the recorded mask's
        ``open``). Without it a key is open where the exact map's weight is above 0, which a sparse
        backend (entmax, top-k) breaks: it puts exact zeros on open keys.
    """
    n_layers, n_rows, length, _ = maps.shape
    n_graphs, seq_len = padding.shape
    if n_rows != n_graphs * n_heads or length != seq_len + 1:
        raise ValueError(
            f"maps of shape {tuple(maps.shape)} do not fit {n_graphs} graphs x {n_heads} heads of "
            f"{seq_len} tokens plus CLS."
        )
    if reference is not None and reference.shape != maps.shape:
        raise ValueError(f"reference of shape {tuple(reference.shape)} does not match maps {tuple(maps.shape)}.")
    maps = maps.detach()
    reference = None if reference is None else reference.detach()

    out: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    for b in range(n_graphs):
        cells = (~padding[b].bool()).nonzero().squeeze(1)
        if cells.numel() == 0:
            continue
        dist = None
        if pos_um is not None:
            where = pos_um[b, cells].double()
            dist = torch.cdist(where, where, compute_mode="donot_use_mm_for_euclid_dist")  # [n, n]
        # One head at a time: a few n x n float64 temporaries, whatever the number of heads.
        for layer in range(n_layers):
            for head in range(n_heads):
                row = b * n_heads + head
                queries = maps[layer, row][cells].double()  # [n, S + 1]: cell queries, every key
                exact = None if reference is None else reference[layer, row][cells].double()
                on_cells = queries[:, cells]  # [n, n]
                mass = on_cells.sum(-1)
                kept = mass > MIN_CELL_MASS
                share = on_cells / mass.clamp_min(MIN_CELL_MASS)[:, None]
                # The keys a query can attend to: the mask's, else a weight above 0 in the exact map.
                if allowed is not None:
                    open_keys = allowed[b][cells][:, cells].to(queries.device)
                else:
                    open_keys = (queries if exact is None else exact)[:, cells] > 0
                n_open = open_keys.sum(-1).double().clamp_min(1.0)
                stats = {
                    "cls_mass": queries[:, seq_len],
                    "effective_keys": torch.special.entr(share).sum(-1).exp(),
                }
                stats["effective_keys_rel"] = stats["effective_keys"] / n_open
                if dist is not None:
                    stats["span_um"] = (share * dist).sum(-1)
                    uniform = (dist * open_keys).sum(-1) / n_open
                    stats["span_rel"] = stats["span_um"] / uniform.clamp_min(1e-12)
                    for radius in radii_um:
                        stats[radius_metric(radius)] = (share * (dist > float(radius))).sum(-1)
                if exact is not None:
                    stats["tv_softmax"] = 0.5 * (queries - exact).abs().sum(-1)
                for metric, values in stats.items():
                    picked = values if metric in _EVERY_ROW else values[kept]
                    out[(metric, layer, head)].append(picked.cpu().numpy())
    return out


def summarize(per_cell: dict[tuple[str, int, int], list[np.ndarray]]) -> dict[str, float]:
    """Scalars from :func:`head_statistics`: per head, then means over heads and over everything.

    A head's value is the mean over every cell query collected (the span also gets its
    quartiles); a layer's ``_mean`` is the mean of its heads' values, and the overall ``_mean`` that
    of every layer's heads. A head without a single qualifying row logs nothing for that metric.
    """
    results: dict[str, float] = {}
    per_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (metric, layer, head), chunks in sorted(per_cell.items()):
        values = np.concatenate(chunks) if chunks else np.empty(0)
        if values.size == 0:
            continue
        named = {metric: float(values.mean())}
        if metric == "span_um":
            for q in SPAN_QUANTILES:
                named[f"span_um_q{q}"] = float(np.percentile(values, q))
        for name, value in named.items():
            results[attention_metric_name(name, layer, head)] = value
            per_layer[(name, layer)].append(value)
    overall: dict[str, list[float]] = defaultdict(list)
    for (name, layer), values in per_layer.items():
        results[attention_metric_name(name, layer)] = float(np.mean(values))
        overall[name].extend(values)
    for name, values in overall.items():
        results[attention_metric_name(name)] = float(np.mean(values))
    return results


@torch.no_grad()
def _refuse_pooled(transformer, what: str) -> None:
    """A pooled model's attention is between patches; these readouts index it by cell."""
    if getattr(transformer, "pooling", None) is not None:
        raise ValueError(
            f"{what} reads one attention row per cell, but a pooled model (pooling.kind {transformer.pooling.kind!r}) "
            "attends between patches. Its cell-level readouts are the head scores, the probes and the receptive field."
        )


def attention_diagnostics(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    spatial_unit_um: float = 1.0,
    radii_um: Sequence[float] = (),
    max_graphs: int = 4,
    masked: bool = False,
) -> dict[str, float]:
    """Run ``module`` over the first ``max_graphs`` graphs of ``data_list`` and diagnose its attention.

    Parameters
    ----------
    module
        A module with a transformer: a ``GlobalModule`` or a combined module (its
        ``global_module``). A module without one gets a warning and an empty result.
    data_list
        PyG ``Data`` objects, e.g. ``datamodule.val_data``; read in order, one graph per pass.
    prediction_task, prediction_level
        Passed straight to ``module._common_step``.
    device
        Device to move each graph to.
    spatial_unit_um
        ``dataset.spatial_unit_um``: µm per unit of the graphs' ``pos``.
    radii_um
        Radii (µm) for the mass beyond them.
    max_graphs
        How many graphs to read; 0 or less reads all of them.
    masked
        False blanks every mask first: a clean pass, every cell attending with its own expression.
        True keeps the masks the graphs carry, so the module corrupts its input as in training
        (:func:`attention_by_query` reads that pass by query depth).

    Returns
    -------
    dict
        ``{name: value}``, names from :func:`attention_metric_name`. Span and radii are missing
        (with a warning) when the graphs carry no ``pos``; the TV distance is there only for a
        backend that estimates softmax.
    """
    transformer = getattr(module, "global_module", module)
    if not hasattr(transformer, "recording_attention"):
        warnings.warn(
            "attention diagnostics: this module has no transformer; nothing to read.", RuntimeWarning, stacklevel=2
        )
        return {}
    _refuse_pooled(transformer, "attention diagnostics")
    graphs = list(data_list)
    if max_graphs > 0:
        graphs = graphs[:max_graphs]

    per_cell: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    token_cosine: dict[int, list[float]] = defaultdict(list)
    without_pos = False
    was_training = module.training
    module.eval()
    loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
    try:
        with _rng_untouched(device), transformer.recording_attention(reference=True) as record:
            for batch in loader:
                batch = batch.to(device)
                # Clean: nothing blanked, so attention is read on the real expression. New tensors,
                # never written in place -- they may share storage with the split's Data objects.
                if not masked and getattr(batch, "mask", None) is not None:
                    batch.mask = torch.zeros_like(batch.mask)
                if not masked and getattr(batch, "gene_mask", None) is not None:
                    batch.gene_mask = torch.zeros_like(batch.gene_mask)
                module._common_step(batch, prediction_task, prediction_level)
                passes = list(record)
                record.clear()
                pos = getattr(batch, "pos", None)
                without_pos |= pos is None
                for entry in passes:
                    if entry.get("maps") is None:
                        continue
                    padding = entry["src_padding_mask"]
                    pos_um = None
                    if pos is not None:
                        pos_um = pad_like(
                            pos.to(device=padding.device, dtype=torch.float64) * float(spatial_unit_um),
                            batch.batch,
                            entry["index_nodes"],
                            padding.shape[1],
                        )
                    found = head_statistics(
                        entry["maps"],
                        padding,
                        n_heads=int(transformer.n_heads),
                        pos_um=pos_um,
                        radii_um=radii_um,
                        reference=entry.get("reference"),
                        allowed=entry.get("open"),
                    )
                    for key, values in found.items():
                        per_cell[key].extend(values)
                    if entry.get("token_cosine") is not None:
                        for layer, values in enumerate(entry["token_cosine"].cpu().tolist()):
                            token_cosine[layer].extend(values)
    finally:
        module.train(was_training)

    if without_pos:
        warnings.warn(
            "attention diagnostics: the graphs carry no `pos`, so span and the radii are skipped. "
            "Set dataset.spatial_key so the coordinates are attached.",
            RuntimeWarning,
            stacklevel=2,
        )
    results = summarize(per_cell)
    for layer, values in token_cosine.items():
        results[attention_metric_name("token_cosine", layer)] = float(np.mean(values))
    if token_cosine:
        results[attention_metric_name("token_cosine")] = float(np.mean([np.mean(v) for v in token_cosine.values()]))
    return results


@torch.no_grad()
def attention_by_query(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    spatial_unit_um: float,
    depth_bins_um: Sequence[float],
    radii_um: Sequence[float] = (),
    distance_bins_um: Sequence[float] = (),
    edges: pd.DataFrame | None = None,
    tolerance_um: float = 10.0,
    max_graphs: int = 0,
) -> dict[str, pd.DataFrame]:
    """The raw attention of a masked pass, by how deep each query lies in the hidden area.

    The graphs keep the masks they carry (e.g. discs from ``masked_probing.draw_masks``) and the
    module corrupts its input with them as in training. A hidden cell then queries with its mask
    token and position in the first layer, and with what it gathered in later ones. A clean pass
    reads a regime training never shows: every cell holds its own expression, a receiver its own
    response, so nothing needs to look anywhere.

    Each cell query falls in a group -- ``"visible"``, or the depth bin (``depth_bins_um``) of a
    wholly hidden cell, its distance to the nearest cell with a visible entry -- and is read over the
    cell keys it can attend to (a weight above 0 in the exact map: the Performer's reference, else the
    map itself), against uniform attention over them:

    * ``span_um``, ``span_rel``, ``effective_rel``, ``beyond_<r>um`` -- span, span over the uniform
      span, effective keys over the number of keys, the share beyond each radius
      (:func:`~interscale.evaluation._spatial_weights.row_statistics`);
    * ``visible_mass`` -- the share of the cell mass on visible cells; ``visible_lift`` -- that over
      the visible cells' share of the keys: 1 is blind to what is hidden, above 1 prefers what can
      be seen;
    * ``cls_mass`` -- the weight on CLS; ``tv_softmax`` -- the Performer's distance to softmax;
    * ``token_cosine`` (query ``"all"``) -- the mean cosine of the cell tokens entering each layer.

    Consumes no RNG and leaves the module as it was, like :func:`attention_diagnostics`.

    Parameters
    ----------
    edges
        Planted ``sender``, ``receiver``, ``range_class`` pairs by position in the AnnData the graphs
        were built from (:func:`~interscale.evaluation._spatial_weights.graph_edges`).
    tolerance_um
        How far a comparison cell's distance to the receiver may differ from the sender's.

    Returns
    -------
    dict
        ``"by_query"``: ``layer, head, query, metric, value, n_queries`` -- per head (``h<k>``) and
        their mean (``mean``), each a mean over the group's queries of every graph. ``"profile"``
        (with ``distance_bins_um``): ``layer, query, distance_um, lift, n_queries`` -- the head-mean
        map's weight per distance bin over what uniform attention puts there (1 is uniform).
        ``"edges"`` (with ``edges``): ``layer, range_class, metric, value, n_pairs`` for the planted
        pairs with a wholly hidden receiver and a visible sender, from the head-mean map --
        ``coverage``, the share whose sender the receiver can attend to, and ``matched_rank``, the
        sender's rank among the visible cells about as far from the receiver
        (:func:`~interscale.evaluation._spatial_weights.matched_ranks`): 0.5 is attention blind to
        the edge, a distance kernel for one; 1 the top.
    """
    transformer = getattr(module, "global_module", module)
    if not hasattr(transformer, "recording_attention"):
        warnings.warn(
            "attention_by_query: this module has no transformer; nothing to read.", RuntimeWarning, stacklevel=2
        )
        return {}
    _refuse_pooled(transformer, "attention_by_query")
    graphs = list(data_list)
    if max_graphs > 0:
        graphs = graphs[:max_graphs]
    groups = [VISIBLE, *bin_labels(depth_bins_um)]
    unit = float(spatial_unit_um)
    sums: dict[tuple, list] = {}
    profile: dict[tuple, np.ndarray] = {}
    edge_sums: dict[tuple, np.ndarray] = {}
    token_cosine: dict[int, list[float]] = defaultdict(list)
    n_bins = max(len(distance_bins_um) - 1, 0)
    n_heads = 0
    was_training = module.training
    module.eval()
    loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
    try:
        with _rng_untouched(device), transformer.recording_attention(reference=True) as record:
            for batch in loader:
                batch = batch.to(device)
                if getattr(batch, "pos", None) is None:
                    raise ValueError("attention_by_query reads distances, but a graph carries no `pos`.")
                module._common_step(batch, prediction_task, prediction_level)
                entry = record[0]
                record.clear()
                if entry.get("maps") is None:
                    continue
                maps, reference = entry["maps"], entry.get("reference")
                dev = maps.device
                # The keys a query may attend to, from the mask when it was recorded: a sparse
                # backend's zeros are not blocked keys.
                allowed = entry.get("open")
                padding = entry["src_padding_mask"][0].bool().to(dev)
                seq_len = padding.shape[0]
                spots = (~padding).nonzero().squeeze(1)  # the cells' token positions, CLS left out
                nodes = torch.as_tensor(entry["index_nodes"][0], dtype=torch.long, device=dev)
                hidden_cell = hidden_cells(module, batch).to(dev)
                depth = hidden_depth(batch.pos.to(dev), hidden_cell, batch.batch.to(dev)) * unit
                hid = hidden_cell[nodes]
                where = batch.pos.to(dev)[nodes].double() * unit
                dist = torch.cdist(where, where, compute_mode="donot_use_mm_for_euclid_dist")
                group = query_groups(depth[nodes].cpu().numpy(), hid.cpu().numpy(), depth_bins_um)
                n_layers, n_heads = maps.shape[0], maps.shape[1]

                for layer in range(n_layers):
                    for head in range(n_heads):
                        rows = maps[layer, head][spots].double()  # [n, S + 1]
                        exact = rows if reference is None else reference[layer, head][spots].double()
                        attn = rows[:, spots]
                        open_keys = allowed[0][spots][:, spots].to(dev) if allowed is not None else exact[:, spots] > 0
                        stats = row_statistics(attn, dist, open_keys, radii_um)
                        cell_mass = torch.where(open_keys, attn, torch.zeros_like(attn)).sum(1)
                        visible_keys = open_keys & ~hid[None, :]
                        visible_mass = torch.where(visible_keys, attn, torch.zeros_like(attn)).sum(1)
                        visible_mass = visible_mass / cell_mass.clamp_min(torch.finfo(torch.float64).tiny)
                        visible_share = visible_keys.sum(1).double() / open_keys.sum(1).double().clamp_min(1.0)
                        nan = torch.full_like(visible_mass, float("nan"))
                        stats["visible_mass"] = torch.where(cell_mass > 0, visible_mass, nan)
                        stats["visible_lift"] = torch.where(
                            (cell_mass > 0) & (visible_share > 0), visible_mass / visible_share.clamp_min(1e-12), nan
                        )
                        stats["cls_mass"] = rows[:, seq_len]
                        if reference is not None:
                            stats["tv_softmax"] = 0.5 * (rows - exact).abs().sum(1)
                        for metric, values in stats.items():
                            values = values.cpu().numpy()
                            for g in groups:
                                add_finite(sums, (layer, head, g, metric), values, group == g)

                    mean_rows = maps[layer].mean(0)[spots].double()
                    mean_exact = mean_rows if reference is None else reference[layer].mean(0)[spots].double()
                    mean_attn = mean_rows[:, spots]
                    mean_open = allowed[0][spots][:, spots].to(dev) if allowed is not None else mean_exact[:, spots] > 0
                    if n_bins:
                        weight_in, uniform_in = distance_profile(mean_attn, dist, mean_open, distance_bins_um)
                        defined = torch.isfinite(weight_in).all(1).cpu().numpy()
                        weight_in, uniform_in = weight_in.cpu().numpy(), uniform_in.cpu().numpy()
                        for g in groups:
                            pick = (group == g) & defined
                            if pick.any():
                                acc = profile.setdefault((layer, g), np.zeros((3, n_bins)))
                                acc += [weight_in[pick].sum(0), uniform_in[pick].sum(0), np.full(n_bins, pick.sum())]
                    if edges is not None:
                        pairs = _token_pairs(edges, batch, nodes, hid)
                        if len(pairs):
                            r = torch.as_tensor(pairs["receiver"].to_numpy(), device=dev)
                            s = torch.as_tensor(pairs["sender"].to_numpy(), device=dev)
                            covered = mean_open[r, s].cpu().numpy()
                            ranks = (
                                matched_ranks(
                                    mean_attn,
                                    dist,
                                    mean_open & ~hid[None, :],
                                    r,
                                    s,
                                    pair_exclusions(pairs, len(nodes), dev),
                                    tolerance_um,
                                )
                                .cpu()
                                .numpy()
                            )
                            classes = pairs["range_class"].to_numpy()
                            for range_class in [*np.unique(classes), "any"]:
                                pick = np.ones(len(classes), bool) if range_class == "any" else classes == range_class
                                ranked = pick & np.isfinite(ranks)
                                acc = edge_sums.setdefault((layer, str(range_class)), np.zeros(4))
                                acc += [pick.sum(), covered[pick].sum(), ranks[ranked].sum(), ranked.sum()]
                if entry.get("token_cosine") is not None:
                    for layer, values in enumerate(entry["token_cosine"].cpu().tolist()):
                        token_cosine[layer].extend(values)
    finally:
        module.train(was_training)

    rows = []
    for (layer, head, g, metric), (total, count) in sorted(
        sums.items(), key=lambda kv: (kv[0][0], kv[0][1], groups.index(kv[0][2]), kv[0][3])
    ):
        rows.append(
            {
                "layer": layer,
                "head": f"h{head}",
                "query": g,
                "metric": metric,
                "value": total / count,
                "n_queries": count,
            }
        )
    by_query = pd.DataFrame(rows, columns=["layer", "head", "query", "metric", "value", "n_queries"])
    if len(by_query):
        means = by_query.groupby(["layer", "query", "metric"], sort=False).agg(
            value=("value", "mean"), n_queries=("n_queries", "max")
        )
        by_query = pd.concat([by_query, means.reset_index().assign(head="mean")], ignore_index=True)
    cosines = [
        {
            "layer": layer,
            "head": "mean",
            "query": "all",
            "metric": "token_cosine",
            "value": float(np.mean(v)),
            "n_queries": len(v),
        }
        for layer, v in sorted(token_cosine.items())
    ]
    by_query = pd.concat([by_query, pd.DataFrame(cosines, columns=by_query.columns)], ignore_index=True)
    out = {"by_query": by_query}
    if n_bins:
        labels = bin_labels(distance_bins_um)
        rows = []
        for (layer, g), (weight, uniform, count) in sorted(
            profile.items(), key=lambda kv: (kv[0][0], groups.index(kv[0][1]))
        ):
            for b, label in enumerate(labels):
                lift = weight[b] / uniform[b] if uniform[b] > 0 else np.nan
                rows.append(
                    {"layer": layer, "query": g, "distance_um": label, "lift": lift, "n_queries": int(count[b])}
                )
        out["profile"] = pd.DataFrame(rows, columns=["layer", "query", "distance_um", "lift", "n_queries"])
    if edges is not None:
        rows = []
        for (layer, range_class), (n, covered, rank_sum, ranked) in sorted(edge_sums.items()):
            rows.append(
                {
                    "layer": layer,
                    "range_class": range_class,
                    "metric": "coverage",
                    "value": covered / n if n else np.nan,
                    "n_pairs": int(n),
                }
            )
            rows.append(
                {
                    "layer": layer,
                    "range_class": range_class,
                    "metric": "matched_rank",
                    "value": rank_sum / ranked if ranked else np.nan,
                    "n_pairs": int(ranked),
                }
            )
        out["edges"] = pd.DataFrame(rows, columns=["layer", "range_class", "metric", "value", "n_pairs"])
    return out


def _token_pairs(edges: pd.DataFrame, batch, nodes: torch.Tensor, hid: torch.Tensor) -> pd.DataFrame:
    """The planted pairs of ``batch``'s graph as token indices, where the receiver is wholly hidden and the sender visible.

    ``nodes`` maps each cell token to its node; ``hid`` marks the tokens whose cells are hidden.
    """
    pairs = graph_edges(edges, batch.obs_names)
    token = np.full(batch.num_nodes, -1, dtype=np.int64)
    token[nodes.cpu().numpy()] = np.arange(len(nodes))
    sender, receiver = token[pairs["sender"].to_numpy()], token[pairs["receiver"].to_numpy()]
    hidden = hid.cpu().numpy()
    keep = (sender >= 0) & (receiver >= 0)
    keep[keep] &= hidden[receiver[keep]] & ~hidden[sender[keep]]
    return pd.DataFrame(
        {"sender": sender[keep], "receiver": receiver[keep], "range_class": pairs["range_class"].to_numpy()[keep]}
    )
