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
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation.online_probes import PROBE_PREFIX, _rng_untouched
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


def radius_metric(radius_um: float) -> str:
    """``300`` -> ``"beyond_300um"``, ``37.5`` -> ``"beyond_37p5um"``: no dot, which wandb reads as nesting."""
    return "beyond_" + np.format_float_positional(float(radius_um), trim="-").replace(".", "p") + "um"


def head_statistics(
    maps: torch.Tensor,
    padding: torch.Tensor,
    *,
    n_heads: int,
    pos_um: torch.Tensor | None = None,
    radii_um: Sequence[float] = (),
    reference: torch.Tensor | None = None,
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
                on_cells = queries[:, cells]  # [n, n]
                mass = on_cells.sum(-1)
                kept = mass > MIN_CELL_MASS
                share = on_cells / mass.clamp_min(MIN_CELL_MASS)[:, None]
                stats = {
                    "cls_mass": queries[:, seq_len],
                    "effective_keys": torch.special.entr(share).sum(-1).exp(),
                }
                if dist is not None:
                    stats["span_um"] = (share * dist).sum(-1)
                    for radius in radii_um:
                        stats[radius_metric(radius)] = (share * (dist > float(radius))).sum(-1)
                if reference is not None:
                    exact = reference[layer, row][cells].double()
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
                if getattr(batch, "mask", None) is not None:
                    batch.mask = torch.zeros_like(batch.mask)
                if getattr(batch, "gene_mask", None) is not None:
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
