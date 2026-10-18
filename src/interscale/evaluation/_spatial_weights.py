"""Per-query weights over a graph's cells -- attention rows or attributions -- against uniform weights.

Shared by :func:`~interscale.evaluation.attention_diagnostics.attention_by_query` (the raw attention
of a masked pass) and :func:`~interscale.evaluation.receptive_field.receptive_field` (gradient
attributions), so both read the same numbers against the same references. The functions take one
graph's rows: ``weights`` ``[Q, N]``, non-negative; ``dist`` ``[Q, N]``, µm from each query to every
cell; and ``keep`` ``[Q, N]``, the cells each row is read over -- the keys a query can attend to, or
the visible cells. A row without weight on its kept cells is undefined (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

#: The query group of a cell with any visible entry; a wholly hidden cell's group is its depth bin.
VISIBLE = "visible"


def radius_metric(radius_um: float) -> str:
    """``300`` -> ``"beyond_300um"``, ``37.5`` -> ``"beyond_37p5um"``: no dot, which wandb reads as nesting."""
    return "beyond_" + np.format_float_positional(float(radius_um), trim="-").replace(".", "p") + "um"


def bin_labels(bins) -> list[str]:
    """The labels of the bins ``(e0, e1], (e1, e2], ...``, in order, as ``score_heads`` writes them."""
    return [str(interval) for interval in pd.IntervalIndex.from_breaks([float(b) for b in bins], closed="right")]


def query_groups(depth_um: np.ndarray, hidden: np.ndarray, depth_bins_um) -> np.ndarray:
    """Each cell's group: :data:`VISIBLE`, or the depth bin of a wholly hidden cell.

    ``hidden`` ``[N]`` marks the wholly hidden cells and ``depth_um`` ``[N]`` holds their distance to
    the nearest cell with a visible entry. A hidden cell outside every bin gets ``""``, read in no group.
    """
    depth = np.asarray(depth_um, dtype=float)
    hidden = np.asarray(hidden, dtype=bool)
    groups = np.full(len(depth), VISIBLE, dtype=object)
    cut = pd.cut(pd.Series(depth), bins=[float(b) for b in depth_bins_um], right=True)
    labels = np.where(cut.notna().to_numpy(), cut.astype(str).to_numpy(), "")
    groups[hidden] = labels[hidden]
    return groups


def hidden_cells(module, batch) -> torch.Tensor:
    """``[N]`` the cells ``module`` hides wholly in ``batch``: by ``gene_mask`` under gene masking, else ``mask``."""
    if not getattr(module, "masked_nodes", True) or getattr(batch, "mask", None) is None:
        return torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
    gene_mask = getattr(batch, "gene_mask", None) if getattr(module, "mask_strategy", "node") == "gene" else None
    return batch.mask.bool() if gene_mask is None else gene_mask.bool().all(dim=1)


def add_finite(sums: dict, key, values: np.ndarray, pick: np.ndarray) -> None:
    """Add the finite ``values[pick]`` to the running ``[sum, count]`` at ``sums[key]``."""
    pick = pick & np.isfinite(values)
    if pick.any():
        acc = sums.setdefault(key, [0.0, 0])
        acc[0] += float(values[pick].sum())
        acc[1] += int(pick.sum())


def row_statistics(
    weights: torch.Tensor, dist: torch.Tensor, keep: torch.Tensor, radii_um=()
) -> dict[str, torch.Tensor]:
    """Per row, over its kept cells (NaN for a row without weight on them).

    * ``span_um`` -- the weighted mean distance. ``span_rel`` -- that over the mean distance to the
      kept cells, the span of uniform weights: 1 is uniform, below 1 nearer than uniform.
    * ``effective_rel`` -- ``exp`` of the entropy of the row's shares, over the number of kept cells:
      1 is uniform, ``1/n`` all on one cell.
    * ``beyond_<r>um`` -- the share of the weight farther than ``r`` µm.
    """
    d = dist.double()
    w = torch.where(keep, weights.double(), torch.zeros_like(d))
    mass = w.sum(1)
    defined = mass > 0
    share = w / mass.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
    n_keep = keep.sum(1).double().clamp_min(1.0)
    uniform_span = torch.where(keep, d, torch.zeros_like(d)).sum(1) / n_keep
    span = (share * d).sum(1)
    nan = torch.full_like(span, float("nan"))
    stats = {
        "span_um": span,
        "span_rel": torch.where(uniform_span > 0, span / uniform_span.clamp_min(1e-12), nan),
        "effective_rel": torch.special.entr(share).sum(1).exp() / n_keep,
    }
    for radius in radii_um:
        stats[radius_metric(radius)] = (share * (d > float(radius))).sum(1)
    return {name: torch.where(defined, value, nan) for name, value in stats.items()}


def distance_profile(
    weights: torch.Tensor, dist: torch.Tensor, keep: torch.Tensor, bins_um
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row and distance bin: the share of the row's weight there, and the share of its kept cells.

    The second is what uniform weights would put in the bin. Both ``[Q, B]`` for the bins
    ``(e0, e1], ...`` of ``bins_um``; a row without weight on its kept cells is NaN. The lift of a
    group of rows is the ratio of the two, each summed over the rows: 1 is uniform.
    """
    d = dist.double()
    w = torch.where(keep, weights.double(), torch.zeros_like(d))
    mass = w.sum(1, keepdim=True)
    share = w / mass.clamp_min(torch.finfo(torch.float64).tiny)
    present = keep.double() / keep.sum(1, keepdim=True).double().clamp_min(1.0)
    edges = [float(b) for b in bins_um]
    weight_in, uniform_in = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (d > low) & (d <= high)
        weight_in.append((share * inside).sum(1))
        uniform_in.append((present * inside).sum(1))
    weight_in, uniform_in = torch.stack(weight_in, dim=1), torch.stack(uniform_in, dim=1)
    undefined = (mass.squeeze(1) <= 0)[:, None]
    nan = torch.full_like(weight_in, float("nan"))
    return torch.where(undefined, nan, weight_in), torch.where(undefined, nan, uniform_in)


def graph_edges(edges: pd.DataFrame, obs_names, *, offset: int = 1) -> pd.DataFrame:
    """The planted ``edges`` inside one graph, as node indices of that graph.

    ``edges`` names cells by position in the AnnData the graphs were built from (``sender``,
    ``receiver``, ``range_class``, and any other column, carried along); ``prepare_geome_dataset``
    names each graph's nodes by that position plus ``offset``, and the graphs carry those names as
    ``obs_names``. Pairs leaving the graph go.
    """
    names = obs_names.detach().cpu().numpy() if isinstance(obs_names, torch.Tensor) else np.asarray(obs_names)
    position = np.rint(names.astype(np.float64)).astype(np.int64) - int(offset)
    node = pd.Series(np.arange(len(position)), index=position)
    sender = edges["sender"].to_numpy().astype(np.int64)
    receiver = edges["receiver"].to_numpy().astype(np.int64)
    inside = np.isin(sender, position) & np.isin(receiver, position)
    out = {
        "sender": node.loc[sender[inside]].to_numpy(),
        "receiver": node.loc[receiver[inside]].to_numpy(),
        "range_class": edges["range_class"].astype(str).to_numpy()[inside],
    }
    for column in edges.columns:
        if column not in out:
            out[column] = edges[column].to_numpy()[inside]
    return pd.DataFrame(out)


#: Planted pairs ranked at once in :func:`matched_ranks`: a few ``[pairs, N]`` temporaries per chunk.
RANK_CHUNK = 512


def matched_ranks(
    weights: torch.Tensor,
    dist: torch.Tensor,
    keep: torch.Tensor,
    rows: torch.Tensor,
    senders: torch.Tensor,
    exclude: torch.Tensor,
    tolerance_um: float,
) -> torch.Tensor:
    """Each planted pair's sender ranked among the kept cells about as far from the receiver.

    ``rows`` ``[P]`` picks each pair's receiver row and ``senders`` ``[P]`` its sender column;
    ``exclude`` ``[P, N]`` marks cells that are no comparison (the receiver's other senders). The
    comparisons are kept, not excluded, not the sender, and within ``tolerance_um`` of the sender's
    distance to the receiver. Rank ``(below + ties / 2) / comparisons``: 0.5 is what weights blind to
    the edge but not to distance score -- a distance kernel, say -- and 1 the top. NaN where the
    sender is not kept or nothing compares.
    """
    out = []
    for start in range(0, len(rows), RANK_CHUNK):
        r, s = rows[start : start + RANK_CHUNK], senders[start : start + RANK_CHUNK]
        w, d = weights[r].double(), dist[r].double()
        w_s, d_s = w.gather(1, s[:, None]), d.gather(1, s[:, None])
        compare = keep[r] & ~exclude[start : start + RANK_CHUNK] & ((d - d_s).abs() <= float(tolerance_um))
        compare[torch.arange(len(r), device=compare.device), s] = False
        n = compare.sum(1)
        below = ((w < w_s) & compare).sum(1)
        ties = ((w == w_s) & compare).sum(1)
        rank = (below.double() + 0.5 * ties.double()) / n.clamp_min(1).double()
        valid = keep[r].gather(1, s[:, None]).squeeze(1) & (n > 0) & torch.isfinite(w_s.squeeze(1))
        out.append(torch.where(valid, rank, torch.full_like(rank, float("nan"))))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.float64, device=weights.device)


def pair_exclusions(pairs: pd.DataFrame, n_cells: int, device) -> torch.Tensor:
    """``[P, N]``: for each pair, every sender of its receiver -- the cells :func:`matched_ranks` skips."""
    exclude = torch.zeros(len(pairs), n_cells, dtype=torch.bool, device=device)
    senders = pairs.groupby("receiver")["sender"].apply(lambda s: np.unique(s.to_numpy())).to_dict()
    for p, receiver in enumerate(pairs["receiver"].to_numpy()):
        exclude[p, torch.as_tensor(senders[receiver], device=device)] = True
    return exclude
