"""Niche patches: the spatially connected pieces of a categorical label (``.claude/lung_plan.md`` L2-L3, ``.claude/masking_plan.md`` M3).

A niche label (or any categorical annotation) is scattered: one niche turns up in several places of a slide. A **patch** is one
connected piece of it -- a connected component of the slide's own neighbour graph (``edge_index``, the edges the local component
reads) once every edge between two cells of different labels is cut -- so two places of one niche are two patches. Three options read
the same patches:

* the hierarchy's top level (``hierarchy.top_level.mode: obs_label``), finer clusters built inside each patch;
* the mask unit (``dataset.mask_block_kind: label``): whole patches hidden in place of discs;
* pooled tokens (``pooling.kind: label``): one token per patch.

Two adjustments, both deterministic, applied in this order:

* ``min_cells``: a patch with fewer cells joins the neighbouring patch it shares the most edges with (ties: the larger, then the lower
  id); one with no edge to another patch joins the one whose centroid is nearest. Smallest first, until none is left or one patch
  remains. A merged patch can hold cells of more than one label: noisy labels make many one-cell patches, and a coarse level or a
  sequence of tokens cannot afford one each.
* ``max_cells``: a patch with more cells is cut by the kdtree (``hierarchy.kd_parts``, splits at the cell-count quantile across the
  longer side) into ``ceil(size / max_cells)`` pieces of about equal size, so one large niche does not become one huge unit.

A cell with no label (an all-zero one-hot row: NaN in ``obs``) has the code -1, which is one more label.

**Leakage.** A niche label is an annotation, usually computed from every cell's expression, hidden cells included. Its patches therefore
say something about what a hidden cell expresses -- unlike positions, the only thing a disc, a grid patch or a kdtree cluster reads. Read
every arm built on them as an annotation arm (an upper bound: "does a niche prior help?"), not as the default.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import torch


def niche_codes(data, name: str) -> torch.Tensor:
    """``[N]`` integer codes of the one-hot annotation ``name`` on ``data``; -1 for a cell with no label.

    Raises
    ------
    AttributeError
        If the annotation is absent (the obs column must be in ``dataset.extra_obs_keys``, or be one of the named roles).
    """
    value = getattr(data, name, None)
    if value is None:
        raise AttributeError(
            f"Batch carries no {name!r}; the niche label reaches the graphs only when its obs column is in "
            "dataset.extra_obs_keys (or is a named role such as celltype_key)."
        )
    codes = value.argmax(dim=1)
    return torch.where(value.sum(dim=1) > 0, codes, torch.full_like(codes, -1))


def _renumber(patch: np.ndarray) -> np.ndarray:
    """``patch`` numbered 0 .. P - 1 in the order of each patch's first cell."""
    _, first, inverse = np.unique(patch, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inverse.reshape(-1)]


def label_components(edge_index, labels, num_nodes: int) -> np.ndarray:
    """``[N]``: the connected component of each cell on the edges whose two ends share a label, numbered by first cell."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    labels = np.asarray(labels)
    if len(labels) != num_nodes:
        raise ValueError(f"labels has {len(labels)} entries for {num_nodes} cells.")
    edges = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    keep = labels[edges[0]] == labels[edges[1]]
    src, dst = edges[0, keep], edges[1, keep]
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(num_nodes, num_nodes))
    _, comp = connected_components(graph, directed=True, connection="weak")
    return _renumber(comp)


def merge_small_patches(patch, edge_index, min_cells: int, pos=None) -> np.ndarray:
    """``patch`` with every patch under ``min_cells`` cells merged into a neighbour (module docstring), renumbered.

    Without ``pos`` a patch with no edge to another patch stays as it is.
    """
    patch = _renumber(np.asarray(patch))
    k = int(patch.max()) + 1 if len(patch) else 0
    if int(min_cells) <= 1 or k <= 1:
        return patch
    size = np.bincount(patch, minlength=k).astype(np.int64)
    sums = np.zeros((k, 2))
    if pos is not None:
        np.add.at(sums, patch, np.asarray(pos, dtype=np.float64)[:, :2])
    edges = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    a, b = patch[edges[0]], patch[edges[1]]
    cross = a != b
    pairs, weight = np.unique(np.stack([a[cross], b[cross]], 1), axis=0, return_counts=True)
    adjacent: list[dict] = [{} for _ in range(k)]
    for (u, v), w in zip(pairs.tolist(), weight.tolist(), strict=True):
        adjacent[u][v] = adjacent[u].get(v, 0) + w
        adjacent[v][u] = adjacent[v].get(u, 0) + w
    into = np.arange(k)
    alive = np.ones(k, dtype=bool)
    heap = [(int(size[c]), c) for c in range(k) if size[c] < min_cells]
    heapq.heapify(heap)
    n_alive = k
    while heap and n_alive > 1:
        s, c = heapq.heappop(heap)
        if not alive[c] or size[c] != s:
            continue
        if adjacent[c]:
            target = max(adjacent[c], key=lambda d: (adjacent[c][d], size[d], -d))
        elif pos is not None:
            others = np.flatnonzero(alive)
            others = others[others != c]
            gap = ((sums[others] / size[others, None] - sums[c] / size[c]) ** 2).sum(1)
            target = int(others[np.argmin(gap)])
        else:
            continue
        for d, w in adjacent[c].items():
            if d == target:
                continue
            adjacent[target][d] = adjacent[target].get(d, 0) + w
            adjacent[d][target] = adjacent[d].get(target, 0) + w
            del adjacent[d][c]
        adjacent[target].pop(c, None)
        adjacent[c] = {}
        size[target] += size[c]
        sums[target] += sums[c]
        alive[c] = False
        into[c] = target
        n_alive -= 1
        if size[target] < min_cells:
            heapq.heappush(heap, (int(size[target]), target))
    while True:
        deeper = into[into]
        if np.array_equal(deeper, into):
            break
        into = deeper
    return _renumber(into[patch])


def split_large_patches(patch, pos, max_cells: int) -> np.ndarray:
    """``patch`` with every patch over ``max_cells`` cells cut into ``ceil(size / max_cells)`` kdtree pieces, renumbered."""
    from interscale.module.global_modules.hierarchy import kd_parts

    patch = _renumber(np.asarray(patch))
    if int(max_cells) <= 0 or not len(patch):
        return patch
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    size = np.bincount(patch)
    out = patch.copy()
    next_id = len(size)
    order = np.argsort(patch, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(size)])
    for c in np.flatnonzero(size > max_cells):
        members = order[bounds[c] : bounds[c + 1]]
        for piece in kd_parts(pos, members, math.ceil(size[c] / max_cells))[1:]:
            out[piece] = next_id
            next_id += 1
    return _renumber(out)


def label_patches(
    edge_index, labels, num_nodes: int, *, pos=None, min_cells: int = 1, max_cells: int = 0
) -> np.ndarray:
    """``[N]`` the niche patch of each cell: label components, small ones merged, large ones split (module docstring).

    Parameters
    ----------
    edge_index
        ``[2, E]`` the graph's edges (cell indices 0 .. N - 1).
    labels
        ``[N]`` integer label codes.
    num_nodes
        ``N``.
    pos
        ``[N, >= 2]`` positions, needed by ``max_cells`` and by merging a patch with no edge to another.
    min_cells, max_cells
        Merge patches under ``min_cells`` cells (1: none), split patches over ``max_cells`` (0: none).
    """
    if int(max_cells) > 0 and pos is None:
        raise ValueError("splitting patches over max_cells cuts them by position; pass pos.")
    patch = label_components(edge_index, labels, num_nodes)
    patch = merge_small_patches(patch, edge_index, min_cells, pos)
    return split_large_patches(patch, pos, max_cells) if int(max_cells) > 0 else patch


def graph_label_patches(data, name: str, *, min_cells: int = 1, max_cells: int = 0) -> torch.Tensor:
    """``[N]`` (on the CPU) the niche patches of one graph ``data`` by its annotation ``name``."""
    pos = getattr(data, "pos", None)
    patch = label_patches(
        data.edge_index.cpu().numpy(),
        niche_codes(data, name).cpu().numpy(),
        int(data.num_nodes),
        pos=None if pos is None else pos.detach().cpu().double().numpy(),
        min_cells=min_cells,
        max_cells=max_cells,
    )
    return torch.as_tensor(patch, dtype=torch.long)


def batch_label_patches(batched_data, name: str, *, min_cells: int = 1, max_cells: int = 0) -> tuple[torch.Tensor, int]:
    """``(patch [N], n_patches)``: the niche patches of every graph of a batch, batch-global and ordered by graph.

    Each graph's patches are built on its own edges and are contiguous in the numbering, as ``pooling.grid_patches`` numbers
    its squares. The graphs' cells must be contiguous in the batch (as PyG collates them).
    """
    graph = batched_data.batch.long()
    graph_np = graph.cpu().numpy()
    if len(graph_np) > 1 and (np.diff(graph_np) < 0).any():
        raise ValueError("niche patches need each graph's cells contiguous in the batch, as PyG collates them.")
    codes = niche_codes(batched_data, name).cpu().numpy()
    edges = batched_data.edge_index.cpu().numpy()
    pos = getattr(batched_data, "pos", None)
    pos = None if pos is None else pos.detach().cpu().double().numpy()
    n_graphs = int(graph_np.max()) + 1 if len(graph_np) else 0
    counts = np.bincount(graph_np, minlength=n_graphs)
    starts = np.cumsum(counts) - counts
    edge_graph = graph_np[edges[0]] if edges.shape[1] else np.zeros(0, dtype=np.int64)
    out = np.empty(len(graph_np), dtype=np.int64)
    total = 0
    for g in range(n_graphs):
        lo, hi = int(starts[g]), int(starts[g] + counts[g])
        if hi == lo:
            continue
        local = label_patches(
            edges[:, edge_graph == g] - lo,
            codes[lo:hi],
            hi - lo,
            pos=None if pos is None else pos[lo:hi],
            min_cells=min_cells,
            max_cells=max_cells,
        )
        out[lo:hi] = local + total
        total += int(local.max()) + 1
    return torch.as_tensor(out, dtype=torch.long, device=graph.device), total
