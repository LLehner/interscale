"""The cluster hierarchy of the hierarchical attention (``.claude/hierarchical_attention_plan.md``, T1).

Per slide, coarse levels ``0`` (regions, the coarsest) .. ``L - 1`` over the cells (level ``L``), each cluster the union of its
children. Built from the cells' positions alone -- never from expression -- so no hidden value can decide which cluster a cell
joins, and the same slide always gets the same hierarchy (cached by a fingerprint of its positions).

The top level and the finer ones are built separately:

* the top level (``top_mode``): ``tree`` -- the builder over every cell; ``patches`` -- a square grid by position
  (``pooling.grid_patches``), patches below ``min_cells`` merged into the nearest larger one;
* every finer level: each parent split by the builder into a number of children proportional to its cells, so the clusters of
  a level hold about the same number of cells. ``kdtree``: recursive splits at the cell-count quantile across the longer
  side -- balanced by cell count, so empty space (an FOV gap, a hole) makes no empty cluster. ``random``: a balanced random
  partition, the negative control (clusters with no spatial meaning).

``shift`` builds the second hierarchy of a Swin-style pair: the kdtree on coordinates turned by 45 degrees (its boundaries
cross the first one's), a patch grid moved by half a patch, a random partition from another seed. The layers alternate
between the two.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch

HIERARCHY_BUILDERS = ("kdtree", "random")
TOP_LEVEL_MODES = ("tree", "patches")


@dataclass
class Hierarchy:
    """One slide's hierarchy: ``L`` coarse levels over ``n`` cells, every array indexed by the cells' order as given.

    Level ``L`` is the cells themselves. Lists indexed by level hold ``L + 1`` entries where a quantity is defined for the
    cells too (``counts``, ``centroid``, ``up``), else ``L`` (the coarse levels).
    """

    counts: list[int]  # clusters per level, coarsest first; counts[L] = n
    assign: list[torch.Tensor]  # per coarse level: [n] the cluster of each cell
    up: list[torch.Tensor | None]  # per level: [n_l] its parent at level l - 1 (None at level 0)
    children: list[torch.Tensor]  # per coarse level: [n_l, m_l] the ids of its children at level l + 1, -1 padded
    n_children: list[torch.Tensor]  # per coarse level: [n_l]
    centroid: list[torch.Tensor]  # per level: [n_l, 2] mean position of a cluster's cells (level L: the cells), um
    spread: list[torch.Tensor]  # per coarse level: [n_l] root mean square distance of its cells to the centroid, um
    sizes: list[torch.Tensor]  # per coarse level: [n_l] cells
    neighbours: list[
        torch.Tensor
    ]  # per coarse level: [n_l, k] nearest clusters by centroid, the cluster itself first; -1 pad
    build_seconds: float = 0.0
    _devices: dict = field(default_factory=dict, repr=False)

    @property
    def n_levels(self) -> int:
        """``L``: the number of coarse levels."""
        return len(self.assign)

    def to(self, device) -> Hierarchy:
        """This hierarchy on ``device`` (kept, so a slide seen again is not copied again)."""
        device = torch.device(device)
        if device == self.centroid[-1].device:
            return self
        key = str(device)
        if key not in self._devices:

            def move(values):
                return [v.to(device) if v is not None else None for v in values]

            self._devices[key] = Hierarchy(
                list(self.counts),
                move(self.assign),
                move(self.up),
                move(self.children),
                move(self.n_children),
                move(self.centroid),
                move(self.spread),
                move(self.sizes),
                move(self.neighbours),
                self.build_seconds,
            )
        return self._devices[key]

    def stats(self) -> dict:
        """Per coarse level: clusters, cells per cluster (min, median, max) and the spread's spatial compactness."""
        out = {"build_seconds": self.build_seconds}
        for level in range(self.n_levels):
            sizes = self.sizes[level].float().cpu()
            spread = self.spread[level].float().cpu()
            out[f"l{level}_clusters"] = int(self.counts[level])
            out[f"l{level}_cells_min"] = float(sizes.min())
            out[f"l{level}_cells_median"] = float(sizes.median())
            out[f"l{level}_cells_max"] = float(sizes.max())
            out[f"l{level}_spread_um_median"] = float(spread.median())
        return out


def _rotate(pos: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return pos @ rot.T


def kd_parts(pos: np.ndarray, members: np.ndarray, k: int) -> list[np.ndarray]:
    """``members`` (indices into ``pos``) in ``k`` parts of about equal size, by recursive splits across the longer side.

    Each split cuts at the cell-count quantile ``floor(k / 2) / k`` along the axis of larger extent; ties in the coordinate
    (a lattice) are broken by the other coordinate, then the index, so the result is deterministic. Parts come in
    depth-first order (left / lower first), so neighbouring parts tend to have neighbouring numbers.
    """
    k = max(1, min(int(k), len(members)))
    parts, stack = [], [(np.asarray(members), k)]
    while stack:
        idx, kk = stack.pop()
        if kk <= 1 or len(idx) <= 1:
            parts.append(idx)
            continue
        left = kk // 2
        sub = pos[idx]
        axis = int(np.argmax(sub.max(0) - sub.min(0)))
        order = np.lexsort((idx, sub[:, 1 - axis], sub[:, axis]))
        cut = min(max(int(round(len(idx) * left / kk)), 1), len(idx) - 1)
        stack.append((idx[order[cut:]], kk - left))
        stack.append((idx[order[:cut]], left))
    return parts


def random_parts(rng: np.random.Generator, members: np.ndarray, k: int) -> list[np.ndarray]:
    """``members`` in ``k`` random parts of about equal size."""
    k = max(1, min(int(k), len(members)))
    return [np.sort(p) for p in np.array_split(rng.permutation(np.asarray(members)), k)]


def patch_labels(pos: np.ndarray, *, size_um: float, n_patches: int, offset: float, min_cells: int) -> np.ndarray:
    """``[n]`` patch of each cell: a square grid by position (``size_um``, or about ``n_patches``), small patches merged.

    A patch with fewer than ``min_cells`` cells joins the patch of larger size whose centroid is nearest. Patches are numbered
    by the grid's column and row.
    """
    from interscale.module.global_modules.pooling import grid_patches

    t = torch.as_tensor(pos, dtype=torch.float64)
    patch, n = grid_patches(
        t, torch.zeros(len(t), dtype=torch.long), size=float(size_um), n_patches=int(n_patches), offset=offset
    )
    labels = patch.numpy()
    counts = np.bincount(labels, minlength=n)
    big = np.flatnonzero(counts >= max(1, int(min_cells)))
    if len(big) and len(big) < n:
        centroid = np.zeros((n, 2))
        np.add.at(centroid, labels, pos)
        centroid /= np.maximum(counts, 1)[:, None]
        small = np.flatnonzero((counts < max(1, int(min_cells))) & (counts > 0))
        d = ((centroid[small][:, None, :] - centroid[big][None, :, :]) ** 2).sum(-1)
        target = np.arange(n)
        target[small] = big[d.argmin(1)]
        labels = target[labels]
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def _labels(parts: list[np.ndarray], n: int, start: int = 0) -> tuple[np.ndarray, int]:
    out = np.empty(n, dtype=np.int64)
    for k, p in enumerate(parts):
        out[p] = start + k
    return out, start + len(parts)


def build_hierarchy(
    pos_um,
    *,
    levels=(),
    cells_per_cluster=(),
    builder: str = "kdtree",
    top_mode: str = "tree",
    patch_size_um: float = 0.0,
    min_cells: int = 1,
    neighbors: int = 8,
    shift: bool = False,
    seed: int = 0,
) -> Hierarchy:
    """The hierarchy over cells at ``pos_um`` ``[n, 2]`` (µm), on the CPU.

    Parameters
    ----------
    levels, cells_per_cluster
        The coarse levels, coarsest first: a number of clusters per level, or the cells a cluster should hold (exactly one of
        the two is given; ``cells_per_cluster`` keeps the size of a cluster across slides of different size). Each level
        has at least as many clusters as the one above.
    builder
        ``kdtree`` or ``random`` (``HIERARCHY_BUILDERS``): how every finer level splits its parents (and the top level under
        ``top_mode`` ``tree``).
    top_mode
        ``tree`` or ``patches`` (``TOP_LEVEL_MODES``); with ``patches`` the top level is a grid of side ``patch_size_um``, or
        about ``levels[0]`` patches (``cells_per_cluster[0]`` cells each) when it is 0, and patches under ``min_cells`` merge.
    neighbors
        Spatial neighbour clusters listed per cluster (nearest centroids), besides the cluster itself.
    shift
        Build the shifted twin (module docstring).
    seed
        For the ``random`` builder.
    """
    started = time.perf_counter()
    pos = np.asarray(pos_um, dtype=np.float64)
    n = len(pos)
    if (len(levels) > 0) == (len(cells_per_cluster) > 0):
        raise ValueError("give exactly one of levels (clusters per coarse level) and cells_per_cluster.")
    if builder not in HIERARCHY_BUILDERS:
        raise ValueError(f"builder must be one of {HIERARCHY_BUILDERS}, got {builder!r}.")
    if top_mode not in TOP_LEVEL_MODES:
        raise ValueError(f"top_mode must be one of {TOP_LEVEL_MODES}, got {top_mode!r}.")
    if len(levels):
        targets = [int(c) for c in levels]
    else:
        targets = [max(1, int(round(n / float(c)))) for c in cells_per_cluster]
    split_pos = _rotate(pos, 45.0) if shift and builder == "kdtree" else pos
    rng = np.random.default_rng([int(seed), n, 1 if shift else 0])

    assign: list[np.ndarray] = []
    all_cells = np.arange(n)
    for level, target in enumerate(targets):
        target = max(1, min(target, n))
        if level == 0:
            if top_mode == "patches":
                labels = patch_labels(
                    pos,
                    size_um=float(patch_size_um),
                    n_patches=0 if patch_size_um > 0 else target,
                    offset=0.5 if shift else 0.0,
                    min_cells=min_cells,
                )
            else:
                parts = (
                    kd_parts(split_pos, all_cells, target)
                    if builder == "kdtree"
                    else random_parts(rng, all_cells, target)
                )
                labels, _ = _labels(parts, n)
            assign.append(labels)
            continue
        above = assign[-1]
        order = np.argsort(above, kind="stable")
        bounds = np.searchsorted(above[order], np.arange(above.max() + 2))
        parts = []
        for c in range(above.max() + 1):
            members = order[bounds[c] : bounds[c + 1]]
            k = max(1, int(round(target * len(members) / n)))
            parts += kd_parts(split_pos, members, k) if builder == "kdtree" else random_parts(rng, members, k)
        labels, _ = _labels(parts, n)
        assign.append(labels)

    L = len(assign)
    counts = [int(a.max()) + 1 for a in assign] + [n]
    up: list[np.ndarray | None] = [None]
    for level in range(1, L + 1):
        parent = np.zeros(counts[level], dtype=np.int64)
        child_of_cell = assign[level] if level < L else np.arange(n)
        parent[child_of_cell] = assign[level - 1]
        up.append(parent)
    children, n_children = [], []
    for level in range(L):
        parent = up[level + 1]
        order = np.argsort(parent, kind="stable")
        count = np.bincount(parent, minlength=counts[level])
        width = int(count.max())
        table = np.full((counts[level], width), -1, dtype=np.int64)
        slot = np.arange(len(order)) - np.repeat(np.cumsum(count) - count, count)
        table[parent[order], slot] = order
        children.append(table)
        n_children.append(count)
    centroid, spread, sizes, neighbours = [], [], [], []
    for level in range(L):
        a = assign[level]
        size = np.bincount(a, minlength=counts[level])
        c = np.zeros((counts[level], 2))
        np.add.at(c, a, pos)
        c /= np.maximum(size, 1)[:, None]
        sq = np.zeros(counts[level])
        np.add.at(sq, a, ((pos - c[a]) ** 2).sum(1))
        centroid.append(c)
        spread.append(np.sqrt(sq / np.maximum(size, 1)))
        sizes.append(size)
        neighbours.append(_neighbour_table(c, int(neighbors)))
    centroid.append(pos)

    def t(values, dtype=torch.long):
        return [torch.as_tensor(v, dtype=dtype) if v is not None else None for v in values]

    return Hierarchy(
        counts=counts,
        assign=t(assign),
        up=t(up),
        children=t(children),
        n_children=t(n_children),
        centroid=t(centroid, torch.float32),
        spread=t(spread, torch.float32),
        sizes=t(sizes),
        neighbours=t(neighbours),
        build_seconds=time.perf_counter() - started,
    )


def flat_hierarchy(pos_um) -> Hierarchy:
    """No coarse level: the cells alone (``L = 0``), so the hierarchical layer is the dense attention."""
    pos = torch.as_tensor(np.asarray(pos_um, dtype=np.float64), dtype=torch.float32)
    return Hierarchy(
        counts=[len(pos)],
        assign=[],
        up=[None],
        children=[],
        n_children=[],
        centroid=[pos],
        spread=[],
        sizes=[],
        neighbours=[],
    )


def _neighbour_table(centroid: np.ndarray, k: int) -> np.ndarray:
    """``[n, min(k, n - 1) + 1]``: each cluster itself, then its ``k`` nearest other clusters by centroid."""
    from scipy.spatial import cKDTree

    n = len(centroid)
    width = min(k, n - 1) + 1
    if width <= 1:
        return np.arange(n)[:, None]
    _, found = cKDTree(centroid).query(centroid, k=min(width + 1, n))
    found = np.atleast_2d(found)
    table = np.empty((n, width), dtype=np.int64)
    for i in range(n):
        others = [int(j) for j in found[i] if j != i]
        table[i, 0] = i
        table[i, 1:] = others[: width - 1]
    return table


class HierarchyCache:
    """Hierarchies by slide: a slide's positions (and the settings) map to the one hierarchy built for them.

    The positions never change between steps, so a slide is built once per process; ``max_entries`` bounds the memory.
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = int(max_entries)
        self._entries: OrderedDict = OrderedDict()

    def get(self, pos_um: torch.Tensor, **settings) -> Hierarchy:
        array = np.ascontiguousarray(pos_um.detach().double().cpu().numpy())
        key = (
            hashlib.sha1(array.tobytes()).hexdigest(),
            array.shape,
            tuple(sorted((k, repr(v)) for k, v in settings.items())),
        )
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        has_levels = len(settings.get("levels", ())) or len(settings.get("cells_per_cluster", ()))
        built = build_hierarchy(array, **settings) if has_levels else flat_hierarchy(array)
        self._entries[key] = built
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return built
