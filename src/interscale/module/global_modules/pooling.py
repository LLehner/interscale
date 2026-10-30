"""Pooled tokens: the global component on patches of cells instead of single cells (``.claude/roadmap.md``, T1b).

The transformer sees one token per *patch* -- the mean of its visible cells' tokens, placed at the
patch's centroid -- and every cell's global embedding is its patch's. The tissue is covered by
``S`` tokens that number 10-1000 times fewer than its cells, so whole sections fit dense attention; a
patch token is a denoised summary of its cells, so the model does not have to look at single cells; and
because the embedding is shared by a patch, it cannot carry one cell's own token (the global head cannot
copy the local one). The loss, the metrics and the evaluation stay at the cell level.

Four pieces, in the order a step meets them:

* :func:`grid_patches` -- which patch each cell belongs to. A grid of squares by position alone
  (``size_um``, or the side that gives about ``n_patches`` patches per slide), so no expression value can
  decide a cell's patch. More ways to assign cells plug in next to it (``ASSIGNMENTS``): ``label``, one
  patch per connected piece of a niche label (``tl.niches.batch_label_patches``; ``.claude/lung_plan.md``,
  L3), small pieces merged (``min_cells``), large ones cut (``max_cells``). The label is an annotation
  computed from expression, hidden cells included, so it is the one assignment that does not read
  positions alone: an annotation arm, not the default.
* :meth:`PatchPooling.plan` and :meth:`PatchPooling.pool` -- the patch tokens: the mean over the patch's
  *visible* cells (a cell hidden whole contributes nothing), a learned token at the position of a patch
  with no visible cell, and optionally the log cell count and the spread of the cells' tokens
  (``features``), all starting at zero so a patch of one cell is that cell's token.
* :class:`PooledBatch` -- the patches as a batch of nodes, which is all ``common_step_local_to_global``
  reads (positions, graph index, a mask, no edges), so every attention backend, bias and encoding that
  needs coordinates works on patches unchanged.
* :func:`cell_layout` -- the patches' output back to one row per cell in the layout the decoders and
  every consumer of a step's output read (left-padded, CLS last, cells in batch order).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

#: How cells are assigned to patches (``pooling.kind``); ``none`` is no pooling.
POOLING_KINDS = ("none", "grid", "label")
#: Extra token features beyond the mean (``pooling.features``).
POOL_FEATURES = ("count", "std")


class PoolPlan(NamedTuple):
    """Which patch each cell belongs to, for one batch."""

    patch: torch.Tensor  # [N] batch-global patch index of each cell; patches are ordered by graph
    n_patches: int
    patch_graph: torch.Tensor  # [P] the graph each patch lies on
    centroid: torch.Tensor  # [P, 2] mean position of a patch's cells, in the units of ``pos``
    n_cells: torch.Tensor  # [P] cells in each patch


class PooledBatch:
    """The patches of a batch as a batch of nodes: what the transformer's input step reads off a graph batch.

    ``pos`` and ``batch`` are the patches' centroids and graph index; ``mask`` marks the patches with no
    visible cell; there are no edges (a pooled model has no long-range mask to build from them).
    """

    def __init__(self, pos: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor):
        self.pos, self.batch, self.mask = pos, batch, mask
        self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos.device)


def _occupied(rel: torch.Tensor, size: float, offset: float) -> int:
    """How many grid squares of side ``size`` hold at least one of the points ``rel`` ``[n, 2]``."""
    cell = torch.floor(rel / size + offset).long()
    return int(torch.unique(cell[:, 0] * (int(cell[:, 1].max()) + 1) + cell[:, 1]).numel())


def side_for_patches(rel: torch.Tensor, target: int, offset: float = 0.0, steps: int = 20) -> float:
    """The side of a square grid under which about ``target`` squares hold a point of ``rel`` ``[n, 2]``.

    Bisection in log space on the number of occupied squares, which falls as the side grows (not
    strictly: a square grid lines up with the points differently at every side), so the side with the
    count nearest ``target`` among those tried is returned. ``target`` at or above the number of points
    gives a side small enough for one point per square.
    """
    n = len(rel)
    extent = float(rel.max()) if n else 0.0
    if n <= 1 or extent <= 0:
        return 1.0
    hi = extent * 1.0001
    lo = hi / (2.0 * math.sqrt(n))
    if target >= n:
        return lo
    best, best_gap = hi, abs(1 - target)
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        count = _occupied(rel, mid, offset)
        gap = abs(count - target)
        if gap < best_gap:
            best, best_gap = mid, gap
        if count == target:
            break
        if count > target:
            lo = mid
        else:
            hi = mid
    return best


def grid_patches(
    pos: torch.Tensor, graph: torch.Tensor, *, size: float, n_patches: int = 0, offset: float = 0.0
) -> tuple[torch.Tensor, int]:
    """``(patch, n_patches)``: a square grid over each graph's cells, by position alone.

    The grid of a graph starts at its cells' minimum corner, so a slide's place in the scanner's
    coordinates does not matter; ``offset`` (a fraction of the side) shifts it. With ``size`` > 0 every
    graph gets squares of that side (in the units of ``pos``); with ``n_patches`` > 0 each graph gets the
    side that gives about that many occupied squares (:func:`side_for_patches`), so slides of different
    extent have the same number of patches. Patch indices are batch-global, ordered by graph and then by
    column and row of the square, so each graph's patches are contiguous.
    """
    n_graphs = int(graph.max()) + 1
    rel = torch.empty_like(pos, dtype=torch.float32)
    side = torch.empty(n_graphs, dtype=torch.float32, device=pos.device)
    for g in range(n_graphs):
        rows = graph == g
        rel[rows] = pos[rows].float() - pos[rows].float().min(0).values
        side[g] = side_for_patches(rel[rows], n_patches, offset) if n_patches else float(size)
    cell = torch.floor(rel / side[graph][:, None] + offset).long()
    ny = int(cell[:, 1].max()) + 1
    nx = int(cell[:, 0].max()) + 1
    key = (graph.long() * nx + cell[:, 0]) * ny + cell[:, 1]
    unique, patch = torch.unique(key, return_inverse=True)
    return patch, int(unique.numel())


def label_patches(batched_data, *, label: str, min_cells: int = 1, max_cells: int = 0) -> tuple[torch.Tensor, int]:
    """``(patch, n_patches)``: one patch per connected piece of the niche label ``label`` (a Data attribute), per graph.

    Batch-global and ordered by graph, as :func:`grid_patches` numbers its squares (``tl.niches.batch_label_patches``).
    """
    from interscale.tl.niches import batch_label_patches

    return batch_label_patches(batched_data, label, min_cells=min_cells, max_cells=max_cells)


#: How cells are assigned to patches, by ``pooling.kind``.
ASSIGNMENTS = {"grid": grid_patches, "label": label_patches}


class PatchPooling(nn.Module):
    """Cells to patch tokens and back: the mean of the visible cells' tokens, plus optional extras.

    Parameters
    ----------
    n_embed
        Width of a token.
    kind
        How cells are assigned to patches (``POOLING_KINDS``; ``grid`` or ``label``).
    size_um, n_patches
        ``grid``: the grid's side in µm, or the number of patches wanted per slide (exactly one is positive).
    offset
        ``grid``: shift of the grid as a fraction of its side, in [0, 1).
    label, min_cells, max_cells
        ``label``: the Data attribute of the niche label, the size under which a patch merges into a neighbour (1: none)
        and over which it is cut into kdtree pieces (0: none).
    features
        Any of ``count`` (a learned vector times the log number of visible cells) and ``std`` (a learned
        linear map of the per-dimension spread of the visible cells' tokens), each starting at zero.
    spatial_unit_um
        µm per unit of ``pos``.
    """

    def __init__(
        self,
        n_embed: int,
        *,
        kind: str = "grid",
        size_um: float = 0.0,
        n_patches: int = 0,
        offset: float = 0.0,
        features=(),
        spatial_unit_um: float = 1.0,
        label: str = "",
        min_cells: int = 1,
        max_cells: int = 0,
    ):
        super().__init__()
        if kind not in ASSIGNMENTS:
            raise ValueError(f"pooling.kind must be one of {sorted(ASSIGNMENTS)}, got {kind!r}.")
        unknown = sorted(set(features) - set(POOL_FEATURES))
        if unknown:
            raise ValueError(f"pooling.features names {unknown}; choose from {list(POOL_FEATURES)}.")
        if kind == "label" and not label:
            raise ValueError("pooling.kind label needs the niche label's Data attribute.")
        self.kind, self.size_um, self.n_patches = kind, float(size_um), int(n_patches)
        self.offset, self.features, self.spatial_unit_um = float(offset), tuple(features), float(spatial_unit_um)
        self.label, self.min_cells, self.max_cells = str(label), int(min_cells), int(max_cells)
        # A patch with no visible cell has no token of its own: one learned vector stands in, drawn from a
        # private generator so that building it moves no other initial weight.
        generator = torch.Generator().manual_seed(0)
        self.fill = nn.Parameter(torch.randn(n_embed, generator=generator))
        if "count" in self.features:
            self.count_embed = nn.Parameter(torch.zeros(n_embed))
        if "std" in self.features:
            self.std_embed = nn.Linear(n_embed, n_embed)
            nn.init.zeros_(self.std_embed.weight)
            nn.init.zeros_(self.std_embed.bias)

    def extra_repr(self) -> str:
        if self.kind == "label":
            return (
                f"kind=label, label={self.label}, min_cells={self.min_cells}, max_cells={self.max_cells}, "
                f"features={list(self.features)}"
            )
        side = f"{self.size_um:g} um" if self.size_um else f"~{self.n_patches} patches"
        return f"kind={self.kind}, {side}, offset={self.offset:g}, features={list(self.features)}"

    def plan(self, batched_data, max_patches: int | None = None) -> PoolPlan:
        """Assign the cells of a batch to patches: by ``batched_data.pos`` alone (``grid``), or by the niche label on the
        graph's edges (``label``).
        """
        pos = getattr(batched_data, "pos", None)
        if pos is None:
            raise ValueError(
                "pooling needs cell coordinates, but the graphs carry no `pos`. Set dataset.spatial_key "
                "(e.g. 'spatial') so they are attached."
            )
        graph = batched_data.batch.long()
        if self.kind == "label":
            patch, n = ASSIGNMENTS["label"](
                batched_data, label=self.label, min_cells=self.min_cells, max_cells=self.max_cells
            )
        else:
            patch, n = ASSIGNMENTS[self.kind](
                pos, graph, size=self.size_um / self.spatial_unit_um, n_patches=self.n_patches, offset=self.offset
            )
        patch_graph = torch.zeros(n, dtype=torch.long, device=pos.device)
        patch_graph[patch] = graph
        n_cells = torch.bincount(patch, minlength=n)
        centroid = torch.zeros(n, 2, dtype=torch.float32, device=pos.device).index_add(0, patch, pos.float()[:, :2])
        centroid = centroid / n_cells[:, None].clamp_min(1)
        if max_patches is not None:
            most = int(torch.bincount(patch_graph).max())
            if most > max_patches:
                raise ValueError(
                    f"pooling gives {most} patches on one slide, more than max_seq_len ({max_patches}), so the "
                    "transformer would drop some. Raise max_seq_len, or use larger patches (pooling.size_um, or "
                    "pooling.min_cells for niche patches) or fewer (pooling.n_patches)."
                )
        return PoolPlan(patch, n, patch_graph, centroid.to(pos.dtype), n_cells)

    def pool(self, emb: torch.Tensor, hidden: torch.Tensor, plan: PoolPlan) -> tuple[torch.Tensor, torch.Tensor]:
        """``([P, E] patch tokens, [P] visible cells per patch)`` from the cell tokens ``emb`` ``[N, E]``.

        ``hidden`` ``[N]`` marks the cells hidden whole; they weigh 0, so a token never holds a hidden
        cell's value. A patch with no visible cell gets the learned fill.
        """
        weight = (~hidden.bool()).to(emb.dtype)
        n_patches = plan.n_patches
        visible = torch.zeros(n_patches, dtype=emb.dtype, device=emb.device).index_add(0, plan.patch, weight)
        total = torch.zeros(n_patches, emb.shape[1], dtype=emb.dtype, device=emb.device)
        total = total.index_add(0, plan.patch, emb * weight[:, None])
        mean = total / visible.clamp_min(1)[:, None]
        token = torch.where(visible[:, None] > 0, mean, self.fill.to(emb.dtype).expand_as(mean))
        if "count" in self.features:
            token = token + self.count_embed.to(emb.dtype) * torch.log1p(visible)[:, None]
        if "std" in self.features:
            squares = torch.zeros_like(total).index_add(0, plan.patch, (emb - mean[plan.patch]).pow(2) * weight[:, None])
            spread = (squares / visible.clamp_min(1)[:, None] + 1e-6).sqrt()
            token = token + self.std_embed(spread)
        return token, visible

    def batch(self, plan: PoolPlan, visible: torch.Tensor) -> PooledBatch:
        """The patches as a batch of nodes; a patch with no visible cell is the ``mask`` the padding keeps."""
        return PooledBatch(plan.centroid, plan.patch_graph, visible <= 0)


def cell_layout(cell_rows: torch.Tensor, cls_row: torch.Tensor, graph: torch.Tensor, n_graphs: int):
    """One row per cell in the transformer's output layout: ``(embedding, padding mask, index_nodes)``.

    ``cell_rows`` ``[N, E]`` in batch order and ``cls_row`` ``[B, E]`` (the CLS output). The embedding is
    ``[S + 1, B, E]``, ``S`` the largest slide's cell count, each graph's cells in the last positions
    before CLS (left-padded, as ``pad_batch`` lays out tokens), CLS at the last position; the mask is
    ``[B, S + 1]``, True at padding; ``index_nodes`` lists each graph's cells, all of them, in order.
    """
    counts = torch.bincount(graph, minlength=n_graphs)
    length = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(graph), device=graph.device) - starts[graph]
    column = length - counts[graph] + rank
    padded = cell_rows.new_zeros(length, n_graphs, cell_rows.shape[1])
    padded[column, graph] = cell_rows
    embedding = torch.cat([padded, cls_row[None]], dim=0)
    padding = torch.arange(length + 1, device=graph.device)[None, :] < (length - counts)[:, None]
    return embedding, padding, [list(range(int(c))) for c in counts]


def build_pooling(cfg, n_embed: int) -> PatchPooling | None:
    """The configured :class:`PatchPooling` (``model.global_component.parameters.pooling``), or None for ``kind: none``."""
    params = cfg.model.global_component.get("parameters", None)
    pooling = params.get("pooling", None) if params is not None else None
    if pooling is None or pooling.get("kind", "none") == "none":
        return None
    label = ""
    if pooling.kind == "label":
        from interscale.tl.geome_utils import label_attribute

        label = label_attribute(cfg, str(pooling.label_key))
    return PatchPooling(
        n_embed,
        kind=str(pooling.kind),
        size_um=float(pooling.size_um),
        n_patches=int(pooling.n_patches),
        offset=float(pooling.offset),
        features=list(pooling.features),
        spatial_unit_um=float(cfg.dataset.get("spatial_unit_um", 1.0)),
        label=label,
        min_cells=int(pooling.get("min_cells", 1)),
        max_cells=int(pooling.get("max_cells", 0)),
    )
