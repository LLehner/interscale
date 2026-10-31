"""Net attention flow on a slide, drawn as streams on the tissue (``.claude/roadmap.md``, task 5; the manuscript's net streams
of :mod:`~interscale.evaluation.net_streams`, for whole sections).

**What is drawn.** The last layer's head-mean attention over cells, ``A_ij`` (cell ``i`` reads cell ``j``; the weight on CLS
is not a cell and is left out). The *net* flow of a pair is ``F_ij = A_ij - A_ji`` as in the manuscript's ``M - M^T``: positive
when ``i`` reads ``j`` more than ``j`` reads ``i``, so information goes ``j -> i`` (``j`` is the source, ``i`` the receiver).
Each pair with ``F > 0`` is an arrow of length ``F`` from ``x_j`` to ``x_i``, deposited at its midpoint; the deposits are summed
on a grid and smoothed with a Gaussian, and divided by the smoothed cell count, so the field at a place is the **net current
of attention per cell** through it. A stream follows that field. No label enters: the cells are drawn grey, the cells hidden in
the pass the model was asked about are tinted.

**Routed attention.** A routed row holds the keys of its parent's candidates only, so a pair can be known in one row and absent
from the other, where it counts as 0 and would show as a net flow of the full weight. By default only the *mutual* pairs, known
in both rows, enter (``mutual``); the share of the attention mass that sits on pairs known in one row only is reported, so how
much that drops is visible. With dense attention every pair is mutual and this is the manuscript's ``M - M^T``.

**What it is, checked.** A picture of where the model reads, not a measurement of signalling. On synth_spot's trained windowed
models (notes.md, 2026-10-31) the head mean's top 5 holds a planted short-range sender for 97% of the receivers (the untrained twin
69%: the distance window alone) and a mid-range one for 73% (twin 7%), and the net flow points from the sender to the receiver for 61%
and 94% of the pairs (twin 49% and 42%); it finds none of the dense, sparse or directional senders, and integrated gradients are
stronger (mid range 100%). Part of any net flow is geometry -- density, the tissue's edge, the layout of the mask (a hidden cell reads
the visible ones around it, which read it little). Read it against the untrained twin, drawn by the same code. The hierarchy's coarse
levels are not in it (the cell level only), nor the first layer.

**Layout.** A section is a few tissue blocks (CosMx: its FOVs) with empty slide between them. :func:`tissue_blocks` finds
them, :func:`plot_attention_flow` packs one tile per block into one figure at one scale, so nothing but tissue is drawn.

Everything but the plot is a function of arrays (:func:`attention_matrix` from the rows ``cell_rows`` of
:mod:`~interscale.evaluation.hierarchy_readouts` gives), so it tests on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix

#: The arrays :func:`flow_arrays` puts into a slide's npz (besides ``pos_um`` and ``visible``).
GRID_KEYS = ("flow_ux", "flow_uy", "flow_mass", "flow_cells", "flow_origin", "flow_grid_um")
#: The per-cell and per-pair arrays it adds: each cell's net flow out and in, and the sampled receivers' top sources.
CELL_KEYS = ("flow_out", "flow_in", "pair_receiver", "pair_source", "pair_flow")


def attention_matrix(keys, weights, n: int) -> csr_matrix:
    """``A [n, n]`` (float32): the head mean of ``weights`` ``[H, n, K]`` over the cell keys ``keys`` ``[n, K]``.

    Key ids outside ``[0, n)`` (CLS is ``n``, padding ``-1``) and a cell's own key are left out; a key listed twice adds up.
    """
    keys = np.asarray(keys)
    w = np.asarray(weights, dtype=np.float32).mean(0)
    rows = np.broadcast_to(np.arange(n)[:, None], keys.shape)
    ok = (keys >= 0) & (keys < n) & (keys != rows) & (w > 0)
    return coo_matrix((w[ok], (rows[ok], keys[ok])), shape=(n, n)).tocsr()


def net_flow_pairs(A: csr_matrix, *, mutual: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(source, receiver, flow, mutual_share)``: every pair with a positive net flow ``F_ij = A_ij - A_ji``.

    ``source[k] = j`` and ``receiver[k] = i`` for ``F_ij > 0``: ``i`` reads ``j`` more than ``j`` reads ``i``. With ``mutual``
    only pairs present in both rows count. ``mutual_share`` is the share of the attention mass (``sum A``) on pairs present in
    both rows (1 for dense attention).
    """
    A = csr_matrix(A, dtype=np.float32)
    both = A.sign().multiply(A.T.sign()).tocsr()
    total = float(A.sum())
    mutual_share = float(A.multiply(both).sum()) / total if total > 0 else float("nan")
    F = A - A.T
    if mutual:
        F = F.multiply(both)
    F = F.tocoo()
    keep = F.data > 0
    return F.col[keep].astype(np.int64), F.row[keep].astype(np.int64), F.data[keep].astype(np.float32), mutual_share


def flow_grid(pos, source, receiver, flow, *, grid_um: float = 10.0) -> dict[str, np.ndarray]:
    """The deposits of the pairs on a grid of ``grid_um`` squares over the cells' bounding box (``flow_*`` arrays ``[nx, ny]``).

    ``flow_ux`` / ``flow_uy``: the sum of ``F`` times the unit vector from source to receiver, in the bin of the pair's
    midpoint; ``flow_mass``: the sum of ``F``; ``flow_cells``: cells per bin. ``flow_origin`` and ``flow_grid_um`` place the grid.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    low = pos.min(0) - grid_um
    shape = np.ceil((pos.max(0) + grid_um - low) / grid_um).astype(np.int64) + 1
    size = int(shape[0] * shape[1])
    delta = pos[receiver] - pos[source]
    length = np.hypot(delta[:, 0], delta[:, 1])
    ok = length > 0
    delta, length, f = delta[ok], length[ok], np.asarray(flow, dtype=np.float64)[ok]
    mid = (pos[source[ok]] + pos[receiver[ok]]) / 2.0
    cell = np.floor((mid - low) / grid_um).astype(np.int64)
    flat = cell[:, 0] * shape[1] + cell[:, 1]
    unit = delta / length[:, None]
    at = np.floor((pos - low) / grid_um).astype(np.int64)

    def binned(values=None, index=flat):
        return np.bincount(index, weights=values, minlength=size).reshape(shape).astype(np.float32)

    return {
        "flow_ux": binned(f * unit[:, 0]),
        "flow_uy": binned(f * unit[:, 1]),
        "flow_mass": binned(f),
        "flow_cells": binned(None, at[:, 0] * shape[1] + at[:, 1]),
        "flow_origin": low.astype(np.float64),
        "flow_grid_um": np.asarray(float(grid_um)),
    }


def flow_summary(pos, visible, source, receiver, flow, mutual_share: float = float("nan")) -> dict[str, float]:
    """One slide's numbers: the net flow per cell, where it goes (hidden cells), how far, and the mutual share.

    ``to_hidden``: the share of the net flow whose receiver is a hidden cell; ``visible_to_hidden``: from a visible source to
    a hidden receiver; ``hidden_cells``: the share of cells that are hidden (what ``to_hidden`` would be by chance);
    ``length_um``: the flow-weighted mean distance between source and receiver.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    hidden = np.asarray(visible) == 0
    f = np.asarray(flow, dtype=np.float64)
    total = float(f.sum())
    if total <= 0:
        nan = float("nan")
        return {
            "net_flow_per_cell": 0.0,
            "to_hidden": nan,
            "visible_to_hidden": nan,
            "hidden_cells": float(hidden.mean()),
            "length_um": nan,
            "mutual_share": mutual_share,
        }
    distance = np.linalg.norm(pos[receiver] - pos[source], axis=1)
    return {
        "net_flow_per_cell": total / len(pos),
        "to_hidden": float(f[hidden[receiver]].sum() / total),
        "visible_to_hidden": float(f[hidden[receiver] & ~hidden[source]].sum() / total),
        "hidden_cells": float(hidden.mean()),
        "length_um": float((f * distance).sum() / total),
        "mutual_share": mutual_share,
    }


def cell_strengths(n: int, source, receiver, flow) -> tuple[np.ndarray, np.ndarray]:
    """``(out, in)`` ``[n]``: each cell's net flow sent (as a source) and received (as a receiver). ``out - in`` is the exact
    divergence of the net flow at the cell: positive for a source (read more than it reads), negative for a sink.
    """
    f = np.asarray(flow, dtype=np.float64)
    out = np.bincount(np.asarray(source, dtype=np.int64), weights=f, minlength=n)
    into = np.bincount(np.asarray(receiver, dtype=np.int64), weights=f, minlength=n)
    return out.astype(np.float32), into.astype(np.float32)


def sample_pairs(
    visible, source, receiver, flow, *, receivers: int = 2000, top: int = 5, seed: int = 0
) -> dict[str, np.ndarray]:
    """The ``top`` sources of up to ``receivers`` receivers, by net flow: ``pair_receiver`` ``[R]``, ``pair_source`` and
    ``pair_flow`` ``[R, top]`` (-1 and 0 where a receiver has fewer). Hidden cells with an inflow are drawn first (the cells the
    model is asked about), else any cell with one; from a private generator, sorted.
    """
    n = len(visible)
    receiver, source = np.asarray(receiver, dtype=np.int64), np.asarray(source, dtype=np.int64)
    flow = np.asarray(flow, dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    has[receiver] = True
    hidden = np.asarray(visible) == 0
    pool = np.flatnonzero(has & hidden)
    if not len(pool):
        pool = np.flatnonzero(has)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(pool, size=min(int(receivers), len(pool)), replace=False)) if len(pool) else pool
    slot = np.full(n, -1, dtype=np.int64)
    slot[chosen] = np.arange(len(chosen))
    keep = slot[receiver] >= 0
    r, src, f = receiver[keep], source[keep], flow[keep]
    order = np.lexsort((-f, r))
    r, src, f = r[order], src[order], f[order]
    start = np.searchsorted(r, r, side="left")
    rank = np.arange(len(r)) - start
    take = rank < top
    pair_source = np.full((len(chosen), top), -1, dtype=np.int64)
    pair_flow = np.zeros((len(chosen), top), dtype=np.float32)
    pair_source[slot[r[take]], rank[take]] = src[take]
    pair_flow[slot[r[take]], rank[take]] = f[take]
    return {"pair_receiver": chosen.astype(np.int64), "pair_source": pair_source, "pair_flow": pair_flow}


def flow_arrays(keys, weights, pos_um, visible, *, grid_um: float = 10.0, mutual: bool = True, receivers: int = 2000):
    """``(arrays, summary)`` of one slide from its cell-level rows: the npz arrays (``pos_um``, ``visible``, the grid, each cell's
    ``flow_out`` / ``flow_in`` (:func:`cell_strengths`) and the sampled pairs (:func:`sample_pairs`)) and :func:`flow_summary`.
    ``keys`` ``[n, K]`` and ``weights`` ``[H, n, K]`` as ``hierarchy_readouts.cell_rows`` returns them.
    """
    pos = np.asarray(pos_um, dtype=np.float32)
    n = len(pos)
    A = attention_matrix(keys, weights, n)
    source, receiver, flow, mutual_share = net_flow_pairs(A, mutual=mutual)
    arrays = {"pos_um": pos, "visible": np.asarray(visible, dtype=np.float32)}
    arrays |= flow_grid(pos, source, receiver, flow, grid_um=grid_um)
    arrays["flow_out"], arrays["flow_in"] = cell_strengths(n, source, receiver, flow)
    arrays |= sample_pairs(arrays["visible"], source, receiver, flow, receivers=receivers)
    return arrays, flow_summary(pos, visible, source, receiver, flow, mutual_share)


# ------------------------------------------------------------------------------------------------- the tissue


def _tight(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    """The bounding box ``(x0, x1, y0, y1)`` of the occupied bins of ``mask[x0:x1, y0:y1]``, or None when there are none."""
    sub = mask[x0:x1, y0:y1]
    rows, cols = np.flatnonzero(sub.any(1)), np.flatnonzero(sub.any(0))
    if not len(rows):
        return None
    return x0 + int(rows[0]), x0 + int(rows[-1]) + 1, y0 + int(cols[0]), y0 + int(cols[-1]) + 1


def rectangles(mask: np.ndarray, gain: float = 0.75) -> list[tuple[int, int, int, int]]:
    """Boxes ``(x0, x1, y0, y1)`` of the occupied bins of ``mask``, cut until each is mostly tissue.

    A box is cut along the grid line that makes the two parts' boxes smallest, when their areas add up to less than ``gain``
    times its own: an L of two fields cut at their corner, a rectangle (or one with a notch) left whole.
    """
    first = _tight(mask, 0, mask.shape[0], 0, mask.shape[1])
    todo, out = ([first] if first is not None else []), []
    while todo:
        box = todo.pop()
        x0, x1, y0, y1 = box
        best, best_area = None, gain * (x1 - x0) * (y1 - y0)
        for axis in (0, 1):
            for c in range((x0, y0)[axis] + 1, (x1, y1)[axis]):
                low = _tight(mask, x0, c, y0, y1) if axis == 0 else _tight(mask, x0, x1, y0, c)
                high = _tight(mask, c, x1, y0, y1) if axis == 0 else _tight(mask, x0, x1, c, y1)
                if low is None or high is None:
                    continue
                area = sum((b[1] - b[0]) * (b[3] - b[2]) for b in (low, high))
                if area < best_area:
                    best, best_area = (low, high), area
        if best is None:
            out.append(box)
        else:
            todo += list(best)
    return out


def tissue_blocks(pos, *, link_um: float = 60.0, min_cells: int = 100) -> np.ndarray:
    """``[n]`` the tissue block of each cell, numbered top to bottom then left to right; -1 for a dropped cell.

    Cells are binned into squares of ``link_um / 2``; occupied squares are joined to their 8 neighbours after a one-square
    dilation, so gaps in the tissue narrower than ``link_um`` do not split a piece and pieces further apart do not merge. A
    piece that is not a rectangle (two fields touching at a corner) is cut into rectangles (:func:`rectangles`), so a tile
    drawn around a block holds tissue and not empty slide. Blocks under ``min_cells`` cells (stray cells) are dropped.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    step = max(float(link_um) / 2.0, 1e-9)
    low = pos.min(0)
    at = np.floor((pos - low) / step).astype(np.int64)
    occupied = np.zeros(at.max(0) + 1, dtype=bool)
    occupied[at[:, 0], at[:, 1]] = True
    joined, _ = ndimage.label(ndimage.binary_dilation(occupied, structure=np.ones((3, 3), dtype=bool)), np.ones((3, 3)))
    piece = joined[at[:, 0], at[:, 1]].astype(np.int64) - 1
    block = np.full(len(pos), -1, dtype=np.int64)
    count = 0
    for p in np.unique(piece):
        member = np.flatnonzero(piece == p)
        corner = at[member].min(0)
        local = np.zeros(at[member].max(0) - corner + 1, dtype=bool)
        local[at[member, 0] - corner[0], at[member, 1] - corner[1]] = True
        for x0, x1, y0, y1 in rectangles(local):
            inside = (
                (at[member, 0] - corner[0] >= x0)
                & (at[member, 0] - corner[0] < x1)
                & (at[member, 1] - corner[1] >= y0)
                & (at[member, 1] - corner[1] < y1)
            )
            block[member[inside]] = count
            count += 1
    ids, sizes = np.unique(block[block >= 0], return_counts=True)
    ids = ids[sizes >= min_cells]
    if not len(ids):
        return np.full(len(pos), -1, dtype=np.int64)
    centre = np.array([pos[block == b].mean(0) for b in ids])
    band = np.round(-centre[:, 1] / (4.0 * link_um))  # larger y first: the plots keep the slide's axes
    order = np.lexsort((centre[:, 0], band))
    rank = np.full(count + 1, -1, dtype=np.int64)
    rank[ids[order]] = np.arange(len(ids))
    return rank[block]


def tile_layout(sizes_um, *, width_in: float = 16.0, tile_in: float = 3.8, gap_in: float = 0.12):
    """``(boxes [B, 4], scale, height_in)``: the tiles of boxes ``sizes_um`` ``[B, 2]`` (width, height in µm) packed in rows.

    One scale (inches per µm) for every tile: the median box is ``tile_in`` wide, and the widest never exceeds ``width_in``.
    ``boxes[k] = (left, top, width, height)`` in inches from the top left corner; a row wraps when the next tile does not fit.
    """
    sizes = np.asarray(sizes_um, dtype=np.float64)
    scale = min(tile_in / float(np.median(sizes[:, 0])), (width_in - gap_in) / float(sizes[:, 0].max()))
    w, h = sizes[:, 0] * scale, sizes[:, 1] * scale
    boxes, x, y, row = [], 0.0, 0.0, 0.0
    for k in range(len(w)):
        if x > 0 and x + w[k] > width_in + 1e-9:
            x, y, row = 0.0, y + row + gap_in, 0.0
        boxes.append((x, y, w[k], h[k]))
        x += w[k] + gap_in
        row = max(row, h[k])
    return np.asarray(boxes), float(scale), float(y + row)


def smoothed_field(data, bandwidth_um: float = 30.0, min_density: float = 0.3):
    """``(U, V, tissue)`` ``[nx, ny]`` from a slide's ``flow_*`` arrays: the net current per cell, Gaussian-smoothed.

    ``U = smooth(ux) / smooth(cells)``; ``tissue`` marks the bins whose smoothed cell count is at least ``min_density`` of the
    mean over occupied bins (outside it the field is 0 and no stream is drawn).
    """
    grid = float(data["flow_grid_um"])
    sigma = max(float(bandwidth_um) / grid, 1e-3)
    ux, uy, cells = (
        ndimage.gaussian_filter(np.asarray(data[k], dtype=np.float64), sigma)
        for k in ("flow_ux", "flow_uy", "flow_cells")
    )
    occupied = np.asarray(data["flow_cells"]) > 0
    tissue = cells >= float(min_density) * cells[occupied].mean()
    safe = np.where(tissue, cells, 1.0)
    return np.where(tissue, ux / safe, 0.0), np.where(tissue, uy / safe, 0.0), tissue


def smoothed_net(data, bandwidth_um: float = 30.0, min_density: float = 0.3):
    """``(net_grid [nx, ny], net_cell [n], tissue)``: each cell's ``flow_out - flow_in`` (the divergence of the net flow)
    Gaussian-smoothed per cell on the slide's grid, and read back at every cell. Positive: a source, negative: a sink.
    """
    if "flow_out" not in data:
        raise ValueError("this npz has no flow_out / flow_in: write it again with flow_maps.flow_arrays.")
    grid = float(data["flow_grid_um"])
    origin = np.asarray(data["flow_origin"], dtype=np.float64)
    pos = np.asarray(data["pos_um"], dtype=np.float64)[:, :2]
    shape = np.asarray(data["flow_cells"]).shape
    at = np.clip(np.floor((pos - origin) / grid).astype(np.int64), 0, np.array(shape) - 1)
    net = np.asarray(data["flow_out"], dtype=np.float64) - np.asarray(data["flow_in"], dtype=np.float64)
    deposit = np.zeros(shape)
    np.add.at(deposit, (at[:, 0], at[:, 1]), net)
    sigma = max(float(bandwidth_um) / grid, 1e-3)
    smooth, cells = (
        ndimage.gaussian_filter(deposit, sigma),
        ndimage.gaussian_filter(np.asarray(data["flow_cells"], dtype=np.float64), sigma),
    )
    occupied = np.asarray(data["flow_cells"]) > 0
    tissue = cells >= float(min_density) * cells[occupied].mean()
    value = np.where(tissue, smooth / np.where(tissue, cells, 1.0), 0.0)
    return value, value[at[:, 0], at[:, 1]], tissue


class _Tiles:
    """One slide's figure: a tile per tissue block at one scale (only tissue), a header with a title, a caption, a colour bar
    and a 100 µm bar. ``axes[k]`` draws block ``k``; ``members[k]`` its cells.
    """

    def __init__(self, data, *, link_um, min_cells, width_in, tile_in, dpi):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.pos = np.asarray(data["pos_um"], dtype=np.float64)[:, :2]
        self.hidden = np.asarray(data["visible"]) == 0
        block = tissue_blocks(self.pos, link_um=link_um, min_cells=min_cells)
        ids = np.unique(block[block >= 0])
        if not len(ids):
            raise ValueError(f"no tissue block of at least {min_cells} cells to draw.")
        margin = 0.5 * float(link_um)
        self.members = [block == b for b in ids]
        self.boxes_um = [(self.pos[m].min(0) - margin, self.pos[m].max(0) + margin) for m in self.members]
        boxes, self.scale, height = tile_layout(
            np.array([h - lo for lo, h in self.boxes_um]), width_in=width_in, tile_in=tile_in
        )
        self.header = 0.95
        self.total = height + self.header + 0.06
        self.width = max(float((boxes[:, 0] + boxes[:, 2]).max()), 9.0)
        self.fig = plt.figure(figsize=(self.width, self.total), dpi=dpi, facecolor="white")
        self.dpi = dpi
        self.axes = []
        for k, (low, high) in enumerate(self.boxes_um):
            left, top, w, h = boxes[k]
            ax = self.fig.add_axes(
                [left / self.width, 1 - (self.header + top + h) / self.total, w / self.width, h / self.total]
            )
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.set_facecolor("#fbfbfb")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#dddddd")
                spine.set_linewidth(0.6)
            self.axes.append(ax)
        self.drawn = int((block >= 0).sum())
        self.dot = max(1.5, (8.0 * self.scale * 72.0) ** 2)  # a dot of 8 µm: points^2 at this scale

    def label(self, k, extra=""):
        ax = self.axes[k]
        low, high = self.boxes_um[k]
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.text(
            0.012,
            0.985,
            f"{k + 1} · {int(self.members[k].sum()):,} cells{extra}",
            transform=ax.transAxes,
            fontsize=7,
            color="#444444",
            va="top",
            zorder=6,
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 1.5, "linewidth": 0},
        )

    def finish(self, out_path, title, caption, mappable, bar_label):
        from matplotlib.lines import Line2D

        fig, total, width = self.fig, self.total, self.width
        fig.text(0.004, 1 - 0.12 / total, title, fontsize=11, weight="bold", va="top", ha="left")
        fig.text(0.004, 1 - 0.38 / total, caption, fontsize=8, va="top", ha="left", color="#444444")
        right = (width - 3.1) / width
        cax = fig.add_axes([right, 1 - 0.30 / total, 2.9 / width, 0.08 / total])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(bar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        bar = 100.0 * self.scale
        fig.add_artist(Line2D([right, right + bar / width], [1 - 0.80 / total] * 2, color="#333333", lw=2))
        fig.text(right + bar / width + 0.004, 1 - 0.80 / total, "100 µm", fontsize=7, va="center")
        fig.savefig(out_path, dpi=self.dpi, facecolor="white")
        self.plt.close(fig)
        return {"blocks": len(self.axes), "cells": self.drawn, "dropped": int(len(self.pos) - self.drawn)}


#: Sinks blue, sources red, nothing grey: the source-sink colour map.
def _diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("source_sink", ["#2166ac", "#92c5de", "#e3e3e3", "#f4a582", "#b2182b"])


def plot_attention_flow(
    source,
    out_path,
    *,
    title: str = "",
    bandwidth_um: float = 30.0,
    cells: bool = True,
    shade: bool = True,
    link_um: float = 60.0,
    min_cells: int = 100,
    width_in: float = 16.0,
    tile_in: float = 3.8,
    density: float = 0.9,
    min_flow: float = 0.1,
    dpi: int = 120,
) -> dict:
    """One figure of one slide: a tile per tissue block, the cells and the streams of the net attention flow.

    ``source``: the npz :func:`flow_arrays` wrote (path) or its arrays. All tiles share one scale, so a stream's width and
    colour (the net current per cell, one colour bar) compare across tiles, and nothing but tissue is drawn. ``cells``: draw
    the cells (grey; hidden ones tinted) under the streams, else the streams alone; ``shade``: the smoothed source-sink value
    (:func:`smoothed_net`) behind them, sources red and sinks blue (needs ``flow_out``; skipped for an older npz); ``min_flow``:
    no stream where the net current is under that share of the colour scale's end (a weak field is noise, drawn as tangle).
    Returns ``{"blocks": B, "cells": drawn, "dropped": cells in dropped blocks}``.
    """
    from matplotlib.colors import Normalize

    data = np.load(source) if isinstance(source, str | Path) else source
    t = _Tiles(data, link_um=link_um, min_cells=min_cells, width_in=width_in, tile_in=tile_in, dpi=dpi)
    U, V, tissue = smoothed_field(data, bandwidth_um)
    grid, origin = float(data["flow_grid_um"]), np.asarray(data["flow_origin"], dtype=np.float64)
    magnitude = np.hypot(U, V)
    shape = np.array(tissue.shape) - 1
    windows = []
    inside = np.zeros_like(tissue)
    for low, high in t.boxes_um:
        lo = np.clip(np.floor((low - origin) / grid).astype(int), 0, shape)
        hi = np.clip(np.ceil((high - origin) / grid).astype(int), 0, shape)
        windows.append((lo, hi))
        inside[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1] = True
    reference = float(np.quantile(magnitude[tissue & inside], 0.99)) if (tissue & inside).any() else 0.0
    reference = reference if reference > 0 else 1.0
    norm = Normalize(0.0, reference)
    net = None
    if shade and "flow_out" in data:
        net, _, _ = smoothed_net(data, bandwidth_um)
        limit = float(np.quantile(np.abs(net[tissue & inside]), 0.99)) if (tissue & inside).any() else 0.0
        net_norm = Normalize(-(limit or 1.0), limit or 1.0)
    grey, tint, stream_map = "#cfcfcf", "#f0a58a", t.plt.get_cmap("viridis")
    for k, ax in enumerate(t.axes):
        mine, hidden, pos = t.members[k], t.hidden, t.pos
        lo, hi = windows[k]
        sl = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1))
        xs = origin[0] + (np.arange(lo[0], hi[0] + 1) + 0.5) * grid
        ys = origin[1] + (np.arange(lo[1], hi[1] + 1) + 0.5) * grid
        if net is not None:
            ax.imshow(
                np.ma.array(net[sl].T, mask=~tissue[sl].T),
                origin="lower",
                extent=(xs[0] - grid / 2, xs[-1] + grid / 2, ys[0] - grid / 2, ys[-1] + grid / 2),
                cmap=_diverging(),
                norm=net_norm,
                interpolation="bilinear",
                zorder=0,
                aspect="auto",
            )
        if cells:
            alpha = 0.45 if net is not None else 1.0
            ax.scatter(
                pos[mine & ~hidden, 0],
                pos[mine & ~hidden, 1],
                s=t.dot,
                c=grey,
                alpha=alpha,
                marker="o",
                linewidths=0,
                zorder=1,
            )
            ax.scatter(
                pos[mine & hidden, 0],
                pos[mine & hidden, 1],
                s=t.dot,
                c=tint,
                alpha=alpha,
                marker="o",
                linewidths=0,
                zorder=2,
            )
        mask = ~tissue[sl].T | (magnitude[sl].T < float(min_flow) * reference)
        m = np.ma.array(magnitude[sl].T, mask=mask)
        if m.count() and float(m.max()) > 0 and len(xs) > 2 and len(ys) > 2:
            strength = np.ma.filled(m, 0.0)
            ax.streamplot(
                xs,
                ys,
                np.ma.array(U[sl].T, mask=mask),
                np.ma.array(V[sl].T, mask=mask),
                color=strength,
                cmap=stream_map,
                norm=norm,
                linewidth=0.4 + 1.8 * np.clip(strength / reference, 0, 1),
                density=density,
                arrowsize=0.9,
                zorder=3,
            )
        t.label(k)
    caption = (
        "Net attention flow, last layer, head mean: A - A^T over cell pairs. Streams point from the cells being read to the cells that read them; "
        "width and colour: net current per cell.\n"
        + (
            "Behind them the source-sink value (out - in, smoothed): red sources, blue sinks. "
            if net is not None
            else ""
        )
        + f"Hidden cells (tinted) are what the model was asked about; {t.drawn:,} of {len(t.pos):,} cells in {len(t.axes)} blocks."
    )
    return t.finish(
        out_path, title, caption, t.plt.cm.ScalarMappable(norm=norm, cmap=stream_map), "net current per cell (a.u.)"
    )


def plot_source_sink(
    source,
    out_path,
    *,
    title: str = "",
    bandwidth_um: float = 30.0,
    hubs: float = 0.01,
    link_um: float = 60.0,
    min_cells: int = 100,
    width_in: float = 16.0,
    tile_in: float = 3.8,
    dpi: int = 120,
) -> dict:
    """The source-sink map of one slide: every cell coloured by its smoothed ``out - in`` (:func:`smoothed_net`), red where it
    is read more than it reads (a source), blue where it reads more (a sink), on one symmetric scale for the slide; the
    ``hubs`` share of cells with the largest |value| ringed (sources dark red, sinks dark blue), hidden cells marked by a dot.
    """
    from matplotlib.colors import Normalize

    data = np.load(source) if isinstance(source, str | Path) else source
    t = _Tiles(data, link_um=link_um, min_cells=min_cells, width_in=width_in, tile_in=tile_in, dpi=dpi)
    _, value, _ = smoothed_net(data, bandwidth_um)
    drawn = np.any(t.members, axis=0)
    limit = float(np.quantile(np.abs(value[drawn]), 0.99)) if drawn.any() else 0.0
    norm = Normalize(-(limit or 1.0), limit or 1.0)
    cut = float(np.quantile(np.abs(value[drawn]), 1.0 - float(hubs))) if drawn.any() and hubs > 0 else np.inf
    cmap = _diverging()
    for k, ax in enumerate(t.axes):
        mine = t.members[k]
        order = np.flatnonzero(mine)[np.argsort(np.abs(value[mine]))]  # strongest last, on top
        ax.scatter(
            t.pos[order, 0],
            t.pos[order, 1],
            s=t.dot,
            c=value[order],
            cmap=cmap,
            norm=norm,
            marker="o",
            linewidths=0,
            zorder=1,
        )
        hid = mine & t.hidden
        ax.scatter(
            t.pos[hid, 0], t.pos[hid, 1], s=max(0.3, t.dot * 0.08), c="#333333", marker="o", linewidths=0, zorder=2
        )
        for sign, colour in ((1, "#67001f"), (-1, "#053061")):
            hub = mine & (np.abs(value) >= cut) & (np.sign(value) == sign)
            ax.scatter(
                t.pos[hub, 0],
                t.pos[hub, 1],
                s=t.dot * 4,
                facecolors="none",
                edgecolors=colour,
                linewidths=0.7,
                zorder=3,
            )
        t.label(k)
    caption = (
        "Sources and sinks of the net attention flow (last layer, head mean): each cell's out - in (how much it is read minus how much it reads), "
        f"smoothed over {bandwidth_um:g} µm.\nRed: sources, blue: sinks; ringed: the {hubs:.0%} strongest. Black dots: hidden cells, the receivers by "
        f"construction, so sinks follow the mask -- compare with the untrained twin. {t.drawn:,} cells in {len(t.axes)} blocks."
    )
    return t.finish(
        out_path,
        title,
        caption,
        t.plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        "out - in per cell (a.u.): sink  <-  ->  source",
    )


def plot_who_whom(
    source,
    out_path,
    *,
    title: str = "",
    panels: int = 8,
    view_um: float = 260.0,
    min_hidden: int = 20,
    columns: int = 4,
    panel_in: float = 3.6,
    dpi: int = 130,
) -> dict:
    """Who influences whom: zoom panels of ``view_um`` on the ``panels`` largest patches of hidden cells, each sampled receiver
    (:func:`sample_pairs`) joined by arrows from its top sources, width and colour by the pair's net flow (one scale for the
    figure). Cells grey, hidden ones tinted. Returns ``{"panels": drawn, "pairs": arrows drawn}``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    data = np.load(source) if isinstance(source, str | Path) else source
    if "pair_receiver" not in data:
        raise ValueError("this npz has no sampled pairs: write it again with flow_maps.flow_arrays.")
    pos = np.asarray(data["pos_um"], dtype=np.float64)[:, :2]
    hidden = np.asarray(data["visible"]) == 0
    receivers, sources, flows = data["pair_receiver"], data["pair_source"], data["pair_flow"]
    patch = np.full(len(pos), -1, dtype=np.int64)
    if hidden.sum() >= min_hidden:
        patch[hidden] = tissue_blocks(pos[hidden], link_um=25.0, min_cells=min_hidden)
    ids, counts = np.unique(patch[patch >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")][: int(panels)]
    centres = [pos[patch == p].mean(0) for p in order]
    if not centres:  # nothing hidden: the receivers with the strongest inflow
        strongest = np.argsort(-flows.sum(1))[: int(panels)]
        centres = [pos[receivers[i]] for i in strongest]
    valid = sources >= 0
    reference = float(np.quantile(flows[valid], 0.99)) if valid.any() else 1.0
    norm = Normalize(0.0, reference or 1.0)
    cmap = plt.get_cmap("magma_r")
    rows = int(np.ceil(len(centres) / columns)) or 1
    header = 0.85
    fig = plt.figure(figsize=(columns * panel_in, rows * panel_in + header), dpi=dpi, facecolor="white")
    total_h = rows * panel_in + header
    half = float(view_um) / 2.0
    dot = max(2.0, (8.0 * panel_in / view_um * 72.0) ** 2)
    drawn = 0
    for k, centre in enumerate(centres):
        r, c = divmod(k, columns)
        ax = fig.add_axes(
            [
                c / columns + 0.004,
                1 - (header + (r + 1) * panel_in) / total_h + 0.004,
                1 / columns - 0.008,
                panel_in / total_h - 0.008,
            ]
        )
        low, high = centre - half, centre + half
        near = np.all((pos >= low) & (pos <= high), axis=1)
        ax.scatter(pos[near & ~hidden, 0], pos[near & ~hidden, 1], s=dot, c="#d0d0d0", linewidths=0, zorder=1)
        ax.scatter(pos[near & hidden, 0], pos[near & hidden, 1], s=dot, c="#f0a58a", linewidths=0, zorder=2)
        here = near[receivers]
        src, f = sources[here], flows[here]
        rec = np.repeat(receivers[here][:, None], sources.shape[1], axis=1)
        ok = src >= 0
        start, end, f = pos[src[ok]], pos[rec[ok]], f[ok]
        strongest = np.argsort(f)  # weakest first, strongest on top
        start, end, f = start[strongest], end[strongest], f[strongest]
        if len(f):
            ax.quiver(
                start[:, 0],
                start[:, 1],
                end[:, 0] - start[:, 0],
                end[:, 1] - start[:, 1],
                f,
                cmap=cmap,
                norm=norm,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.004 + 0.008 * np.clip(f / (reference or 1.0), 0, 1).mean(),
                headwidth=3.5,
                headlength=4.0,
                headaxislength=3.5,
                zorder=3,
            )
            ax.scatter(end[:, 0], end[:, 1], s=dot * 0.6, c="#222222", linewidths=0, zorder=4)
        drawn += len(f)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#fbfbfb")
        ax.text(
            0.02,
            0.97,
            f"{k + 1} · {int(here.sum())} receivers",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="#444444",
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 1.5, "linewidth": 0},
            zorder=5,
        )
    fig.text(0.004, 1 - 0.12 / total_h, title, fontsize=11, weight="bold", va="top", ha="left")
    fig.text(
        0.004,
        1 - 0.38 / total_h,
        f"Who influences whom (net attention flow, last layer, head mean): {view_um:g} µm around the largest patches of hidden cells. Arrows run from each sampled\n"
        "receiver's top-5 sources (where it reads more than it is read) to the receiver (black dot); colour: the pair's net flow. Hidden cells tinted.",
        fontsize=8,
        va="top",
        ha="left",
        color="#444444",
    )
    cax = fig.add_axes([1 - 3.1 / (columns * panel_in), 1 - 0.30 / total_h, 2.9 / (columns * panel_in), 0.08 / total_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("net flow of a pair (a.u.)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return {"panels": len(centres), "pairs": int(drawn)}
