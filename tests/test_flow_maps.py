"""Net attention flow on a slide (``evaluation/flow_maps.py``; roadmap.md, task 5).

Pinned here, each against a value worked out by hand:

* **the matrix** -- the head mean over the cell keys of a row; CLS, padding, out-of-range ids and a cell's own key are left out;
* **the net flow** -- ``F_ij = A_ij - A_ji``, a pair's positive entry goes from the cell read less (the source) to the cell
  that reads it more (the receiver); a pair known in one row only counts only without ``mutual``, and the mutual share says how
  much attention mass that is;
* **the field** -- the deposits conserve the flow and sit at the pair's midpoint; a planted asymmetry (every cell reads its left
  neighbour more) gives a field along +x of the planted size; the smoothing divides by the cell count and leaves empty bins out;
* **the numbers** -- the share into hidden cells, from visible to hidden, the length;
* **the tissue** -- blocks by gaps, a corner-touching pair cut into two rectangles, a notch not cut, stray cells dropped, numbering top
  to bottom; the tiles one scale, inside the figure, not overlapping, no tile wider than the figure;
* **through the model** -- dense attention and routing over every parent give one flow, ``hierarchy_readouts(flow_dir=...)`` writes
  the arrays and the ``hier_flow`` table without drawing on the RNG, and the figure is drawn from them.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.sparse import csr_matrix

from interscale.evaluation.flow_maps import (
    attention_matrix,
    flow_arrays,
    flow_grid,
    flow_summary,
    net_flow_pairs,
    plot_attention_flow,
    rectangles,
    smoothed_field,
    tile_layout,
    tissue_blocks,
)

SPACING = 10.0


def _sheet(nx, ny, x0=0.0, y0=0.0, step=SPACING):
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.stack([x0 + ii.ravel() * step, y0 + jj.ravel() * step], 1).astype(np.float64)


def _left_reads_more(pos, left=0.5, right=0.1, step=SPACING):
    """Rows over a lattice of ``step``: every cell reads its left neighbour with ``left`` and its right one with ``right``."""
    n = len(pos)
    at = {tuple(np.round(p / step).astype(int)): i for i, p in enumerate(pos)}
    keys = -np.ones((n, 2), dtype=np.int64)
    weights = np.zeros((1, n, 2), dtype=np.float32)
    for (x, y), i in at.items():
        if (x - 1, y) in at:
            keys[i, 0], weights[0, i, 0] = at[(x - 1, y)], left
        if (x + 1, y) in at:
            keys[i, 1], weights[0, i, 1] = at[(x + 1, y)], right
    return torch.as_tensor(keys), torch.as_tensor(weights)


# --------------------------------------------------------------------------- the matrix and the net flow


def test_the_matrix_is_the_head_mean_over_cell_keys_only():
    keys = torch.tensor(
        [[1, 4, -1], [0, 2, 1], [3, -1, -1], [2, 0, 5]]
    )  # 4 is CLS, -1 padding, row 1's key 1 is its own, 5 out of range
    weights = torch.tensor(
        [
            [[0.2, 0.5, 0.0], [0.1, 0.3, 0.4], [0.6, 0.0, 0.0], [0.3, 0.2, 0.1]],
            [[0.4, 0.3, 0.0], [0.3, 0.1, 0.2], [0.2, 0.0, 0.0], [0.1, 0.4, 0.3]],
        ]
    )
    A = attention_matrix(keys, weights, 4).toarray()
    expected = np.array([[0, 0.3, 0, 0], [0.2, 0, 0.2, 0], [0, 0, 0, 0.4], [0.3, 0, 0.2, 0]])
    assert np.allclose(A, expected)
    twice = attention_matrix(
        torch.tensor([[1, 1], [-1, -1]]), torch.tensor([[[0.2, 0.3], [0.0, 0.0]]]), 2
    ).toarray()  # a key listed twice adds up
    assert np.allclose(twice, [[0, 0.5], [0, 0]])


def test_a_pair_flows_from_the_cell_read_less_to_the_one_reading_more_and_mutual_pairs_only_count_by_default():
    A = np.zeros((3, 3), dtype=np.float32)
    A[0, 1], A[1, 0] = 0.5, 0.1  # 0 reads 1 more than 1 reads 0: 1 -> 0, net 0.4
    A[1, 2], A[2, 1] = 0.3, 0.3  # equal: no net flow
    A[0, 2] = 0.2  # known in row 0 only
    source, receiver, flow, share = net_flow_pairs(csr_matrix(A))
    assert (source.tolist(), receiver.tolist()) == ([1], [0]) and np.allclose(flow, [0.4])
    assert np.isclose(share, 1.2 / 1.4)  # the mass on the pairs both rows know
    source, receiver, flow, _ = net_flow_pairs(csr_matrix(A), mutual=False)
    pairs = sorted(zip(source.tolist(), receiver.tolist(), flow.tolist(), strict=True))
    assert [(a, b) for a, b, _ in pairs] == [(1, 0), (2, 0)] and np.allclose([f for *_, f in pairs], [0.4, 0.2])


def test_the_net_flow_is_the_antisymmetric_part_of_a_random_mutual_matrix():
    rng = np.random.default_rng(0)
    n = 30
    dense = rng.random((n, n)) * (rng.random((n, n)) < 0.3)
    dense = dense * (dense > 0).T  # mutual: a pair is in both rows or in neither
    np.fill_diagonal(dense, 0)
    source, receiver, flow, share = net_flow_pairs(csr_matrix(dense.astype(np.float32)))
    assert np.isclose(share, 1.0, atol=1e-6)
    brute = np.maximum(dense - dense.T, 0)  # brute[i, j] > 0: i reads j more, the flow goes j -> i
    found = np.zeros((n, n))
    found[receiver, source] = flow
    assert np.allclose(found, brute, atol=1e-6)
    assert (found * found.T == 0).all()  # a pair carries one direction only


# --------------------------------------------------------------------------- the field


def test_the_deposit_conserves_the_flow_and_sits_at_the_midpoint():
    pos = np.array([[0.0, 0.0], [20.0, 0.0], [100.0, 100.0]])
    grid = flow_grid(pos, np.array([0]), np.array([1]), np.array([0.7], dtype=np.float32), grid_um=10.0)
    assert np.isclose(grid["flow_mass"].sum(), 0.7, atol=1e-6) and grid["flow_cells"].sum() == 3
    at = np.argwhere(grid["flow_mass"] > 0)
    assert at.shape == (1, 2)
    midpoint = np.floor((np.array([10.0, 0.0]) - grid["flow_origin"]) / 10.0).astype(int)
    assert tuple(at[0]) == tuple(midpoint)  # the bin of the midpoint (10, 0), not of either end
    assert np.isclose(grid["flow_ux"][tuple(at[0])], 0.7, atol=1e-6) and np.isclose(grid["flow_uy"][tuple(at[0])], 0.0)
    down = flow_grid(
        pos, np.array([1]), np.array([0]), np.array([0.7], dtype=np.float32)
    )  # the flow reversed: towards -x
    assert np.isclose(down["flow_ux"].sum(), -0.7, atol=1e-6)
    empty = flow_grid(pos, np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    assert empty["flow_mass"].sum() == 0 and empty["flow_cells"].sum() == 3


def test_a_planted_asymmetry_gives_a_field_along_plus_x_of_the_planted_size():
    pos = _sheet(40, 20)
    keys, weights = _left_reads_more(pos, left=0.5, right=0.1)
    arrays, summary = flow_arrays(keys, weights, pos, np.ones(len(pos)), grid_um=10.0)
    assert np.isclose(summary["mutual_share"], 1.0) and np.isclose(summary["length_um"], SPACING)
    U, V, tissue = smoothed_field(arrays, bandwidth_um=30.0)
    inner = np.zeros_like(tissue)
    inner[8:-8, 8:-8] = True
    inner &= tissue
    assert inner.sum() > 100
    assert (
        np.allclose(U[inner], 0.4, atol=0.03) and np.abs(V[inner]).max() < 0.03
    )  # 0.5 - 0.1 = 0.4 per pair, one pair per cell
    # the other way round: reading the right neighbour more sends the flow towards -x
    keys, weights = _left_reads_more(pos, left=0.1, right=0.5)
    arrays, _ = flow_arrays(keys, weights, pos, np.ones(len(pos)))
    U, _, tissue = smoothed_field(arrays)
    assert U[inner & tissue].mean() < -0.35


def test_the_current_is_per_cell_not_per_area():
    """Cells 5 um apart put four cells in a 10 um bin, each with its pair of 0.4: the field is still 0.4, not 1.6."""
    pos = _sheet(80, 40, step=5.0)
    arrays, _ = flow_arrays(*_left_reads_more(pos, step=5.0), pos, np.ones(len(pos)), grid_um=10.0)
    assert arrays["flow_cells"].max() >= 4
    U, _, tissue = smoothed_field(arrays, bandwidth_um=30.0)
    inner = np.zeros_like(tissue)
    inner[8:-8, 8:-8] = True
    assert np.allclose(U[inner & tissue], 0.4, atol=0.03)


def test_the_smoothing_is_a_current_per_cell_and_leaves_empty_bins_out():
    pos = np.concatenate([_sheet(20, 20), _sheet(20, 20, x0=1000.0)])
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, np.ones(len(pos)))
    U, V, tissue = smoothed_field(arrays)
    assert not tissue[
        np.floor((np.array([500.0, 100.0]) - arrays["flow_origin"]) / 10.0).astype(int)[0], 10
    ]  # the gap is not tissue
    assert (U[~tissue] == 0).all() and (V[~tissue] == 0).all()
    # a field twice as dense in cells (cells 5 um apart would double the counts) does not double the per-cell current
    far = smoothed_field(arrays, bandwidth_um=60.0)[0]
    assert np.isclose(far[tissue].max(), 0.4, atol=0.05)


def test_the_summary_numbers():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0], [60.0, 0.0]])
    visible = np.array([1, 1, 0, 0])
    source, receiver, flow = np.array([0, 1, 3]), np.array([1, 2, 2]), np.array([0.4, 0.2, 0.4])
    summary = flow_summary(pos, visible, source, receiver, flow, mutual_share=0.9)
    assert np.isclose(summary["net_flow_per_cell"], 1.0 / 4)
    assert np.isclose(
        summary["to_hidden"], 0.6 / 1.0
    )  # receivers 1, 2, 2: the flows into cell 2 (hidden) are 0.2 + 0.4
    assert np.isclose(summary["visible_to_hidden"], 0.2)  # 1 -> 2 only (3 is hidden)
    assert np.isclose(summary["hidden_cells"], 0.5) and np.isclose(summary["mutual_share"], 0.9)
    assert np.isclose(summary["length_um"], (0.4 * 10 + 0.2 * 20 + 0.4 * 30) / 1.0)
    none = flow_summary(pos, visible, np.array([], dtype=int), np.array([], dtype=int), np.array([]))
    assert none["net_flow_per_cell"] == 0 and np.isnan(none["to_hidden"])


# --------------------------------------------------------------------------- the tissue and the tiles


def test_blocks_follow_the_gaps_and_a_narrow_gap_does_not_split_one():
    top = _sheet(30, 20, x0=0.0, y0=600.0)
    bottom = _sheet(30, 20, x0=0.0, y0=0.0)
    pos = np.concatenate([bottom, top])
    block = tissue_blocks(pos, link_um=60.0, min_cells=100)
    assert set(block) == {0, 1}
    assert (block[len(bottom) :] == 0).all() and (block[: len(bottom)] == 1).all()  # larger y first
    split = np.concatenate([_sheet(15, 20), _sheet(15, 20, x0=15 * SPACING + 30.0)])  # a 30 um gap inside one field
    assert set(tissue_blocks(split, link_um=60.0, min_cells=100)) == {0}


def test_stray_cells_are_dropped_and_a_block_is_numbered_left_to_right_within_a_row():
    right, left = _sheet(25, 20, x0=800.0), _sheet(25, 20, x0=0.0)
    stray = np.array([[400.0, 1500.0], [410.0, 1500.0]])
    pos = np.concatenate([right, left, stray])
    block = tissue_blocks(pos, min_cells=100)
    assert (
        (block[-2:] == -1).all()
        and (block[len(right) : len(right) + len(left)] == 0).all()
        and (block[: len(right)] == 1).all()
    )


def test_two_fields_touching_at_a_corner_are_cut_into_rectangles_and_a_notch_is_not():
    first, second = (
        _sheet(30, 20, x0=0.0, y0=0.0),
        _sheet(30, 20, x0=30 * SPACING, y0=20 * SPACING),
    )  # touch at one corner
    pos = np.concatenate([first, second])
    block = tissue_blocks(pos, link_um=60.0, min_cells=100)
    assert sorted(set(block)) == [0, 1]
    for b in (0, 1):
        mine = pos[block == b]
        box = (mine.max(0) - mine.min(0) + SPACING).prod()
        assert len(mine) * SPACING**2 / box > 0.9  # the tile holds tissue, not slide
    notched = _sheet(30, 20)
    notched = notched[~((notched[:, 0] < 60) & (notched[:, 1] > 150))]  # a corner bitten off, ~4% of the area
    assert set(tissue_blocks(notched, link_um=60.0, min_cells=100)) == {0}
    assert set(tissue_blocks(np.concatenate([_sheet(20, 20), _sheet(20, 20, x0=20 * SPACING)]), min_cells=100)) == {
        0
    }  # a joined pair: one rectangle


def test_rectangles_cut_an_l_and_keep_a_rectangle():
    mask = np.zeros((20, 20), dtype=bool)
    mask[:20, :8] = True
    mask[:8, 8:20] = True  # an L
    boxes = rectangles(mask)
    assert len(boxes) == 2
    assert sum((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in boxes) <= 0.75 * 20 * 20
    covered = np.zeros_like(mask)
    for x0, x1, y0, y1 in boxes:
        covered[x0:x1, y0:y1] = True
    assert (covered | ~mask).all()  # every bin of the tissue is in a box
    assert rectangles(np.ones((12, 9), dtype=bool)) == [(0, 12, 0, 9)]
    assert rectangles(np.zeros((4, 4), dtype=bool)) == []


def test_tiles_share_one_scale_stay_inside_the_figure_and_do_not_overlap():
    sizes = np.array(
        [[1000.0, 650.0], [985.0, 657.0], [1500.0, 700.0], [980.0, 650.0], [1000.0, 640.0], [990.0, 660.0]]
    )
    boxes, scale, height = tile_layout(sizes, width_in=10.0, tile_in=3.0, gap_in=0.1)
    assert np.allclose(boxes[:, 2] / sizes[:, 0], scale) and np.allclose(boxes[:, 3] / sizes[:, 1], scale)  # one scale
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 0] + boxes[:, 2] <= 10.0 + 1e-9).all()
    assert (boxes[:, 1] + boxes[:, 3] <= height + 1e-9).all()
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            apart = (
                boxes[a, 0] + boxes[a, 2] <= boxes[b, 0] + 1e-9
                or boxes[b, 0] + boxes[b, 2] <= boxes[a, 0] + 1e-9
                or boxes[a, 1] + boxes[a, 3] <= boxes[b, 1] + 1e-9
                or boxes[b, 1] + boxes[b, 3] <= boxes[a, 1] + 1e-9
            )
            assert apart
    assert len(set(np.round(boxes[:, 1], 6))) > 1  # the rows wrapped
    wide = tile_layout(np.array([[1000.0, 600.0], [1000.0, 600.0], [8000.0, 600.0]]), width_in=10.0, tile_in=3.0)
    assert (
        wide[0][:, 0] + wide[0][:, 2] <= 10.0 + 1e-9
    ).all()  # the median says 3 in per 1000 um, the 8000 um block would be 24 in: the scale shrinks
    assert wide[1] < 3.0 / 1000.0 / 2


# --------------------------------------------------------------------------- the figure


def test_the_figure_is_drawn_from_the_arrays_one_tile_per_block(tmp_path):
    pos = np.concatenate([_sheet(40, 25), _sheet(40, 25, x0=1500.0, y0=300.0), _sheet(30, 20, x0=0.0, y0=900.0)])
    visible = np.ones(len(pos))
    visible[500:560] = 0
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, visible)
    out = plot_attention_flow(arrays, tmp_path / "flow.png", title="toy", min_cells=50)
    assert out == {"blocks": 3, "cells": len(pos), "dropped": 0} and (tmp_path / "flow.png").stat().st_size > 5000
    np.savez_compressed(tmp_path / "slide.npz", **arrays)
    again = plot_attention_flow(
        tmp_path / "slide.npz", tmp_path / "again.png", cells=False, min_cells=50
    )  # from the file, streams alone
    assert again["blocks"] == 3 and (tmp_path / "again.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="no tissue block"):
        plot_attention_flow(arrays, tmp_path / "none.png", min_cells=10_000)


def test_the_figure_has_an_axes_per_block_around_its_cells_with_the_hidden_ones_tinted(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    pos = np.concatenate([_sheet(40, 25), _sheet(40, 25, x0=1500.0, y0=300.0)])
    visible = np.ones(len(pos))
    visible[100:160] = 0  # 60 hidden cells in the first block
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, visible)
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig=None: seen.append(fig))
    plot_attention_flow(arrays, tmp_path / "f.png", min_cells=50, tile_in=3.0)
    fig = seen[0]
    tiles = [ax for ax in fig.axes if ax.get_xlim()[1] - ax.get_xlim()[0] > 100]  # the colour bar is not a tile
    assert len(tiles) == 2
    for ax, block in zip(
        tiles, (pos[1000:], pos[:1000]), strict=True
    ):  # numbered top to bottom: the sheet at y = 300 first
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        assert (
            x0 <= block[:, 0].min() and x1 >= block[:, 0].max() and y0 <= block[:, 1].min() and y1 >= block[:, 1].max()
        )
        assert (x1 - x0) < (block[:, 0].max() - block[:, 0].min()) + 80  # around the cells, not around the slide
    dots = {
        ax: [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())] for ax in tiles
    }  # the scatters
    tinted = [
        sum(
            len(c.get_offsets())
            for c in dots[ax]
            if np.allclose(c.get_facecolor()[0][:3], (0.941, 0.647, 0.541), atol=0.01)
        )
        for ax in tiles
    ]
    assert tinted == [0, 60]  # only the hidden cells are tinted, in their own tile
    assert sum(len(c.get_offsets()) for ax in tiles for c in dots[ax]) == 2000  # every cell drawn once
    # the tiles do not overlap, in figure coordinates
    boxes = [ax.get_position() for ax in tiles]
    assert boxes[0].x1 <= boxes[1].x0 + 1e-9 or boxes[0].y1 <= boxes[1].y0 + 1e-9 or boxes[1].y1 <= boxes[0].y0 + 1e-9


def test_no_stream_is_drawn_where_the_flow_is_weaker_than_min_flow(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pos = _sheet(40, 25)
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, np.ones(len(pos)))
    counts = {}
    for name, floor in (
        ("shown", 0.1),
        ("none", 2.0),
    ):  # the colour scale ends at the 99th percentile: nothing exceeds twice that
        seen = []
        monkeypatch.setattr(plt, "close", lambda fig=None, seen=seen: seen.append(fig))
        plot_attention_flow(arrays, tmp_path / f"{name}.png", min_cells=50, min_flow=floor)
        tile = max(seen[0].axes, key=lambda ax: ax.get_xlim()[1] - ax.get_xlim()[0])
        counts[name] = sum(isinstance(c, LineCollection) and len(c.get_segments()) > 0 for c in tile.collections)
    assert counts["shown"] > 0 and counts["none"] == 0


def test_the_figure_has_no_blank_slide_between_blocks(tmp_path):
    """Two fields 5 mm apart: the picture is as wide as its tiles, not as the slide."""
    import matplotlib.image as mpimg

    pos = np.concatenate([_sheet(60, 40), _sheet(60, 40, x0=5000.0, y0=3000.0)])
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, np.ones(len(pos)))
    plot_attention_flow(arrays, tmp_path / "wide.png", tile_in=3.0, min_cells=100)
    image = mpimg.imread(tmp_path / "wide.png")
    # the two tiles side by side (~6 in; the header keeps the figure 9 in wide) and ~3.4 in tall, where the slide at the tiles'
    # scale would be ~28 x 17 in
    assert image.shape[1] / 120 <= 9.01 and image.shape[0] / 120 < 5.0


# --------------------------------------------------------------------------- through the model


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "hier_helpers", Path(__file__).with_name("test_hierarchical_attention.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_attention_and_routing_over_every_parent_give_one_flow():
    from interscale.evaluation.hierarchy_readouts import cell_rows

    h = _helpers()
    batch = h._batch((6,))
    base = {"prior_mode": "none", "readout": "concat", "levels": [3, 9]}
    dense = h._stir(h._build(h._config({**base, "candidates.mode": "dense"}))).eval()
    routed = h._build(
        h._config({**base, "candidates.mode": "routed", "candidates.topk": [100], "candidates.cap_per_query": 4096})
    ).eval()
    routed.load_state_dict(dense.state_dict(), strict=False)
    matrices = []
    for module in (dense, routed):
        entry = h._record(module, batch)[-1]
        keys, weights = cell_rows(entry)
        matrices.append(attention_matrix(keys, weights, 36))
    assert matrices[0].nnz > 0 and np.allclose(matrices[0].toarray(), matrices[1].toarray(), atol=1e-6)
    flows = [net_flow_pairs(m) for m in matrices]
    assert np.isclose(flows[0][3], 1.0) and np.allclose(flows[0][2].sum(), flows[1][2].sum(), atol=1e-5)


def test_hierarchy_readouts_writes_the_flow_without_touching_the_rng(tmp_path):
    from interscale.evaluation.hierarchy_readouts import hierarchy_readouts

    h = _helpers()
    graph, other = h._lattice(8, seed=2, programme=False), h._lattice(6, seed=5, programme=False)
    graph.mask[:12], other.mask[:9] = True, True  # cells hidden whole
    graph.obs_names = torch.arange(64) + 1
    module = h._stir(
        h._build(h._config({"prior_mode": "none", "readout": "concat", "levels": [4, 16]}, strategy="node"))
    ).eval()
    state = torch.random.get_rng_state()
    tables = hierarchy_readouts(
        module,
        [graph, other],
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        flow_dir=tmp_path / "flow",
        graph_names=["a", "b"],
    )
    assert torch.equal(torch.random.get_rng_state(), state)
    assert sorted(p.name for p in (tmp_path / "flow").iterdir()) == [
        "a.npz",
        "b.npz",
    ]  # every graph, no export_dir files elsewhere
    arrays = np.load(tmp_path / "flow" / "a.npz")
    assert set(arrays.files) >= {
        "pos_um",
        "visible",
        "flow_ux",
        "flow_uy",
        "flow_mass",
        "flow_cells",
        "flow_origin",
        "flow_grid_um",
    }
    assert arrays["pos_um"].shape == (64, 2) and np.isclose(arrays["flow_cells"].sum(), 64)
    flow = tables["hier_flow"]
    assert list(flow["graph"]) == ["a", "b"] and np.isfinite(flow[["net_flow_per_cell", "length_um"]].to_numpy()).all()
    assert np.allclose(flow["hidden_cells"], [12 / 64, 9 / 36]) and (flow["mutual_share"] > 0).all()
    from torch_geometric.data import Batch

    from interscale.evaluation.hierarchy_readouts import cell_rows

    batch = Batch.from_data_list([graph])
    batch.obs_names = graph.obs_names
    record = h._record(module, batch)
    assert len(record) == 2
    for layer in (record[-1], record[0]):  # the export is the last layer's, and the layers differ
        keys, weights = cell_rows(layer)
        direct, _ = flow_arrays(keys, weights, layer["pos_um"], layer["visible"])
        same = np.allclose(direct["flow_ux"], arrays["flow_ux"], atol=1e-6)
        assert same == (layer is record[-1])
    plot_attention_flow(
        tmp_path / "flow" / "a.npz", tmp_path / "a.png", link_um=250.0, bandwidth_um=150.0, min_cells=10
    )  # 100 um spacing
    assert (tmp_path / "a.png").stat().st_size > 1000
    nothing = hierarchy_readouts(
        module, [graph], prediction_task="regression", prediction_level="node", device=torch.device("cpu")
    )
    assert "hier_flow" not in nothing  # off unless asked


# --------------------------------------------------------------------------- sources, sinks, who influences whom


def _star(n_side=21, radius=50.0):
    """A lattice whose cells within ``radius`` of the centre read the centre cell (0.5) and are read by it 0.1: a planted source."""

    pos = _sheet(n_side, n_side)
    centre = int(np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1)))
    readers = np.flatnonzero((np.linalg.norm(pos - pos[centre], axis=1) <= radius) & (np.arange(len(pos)) != centre))
    keys = -np.ones((len(pos), len(readers)), dtype=np.int64)
    weights = np.zeros((1, len(pos), len(readers)), dtype=np.float32)
    keys[readers, 0], weights[0, readers, 0] = centre, 0.5
    keys[centre], weights[0, centre] = readers, 0.1
    return pos, centre, readers, torch.as_tensor(keys), torch.as_tensor(weights)


def test_out_and_in_strength_are_the_divergence_of_the_net_flow():
    from interscale.evaluation.flow_maps import cell_strengths

    source, receiver, flow = np.array([0, 0, 1, 3]), np.array([1, 2, 2, 2]), np.array([0.4, 0.1, 0.2, 0.3])
    out, into = cell_strengths(4, source, receiver, flow)
    assert np.allclose(out, [0.5, 0.2, 0, 0.3]) and np.allclose(into, [0, 0.4, 0.6, 0])
    assert np.isclose((out - into).sum(), 0.0)  # what leaves one cell enters another
    pos, centre, readers, keys, weights = _star()
    arrays, _ = flow_arrays(keys, weights, pos, np.ones(len(pos)))
    net = arrays["flow_out"] - arrays["flow_in"]
    assert np.isclose(net[centre], 0.4 * len(readers)) and np.allclose(
        net[readers], -0.4
    )  # the planted source, its sinks
    assert np.isclose(net.sum(), 0.0, atol=1e-5)


def test_the_smoothed_source_sink_value_has_the_planted_sign():
    from interscale.evaluation.flow_maps import smoothed_net

    pos, centre, readers, keys, weights = _star(radius=40.0)
    arrays, _ = flow_arrays(keys, weights, pos, np.ones(len(pos)))
    grid, value, tissue = smoothed_net(arrays, bandwidth_um=4.0)
    ring = readers[
        np.linalg.norm(pos[readers] - pos[centre], axis=1) >= 20
    ]  # the centre's own bin bleeds into its first ring
    assert value[centre] > 0 and (value[ring] < 0).all()
    far = np.linalg.norm(pos - pos[centre], axis=1) > 120
    assert np.allclose(value[far], 0.0, atol=1e-6)  # nothing happens away from the star
    old = {k: v for k, v in arrays.items() if k not in ("flow_out", "flow_in")}
    with pytest.raises(ValueError, match="flow_out"):
        smoothed_net(old)


def test_sampled_pairs_take_hidden_receivers_and_their_strongest_sources_first():
    from interscale.evaluation.flow_maps import sample_pairs

    visible = np.array([1, 1, 1, 0, 0, 1])
    source = np.array([0, 1, 2, 0, 2, 0, 1])
    receiver = np.array([3, 3, 3, 4, 4, 5, 5])
    flow = np.array([0.1, 0.5, 0.3, 0.2, 0.7, 0.9, 0.4], dtype=np.float32)
    pairs = sample_pairs(visible, source, receiver, flow, receivers=10, top=2)
    assert pairs["pair_receiver"].tolist() == [
        3,
        4,
    ]  # the hidden cells; cell 5 (visible) is not drawn while hidden ones exist
    assert pairs["pair_source"].tolist() == [[1, 2], [2, 0]] and np.allclose(
        pairs["pair_flow"], [[0.5, 0.3], [0.7, 0.2]]
    )
    one = sample_pairs(visible, source, receiver, flow, receivers=1, top=4)
    assert len(one["pair_receiver"]) == 1 and one["pair_source"].shape == (1, 4) and (one["pair_source"][0, 3] == -1)
    assert np.array_equal(
        sample_pairs(visible, source, receiver, flow, receivers=1)["pair_receiver"], one["pair_receiver"]
    )  # seeded
    none_hidden = sample_pairs(np.ones(6), source, receiver, flow, receivers=10, top=1)
    assert none_hidden["pair_receiver"].tolist() == [3, 4, 5]


def test_the_source_sink_map_and_the_who_whom_panels_are_drawn(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    from interscale.evaluation.flow_maps import plot_source_sink, plot_who_whom

    pos, centre, readers, keys, weights = _star(n_side=30, radius=40.0)
    visible = np.ones(len(pos))
    visible[readers] = 0  # the readers are the hidden cells: one patch
    arrays, _ = flow_arrays(keys, weights, pos, visible)
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig=None: seen.append(fig))
    out = plot_source_sink(arrays, tmp_path / "ss.png", min_cells=50, bandwidth_um=4.0, hubs=0.01)
    assert out["blocks"] == 1 and (tmp_path / "ss.png").stat().st_size > 1000
    tile = max(seen[-1].axes, key=lambda ax: ax.get_xlim()[1] - ax.get_xlim()[0])
    dots = [c for c in tile.collections if isinstance(c, PathCollection)]
    assert sum(len(c.get_offsets()) for c in dots[:1]) == len(pos)  # every cell coloured once
    rings = sum(len(c.get_offsets()) for c in dots[2:])
    assert (
        1 <= rings <= int(np.ceil(0.01 * len(pos))) + 2 and np.isclose(dots[2].get_offsets()[0], pos[centre]).all()
    )  # the source ringed
    panels = plot_who_whom(arrays, tmp_path / "ww.png", panels=8, min_hidden=5)
    assert panels == {"panels": 1, "pairs": len(readers)}  # one hidden patch, every reader's one source drawn
    with pytest.raises(ValueError, match="sampled pairs"):
        plot_who_whom({k: v for k, v in arrays.items() if not k.startswith("pair_")}, tmp_path / "x.png")


def test_the_streams_carry_the_source_sink_shading_unless_switched_off(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    pos = _sheet(40, 25)
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, np.ones(len(pos)))
    images = {}
    for shade in (True, False):
        seen = []
        monkeypatch.setattr(plt, "close", lambda fig=None, seen=seen: seen.append(fig))
        plot_attention_flow(arrays, tmp_path / f"s{shade}.png", min_cells=50, shade=shade)
        images[shade] = sum(len(ax.images) for ax in seen[0].axes)
    assert images == {True: 1, False: 0}
    old = {k: v for k, v in arrays.items() if k not in ("flow_out", "flow_in")}
    plot_attention_flow(
        old, tmp_path / "old.png", min_cells=50
    )  # an npz from before the strengths: no shading, no error


def test_the_sample_is_seeded_and_a_bin_of_several_cells_averages_them(tmp_path):
    from interscale.evaluation.flow_maps import sample_pairs, smoothed_net

    n = 400
    visible = np.zeros(n)
    receiver, source = np.arange(1, n), np.arange(n - 1)
    flow = np.linspace(0.1, 1.0, n - 1).astype(np.float32)
    draws = [sample_pairs(visible, source, receiver, flow, receivers=20)["pair_receiver"] for _ in range(2)]
    assert np.array_equal(*draws) and len(draws[0]) == 20
    assert not np.array_equal(
        draws[0], sample_pairs(visible, source, receiver, flow, receivers=20, seed=1)["pair_receiver"]
    )
    # cells 5 um apart: four to a 10 um bin; with no smoothing a bin's value is the mean of its cells' out - in
    pos = _sheet(20, 20, step=5.0)
    arrays, _ = flow_arrays(*_left_reads_more(pos, step=5.0), pos, np.ones(len(pos)), grid_um=10.0)
    _, value, _ = smoothed_net(arrays, bandwidth_um=1e-6)
    net = arrays["flow_out"] - arrays["flow_in"]
    at = np.floor((pos - arrays["flow_origin"]) / 10.0).astype(int)
    for b in {tuple(a) for a in at[:40]}:
        same = (at == b).all(1)
        assert same.sum() >= 2 and np.allclose(value[same], net[same].mean(), atol=1e-5)


def test_a_who_whom_panel_draws_only_the_receivers_in_its_view(tmp_path):
    from interscale.evaluation.flow_maps import plot_who_whom

    pos = _sheet(80, 30)
    visible = np.ones(len(pos))
    big = (pos[:, 0] < 100) & (pos[:, 1] < 100)  # 100 hidden cells in one corner
    small = (pos[:, 0] > 600) & (pos[:, 0] < 650) & (pos[:, 1] > 100) & (pos[:, 1] < 150)  # 16 hidden cells 500 um away
    visible[big | small] = 0
    arrays, _ = flow_arrays(*_left_reads_more(pos), pos, visible)
    out = plot_who_whom(arrays, tmp_path / "ww.png", panels=1, view_um=260.0, min_hidden=5)
    hidden_receivers = arrays["pair_receiver"]
    in_big = big[hidden_receivers] & (arrays["pair_source"][:, 0] >= 0)
    assert out["panels"] == 1 and out["pairs"] == int(in_big.sum()) and small[hidden_receivers].any()
