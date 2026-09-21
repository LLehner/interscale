"""The ring/core split, and the guards around the radius that defines it.

The failure this file exists for is an annulus that is not where it is supposed to be. A ring that
starts too far in overlaps what the transformer cannot attend to; one that starts too far out
leaves a band nothing optimises. Neither raises, and both produce a loss curve that falls.
"""

import pytest

torch = pytest.importorskip("torch")

from interscale.train.ring_nce import (
    extra_negatives,
    pooled_ring_and_core,
    ring_membership,
    similarity_gap,
    usable_anchors,
)


def path_edges(n):
    """A 0-1-...-(n-1) path, both directions, so hop count is exactly positional distance."""
    src = torch.arange(n - 1)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


# --------------------------------------------------------------------------- the annulus


def test_the_ring_is_the_outer_set_minus_the_core():
    ring, core = ring_membership(torch.tensor([4]), path_edges(9), 9, inner_hops=1, outer_hops=3)

    assert core[0].nonzero().flatten().tolist() == [3, 5]
    assert ring[0].nonzero().flatten().tolist() == [1, 2, 6, 7]


def test_ring_and_core_never_share_a_cell():
    ring, core = ring_membership(torch.arange(9), path_edges(9), 9, inner_hops=2, outer_hops=4)

    assert not (ring & core).any()


def test_the_anchor_is_in_neither_set():
    """It is the thing being compared against both, so its presence in either makes that side
    identifiable by a projection onto the anchor itself."""
    ring, core = ring_membership(torch.arange(9), path_edges(9), 9, inner_hops=1, outer_hops=3)
    diagonal = torch.arange(9)

    assert not ring[diagonal, diagonal].any()
    assert not core[diagonal, diagonal].any()


def test_widening_the_inner_radius_moves_cells_from_ring_to_core():
    narrow_ring, narrow_core = ring_membership(torch.tensor([4]), path_edges(9), 9, 1, 3)
    wide_ring, wide_core = ring_membership(torch.tensor([4]), path_edges(9), 9, 2, 3)

    assert wide_core.sum() > narrow_core.sum()
    assert wide_ring.sum() < narrow_ring.sum()
    # The union is the same set either way: only the boundary between them moved.
    assert torch.equal(narrow_ring | narrow_core, wide_ring | wide_core)


def test_an_outer_radius_inside_the_inner_one_is_rejected():
    """Otherwise the ring is empty for every anchor, every row is dropped, and the term reports a
    zero loss that looks like a converged one."""
    with pytest.raises(ValueError, match="must exceed the inner radius"):
        ring_membership(torch.tensor([0]), path_edges(5), 5, inner_hops=3, outer_hops=2)


def test_a_ring_cannot_cross_into_another_slide():
    """Two disconnected components stand in for two graphs; `edge_index` carries no cross-graph
    edges, so containment is structural rather than a filter that could be forgotten."""
    edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])

    ring, core = ring_membership(torch.tensor([0]), edges, 6, inner_hops=1, outer_hops=5)

    assert set((ring | core)[0].nonzero().flatten().tolist()) <= {1, 2}


# --------------------------------------------------------------------------- usability


def test_an_anchor_with_no_ring_is_dropped():
    """A short path gives the end cell a core but nothing beyond it. Both sides pool to zeros,
    which is an ordinary-looking embedding, so this has to be checked rather than noticed."""
    ring, core = ring_membership(torch.tensor([0]), path_edges(3), 3, inner_hops=2, outer_hops=3)
    _, _, keep = pooled_ring_and_core(ring, core, torch.randn(3, 4))

    assert keep.tolist() == [False]


def test_an_anchor_with_both_sides_is_kept():
    ring, core = ring_membership(torch.tensor([4]), path_edges(9), 9, inner_hops=1, outer_hops=3)
    _, _, keep = pooled_ring_and_core(ring, core, torch.randn(9, 4))

    assert keep.tolist() == [True]


def test_usable_needs_both_sides_not_either():
    assert usable_anchors(torch.tensor([3, 0, 2]), torch.tensor([1, 4, 0])).tolist() == [True, False, False]


# --------------------------------------------------------------------------- extra negatives


def test_an_anchor_never_draws_its_own_core_as_an_extra_negative():
    """It is already negative 0, so drawing it again would put the same vector in the denominator
    twice and quietly halve the weight of every other one."""
    cores = torch.eye(6)

    drawn = extra_negatives(cores, n_extra=3)

    for row in range(6):
        assert not torch.allclose(drawn[row], cores[row].expand(3, 6))


def test_extra_negatives_are_capped_at_the_number_of_other_anchors():
    drawn = extra_negatives(torch.randn(3, 5), n_extra=10)

    assert drawn.shape == (3, 2, 5)


def test_a_single_anchor_yields_no_extra_negatives():
    drawn = extra_negatives(torch.randn(1, 5), n_extra=4)

    assert drawn.shape == (1, 0, 5)


def test_zero_extra_negatives_is_the_requested_one_negative_design():
    drawn = extra_negatives(torch.randn(8, 5), n_extra=0)

    assert drawn.shape == (8, 0, 5)


# --------------------------------------------------------------------------- the diagnostic


def test_the_gap_is_negative_when_the_core_is_the_more_similar_side():
    """Which is the expected starting state: spatial autocorrelation and the GCN aggregate both
    put the anchor closer to its own neighbourhood than to anything further out."""
    anchor = torch.tensor([[1.0, 0.0]])

    gap = similarity_gap(anchor, ring=torch.tensor([[0.0, 1.0]]), core=torch.tensor([[1.0, 0.0]]))

    assert gap.item() == pytest.approx(-1.0)


def test_the_gap_rises_as_the_anchor_turns_toward_its_ring():
    anchor = torch.tensor([[1.0, 0.0]])
    core = torch.tensor([[0.0, 1.0]])

    away = similarity_gap(anchor, torch.tensor([[-1.0, 0.0]]), core)
    toward = similarity_gap(anchor, torch.tensor([[1.0, 0.0]]), core)

    assert toward > away


def test_the_gap_is_scale_invariant():
    anchor, ring, core = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)

    base = similarity_gap(anchor, ring, core)
    scaled = similarity_gap(anchor * 5, ring * 2, core * 9)

    assert torch.allclose(base, scaled, atol=1e-5)
