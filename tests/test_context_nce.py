"""Stage 2: the index arithmetic and the negative-sampling policy.

Everything pinned here is something that fails *silently* -- a wrong k-hop set, a false negative
in the denominator, a composition match that does not match -- and produces a plausible loss
curve while training on the wrong thing. The loss going down is not evidence for any of it.
"""

import pytest

torch = pytest.importorskip("torch")

from interscale.train.context_nce import (
    build_inverse,
    composition,
    histogram_gap,
    info_nce,
    khop_membership,
    match_composition,
    overlaps,
    pool_contexts,
    selection_is_clean,
    to_token_edges,
    valid_contexts,
)


def path_edges(n):
    """A 0-1-2-...-(n-1) path, both directions, so hop counts are exactly positional distance."""
    src = torch.arange(n - 1)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


# --------------------------------------------------------------------------- index arithmetic


def test_inverse_maps_tokens_back_and_marks_dropped_nodes():
    kept = torch.tensor([0, 2, 3])
    inverse = build_inverse(kept, num_nodes=5)

    assert inverse.tolist() == [0, -1, 1, 2, -1]


def test_edges_are_dropped_when_an_endpoint_did_not_survive_padding():
    # Node 1 produced no token, so both edges touching it must go.
    inverse = build_inverse(torch.tensor([0, 2]), num_nodes=3)
    edges = torch.tensor([[0, 1, 0], [1, 2, 2]])

    kept = to_token_edges(edges, inverse)

    assert kept.tolist() == [[0], [1]]


# --------------------------------------------------------------------------- k-hop contexts


def test_one_hop_context_is_the_immediate_neighbours_without_the_centre():
    members = khop_membership(torch.tensor([2]), path_edges(5), n_tokens=5, hops=1)

    assert members[0].nonzero().flatten().tolist() == [1, 3]


def test_two_hop_context_grows_by_exactly_one_step():
    members = khop_membership(torch.tensor([2]), path_edges(5), n_tokens=5, hops=2)

    assert members[0].nonzero().flatten().tolist() == [0, 1, 3, 4]


def test_the_centre_is_never_in_its_own_context():
    """The anchor is what the context must predict; leaving it in makes the positive trivial."""
    members = khop_membership(torch.arange(5), path_edges(5), n_tokens=5, hops=3)

    assert not members[torch.arange(5), torch.arange(5)].any()


def test_propagation_is_symmetric_even_when_edge_index_is_one_directional():
    """`edge_index` need not carry both directions; a one-way walk would give the two endpoints
    of one edge different contexts, which no assertion downstream would catch."""
    one_way = torch.tensor([[0, 1], [1, 2]])

    members = khop_membership(torch.tensor([2]), one_way, n_tokens=3, hops=1)

    assert members[0].nonzero().flatten().tolist() == [1]


def test_an_isolated_cell_has_an_empty_context_and_is_reported_as_invalid():
    edges = torch.tensor([[0], [1]])  # node 2 is isolated
    members = khop_membership(torch.tensor([2]), edges, n_tokens=3, hops=2)
    _, sizes = pool_contexts(members, torch.randn(3, 4))

    assert sizes.tolist() == [0]
    assert not valid_contexts(sizes).any()


# --------------------------------------------------------------------------- pooling


def test_pooling_is_a_uniform_mean_over_the_set_not_a_degree_weighted_one():
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    members = khop_membership(torch.tensor([2]), path_edges(5), n_tokens=5, hops=2)

    pooled, sizes = pool_contexts(members, features)

    assert sizes.tolist() == [4]
    assert pooled.item() == pytest.approx((1 + 2 + 4 + 5) / 4)


def test_an_empty_context_pools_to_zero_rather_than_nan():
    """A NaN would propagate into the loss and be obvious; a zero is a legitimate embedding and
    would quietly train the model to align with the origin. Hence `valid_contexts`."""
    members = torch.zeros(1, 3, dtype=torch.bool)

    pooled, sizes = pool_contexts(members, torch.randn(3, 4))

    assert torch.isfinite(pooled).all()
    assert sizes.tolist() == [0]


# --------------------------------------------------------------------------- composition


def test_histograms_are_proportions_so_context_size_cannot_be_the_discriminator():
    types = torch.tensor([0, 0, 1, 1, 0, 1])
    small = torch.tensor([[True, False, True, False, False, False]])
    large = torch.tensor([[True, True, True, True, False, False]])

    hist = composition(torch.cat([small, large]), types, n_types=2)

    assert torch.allclose(hist[0], hist[1])
    assert torch.allclose(hist.sum(dim=1), torch.ones(2))


def test_overlap_is_detected_from_a_single_shared_cell():
    a = torch.tensor([[True, True, False, False]])
    shares_one = torch.tensor([[False, True, False, False]])
    disjoint = torch.tensor([[False, False, True, True]])

    flags = overlaps(a, torch.cat([shares_one, disjoint]))

    assert flags.tolist() == [[True, False]]


# --------------------------------------------------------------------------- negative selection


def test_matching_picks_the_nearest_histogram_not_the_nearest_index():
    anchor = torch.tensor([[0.5, 0.5]])
    candidates = torch.tensor([[1.0, 0.0], [0.4, 0.6], [0.0, 1.0]])
    rejected = torch.zeros(1, 3, dtype=torch.bool)

    chosen = match_composition(anchor, candidates, rejected, n_negatives=1)

    assert chosen.tolist() == [[1]]


def test_a_rejected_candidate_loses_to_every_survivor_however_well_it_matches():
    """An overlapping context is a false negative; a perfect histogram match does not redeem it."""
    anchor = torch.tensor([[0.5, 0.5]])
    candidates = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    rejected = torch.tensor([[True, False]])

    chosen = match_composition(anchor, candidates, rejected, n_negatives=1)

    assert chosen.tolist() == [[1]]


def test_a_row_filled_from_rejected_candidates_is_reported_as_unclean():
    """When a slide cannot supply enough non-overlapping contexts the row is filled anyway, and
    the diagnostic is the only thing that says the denominator holds false negatives."""
    anchor = torch.tensor([[0.5, 0.5]])
    candidates = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    rejected = torch.tensor([[True, False]])

    chosen = match_composition(anchor, candidates, rejected, n_negatives=2)

    assert selection_is_clean(chosen, rejected).tolist() == [False]


def test_a_fully_clean_row_is_reported_as_clean():
    rejected = torch.zeros(1, 3, dtype=torch.bool)
    chosen = match_composition(torch.rand(1, 2), torch.rand(3, 2), rejected, n_negatives=2)

    assert selection_is_clean(chosen, rejected).tolist() == [True]


# --------------------------------------------------------------------------- the objective


def test_info_nce_falls_as_the_positive_becomes_more_similar():
    anchors = torch.tensor([[1.0, 0.0]])
    negatives = torch.tensor([[[0.0, 1.0]]])

    aligned = info_nce(anchors, torch.tensor([[1.0, 0.0]]), negatives, temperature=0.5)
    opposed = info_nce(anchors, torch.tensor([[-1.0, 0.0]]), negatives, temperature=0.5)

    assert aligned < opposed


def test_info_nce_is_scale_invariant_because_it_normalises():
    anchors, positives = torch.randn(4, 8), torch.randn(4, 8)
    negatives = torch.randn(4, 3, 8)

    base = info_nce(anchors, positives, negatives, temperature=0.5)
    scaled = info_nce(anchors * 10, positives * 3, negatives * 7, temperature=0.5)

    assert base == pytest.approx(scaled.item(), abs=1e-5)


def test_chance_level_is_log_of_the_class_count():
    """With everything orthogonal the loss must sit at ln(1 + K); a value below that on an
    untrained model means something is leaking rather than being learned."""
    anchors = torch.tensor([[1.0, 0.0, 0.0]])
    positives = torch.tensor([[0.0, 1.0, 0.0]])
    negatives = torch.tensor([[[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]])

    loss = info_nce(anchors, positives, negatives, temperature=1.0)

    assert loss.item() == pytest.approx(torch.log(torch.tensor(3.0)).item(), abs=1e-5)


# --------------------------------------------------------------------------- shared grouping


def test_slide_annotation_is_preferred_over_the_graph_index():
    """`slide_codes` is shared by the VICReg variance hinge and the contrastive negative pool.
    Several windows can come from one slide, and it is the SLIDE that carries the batch effect
    both of them are trying to keep out -- so a graph index standing in for it would quietly
    re-admit exactly what the grouping exists to exclude."""
    from torch_geometric.data import Data

    from interscale.train.aux_losses import slide_codes

    batch = Data(batch=torch.tensor([0, 0, 1, 1]), slide=torch.tensor([[1.0, 0.0]] * 4))

    codes = slide_codes(batch, None, n_rows=4)

    assert codes.tolist() == [0, 0, 0, 0]


def test_the_graph_index_is_the_fallback_when_no_slide_is_attached():
    from torch_geometric.data import Data

    from interscale.train.aux_losses import slide_codes

    batch = Data(batch=torch.tensor([0, 0, 1, 1]))

    assert slide_codes(batch, None, n_rows=4).tolist() == [0, 0, 1, 1]


def test_grouping_is_brought_into_token_order_by_the_index():
    from torch_geometric.data import Data

    from interscale.train.aux_losses import slide_codes

    batch = Data(batch=torch.tensor([0, 0, 1, 1]))

    codes = slide_codes(batch, torch.tensor([3, 0]), n_rows=2)

    assert codes.tolist() == [1, 0]


def test_none_disables_grouping():
    from torch_geometric.data import Data

    from interscale.train.aux_losses import slide_codes

    assert slide_codes(Data(batch=torch.zeros(4, dtype=torch.long)), None, 4, "none") is None


# --------------------------------------------------------------------------- match quality


def test_the_gap_is_zero_when_the_chosen_negative_has_the_anchor_composition():
    anchor = torch.tensor([[0.5, 0.5]])
    candidates = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    selected = torch.tensor([[0]])

    assert histogram_gap(anchor, candidates, selected).item() == pytest.approx(0.0)


def test_the_gap_reports_the_distance_actually_taken_not_the_best_available():
    """Matching is selection from a uniformly drawn bank, so an anchor with a rare local
    composition takes the closest of a bad bank. This is the only number that says so."""
    anchor = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[0.0, 1.0]])
    selected = torch.tensor([[0]])

    assert histogram_gap(anchor, candidates, selected).item() == pytest.approx(2.0)


def test_matching_gives_a_smaller_gap_than_drawing_at_random():
    """The stage's central claim, at the level the sampler can be tested at."""
    torch.manual_seed(0)
    anchor = torch.rand(16, 4)
    anchor = anchor / anchor.sum(dim=1, keepdim=True)
    candidates = torch.rand(64, 4)
    candidates = candidates / candidates.sum(dim=1, keepdim=True)
    rejected = torch.zeros(16, 64, dtype=torch.bool)

    matched = match_composition(anchor, candidates, rejected, n_negatives=4, match=True)
    random = match_composition(anchor, candidates, rejected, n_negatives=4, match=False)

    assert histogram_gap(anchor, candidates, matched).mean() < histogram_gap(
        anchor, candidates, random
    ).mean()
