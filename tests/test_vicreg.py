"""The three VICReg terms.

Each has one job and a characteristic failure. The tests below pin the jobs, and — for the
variance term — the one deviation from the paper that this codebase needs: a per-group hinge,
because a batch-level one can be satisfied entirely by between-slide variance while every cell
within a slide collapses.
"""

import pytest

torch = pytest.importorskip("torch")

from interscale.train.vicreg import covariance_term, invariance_term, variance_term


def _spread(n=256, d=8, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * scale


# --------------------------------------------------------------------------- variance


def test_variance_is_near_zero_when_every_dimension_has_unit_spread():
    assert variance_term(_spread()) < 0.05


def test_variance_is_maximal_under_full_collapse():
    """Every cell on one point: the hinge should read ~gamma."""
    assert variance_term(torch.zeros(256, 8)) == pytest.approx(1.0, abs=0.02)


def test_variance_penalises_a_shrunken_embedding_proportionally():
    assert variance_term(_spread(scale=0.25)) > variance_term(_spread(scale=0.75)) > 0


def test_variance_acts_on_the_standard_deviation_not_the_variance():
    """The paper's point: with var in the hinge the gradient vanishes exactly at collapse."""
    z = torch.zeros(64, 4, requires_grad=True)
    variance_term(z + torch.randn(64, 4) * 1e-4).backward()

    assert z.grad is not None and z.grad.abs().sum() > 0


# --------------------------------------------------------------------------- grouping


def test_between_group_spread_alone_does_not_satisfy_the_hinge():
    """The failure this grouping exists for: two collapsed slides far apart look fine globally."""
    collapsed_a = torch.zeros(64, 4)
    collapsed_b = torch.ones(64, 4) * 3.0
    z = torch.cat([collapsed_a, collapsed_b])
    groups = torch.cat([torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)])

    assert variance_term(z) < 0.1, "batch-level hinge is fooled -- this is the premise"
    assert variance_term(z, groups=groups) == pytest.approx(1.0, abs=0.02)


def test_grouped_variance_matches_ungrouped_for_a_single_group():
    z = _spread()
    groups = torch.zeros(len(z), dtype=torch.long)

    assert variance_term(z, groups=groups) == pytest.approx(float(variance_term(z)), abs=1e-6)


def test_a_singleton_group_is_skipped_not_counted_as_collapsed():
    """One row has no spread to measure; scoring it 0 std would add a full penalty for nothing."""
    z = torch.cat([_spread(n=64), torch.randn(1, 8)])
    groups = torch.cat([torch.zeros(64, dtype=torch.long), torch.tensor([1])])

    assert variance_term(z, groups=groups) == pytest.approx(float(variance_term(_spread(n=64))), abs=1e-6)


def test_all_singleton_groups_yield_zero_rather_than_an_error():
    z = _spread(n=4)
    assert variance_term(z, groups=torch.arange(4)) == 0.0


# --------------------------------------------------------------------------- covariance


def test_covariance_is_small_for_independent_dimensions():
    assert covariance_term(_spread()) < 0.1


def test_covariance_catches_redundant_dimensions():
    """Informational collapse: the dimensions are all copies of one another."""
    z = _spread()[:, :1].repeat(1, 8)

    assert covariance_term(z) > 5.0


def test_covariance_excludes_the_diagonal_entirely():
    """Uncorrelated dimensions score zero whatever their individual variances are.

    Built from columns orthogonal to the all-ones vector, so they are centred *and* mutually
    orthogonal: the sample covariance matrix is exactly diagonal. Scaling those columns then
    changes only the diagonal, which the term drops.
    """
    n, d = 64, 5
    g = torch.Generator().manual_seed(0)
    basis, _ = torch.linalg.qr(torch.cat([torch.ones(n, 1), torch.randn(n, d, generator=g)], dim=1))
    z = basis[:, 1:]  # orthogonal to ones => centred; mutually orthogonal => zero covariance
    scaled = z * torch.tensor([1.0, 10.0, 0.1, 5.0, 2.0])

    assert covariance_term(z) == pytest.approx(0.0, abs=1e-8)
    assert covariance_term(scaled) == pytest.approx(0.0, abs=1e-8)


def test_covariance_is_not_scale_free_across_correlated_dimensions():
    """Worth knowing rather than assuming away: off-diagonals scale with the dimensions.

    ``cov(10a, b) == 10 cov(a, b)``, so inflating one dimension inflates its cross terms and the
    squared term grows ~100x. This is why the variance and covariance terms are complementary
    rather than redundant: once the variance hinge holds every dimension near unit spread, the
    off-diagonal covariance is effectively a correlation and comparable across runs.
    """
    z = _spread()
    scaled = z.clone()
    scaled[:, 0] *= 10

    assert covariance_term(scaled) > 5 * covariance_term(z)


def test_covariance_of_a_single_row_is_zero_not_a_division_by_zero():
    assert covariance_term(torch.randn(1, 8)) == 0.0


# --------------------------------------------------------------------------- invariance


def test_invariance_is_zero_for_identical_views():
    z = _spread()
    assert invariance_term(z, z) == 0.0


def test_invariance_grows_with_disagreement():
    z = _spread()
    assert invariance_term(z, z + 0.5) > invariance_term(z, z + 0.1) > 0


def test_mismatched_shapes_are_rejected():
    """Row i of each view must be the same cell; a shape difference is the detectable half."""
    with pytest.raises(ValueError, match="same cells"):
        invariance_term(_spread(n=10), _spread(n=12))
