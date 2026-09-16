"""The three VICReg terms, as plain functions over embedding matrices.

Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised
Learning*, ICLR 2022. Three terms with separate jobs:

* **variance** -- a hinge keeping each dimension's standard deviation above ``gamma``, which stops
  every cell collapsing onto one point;
* **covariance** -- the squared off-diagonal of the covariance matrix, which stops the dimensions
  carrying redundant copies of each other (*informational* collapse);
* **invariance** -- mean squared distance between two views of the same cell, which is the only
  term that constitutes a *task*.

The first two need no pairing and no negatives, which is why they are worth having on their own
beside a reconstruction loss; the third needs two views and is what makes VICReg a standalone
objective. See ``.claude/contrastive_plan.md``.

Nothing here takes a config or a batch. That is deliberate: the arithmetic is worth testing
without a model, and the decisions about *which* embeddings and *which* grouping to hand it
belong to the caller.

One deviation from the paper, and it matters here
-------------------------------------------------
:func:`variance_term` takes an optional ``groups`` and, when given, computes the hinge **within
each group and averages**, rather than over the whole batch. With one slide per graph and several
graphs per batch, a batch-level hinge can be satisfied entirely by *between*-slide variance -- the
batch effect -- while every cell inside a slide collapses to a single point. The within-slide
spread is the one that means "cells in this tissue differ from each other".

The covariance term stays batch-level by default: it is a marginal statistic, and mixing slides
in it is harmless.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Target standard deviation for the variance hinge; 1 in the paper.
DEFAULT_GAMMA = 1.0
#: Added inside the square root, so the gradient stays finite at zero variance. 1e-4 in the paper.
DEFAULT_EPS = 1e-4


def variance_term(
    z: torch.Tensor,
    groups: torch.Tensor | None = None,
    gamma: float = DEFAULT_GAMMA,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Hinge on the per-dimension standard deviation: ``mean_j max(0, gamma - std(z_j))``.

    Parameters
    ----------
    z
        ``[N, D]`` embeddings.
    groups
        Optional ``[N]`` integer labels. When given, the hinge is computed within each group and
        averaged over the groups present, which is what keeps the term from being satisfied by
        between-group (i.e. between-slide) variance alone. Groups with fewer than two rows are
        skipped -- a standard deviation over one sample is not defined, and counting it as zero
        would add a full-strength penalty for a group that simply has no spread to measure.
    gamma
        Target standard deviation.
    eps
        Added under the square root. The paper is explicit that the hinge must act on the standard
        deviation and not the variance: at ``z`` near its mean the gradient of the variance
        vanishes, so the term stops pushing exactly when collapse is happening.

    Returns
    -------
    torch.Tensor
        Scalar. Zero when every dimension already has standard deviation at or above ``gamma``.
    """
    if groups is None:
        return _variance(z, gamma, eps)

    totals = []
    for group in torch.unique(groups):
        rows = z[groups == group]
        if rows.shape[0] < 2:
            continue
        totals.append(_variance(rows, gamma, eps))
    if not totals:
        return z.new_zeros(())
    return torch.stack(totals).mean()


def _variance(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
    return F.relu(gamma - std).mean()


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    """Sum of the squared off-diagonal covariances, scaled by ``1 / D``.

    Parameters
    ----------
    z
        ``[N, D]`` embeddings. Needs at least two rows; fewer returns zero, since a covariance
        over one sample is undefined rather than zero.

    Returns
    -------
    torch.Tensor
        Scalar, zero when the dimensions are exactly decorrelated.
    """
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())
    centred = z - z.mean(dim=0, keepdim=True)
    cov = (centred.T @ centred) / (n - 1)
    off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diagonal.pow(2).sum() / d


def invariance_term(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between two views of the same cells, row for row.

    Parameters
    ----------
    z_a, z_b
        ``[N, D]`` embeddings of the SAME cells in the SAME order. That alignment is the whole
        content of the term: paired with mismatched rows it still produces a plausible number and
        trains the model towards nothing.

    Returns
    -------
    torch.Tensor
        Scalar.

    Raises
    ------
    ValueError
        If the two shapes differ, which is the one form of misalignment that is detectable here.
    """
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"invariance_term needs two views of the same cells: got {tuple(z_a.shape)} and "
            f"{tuple(z_b.shape)}. Row i of each must be the same cell."
        )
    return F.mse_loss(z_a, z_b)
