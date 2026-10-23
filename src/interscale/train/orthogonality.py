"""Orthogonality between the parts of a prediction, and within one embedding: the terms of decoder_plan.md, D4.

Pure functions on tensors; the auxiliary losses that read them off a step's output are the ``orth_*`` terms of
:mod:`interscale.train.aux_losses`. Rows are cells (for the loadings: genes), columns dimensions. Every value lies in
[0, 1] and is scale-free: a part's size cannot set it to zero. The raw Frobenius penalty of Domain Separation Networks
(Bousmalis et al. 2016, ``||H_c^T H_p||_F^2``) can be zeroed that way under an additive decoder, which rescales each
part freely.

What each one is invariant to is what decides what it can be met by without removing information:

* :func:`linear_cka` -- scale and rotation of each side (Kornblith et al. 2019). It weighs directions by their variance,
  so a large direction the decoder ignores lowers it while a small one carries a copy.
* :func:`canonical_overlap` -- any invertible linear map of each side up to the ridge: the mean squared canonical
  correlation of two cell-by-dimension matrices, or the mean squared cosine of the principal angles between two
  gene-by-dimension loading matrices. This is what a linear decoder can undo, so a copy cannot hide in a small direction
  (below the ridge it can).
* :func:`kernel_cka` -- the normalised HSIC of two RBF kernels: nonlinear dependence.
* :func:`cross_cosine` -- the Orthogonal Projection Loss's term between two classes (Ranasinghe et al. 2021), with the
  two parts as the classes. With two decoders a rotation of one part meets it without removing anything: a control.
* :func:`gene_correlation` -- per gene, the correlation of two parts' predictions: an orthogonal sum per gene.
* :func:`decorrelation` and :func:`loading_orthogonality` -- within one embedding or one decoder: a basis in which the
  dimensions, or their gene programmes, do not overlap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Below this a norm, a variance or a denominator counts as zero.
EPS = 1e-12


def centre_by_group(x: torch.Tensor, groups: torch.Tensor | None = None) -> torch.Tensor:
    """``x`` minus its rows' mean within each group: each graph centred on its own (one group without ``groups``)."""
    if groups is None:
        return x - x.mean(dim=0, keepdim=True)
    _, inverse = torch.unique(groups, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    sums = x.new_zeros(n_groups, x.shape[1]).index_add_(0, inverse, x)
    counts = torch.bincount(inverse, minlength=n_groups).clamp_min(1).to(x.dtype)
    return x - (sums / counts[:, None])[inverse]


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear CKA of two row-aligned, column-centred matrices: ``||X^T Y||^2 / (||X^T X|| ||Y^T Y||)``, Frobenius norms.

    1 when one is a rotation and rescaling of the other, 0 when no column of one correlates with a column of the other.
    """
    cross = (x.T @ y).pow(2).sum()
    norm = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return cross / norm.clamp_min(EPS)


def _whitener(gram: torch.Tensor, ridge: float) -> torch.Tensor:
    """The lower Cholesky factor of ``gram`` plus ``ridge`` times its mean diagonal (detached) on the diagonal."""
    scale = torch.diagonal(gram).mean().detach().clamp_min(EPS)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.cholesky(gram + ridge * scale * eye)


def _unit_columns(x: torch.Tensor) -> torch.Tensor:
    """Each column divided by its root mean square (a zero column stays zero): an invertible map of the column space."""
    return x / x.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(EPS)


def canonical_overlap(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Mean squared canonical correlation of two row-aligned matrices: ``||Lx^-1 X^T Y Ly^-T||^2 / min(dx, dy)``.

    Each column is first scaled to unit root mean square -- an invertible map, so no canonical correlation changes, but
    a column inflated far beyond the others can no longer raise the ridge over the rest. Then
    ``Lx Lx^T = X^T X + ridge * mean(diag) * I``, and likewise for ``Y``. With centred columns (cells as rows) the sum
    is that of the squared canonical correlations; with uncentred loadings (genes as rows) that of the squared cosines
    of the principal angles between the two column spaces. 1 when one side is an invertible linear map of the other (as
    the ridge goes to 0), 0 when the two spaces are orthogonal; unchanged by any invertible linear map of either side,
    except for directions -- combinations of columns -- with less variance than the ridge, which escape. Rows must
    outnumber the columns by a good margin, or canonical correlations come out high by chance.
    """
    x, y = _unit_columns(x), _unit_columns(y)
    lx, ly = _whitener(x.T @ x, ridge), _whitener(y.T @ y, ridge)
    whitened = torch.linalg.solve_triangular(lx, x.T @ y, upper=False)  # Lx^-1 X^T Y
    whitened = torch.linalg.solve_triangular(ly, whitened.T, upper=False)  # Ly^-1 (Lx^-1 X^T Y)^T
    return whitened.pow(2).sum() / min(x.shape[1], y.shape[1])


def _rbf(x: torch.Tensor) -> torch.Tensor:
    """The RBF kernel of the rows, its bandwidth the median distance between distinct rows (detached)."""
    sq = (x * x).sum(dim=1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)).clamp_min(0.0)
    off = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    sigma2 = d2[off].median().detach().clamp_min(EPS)
    return torch.exp(-d2 / (2.0 * sigma2))


def _centre_kernel(k: torch.Tensor) -> torch.Tensor:
    """``H K H`` with ``H = I - 11^T / n``: rows and columns centred."""
    return k - k.mean(dim=0, keepdim=True) - k.mean(dim=1, keepdim=True) + k.mean()


def kernel_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CKA of the RBF kernels of two row-aligned matrices: the normalised HSIC, ``<Kx, Ky> / (||Kx|| ||Ky||)`` centred.

    Unlike :func:`linear_cka` it sees nonlinear dependence (one side a nonlinear function of the other). The bandwidths
    follow the median distance, so each side's scale drops out. Costs ``n^2`` per side: read on a subsample of rows.
    """
    kx, ky = _centre_kernel(_rbf(x)), _centre_kernel(_rbf(y))
    norm = torch.sqrt((kx * kx).sum() * (ky * ky).sum())
    return (kx * ky).sum() / norm.clamp_min(EPS)


def cross_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The mean ``|cos|`` between every row of ``x`` and every row of ``y``: OPL's term between two classes.

    The paper's code takes the absolute cosine of each pair (its equation the absolute value of their sum, which lets
    pairs cancel); this follows the code. Raw vectors, uncentred, as the paper reads them. 0 when the two point clouds
    lie in orthogonal subspaces of the shared feature space.
    """
    return (F.normalize(x, dim=1) @ F.normalize(y, dim=1).T).abs().mean()


def _centre_masked(x: torch.Tensor, mask: torch.Tensor, groups: torch.Tensor | None) -> torch.Tensor:
    """``x`` minus, per column and group, the mean over the rows where ``mask`` (float) is 1."""
    if groups is None:
        groups = torch.zeros(len(x), dtype=torch.long, device=x.device)
    _, inverse = torch.unique(groups, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    sums = x.new_zeros(n_groups, x.shape[1]).index_add_(0, inverse, x * mask)
    counts = x.new_zeros(n_groups, x.shape[1]).index_add_(0, inverse, mask).clamp_min(1.0)
    return x - (sums / counts)[inverse]


def gene_correlation(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None, groups: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean over genes (columns) of the squared correlation of ``a[:, g]`` and ``b[:, g]`` over the rows ``mask[:, g]``.

    Each group (graph) is centred on its own, per gene, over its masked rows. A gene where either part does not vary is
    left out; with none left the result is a zero that keeps the graph (no gradient). 0 when, for every gene, the two
    parts are uncorrelated over the cells: the prediction is then an orthogonal sum, and each part's share of a gene's
    variance is its own variance over the whole's, with no cross term.
    """
    m = torch.ones_like(a) if mask is None else mask.to(a.dtype)
    ac, bc = _centre_masked(a, m, groups) * m, _centre_masked(b, m, groups) * m
    cross = (ac * bc).sum(dim=0)
    norm = torch.sqrt((ac * ac).sum(dim=0) * (bc * bc).sum(dim=0))
    defined = norm > EPS
    if not bool(defined.any()):
        return (a.sum() + b.sum()) * 0.0
    return (cross[defined] / norm[defined]).pow(2).mean()


def _mean_off_diagonal_square(c: torch.Tensor) -> torch.Tensor:
    """The mean of the squared entries of a square matrix off its diagonal; 0 for fewer than two rows."""
    d = c.shape[0]
    if d < 2:
        return c.sum() * 0.0
    off = c - torch.diag(torch.diagonal(c))
    return off.pow(2).sum() / (d * (d - 1))


def decorrelation(x: torch.Tensor, groups: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared correlation between distinct columns of ``x`` over its rows, each group centred on its own.

    A column that does not vary is left out. 0 when the dimensions are uncorrelated over the cells: the VICReg
    covariance term (``train.vicreg``) with each dimension standardised, so no dimension's size counts.
    """
    xc = centre_by_group(x, groups)
    std = xc.pow(2).mean(dim=0).sqrt()
    keep = std > EPS
    z = xc[:, keep] / std[keep]
    return _mean_off_diagonal_square(z.T @ z / len(z))


def loading_orthogonality(w: torch.Tensor) -> torch.Tensor:
    """Mean squared cosine between distinct columns of a gene-by-dimension loading matrix ``w``.

    0 when every latent dimension loads a gene programme orthogonal to every other's: the soft orthogonality
    ``||W^T W - I||^2`` (Bansal et al. 2018) with the columns normalised, so a column's size does not count. A zero
    column contributes nothing.
    """
    wn = F.normalize(w, dim=0)
    return _mean_off_diagonal_square(wn.T @ wn)
