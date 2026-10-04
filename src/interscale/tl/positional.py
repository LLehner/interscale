"""Per-graph inputs that positional encodings precompute once, when the graphs are built."""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def _adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    """The graph as an unweighted, undirected ``[n, n]`` adjacency without self loops."""
    src, dst = edge_index.detach().cpu().numpy()
    keep = src != dst
    adj = sp.coo_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(n, n)).tocsr()
    return ((adj + adj.T) > 0).astype(np.float64)


def _normalised(adj: sp.csr_matrix) -> sp.csr_matrix:
    """``D^-1/2 A D^-1/2``, with isolated nodes' rows and columns left at zero."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros(adj.shape[0])
    inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
    return (sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)).tocsr()


def laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int, *, dense_max_nodes: int = 500) -> torch.Tensor:
    """``[num_nodes, k]`` lowest non-trivial eigenvectors of the symmetric normalised Laplacian.

    ``L = I - D^-1/2 A D^-1/2`` of the graph as an unweighted, undirected adjacency without self
    loops -- the same neighbour graph the local component runs on. Columns are ordered by
    eigenvalue, unit-norm, and zero-padded when the graph has fewer than ``k`` non-trivial ones.

    Three things differ from PyG's ``AddLaplacianEigenvectorPE``, all because tissue graphs are not
    the connected molecules it assumes:

    * **Trivial eigenvectors are dropped per component.** Every connected component with an edge
      has its own zero eigenvalue, whose eigenvector only says which component a cell is in. PyG
      drops the first column; here exactly as many are dropped as there are such components. That
      count is exact, so no tolerance has to separate a zero eigenvalue from a genuinely small
      one -- on a long, thin graph the first non-trivial eigenvalue can be ~1e-6. An isolated
      cell has ``L_ii = 1``, an ordinary eigenvalue, and is not trivial.
    * **Signs are canonical, not random.** An eigenvector is only defined up to sign. Each column
      is flipped so its largest-magnitude entry is positive, so the same graph gives the same
      encoding in training and in ``get_model_output``, which rebuilds the graphs. Sign
      *invariance* is taught by flipping in training (``pe.lap.sign_flip``), not baked in here.
    * **Large graphs use shift-invert ``eigsh``** around a point just below 0, which converges on
      the smallest eigenvalues without a dense ``N x N`` decomposition, from a fixed start vector
      so the result is reproducible.

    A degenerate eigenvalue (a near-regular grid) leaves the basis within its eigenspace arbitrary;
    canonical signs do not fix that, and it is the ambiguity SignNet-style encoders exist for.
    """
    n = int(num_nodes)
    out = np.zeros((n, k), dtype=np.float32)
    if n == 0:
        return torch.from_numpy(out)

    adj = _adjacency(edge_index, n)
    lap = sp.identity(n, format="csr") - _normalised(adj)

    n_components, labels = connected_components(adj, directed=False)
    n_trivial = int((np.bincount(labels, minlength=n_components) > 1).sum())
    n_wanted = min(n, n_trivial + k)

    if n <= dense_max_nodes or n_wanted >= n - 1:
        vals, vecs = np.linalg.eigh(lap.toarray())
    else:
        vals, vecs = eigsh(lap.tocsc(), k=n_wanted, sigma=-1e-3, which="LM", v0=np.ones(n))
    order = np.argsort(vals)
    vecs = vecs[:, order][:, n_trivial : n_trivial + k]

    if vecs.shape[1]:
        peak = np.abs(vecs).argmax(axis=0)
        signs = np.sign(vecs[peak, np.arange(vecs.shape[1])])
        signs[signs == 0] = 1.0
        vecs = vecs * signs
    out[:, : vecs.shape[1]] = vecs
    return torch.from_numpy(out)


def random_walk_pe(edge_index: torch.Tensor, num_nodes: int, steps: int) -> torch.Tensor:
    """``[num_nodes, steps]`` return probabilities of a random walk after t = 1..steps steps.

    ``diag((D^-1 A)^t)``: the probability that a walk from a cell, stepping to a uniformly chosen
    neighbour each time, is back at that cell after t steps -- LSPE's RWPE, GraphGPS's RWSE, and the
    same values as PyG's ``AddRandomWalkPE``. Same graph as :func:`laplacian_pe`. Column 1 is
    ``diag(D^-1 A)``, zero without self loops; it is kept so the encoding matches that definition.
    Isolated cells, which have no walk, get zeros.

    The full powers are never formed. ``D^-1 A`` is similar to the symmetric ``S = D^-1/2 A D^-1/2``
    through a diagonal matrix, which leaves the diagonal of every power unchanged, and for a
    symmetric ``S``, ``diag(S^(a+b))_i`` is the dot product of row ``i`` of ``S^a`` and ``S^b``. So
    powers only up to ``ceil(steps / 2)`` are needed, and a power's fill-in is everything within that
    many hops: at half the steps, a fraction of what the full powers would hold on a slide.
    """
    n = int(num_nodes)
    out = np.zeros((n, steps), dtype=np.float32)
    if n == 0 or steps == 0:
        return torch.from_numpy(out)

    sym = _normalised(_adjacency(edge_index, n))
    powers = [sp.identity(n, format="csr")]
    for _ in range((steps + 1) // 2):
        powers.append((powers[-1] @ sym).tocsr())
    for t in range(1, steps + 1):
        a, b = (t + 1) // 2, t // 2
        out[:, t - 1] = np.asarray(powers[a].multiply(powers[b]).sum(axis=1)).ravel()
    return torch.from_numpy(out)
