"""Per-graph inputs that positional encodings precompute once, when the graphs are built."""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


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

    src, dst = edge_index.detach().cpu().numpy()
    keep = src != dst
    adj = sp.coo_matrix((np.ones(int(keep.sum())), (src[keep], dst[keep])), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float64)

    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros(n)
    inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
    lap = sp.identity(n, format="csr") - sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)

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
