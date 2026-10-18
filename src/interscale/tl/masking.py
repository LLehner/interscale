"""Input corruption for the masked-reconstruction objective.

Two granularities are available, selected by ``mask_strategy``:

``"node"`` -- cell masking (the original behaviour)
    A Bernoulli subset of *cells* has its entire expression vector replaced by ``MASK_VALUE``,
    and the loss is evaluated on all G genes of those cells. A masked cell carries no
    information about itself, so the only thing the model can condition on is its neighbourhood,
    and E[x_i | neighbours of i] is close to the population mean. A near-constant predictor is
    therefore a strong solution to this objective, which is what makes it look like the model
    "learns the mean instead of reconstructing".

``"gene"`` -- per-entry masking (GraphMAE / MAE style)
    Every cell keeps most of its expression vector; a Bernoulli subset of *entries* ``(cell,
    gene)`` is replaced by ``MASK_VALUE``, drawn independently per cell. The loss is evaluated
    on those entries only (see :func:`masked_loss` below). The model now has the
    cell's remaining genes to condition on, so within-cell co-expression -- not just the
    population mean -- is available and rewarded.

``"gene"`` with a gene set (``dataset.mask_genes``) -- programme masking
    A fixed set of genes -- e.g. every response gene of a signalling programme -- is hidden in
    the selected cells, and nothing else: each cell is selected with probability
    ``mask_percentage`` (1.0: every cell), plus the cells in spatial blocks if those are on. The
    loss is on the hidden entries, as under ``"gene"``. With the set hidden in every cell no
    neighbour shows it either, so the programme can only be predicted from what drives it --
    senders, sources, receptors, position -- which is what a model of cell-cell communication
    should be able to do and a smoother of the programme itself cannot.

Spatial blocks (``dataset.mask_block_fraction`` > 0) -- on top of either strategy
    Whole cells in discs of ``dataset.mask_block_radius_um`` (:func:`sample_block_mask`) are
    hidden as well: under ``"node"`` they join the masked cells, under ``"gene"`` every entry of
    their rows is masked. Both strategies leave each hidden entry's neighbourhood visible, so the
    local component alone can fill it in; a disc wider than its reach cannot be, which is what
    asks the global component for context from beyond it. Under ``"node"`` the Bernoulli rate may
    then be 0, making the discs the only targets.

Note on GraphMAE (arXiv:2205.10803): GraphMAE itself masks whole nodes, exactly as ``"node"``
does here. What it changes to avoid the trivial solution is the *criterion* (scaled cosine
error, already available as ``interscale.train.losses.SCELoss``), a learnable ``[MASK]`` token
rather than a zero vector, re-mask decoding, and a GNN decoder. Per-entry masking is the
orthogonal knob this module adds; the two are independent and can be ablated together.

THE FILL VALUE IS -1, NOT 0, AND IS NOT CONFIGURABLE. 62% of legnini23's ``log1p_norm`` entries
are exactly 0, and 648 of its cells are all-zero outright, so a zero fill makes a masked position
indistinguishable from a real measurement -- under gene masking the corruption becomes invisible
and the model cannot identify the entries it is being asked to reconstruct. ``MASK_VALUE = -1``
is outside the range of any log1p-normalised layer, so a masked position is unambiguous.

This was measured, not assumed (legnini23, gene masking at rate 0.25, 3 seeds):

    fill 0     val_concordance 0.0696 +/- 0.0113   (CV 16.2%)
    fill -1    val_concordance 0.0799 +/- 0.0046   (CV  5.8%)

-1 wins on every seed and cuts the run-to-run spread by ~2.5x, to below the cell-masking
baseline's own CV of 10.6%. A learnable per-gene [MASK] token (GraphMAE's design) was also tried
and scored identically to the fixed -1 to three decimals: the token drifts only ~0.02-0.035 over
100 epochs from any initialisation, so the fill behaves as a constant, not as something worth
learning. It was removed rather than kept as an option.
"""

import math

import torch
from torch_geometric.data import Batch

# -1, not 0: outside the range of any log1p-normalised expression layer (which is >= 0), so a
# masked position can never be confused with a real measurement. See the module docstring for the
# measurement that settled this. Changing it back to 0 silently un-does that result.
MASK_VALUE = -1.0

MASK_STRATEGIES = ("node", "gene")


def block_masking_on(cfg) -> bool:
    """Whether ``cfg`` hides spatial blocks (``dataset.mask_block_fraction`` > 0).

    The modules and the trainer ask this next to ``mask_percentage > 0`` when deciding whether the
    reconstruction input is corrupted at all: with discs alone the Bernoulli rate is 0, and a
    module that only looked at the rate would pass the input through unmasked, training an
    autoencoder.
    """
    return float(cfg.dataset.get("mask_block_fraction", 0.0)) > 0


def sample_node_mask(num_nodes: int, pct: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw a per-cell mask: each cell is masked independently with probability ``pct``.

    Parameters
    ----------
    num_nodes
        Number of cells in the graph.
    pct
        Per-cell masking probability.
    generator
        Optional RNG, for reproducible draws.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[num_nodes]``. At least one cell is always masked, otherwise
        the graph contributes no supervision at all.
    """
    mask = torch.rand(num_nodes, generator=generator) < pct
    if not mask.any():
        mask[torch.randint(num_nodes, (1,), generator=generator)] = True
    return mask


def sample_gene_mask(
    num_nodes: int, num_genes: int, pct: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw a per-entry mask: each ``(cell, gene)`` entry is masked independently with prob ``pct``.

    The draw is independent per cell, so different cells lose different genes. That is
    deliberate -- a mask shared across all cells of a graph would let the model learn a fixed
    "these G_masked genes are always missing" shortcut, and would make each step's supervision
    a single gene subset rather than |cells| different ones.

    Parameters
    ----------
    num_nodes
        Number of cells in the graph.
    num_genes
        Number of genes (columns of ``data.x``).
    pct
        Per-entry masking probability.
    generator
        Optional RNG, for reproducible draws.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[num_nodes, num_genes]``. Every row has at least one masked
        entry, so every cell contributes to the loss and the per-cell cosine metric is defined
        for all of them.
    """
    mask = torch.rand(num_nodes, num_genes, generator=generator) < pct

    # Rows that came up all-False would silently drop out of the loss and make per-cell metrics
    # undefined; give each of them exactly one masked gene.
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        fill = torch.randint(num_genes, (int(empty_rows.sum()),), generator=generator)
        mask[empty_rows, fill] = True
    return mask


def resolve_mask_genes(selectors, var) -> list[int]:
    """The columns of ``var`` that ``dataset.mask_genes`` selects, in ``var`` order.

    Each selector is a gene name (an index entry of ``var``) or ``"<column>=<value>"``, which takes
    every gene whose ``var[column]`` reads ``value`` (compared as text). A selector matching nothing
    raises, so a misspelt gene or value never shrinks the hidden set silently.

    Parameters
    ----------
    selectors
        ``dataset.mask_genes``.
    var
        ``adata.var``, in the column order of ``data.x``.
    """
    names = [str(g) for g in var.index]
    position = {g: k for k, g in enumerate(names)}
    chosen = set()
    for selector in selectors:
        selector = str(selector)
        if "=" in selector:
            column, value = selector.split("=", 1)
            if column not in var.columns:
                raise ValueError(f"dataset.mask_genes selector {selector!r}: var has no column {column!r}.")
            hits = [k for k, v in enumerate(var[column].astype(str)) if v == value]
        else:
            hits = [position[selector]] if selector in position else []
        if not hits:
            raise ValueError(f"dataset.mask_genes selector {selector!r} matches no gene.")
        chosen.update(hits)
    return sorted(chosen)


def sample_gene_set_mask(
    num_nodes: int,
    num_genes: int,
    columns,
    cell_rate: float,
    blocks: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hide the genes ``columns`` in a selection of cells: ``[num_nodes, num_genes]`` boolean.

    A cell is selected with probability ``cell_rate`` (1.0: every cell, drawing nothing), or when
    ``blocks`` ``[num_nodes]`` marks it. At least one cell is selected, so every graph contributes to
    the loss.
    """
    if cell_rate >= 1.0:
        cells = torch.ones(num_nodes, dtype=torch.bool)
    elif cell_rate > 0:
        cells = torch.rand(num_nodes, generator=generator) < cell_rate
    else:
        cells = torch.zeros(num_nodes, dtype=torch.bool)
    if blocks is not None:
        cells = cells | blocks
    if not cells.any():
        cells[torch.randint(num_nodes, (1,), generator=generator)] = True
    genes = torch.zeros(num_genes, dtype=torch.bool)
    genes[torch.as_tensor(list(columns), dtype=torch.long)] = True
    return cells[:, None] & genes[None, :]


def sample_block_mask(
    pos: torch.Tensor, radius: float, fraction: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw spatial blocks: discs of cells, until at least ``fraction`` of the cells lie in one.

    Each disc is every cell within ``radius`` of a centre cell, the centre drawn uniformly among
    the cells no disc covers yet, so every disc adds at least its centre and the draw ends. A disc
    wider than the local component's reach holds cells whose whole neighbourhood is hidden: no
    message-passing layer can fill those in, and only context from beyond the disc can.

    Parameters
    ----------
    pos
        ``[N, D]`` cell coordinates.
    radius
        Disc radius, in the units of ``pos``.
    fraction
        Share of the cells to cover, in (0, 1). Discs are whole, so the result overshoots it by at
        most one disc.
    generator
        Optional RNG, for reproducible draws.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[N]``.
    """
    pos = pos.double()
    num_nodes = pos.shape[0]
    target = max(1, math.ceil(fraction * num_nodes))
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    while int(mask.sum()) < target:
        free = torch.where(~mask)[0]
        centre = free[torch.randint(len(free), (1,), generator=generator)]
        mask |= (pos - pos[centre]).norm(dim=1) <= radius
    return mask


def apply_mask(batched_data: Batch, mask_strategy: str = "node"):
    """Corrupt ``batched_data.x`` at the granularity named by ``mask_strategy``.

    Under ``"gene"`` the batch's ``gene_mask`` ``[N, G]`` selects the entries to overwrite, and it
    is handed back so the loss can be restricted to them. Under ``"node"`` every gene of every
    cell selected by ``.mask`` is overwritten and the returned entry mask is ``None``, meaning
    "score the full rows".

    The strategy is a parameter rather than being inferred from whether a ``gene_mask`` attribute
    happens to be present. Sniffing the attribute made a stale ``gene_mask`` -- left on a ``Data``
    object reused across strategies -- silently override the configured strategy, which the
    dataloader then had to defend against by deleting the attribute.

    Args:
        batched_data (Batch): batch carrying ``.mask`` ``[N]``, plus ``.gene_mask`` ``[N, G]`` when
            ``mask_strategy == "gene"``.
        mask_strategy: one of :data:`MASK_STRATEGIES`.

    Returns
    -------
        batched_data_w_mask (Batch):
            Copy of the batch with the masked entries set to ``MASK_VALUE``.
        mask_idx (torch.Tensor):
            Indices of the cells that carry at least one masked entry -- i.e. the rows on which
            predictions are scored.
        entry_mask (torch.Tensor | None):
            ``[N, G]`` boolean over the *full* node ordering under gene masking, ``None`` under
            cell masking. Callers must subset it with the same indices they use for ``y_true``.

    Example:
        Data object:
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        data = Data(x=x, edge_index=edge_index, mask=mask)
        ----
        mask_idx = torch.tensor([0, 2])
        masked_values = torch.tensor([[-1., -1.], [3., 4.], [-1., -1.], [7., 8.]])
    """
    assert batched_data.mask is not None, "Mask is not set in the batch."
    assert mask_strategy in MASK_STRATEGIES, f"mask_strategy must be one of {MASK_STRATEGIES}, got {mask_strategy!r}."

    gene_mask = getattr(batched_data, "gene_mask", None) if mask_strategy == "gene" else None
    masked_values = batched_data.x.clone()

    if gene_mask is None:
        assert mask_strategy == "node", "mask_strategy='gene' but the batch carries no gene_mask."
        mask = batched_data.mask
        mask_idx = torch.where(mask == 1)[0]  # TODO into 2D array [B, N_batched_nodes]
        masked_values[mask] = MASK_VALUE
        entry_mask = None
    else:
        gene_mask = gene_mask.bool()
        assert gene_mask.shape == batched_data.x.shape, (
            f"Mismatch: gene_mask.shape: {tuple(gene_mask.shape)}, x.shape: {tuple(batched_data.x.shape)}"
        )
        masked_values[gene_mask] = MASK_VALUE
        mask_idx = torch.where(gene_mask.any(dim=1))[0]
        entry_mask = gene_mask

    batched_data_w_mask = batched_data.clone()
    batched_data_w_mask.x = masked_values
    return batched_data_w_mask, mask_idx, entry_mask


# Losses whose value depends on the *arrangement* of a row, not just on the individual entries:
# they normalise or centre along dim=-1. Selecting entries out of them would change what a "row"
# is, so those get the masked entries zeroed in both tensors instead -- which restricts every sum,
# dot product and norm involved to the masked coordinates, leaving the row structure intact.
_ROW_STRUCTURED_LOSSES = ("SCELoss", "SCE_EntropyATT_Loss", "BalancedPearsonCorrelationLoss")


def masked_row_std(y: torch.Tensor, entry_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row standard deviation of ``y`` over the masked entries only, shape ``[N, 1]``.

    Note: this is the *spread* argument the existing ``GaussianNLL`` branch passes, and it is
    deliberately std, not variance. ``nn.GaussianNLLLoss`` documents its third argument as a
    variance, and the pre-existing unmasked branch has always passed ``torch.std(...)`` there --
    a real bug, but one the published cell-masking runs were trained with. Matching it keeps the
    two masking arms comparable; fix both call sites together, never just this one, or the
    ablation stops being an ablation.
    """
    m = entry_mask.to(y.dtype)
    n = m.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (y * m).sum(dim=1, keepdim=True) / n
    var = (((y * m) ** 2).sum(dim=1, keepdim=True) / n - mean**2).clamp(min=0)
    return var.sqrt().clamp(min=eps)


def _plain_loss(loss_fn, loss_type: str, y_pred: torch.Tensor, y_true: torch.Tensor):
    """``loss_fn`` over everything it is given, with GaussianNLL's third argument supplied."""
    if loss_type == "GaussianNLL":
        return loss_fn(y_pred, y_true, torch.std(y_true, dim=1, keepdim=True))
    return loss_fn(y_pred, y_true)


def masked_loss(loss_fn, loss_type: str, y_pred: torch.Tensor, y_true: torch.Tensor, entry_mask=None):
    """Evaluate a reconstruction loss on the masked entries only.

    Under cell masking (``entry_mask is None``) every entry of every scored row was blanked, so
    this is just ``loss_fn(y_pred, y_true)`` and the behaviour is unchanged. Under gene masking
    most entries of a scored row were *given to the model as input*; including them would let
    the identity map dominate the objective and would make the reported loss incomparable to the
    cell-masking arm.

    Parameters
    ----------
    loss_fn
        The configured criterion, e.g. ``nn.SmoothL1Loss()``.
    loss_type
        Its name as it appears in ``optim.loss``; selects how the restriction is applied.
    y_pred, y_true
        ``[N, G]`` predictions and targets for the scored rows.
    entry_mask
        ``[N, G]`` boolean, or ``None``. Under cell masking this is the per-cell mask broadcast
        over all genes, so it is all-True or all-False per row and the row branch below fires.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if entry_mask is None:
        return _plain_loss(loss_fn, loss_type, y_pred, y_true)

    # Whole-row mask, i.e. cell masking expressed at entry level: every row is either entirely
    # in or entirely out. Subset the ROWS rather than the entries, which keeps each row intact --
    # a row-normalising criterion (SCE's cosine, Pearson) is only meaningful on a whole row, and
    # zeroing the excluded rows instead would feed it rows of zeros that dilute the mean.
    # This branch reproduces the pre-all-cells behaviour of cell masking exactly.
    rows_in = entry_mask.all(dim=1)
    if bool((rows_in | (~entry_mask).all(dim=1)).all()):
        return _plain_loss(loss_fn, loss_type, y_pred[rows_in], y_true[rows_in])

    if loss_type in _ROW_STRUCTURED_LOSSES:
        # Only the rows with a masked entry. A row without one has no masked coordinates, and
        # zeroed it would enter a cosine criterion as an all-zero row -- an error of exactly 1 that
        # dilutes the mean and that no prediction can change.
        rows = entry_mask.any(dim=1)
        m = entry_mask[rows].to(y_pred.dtype)
        return loss_fn(y_pred[rows] * m, y_true[rows] * m)

    if loss_type == "GaussianNLL":
        # std, matching the unmasked branch above -- see masked_row_std for why.
        spread = masked_row_std(y_true, entry_mask).expand_as(y_true)
        return loss_fn(y_pred[entry_mask], y_true[entry_mask], spread[entry_mask])

    # Element-wise criteria (MSELoss, SmoothL1, ...) reduce over whatever they are given, so
    # handing them the selected entries as a flat vector is exactly a mean over masked entries.
    return loss_fn(y_pred[entry_mask], y_true[entry_mask])


def _local_reach(edge_index: torch.Tensor, num_nodes: int, n_hops: int) -> torch.Tensor:
    """Boolean ``[num_nodes, num_nodes]`` matrix: which nodes the GNN can already see from each node.

    ``reach[i, j]`` is True when ``j`` lies within ``n_hops`` of ``i``, the diagonal included --
    a cell is part of its own receptive field. ``n_hops`` should be the number of message-passing
    layers of the local component, since that is exactly its receptive field.

    The multi-hop closure is done with sparse matmuls, so the cost tracks the number of edges
    rather than ``num_nodes ** 2``; only the final densification is quadratic, and that is
    unavoidable because the attention mask itself is dense.
    """
    device = edge_index.device

    if n_hops <= 1:
        # No closure to compute, so skip the sparse round trip: scattering straight into the
        # boolean matrix avoids materialising an n x n float one just to threshold it.
        dense = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        dense[edge_index[0], edge_index[1]] = True
    else:
        values = torch.ones(edge_index.shape[1], device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()

        reach, frontier = adj, adj
        for _ in range(n_hops - 1):
            frontier = torch.sparse.mm(frontier, adj).coalesce()
            reach = (reach + frontier).coalesce()
        dense = reach.to_dense() > 0
    # Spatial neighbour graphs are built symmetric, but a directed edge_index would otherwise
    # leave the mask asymmetric and block only one direction of a pair the GNN mixed both ways.
    # `dense |= dense.T` aliases its own memory, so this is an out-of-place or.
    dense = dense | dense.transpose(0, 1)
    dense.fill_diagonal_(True)
    return dense


def _local_reach_pairs(edge_index: torch.Tensor, num_nodes: int, n_hops: int) -> torch.Tensor:
    """``[2, P]`` pairs ``(i, j)`` of :func:`_local_reach`'s closure, never densified.

    The same set as the True entries of ``_local_reach(edge_index, num_nodes, n_hops)``: the
    ``n_hops`` closure, symmetrised, with the diagonal. Memory tracks the number of pairs rather
    than ``num_nodes ** 2``, which is what lets a backend that never forms the attention matrix
    keep the long-range mask on a whole slide.
    """
    device = edge_index.device
    if n_hops <= 1 or edge_index.numel() == 0:
        pairs = edge_index
    else:
        values = torch.ones(edge_index.shape[1], device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
        reach, frontier = adj, adj
        for _ in range(n_hops - 1):
            frontier = torch.sparse.mm(frontier, adj).coalesce()
            reach = (reach + frontier).coalesce()
        pairs = reach.indices()
    diagonal = torch.arange(num_nodes, device=device).expand(2, -1)
    pairs = torch.cat([pairs, pairs.flip(0), diagonal], dim=1)
    return torch.unique(pairs, dim=1)


def _kept_positions(num_nodes: int, kept_nodes, seq_len: int, device) -> torch.Tensor:
    """``[num_nodes]``: each node's token position in the padded layout, or -1 for a node ``pad_batch`` dropped.

    ``pad_batch`` left-pads, so a graph's kept nodes are the last ``len(kept_nodes)`` positions
    before the CLS slot at ``seq_len``, in the order ``kept_nodes`` lists them.
    """
    kept = torch.as_tensor(kept_nodes, dtype=torch.long, device=device)
    position = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    position[kept] = seq_len - len(kept) + torch.arange(len(kept), device=device)
    return position


def blocked_pairs(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    index_nodes: list,
    *,
    n_hops: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``[3, P]`` ``(graph, query, key)`` positions of every pair the attention may not use.

    The same pairs as the boolean masks, as a list: for ``n_hops >= 1`` the True entries of
    :func:`create_transformer_attention_mask_from_edges` (one head's worth), for ``n_hops`` 0 those
    of :func:`attn_mask_diagonal`. Positions are in the padded layout -- ``pad_batch`` left-pads, so
    graph ``b``'s kept tokens are the last ``len(index_nodes[b])`` before the CLS slot -- and the
    CLS slot never appears. The closure is taken on the whole graph and then restricted to the kept
    nodes, so a path through a node ``pad_batch`` dropped still blocks its two ends. Nothing here is
    ``S x S``.

    Parameters
    ----------
    edge_index
        ``[2, num_edges]`` edge index of the whole batch, with PyG's per-graph node offsets.
    batch
        ``[num_nodes]`` graph assignment per node.
    index_nodes
        Per graph, the node indices ``pad_batch`` kept, relative to that graph's own node order.
    n_hops
        Radius of the blocked neighbourhood in message-passing steps; 0 blocks each token's
        attention to itself only (every position, padding included, as ``attn_mask_diagonal``).
    device
        Device for the returned tensor. Defaults to ``edge_index``'s.
    """
    device = edge_index.device if device is None else device
    if n_hops < 0:
        raise ValueError(f"n_hops must be >= 0, got {n_hops}.")
    seq_len = max(len(nodes) for nodes in index_nodes)
    if n_hops == 0:
        positions = torch.arange(seq_len, device=device)
        graphs = torch.arange(len(index_nodes), device=device).repeat_interleave(seq_len)
        return torch.stack([graphs, positions.repeat(len(index_nodes)), positions.repeat(len(index_nodes))])

    batch = batch.to(torch.long)
    out = []
    for b, kept_nodes in enumerate(index_nodes):
        nodes_b = torch.nonzero(batch == b, as_tuple=False).flatten()
        n_b = int(nodes_b.numel())
        if n_b == 0:
            continue
        # PyG batches graphs by concatenation: a graph's local indices are the global ones minus
        # the offset of its first node.
        offset = int(nodes_b[0].item())
        in_graph = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
        pairs = _local_reach_pairs(edge_index[:, in_graph] - offset, n_b, n_hops)

        position = _kept_positions(n_b, kept_nodes, seq_len, pairs.device)
        query, key = position[pairs[0]], position[pairs[1]]
        keep = (query >= 0) & (key >= 0)
        out.append(torch.stack([torch.full_like(query[keep], b), query[keep], key[keep]]))
    if not out:
        return torch.zeros((3, 0), dtype=torch.long, device=device)
    return torch.cat(out, dim=1).to(device)


#: Edge types of an attention pattern (:func:`attention_pattern`), in the order of their ids.
PATTERN_EDGE_TYPES = ("graph", "expander", "virtual")


_MASK64 = 2**64 - 1


def _splitmix64(value: int) -> int:
    """One step of SplitMix64: a fixed, version-independent mixing of a 64-bit integer."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def graph_seed(seed: int, num_nodes: int, edge_index: torch.Tensor | None) -> int:
    """A seed determined by the graph itself: its size and its edges, combined with ``seed``.

    The same graph -- in training, in evaluation, rebuilt by ``get_model_output`` -- always gets the
    same seed, whatever batch it lands in and in whatever order, so a random structure drawn from it
    (an expander) is fixed per graph without being stored on it. Mixed with SplitMix64 rather than
    Python's ``hash``, whose tuple hashing may change between Python versions.
    """
    n_edges, edge_sum = 0, 0
    if edge_index is not None and edge_index.numel():
        n_edges = int(edge_index.shape[1])
        edge_sum = int((edge_index[0].long() * 1_000_003 + edge_index[1].long()).sum().item())
    mixed = 0
    for value in (int(seed), int(num_nodes), n_edges, edge_sum):
        mixed = _splitmix64(mixed ^ (value & _MASK64))
    return mixed & (2**63 - 1)


def random_expander(num_nodes: int, degree: int, generator: torch.Generator) -> torch.Tensor:
    """``[2, E]`` an (approximately) ``degree``-regular random graph: ``degree / 2`` random permutations, both ways.

    Each permutation ``σ`` adds the edges ``i -> σ(i)`` and ``σ(i) -> i``, so every node gets
    ``degree`` edges; random regular graphs made this way are expanders with high probability
    (Friedman's theorem), which is what Exphormer relies on for mixing in few layers. Self-loops and
    repeated edges are dropped, so a few nodes end up with fewer. Drawn on the CPU from
    ``generator``; ``degree`` must be even.
    """
    if degree % 2:
        raise ValueError(f"the expander is built from degree / 2 permutations, so degree must be even; got {degree}.")
    if num_nodes < 2 or degree == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    nodes = torch.arange(num_nodes)
    perms = [torch.randperm(num_nodes, generator=generator) for _ in range(degree // 2)]
    src = torch.cat([nodes.repeat(len(perms)), torch.cat(perms)])
    dst = torch.cat([torch.cat(perms), nodes.repeat(len(perms))])
    keep = src != dst
    return torch.unique(torch.stack([src[keep], dst[keep]]), dim=1)


def attention_pattern(
    edge_index: torch.Tensor | None,
    batch: torch.Tensor,
    index_nodes: list,
    blocked: torch.Tensor,
    *,
    graph_edges: bool,
    degree: int,
    seed: int = 0,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pairs an Exphormer-style attention attends along: ``([3, E] (graph, query, key), [E] edge type)``.

    Three sources (Shirzad et al., ICML 2023), with the types of :data:`PATTERN_EDGE_TYPES`:

    * ``graph`` -- the neighbour graph's edges, when ``graph_edges``; an edge ``src -> dst`` is the
      query ``dst`` attending to the key ``src``, PyG's direction of a message.
    * ``expander`` -- a :func:`random_expander` of ``degree`` on the whole graph, drawn from a seed
      the graph itself determines (:func:`graph_seed`), so it is the same every time the graph is
      seen; or, given ``generator`` (training with resampling), a fresh one from it.
    * ``virtual`` -- the CLS slot, linked both ways to every cell: every position's query attends
      to CLS (padding rows included, so no row is ever empty), CLS attends to every kept cell and to
      itself.

    Then ``blocked`` (:func:`blocked_pairs`) is removed, whatever a pair's source -- under the
    long-range mask that removes every graph edge, since the GNN's reach contains them. Both graph
    and expander edges are drawn on the whole graph and restricted to the nodes ``pad_batch`` kept,
    laid out as it lays out tokens; padding is never a key. A pair found by two sources keeps the
    first type above. Nothing here is ``S x S``.
    """
    device = blocked.device if device is None else device
    batch = batch.to(torch.long)
    seq_len = max(len(nodes) for nodes in index_nodes)
    cls = seq_len
    width = seq_len + 1

    graphs, queries, keys, types = [], [], [], []
    for b, kept_nodes in enumerate(index_nodes):
        nodes_b = torch.nonzero(batch == b, as_tuple=False).flatten()
        n_b = int(nodes_b.numel())
        if n_b == 0:
            continue
        offset = int(nodes_b[0].item())
        local = torch.zeros((2, 0), dtype=torch.long, device=batch.device)
        if edge_index is not None:
            in_graph = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
            local = edge_index[:, in_graph] - offset
        position = _kept_positions(n_b, kept_nodes, seq_len, local.device)

        sources = []
        if graph_edges and local.numel():
            sources.append((local[1], local[0], 0))  # query dst, key src
        if degree:
            gen = generator if generator is not None else torch.Generator().manual_seed(graph_seed(seed, n_b, local))
            expander = random_expander(n_b, degree, gen).to(local.device)
            sources.append((expander[0], expander[1], 1))
        for query, key, edge_type in sources:
            query, key = position[query], position[key]
            keep = (query >= 0) & (key >= 0)
            graphs.append(torch.full((int(keep.sum()),), b, dtype=torch.long, device=local.device))
            queries.append(query[keep])
            keys.append(key[keep])
            types.append(torch.full((int(keep.sum()),), edge_type, dtype=torch.long, device=local.device))

    cell_pairs = torch.zeros((3, 0), dtype=torch.long, device=device)
    cell_types = torch.zeros((0,), dtype=torch.long, device=device)
    if graphs:
        pairs = torch.stack([torch.cat(graphs), torch.cat(queries), torch.cat(keys)]).to(device)
        pair_types = torch.cat(types).to(device)
        codes = (pairs[0] * width + pairs[1]) * width + pairs[2]
        # Deduplicate, keeping each pair's first occurrence: graph edges come before expander edges.
        unique, inverse = torch.unique(codes, return_inverse=True)
        first = torch.full((len(unique),), len(codes), dtype=torch.long, device=device)
        first = first.scatter_reduce(0, inverse, torch.arange(len(codes), device=device), reduce="amin")
        blocked_codes = ((blocked[0] * width + blocked[1]) * width + blocked[2]).to(device)
        first = first[~torch.isin(unique, blocked_codes)]
        cell_pairs, cell_types = pairs[:, first], pair_types[first]

    virtual = []
    for b, kept_nodes in enumerate(index_nodes):
        everyone = torch.arange(seq_len, device=device)
        kept = torch.arange(seq_len - len(kept_nodes), seq_len, device=device)
        virtual.append(torch.stack([torch.full_like(everyone, b), everyone, torch.full_like(everyone, cls)]))
        virtual.append(torch.stack([torch.full_like(kept, b), torch.full_like(kept, cls), kept]))
        virtual.append(torch.tensor([[b], [cls], [cls]], device=device))
    virtual = torch.cat(virtual, dim=1)
    pairs = torch.cat([cell_pairs, virtual], dim=1)
    edge_types = torch.cat([cell_types, torch.full((virtual.shape[1],), 2, dtype=torch.long, device=device)])
    return pairs, edge_types


def create_transformer_attention_mask_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch: torch.Tensor,
    index_nodes: list,
    num_heads: int,
    *,
    n_hops: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Block the transformer from attending inside the local component's receptive field.

    This is the ``M = 1 - A`` mask of the paper, with ``A`` taken as the ``n_hops`` closure of the
    spatial neighbour graph rather than just its direct edges: the point is to stop the two
    components from re-deriving the same signal, and a 2-layer GCN has already mixed the 2-hop
    neighbourhood. The diagonal is blocked with it -- a cell's own state is what the local
    embedding is.

    **Why this cannot produce NaN.** A softmax row that is entirely ``-inf`` is NaN, which is the
    failure this mask invites: a cell in a dense region can easily have every other cell of its
    window inside its own neighbourhood. Two properties rule it out here.

    * The CLS token is never blocked, as a query or as a key. Every row therefore keeps at least
      one attendable key, whatever the graph looks like. It is also never a padded key, so the
      merge with ``src_key_padding_mask`` cannot take that guarantee away.
    * The mask is boolean and is built by indexing, never by arithmetic. Forming it as
      ``(1 - A) * -inf`` -- the obvious reading of "inverse adjacency" -- puts ``0 * -inf`` on
      every connected pair, and that is NaN before the softmax ever runs.

    Padding positions are left unblocked for the same reason: their rows are meaningless but must
    still normalise, and ``src_key_padding_mask`` is what actually removes them as keys.

    Parameters
    ----------
    edge_index
        ``[2, num_edges]`` edge index of the whole batch, with PyG's per-graph node offsets.
    num_nodes
        Number of nodes in the batch; used to check ``batch`` and ``edge_index`` agree.
    batch
        ``[num_nodes]`` graph assignment per node.
    index_nodes
        Per graph, the indices of the nodes ``pad_batch`` kept, relative to that graph's own node
        order. Its lengths define the sequence length.
    num_heads
        Number of attention heads; the mask is repeated for each.
    n_hops
        Radius of the blocked neighbourhood, in message-passing steps. Pass the local component's
        ``num_layers``.
    device
        Device for the returned mask. Defaults to ``edge_index``'s.

    Returns
    -------
    torch.Tensor
        Boolean ``[num_batch * num_heads, S + 1, S + 1]`` mask, ``True`` where attention is
        blocked, with the CLS token in the last position. Ordered graph-major, matching what
        :class:`torch.nn.MultiheadAttention` expects of a 3-D ``attn_mask``.
    """
    device = edge_index.device if device is None else device
    if n_hops < 1:
        # `_local_reach` would read 0 as 1 and block the direct neighbours of a model whose local
        # component mixed none of them.
        raise ValueError(
            f"n_hops must be >= 1, got {n_hops}. A reach of 0 blocks only the diagonal; use attn_mask_diagonal."
        )
    batch = batch.to(torch.long)
    if batch.numel() != num_nodes:
        raise ValueError(f"batch has {batch.numel()} entries but num_nodes is {num_nodes}")

    num_batch = int(batch.max().item()) + 1
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    mask = torch.zeros((num_batch, max_seq_len + 1, max_seq_len + 1), dtype=torch.bool, device=device)

    for b in range(num_batch):
        nodes_b = torch.nonzero(batch == b, as_tuple=False).flatten()
        n_b = int(nodes_b.numel())
        if n_b == 0:
            continue
        # PyG batches graphs by concatenation, so a graph's nodes are contiguous and its local
        # indices are the global ones minus the offset of its first node.
        offset = int(nodes_b[0].item())
        in_graph = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
        local_edges = edge_index[:, in_graph] - offset

        reach = _local_reach(local_edges, n_b, n_hops)

        kept = torch.as_tensor(index_nodes[b], dtype=torch.long, device=reach.device)
        block = reach[kept][:, kept].to(device)

        # pad_batch left-pads, so the kept tokens sit in the LAST len(kept) positions before the
        # CLS slot. Writing the block anywhere else silently masks the wrong pairs.
        s = int(kept.numel())
        lo = max_seq_len - s
        mask[b, lo:max_seq_len, lo:max_seq_len] = block

    if bool(mask.all(dim=-1).any()):
        raise RuntimeError("a query row is fully blocked; softmax would be NaN")

    # (N * num_heads, L, S) is indexed graph-major: repeat_interleave, not repeat.
    return mask.repeat_interleave(num_heads, dim=0)


def attn_mask_diagonal(batch: torch.Tensor, index_nodes: list, num_heads: int, device: torch.device) -> torch.Tensor:
    """Block self-attention only: the weakest mask, and the default.

    Returns the same boolean convention as
    :func:`create_transformer_attention_mask_from_edges` (``True`` = blocked), so the two are
    interchangeable at the call site. The CLS slot in the last position is left open.
    """
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    batch_size = int(batch.max().item()) + 1
    attention_mask = torch.zeros(
        (num_heads * batch_size, max_seq_len + 1, max_seq_len + 1), device=device, dtype=torch.bool
    )
    diag_idx = torch.arange(max_seq_len, device=device)
    attention_mask[:, diag_idx, diag_idx] = True
    return attention_mask
