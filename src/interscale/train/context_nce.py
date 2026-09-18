"""Context-contrast mechanics: k-hop contexts, pooled contexts, composition-matched negatives.

Stage 2 of ``.claude/contrastive_plan.md``. The objective asks a cell's representation to be
predictive of the tissue context it sits in -- an InfoNCE between an anchor token and a *pooled
neighbourhood*, rather than between two augmented copies of the same cell. The pairing is
structural, so it needs one forward pass, no augmentation policy and no second encoder pass.

Everything here is a plain function over tensors. Nothing takes a config, a batch or a module,
for the same reason as :mod:`interscale.train.vicreg`: the index arithmetic is the part that
fails silently, and it is worth testing without a model in the way.

The two nuisance variables, and why both are held fixed
------------------------------------------------------
A naive "anchor vs. its neighbourhood, against random cell sets" is solved without learning
anything about interaction, twice over:

* **Composition.** Niche composition -- which cell types are around me -- is already inside the
  2-hop GCN and predicts most targets. If the negative context has a different type histogram
  from the positive, "count the types" separates them and the term teaches nothing new. So a
  negative is *selected* to match the anchor's positive histogram as closely as the slide allows
  (:func:`match_composition`).

* **Spatial coherence.** A negative pooled from cells scattered across the slide regresses toward
  the slide mean, while a true context is a contiguous patch. Given how smooth the fields in this
  data are -- ``dist_to_center`` correlates 0.997 with its own neighbourhood mean, measured
  2026-09-18 -- "which of these is spatially coherent" is a sufficient discriminator that has
  nothing to do with interaction, and it would look like a healthy loss curve the whole way down.
  So a negative context is by default **another cell's actual k-hop neighbourhood**, built by the
  same operator as the positive.

``scattered`` remains available as the ablation: it is the term that says how much of the loss was
ever about arrangement rather than about either nuisance variable.

False negatives
---------------
Two overlapping neighbourhoods are the same context, so a candidate sharing any cell with the
anchor's own context is dropped rather than ranked (:func:`overlaps`). This is the set-level
version of the plan's "exclude the anchor's k-hop neighbours from the denominator": here the
denominator holds contexts, so exclusion is by intersection rather than by hop count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Distance written into a rejected (overlapping) candidate before ranking, so it sorts last
#: without introducing a NaN or an inf into anything differentiable.
_REJECTED = 1e9


def to_token_edges(edge_index: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Re-express ``edge_index`` in token order, dropping edges whose endpoints were not kept.

    ``edge_index`` is in batch node order; tokens are in kept order. ``inverse`` is the map from
    the first to the second, ``-1`` where a node produced no token.

    Cells whose neighbours were dropped by ``pad_batch`` end up with truncated contexts. With
    sequences that fit inside ``max_seq_len`` nothing is dropped and the filter is all-True --
    but it has to be here for the day someone lowers ``max_seq_len``.

    Parameters
    ----------
    edge_index
        ``[2, E]`` in batch node order.
    inverse
        ``[num_nodes]``, token index per node or ``-1``.

    Returns
    -------
    torch.Tensor
        ``[2, E_kept]`` in token order.
    """
    src, dst = edge_index
    keep = (inverse[src] >= 0) & (inverse[dst] >= 0)
    return torch.stack([inverse[src[keep]], inverse[dst[keep]]])


def build_inverse(padded_node_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """``[num_nodes]`` map from batch node index to token index, ``-1`` where no token was kept."""
    inverse = torch.full((num_nodes,), -1, dtype=torch.long, device=padded_node_idx.device)
    inverse[padded_node_idx] = torch.arange(len(padded_node_idx), device=padded_node_idx.device)
    return inverse


def khop_membership(centers: torch.Tensor, token_edges: torch.Tensor, n_tokens: int, hops: int) -> torch.Tensor:
    """``[len(centers), n_tokens]`` boolean: is token *j* within ``hops`` of centre *i*?

    The centre itself is **excluded** from its own context. That is not cosmetic: the anchor's own
    token is what the context is being asked to predict, and leaving it in makes the positive
    trivially identifiable by a projection onto itself.

    Computed by ``hops`` rounds of sparse propagation from the centres only, rather than by
    materialising a ``[n_tokens, n_tokens]`` reachability matrix -- with a few thousand cells per
    slide the latter is the larger of the two by a wide margin, and it grows the wrong way.

    Parameters
    ----------
    centers
        ``[C]`` token indices to expand from.
    token_edges
        ``[2, E]`` edges in token order, as returned by :func:`to_token_edges`.
    n_tokens
        Number of tokens, i.e. the width of the returned membership matrix.
    hops
        Number of propagation rounds; ``1`` is the immediate neighbourhood.

    Returns
    -------
    torch.Tensor
        ``[C, n_tokens]`` boolean membership, centres excluded from their own rows.
    """
    device = centers.device
    frontier = torch.zeros((len(centers), n_tokens), dtype=torch.bool, device=device)
    frontier[torch.arange(len(centers), device=device), centers] = True
    reached = frontier.clone()

    if token_edges.numel():
        src, dst = token_edges
        # Symmetric propagation: the spatial graph is undirected in intent, but `edge_index` is
        # not guaranteed to carry both directions, and a one-directional walk would silently give
        # a different context to each endpoint of the same edge.
        values = torch.ones(src.numel() * 2, device=device)
        indices = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        adjacency = torch.sparse_coo_tensor(indices, values, (n_tokens, n_tokens)).coalesce()

        for _ in range(hops):
            # [n_tokens, C] = [n_tokens, n_tokens] @ [n_tokens, C]
            grown = torch.sparse.mm(adjacency, reached.to(values.dtype).t()).t() > 0
            reached = reached | grown

    reached[torch.arange(len(centers), device=device), centers] = False
    return reached


def pool_contexts(membership: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool ``features`` over each row's members.

    A **uniform** mean over the k-hop set, not a degree-weighted propagation. The composition
    match is a statement about the *set*, so the pooling has to weight the set the way the
    histogram counts it, or the two describe different objects.

    Returns
    -------
    pooled : torch.Tensor
        ``[C, F]``. Rows with no members are zero; :func:`valid_contexts` is how a caller finds
        and drops them, since a zero row is a legitimate embedding value and cannot be sniffed.
    sizes : torch.Tensor
        ``[C]`` member count per row.
    """
    weights = membership.to(features.dtype)
    sizes = weights.sum(dim=1)
    pooled = weights @ features
    return pooled / sizes.clamp(min=1).unsqueeze(1), sizes.to(torch.long)


def valid_contexts(sizes: torch.Tensor) -> torch.Tensor:
    """``[C]`` boolean: rows with at least one member.

    An isolated cell has an empty k-hop set, so it has no context to be predictive of. Such an
    anchor contributes a pooled vector of zeros, which is a perfectly ordinary-looking embedding
    and would train the model to align with the origin.
    """
    return sizes > 0


def composition(membership: torch.Tensor, celltype_codes: torch.Tensor, n_types: int) -> torch.Tensor:
    """``[C, n_types]`` normalised cell-type histogram of each context.

    Normalised because the match is about *proportions*: two neighbourhoods of different size can
    still have the same composition, and requiring equal counts would silently make context size
    the discriminating variable instead.
    """
    # Both operands are made float explicitly. `membership` is boolean and a bool matmul is not
    # defined, so deriving either dtype from it raises -- loudly here, but it would be a dtype
    # promotion surprise in any arrangement that happened not to raise.
    one_hot = F.one_hot(celltype_codes, num_classes=n_types).to(torch.float32)
    counts = membership.to(torch.float32) @ one_hot
    return counts / counts.sum(dim=1, keepdim=True).clamp(min=1)


def overlaps(anchor_membership: torch.Tensor, candidate_membership: torch.Tensor) -> torch.Tensor:
    """``[A, C]`` boolean: does candidate *c*'s context share any cell with anchor *a*'s?

    An overlapping context is the same context, so it is a false negative -- and false negatives
    are the failure mode this whole design is arranged around, since here the most similar
    contexts are exactly the ones most likely to share the anchor's interaction program.
    """
    return (anchor_membership.to(torch.float32) @ candidate_membership.to(torch.float32).t()) > 0


def match_composition(
    anchor_hist: torch.Tensor,
    candidate_hist: torch.Tensor,
    rejected: torch.Tensor,
    n_negatives: int,
    generator: torch.Generator | None = None,
    match: bool = True,
) -> torch.Tensor:
    """Pick ``n_negatives`` candidates per anchor, preferring matched cell-type composition.

    The selection is the whole point of the stage: with the histogram held fixed across positive
    and negative, composition carries zero information about which is which, and the only way left
    to reduce the loss is to encode which specific states are present and how they are arranged.

    Parameters
    ----------
    anchor_hist
        ``[A, T]`` normalised histogram of each anchor's positive context.
    candidate_hist
        ``[C, T]`` the same for each candidate context.
    rejected
        ``[A, C]`` boolean, ``True`` where the candidate may not be used for that anchor
        (overlapping context, or an empty one).
    n_negatives
        How many to return per anchor. Capped at the number of candidates.
    generator
        Seeded generator for the ``match=False`` ablation. Unused when matching.
    match
        ``False`` draws uniformly at random among the survivors instead -- the ablation that says
        how much the matching was worth.

    Returns
    -------
    torch.Tensor
        ``[A, n_negatives]`` candidate indices. When an anchor has fewer survivors than
        ``n_negatives``, rejected candidates are drawn to fill the row rather than the row being
        dropped; :func:`selection_is_clean` reports whether that happened.
    """
    n_candidates = candidate_hist.shape[0]
    k = min(n_negatives, n_candidates)

    if match:
        # L1 between normalised histograms: a proportion difference, summed over types.
        distance = torch.cdist(anchor_hist.unsqueeze(0), candidate_hist.unsqueeze(0), p=1).squeeze(0)
    else:
        distance = torch.rand(
            (anchor_hist.shape[0], n_candidates), device=anchor_hist.device, generator=generator
        )

    distance = distance + rejected.to(distance.dtype) * _REJECTED
    return distance.topk(k, dim=1, largest=False).indices


def histogram_gap(
    anchor_hist: torch.Tensor, candidate_hist: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    """``[A]`` mean L1 distance between each anchor's positive histogram and its chosen negatives'.

    The instrument for the stage's central claim. Composition being held fixed across positive and
    negative is what makes "count the cell types" a losing strategy, and *selecting* the nearest
    available candidate is not the same as finding a close one: centres are drawn uniformly, so an
    anchor whose local composition is rare on its slide gets the closest of a bad bank. Nothing
    else here would say so -- :func:`selection_is_clean` reports whether a negative was *false*,
    not whether it was *matched*, and the loss falls either way.

    Both histograms are normalised, so the value lies in ``[0, 2]``: 0 is an exact composition
    match, and the scale to read it against is the same number from a ``match_composition: False``
    run, which is what no matching at all looks like on this data.
    """
    distance = torch.cdist(anchor_hist.unsqueeze(0), candidate_hist.unsqueeze(0), p=1).squeeze(0)
    return distance.gather(1, selected).mean(dim=1)


def selection_is_clean(selected: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """``[A]`` boolean: did every negative chosen for this anchor come from the survivors?

    False for an anchor whose slide could not supply ``n_negatives`` non-overlapping contexts, so
    a rejected one was drawn to fill the row. That is a false negative in the denominator, and it
    is reported rather than silently tolerated -- on a small slide, or with ``context_hops`` set
    high enough that every neighbourhood touches every other, it can be *most* rows.
    """
    return ~rejected.gather(1, selected).any(dim=1)


def info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE with one positive and ``K`` negatives per anchor, on L2-normalised inputs.

    Parameters
    ----------
    anchors
        ``[A, D]`` projected anchor tokens.
    positives
        ``[A, D]`` projected positive contexts, row-aligned with ``anchors``.
    negatives
        ``[A, K, D]`` projected negative contexts.
    temperature
        Softmax temperature. High (0.5) rather than low (0.1) on purpose: the softmax denominator
        already weights negatives by similarity, which is soft hard-negative mining, and here the
        hardness ranking and the false-negative ranking are the same ranking.

    Returns
    -------
    torch.Tensor
        Scalar mean cross-entropy over anchors, with the positive as class 0.
    """
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    positive_logit = (anchors * positives).sum(dim=-1, keepdim=True)  # [A, 1]
    negative_logits = torch.einsum("ad,akd->ak", anchors, negatives)  # [A, K]
    logits = torch.cat([positive_logit, negative_logits], dim=1) / temperature

    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)
