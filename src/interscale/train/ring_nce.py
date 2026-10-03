"""Ring contrast: the anchor's long-range environment against the neighbourhood the GNN already has.

The positive is the cells in a hop annulus around the anchor -- beyond the attention mask's reach,
within a configured outer bound -- pooled. The negative is the anchor's own inner neighbourhood,
the set the local component aggregates over, pooled the same way.

That is the architecture's ``M = 1 - A`` mask restated as an objective. The mask *forbids* the
transformer from attending inside the GNN's receptive field; this term *asks* the resulting
embedding to describe what lies outside it rather than what lies inside.

Why an annulus rather than a single distant cell
------------------------------------------------
A random distant cell on the same slide makes "which slide am I on" a near-optimal answer, and
carries mostly its own identity. Pooling a region instead gives a description of the tissue
environment. The annulus is also *centred on the anchor*, so positive and negative sit at almost
the same place in every smooth spatial field -- which removes the "which of these is near me"
shortcut that sinks a distant-cell positive on data where every long-range field is smooth.

The thing to watch
------------------
The inner neighbourhood is naturally MORE similar to the anchor than the ring is: spatial
autocorrelation puts it there, and the GCN aggregate puts the anchor's own expression inside its
neighbours' embeddings. So this term starts by asking the model to move against that gradient, and
``_ring_nce_acc`` near 0 early in training is expected rather than alarming. What would be alarming
is it staying there -- that is the term failing to find anything in the ring that the core does not
also have.

``anchors_masked_only`` matters more here than anywhere else for the same reason: an unmasked
anchor's own transcriptome is inside its core's pooled vector, which makes the *negative*
identifiable by self-detection.
"""

from __future__ import annotations

import torch

from interscale.train.context_nce import khop_membership, pool_contexts, valid_contexts


def ring_membership(
    centers: torch.Tensor,
    token_edges: torch.Tensor,
    n_tokens: int,
    inner_hops: int,
    outer_hops: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``([C, n_tokens], [C, n_tokens])`` -- the annulus and the core it excludes.

    The core is everything within ``inner_hops``; the ring is everything within ``outer_hops``
    that is not in the core. Both exclude the centre itself.

    Neither can leave the centre's slide: ``edge_index`` carries no cross-graph edges, so a hop
    walk is confined to one graph by construction rather than by a filter that could be forgotten.

    Parameters
    ----------
    centers
        ``[C]`` token indices.
    token_edges
        ``[2, E]`` edges in token order.
    n_tokens
        Width of the returned membership matrices.
    inner_hops
        The local component's receptive field, and what the attention mask blocks. Comes from
        :func:`interscale.tl.local_reach_hops` on the trained module, not from a knob of its own.
    outer_hops
        Outer bound of the ring. Unbounded would make the positive the slide mean, which is the
        degenerate case this parameter exists to avoid.

    Returns
    -------
    ring : torch.Tensor
        ``[C, n_tokens]`` boolean, the annulus.
    core : torch.Tensor
        ``[C, n_tokens]`` boolean, the inner neighbourhood.

    Raises
    ------
    ValueError
        If ``outer_hops <= inner_hops``, which makes the ring empty for every centre -- a state
        that otherwise shows up only as every anchor being silently dropped.
    """
    if outer_hops <= inner_hops:
        raise ValueError(
            f"ring_outer_hops ({outer_hops}) must exceed the inner radius ({inner_hops}), which is "
            "the attention mask's reach. Otherwise the annulus is empty for every anchor and the "
            "term has no positive to contrast."
        )
    core = khop_membership(centers, token_edges, n_tokens, inner_hops)
    outer = khop_membership(centers, token_edges, n_tokens, outer_hops)
    return outer & ~core, core


def usable_anchors(ring_sizes: torch.Tensor, core_sizes: torch.Tensor) -> torch.Tensor:
    """``[C]`` boolean: centres that have both a non-empty ring and a non-empty core.

    An anchor missing either side has no contrast to make. Both pool to a vector of zeros, which
    is a perfectly ordinary-looking embedding -- so without this the term would train those rows
    to align with the origin and report nothing unusual.
    """
    return valid_contexts(ring_sizes) & valid_contexts(core_sizes)


def pooled_ring_and_core(
    ring: torch.Tensor, core: torch.Tensor, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool both sets and report which rows are usable.

    Returns
    -------
    ring_pooled, core_pooled : torch.Tensor
        ``[C, F]`` each.
    keep : torch.Tensor
        ``[C]`` boolean from :func:`usable_anchors`.
    """
    ring_pooled, ring_sizes = pool_contexts(ring, features)
    core_pooled, core_sizes = pool_contexts(core, features)
    return ring_pooled, core_pooled, usable_anchors(ring_sizes, core_sizes)


def extra_negatives(
    core_pooled: torch.Tensor, n_extra: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """``[A, n_extra, F]`` further negatives, drawn from *other* anchors' cores on the same slide.

    The requested design has one negative per anchor -- its own core -- which makes the objective
    a two-way softmax whose gradient is correspondingly weak. Filling the rest of the denominator
    with other cells' cores keeps every negative "a neighbourhood the GNN already covers", so the
    statement the term makes is unchanged while the discrimination gets harder.

    Drawn with an index shift rather than uniformly, so an anchor can never draw its own core as
    one of its extra negatives -- that would put the same vector in the denominator twice and
    quietly halve its weight.
    """
    n_anchors = core_pooled.shape[0]
    if n_extra <= 0 or n_anchors < 2:
        return core_pooled.new_zeros((n_anchors, 0, core_pooled.shape[1]))

    n_extra = min(n_extra, n_anchors - 1)
    shifts = torch.randint(
        1, n_anchors, (n_anchors, n_extra), device=core_pooled.device, generator=generator
    )
    rows = torch.arange(n_anchors, device=core_pooled.device).unsqueeze(1)
    return core_pooled[(rows + shifts) % n_anchors]


def similarity_gap(anchors: torch.Tensor, ring: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
    """``[A]`` cos(anchor, ring) - cos(anchor, core): positive when the term is winning.

    The diagnostic that says which way the representation is actually pointing, on a scale that
    does not depend on how many negatives are in the denominator. It starts negative -- the core is
    naturally the more similar of the two -- so the number to watch is whether it rises, not
    whether it is above zero.
    """
    normalise = torch.nn.functional.normalize
    a = normalise(anchors, dim=-1)
    return (a * normalise(ring, dim=-1)).sum(-1) - (a * normalise(core, dim=-1)).sum(-1)
