"""The value ``_common_step`` returns: named fields rather than a positional tuple.

Every auxiliary loss added beside reconstruction needs something different out of the forward
pass -- attention weights, the padded token embeddings, the mapping from token to slide -- and a
positional tuple makes each of those a signature change propagated through four modules and three
call sites. These dataclasses are the extension point: a new loss reads a field, and adding a
field breaks nothing.

Two coordinate systems meet here, and confusing them is the main source of silent errors:

* **batch node order** -- ``batch.x``, ``batch.batch``, ``edge_index`` and any local embedding,
  all ``[N, ...]`` over every cell in the batch.
* **padded order** -- ``global_embedding`` ``[S, B, E]``, one column per graph, padded to
  ``max_seq_len`` with a CLS token at the LAST sequence position.

:attr:`ViewOutput.padded_node_idx` is the bridge: the batch-global node index of each kept token,
in the order the tokens come out. Index anything that lives in batch node order with it to bring
it into token order. It is computed inside ``_process_batch_for_metrics`` and was previously
discarded; it is carried here so that every consumer uses the same one instead of re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ViewOutput:
    """One forward pass over one (possibly corrupted) copy of the batch.

    A single-view run has exactly one of these. From the two-view stage on there is one per
    view, all produced from the same batch with independent corruption draws.

    Attributes
    ----------
    local_embedding
        ``[N, E]`` in batch node order, or ``None`` for a module with no local component.
    global_embedding
        ``[S, B, E]`` in padded order, CLS at the last sequence position, or ``None`` for a
        module with no global component.
    src_padding_mask
        ``[B, S]`` boolean, ``True`` where the position is padding. ``None`` when there is no
        padding step.
    padded_node_idx
        ``[N_kept]`` batch-global node index of each kept token, in token order. ``None`` for
        graph-level prediction, where tokens are not gathered per cell.
    attn
        Stacked per-layer attention weights, or ``None``.
    """

    local_embedding: torch.Tensor | None = None
    global_embedding: torch.Tensor | None = None
    src_padding_mask: torch.Tensor | None = None
    padded_node_idx: torch.Tensor | None = None
    attn: torch.Tensor | None = None

    def tokens(self) -> torch.Tensor:
        """``[N_kept, E]`` real cell tokens -- see :func:`gather_tokens`."""
        if self.global_embedding is None or self.src_padding_mask is None:
            raise ValueError("This view has no padded global embedding to gather tokens from.")
        return gather_tokens(self.global_embedding, self.src_padding_mask)


@dataclass
class StepOutput:
    """What ``_common_step`` returns, for every module type.

    Attributes
    ----------
    y_pred, y_true
        Predictions and targets, already aligned. Shapes depend on task and level; for the
        dual-decoder module they are the stacked ``[2N, ...]`` local-then-global form.
    entry_mask
        ``[N, F]`` boolean marking the entries the *loss* is restricted to, or ``None`` for
        classification. Metrics are computed over everything; only the loss is restricted.
    views
        One :class:`ViewOutput` per forward pass. Length 1 until the two-view stage.
    """

    y_pred: torch.Tensor
    y_true: torch.Tensor
    entry_mask: torch.Tensor | None = None
    views: list[ViewOutput] = field(default_factory=list)

    @property
    def view(self) -> ViewOutput:
        """The reconstruction view -- the first one, and the only one in a single-view run."""
        return self.views[0]

    @property
    def local_embedding(self) -> torch.Tensor | None:
        """Shorthand for ``self.view.local_embedding``."""
        return self.view.local_embedding

    @property
    def global_embedding(self) -> torch.Tensor | None:
        """Shorthand for ``self.view.global_embedding``."""
        return self.view.global_embedding

    @property
    def attn(self) -> torch.Tensor | None:
        """Shorthand for ``self.view.attn``."""
        return self.view.attn


def gather_cls(global_embedding: torch.Tensor) -> torch.Tensor:
    """The CLS token of each graph: ``[B, E]``.

    The CLS token is *appended*, so it sits at the LAST sequence position -- not the first, as
    BERT-style diagrams suggest. It is the one position every cell can attend to, which is also
    what keeps a fully blocked long-range attention row from being all ``-inf``.
    """
    return global_embedding[-1, :, :]


def gather_tokens(global_embedding: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
    """Real cell tokens as ``[N_kept, E]``, CLS and padding removed, in ``padded_node_idx`` order.

    The single place that knows which end of the sequence is which. Two things it exists to stop,
    both of which fail silently rather than raising:

    * **CLS in the token set.** CLS attends to every cell and is similar to everything, so as a
      contrastive negative it contributes a large, meaningless repulsive gradient to every anchor,
      and as a reconstruction target it is a row with no cell behind it.
    * **Padding in the token set.** Padded positions are identical to one another, so they inflate
      any covariance and satisfy any variance floor without a single real cell varying.

    ``pad_batch`` LEFT-pads, so a graph's tokens occupy the *last* positions before CLS. Boolean
    indexing by ``~src_padding_mask`` is order-preserving either way, which is why this works
    without knowing the padding side -- but nothing else here may assume the tokens start at 0.

    Parameters
    ----------
    global_embedding
        ``[S + 1, B, E]`` padded transformer output, CLS last.
    src_padding_mask
        ``[B, S + 1]`` boolean, ``True`` where the position is padding.

    Returns
    -------
    torch.Tensor
        ``[N_kept, E]``, row ``i`` being the cell named by ``padded_node_idx[i]``.
    """
    h = global_embedding[:-1]  # drop CLS -> [S, B, E]
    h = torch.permute(h, (1, 0, 2))  # [B, S, E]
    return h[~src_padding_mask[:, :-1]]  # [N_kept, E]
