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
