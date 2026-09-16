"""Auxiliary loss terms added beside the reconstruction criterion.

The reconstruction loss reads ``(y_pred, y_true)`` and nothing else, which is why every term that
needs something different -- attention weights, the padded token embeddings, which slide a token
came from -- has so far been bolted on with a branch on ``loss_type``. This module is the
alternative: a term declares what it needs by reading named fields off
:class:`~interscale.module.base.StepOutput`, and the training plan weights, sums and logs whatever
terms are configured without knowing what any of them are.

Adding a term means writing an :class:`AuxLoss` subclass, registering it in :data:`AUX_LOSSES`, and
giving it a default weight of ``0.0`` in ``cfg.optim.aux_loss_weights``. Nothing else changes: a
term with weight ``0.0`` is never constructed, so it costs nothing and cannot alter a run.

How many forward passes to run is derived, not configured. Each term declares
:attr:`AuxLoss.requires_views`, and the module runs the maximum over the enabled terms -- so
enabling a two-view loss turns on two-view mode by itself. A separate switch would be one more
thing to forget, and forgetting it produces a run that trains quietly on the wrong objective.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from interscale.module.base import StepOutput


class AuxLoss(nn.Module):
    """Base class for a loss term computed from a :class:`StepOutput`.

    Attributes
    ----------
    requires_views
        How many forward passes over (differently corrupted copies of) the batch this term needs.
        ``1`` for anything computed from a single pass; ``2`` for a two-view contrastive term.
    """

    requires_views: int = 1

    @abstractmethod
    def forward(self, out: StepOutput, batch) -> dict[str, torch.Tensor]:
        """Compute the term.

        Parameters
        ----------
        out
            The step output, carrying one :class:`~interscale.module.base.ViewOutput` per view.
        batch
            The collated batch, for anything that lives in batch node order -- ``batch.batch`` for
            the slide a token belongs to, ``edge_index`` for its spatial neighbours. Index them
            with ``out.view.padded_node_idx`` to reach token order.

        Returns
        -------
        dict[str, torch.Tensor]
            Named scalar terms. Names are logged verbatim under the mode prefix, so a term may
            report several numbers (a total and its parts) and each will appear separately.
        """


def aligned_view_tokens(out: StepOutput) -> list[torch.Tensor]:
    """Token matrices for every view, after checking that row *i* is the same cell in each.

    Any two-view term is a statement about *pairs of the same cell*, and the one way that
    silently stops being true here is ``pad_batch``: when a graph is larger than ``max_seq_len``
    it keeps a RANDOM subset, so two passes over one batch can retain different cells. The shapes
    still match, the loss still produces a plausible number, and the model trains towards nothing.

    Returns
    -------
    list of torch.Tensor
        One ``[N_kept, E]`` matrix per view, all row-aligned.

    Raises
    ------
    ValueError
        If the views kept different cells, naming the cause and the fix.
    """
    reference = out.view.padded_node_idx
    for i, view in enumerate(out.views[1:], start=1):
        if reference is None or view.padded_node_idx is None:
            continue
        if not torch.equal(reference, view.padded_node_idx):
            raise ValueError(
                f"View 0 and view {i} kept different cells, so a per-cell pairing between them is "
                "meaningless. This happens when a graph is larger than "
                "model.global_component.parameters.max_seq_len, where pad_batch subsamples "
                "tokens at random per pass. Raise max_seq_len to at least the largest graph."
            )
    return [view.tokens() for view in out.views]


#: Name -> class. A name here must also appear in ``cfg.optim.aux_loss_weights``.
AUX_LOSSES: dict[str, type[AuxLoss]] = {}


def register_aux_loss(name: str):
    """Class decorator adding an :class:`AuxLoss` to :data:`AUX_LOSSES` under ``name``."""

    def decorator(cls: type[AuxLoss]) -> type[AuxLoss]:
        if name in AUX_LOSSES:
            raise ValueError(f"An auxiliary loss is already registered as {name!r}.")
        AUX_LOSSES[name] = cls
        return cls

    return decorator


class CompositeAuxLoss(nn.Module):
    """The enabled auxiliary terms, their weights, and the weighted sum of what they report.

    Holds the terms as a :class:`~torch.nn.ModuleDict` so any parameters they own (a projection
    head, an expander) are reachable through ``.parameters()`` and land in the optimiser.
    """

    def __init__(self, terms: dict[str, AuxLoss], weights: dict[str, float]):
        super().__init__()
        self.terms = nn.ModuleDict(terms)
        self.weights = dict(weights)

    def __bool__(self) -> bool:
        """False when nothing is enabled, so callers can skip the whole path."""
        return len(self.terms) > 0

    @property
    def requires_views(self) -> int:
        """Forward passes needed to satisfy every enabled term; 1 when none are enabled."""
        return max((t.requires_views for t in self.terms.values()), default=1)

    def forward(self, out: StepOutput, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate every enabled term.

        Returns
        -------
        total : torch.Tensor
            The weighted sum to add to the reconstruction loss. A zero scalar when nothing is
            enabled -- deliberately a tensor, so the caller needs no special case.
        reported : dict[str, torch.Tensor]
            Every scalar each term reported, unweighted, for logging. Weighting is not folded in
            because the unweighted value is the one that is comparable across runs with different
            weights.
        """
        device = out.y_pred.device
        total = torch.zeros((), device=device, dtype=out.y_pred.dtype)
        reported: dict[str, torch.Tensor] = {}

        for name, term in self.terms.items():
            values = term(out, batch)
            weight = self.weights[name]
            for key, value in values.items():
                if key in reported:
                    raise ValueError(f"Two auxiliary terms both reported {key!r}; names must be unique.")
                reported[key] = value
                total = total + weight * value

        return total, reported


def build_aux_losses(cfg: CN, module: nn.Module | None = None) -> CompositeAuxLoss:
    """Construct the auxiliary terms whose weight is non-zero.

    A weight of ``0.0`` means the term is not constructed at all, rather than constructed and
    multiplied by zero: an unbuilt term adds no parameters, runs no forward pass, and cannot
    change a number. That is what makes every one of these opt-in.

    Parameters
    ----------
    cfg
        The full config. Reads ``cfg.optim.aux_loss_weights`` and ``cfg.optim.contrastive``.
    module
        The module being trained, for terms that need its dimensions (``n_embed``). Optional so
        that the scaffolding can be built and tested without one.

    Returns
    -------
    CompositeAuxLoss
        Empty when every weight is zero, which is the default for every existing config.

    Raises
    ------
    ValueError
        If a non-zero weight names a term that is not registered -- a typo in a sweep would
        otherwise silently train without the term it claims to be measuring.
    """
    weights = {name: float(value) for name, value in dict(cfg.optim.aux_loss_weights).items()}
    enabled = {name: w for name, w in weights.items() if w != 0.0}

    unknown = sorted(set(enabled) - set(AUX_LOSSES))
    if unknown:
        raise ValueError(
            f"optim.aux_loss_weights enables unregistered auxiliary losses {unknown}; "
            f"known terms are {sorted(AUX_LOSSES)}."
        )

    terms = {name: AUX_LOSSES[name](cfg, module) for name in enabled}
    return CompositeAuxLoss(terms, enabled)
