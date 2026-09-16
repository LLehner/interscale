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

import warnings
from abc import abstractmethod

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from interscale.module.base import StepOutput
from interscale.train.vicreg import covariance_term, invariance_term, variance_term


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
            Named scalar terms, each added to the objective and logged under the mode prefix.

            A name beginning with ``_`` is **reported but not summed** -- a diagnostic rather
            than part of the objective. Embedding standard deviation is the motivating case:
            it is the number that reveals a slow collapse, and it must not also be optimised.
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
            enabled -- deliberately a tensor, so the caller needs no special case. Names
            beginning with ``_`` are excluded: they are diagnostics, not objectives.
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
                if not key.startswith("_"):
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


def build_expander(n_input: int, dims: list[int]) -> nn.Module:
    """The MLP VICReg's terms are computed on: ``Linear -> BN -> ReLU`` per hidden, then Linear.

    The paper puts the variance and covariance terms on an *expanded* embedding -- 8192-d against
    a 2048-d representation -- and its own sweep says the width matters a great deal (Table 12:
    256-d reaches 55.9%, 8192-d 68.6%). This model's ``n_embed`` is 16, where decorrelating every
    dimension is a far harsher constraint and competes directly with the decoder's need for them.
    Expanding first is what keeps the regulariser off the representation itself.

    An empty ``dims`` gives the identity, which applies the terms to the embedding directly -- a
    deliberate option, and the ablation that says whether the expander earned its parameters.
    """
    if not dims:
        return nn.Identity()
    layers: list[nn.Module] = []
    in_dim = n_input
    for width in dims[:-1]:
        layers += [nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU(inplace=True)]
        in_dim = width
    layers.append(nn.Linear(in_dim, dims[-1]))
    return nn.Sequential(*layers)



def _encoder_dropout(cfg) -> float:
    """Largest dropout rate configured anywhere in the encoder.

    Two passes over one batch currently differ only through the encoder's own stochasticity, so
    this is what decides whether a two-view term has anything to compare. Reads defensively: a
    component may be absent (``GlobalModel`` has no local one) or carry no ``parameters`` node at
    all (``Precomputed``).
    """
    rates = [0.0]
    for component, name in (
        (getattr(cfg.model, "local_component", None), "dropout_local"),
        (getattr(cfg.model, "global_component", None), "dropout_global"),
    ):
        params = getattr(component, "parameters", None) if component is not None else None
        rate = getattr(params, name, None) if params is not None else None
        if rate is not None:
            rates.append(float(rate))
    return max(rates)


@register_aux_loss("vicreg")
class VICRegAuxLoss(AuxLoss):
    """VICReg as an auxiliary term: variance + covariance always, invariance when two views run.

    Three configurations, all reached from the same class:

    * ``vicreg_mu``/``vicreg_nu`` only -- one forward pass, no pairing, no negatives. A collapse
      regulariser beside the reconstruction loss; the cheapest useful thing in the plan.
    * all three coefficients, with ``optim.loss`` still set -- the hybrid.
    * all three, with ``optim.loss: none`` -- VICReg as the whole objective.

    ``requires_views`` is 2 exactly when the invariance coefficient is non-zero, so asking for
    the invariance term is what turns two-view mode on. Nothing else needs setting, and nothing
    else can be forgotten.

    Which embedding the terms read is ``optim.contrastive.embedding``: ``"global"`` for the
    transformer's per-cell tokens, ``"local"`` for the graph component's, or ``"auto"`` (the
    default) for whichever the model has -- which is what lets one config block serve
    ``LocalModel``, ``GlobalModel`` and ``CombinedModel``.
    """

    def __init__(self, cfg, module=None):
        super().__init__()
        contrastive = cfg.optim.contrastive
        self.lam = float(contrastive.vicreg_lambda)
        self.mu = float(contrastive.vicreg_mu)
        self.nu = float(contrastive.vicreg_nu)
        self.embedding = contrastive.embedding
        self.group_by = contrastive.vicreg_group

        # An instance attribute shadowing the class one: how many passes to run is a property of
        # THIS configuration, not of the class.
        self.requires_views = 2 if self.lam > 0 else 1

        n_embed = getattr(module, "n_embed", None) if module is not None else None
        self.expander = build_expander(n_embed, list(contrastive.expander_dims)) if n_embed else nn.Identity()

        if self.lam == 0 and self.mu == 0 and self.nu == 0:
            raise ValueError(
                "aux_loss_weights.vicreg is non-zero but every VICReg coefficient "
                "(optim.contrastive.vicreg_lambda/mu/nu) is 0, so the term contributes nothing."
            )
        if self.mu == 0 and self.lam > 0:
            warnings.warn(
                "VICReg with an invariance term but no variance term (vicreg_mu = 0) collapses: "
                "mapping every cell to one point satisfies invariance exactly. The paper's "
                "Table 7 reports collapse for every such combination.",
                UserWarning,
                stacklevel=2,
            )
        if self.lam > 0 and _encoder_dropout(cfg) == 0:
            warnings.warn(
                "vicreg_lambda > 0 asks for an invariance term between two views, but every "
                "encoder dropout is 0 -- and dropout is currently the only thing that makes two "
                "passes differ. The views are then identical, the invariance term is exactly 0.0, "
                "and VICReg degenerates to variance+covariance, which its own Table 7 reports as "
                "collapse. Set model.local_component.parameters.dropout_local or "
                "model.global_component.parameters.dropout_global above 0, or wait for the "
                "Stage 3 view sampler, which corrupts the input instead.",
                UserWarning,
                stacklevel=2,
            )

    def _embeddings(self, out: StepOutput) -> list[torch.Tensor]:
        """One ``[N, E]`` matrix per view, from whichever embedding is configured."""
        wants_global = self.embedding == "global" or (
            self.embedding == "auto" and out.view.global_embedding is not None
        )
        if wants_global:
            if out.view.global_embedding is None:
                raise ValueError(
                    "optim.contrastive.embedding is 'global' but this module produced none. Use "
                    "'local' or 'auto' for a LocalModel."
                )
            return aligned_view_tokens(out)

        if out.view.local_embedding is None:
            raise ValueError(
                "optim.contrastive.embedding is 'local' but this module produced none. Use "
                "'global' or 'auto' for a GlobalModel."
            )
        return [view.local_embedding for view in out.views]

    def _groups(self, out: StepOutput, batch, n_rows: int) -> torch.Tensor | None:
        """Row labels for the per-slide variance hinge, or None to pool the whole batch.

        Prefers an attached ``slide`` annotation over the graph index, because several windows
        can come from one slide and it is the SLIDE that carries the batch effect the grouping
        exists to keep out of the hinge.
        """
        if self.group_by == "none":
            return None

        index = out.view.padded_node_idx
        slide = getattr(batch, "slide", None)
        if slide is not None:
            codes = slide.argmax(dim=1) if slide.dim() > 1 else slide
            return codes[index] if index is not None else codes[:n_rows]

        graph = getattr(batch, "batch", None)
        if graph is None:
            return None
        return graph[index] if index is not None else graph[:n_rows]

    def forward(self, out: StepOutput, batch) -> dict[str, torch.Tensor]:
        """The VICReg objective, with each coefficient's contribution reported separately."""
        embeddings = self._embeddings(out)
        projected = [self.expander(z) for z in embeddings]
        groups = self._groups(out, batch, projected[0].shape[0])

        terms: dict[str, torch.Tensor] = {}

        if self.mu:
            variance = torch.stack([variance_term(z, groups=groups) for z in projected]).mean()
            terms["vicreg_var"] = self.mu * variance
        if self.nu:
            covariance = torch.stack([covariance_term(z) for z in projected]).mean()
            terms["vicreg_cov"] = self.nu * covariance
        if self.lam:
            terms["vicreg_inv"] = self.lam * invariance_term(projected[0], projected[1])

        # Not summed (leading underscore): the collapse diagnostic, which VICReg's Figure 4 reads
        # to see a representation shrinking long before a loss curve shows it.
        terms["_vicreg_std"] = projected[0].std(dim=0).mean().detach()
        return terms
