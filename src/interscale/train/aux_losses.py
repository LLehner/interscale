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
from interscale.train.context_nce import (
    build_inverse,
    composition,
    histogram_gap,
    info_nce,
    khop_membership,
    match_composition,
    overlaps,
    pool_contexts,
    selection_is_clean,
    to_token_edges,
    valid_contexts,
)
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



def build_projector(n_input: int, dims: list[int]) -> nn.Module:
    """The MLP the InfoNCE terms read through -- same shape as the expander, different job.

    A separate head from VICReg's on purpose: the two families disagree about normalisation.
    InfoNCE reads an L2-normalised projection (SimCLR Table 5: 64.4 against 57.2 without it),
    VICReg explicitly does not (its Table 8 reports L2 costing 3.5%). Sharing one head would force
    one of them onto the other's convention, and nothing about the resulting number would say so.

    The normalisation itself lives in :func:`~interscale.train.context_nce.info_nce` rather than
    here, so the head stays a plain MLP and the loss owns its own convention.
    """
    return build_expander(n_input, dims)


def slide_codes(batch, index: torch.Tensor | None, n_rows: int, group_by: str = "auto") -> torch.Tensor | None:
    """Per-row slide label, in the row order of whatever embedding the caller is holding.

    Prefers an attached ``slide`` annotation over the graph index, because several windows can
    come from one slide and it is the SLIDE that carries the batch effect every consumer here is
    trying to keep out of its statistics -- the VICReg variance hinge and the contrastive negative
    pool alike.

    One function rather than one per term: both need exactly this mapping, and two copies of it
    would drift the moment one of them learned about a new field.

    Parameters
    ----------
    batch
        The collated batch.
    index
        ``padded_node_idx``, to bring a batch-node-order field into token order, or ``None`` when
        the caller's rows are already in batch node order.
    n_rows
        Number of rows the caller holds, used only when ``index`` is ``None``.
    group_by
        ``"auto"`` for slide-then-graph as described; ``"none"`` to pool everything (returns
        ``None``).

    Returns
    -------
    torch.Tensor | None
        ``[n_rows]`` integer codes, or ``None`` when there is nothing to group by.
    """
    if group_by == "none":
        return None

    slide = getattr(batch, "slide", None)
    if slide is not None:
        codes = slide.argmax(dim=1) if slide.dim() > 1 else slide
        return codes[index] if index is not None else codes[:n_rows]

    graph = getattr(batch, "batch", None)
    if graph is None:
        return None
    return graph[index] if index is not None else graph[:n_rows]


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
        """Row labels for the per-slide variance hinge, or None to pool the whole batch."""
        return slide_codes(batch, out.view.padded_node_idx, n_rows, self.group_by)

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


@register_aux_loss("context_nce")
class ContextNCE(AuxLoss):
    """A cell against its own neighbourhood, with composition-matched negative contexts.

    Stage 2 of ``.claude/contrastive_plan.md``, and the first term here that makes a positive
    claim about *content* rather than about the shape of the embedding distribution: a cell's
    representation should be predictive of the tissue context it sits in. VICReg says "do not
    collapse", which an embedding of pure slide identity satisfies perfectly.

    One forward pass. The positive pair is not two augmented copies of one cell but two parts of
    one sample -- the anchor token and its pooled k-hop neighbourhood -- so the spatial graph
    supplies the pairing and ``requires_views`` stays 1. This is the CPC / Deep InfoMax family
    rather than the SimCLR one, which is also why the term is a real number in validation: the
    pairing is structural, where dropout-only views coincide under ``eval()`` and make
    ``val_vicreg_inv`` identically zero.

    The negatives are the design. Per slide, a **bank** of candidate contexts is built once per
    step and shared by every anchor on that slide -- negatives are contexts, not cells, so
    building them per anchor would cost ``n_anchors * n_negatives`` k-hop expansions instead of
    ``n_candidates``. Each anchor then takes the candidates whose cell-type histogram is closest
    to its own positive context's, having first discarded every candidate whose context overlaps
    its own. Composition is thereby held fixed across the positive and the negatives, so it
    carries no information about which is which, and the only way left to reduce the loss is to
    encode which specific states are present and how they are arranged.

    Two diagnostics are reported and not summed: the fraction of anchors whose negatives were all
    genuinely non-overlapping, and the fraction where the positive came out on top. The first is
    the one to watch -- on a small slide, or with ``context_hops`` high enough that every
    neighbourhood touches every other, most rows can be filled with false negatives, and the loss
    looks entirely normal while that happens.
    """

    requires_views = 1

    def __init__(self, cfg, module=None):
        super().__init__()
        contrastive = cfg.optim.contrastive
        self.temperature = float(contrastive.temperature)
        self.embedding = contrastive.embedding
        self.hops = int(contrastive.context_hops)
        self.n_anchors = int(contrastive.n_anchors)
        self.n_negatives = int(contrastive.n_negatives)
        self.n_candidates = int(contrastive.n_candidates)
        self.negative_context = contrastive.negative_context
        self.match = bool(contrastive.match_composition)
        self.masked_only = bool(contrastive.anchors_masked_only)

        if self.negative_context not in ("neighbourhood", "scattered"):
            raise ValueError(
                "optim.contrastive.negative_context must be 'neighbourhood' or 'scattered', "
                f"got {self.negative_context!r}."
            )
        if self.hops < 1:
            raise ValueError(f"optim.contrastive.context_hops must be >= 1, got {self.hops}.")
        if self.n_negatives < 1:
            raise ValueError(f"optim.contrastive.n_negatives must be >= 1, got {self.n_negatives}.")
        if self.n_candidates <= self.n_negatives:
            raise ValueError(
                f"optim.contrastive.n_candidates ({self.n_candidates}) must exceed n_negatives "
                f"({self.n_negatives}): the bank is what the composition match selects FROM, so "
                "with no surplus every candidate is taken regardless of its histogram and the "
                "matching -- the whole point of the term -- silently does nothing."
            )

        n_embed = getattr(module, "n_embed", None) if module is not None else None
        self.projector = build_projector(n_embed, list(contrastive.projector_dims)) if n_embed else nn.Identity()

    # ------------------------------------------------------------------ inputs

    def _features(self, out: StepOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """``([N_rows, E], [N_rows])`` -- the embedding to contrast and each row's batch node id.

        The node id is what every structural lookup here goes through, and the two embeddings
        reach it differently: global tokens are a padded subset named by ``padded_node_idx``,
        while a local embedding is already one row per batch node.
        """
        view = out.view
        wants_global = self.embedding == "global" or (self.embedding == "auto" and view.global_embedding is not None)

        if wants_global:
            if view.global_embedding is None:
                raise ValueError(
                    "optim.contrastive.embedding is 'global' but this module produced none. Use "
                    "'local' or 'auto' for a LocalModel."
                )
            if view.padded_node_idx is None:
                raise ValueError(
                    "context_nce needs padded_node_idx to map tokens onto edge_index, and this "
                    "step produced none -- graph-level prediction does not gather per-cell "
                    "tokens. The term is node-level only."
                )
            return view.tokens(), view.padded_node_idx

        if view.local_embedding is None:
            raise ValueError(
                "optim.contrastive.embedding is 'local' but this module produced none. Use "
                "'global' or 'auto' for a GlobalModel."
            )
        local = view.local_embedding
        return local, torch.arange(local.shape[0], device=local.device)

    def _celltypes(self, batch, node_idx: torch.Tensor) -> tuple[torch.Tensor, int]:
        """``([N_rows], n_types)`` cell-type codes per row.

        Without an attached annotation there is nothing to match on, and an unmatched negative
        makes the term a composition classifier -- the exact thing it exists to rule out. So this
        raises rather than falling back to a single pseudo-type, which would leave the run looking
        healthy while measuring something else entirely.
        """
        if not self.match:
            # The `match_composition: False` ablation ranks candidates at random, so no histogram
            # is consulted and no annotation is needed. Demanding one anyway would make the
            # ablation unavailable on exactly the datasets where it is cheapest to run.
            return torch.zeros(len(node_idx), dtype=torch.long, device=node_idx.device), 1

        celltype = getattr(batch, "celltype", None)
        if celltype is None:
            raise ValueError(
                "optim.contrastive.match_composition is True but no cell-type annotation is "
                "attached. Set dataset.celltype_key, or set match_composition: False to draw "
                "negatives uniformly -- but note that an unmatched negative lets cell-type "
                "composition alone separate positive from negative, which is what this term is "
                "designed to prevent."
            )
        codes = celltype.argmax(dim=1) if celltype.dim() > 1 else celltype.long()
        n_types = celltype.shape[1] if celltype.dim() > 1 else int(codes.max().item()) + 1
        return codes[node_idx], n_types

    def _anchor_pool(self, batch, node_idx: torch.Tensor) -> torch.Tensor:
        """``[N_rows]`` boolean: which rows may serve as anchors.

        Under ``anchors_masked_only`` the answer is the masked cells, and the reason is the
        identity shortcut in its graph-shaped form: an unmasked anchor's own expression is in the
        encoder input *and* -- because the GCN aggregates over neighbourhoods -- inside its own
        neighbours' embeddings, so the positive context is identifiable by detecting the anchor's
        own transcriptome in it. Masking the anchor removes both copies at once.
        """
        if not self.masked_only:
            return torch.ones(len(node_idx), dtype=torch.bool, device=node_idx.device)
        mask = getattr(batch, "mask", None)
        if mask is None:
            return torch.ones(len(node_idx), dtype=torch.bool, device=node_idx.device)
        return mask.bool()[node_idx]

    # ------------------------------------------------------------------ sampling

    @staticmethod
    def _sample(pool: torch.Tensor, n: int) -> torch.Tensor:
        """``min(n, len(pool))`` entries of ``pool`` without replacement; all of it when ``n`` is 0."""
        if n <= 0 or len(pool) <= n:
            return pool
        return pool[torch.randperm(len(pool), device=pool.device)[:n]]

    def _scattered(self, sizes: torch.Tensor, in_slide: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """``[C, n_tokens]`` random cell sets of the given sizes, drawn from one slide.

        The ablation arm. Sizes are copied from the neighbourhood contexts they replace, so the
        only variable this changes is contiguity -- if it were also free to change context size,
        size would become the discriminator and the comparison would say nothing.
        """
        scores = torch.rand((len(sizes), n_tokens), device=sizes.device)
        outside = torch.ones(n_tokens, dtype=torch.bool, device=sizes.device)
        outside[in_slide] = False
        scores[:, outside] = -1.0

        order = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(n_tokens, device=sizes.device).expand(len(sizes), -1))
        return ranks < sizes.clamp(max=len(in_slide)).unsqueeze(1)

    # ------------------------------------------------------------------ the term

    def forward(self, out: StepOutput, batch) -> dict[str, torch.Tensor]:
        """InfoNCE over anchors from every slide in the batch, pooled into one scalar."""
        features, node_idx = self._features(out)
        projected = self.projector(features)
        n_tokens = projected.shape[0]
        device = projected.device

        inverse = build_inverse(node_idx, int(batch.num_nodes))
        token_edges = to_token_edges(batch.edge_index, inverse)
        slides = slide_codes(batch, node_idx, n_tokens)
        if slides is None:
            slides = torch.zeros(n_tokens, dtype=torch.long, device=device)
        types, n_types = self._celltypes(batch, node_idx)
        eligible = self._anchor_pool(batch, node_idx)

        losses: list[torch.Tensor] = []
        clean: list[torch.Tensor] = []
        correct: list[torch.Tensor] = []
        gap: list[torch.Tensor] = []

        for slide in slides.unique():
            rows = (slides == slide).nonzero(as_tuple=True)[0]
            anchor_pool = rows[eligible[rows]]
            if len(anchor_pool) == 0 or len(rows) < 2:
                continue

            anchors = self._sample(anchor_pool, self.n_anchors)
            centers = self._sample(rows, self.n_candidates)

            anchor_members = khop_membership(anchors, token_edges, n_tokens, self.hops)
            anchor_pooled, anchor_sizes = pool_contexts(anchor_members, projected)
            keep = valid_contexts(anchor_sizes)
            if not bool(keep.any()):
                continue

            candidate_members = khop_membership(centers, token_edges, n_tokens, self.hops)
            _, candidate_sizes = pool_contexts(candidate_members, projected)
            if self.negative_context == "scattered":
                candidate_members = self._scattered(candidate_sizes, rows, n_tokens)
            candidate_pooled, candidate_sizes = pool_contexts(candidate_members, projected)

            anchors, anchor_members = anchors[keep], anchor_members[keep]
            anchor_pooled = anchor_pooled[keep]

            # A candidate is unusable if its context is empty, if it shares a cell with the
            # anchor's context (same context under another name), or if it CONTAINS the anchor --
            # the last is not implied by the first two, since the anchor is excluded from its own
            # context, and it is the most direct false negative of the three.
            rejected = overlaps(anchor_members, candidate_members)
            rejected = rejected | candidate_members[:, anchors].t()
            rejected = rejected | ~valid_contexts(candidate_sizes).unsqueeze(0)

            anchor_hist = composition(anchor_members, types, n_types)
            candidate_hist = composition(candidate_members, types, n_types)
            selected = match_composition(
                anchor_hist, candidate_hist, rejected, self.n_negatives, match=self.match
            )

            negatives = candidate_pooled[selected]  # [A, K, D]
            losses.append(info_nce(projected[anchors], anchor_pooled, negatives, self.temperature))
            clean.append(selection_is_clean(selected, rejected).to(projected.dtype))
            gap.append(histogram_gap(anchor_hist, candidate_hist, selected).detach())
            with torch.no_grad():
                sim_pos = torch.nn.functional.cosine_similarity(projected[anchors], anchor_pooled, dim=-1)
                sim_neg = torch.einsum(
                    "ad,akd->ak",
                    torch.nn.functional.normalize(projected[anchors], dim=-1),
                    torch.nn.functional.normalize(negatives, dim=-1),
                )
                correct.append((sim_pos.unsqueeze(1) > sim_neg).all(dim=1).to(projected.dtype))

        if not losses:
            # Nothing scoreable this step -- every slide too small, or no masked anchor survived.
            # Returned through `projected` so the value stays attached to the graph: a detached
            # zero would make this term the only one in a `optim.loss: none` run with no gradient
            # path at all, which errors rather than no-ops.
            zero = projected.sum() * 0.0
            return {
                "context_nce": zero,
                "_context_nce_clean": zero.detach(),
                "_context_nce_acc": zero.detach(),
                "_context_nce_hist_gap": zero.detach(),
            }

        return {
            "context_nce": torch.stack(losses).mean(),
            # Not summed: diagnostics. `clean` is the one that matters -- it is the fraction of
            # anchors whose negatives were all genuinely non-overlapping, and a low value means
            # the denominator is full of false negatives while the loss curve looks fine.
            "_context_nce_clean": torch.cat(clean).mean().detach(),
            "_context_nce_acc": torch.cat(correct).mean().detach(),
            # The matching's own report card: how far the chosen negatives' cell-type histograms
            # actually sat from the positive's. `clean` says the negatives were not false; this
            # says they were not trivially distinguishable by composition, which is the claim the
            # whole stage rests on. Read against the same number from a `match_composition: False`
            # run -- alone it has no scale.
            "_context_nce_hist_gap": torch.cat(gap).mean().detach(),
        }
