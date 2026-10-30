"""Hierarchical coarse-to-fine attention (``.claude/hierarchical_attention_plan.md``, T2-T6).

Each layer of the global transformer makes one coarse-to-fine pass over a slide's cluster hierarchy
(:mod:`~interscale.module.global_modules.hierarchy`): levels ``0`` (regions) .. ``L - 1``, then the cells (level ``L``).

1. **Pool bottom-up** (the vertical block): the tokens of level ``l - 1`` from those of level ``l`` -- ``mean`` over the
   visible cells, or ``attn_pool`` (a learned query per level, starting as the mean). A cell hidden whole weighs 0, so a
   cluster token never holds a hidden cell's value; a cluster with no visible cell gets a learned token.
2. **Level 0:** dense attention among the region tokens and CLS, with the distance bias between centroids. CLS queries here
   only, and stays an open key with bias 0 at every level.
3. **Levels 1 .. L:** attention among the level's tokens, steered by the attention of the level above (``prior_mode``):
   ``none`` (levels independent), ``soft`` (``+ lambda_h log(P + eps)``, ``P_ij = A_{p(i), p(j)} / n_{p(j)}``),
   ``factorized`` (``A_ij = A_{p(i), p(j)} softmax_{j' in p(j)}(s_ij')``, a hierarchical softmax), ``factorized_percell``
   (each query weighs the clusters itself, ``softmax_c(q_i . k_c + B_ic + lambda_h log A_{p(i), c})``, then the
   within-cluster softmax). ``direction_prior`` adds ``mu_h conf_{p(i)} cos(x_j - x_i, v_{p(i)})``, the parent's coarse
   arrow. Head ``h`` reads the prior of head ``h``.
4. **Candidates** (``candidates``): ``dense`` -- every token of the level is a key (the prior without pruning); ``routed`` --
   the queries of one parent share their keys: the children of the parent's top-k targets, of its spatial neighbours (itself
   included) and of ``explore_n`` exploration parents, at most ``cap_per_query`` keys. Computed a block of parents at a time
   and never as an ``N x N`` matrix.
5. **Readout:** each cell's output fused with its ancestors' (``none``, ``concat``, ``transformer``), then the layer's output
   projection; the residual, norms and feed-forward block are the encoder layer's own.

With ``prior_mode soft``, ``lambda`` 0, dense candidates and one level of singleton clusters (or no coarse level at all) the
layer computes the transformer's dense attention exactly, with the same weights (tests).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from interscale.module.global_modules.transformer_encoder_layer import (
    CustomTransformerEncoderLayer,
    mean_pairwise_cosine,
)

HIERARCHY_KINDS = ("none", "coarse_to_fine")
PRIOR_MODES = ("none", "soft", "factorized", "factorized_percell")
HIER_POOLINGS = ("mean", "attn_pool")
HIER_POOL_FEATURES = ("count", "spread")
READOUTS = ("none", "concat", "transformer")
CANDIDATE_MODES = ("dense", "routed")
EXPLORE_STRATEGIES = ("uniform", "prior_sampled")
LEARNABLE = ("per_head", "global", "fixed")
PRIOR_REFRESH = ("per_layer", "once")
COARSE_BIAS = ("shared", "own", "none")
#: Added to a prior before its logarithm, so a cluster the parent ignores gets a large finite penalty, not -inf.
PRIOR_EPS = 1e-6
#: Floor of the parent weight in the factorized modes, whose logarithm is the weight itself.
LOG_FLOOR = 1e-30


@dataclass(frozen=True)
class HierarchySettings:
    """``model.global_component.parameters.hierarchy``, parsed (:func:`hierarchy_settings`)."""

    levels: tuple = ()
    cells_per_cluster: tuple = ()
    builder: str = "kdtree"
    top_mode: str = "tree"
    patch_size_um: float = 0.0
    min_cells: int = 1
    split_by_space: str = "components"
    top_label: str = ""  # obs_label: the Data attribute the label arrives under (geome_utils.label_attribute)
    shift: bool = False
    neighbors: int = 8
    pooling: str = "mean"
    pool_features: tuple = ()
    prior_mode: str = "soft"
    size_correction: bool = True
    direction_prior: bool = False
    lambda_init: float = 1.0
    lambda_learnable: str = "per_head"
    mu_init: float = 0.5
    mu_learnable: str = "per_head"
    prior_refresh: str = "per_layer"
    share_level_params: bool = True
    coarse_bias: str = "shared"
    readout: str = "none"
    candidates: str = "dense"
    topk: tuple = (8,)
    explore_n: int = 0
    explore_strategy: str = "uniform"
    cap_per_query: int = 512
    chunk_pairs: int = 2**24
    seed: int = 0
    spatial_unit_um: float = 1.0

    @property
    def n_levels(self) -> int:
        return len(self.levels) or len(self.cells_per_cluster)

    def build_kwargs(self, shift: bool = False) -> dict:
        """What :func:`~interscale.module.global_modules.hierarchy.build_hierarchy` takes."""
        return {
            "levels": tuple(int(c) for c in self.levels),
            "cells_per_cluster": tuple(float(c) for c in self.cells_per_cluster),
            "builder": self.builder,
            "top_mode": self.top_mode,
            "patch_size_um": float(self.patch_size_um),
            "min_cells": int(self.min_cells),
            "split_by_space": self.split_by_space,
            "neighbors": int(self.neighbors),
            "shift": bool(shift),
            "seed": int(self.seed),
        }

    def topk_at(self, level: int) -> int:
        """The top-k of the routed level ``level`` (1 .. L): ``topk``'s entry for it, or its last."""
        values = tuple(self.topk) or (8,)
        return int(values[min(level - 1, len(values) - 1)])


def hierarchy_settings(cfg) -> HierarchySettings | None:
    """The configured :class:`HierarchySettings`, or None for ``kind: none``."""
    params = cfg.model.global_component.get("parameters", None)
    hier = params.get("hierarchy", None) if params is not None else None
    if hier is None or hier.get("kind", "none") == "none":
        return None
    cand = hier.candidates
    top_label = ""
    if hier.top_level.mode == "obs_label":
        from interscale.tl.geome_utils import label_attribute

        top_label = label_attribute(cfg, str(hier.top_level.obs_key))
    return HierarchySettings(
        levels=tuple(int(c) for c in hier.levels),
        cells_per_cluster=tuple(float(c) for c in hier.cells_per_cluster),
        builder=str(hier.builder),
        top_mode=str(hier.top_level.mode),
        patch_size_um=float(hier.top_level.patch_size_um),
        min_cells=int(hier.top_level.min_cells),
        split_by_space=str(hier.top_level.get("split_by_space", "components")),
        top_label=top_label,
        shift=bool(hier.shift),
        neighbors=int(hier.neighbors),
        pooling=str(hier.pooling),
        pool_features=tuple(hier.pool_features),
        prior_mode=str(hier.prior_mode),
        size_correction=bool(hier.size_correction),
        direction_prior=bool(hier.direction_prior),
        lambda_init=float(hier.lambda_init),
        lambda_learnable=str(hier.lambda_learnable),
        mu_init=float(hier.mu_init),
        mu_learnable=str(hier.mu_learnable),
        prior_refresh=str(hier.prior_refresh),
        share_level_params=bool(hier.share_level_params),
        coarse_bias=str(hier.coarse_bias),
        readout=str(hier.readout),
        candidates=str(cand.mode),
        topk=tuple(int(k) for k in cand.topk),
        explore_n=int(cand.explore_n),
        explore_strategy=str(cand.explore_strategy),
        cap_per_query=int(cand.cap_per_query),
        chunk_pairs=int(cand.chunk_pairs),
        seed=int(cfg.optim.get("seed", 0) or 0),
        spatial_unit_um=float(cfg.dataset.get("spatial_unit_um", 1.0)),
    )


@dataclass
class GraphLayout:
    """One graph of a batch, as the hierarchical layers read it."""

    start: int  # its first cell's position in the padded sequence (left padding: cells fill start .. S - 1)
    pos_um: torch.Tensor  # [n, 2] the kept cells' positions, in sequence order
    visible: torch.Tensor  # [n] 1 for a cell with a visible entry, 0 for one hidden whole
    hierarchies: list  # one Hierarchy, or two under shift (layers alternate)
    nodes: list | None = None  # the graph-local node indices pad_batch kept, in sequence order


class HierarchicalContext:
    """What the hierarchical layers need for one pass besides the tokens: the graphs' layout and hierarchies, the biases.

    The value of ``GlobalInput.attention_mask`` for a hierarchical transformer (no ``S x S`` mask is formed). ``cache`` holds
    the first layer's coarse attention under ``prior_refresh: once``; ``record``, when a list, receives every layer's export
    (:meth:`HierarchicalEncoderLayer._export`).
    """

    def __init__(self, graphs: list[GraphLayout], *, bias=None, coarse_biases=None, generator=None, training=False):
        self.graphs = graphs
        self.bias = bias
        self.coarse_biases = coarse_biases  # None (no bias above the cells), or one per coarse level
        self.generator = generator
        self.training = training
        self.cache: dict = {}
        self.record: list | None = None


def offset_bias(bias, offsets_um: torch.Tensor) -> torch.Tensor | None:
    """The enabled attention biases (``positional_encodings.AttentionBias``) at offsets ``key - query`` ``[..., 2]`` (µm): ``[..., heads]``.

    Biases of the distance (every ``distance`` kind) and of the offset (``fourier``); none other has a form for an arbitrary
    pair (the validator refuses them). None when ``bias`` is None.
    """
    if bias is None:
        return None
    total = 0
    for name, encoder in bias.encoders.items():
        need = bias.requires[name]
        if need == "pos":
            total = total + encoder._lookup(offsets_um.norm(dim=-1))
        elif need == "pos_coords":
            phase = offsets_um.double() @ encoder.frequencies.double().T
            a, c = encoder.coefficients()
            total = total + (phase.cos().to(offsets_um.dtype) @ a.T + phase.sin().to(offsets_um.dtype) @ c.T)
        else:
            raise ValueError(
                f"the attention bias '{name}' reads neither distances nor offsets; the hierarchy cannot place it."
            )
    return total


def segment_logsumexp(scores: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """``[..., n_groups]``: the log-sum-exp of ``scores`` ``[..., K]`` over the keys of each group (``index`` ``[K]``).

    A group whose every key is blocked (-inf) gets a large negative finite value with a zero gradient, never -inf or a NaN
    gradient (``torch.logsumexp`` gives NaN gradients there): its keys are blocked anyway.
    """
    shape = scores.shape[:-1] + (n_groups,)
    expanded = index.expand_as(scores)
    top = scores.new_full(shape, float("-inf")).scatter_reduce(-1, expanded, scores, "amax", include_self=True)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top)).detach()
    total = scores.new_zeros(shape).scatter_add(-1, expanded, (scores - top.gather(-1, expanded)).exp())
    return total.clamp_min(LOG_FLOOR).log() + top


def safe_logsumexp(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``torch.logsumexp`` with the fully blocked case of :func:`segment_logsumexp`: finite, and a zero gradient."""
    top = scores.amax(dim, keepdim=True)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top)).detach()
    return ((scores - top).exp().sum(dim, keepdim=True).clamp_min(LOG_FLOOR).log() + top).squeeze(dim)


def within_cluster(scores: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """``scores - lse`` (the log of the within-cluster softmax), -inf where the score is (a blocked key), never NaN."""
    blocked = torch.isneginf(scores)
    return torch.where(
        blocked, torch.full_like(scores, float("-inf")), scores - torch.where(blocked, torch.zeros_like(lse), lse)
    )


@dataclass
class SparseRows:
    """A routed level's attention: the queries of one parent (a group) share their keys.

    ``keys`` ``[G, K]`` the key ids per group (the level's token ids, CLS = the level's count, -1 padding); ``weights``
    ``[G, H, m, K]`` per group, head and query slot; ``q_group`` / ``q_slot`` ``[n]`` where each query sits.
    """

    keys: torch.Tensor
    weights: torch.Tensor
    q_group: torch.Tensor
    q_slot: torch.Tensor

    def rows(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(keys [len(ids), K], weights [H, len(ids), K])``: the rows of the queries ``ids``."""
        group, slot = self.q_group[ids], self.q_slot[ids]
        return self.keys[group], self.weights[group, :, slot].permute(1, 0, 2)

    def detach_cpu(self) -> SparseRows:
        return SparseRows(self.keys.cpu(), self.weights.detach().cpu(), self.q_group.cpu(), self.q_slot.cpu())


def prior_rows(attention, ids: torch.Tensor, n_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(keys [Q, K], weights [H, Q, K])``: rows ``ids`` of a level's attention, dense ``[H, Q', n + 1]`` or :class:`SparseRows`."""
    if isinstance(attention, SparseRows):
        return attention.rows(ids)
    keys = torch.arange(n_tokens + 1, device=attention.device).expand(len(ids), -1)
    return keys, attention[:, ids]


def lookup(keys: torch.Tensor, weights: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """``[H, Q, W]``: the weight of each row of ``keys``/``weights`` at the key ids ``wanted`` ``[Q, W]`` (0 where absent)."""
    sorted_keys, order = keys.sort(-1)
    at = torch.searchsorted(sorted_keys, wanted.clamp_min(0)).clamp_max(keys.shape[-1] - 1)
    hit = (sorted_keys.gather(-1, at) == wanted) & (wanted >= 0)
    column = order.gather(-1, at)
    found = weights.gather(-1, column.expand(weights.shape[0], -1, -1))
    return torch.where(hit, found, torch.zeros_like(found))


def coarse_direction(keys, weights, centroid, n_tokens):
    """Per row, the parent's coarse arrow and its confidence, from its attention over clusters (CLS left out).

    ``keys`` ``[Q, K]``, ``weights`` ``[H, Q, K]`` (the rows of the queries' parents), ``centroid`` ``[n, 2]``: returns
    ``v`` ``[H, Q, 2]`` (``sum_c' A_{c c'} (x_c' - x_c)``, the row's cluster ``c`` given by ``rows``) is computed by the caller;
    here ``(target [H, Q, 2], conf [H, Q])``: the weighted mean target position and ``1 - H / log |C|``.
    """
    cluster = (keys >= 0) & (keys < n_tokens)
    w = torch.where(cluster, weights, torch.zeros_like(weights))
    mass = w.sum(-1, keepdim=True)
    share = w / mass.clamp_min(1e-12)
    target = torch.einsum("hqk,qkc->hqc", share, centroid[keys.clamp(0, n_tokens - 1)])
    entropy = -(share * share.clamp_min(1e-30).log()).sum(-1)
    count = cluster.sum(-1).clamp_min(2).to(weights.dtype)
    conf = (1.0 - entropy / count.log()).clamp(0.0, 1.0)
    return target, conf


def direction_cosine(offsets: torch.Tensor, arrow: torch.Tensor, scale) -> torch.Tensor:
    """The cosine of the angle between ``offsets`` and ``arrow`` (last dims), softened for short arrows.

    The arrow is normalised by ``sqrt(|arrow|^2 + scale^2)`` instead of ``|arrow|``: an arrow much shorter than ``scale`` (a
    parent that reads mostly itself, as every narrow head does) carries no direction, and the gradient through it is bounded
    by ``1 / scale``. Dividing by ``|arrow|`` itself sent gradients of ~1e11 through near-zero arrows, and Adam then froze every
    parameter (synth_spot Stage A, 2026-10-26). The offsets carry no gradient; a zero offset (a key on its query) gives 0.
    """
    unit = offsets / offsets.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    soft = arrow / (arrow.pow(2).sum(-1, keepdim=True) + scale**2).sqrt()
    return (unit * soft).sum(-1)


def _weight(shape, init, learnable):
    value = torch.full(shape, float(init))
    return nn.Parameter(value) if learnable != "fixed" else value


class HierarchicalEncoderLayer(CustomTransformerEncoderLayer):
    """A pre-norm encoder layer whose attention block is one coarse-to-fine pass (module docstring).

    The attention's projections are the layer's ``self_attn`` ones (``in_proj``, ``out_proj``), so at the default the layer
    has the dense layer's weights at the same seed; every parameter it adds starts at zero or comes from a private generator,
    drawing nothing from the global one. The per-pass layout arrives as ``hier_context`` (a :class:`HierarchicalContext`),
    set and cleared around each call by ``TransformerNodeEncoderHook.forward``.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation, *, settings, center_qk=False
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            norm_first=True,
            attention_factory=None,
            center_qk=center_qk,
        )
        self.settings = settings
        self.n_heads, self.d_model = int(nhead), int(d_model)
        self.head_dim = self.d_model // self.n_heads
        self.layer_index = 0
        self.hier_context = None
        self.attention_dropout = float(dropout)
        L, E, H = settings.n_levels, self.d_model, self.n_heads
        gen = torch.Generator().manual_seed(0)
        # Pooling: a learned token for a cluster without a visible cell, the attention-pool queries (zero: the mean) and
        # the cluster features (zero: absent at the start).
        self.pool_fill = nn.Parameter(torch.randn(max(L, 1), E, generator=gen))
        if settings.pooling == "attn_pool":
            self.pool_query = nn.Parameter(torch.zeros(max(L, 1), E))
        if "count" in settings.pool_features:
            self.pool_count = nn.Parameter(torch.zeros(max(L, 1), E))
        if "spread" in settings.pool_features:
            self.pool_spread = nn.Parameter(torch.zeros(max(L, 1), E))
        # Projections of the coarse levels: copies of the cells' (no draw), learned apart unless shared.
        if not settings.share_level_params and L > 0:
            self.level_in_weight = nn.Parameter(
                self.self_attn.in_proj_weight.detach().clone().unsqueeze(0).repeat(L, 1, 1)
            )
            self.level_in_bias = nn.Parameter(self.self_attn.in_proj_bias.detach().clone().unsqueeze(0).repeat(L, 1))
        lam_shape = (max(L, 1), H if settings.lambda_learnable == "per_head" else 1)
        mu_shape = (max(L, 1), H if settings.mu_learnable == "per_head" else 1)
        if settings.prior_mode in ("soft", "factorized_percell"):
            weight = _weight(lam_shape, settings.lambda_init, settings.lambda_learnable)
            if isinstance(weight, nn.Parameter):
                self.prior_lambda = weight
            else:
                self.register_buffer("prior_lambda", weight)
        if settings.direction_prior:
            weight = _weight(mu_shape, settings.mu_init, settings.mu_learnable)
            if isinstance(weight, nn.Parameter):
                self.prior_mu = weight
            else:
                self.register_buffer("prior_mu", weight)
        if settings.readout == "concat" and L > 0:
            self.readout_weight = nn.Parameter(torch.zeros(L, E, E))
        if settings.readout == "transformer" and L > 0:
            self.readout_query = nn.Parameter(torch.zeros(E, E))
            self.readout_key = nn.Parameter(torch.randn(E, E, generator=gen) / math.sqrt(E))
            self.readout_level = nn.Parameter(torch.zeros(L + 1, E))

    # ------------------------------------------------------------------ the block

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        ctx = self.hier_context
        if ctx is None:
            raise RuntimeError(
                "a hierarchical layer needs the pass's HierarchicalContext (TransformerNodeEncoderHook.forward sets it)."
            )
        seq_len, n_graphs, _ = x.shape
        seq_len -= 1  # CLS last
        if self.record_tokens:
            cells = torch.zeros(n_graphs, seq_len + 1, dtype=torch.bool, device=x.device)
            for g, graph in enumerate(ctx.graphs):
                cells[g, graph.start : seq_len] = True
            self.token_cosine = mean_pairwise_cosine(x.detach(), cells)
        out = x.new_zeros(x.shape)
        maps = [] if self.keep_maps else None
        for g, graph in enumerate(ctx.graphs):
            cell_out, cls_out, cell_map = self._attention(x[graph.start : seq_len, g], x[seq_len, g], graph, ctx, g)
            out[graph.start : seq_len, g] = cell_out
            out[seq_len, g] = cls_out
            if maps is not None:
                maps.append((graph.start, cell_map))
        if maps is not None:
            self.save_attn_output_weights(self._layout_maps(maps, seq_len, x))
        else:
            self.save_attn_output_weights(None)
        return self.dropout1(out)

    def _layout_maps(self, maps, seq_len, x):
        """The cell-level maps as the dense transformer's ``[B * H, S + 1, S + 1]`` (None if a graph has none: routed)."""
        if any(m is None for _, m in maps):
            return None
        H = self.n_heads
        full = x.new_zeros(len(maps), H, seq_len + 1, seq_len + 1)
        for g, (start, (cell_rows, cls_row)) in enumerate(maps):
            n = seq_len - start
            full[g, :, start:seq_len, start:seq_len] = cell_rows[:, :, :n]
            full[g, :, start:seq_len, seq_len] = cell_rows[:, :, n]
            full[g, :, seq_len, start:seq_len] = cls_row[:, :n]
            full[g, :, seq_len, seq_len] = cls_row[:, n]
        return full.reshape(len(maps) * H, seq_len + 1, seq_len + 1)

    # ------------------------------------------------------------------ pieces

    def _in_proj(self, level: int, L: int):
        if level < L and hasattr(self, "level_in_weight"):
            return self.level_in_weight[level], self.level_in_bias[level]
        return self.self_attn.in_proj_weight, self.self_attn.in_proj_bias

    def _project(self, level: int, L: int, tokens: torch.Tensor, cls: torch.Tensor, keys_only: bool = False):
        """``(q, k, v)``, each ``[n + 1, H, d]`` (CLS last): the level's tokens through its projections, ``center_qk`` applied."""
        E, H, d = self.d_model, self.n_heads, self.head_dim
        w, b = self._in_proj(level, L)
        x = torch.cat([tokens, cls[None]], 0)
        qk_in = x
        if self.center_qk:
            qk_in = F.layer_norm(x - tokens.mean(0, keepdim=True), (E,))
        k = F.linear(qk_in, w[E : 2 * E], b[E : 2 * E]).view(-1, H, d)
        if keys_only:
            return None, k, None
        q = F.linear(qk_in, w[:E], b[:E]).view(-1, H, d)
        v = F.linear(x, w[2 * E :], b[2 * E :]).view(-1, H, d)
        return q, k, v

    def _pool(self, level: int, child_tokens, child_weight, parent, n_parents, hier):
        """Level ``level``'s tokens ``[n_parents, E]`` and visible-cell counts from its children's (``parent`` ``[m]``)."""
        E = self.d_model
        weight = torch.zeros(n_parents, dtype=child_tokens.dtype, device=child_tokens.device).index_add(
            0, parent, child_weight
        )
        if self.settings.pooling == "attn_pool":
            score = child_tokens @ self.pool_query[level].to(child_tokens.dtype) / math.sqrt(E)
            score = torch.where(
                child_weight > 0, score + child_weight.clamp_min(1e-30).log(), torch.full_like(score, float("-inf"))
            )
            top = torch.full((n_parents,), float("-inf"), dtype=score.dtype, device=score.device)
            top = top.scatter_reduce(0, parent, score, "amax", include_self=True)
            top = torch.where(torch.isfinite(top), top, torch.zeros_like(top)).detach()
            e = (score - top[parent]).exp()
            total = torch.zeros(n_parents, dtype=e.dtype, device=e.device).index_add(0, parent, e)
            alpha = e / total[parent].clamp_min(1e-30)
            token = torch.zeros(n_parents, E, dtype=child_tokens.dtype, device=child_tokens.device)
            token = token.index_add(0, parent, child_tokens * alpha[:, None])
        else:
            token = torch.zeros(n_parents, E, dtype=child_tokens.dtype, device=child_tokens.device)
            token = token.index_add(0, parent, child_tokens * child_weight[:, None]) / weight.clamp_min(1e-30)[:, None]
        token = torch.where(weight[:, None] > 0, token, self.pool_fill[level].to(token.dtype).expand_as(token))
        if hasattr(self, "pool_count"):
            token = token + self.pool_count[level].to(token.dtype) * torch.log1p(weight)[:, None]
        if hasattr(self, "pool_spread"):
            token = (
                token
                + self.pool_spread[level].to(token.dtype) * torch.log1p(hier.spread[level].to(token.dtype))[:, None]
            )
        return token, weight

    def _level_bias(self, ctx, level: int, L: int):
        if level == L:
            return ctx.bias
        if ctx.coarse_biases is None:
            return None
        return ctx.coarse_biases[level]

    @staticmethod
    def _arrow_scale(hier, level: int) -> float:
        """The length below which a parent's coarse arrow at ``level - 1`` carries no direction: that level's median cluster
        spread (µm), at least 1 µm."""
        return max(float(hier.spread[level - 1].float().median()), 1.0)

    def _lam(self, level: int) -> torch.Tensor:
        return self.prior_lambda[level - 1]

    def _mu(self, level: int) -> torch.Tensor:
        return self.prior_mu[level - 1]

    def _dropout(self, weights):
        return (
            F.dropout(weights, p=self.attention_dropout, training=self.training)
            if self.attention_dropout > 0
            else weights
        )

    # ------------------------------------------------------------------ attention

    def _attention(self, cells, cls, graph: GraphLayout, ctx: HierarchicalContext, g: int):
        """One graph: ``(cell outputs [n, E], CLS output [E], the cell-level map or None)``."""
        hier = graph.hierarchies[self.layer_index % len(graph.hierarchies)]
        L, E, H = hier.n_levels, self.d_model, self.n_heads
        tokens, weights = [None] * (L + 1), [None] * (L + 1)
        tokens[L], weights[L] = cells, graph.visible.to(cells.dtype)
        for level in range(L - 1, -1, -1):
            tokens[level], weights[level] = self._pool(
                level, tokens[level + 1], weights[level + 1], hier.up[level + 1], hier.counts[level], hier
            )
        projected = [self._project(level, L, tokens[level], cls) for level in range(L + 1)]

        outputs, attention = [None] * (L + 1), [None] * (L + 1)
        outputs[0], attention[0] = self._top_level(projected[0], hier, ctx, L)
        for level in range(1, L + 1):
            prior = attention[level - 1]
            if self.settings.prior_refresh == "once":
                key = (g, self.layer_index % len(graph.hierarchies), level - 1)
                if self.layer_index < len(graph.hierarchies):
                    ctx.cache[key] = prior
                prior = ctx.cache.get(key, prior)
            if self.settings.candidates == "dense":
                outputs[level], attention[level] = self._dense_level(level, projected, tokens, prior, hier, ctx, L, cls)
            else:
                outputs[level], attention[level] = self._routed_level(
                    level, projected, tokens, prior, hier, ctx, L, cls
                )

        if ctx.record is not None:
            ctx.record.append(self._export(g, hier, graph, attention, L))
        cell_out = self._readout(outputs, hier, L)
        cls_out = self.self_attn.out_proj(outputs[0][-1].reshape(E))
        cell_map = None
        if self.keep_maps and self.settings.candidates == "dense":
            cls_row = attention[0][:, -1]  # [H, n_0 + 1]
            if L > 0:
                # CLS reads clusters; each cluster's weight spread over its cells evenly, for the cell-level layout.
                top = hier.assign[0]
                share = cls_row[:, top] / hier.sizes[0][top].to(cls_row.dtype)
                cls_row = torch.cat([share, cls_row[:, -1:]], 1)
            cell_map = (attention[L][:, : hier.counts[L]], cls_row)
        return cell_out, cls_out, cell_map

    def _top_level(self, qkv, hier, ctx, L):
        """Level 0: dense attention among the level's tokens and CLS: ``(output [n_0 + 1, H, d], A^0 [H, n_0 + 1, n_0 + 1])``."""
        q, k, v = qkv
        n = q.shape[0] - 1
        scores = torch.einsum("qhd,khd->hqk", q * self.head_dim**-0.5, k)
        bias = self._level_bias(ctx, 0, L)
        if bias is not None:
            pos = hier.centroid[0].to(q.dtype)
            found = offset_bias(bias, pos[None, :, :] - pos[:, None, :])  # [n, n, H]
            scores = scores + F.pad(found.permute(2, 0, 1), (0, 1, 0, 1))
        if L == 0:
            self_pairs = torch.eye(n + 1, dtype=torch.bool, device=q.device)
            self_pairs[n, n] = False  # CLS reads itself, as in the dense transformer
            scores = scores.masked_fill(self_pairs, float("-inf"))
        weights = scores.softmax(-1)
        out = torch.einsum("hqk,khd->qhd", self._dropout(weights), v)
        return out, weights

    def _dense_level(self, level, projected, tokens, prior, hier, ctx, L, cls):
        """Level ``level`` >= 1 with every token of the level as a key: ``(output [n + 1, H, d], A [H, n, n + 1])``."""
        q, k, v = projected[level]
        n, n_par = hier.counts[level], hier.counts[level - 1]
        mode = self.settings.prior_mode
        q = q[:n]
        scores = torch.einsum("qhd,khd->hqk", q * self.head_dim**-0.5, k)  # [H, n, n + 1]
        pos = hier.centroid[level].to(q.dtype)
        offsets = pos[None, :, :] - pos[:, None, :]  # [n, n, 2]: key - query
        bias = self._level_bias(ctx, level, L)
        if bias is not None:
            scores = scores + F.pad(offset_bias(bias, offsets).permute(2, 0, 1), (0, 1))
        if level == L:
            scores = scores.masked_fill(torch.eye(n, n + 1, dtype=torch.bool, device=q.device), float("-inf"))
        pq = hier.up[level]
        pk = torch.cat([pq, pq.new_tensor([n_par])])  # CLS is its own one-member cluster, index n_par
        parent_rows = prior[:, pq]  # [H, n, n_par + 1]
        if self.settings.direction_prior:
            target, conf = coarse_direction(
                torch.arange(n_par + 1, device=q.device).expand(n, -1),
                parent_rows,
                hier.centroid[level - 1].to(q.dtype),
                n_par,
            )
            arrow = target - hier.centroid[level - 1].to(q.dtype)[pq][None]  # [H, n, 2]
            cos = direction_cosine(offsets[None], arrow[:, :, None, :], self._arrow_scale(hier, level))  # [H, n, n]
            term = self._mu(level)[:, None, None] * conf[:, :, None] * cos
            scores = scores + F.pad(term, (0, 1))
        a = parent_rows[:, :, pk]  # [H, n, n + 1]: A^{l-1}_{p(i), p(j)}
        if mode == "none":
            logits = scores
        elif mode == "soft":
            sizes = torch.cat([hier.n_children[level - 1], hier.n_children[level - 1].new_ones(1)])[pk].to(a.dtype)
            prior_p = a / sizes if self.settings.size_correction else a
            logits = scores + self._lam(level)[:, None, None] * (prior_p + PRIOR_EPS).log()
        else:
            within = within_cluster(scores, segment_logsumexp(scores, pk, n_par + 1).gather(-1, pk.expand_as(scores)))
            if mode == "factorized":
                logits = within + a.clamp_min(LOG_FLOOR).log()
            else:
                _, kbar, _ = self._project(level, L, tokens[level - 1], cls, keys_only=True)  # [n_par + 1, H, d]
                cluster_scores = torch.einsum("qhd,khd->hqk", q * self.head_dim**-0.5, kbar)  # [H, n, n_par + 1]
                if bias is not None:
                    to_centroid = hier.centroid[level - 1].to(q.dtype)[None, :, :] - pos[:, None, :]
                    cluster_scores = cluster_scores + F.pad(offset_bias(bias, to_centroid).permute(2, 0, 1), (0, 1))
                cluster_scores = cluster_scores + self._lam(level)[:, None, None] * (parent_rows + PRIOR_EPS).log()
                log_w = cluster_scores.log_softmax(-1)
                logits = within + log_w.gather(-1, pk.expand(log_w.shape[0], n, -1))
        weights = logits.softmax(-1)
        out = torch.einsum("hqk,khd->qhd", self._dropout(weights), v)
        return out, weights

    # ------------------------------------------------------------------ routing

    def _candidates(self, level, prior, hier, ctx):
        """``(parents [P, C], a [H, P, C], a_cls [H, P])``: each parent's candidate parents and its weight on them."""
        n_par = hier.counts[level - 1]
        device = hier.centroid[level].device
        ids = torch.arange(n_par, device=device)
        keys, weights = prior_rows(prior, ids, n_par)
        is_cluster = (keys >= 0) & (keys < n_par)
        score = torch.where(is_cluster, weights.amax(0), torch.full_like(keys, float("-inf"), dtype=weights.dtype))
        k_top = min(self.settings.topk_at(level), keys.shape[1])
        top_score, top_at = score.topk(k_top, dim=-1)
        top = torch.where(torch.isfinite(top_score), keys.gather(-1, top_at), torch.full_like(top_at, -1))
        parts = [hier.neighbours[level - 1], top]
        if self.training and self.settings.explore_n > 0:
            parts.append(self._explore(weights, is_cluster, keys, top, n_par, ctx))
        merged = torch.cat(parts, 1)
        earlier = (merged[:, :, None] == merged[:, None, :]) & torch.ones(
            merged.shape[1], merged.shape[1], dtype=torch.bool, device=device
        ).tril(-1)
        drop = earlier.any(-1) | (merged < 0)
        order = torch.sort(drop.to(torch.int8), dim=-1, stable=True).indices
        merged = torch.where(drop.gather(-1, order), torch.full_like(merged, -1), merged.gather(-1, order))
        width = hier.children[level - 1].shape[1]
        slots = max(1, min(merged.shape[1], self.settings.cap_per_query // max(width, 1)))
        chosen = int((~drop).sum(-1).max())
        if slots < chosen:
            # The cap is shared out by the widest parent of the level: one parent with many children (a large niche
            # patch on top, say) leaves every query of the slide only a few of the parents it chose.
            warnings.warn(
                f"routed level {level}: a parent with {width} children leaves {slots} of up to {chosen} chosen parents per "
                f"query under cap_per_query {self.settings.cap_per_query}. Balance the level above (top_level.split_by_space "
                "kdtree, smaller clusters) or raise cap_per_query.",
                RuntimeWarning,
                stacklevel=2,
            )
        parents = merged[:, :slots]
        a = lookup(keys, weights, parents)
        a_cls = lookup(keys, weights, torch.full((n_par, 1), n_par, device=device, dtype=torch.long))[..., 0]
        return parents, a, a_cls

    def _explore(self, weights, is_cluster, keys, top, n_par, ctx):
        """``[P, explore_n]`` exploration parents (training only): uniform over the level, or drawn in proportion to the
        parent's head-mean weight outside its top-k, from the pass's private generator.
        """
        n_draw = self.settings.explore_n
        gen = ctx.generator
        device = keys.device
        if self.settings.explore_strategy == "uniform":
            return torch.randint(0, n_par, (keys.shape[0], n_draw), generator=gen).to(device)
        mean = weights.detach().mean(0)
        in_top = (keys[:, :, None] == top[:, None, :]).any(-1)
        mean = torch.where(is_cluster & ~in_top, mean, torch.zeros_like(mean)) + 1e-12 * is_cluster
        picks = torch.multinomial(mean.float().cpu(), n_draw, replacement=True, generator=gen).to(device)
        return keys.gather(-1, picks)

    def _routed_level(self, level, projected, tokens, prior, hier, ctx, L, cls):
        """Level ``level`` >= 1, routed: ``(output [n + 1, H, d], SparseRows)``, a block of parents at a time."""
        q_all, k_all, v_all = projected[level]
        n, n_par = hier.counts[level], hier.counts[level - 1]
        H, d = self.n_heads, self.head_dim
        children = hier.children[level - 1]  # [P, m]
        m = children.shape[1]
        parents, a, a_cls = self._candidates(level, prior, hier, ctx)  # [P, C], [H, P, C], [H, P]
        n_slots = parents.shape[1]
        key_ids = children[parents.clamp_min(0)]  # [P, C, m]
        key_ids = torch.where((parents >= 0)[:, :, None], key_ids, torch.full_like(key_ids, -1)).reshape(n_par, -1)
        pos = hier.centroid[level].to(q_all.dtype)
        bias = self._level_bias(ctx, level, L)
        mode = self.settings.prior_mode
        direction = None
        if self.settings.direction_prior:
            keys, weights = prior_rows(prior, torch.arange(n_par, device=pos.device), n_par)
            target, conf = coarse_direction(keys, weights, hier.centroid[level - 1].to(q_all.dtype), n_par)
            direction = (target - hier.centroid[level - 1].to(q_all.dtype)[None], conf)  # [H, P, 2], [H, P]
            arrow_scale = self._arrow_scale(hier, level)
        kbar = None
        if mode == "factorized_percell":
            _, kbar, _ = self._project(level, L, tokens[level - 1], cls, keys_only=True)
        sizes = hier.n_children[level - 1]
        rows_per_block = max(1, self.settings.chunk_pairs // max(1, H * m * (key_ids.shape[1] + 1)))
        last = level == L
        keep_weights = (not last) or ctx.record is not None or self.keep_maps

        def block(start, stop, q_all, k_all, v_all, a, a_cls, kbar):
            groups = slice(start, stop)
            qi, kid = children[groups], key_ids[groups]  # [G, m], [G, K]
            q = q_all[qi.clamp_min(0)] * d**-0.5  # [G, m, H, d]
            k = torch.cat([k_all[kid.clamp_min(0)], k_all[-1].expand(len(qi), 1, H, d)], 1)  # [G, K + 1, H, d]
            v = torch.cat([v_all[kid.clamp_min(0)], v_all[-1].expand(len(qi), 1, H, d)], 1)
            scores = torch.einsum("gqhd,gkhd->ghqk", q, k)  # [G, H, m, K + 1]
            offsets = pos[kid.clamp_min(0)][:, None, :, :] - pos[qi.clamp_min(0)][:, :, None, :]  # [G, m, K, 2]
            if bias is not None:
                scores = scores + F.pad(offset_bias(bias, offsets).permute(0, 3, 1, 2), (0, 1))
            blocked = (kid < 0)[:, None, :].expand(-1, m, -1)
            if last:
                blocked = blocked | (kid[:, None, :] == qi[:, :, None])
            blocked = F.pad(blocked, (0, 1))[:, None]  # CLS open
            scores = scores.masked_fill(blocked, float("-inf"))
            if direction is not None:
                arrow, conf = direction[0][:, groups], direction[1][:, groups]  # [H, G, 2], [H, G]
                cos = direction_cosine(offsets[None], arrow[:, :, None, None, :], arrow_scale)  # [H, G, m, K]
                term = (self._mu(level)[:, None, None, None] * conf[:, :, None, None] * cos).permute(1, 0, 2, 3)
                scores = scores + F.pad(term, (0, 1))
            slot = torch.arange(kid.shape[1], device=kid.device) // m
            a_keys = torch.cat([a[:, groups][:, :, slot], a_cls[:, groups, None]], -1).permute(1, 0, 2)[
                :, :, None
            ]  # [G, H, 1, K + 1]
            if mode == "none":
                logits = scores
            elif mode == "soft":
                if self.settings.size_correction:
                    count = sizes[parents[groups].clamp_min(0)][:, slot].to(scores.dtype)  # [G, K]
                    count = F.pad(count, (0, 1), value=1.0)[:, None, None]
                    prior_p = a_keys / count
                else:
                    prior_p = a_keys
                logits = scores + self._lam(level)[None, :, None, None] * (prior_p + PRIOR_EPS).log()
            else:
                G = len(qi)
                body = scores[..., :-1].reshape(G, H, m, n_slots, m)
                lse = safe_logsumexp(body, -1)[..., None].expand_as(body).reshape(G, H, m, -1)
                within = torch.cat([within_cluster(scores[..., :-1], lse), torch.zeros_like(scores[..., -1:])], -1)
                if mode == "factorized":
                    logits = within + a_keys.clamp_min(LOG_FLOOR).log()
                else:
                    cand = parents[groups]  # [G, C]
                    kb = torch.cat([kbar[cand.clamp_min(0)], kbar[-1].expand(G, 1, H, d)], 1)  # [G, C + 1, H, d]
                    cs = torch.einsum("gqhd,gchd->ghqc", q, kb)
                    if bias is not None:
                        to_c = (
                            hier.centroid[level - 1].to(q.dtype)[cand.clamp_min(0)][:, None]
                            - pos[qi.clamp_min(0)][:, :, None]
                        )
                        cs = cs + F.pad(offset_bias(bias, to_c).permute(0, 3, 1, 2), (0, 1))
                    a_c = torch.cat([a[:, groups], a_cls[:, groups, None]], -1).permute(1, 0, 2)[:, :, None]
                    cs = cs + self._lam(level)[None, :, None, None] * (a_c + PRIOR_EPS).log()
                    cs = cs.masked_fill(F.pad(cand < 0, (0, 1))[:, None, None], float("-inf"))
                    log_w = cs.log_softmax(-1)
                    logits = within + torch.cat([log_w[..., slot], log_w[..., -1:]], -1)
            weights = logits.softmax(-1)
            out = torch.einsum("ghqk,gkhd->gqhd", self._dropout(weights), v)
            return out, weights

        outs, rows = [], []
        n_blocks = math.ceil(n_par / rows_per_block)
        for b in range(n_blocks):
            start, stop = b * rows_per_block, min(n_par, (b + 1) * rows_per_block)
            if last and not keep_weights and self.training and n_blocks > 1:
                out = torch.utils.checkpoint.checkpoint(
                    lambda *t, s=start, e=stop: block(s, e, *t)[0],
                    q_all,
                    k_all,
                    v_all,
                    a,
                    a_cls,
                    kbar,
                    use_reentrant=False,
                )
                w = None
            else:
                out, w = block(start, stop, q_all, k_all, v_all, a, a_cls, kbar)
            outs.append(out)
            rows.append(w)
        out = torch.cat(outs, 0)  # [P, m, H, d]
        flat = children.reshape(-1)
        valid = flat >= 0
        cell_out = out.reshape(-1, H, d)[valid]
        result = q_all.new_zeros(n + 1, H, d).index_copy(0, flat[valid], cell_out)
        sparse = None
        if keep_weights:
            q_group = torch.zeros(n, dtype=torch.long, device=flat.device)
            q_slot = torch.zeros(n, dtype=torch.long, device=flat.device)
            group_of = torch.arange(n_par, device=flat.device).repeat_interleave(m)
            slot_of = torch.arange(m, device=flat.device).repeat(n_par)
            q_group[flat[valid]] = group_of[valid]
            q_slot[flat[valid]] = slot_of[valid]
            cls_id = torch.full((n_par, 1), n, dtype=torch.long, device=flat.device)
            sparse = SparseRows(torch.cat([key_ids, cls_id], 1), torch.cat(rows, 0), q_group, q_slot)
        return result, sparse

    # ------------------------------------------------------------------ readout

    def _readout(self, outputs, hier, L):
        """Each cell's output fused with its ancestors' (``readout``), through the output projection: ``[n, E]``."""
        E = self.d_model
        n = hier.counts[L]
        cell = outputs[L][:n].reshape(n, E)
        if self.settings.readout == "none" or L == 0:
            return self.self_attn.out_proj(cell)
        ancestors = [outputs[level][: hier.counts[level]].reshape(-1, E)[hier.assign[level]] for level in range(L)]
        if self.settings.readout == "concat":
            mixed = cell
            for level, value in enumerate(ancestors):
                mixed = mixed + value @ self.readout_weight[level].to(value.dtype).T
            return self.self_attn.out_proj(mixed)
        stack = torch.stack(
            [cell, *ancestors[::-1]], 1
        )  # [n, L + 1, E]: the cell, then its ancestors from the nearest up
        query = cell @ self.readout_query.to(cell.dtype).T
        keys = stack @ self.readout_key.to(cell.dtype).T + self.readout_level.to(cell.dtype)
        alpha = torch.einsum("ne,nle->nl", query, keys).div(math.sqrt(E)).softmax(-1)
        return self.self_attn.out_proj((alpha[..., None] * stack).sum(1))

    # ------------------------------------------------------------------ export

    def _export(self, g, hier, graph, attention, L) -> dict:
        """What a recording pass keeps of this layer and graph (detached, on the CPU)."""
        levels = []
        for level, value in enumerate(attention):
            if isinstance(value, SparseRows):
                levels.append(value.detach_cpu())
            elif value is not None:
                levels.append(value.detach().cpu())
            else:
                levels.append(None)
        out = {
            "layer": self.layer_index,
            "graph": g,
            "hierarchy": hier,
            "pos_um": graph.pos_um.detach().cpu(),
            "visible": graph.visible.detach().cpu(),
            "nodes": graph.nodes,
            "attention": levels,
            "topk": [self.settings.topk_at(level) for level in range(1, hier.n_levels + 1)],
            "candidates": self.settings.candidates,
        }
        if hasattr(self, "prior_lambda"):
            out["lambda"] = self.prior_lambda.detach().cpu().clone()
        if hasattr(self, "prior_mu"):
            out["mu"] = self.prior_mu.detach().cpu().clone()
        return out
