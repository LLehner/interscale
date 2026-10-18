"""Attention backends of the global (transformer) component.

Selected by ``model.global_component.parameters.attention``; the staged design is in
``.claude/attention_plan.md``. A backend is the module each transformer layer uses as its
``self_attn``. The default, ``dense`` at the default head width, is torch's multi-head attention as
vendored in ``transformer_utils`` -- the model from before backends existed. Every other backend
builds on :class:`ProjectedAttention`: project to ``heads x head_dim`` queries, keys and values,
rotate queries and keys (RoPE), attend, project back to ``n_embed``.

What every backend keeps:

* **Off is today's model.** With the defaults nothing changes -- no parameter, no RNG draw, no
  checkpoint name.
* **On changes no other initial weight.** A non-default backend is built inside a forked RNG and
  the default attention's draws are then replayed, so everything built after the layer starts from
  the same weights as in a dense model at the same seed. At the default head width the projections
  themselves are drawn like torch's, so dense and another backend share their initial ``q``, ``k``,
  ``v`` and output projections too.
* **An explicit path.** Asked for weights (``need_weights``), a backend computes the same function
  through a path that materialises each head's attention map with autograd -- what relevance and
  the evaluation pipeline read. Training may take a cheaper path.
* **Declared capabilities.** :class:`AttentionSpec` says which mask form, which attention-bias forms
  and whether RoPE a backend can carry; the config validator rejects any other combination at load
  time.

Adding a backend is one :func:`register_attention` entry plus, if it has settings, one config
sub-block in ``get_global_component_cfg``.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch_geometric.utils import softmax as scatter_softmax

from interscale.module.global_modules.transformer_utils import apply_rotary
from interscale.tl.masking import PATTERN_EDGE_TYPES, attention_pattern

#: Forms in which an attention bias can be handed to a backend: ``dense``, the ``[B * heads, S, S]``
#: float mask it is merged into today; ``factors``, low-rank ``U, W`` with ``bias = U W^T``, which a
#: backend folds into its queries and keys; ``pairs``, the bias of each pair of a list, for a backend
#: that attends along one.
BIAS_FORMS = ("dense", "factors", "pairs")


class StructuredMask(NamedTuple):
    """The attention mask for a backend that never forms the ``S x S`` matrix.

    It stands where the dense mask stands -- ``GlobalInput.attention_mask``, handed on to ``forward``
    unchanged by every caller -- and ``forward`` passes it to the layers as their per-batch context.

    ``blocked`` is ``[3, P]``: the ``(graph, query, key)`` positions of every blocked pair, exactly
    the True entries of the dense boolean mask (:func:`~interscale.tl.masking.blocked_pairs`); the
    CLS slot never appears. ``bias_factors``, when attention biases are on, are ``(U, W)``, each
    ``[B * heads, S + 1, r]`` and graph-major like the dense mask, with ``bias = U W^T`` -- zero
    rows at CLS and padding, so their bias is 0 there.

    For a backend that attends along a sparse pattern (Exphormer): ``pairs`` ``[3, E]`` are the
    ``(graph, query, key)`` positions it attends along (blocked pairs already removed), ``pair_types``
    ``[E]`` their edge types (``tl.masking.PATTERN_EDGE_TYPES``), and ``pair_bias`` ``[E, heads]``
    the attention biases on them (``AttentionBias.pairs``), or None.
    """

    blocked: torch.Tensor
    bias_factors: tuple[torch.Tensor, torch.Tensor] | None = None
    pairs: torch.Tensor | None = None
    pair_types: torch.Tensor | None = None
    pair_bias: torch.Tensor | None = None


@dataclass(frozen=True)
class AttentionSpec:
    """What the rest of the code needs to know about one attention backend.

    ``build(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim)`` returns
    the module a layer uses as ``self_attn``, or None for torch's vendored multi-head attention.
    ``validate(attention_cfg)`` raises on settings that would fail mid-run. ``mask`` is the form in
    which it takes the blocked pairs: ``"dense"`` (a boolean ``[B * heads, S+1, S+1]`` mask),
    ``"pairs"`` (a list of ``(graph, query, key)`` positions), or ``"pattern"`` (that list, and the
    sparse pattern it attends along, built by ``pattern``). ``bias_forms`` are the attention-bias
    forms it can carry (:data:`BIAS_FORMS`); ``rotary`` whether it applies RoPE;
    ``training_maps`` whether its training forward passes form the maps a loss could read.
    ``note``, when set, says in a sentence what it cannot carry and why; the validator appends it to
    its rejections. ``pattern``, for ``mask="pattern"``, builds the pairs the backend attends along:
    ``pattern(attention_cfg, *, edge_index, batch, index_nodes, blocked, seed, generator, training,
    device) -> (pairs [3, E], pair_types [E])``.
    """

    build: Callable
    validate: Callable
    mask: str
    bias_forms: frozenset = field(default_factory=frozenset)
    rotary: bool = True
    training_maps: bool = False
    note: str = ""
    pattern: Callable | None = None


ATTENTION_REGISTRY: dict[str, AttentionSpec] = {}


def register_attention(name: str, **spec):
    """Register ``build(...) -> nn.Module | None`` as the attention backend ``name``."""

    def decorator(build):
        if name in ATTENTION_REGISTRY:
            raise ValueError(f"attention backend '{name}' is already registered")
        ATTENTION_REGISTRY[name] = AttentionSpec(build=build, **spec)
        return build

    return decorator


def attention_head_dim(attention_cfg, n_embed: int, n_heads: int) -> int:
    """The width of one attention head: ``attention.head_dim``, or ``n_embed / n_heads`` when it is 0."""
    head_dim = int(attention_cfg.get("head_dim", 0)) if attention_cfg is not None else 0
    if head_dim > 0:
        return head_dim
    if n_embed % n_heads:
        raise ValueError(
            f"n_embed ({n_embed}) is not divisible by n_heads ({n_heads}); set "
            "model.global_component.parameters.attention.head_dim to choose the head width directly."
        )
    return n_embed // n_heads


def attention_kind(attention_cfg) -> str:
    return str(attention_cfg.get("kind", "dense")) if attention_cfg is not None else "dense"


class ProjectedAttention(nn.Module):
    """Multi-head self-attention with its own head width, in torch's ``MultiheadAttention`` interface.

    Queries, keys and values are projected from ``embed_dim`` to ``num_heads * head_dim``
    (``in_proj_weight``, ``[3 * heads * head_dim, embed_dim]``), the output back to ``embed_dim``
    (``out_proj``). The parameter names, shapes at the default head width, and initialisation are
    torch's, so a dense checkpoint loads into any backend and at one seed the projections start
    equal. Subclasses implement :meth:`_attend`.

    Inputs are sequence-first (``batch_first`` is False, which also keeps torch's encoder fast path
    -- that would bypass this module -- switched off).
    """

    batch_first = False
    _qkv_same_embed_dim = True
    #: Every backend but torch's vendored attention takes a per-batch ``context`` and computes its
    #: maps only when asked; the layer reads this to tell them apart.
    is_backend = True
    #: Whether RoPE's rotation of queries and keys keeps its meaning in this backend's similarity.
    supports_rotary = True

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim, self.num_heads, self.head_dim = int(embed_dim), int(num_heads), int(head_dim)
        self.dropout = float(dropout)
        inner = self.num_heads * self.head_dim
        # Created in torch's order -- the output projection draws its (later overwritten) init first,
        # then xavier on the input projection -- so the draws match MultiheadAttention's.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * inner, self.embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * inner)) if bias else None
        self.out_proj = NonDynamicallyQuantizableLinear(inner, self.embed_dim, bias=bias)
        nn.init.xavier_uniform_(self.in_proj_weight)
        if bias:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
        context=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``[L, B, E]`` tokens to ``([L, B, E], weights or None)``; weights are ``[B * heads, L, L]``.

        ``key_padding_mask`` ``[B, L]`` is boolean (True at padding) or float (``-inf`` there), as
        the encoder layer hands it on. ``attn_mask`` is the dense mask, for backends that take one.
        ``context`` carries what a backend needs beyond that (:class:`StructuredMask`). The weights
        are per head, after dropout, laid out like torch's (``average_attn_weights`` is ignored).
        """
        if is_causal:
            raise ValueError("attention backends attend bidirectionally; is_causal is not supported.")
        if query is not key or value.shape != query.shape:
            raise ValueError(
                "attention backends are self-attention only: query and key must be one tensor, and the values "
                "the same tokens (or, under attention.center_qk, the tokens queries and keys were centred from)."
            )
        seq_len, n_graphs, _ = query.shape
        if value is key:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        else:
            # Queries and keys from one input, values from another (`attention.center_qk`).
            inner = self.num_heads * self.head_dim
            w_qk, w_v = self.in_proj_weight.split([2 * inner, inner])
            b_qk, b_v = (None, None) if self.in_proj_bias is None else self.in_proj_bias.split([2 * inner, inner])
            q, k = F.linear(query, w_qk, b_qk).chunk(2, dim=-1)
            v = F.linear(value, w_v, b_v)
        q, k, v = (t.reshape(seq_len, n_graphs * self.num_heads, self.head_dim).transpose(0, 1) for t in (q, k, v))
        if rotary is not None:
            if not self.supports_rotary:
                raise ValueError(f"{type(self).__name__} does not apply RoPE (see its docstring).")
            cos, sin = rotary
            if cos.shape != q.shape:
                raise RuntimeError(
                    f"rotary angles of shape {tuple(cos.shape)} do not match the (batch*heads, tokens, head_dim) "
                    f"queries {tuple(q.shape)}."
                )
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        padding = None
        if key_padding_mask is not None:
            padding = key_padding_mask if key_padding_mask.dtype == torch.bool else torch.isneginf(key_padding_mask)

        out, weights = self._attend(q, k, v, padding, attn_mask, need_weights, context)  # [B*H, L, hd]
        out = out.transpose(0, 1).reshape(seq_len * n_graphs, self.num_heads * self.head_dim)
        out = self.out_proj(out).view(seq_len, n_graphs, self.embed_dim)
        return out, weights

    def _attend(self, q, k, v, padding, attn_mask, need_weights, context):
        """``[B * heads, L, head_dim]`` queries, keys, values to ``(output, weights or None)``.

        ``padding`` is ``[B, L]``, True at padded keys, or None.
        """
        raise NotImplementedError


class DenseAttention(ProjectedAttention):
    """Softmax attention over every pair, the explicit way: the same operations as the vendored path.

    Used for dense attention at a head width other than ``n_embed / n_heads``; at the default width
    it reproduces torch's attention given the same weights.
    """

    def _attend(self, q, k, v, padding, attn_mask, need_weights, context):
        n_rows, seq_len, head_dim = q.shape
        mask = None
        if attn_mask is not None:
            # The encoder layer hands masks on as float (-inf where blocked); a boolean one is
            # converted the same way, by indexing.
            if attn_mask.is_floating_point():
                mask = attn_mask.to(q.dtype)
            else:
                mask = q.new_zeros(attn_mask.shape).masked_fill(attn_mask, float("-inf"))
        if padding is not None:
            n_graphs = padding.shape[0]
            pad = q.new_zeros(padding.shape).masked_fill(padding, float("-inf"))
            pad = pad.view(n_graphs, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1).reshape(n_rows, 1, seq_len)
            mask = pad if mask is None else mask + pad

        q_scaled = q / math.sqrt(head_dim)
        if mask is not None:
            weights = torch.baddbmm(mask, q_scaled, k.transpose(-2, -1))
        else:
            weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        weights = self._normalise(weights)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        return torch.bmm(weights, v), (weights if need_weights else None)

    def _normalise(self, scores: torch.Tensor) -> torch.Tensor:
        """``[B * heads, L, L]`` logits (``-inf`` where blocked) to attention weights over the last axis."""
        return torch.softmax(scores, dim=-1)


class EntmaxAttention(DenseAttention):
    """Dense attention normalised by alpha-entmax instead of softmax: sparse maps with exact zeros.

    alpha-entmax (Peters et al. 2019) is softmax at alpha -> 1 and sparsemax at alpha = 2; in
    between, a key whose logit falls more than a threshold below the best gets weight exactly 0, and
    the threshold tightens as alpha grows. Each head has its own alpha, learned
    (``learn_alpha``) as ``1 + sigmoid(a)`` -- in (1, 2), as in Correia et al. 2019 -- or fixed.
    Computed by bisection (the ``entmax`` package), which takes ``-inf`` for blocked keys and gives
    alpha its gradient. A key at weight 0 sends no gradient to its logit.
    """

    #: The lowest alpha the learned form reaches: at alpha -> 1 the bisection's exponent 1 / (alpha - 1) blows up.
    MIN_ALPHA = 1.01

    def __init__(self, *args, alpha: float = 1.5, learn_alpha: bool = True, n_iter: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_iter = int(n_iter)
        if learn_alpha:
            # alpha = 1 + sigmoid(a): no RNG draw, every head starts at the configured alpha.
            start = math.log((alpha - 1.0) / (2.0 - alpha))
            self.alpha_logit = nn.Parameter(torch.full((self.num_heads,), start))
        else:
            self.register_buffer("fixed_alpha", torch.full((self.num_heads,), float(alpha)))

    def alpha(self) -> torch.Tensor:
        """``[heads]``: each head's alpha."""
        if hasattr(self, "alpha_logit"):
            return (1.0 + torch.sigmoid(self.alpha_logit)).clamp(min=self.MIN_ALPHA)
        return self.fixed_alpha

    def extra_repr(self) -> str:
        values = ", ".join(f"{a:.3f}" for a in self.alpha().detach().cpu().tolist())
        return f"{super().extra_repr()}, alpha=[{values}], learned={hasattr(self, 'alpha_logit')}"

    def _normalise(self, scores: torch.Tensor) -> torch.Tensor:
        from entmax import entmax_bisect

        n_rows = scores.shape[0]
        alpha = self.alpha().to(scores.dtype).repeat(n_rows // self.num_heads).view(n_rows, 1, 1)
        return entmax_bisect(scores, alpha=alpha, dim=-1, n_iter=self.n_iter)


class TopKAttention(DenseAttention):
    """Dense attention over each query's ``k`` highest-scoring open keys only, softmax among them.

    A hard attention budget: every other key gets weight exactly 0, so a query mixes at most ``k``
    keys -- exactly ``k`` are kept, ties at the cut broken by ``topk``'s order, which matters: tokens
    of hidden cells are identical and many keys lie at equal distances on a lattice. The cut is taken
    on the full logits -- content, biases and the mask's ``-inf`` -- and gradients flow through the
    kept logits. With ``k`` at or above the number of open keys it is dense softmax attention.
    """

    def __init__(self, *args, k: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, k={self.k}"

    def _normalise(self, scores: torch.Tensor) -> torch.Tensor:
        if self.k < scores.shape[-1]:
            kept = torch.zeros_like(scores, dtype=torch.bool)
            kept.scatter_(-1, scores.topk(self.k, dim=-1).indices, True)
            scores = scores.masked_fill(~kept, float("-inf"))
        return torch.softmax(scores, dim=-1)


def orthogonal_gaussian(num_features: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    """``[num_features, dim]`` Gaussian rows, orthogonal within each block of ``dim`` (Performer's ORF).

    Blocks of ``dim`` exactly orthogonal unit directions, each row then scaled by the norm of an
    independent ``dim``-dimensional Gaussian, so every row is still marginally ``N(0, I)`` -- which
    keeps the positive-feature estimator unbiased -- while the rows of a block cover the directions
    evenly, lowering its variance (Choromanski et al., ICLR 2021).
    """
    blocks = []
    for start in range(0, num_features, dim):
        q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float64))
        # QR alone is not uniformly distributed over the orthogonal group: fixing the signs of R's
        # diagonal makes it so (Mezzadri 2007). Without it the rows' directions are not uniform and
        # the estimate is biased -- measured ~7% too high on exp(x . y) at |x|, |y| ~ 1.
        q = q * torch.sign(torch.diagonal(r))
        blocks.append(q.T[: min(dim, num_features - start)])
    norms = torch.randn(num_features, dim, generator=generator, dtype=torch.float64).norm(dim=1, keepdim=True)
    return (torch.cat(blocks) * norms).float()


def head_pairs(pairs: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(graph, query, key)`` pairs ``[3, P]`` as ``(rows, query, key)`` for every head, pair-major.

    ``rows`` index the ``B * heads`` rows of the split heads, graph-major: entry ``p * heads + h``
    is pair ``p`` in head ``h``.
    """
    graph, query, key = pairs
    heads = torch.arange(num_heads, device=pairs.device)
    rows = (graph[:, None] * num_heads + heads).reshape(-1)
    return rows, query.repeat_interleave(num_heads), key.repeat_interleave(num_heads)


class KernelAttention(ProjectedAttention):
    """Attention whose similarity factorises, ``phi(q) . phi(k)`` with positive features, never forming ``S x S``.

    Attention is ``D^-1 phi(Q) (phi(K)^T V)``, ``D = diag(phi(Q) phi(K)^T 1)``: O(S * F * d) for ``F``
    features per token. Subclasses supply the feature map (:meth:`_feature_maps`); everything else
    is shared here.

    * **The long-range mask, by subtraction.** Every blocked pair's term is removed from both sums,
      ``num_i -= phi(q_i) . phi(k_j) v_j`` and ``den_i`` likewise, over the pairs of a
      :class:`StructuredMask`. That is exact with respect to the kernel: the features are positive,
      so what remains is a sum of positive terms over the open keys, and CLS is always open. The
      sums and the subtraction run in float64, so a query whose mass sits almost entirely on its
      blocked neighbourhood loses no precision to the cancellation.
    * **Padded keys** have their features zeroed.
    * **Maps.** Asked for weights, it takes the explicit path -- ``phi(Q) phi(K)^T`` with the blocked
      entries zeroed, normalised -- which gives the same numbers as the factorised path.
    * **Rank.** ``phi(Q) phi(K)^T`` has rank at most ``F``, so each head's map does too, apart from
      the blocked pairs the mask zeroes (a sparse, local correction).
    * No attention dropout: there is no map to drop entries of in the factorised path.
    """

    def _feature_maps(self, q, k, valid, context) -> tuple[torch.Tensor, torch.Tensor]:
        """``[N, L, head_dim]`` queries and keys to positive features ``([N, L, F], [N, L, F])``.

        ``valid`` ``[N, L]`` marks the keys that are not padding; their features must be zero.
        """
        raise NotImplementedError

    def _attend(self, q, k, v, padding, attn_mask, need_weights, context):
        if attn_mask is not None or not isinstance(context, StructuredMask):
            raise ValueError(
                f"{type(self).__name__} takes its mask as a StructuredMask (blocked pairs), never as a "
                "dense attention mask; common_step_local_to_global builds it."
            )
        n_rows, seq_len, _ = q.shape
        n_graphs = n_rows // self.num_heads
        valid = torch.ones(n_rows, seq_len, dtype=torch.bool, device=q.device)
        if padding is not None:
            valid = (~padding).repeat_interleave(self.num_heads, dim=0)
        phi_q, phi_k = self._feature_maps(q, k, valid, context)
        phi_q, phi_k, v64 = phi_q.double(), phi_k.double(), v.double()

        # Each blocked (graph, query, key) for every head.
        rows, query, key = head_pairs(context.blocked, self.num_heads)
        if rows.numel() and int(rows.max()) >= n_graphs * self.num_heads:
            raise ValueError("blocked pairs name a graph beyond this batch.")

        tiny = torch.finfo(torch.float64).tiny
        if need_weights:
            kernel = phi_q @ phi_k.transpose(1, 2)
            kernel = kernel.index_put((rows, query, key), kernel.new_zeros(rows.numel()))
            # The output is computed from the very tensor returned, so a gradient hook on the map
            # (relevance) sees the gradient that flows through it.
            weights = (kernel / kernel.sum(-1, keepdim=True).clamp_min(tiny)).to(q.dtype)
            return weights @ v, weights

        num = phi_q @ (phi_k.transpose(1, 2) @ v64)  # [N, L, d]
        den = phi_q @ phi_k.sum(dim=1).unsqueeze(-1)  # [N, L, 1]
        term = (phi_q[rows, query] * phi_k[rows, key]).sum(-1, keepdim=True)  # [P * heads, 1]
        num = num.index_put((rows, query), -term * v64[rows, key], accumulate=True)
        den = den.index_put((rows, query), -term, accumulate=True)
        # A row with no open key left (impossible with CLS in the sequence) reads 0, as on the
        # explicit path, rather than dividing a rounding residue by nothing.
        out = torch.where(den > tiny, num / den.clamp_min(tiny), torch.zeros_like(num))
        return out.to(q.dtype), None


class PerformerAttention(KernelAttention):
    """Softmax attention estimated with positive orthogonal random features (Performer's FAVOR+).

    With ``x = q / d^(1/4)`` and ``y = k / d^(1/4)`` (``d`` the head width) and ``m`` random rows
    ``w_r``, ``phi(x) = exp(w_r . x - |x|^2 / 2) / sqrt(m)`` gives ``E[phi(x) . phi(y)] = exp(q . k / sqrt(d))``
    -- the softmax kernel, estimated without bias and with positive terms. The mask, the float64
    sums, padding and the explicit path are :class:`KernelAttention`'s. Each head's map has rank at
    most ``m`` (apart from the blocked pairs) whatever the head width.

    * **RoPE** turns q and k before the feature map, so the estimate is of ``exp(q . R_(n-m) k / sqrt(d))``,
      the RoPE softmax kernel, with positive, normalised weights.
    * **Attention biases** arrive as low-rank factors ``U, W`` (``bias = U W^T``) and extend the
      feature map's input, ``x = [q / d^(1/4), U]``, ``y = [k / d^(1/4), W]``, so the estimate is of
      ``exp(q . k / sqrt(d) + bias)``. ``bias_factor_dim`` is their width.
    * **Features** are a buffer ``favor_projection`` (in the checkpoint), drawn from a generator the
      transformer seeds from ``optim.seed``, never from the global RNG. ``redraw_interval`` > 0 draws
      new ones every that many training forward passes; 0 keeps the first draw.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        num_features: int,
        bias_factor_dim: int = 0,
        redraw_interval: int = 0,
        bias: bool = True,
    ):
        super().__init__(embed_dim, num_heads, head_dim, dropout=0.0, bias=bias)
        self.num_features, self.bias_factor_dim = int(num_features), int(bias_factor_dim)
        self.redraw_interval = int(redraw_interval)
        self.feature_generator = torch.Generator().manual_seed(0)  # replaced by the transformer's
        # Set by `TransformerNodeEncoderHook.recording_attention`: the explicit path then also forms
        # the exact softmax this layer estimates, for the diagnostics' total-variation distance.
        self.record_reference = False
        self.reference_weights = None
        self.register_buffer(
            "favor_projection",
            orthogonal_gaussian(self.num_features, self.head_dim + self.bias_factor_dim, self.feature_generator),
        )
        self._training_calls = 0

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", num_features={self.num_features}, bias_factor_dim={self.bias_factor_dim}"
            + f", redraw_interval={self.redraw_interval}"
        )

    def draw_features(self, generator: torch.Generator | None = None) -> None:
        """Draw new random features, from ``generator`` (and keep it for later redraws) or the stored one."""
        if generator is not None:
            self.feature_generator = generator
        fresh = orthogonal_gaussian(self.num_features, self.head_dim + self.bias_factor_dim, self.feature_generator)
        with torch.no_grad():
            self.favor_projection.copy_(fresh.to(self.favor_projection.device))

    def _features(self, x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """``[N, L, m]`` positive features of ``[N, L, d']`` inputs; ``valid`` marks the keys (None: queries).

        A query's stabiliser is its own largest exponent, a key's the largest over a row's valid
        keys: both are constant within one query's normalisation, so they cancel exactly.
        """
        logits = x @ self.favor_projection.to(x.dtype).T - 0.5 * (x * x).sum(-1, keepdim=True)
        if valid is None:
            stab = logits.amax(dim=-1, keepdim=True)
        else:
            stab = logits.masked_fill(~valid[..., None], float("-inf")).amax(dim=(-2, -1), keepdim=True)
        features = torch.exp(logits - stab.detach()) / math.sqrt(self.num_features)
        return features if valid is None else features * valid[..., None]

    def _feature_maps(self, q, k, valid, context):
        if self.training and self.redraw_interval > 0:
            if self._training_calls and self._training_calls % self.redraw_interval == 0:
                self.draw_features()
            self._training_calls += 1

        n_rows, seq_len, head_dim = q.shape
        scale = head_dim**-0.25
        x, y = q * scale, k * scale
        if self.bias_factor_dim:
            if context.bias_factors is None:
                raise ValueError(f"this performer was built for {self.bias_factor_dim} bias dims but got no factors.")
            u, w = context.bias_factors
            if u.shape != (n_rows, seq_len, self.bias_factor_dim):
                raise ValueError(
                    f"bias factors of shape {tuple(u.shape)} do not match ({n_rows}, {seq_len}, {self.bias_factor_dim})."
                )
            x, y = torch.cat([x, u.to(x.dtype)], dim=-1), torch.cat([y, w.to(y.dtype)], dim=-1)
        if self.record_reference:
            self.reference_weights = self._exact_softmax(x, y, valid, context).to(q.dtype)
        return self._features(x, None), self._features(y, valid)

    @torch.no_grad()
    def _exact_softmax(self, x, y, valid, context) -> torch.Tensor:
        """``[N, L, L]``: the softmax attention this layer estimates, ``softmax(x . y)`` with the mask.

        ``x . y = q . k / sqrt(d) + bias``, the exponent FAVOR+ estimates ``exp`` of; blocked pairs
        and padded keys are excluded, as in the estimate. Float64, so a sharp row stays exact.
        """
        logits = x.double() @ y.double().transpose(1, 2)
        rows, query, key = head_pairs(context.blocked, self.num_heads)
        logits = logits.index_put((rows, query, key), logits.new_full((rows.numel(),), float("-inf")))
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        return torch.softmax(logits, dim=-1)


class LinearAttention(KernelAttention):
    """Linear attention (Katharopoulos et al., ICML 2020): the similarity ``phi(q) . phi(k)``, ``phi(x) = elu(x) + 1``.

    A fixed, deterministic positive feature map: not an estimate of softmax but a different
    similarity -- linear in the inputs above zero (``elu(x) + 1 = x + 1``), exponential below. No
    random features, no buffers and no parameters beyond the projections, so its outputs and maps
    are deterministic and a dense checkpoint loads into it as it is. The mask, the float64 sums,
    padding and the explicit path are :class:`KernelAttention`'s. What it is for, next to dense and
    the Performer: a control with the same O(S) structure but neither the softmax kernel nor random
    features; a probe of how much attention rank a task needs; the cheapest backend.

    **Rank.** ``phi`` has as many features as the head has dims, so each head's map has rank at
    most ``head_dim`` (apart from the blocked pairs): every query's attention is a mix of the same
    ``head_dim`` key profiles, and the whole sequence reaches a query through ``head_dim`` summary
    vectors, the rows of ``phi(K)^T V``. Softmax escapes the bound because the exponential of a
    low-rank logit matrix is generally full rank. The head width (``attention.head_dim``) is
    therefore this backend's main knob.

    **What it does not carry**, both because it lacks the exponential softmax has:

    * attention biases -- in softmax a bias ``b`` multiplies a pair's weight by ``exp(b)``; folding a
      bias's factors into ``phi``'s input would be some other modulation, not that factor;
    * RoPE -- turned before ``elu + 1``, the similarity depends on the two cells' absolute positions,
      not on their offset. RoFormer's linear form (rotating the features after ``phi``, in the
      numerator only) keeps the offset but gives weights that can be negative and do not sum to 1,
      which relevance and net flow cannot read.

    The Performer carries both.
    """

    supports_rotary = False

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, bias: bool = True):
        super().__init__(embed_dim, num_heads, head_dim, dropout=0.0, bias=bias)

    def _feature_maps(self, q, k, valid, context):
        if context.bias_factors is not None:
            raise ValueError("linear attention carries no attention bias (see LinearAttention); the performer does.")
        return F.elu(q) + 1.0, (F.elu(k) + 1.0) * valid[..., None]


#: How an Exphormer layer tells its edge types apart: a learned bias per type and head on the logit,
#: or Exphormer's own per-type scaling of each key dimension.
EXPHORMER_EDGE_TYPES = ("bias", "key")


class ExphormerAttention(ProjectedAttention):
    """Softmax attention along a sparse attention graph (Exphormer; Shirzad et al., ICML 2023).

    Each query attends only to its pairs in the pattern (:func:`~interscale.tl.masking.attention_pattern`):
    the neighbour graph's edges, a random expander and the virtual node, CLS, linked to every cell --
    minus every pair the long-range mask blocks. The softmax is exact, over each query's pairs: no
    approximation, and O(E) memory and time for ``E`` pairs instead of ``S x S``.

    * **Edge types.** ``edge_type="bias"`` adds a learned bias per type and head to the logit
      (``edge_type_bias``, zero at the start); ``"key"`` scales each key dimension per type and head,
      as Exphormer's edge embeddings do (``edge_type_scale``, one at the start). Either way the
      layer starts as plain attention along the pattern.
    * **Attention biases** come per pair (``AttentionBias.pairs``), so every bias with a pair form
      works -- the distance profile included.
    * **RoPE** turns queries and keys before the pairs are gathered, as in dense attention.
    * **Dropout** drops attention weights, as in dense attention.
    * **Maps.** Asked for weights, it scatters each pair's weight into the ``[B * heads, S, S]`` map
      (zero off the pattern) and computes the output from it, so a gradient hook on the map sees the
      gradient through it; otherwise the output is a scatter-add over the pairs.

    With the pattern of every open pair it is dense attention, which is how it is tested.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, *, dropout: float = 0.0, edge_type: str = "bias"):
        super().__init__(embed_dim, num_heads, head_dim, dropout=dropout)
        if edge_type not in EXPHORMER_EDGE_TYPES:
            raise ValueError(f"edge_type must be one of {EXPHORMER_EDGE_TYPES}, got {edge_type!r}.")
        self.edge_type = edge_type
        if edge_type == "bias":
            self.edge_type_bias = nn.Parameter(torch.zeros(len(PATTERN_EDGE_TYPES), self.num_heads))
        else:
            self.edge_type_scale = nn.Parameter(torch.ones(len(PATTERN_EDGE_TYPES), self.num_heads, self.head_dim))

    def extra_repr(self) -> str:
        return super().extra_repr() + f", edge_type={self.edge_type}"

    def _attend(self, q, k, v, padding, attn_mask, need_weights, context):
        if attn_mask is not None or not isinstance(context, StructuredMask) or context.pairs is None:
            raise ValueError(
                "the Exphormer attention attends along a pattern, a StructuredMask with pairs; "
                "common_step_local_to_global builds it."
            )
        n_rows, seq_len, head_dim = q.shape
        graph, query, key = context.pairs
        n_pairs = graph.numel()
        # Each pair for every head, pair-major: entry e * heads + h is pair e in head h.
        heads = torch.arange(self.num_heads, device=q.device).repeat(n_pairs)
        rows = graph.repeat_interleave(self.num_heads) * self.num_heads + heads
        query, key = query.repeat_interleave(self.num_heads), key.repeat_interleave(self.num_heads)
        types = context.pair_types.repeat_interleave(self.num_heads)

        keys = k[rows, key]
        if self.edge_type == "key":
            keys = keys * self.edge_type_scale[types, heads]
        score = (q[rows, query] * keys).sum(-1) / math.sqrt(head_dim)
        if self.edge_type == "bias":
            score = score + self.edge_type_bias[types, heads]
        if context.pair_bias is not None:
            score = score + context.pair_bias.reshape(-1).to(score.dtype)

        target = rows * seq_len + query  # the (row, query) each pair belongs to
        weights = scatter_softmax(score, target, num_nodes=n_rows * seq_len)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        if need_weights:
            full = q.new_zeros(n_rows, seq_len, seq_len).index_put((rows, query, key), weights)
            return full @ v, full
        out = q.new_zeros(n_rows * seq_len, head_dim).index_add(0, target, weights[:, None] * v[rows, key])
        return out.view(n_rows, seq_len, head_dim), None


# --------------------------------------------------------------------------- registry entries


def _no_settings(attention_cfg):
    """A backend with no settings of its own: nothing to check."""


@register_attention(
    "dense",
    validate=_no_settings,
    mask="dense",
    bias_forms=frozenset({"dense"}),
    rotary=True,
    training_maps=True,
)
def _build_dense(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    if head_dim * num_heads == embed_dim:
        return None  # torch's vendored attention, exactly as before backends existed
    return DenseAttention(embed_dim, num_heads, head_dim, dropout=dropout)


def _validate_entmax(attention_cfg):
    entmax = attention_cfg.get("entmax", None)
    if entmax is None:
        raise ValueError("attention.kind 'entmax' needs the attention.entmax block.")
    alpha, learn = float(entmax.alpha), bool(entmax.learn_alpha)
    if learn and not EntmaxAttention.MIN_ALPHA <= alpha < 2.0:
        raise ValueError(
            f"attention.entmax.alpha starts a learned alpha = 1 + sigmoid(a), so it must lie in "
            f"[{EntmaxAttention.MIN_ALPHA}, 2), got {alpha}."
        )
    if not learn and not EntmaxAttention.MIN_ALPHA <= alpha <= 2.0:
        raise ValueError(f"attention.entmax.alpha must lie in [{EntmaxAttention.MIN_ALPHA}, 2], got {alpha}.")
    if int(entmax.n_iter) < 1:
        raise ValueError(f"attention.entmax.n_iter must be >= 1, got {entmax.n_iter}.")
    try:
        import entmax as _  # noqa: F401
    except ImportError:
        raise ValueError("attention.kind 'entmax' needs the `entmax` package (pip install entmax).") from None


@register_attention(
    "entmax",
    validate=_validate_entmax,
    mask="dense",
    bias_forms=frozenset({"dense"}),
    rotary=True,
    training_maps=True,
)
def _build_entmax(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    entmax = attention_cfg.entmax
    return EntmaxAttention(
        embed_dim,
        num_heads,
        head_dim,
        dropout=dropout,
        alpha=float(entmax.alpha),
        learn_alpha=bool(entmax.learn_alpha),
        n_iter=int(entmax.n_iter),
    )


def _validate_topk(attention_cfg):
    topk = attention_cfg.get("topk", None)
    if topk is None:
        raise ValueError("attention.kind 'topk' needs the attention.topk block.")
    if int(topk.k) < 1:
        raise ValueError(f"attention.topk.k must be >= 1, got {topk.k}.")


@register_attention(
    "topk",
    validate=_validate_topk,
    mask="dense",
    bias_forms=frozenset({"dense"}),
    rotary=True,
    training_maps=True,
)
def _build_topk(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    return TopKAttention(embed_dim, num_heads, head_dim, dropout=dropout, k=int(attention_cfg.topk.k))


def _validate_performer(attention_cfg):
    performer = attention_cfg.get("performer", None)
    if performer is None:
        raise ValueError("attention.kind 'performer' needs the attention.performer block.")
    if not performer.num_features > 0:
        raise ValueError(f"attention.performer.num_features must be > 0, got {performer.num_features}.")
    if not performer.redraw_interval >= 0:
        raise ValueError(
            f"attention.performer.redraw_interval must be >= 0 (0 keeps one draw), got {performer.redraw_interval}."
        )


@register_attention(
    "performer",
    validate=_validate_performer,
    mask="pairs",
    bias_forms=frozenset({"factors"}),
    rotary=True,
    training_maps=False,
)
def _build_performer(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    performer = attention_cfg.performer
    return PerformerAttention(
        embed_dim,
        num_heads,
        head_dim,
        num_features=int(performer.num_features),
        bias_factor_dim=bias_factor_dim,
        redraw_interval=int(performer.redraw_interval),
    )


#: Below this head width, linear attention's rank bound is tight enough to warn about.
LINEAR_RANK_WARNING = 32


@register_attention(
    "linear",
    validate=_no_settings,
    mask="pairs",
    bias_forms=frozenset(),
    rotary=False,
    training_maps=False,
    note=(
        "elu+1 linear attention has no exponential: a softmax bias (a factor exp(b) on a pair's weight) "
        "has nothing to act on, and RoPE turned before elu+1 no longer depends on offsets alone. "
        "The performer carries both."
    ),
)
def _build_linear(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    if head_dim < LINEAR_RANK_WARNING:
        warnings.warn(
            f"attention.kind 'linear' at a head width of {head_dim}: elu+1 has as many features as the head "
            f"has dims, so every head's attention map has rank <= {head_dim} -- each query mixes the same "
            f"{head_dim} key profiles. Set attention.head_dim (>= {LINEAR_RANK_WARNING}) unless that bound "
            "is what you are measuring.",
            UserWarning,
            stacklevel=2,
        )
    return LinearAttention(embed_dim, num_heads, head_dim)


def _validate_exphormer(attention_cfg):
    exphormer = attention_cfg.get("exphormer", None)
    if exphormer is None:
        raise ValueError("attention.kind 'exphormer' needs the attention.exphormer block.")
    if exphormer.degree < 0 or exphormer.degree % 2:
        raise ValueError(
            "attention.exphormer.degree must be an even number >= 0 (the expander is built from degree / 2 "
            f"permutations; 0 means none), got {exphormer.degree}."
        )
    if exphormer.edge_type not in EXPHORMER_EDGE_TYPES:
        raise ValueError(
            f"attention.exphormer.edge_type must be one of {EXPHORMER_EDGE_TYPES}, got {exphormer.edge_type!r}."
        )


def _exphormer_pattern(attention_cfg, *, edge_index, batch, index_nodes, blocked, seed, generator, training, device):
    """The Exphormer attention graph of a batch: fixed per graph, or redrawn per training step if ``resample``."""
    exphormer = attention_cfg.exphormer
    return attention_pattern(
        edge_index,
        batch,
        index_nodes,
        blocked,
        graph_edges=bool(exphormer.graph_edges),
        degree=int(exphormer.degree),
        seed=seed,
        generator=generator if training and bool(exphormer.resample) else None,
        device=device,
    )


@register_attention(
    "exphormer",
    validate=_validate_exphormer,
    mask="pattern",
    bias_forms=frozenset({"pairs"}),
    rotary=True,
    training_maps=False,
    pattern=_exphormer_pattern,
)
def _build_exphormer(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    return ExphormerAttention(
        embed_dim, num_heads, head_dim, dropout=dropout, edge_type=str(attention_cfg.exphormer.edge_type)
    )


def build_attention_factory(attention_cfg, *, embed_dim: int, num_heads: int, dropout: float, bias_factor_dim: int):
    """A zero-argument builder of one layer's ``self_attn``, or None for torch's vendored attention."""
    kind = attention_kind(attention_cfg)
    if kind not in ATTENTION_REGISTRY:
        raise ValueError(f"unknown attention backend {kind!r}. Known: {sorted(ATTENTION_REGISTRY)}.")
    head_dim = attention_head_dim(attention_cfg, embed_dim, num_heads)
    spec = ATTENTION_REGISTRY[kind]

    def factory():
        return spec.build(
            attention_cfg,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias_factor_dim=bias_factor_dim,
        )

    # Torch's attention needs no factory: the layer builds it exactly where it always has.
    if kind == "dense" and head_dim * num_heads == embed_dim:
        return None
    return factory
