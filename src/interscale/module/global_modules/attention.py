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

from interscale.module.global_modules.transformer_utils import apply_rotary

#: Forms in which an attention bias can be handed to a backend: ``dense``, the ``[B * heads, S, S]``
#: float mask it is merged into today; ``factors``, low-rank ``U, W`` with ``bias = U W^T``, which a
#: backend folds into its queries and keys.
BIAS_FORMS = ("dense", "factors")


class StructuredMask(NamedTuple):
    """The attention mask for a backend that never forms the ``S x S`` matrix.

    It stands where the dense mask stands -- ``GlobalInput.attention_mask``, handed on to ``forward``
    unchanged by every caller -- and ``forward`` passes it to the layers as their per-batch context.

    ``blocked`` is ``[3, P]``: the ``(graph, query, key)`` positions of every blocked pair, exactly
    the True entries of the dense boolean mask (:func:`~interscale.tl.masking.blocked_pairs`); the
    CLS slot never appears. ``bias_factors``, when attention biases are on, are ``(U, W)``, each
    ``[B * heads, S + 1, r]`` and graph-major like the dense mask, with ``bias = U W^T`` -- zero
    rows at CLS and padding, so their bias is 0 there.
    """

    blocked: torch.Tensor
    bias_factors: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class AttentionSpec:
    """What the rest of the code needs to know about one attention backend.

    ``build(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim)`` returns
    the module a layer uses as ``self_attn``, or None for torch's vendored multi-head attention.
    ``validate(attention_cfg)`` raises on settings that would fail mid-run. ``mask`` is the form in
    which it takes the blocked pairs: ``"dense"`` (a boolean ``[B * heads, S+1, S+1]`` mask) or
    ``"pairs"`` (a list of ``(graph, query, key)`` positions). ``bias_forms`` are the attention-bias
    forms it can carry (:data:`BIAS_FORMS`); ``rotary`` whether it applies RoPE;
    ``training_maps`` whether its training forward passes form the maps a loss could read.
    ``note``, when set, says in a sentence what it cannot carry and why; the validator appends it to
    its rejections.
    """

    build: Callable
    validate: Callable
    mask: str
    bias_forms: frozenset = field(default_factory=frozenset)
    rotary: bool = True
    training_maps: bool = False
    note: str = ""


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
        if query is not key or key is not value:
            raise ValueError("attention backends are self-attention only: query, key and value must be one tensor.")
        seq_len, n_graphs, _ = query.shape
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
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
        weights = torch.softmax(weights, dim=-1)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        return torch.bmm(weights, v), (weights if need_weights else None)


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

        # Each blocked (graph, query, key) for every head: rows graph * heads + h, graph-major.
        graph, query, key = context.blocked
        heads = torch.arange(self.num_heads, device=q.device)
        rows = (graph[:, None] * self.num_heads + heads).reshape(-1)
        query, key = query.repeat_interleave(self.num_heads), key.repeat_interleave(self.num_heads)
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
        return self._features(x, None), self._features(y, valid)


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
