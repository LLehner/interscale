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
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from interscale.module.global_modules.transformer_utils import apply_rotary

#: Forms in which an attention bias can be handed to a backend: ``dense``, the ``[B * heads, S, S]``
#: float mask it is merged into today; ``factors``, low-rank ``U, W`` with ``bias = U W^T``, which a
#: backend folds into its queries and keys.
BIAS_FORMS = ("dense", "factors")


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
    """

    build: Callable
    validate: Callable
    mask: str
    bias_forms: frozenset = field(default_factory=frozenset)
    rotary: bool = True
    training_maps: bool = False


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


# --------------------------------------------------------------------------- registry entries


def _validate_dense(attention_cfg):
    pass


@register_attention(
    "dense",
    validate=_validate_dense,
    mask="dense",
    bias_forms=frozenset({"dense"}),
    rotary=True,
    training_maps=True,
)
def _build_dense(attention_cfg, *, embed_dim, num_heads, head_dim, dropout, bias_factor_dim):
    if head_dim * num_heads == embed_dim:
        return None  # torch's vendored attention, exactly as before backends existed
    return DenseAttention(embed_dim, num_heads, head_dim, dropout=dropout)


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
