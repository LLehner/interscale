from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import TransformerEncoderLayer

from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits


def cell_positions(x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
    """``[N, L]`` boolean: which positions of the ``[L, N, E]`` tokens are cells -- not padding, not CLS.

    CLS is the last position of every sequence. ``key_padding_mask`` is True (or ``-inf``) at
    padding, as the encoder hands it on.
    """
    seq_len, n_graphs, _ = x.shape
    if key_padding_mask is None:
        cells = torch.ones(n_graphs, seq_len, dtype=torch.bool, device=x.device)
    elif key_padding_mask.dtype == torch.bool:
        cells = ~key_padding_mask
    else:
        cells = ~torch.isneginf(key_padding_mask)
    cells = cells.clone()
    cells[:, -1] = False
    return cells


def center_tokens(x: Tensor, cells: Tensor) -> Tensor:
    """Each sequence's tokens minus the mean of its cell tokens, layer-normalised without affine weights.

    What queries and keys are computed from under ``attention.center_qk``. Most of every spot's
    expression is the profile all spots share, so raw tokens are nearly identical and their dot
    products barely tell keys apart; subtracting the sequence mean leaves what differs between
    spots, and the normalisation gives it the scale a token had before.
    """
    weight = cells.T.unsqueeze(-1).to(x.dtype)  # [L, N, 1]
    mean = (x * weight).sum(0, keepdim=True) / weight.sum(0, keepdim=True).clamp_min(1.0)
    return F.layer_norm(x - mean, x.shape[-1:])


def mean_pairwise_cosine(x: Tensor, cells: Tensor) -> Tensor:
    """``[N]``: per sequence, the mean cosine between two different cell tokens (1: all identical)."""
    unit = F.normalize(x, dim=-1) * cells.T.unsqueeze(-1).to(x.dtype)
    n = cells.sum(1).to(x.dtype)
    total = unit.sum(0)  # [N, E]
    return ((total * total).sum(-1) - n) / (n * (n - 1)).clamp_min(1.0)


class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    """Overwrite TransformerEncoderLayer from PyTorch to implement attention map and gradient hook in the self-attention block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        attention_factory: Callable | None = None,
        center_qk: bool = False,
        *args,
        **kwargs,
    ):

        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )
        factory_kwargs = {"device": None, "dtype": None}
        if attention_factory is None:
            self.self_attn = MultiHeadAttentionWithEdits(
                d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs
            )
        else:
            # Another attention backend (`attention.py`). Built in a forked RNG, which starts it from
            # the state the default attention would have drawn from -- so at the default head width
            # its projections equal the dense model's at the same seed -- and then the default
            # attention's draws are replayed, so every weight built after this layer is unchanged.
            with torch.random.fork_rng(devices=[]):
                self.self_attn = attention_factory()
            MultiHeadAttentionWithEdits(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.attn_output = None  # To store attention output
        self.attn_output_weights = None
        self.attn_gradients = None  # To store attention weights gradients
        # Whether to hook the attention map for its gradient (relevance), and whether to keep the
        # map at all. Both set per call by `TransformerNodeEncoderHook.forward`; off, a training
        # step neither stores a gradient nor keeps a map alive past the step.
        self.register_hook = False
        self.keep_maps = False
        # ``(cos, sin)`` of the rotary position embedding for the current forward pass, or None.
        # Set and cleared around each call by `TransformerNodeEncoderHook.forward`, because
        # `nn.TransformerEncoder` passes nothing but the masks on to its layers.
        self.rotary = None
        # What an attention backend needs beyond the dense mask for this pass (the blocked pairs and
        # bias factors of `attention.StructuredMask`), set and cleared like `rotary`.
        self.attention_context = None
        # `attention.center_qk`: queries and keys from each sequence's mean-centred tokens
        # (`center_tokens`); values and the residual stream keep the tokens as they are.
        self.center_qk = bool(center_qk)
        # Set by `TransformerNodeEncoderHook.recording_attention`: then each pass stores the mean
        # pairwise cosine of the cell tokens entering attention, before any centring, per sequence.
        self.record_tokens = False
        self.token_cosine = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attn_output(self, attn_output):
        self.attn_output = attn_output

    def get_attn_output(self):
        return self.attn_output

    def save_attn_output_weights(self, attn_output_weights):
        self.attn_output_weights = attn_output_weights

    def get_attn_output_weights(self):
        return self.attn_output_weights

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        """Adjusted self-attention block to save gradients when register_hook is True.

        Output:
            attention output: Tensor
                when input is unbatched (L, E), or when batched (L,N,E)
            attention_weights: Tensor
                attention weigths between L (target sequence) and S (source sequence)
                unbatched: (L,S) or batched: (N,L,S) where
        """
        return self._attend(x, x, attn_mask, key_padding_mask, is_causal)

    def _attend(self, x, memory, attn_mask, key_padding_mask, is_causal):
        """Attention of the tokens ``x`` ``[L, N, E]`` over ``memory`` ``[L, N, E]`` (its keys and values), with the hooks.

        Self-attention is ``memory is x``: queries, keys and values from one sequence, as ever. Anything else is a
        read-out (:meth:`read_context`): the queries come from ``x``, the keys and values from ``memory``, row ``l`` of
        each being the same cell (the mask, the rotation and the padding are laid out once for both).
        """
        # Torch's vendored attention always takes its explicit path (need_weights=True), which is
        # what keeps its numbers unchanged. Other backends compute the map only when it is kept, and
        # may otherwise take a cheaper path to the same output.
        backend = getattr(self.self_attn, "is_backend", False)
        extra = {"context": self.attention_context} if backend else {}
        self_attention = memory is x
        query = key = x
        if not self_attention:
            key = memory
        if self.center_qk or self.record_tokens:
            cells = cell_positions(memory, key_padding_mask)
            if self.record_tokens:
                self.token_cosine = mean_pairwise_cosine(memory.detach(), cells)
            if self.center_qk:
                if self_attention:
                    query = key = center_tokens(x, cells)
                else:
                    query, key = center_tokens(x, cells), center_tokens(memory, cells)
        attn_output, attn_output_weights = self.self_attn(
            query,
            key,
            memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.keep_maps or not backend,
            average_attn_weights=False,  # return weigths per head
            is_causal=is_causal,  # from nn.TransformerEncoderLayer
            rotary=self.rotary,
            **extra,
        )
        self.save_attn_output(attn_output)
        # The map is computed either way (the explicit path is what keeps the numbers unchanged),
        # but only kept when something reads it.
        self.save_attn_output_weights(attn_output_weights if self.keep_maps else None)
        if self.register_hook and attn_output_weights.requires_grad:
            attn_output_weights.register_hook(self.save_attn_gradients)
        # if self.register_hook:
        #     attn_output_weights.register_hook(self.save_attn_gradients) # Note: Source code modification in MultiHeadAttention Module (F.multi_head_attention_forward function) to calculate gradient on weights
        return self.dropout1(attn_output)

    def read_context(self, queries, memory, attn_mask=None, key_padding_mask=None):
        """This layer as a step of a read-out stream: ``queries`` ``[L, N, E]`` read ``memory`` ``[L, N, E]``, then the feed-forward block.

        The pre-norm layer with the query stream in place of the tokens: ``q + attention(norm1(q), norm1(memory))``,
        then ``q + feed_forward(norm2(q))``. The weights are the layer's own (a self-attention layer's projections
        serve as cross-attention). The memory is read, never written: nothing of ``queries`` enters it, and the
        keys and values are the memory's alone -- which is what makes the stream's state a function of the other
        cells' tokens and the position (``TransformerNodeEncoderHook.forward``, ``context_only``).
        """
        if not self.norm_first:
            raise ValueError("read_context is the pre-norm layer; this one is post-norm.")
        queries = queries + self._attend(self.norm1(queries), self.norm1(memory), attn_mask, key_padding_mask, False)
        return queries + self._ff_block(self.norm2(queries))
