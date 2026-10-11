from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules import TransformerEncoderLayer

from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits


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
        # Torch's vendored attention always takes its explicit path (need_weights=True), which is
        # what keeps its numbers unchanged. Other backends compute the map only when it is kept, and
        # may otherwise take a cheaper path to the same output.
        backend = getattr(self.self_attn, "is_backend", False)
        extra = {"context": self.attention_context} if backend else {}
        attn_output, attn_output_weights = self.self_attn(
            x,
            x,
            x,
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
