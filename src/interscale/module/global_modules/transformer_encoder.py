import warnings

import torch
from torch import nn

from interscale.module.base import GlobalModule
from interscale.module.global_modules.transformer_encoder_layer import CustomTransformerEncoderLayer
from interscale.tl import (
    SelfAttentionRelevance,
    attn_mask_diagonal,
    create_transformer_attention_mask_from_edges,
    pad_batch,
)


class TransformerNodeEncoderHook(GlobalModule):
    """
    Sequence of: Dropout → Layer Norm → FC → nonlinearity → Dropout → FC → Dropout → Layer Norm + residual connections
    """

    def __init__(
        self,
        max_seq_len: int,
        n_heads: int = 4,
        act_func: nn.Module = nn.ReLU(),
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout_global: float = 0.1,
        long_range_attention: bool = True,
        local_mask_hops: int = 1,
        positional_encoding: nn.Module | None = None,
        attention_bias: nn.Module | None = None,
        **base_module_kwargs,
    ):

        super().__init__(**base_module_kwargs)
        # Save model parameters
        self.model_type = "TransformerEncoder"
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.act_func = act_func
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_global = dropout_global
        self.long_range_attention = long_range_attention
        # Radius of the neighbourhood the transformer is blocked from, in message-passing steps.
        # It has to match the local component's depth: blocking 1 hop while the GCN mixes 2 leaves
        # the second hop reachable by both components, which is the duplication the mask exists
        # to prevent. The model passes the local module's `receptive_field_hops`; 0 means nothing
        # was mixed, so only self-attention is blocked, exactly as without long-range masking.
        if local_mask_hops < 0:
            raise ValueError(f"local_mask_hops must be >= 0, got {local_mask_hops}.")
        self.local_mask_hops = local_mask_hops
        if long_range_attention and local_mask_hops == 0:
            warnings.warn(
                "long_range_attention is on, but the local component mixes no neighbours into the "
                "embeddings (there is none, or it does no message passing), so there is nothing to "
                "block beyond the cell itself. The mask is the self-only default.",
                UserWarning,
                stacklevel=2,
            )

        # Create Transformer Encoder
        encoder_layer = CustomTransformerEncoderLayer(
            self.n_embed, self.n_heads, self.dim_feedforward, self.dropout_global, self.act_func, norm_first=True
        )
        encoder_norm = nn.LayerNorm(self.n_embed)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, norm=encoder_norm)

        self.norm_input = nn.LayerNorm(self.n_embed)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.n_embed], requires_grad=True))

        # Register self-attention relevance hook
        self.self_attn_relevance = SelfAttentionRelevance(self.transformer_encoder)

        # Node positional encodings (`positional_encodings.py`), or None when the config enables
        # none -- in which case nothing is registered and the module is the one from before PEs.
        self.positional_encoding = positional_encoding
        # Attention biases (`positional_encodings.AttentionBias`), merged into the attention mask in
        # `common_step_local_to_global`, or None -- nothing registered, the mask stays boolean.
        self.attention_bias = attention_bias

    def common_step_local_to_global(self, batched_data, emb: torch.Tensor, eval_step: bool = False):
        """Convert local node embeddings ``[N, E]`` to padded local node embeddings ``[max_seq_len, E]``.

        ``N`` is the number of nodes in the graph and ``E`` the embedding dimension.

        Parameters
        ----------
        batched_data
            PyTorch Geometric batch object.
        emb
            ``torch.Tensor`` of shape ``[N, E]``: embedding of the local model
            or user-provided embeddings.
        eval_step
            If True, the transformer encoder is not masked.

        Returns
        -------
        Tuple of ``(padded_emb, src_padding_mask, index_nodes)``:

        - ``padded_emb`` (``torch.Tensor`` of shape ``[max_seq_len, B, E]``):
          padded local node embeddings.
        - ``src_padding_mask`` (``torch.Tensor`` of shape ``[max_seq_len]``):
          mask indicating padding nodes.
        - ``index_nodes`` (``torch.Tensor`` of shape ``[N]``): indices of the
          nodes in the original graph.
        """
        # Layer normalization
        emb = self.norm_input(emb)
        if self.positional_encoding is not None:
            # Added to the flat [N, E] tokens BEFORE pad_batch: pad_batch subsamples with its own
            # randomness when a graph exceeds max_seq_len, so anything per-cell must ride along
            # here or be gathered by the index_nodes it returns -- never padded a second time.
            emb = emb + self.positional_encoding(batched_data, dtype=emb.dtype, device=emb.device)

        if self.masked_nodes and not eval_step:
            keep_indices = batched_data.mask
        else:
            keep_indices = None

        # Ensure masked nodes are included in padding; during evaluation, no masking is applied
        padded_emb, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
            emb,
            batched_data.batch,
            self.max_seq_len,
            get_mask=False if eval_step else self.masked_nodes,
            keep_indices=None
            if eval_step
            else keep_indices,  # Add parameter to ensure masked nodes are kept (not during evaluation)
        )

        if self.long_range_attention and self.local_mask_hops > 0:
            # Block everything the local component has already seen, so the transformer can only
            # contribute what the GNN could not.
            attention_mask = create_transformer_attention_mask_from_edges(
                batched_data.edge_index,
                batched_data.batch.numel(),
                batched_data.batch,
                index_nodes,
                self.n_heads,
                n_hops=self.local_mask_hops,
                device=emb.device,
            )
        else:
            # default: no attention to self, everything else open
            attention_mask = attn_mask_diagonal(batched_data.batch, index_nodes, self.n_heads, emb.device)

        # Boolean throughout: True is blocked. Keeping it out of float space is what stops the
        # 0 * -inf that an "inverse adjacency" built by arithmetic would produce.
        attention_mask = attention_mask.to(dtype=torch.bool)

        if self.attention_bias is not None:
            # Merged by indexing, never by arithmetic: blocked pairs become -inf, every other pair
            # keeps its finite bias. The bias is laid out from this call's index_nodes, and its CLS
            # row and column are 0, so CLS stays attendable for every query and no row goes NaN.
            # The returned mask is then a float mask, which forward() passes on as an additive one.
            bias = self.attention_bias(batched_data, index_nodes, src_padding_mask, dtype=emb.dtype, device=emb.device)
            attention_mask = bias.masked_fill(attention_mask, float("-inf"))

        return padded_emb, src_padding_mask, index_nodes, attention_mask

    def forward(self, padded_h_node, src_padding_mask, mask=None, register_hook: bool = True):
        """
        N_b_max: maximum number of nodes in the batch
        B: batch size
        H_d: dimension of transformer

        Input:
            padded_h_node: [N_b_max x B X H_d]
                GEX embeddings of the nodes in the batch with padding (0) to the maximum number of nodes in the batch.
            src_padding_mask: [B x N_b_max]
                Matrix indicating the size of the padding mask to be ignored during calculation.
            mask: [H_d, N_b_max x N_b_max]
                matrix indicating the long-range connections (inverse of adjacency matrix). Default: None
        """
        if register_hook:
            for encoder in self.transformer_encoder.layers:
                encoder.register_hook = True

        # append cls embedding
        expand_cls_embedding = self.cls_embedding.expand(
            1, padded_h_node.size(1), -1
        )  # expand cls embedding to the same batch size (1, B, E)
        padded_h_node = torch.cat(
            [padded_h_node, expand_cls_embedding], dim=0
        )  # append cls embedding at the end of the sequence
        # normalize input
        padded_h_node = self.norm_input(padded_h_node)
        assert not torch.any(torch.isnan(padded_h_node)), "normalized padded_h_node contains NaN values"

        zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
        src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        key_padding_mask = src_padding_mask
        if mask is not None and mask.is_floating_point():
            # A float mask (an attention bias merged in) needs a float key-padding mask, or torch
            # warns on every layer that mixing the two types is deprecated. The boolean one is
            # still what this method returns, for the callers that drop padding by it.
            key_padding_mask = torch.zeros_like(src_padding_mask, dtype=mask.dtype).masked_fill(
                src_padding_mask, float("-inf")
            )
        transformer_out = self.transformer_encoder(
            padded_h_node, src_key_padding_mask=key_padding_mask, mask=mask
        )  # (S, B, h_d)

        attn_matrices = []
        if register_hook:
            for i, encoder in enumerate(self.transformer_encoder.layers):
                if hasattr(encoder, "get_attn_output_weights"):
                    attn_matrices.append(encoder.get_attn_output_weights())
                elif hasattr(encoder, "attention_map"):
                    attn_matrices.append(encoder.attention_map)
                elif hasattr(encoder, "attn"):
                    attn_matrices.append(encoder.attn)

                encoder.register_hook = False

        if len(attn_matrices) > 0:
            final_attn = torch.stack(attn_matrices)
        else:
            final_attn = None

        return transformer_out, src_padding_mask, final_attn

    def evaluate(self, batched_data, embedding):
        """Evaluates transformer encoder on a batch of data without masking and registering hook.

        Args:
            batched_data (_type_): _description_

        Returns
        -------
            I: torch.Tensor [N+1, N+1]
                Self-attention relevance matrix with CLS token.
        """
        # evaluation on single graph
        batched_data.batch = torch.Tensor(len(batched_data.obs_names) * [0])
        transformer_in, src_padding_mask, pad_index_nodes, attn_mask = self.common_step_local_to_global(
            batched_data, embedding, eval_step=True
        )

        transformer_out, src_padding_mask, _ = self.forward(
            transformer_in, src_padding_mask, attn_mask, register_hook=True
        )

        last_layer = self.transformer_encoder.layers[-1]
        raw_attn = last_layer.get_attn_output_weights()
        raw_attn = raw_attn.detach().cpu()

        if raw_attn.dim() == 3:
            matrix = raw_attn[0]
        elif raw_attn.dim() == 4:
            matrix = raw_attn[0].mean(dim=0)
        else:
            matrix = raw_attn.squeeze()

        if matrix.dim() == 2:
            diag_val = torch.diag(matrix).mean().item()

        off_diag_mask = ~torch.eye(matrix.shape[0], dtype=bool)
        off_diag_val = matrix[off_diag_mask].mean().item()

        I = self.self_attn_relevance.generate_relevance(transformer_out)

        return transformer_in, transformer_out, src_padding_mask, pad_index_nodes, I

    def get_model_summary(self) -> str:
        """Returns a string containing the model's parameters summary.

        Returns
        -------
            str: Summary string with model parameters
        """
        summary = (
            f"Transformer Encoder Global Component: \n"
            f"max_seq_len: {self.max_seq_len}, \n"
            f"n_heads: {self.n_heads}, \n"
            f"act_func: {self.act_func}, \n"
            f"num_layers: {self.num_layers}, \n"
            f"long_range_attention: {self.long_range_attention}, \n"
            f"local_mask_hops: {self.local_mask_hops}, \n"
            f"positional_encoding: {sorted(self.positional_encoding.encoders) if self.positional_encoding else []}, \n"
            f"attention_bias: {sorted(self.attention_bias.encoders) if self.attention_bias else []}, \n"
        )
        return summary
