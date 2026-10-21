import torch


class SelfAttentionRelevance:
    """Chefer, H., Gur, S. & Wolf, L. Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers.
    Preprint at https://doi.org/10.48550/arXiv.2103.15679 (2021).
    """

    def __init__(self, transformer_encoder):
        """
        Initializes the SelfAttentionRelevance class with a TransformerNodeEncoder.

        Parameters
        ----------
            transformer_encoder: torch.nn.Module
                The transformer encoder to be used for generating relevance.
        """
        self.transformer_encoder = transformer_encoder

    @staticmethod
    def avg_heads(attn_map, grad, n_heads: int | None = None):
        """
        Rule 5 from Chefer et al.: Averages the heads in the attention map after applying gradients.

        Parameters
        ----------
            attn_map: Tensor
                Attention weight (Key and Query dependent), shape: BHxSxS
            grad: Tensor
                Gradients to apply to the attention map, shape: BHxSxS
            n_heads: int, optional
                Heads per graph. Given, the heads are averaged within each graph, ``[B, S, S]``; None
                averages over all ``B * H`` maps at once, which is the same for one graph and mixes
                graphs for more.

        Returns
        -------
            Tensor
                Averaged attention map after applying gradients.
        """
        size = attn_map.shape[-2:]
        cam = (grad.reshape(-1, *size) * attn_map.reshape(-1, *size)).clamp(min=0)
        if n_heads is None:
            return cam.mean(dim=0)
        return cam.view(-1, n_heads, *size).mean(dim=1)

    @staticmethod
    def apply_self_attention_rules(R_ss, cam_ss):
        """
        Rule 6 from Chefer et al.: Applies self-attention rules to update the relevance score.

        Parameters
        ----------
            R_ss: Tensor
                Relevance score matrix.
            cam_ss: Tensor
                Attention map.

        Returns
        -------
            Tensor
                Updated relevance score matrix.
        """
        return torch.matmul(cam_ss, R_ss)

    def generate_relevance(self, transformer_output, category_index=None, target=None):
        """
        Generates the relevance score for a given input and category index.

        Parameters
        ----------
            transformer_output
                ``[S, B, E]``: the encoder's output from a pass that hooked every layer's attention map
                (``forward(..., register_hook=True)``).
            category_index: List[int], optional
                List of indices to consider in the category mask, default is None.
            target: Tensor, optional
                The scalar whose relevance is traced, computed from ``transformer_output`` (e.g. one
                cell's predicted programme). None backpropagates the sum of the output over
                ``category_index``: after the encoder's affine LayerNorm that sum is constant up to the
                norm's weights' deviation from uniform, an arbitrary direction (``.claude/
                linear_probing_docs.md``), so only a given target gives a readable relevance.

        Returns
        -------
            Tensor
                ``[S, S]`` for one graph, ``[B, S, S]`` for more: row ``i`` is what each token
                contributed to token ``i``'s output.
        """
        if target is None:
            category_mask = torch.ones_like(transformer_output)
            if category_index is not None:
                category_mask = torch.zeros_like(transformer_output)
                category_mask[category_index, :, :] = 1
            target = (transformer_output * category_mask).sum()
        self.transformer_encoder.zero_grad()
        target.backward(retain_graph=True)

        num_tokens, n_graphs = self.transformer_encoder.layers[0].get_attn_output().shape[:2]
        eye = torch.eye(num_tokens, num_tokens, device=transformer_output.device, dtype=transformer_output.dtype)
        relevance = eye.expand(n_graphs, -1, -1).clone()

        for encoder in self.transformer_encoder.layers:
            attn_out_weights = encoder.get_attn_output_weights()
            attn_grad = encoder.get_attn_gradients()

            attn_map = self.avg_heads(attn_out_weights, attn_grad, n_heads=encoder.self_attn.num_heads)
            relevance = relevance + self.apply_self_attention_rules(relevance, attn_map)

        return relevance[0] if n_graphs == 1 else relevance
