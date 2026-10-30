import warnings
from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch import nn

from interscale.module.base import GlobalModule
from interscale.module.global_modules.attention import (
    ATTENTION_REGISTRY,
    StructuredMask,
    attention_head_dim,
    attention_kind,
    build_attention_factory,
)
from interscale.module.global_modules.hierarchical_attention import (
    COARSE_BIAS,
    GraphLayout,
    HierarchicalContext,
    HierarchicalEncoderLayer,
)
from interscale.module.global_modules.hierarchy import HierarchyCache
from interscale.module.global_modules.transformer_encoder_layer import CustomTransformerEncoderLayer
from interscale.tl import (
    SelfAttentionRelevance,
    attn_mask_diagonal,
    blocked_pairs,
    create_transformer_attention_mask_from_edges,
    pad_batch,
    pad_like,
)


class _GlobalInputFields(NamedTuple):
    padded_emb: torch.Tensor  # [S, B, E]
    src_padding_mask: torch.Tensor  # [B, S], True at padding
    index_nodes: list  # per graph, the node indices pad_batch kept
    # [B * heads, S + 1, S + 1]: boolean, or float with a bias merged in -- or, for a backend that
    # never forms the S x S matrix, an `attention.StructuredMask` (blocked pairs, bias factors).
    attention_mask: torch.Tensor | StructuredMask
    rotary: tuple[torch.Tensor, torch.Tensor] | None = None


class GlobalInput(_GlobalInputFields):
    """What :meth:`TransformerNodeEncoderHook.common_step_local_to_global` hands to ``forward``.

    Everything in padded order: ``S`` tokens per graph, left-padded, plus CLS appended last by
    ``forward``. ``rotary`` is the ``(cos, sin)`` of the rotary encoding, each
    ``[B * heads, S + 1, head_dim]``, or None when it is off; pass it on as ``forward(...,
    rotary=...)``.

    ``queries`` is one more thing ``forward`` needs under ``context_only: query`` and is not a field of the tuple
    (the five fields are unpacked by position all over): ``[S, B, E]``, each token's node encoding laid out like
    ``padded_emb`` and zero where the model has none, the position half of the position-only queries. None
    otherwise; pass it on as ``forward(..., queries=...)``.
    """

    queries: torch.Tensor | None = None


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
        rotary_encoding: nn.Module | None = None,
        return_attention_maps: bool = False,
        attention=None,
        attention_seed: int = 0,
        hierarchy=None,
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

        # The attention backend (`attention.py`; the `attention` config block, None = dense) and the
        # width of one head, which `attention.head_dim` may set apart from n_embed / n_heads.
        self.attention_kind = attention_kind(attention)
        self.head_dim = attention_head_dim(attention, self.n_embed, self.n_heads)
        attention_factory = build_attention_factory(
            attention,
            embed_dim=self.n_embed,
            num_heads=self.n_heads,
            dropout=self.dropout_global,
            bias_factor_dim=attention_bias.factor_dim() if attention_bias is not None else 0,
        )

        # Create Transformer Encoder
        center_qk = bool(attention.get("center_qk", False)) if attention is not None else False
        # `hierarchy` (`hierarchical_attention.HierarchySettings`, or None): every layer makes a coarse-to-fine pass over a
        # cluster hierarchy of each slide instead of attending over every pair (`.claude/hierarchical_attention_plan.md`).
        # Its layers are the dense layer with another attention block, built from the same draws, so every weight they share
        # is the dense model's at the same seed.
        self.hierarchy = hierarchy
        if hierarchy is not None:
            if self.attention_kind != "dense" or self.head_dim != self.n_embed // self.n_heads:
                raise ValueError(
                    "the hierarchical attention computes its own attention with the dense layer's projections: use "
                    f"attention.kind dense at the default head width (got {self.attention_kind!r}, head width {self.head_dim})."
                )
            encoder_layer = HierarchicalEncoderLayer(
                self.n_embed,
                self.n_heads,
                self.dim_feedforward,
                self.dropout_global,
                self.act_func,
                settings=hierarchy,
                center_qk=center_qk,
            )
        else:
            encoder_layer = CustomTransformerEncoderLayer(
                self.n_embed,
                self.n_heads,
                self.dim_feedforward,
                self.dropout_global,
                self.act_func,
                norm_first=True,
                attention_factory=attention_factory,
                center_qk=center_qk,
            )
        encoder_norm = nn.LayerNorm(self.n_embed)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, norm=encoder_norm)
        if hierarchy is not None:
            for index, layer in enumerate(self.transformer_encoder.layers):
                layer.layer_index = index

        self.norm_input = nn.LayerNorm(self.n_embed)
        self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.n_embed], requires_grad=True))

        # Register self-attention relevance hook
        self.self_attn_relevance = SelfAttentionRelevance(self.transformer_encoder)

        # What the backend can carry, and its randomness: the layers are clones of one, so each
        # draws its own random features (Performer) here, in layer order, from one generator seeded
        # from `attention_seed` (optim.seed) -- never from the global RNG, so enabling a backend
        # shifts neither another weight's initialisation nor the batch order.
        self.attention_spec = ATTENTION_REGISTRY[self.attention_kind]
        self.attention_cfg = attention
        # `attention.visible_keys_only`: no cell attends to a cell hidden whole (`_without_hidden_keys`).
        self.visible_keys_only = bool(attention.get("visible_keys_only", False)) if attention is not None else False
        self.attention_seed = int(attention_seed)
        # Draws a sparse backend's pattern when it is resampled in training (Exphormer's `resample`).
        self.attention_generator = torch.Generator().manual_seed(self.attention_seed)
        generator = torch.Generator().manual_seed(int(attention_seed))
        for layer in self.transformer_encoder.layers:
            if hasattr(layer.self_attn, "draw_features"):
                layer.self_attn.draw_features(generator)

        if self.context_only != "none" and any(
            getattr(layer.self_attn, "is_backend", False) for layer in self.transformer_encoder.layers
        ):
            raise ValueError(
                f"context_only {self.context_only!r} reads the cells' tokens with separate queries, which only torch's "
                f"vendored dense attention takes; attention.kind {self.attention_kind!r} at head width {self.head_dim} "
                "is another backend (it is self-attention only). Use attention.kind dense at the default head width."
            )
        # Node positional encodings (`positional_encodings.py`), or None when the config enables
        # none -- in which case nothing is registered and the module is the one from before PEs.
        self.positional_encoding = positional_encoding
        # Attention biases (`positional_encodings.AttentionBias`), merged into the attention mask in
        # `common_step_local_to_global`, or None -- nothing registered, the mask stays boolean.
        self.attention_bias = attention_bias
        # The rotary encoding (`positional_encodings.RotaryEncoding`): its rotation is computed per
        # batch in `common_step_local_to_global` and applied to queries and keys in every layer by
        # `forward`. None -- nothing registered, the attention is the one from before PEs.
        self.rotary_encoding = rotary_encoding
        if hierarchy is not None:
            if hierarchy.coarse_bias not in COARSE_BIAS:
                raise ValueError(f"hierarchy.coarse_bias must be one of {COARSE_BIAS}, got {hierarchy.coarse_bias!r}.")
            if (
                rotary_encoding is not None
                or self.context_only != "none"
                or self.visible_keys_only
                or long_range_attention
            ):
                raise ValueError(
                    "the hierarchical attention carries no rotary encoding, context-only queries, visible-keys-only "
                    "mask or long-range mask."
                )
            # The coarse levels' distance bias between centroids: the cells' own biases (`shared`), a copy per level learned
            # apart (`own`, copied without drawing), or none.
            if attention_bias is not None and hierarchy.coarse_bias == "own":
                import copy

                self.hier_coarse_bias = nn.ModuleList(copy.deepcopy(attention_bias) for _ in range(hierarchy.n_levels))
            # Hierarchies by slide (built once from the positions), and the generator of the routed exploration draws.
            self.hierarchy_cache = HierarchyCache()
            self.hierarchy_generator = torch.Generator().manual_seed(self.attention_seed + 7919)
        # Set only inside `recording_hierarchy`: the list every hierarchical layer appends its export to.
        self._hierarchy_record = None
        # Whether a training step hands every layer's attention map to its loss (the attention
        # entropy loss reads them). Off, `forward` keeps no map unless asked for relevance.
        self.return_attention_maps = bool(return_attention_maps)
        # Set only inside `recording_attention`: the list each pass appends its maps and layout to.
        self._attention_record = None
        # Set only inside `hooking_attention`: training-style passes hook their maps for relevance too.
        self._hook_attention = False

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
        GlobalInput
            ``(padded_emb, src_padding_mask, index_nodes, attention_mask, rotary)``:

            - ``padded_emb`` ``[S, B, E]``: padded local node embeddings, ``S <= max_seq_len``.
            - ``src_padding_mask`` ``[B, S]``: True at padding.
            - ``index_nodes``: per graph, the indices of the nodes ``pad_batch`` kept.
            - ``attention_mask`` ``[B * heads, S + 1, S + 1]``: True (or ``-inf``) where blocked.
            - ``rotary``: the rotary encoding's ``(cos, sin)``, or None; ``forward`` takes it.
        """
        # Layer normalization
        emb = self.norm_input(emb)
        # Each positional input prepared this pass, by Data key -- the rotary encoding reads the
        # coordinates the node encodings used, so both see the same rotation of each graph.
        pe_inputs = {}
        node_encoding = None
        if self.positional_encoding is not None:
            # Added to the flat [N, E] tokens BEFORE pad_batch: pad_batch subsamples with its own
            # randomness when a graph exceeds max_seq_len, so anything per-cell must ride along
            # here or be gathered by the index_nodes it returns -- never padded a second time.
            node_encoding = self.positional_encoding(batched_data, dtype=emb.dtype, device=emb.device, inputs=pe_inputs)
            emb = emb + node_encoding

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

        if self.hierarchy is not None:
            # No S x S mask: the layers attend along the hierarchy, laid out from this call's index_nodes.
            context = self._hierarchical_context(batched_data, index_nodes, src_padding_mask, eval_step, emb.device)
            if self._attention_record is not None:
                self._attention_record.append(
                    {"index_nodes": index_nodes, "src_padding_mask": src_padding_mask.detach()}
                )
            return GlobalInput(padded_emb, src_padding_mask, index_nodes, context, None)

        if self.attention_spec.mask in ("pairs", "pattern"):
            # The same blocked set as the dense masks below, as a list of pairs. A backend that never
            # forms the S x S matrix takes it this way, with the attention biases in the form it
            # carries: low-rank factors it folds into its queries and keys (Performer), the bias of
            # each pair of the sparse pattern it attends along (Exphormer), or the bias a block of
            # query rows at a time, computed when the block runs (fused). All of it laid out from
            # this call's index_nodes.
            hops = self.local_mask_hops if self.long_range_attention else 0
            edge_index = getattr(batched_data, "edge_index", None)
            blocked = blocked_pairs(edge_index, batched_data.batch, index_nodes, n_hops=hops, device=emb.device)
            factors = pattern = pattern_types = pair_bias = bias_rows = None
            if self.attention_spec.mask == "pattern":
                pattern, pattern_types = self.attention_spec.pattern(
                    self.attention_cfg,
                    edge_index=edge_index,
                    batch=batched_data.batch,
                    index_nodes=index_nodes,
                    blocked=blocked,
                    seed=self.attention_seed,
                    generator=self.attention_generator,
                    training=self.training,
                    device=emb.device,
                )
                if self.attention_bias is not None:
                    pair_bias = self.attention_bias.pairs(
                        batched_data, index_nodes, src_padding_mask, pattern, dtype=emb.dtype, device=emb.device
                    )
            elif self.attention_bias is not None and "pairs" in self.attention_spec.bias_forms:
                bias_rows = self.attention_bias.rows(
                    batched_data, index_nodes, src_padding_mask, dtype=emb.dtype, device=emb.device
                )
            elif self.attention_bias is not None:
                factors = self.attention_bias.factors(
                    batched_data, index_nodes, src_padding_mask, dtype=emb.dtype, device=emb.device
                )
            attention_mask = StructuredMask(blocked, factors, pattern, pattern_types, pair_bias, bias_rows)
        elif self.long_range_attention and self.local_mask_hops > 0:
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

        if not isinstance(attention_mask, StructuredMask):
            # Boolean throughout: True is blocked. Keeping it out of float space is what stops the
            # 0 * -inf that an "inverse adjacency" built by arithmetic would produce.
            attention_mask = attention_mask.to(dtype=torch.bool)

        if self.visible_keys_only and not eval_step:
            if isinstance(attention_mask, StructuredMask):
                raise ValueError(
                    "attention.visible_keys_only needs a dense attention mask; this backend takes a pattern."
                )
            attention_mask = self._without_hidden_keys(
                attention_mask, batched_data, index_nodes, src_padding_mask.shape[1]
            )

        if self.attention_bias is not None and not isinstance(attention_mask, StructuredMask):
            # Merged by indexing, never by arithmetic: blocked pairs become -inf, every other pair
            # keeps its finite bias. The bias is laid out from this call's index_nodes, and its CLS
            # row and column are 0, so CLS stays attendable for every query and no row goes NaN.
            # The returned mask is then a float mask, which forward() passes on as an additive one.
            bias = self.attention_bias(batched_data, index_nodes, src_padding_mask, dtype=emb.dtype, device=emb.device)
            attention_mask = bias.masked_fill(attention_mask, float("-inf"))

        rotary = None
        if self.rotary_encoding is not None:
            # Per token, laid out by this call's index_nodes like the bias: never a second pad_batch.
            rotary = self.rotary_encoding(
                batched_data,
                index_nodes,
                src_padding_mask.shape[1],
                dtype=emb.dtype,
                device=emb.device,
                pos=pe_inputs.get("pos"),
            )

        if self._attention_record is not None:
            entry = {"index_nodes": index_nodes, "src_padding_mask": src_padding_mask.detach()}
            if not isinstance(attention_mask, StructuredMask):
                # Which keys each query may attend to, from the mask itself: a sparse backend
                # (entmax, top-k) puts exact zeros on open keys, so a weight of 0 does not mean
                # blocked. [B, S + 1, S + 1], the first head's (every head is blocked alike).
                blocked = attention_mask if attention_mask.dtype == torch.bool else torch.isneginf(attention_mask)
                n_graphs = src_padding_mask.shape[0]
                entry["open"] = ~blocked.view(n_graphs, -1, *blocked.shape[1:])[:, 0].detach()
            self._attention_record.append(entry)
        given = GlobalInput(padded_emb, src_padding_mask, index_nodes, attention_mask, rotary)
        if self.context_only != "none":
            given.queries = self._query_encoding(node_encoding, batched_data, index_nodes, padded_emb)
        return given

    def _hierarchical_context(
        self, batched_data, index_nodes, src_padding_mask, eval_step: bool, device
    ) -> HierarchicalContext:
        """The pass's :class:`~interscale.module.global_modules.hierarchical_attention.HierarchicalContext`.

        Per graph, the kept cells (``index_nodes``, so a subsampled graph is built on the cells that are in the sequence) in
        sequence order: their positions in µm, which of them is hidden whole (none on a clean pass), and the hierarchy built
        from the positions alone (cached per slide), or two under ``shift``. An ``obs_label`` top level also reads the kept
        cells' labels and the edges among them (a path through a cell ``pad_batch`` dropped does not connect a patch).
        """
        settings = self.hierarchy
        pos = getattr(batched_data, "pos", None)
        if pos is None:
            raise ValueError(
                "the hierarchical attention builds its clusters from cell coordinates, but the graphs carry no `pos`. "
                "Set dataset.spatial_key (e.g. 'spatial')."
            )
        pos = pos[:, :2].to(device=device, dtype=torch.float32) * float(settings.spatial_unit_um)
        hidden = self.wholly_hidden(batched_data).to(device) if not eval_step else None
        batch = batched_data.batch.to(device=device, dtype=torch.long)
        seq_len = src_padding_mask.shape[1]
        # A label top level (`obs_label`) also reads each cell's label and, to find its connected patches, the graph's edges.
        codes = None
        if settings.top_mode == "obs_label":
            from interscale.tl.niches import niche_codes

            codes = niche_codes(batched_data, settings.top_label).to(device)
            edges = batched_data.edge_index.to(device=device, dtype=torch.long)
            local = torch.full((len(batch),), -1, dtype=torch.long, device=device)
        graphs = []
        for g, kept in enumerate(index_nodes):
            rows = (batch == g).nonzero().squeeze(1)[torch.as_tensor(kept, dtype=torch.long, device=device)]
            where = pos[rows]
            visible = (~hidden[rows]).float() if hidden is not None else torch.ones(len(rows), device=device)
            data = {}
            if codes is not None:
                data["labels"] = codes[rows]
                if settings.split_by_space != "none":
                    # The edges among the kept cells, in sequence order.
                    local.fill_(-1)
                    local[rows] = torch.arange(len(rows), device=device)
                    ends = local[edges]
                    data["edge_index"] = ends[:, (ends >= 0).all(0)]
            hierarchies = [self.hierarchy_cache.get(where, **data, **settings.build_kwargs(False)).to(device)]
            if settings.shift:
                hierarchies.append(self.hierarchy_cache.get(where, **data, **settings.build_kwargs(True)).to(device))
            graphs.append(GraphLayout(seq_len - len(kept), where, visible, hierarchies, list(kept)))
        coarse = None
        if self.attention_bias is not None and settings.coarse_bias == "shared":
            coarse = [self.attention_bias] * settings.n_levels
        elif self.attention_bias is not None and settings.coarse_bias == "own":
            coarse = list(self.hier_coarse_bias)
        context = HierarchicalContext(
            graphs,
            bias=self.attention_bias,
            coarse_biases=coarse,
            generator=self.hierarchy_generator,
            training=self.training,
        )
        context.record = self._hierarchy_record
        return context

    @contextmanager
    def recording_hierarchy(self):
        """Record every hierarchical pass made inside: per layer and graph, the hierarchy, the cells' positions and each
        level's attention (dense ``[H, n_l(+1), n_l + 1]``, or the routed rows), detached on the CPU, plus lambda and mu.

        Yields the list the layers append to (``HierarchicalEncoderLayer._export``). The flag is restored on exit.
        """
        if self.hierarchy is None:
            raise ValueError("recording_hierarchy: this transformer is not hierarchical.")
        record, saved = [], self._hierarchy_record
        self._hierarchy_record = record
        try:
            yield record
        finally:
            self._hierarchy_record = saved

    def _query_encoding(self, node_encoding, batched_data, index_nodes, padded_emb) -> torch.Tensor:
        """``[S, B, E]``: the node encodings the tokens were given, alone, laid out as the tokens (zeros without any).

        The position half of ``context_only``'s queries. It is the very tensor that went into the tokens of this
        pass (the encodings draw their random signs and rotations once), taken through the selection ``pad_batch``
        made -- ``pad_like`` with this call's ``index_nodes``, never a second padding.
        """
        seq_len, n_graphs, _ = padded_emb.shape
        if node_encoding is None:
            return padded_emb.new_zeros(padded_emb.shape)
        return pad_like(node_encoding, batched_data.batch, index_nodes, seq_len).transpose(0, 1).to(padded_emb.dtype)

    def _without_hidden_keys(self, mask: torch.Tensor, batched_data, index_nodes, seq_len: int) -> torch.Tensor:
        """``mask`` ``[B * heads, S + 1, S + 1]`` (True: blocked) with every key that is a cell hidden whole blocked too.

        A hidden cell stays a query: its row loses only the hidden keys, and CLS (the last slot, never hidden) stays open,
        so no row is ever empty. The cells are laid out like the tokens (``pad_like`` with this call's ``index_nodes``).
        """
        hidden = self.wholly_hidden(batched_data)
        keys = pad_like(hidden[:, None].float(), batched_data.batch, index_nodes, seq_len)[..., 0] > 0  # [B, S]
        keys = torch.nn.functional.pad(keys, (0, 1))  # CLS open
        n_graphs = keys.shape[0]
        blocked = mask.reshape(n_graphs, self.n_heads, seq_len + 1, seq_len + 1) | keys[:, None, None, :]
        return blocked.reshape(mask.shape)

    @contextmanager
    def recording_attention(self, reference: bool = False):
        """Record every pass made inside: each layer's attention map and the layout of its tokens.

        Yields a list. Each pass through :meth:`common_step_local_to_global` and :meth:`forward`
        appends ``{"index_nodes", "src_padding_mask", "maps", "reference", "token_cosine"}``: ``maps`` is
        ``[layers, B * heads, S + 1, S + 1]``, every layer's map from its explicit path, detached;
        ``open`` (dense masks only) is ``[B, S + 1, S + 1]``, True where a query may attend to a key;
        with ``reference``, ``"reference"`` holds the same for the attention a backend estimates
        (the Performer's exact softmax on the same queries, keys and biases), or None where a
        backend estimates nothing. ``token_cosine`` is ``[layers, B]``: per layer and sequence, the
        mean pairwise cosine of the cell tokens entering attention, before any centring -- near 1
        when the tokens are nearly identical, which leaves content-based attention uniform. What the
        diagnostics read (``evaluation.attention_diagnostics``).

        The passes inside take each backend's explicit path, as ``evaluate()`` does: dense
        attention takes it anyway, so its numbers are unchanged; a kernel backend's explicit path
        computes the same function as its factorised one, agreeing to rounding (~1e-5). The
        reference is computed beside the attention and feeds nothing. Flags are restored on exit,
        even if a pass raises.
        """
        record, saved = [], self.return_attention_maps
        estimators = [
            layer.self_attn for layer in self.transformer_encoder.layers if hasattr(layer.self_attn, "record_reference")
        ]
        self._attention_record, self.return_attention_maps = record, True
        for attention in estimators:
            attention.record_reference = bool(reference)
        for layer in self.transformer_encoder.layers:
            layer.record_tokens = True
        try:
            yield record
        finally:
            self._attention_record, self.return_attention_maps = None, saved
            for attention in estimators:
                attention.record_reference, attention.reference_weights = False, None
            for layer in self.transformer_encoder.layers:
                layer.record_tokens, layer.token_cosine = False, None

    @contextmanager
    def hooking_attention(self):
        """Inside, every pass keeps each layer's attention map and hooks it for its gradient.

        What :meth:`evaluate` does for its unmasked pass, for the passes that do not ask (``register_hook``
        None, e.g. a masked pass through ``_common_step``), so a relevance can be traced from a target computed
        on the masked pass (``SelfAttentionRelevance.generate_relevance(..., target=...)``). The flag is
        restored on exit, even if a pass raises.
        """
        saved, self._hook_attention = self._hook_attention, True
        try:
            yield
        finally:
            self._hook_attention = saved

    def forward(
        self, padded_h_node, src_padding_mask, mask=None, register_hook: bool | None = None, rotary=None, queries=None
    ):
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
            rotary: (cos, sin), each [B * heads, N_b_max + 1, H_d / heads]
                The rotary encoding's rotation from `common_step_local_to_global` (GlobalInput.rotary).
                Required when the rotary encoding is enabled, and only then.
            queries: [N_b_max x B X H_d] or None
                `GlobalInput.queries`: the node encodings, laid out as the tokens. Required under
                ``context_only`` (the queries are one learned token plus these, never a cell's own token),
                and only then.
            register_hook: bool or None
                True: keep every layer's attention map, return them stacked, and hook them for their
                gradient (what relevance needs; `evaluate` passes it). False: none of that. None
                (training steps): return the stacked maps only when the loss reads them
                (`return_attention_maps`), and hook nothing -- unless inside `hooking_attention`, which
                makes None mean True.
        """
        if register_hook is None and self._hook_attention:
            register_hook = True
        if (rotary is None) != (self.rotary_encoding is None):
            raise ValueError(
                "the rotary encoding is enabled, so forward() needs the rotation common_step_local_to_global "
                "returned (GlobalInput.rotary); without it every layer would attend without positions."
                if rotary is None
                else "forward() got a rotation, but this transformer has no rotary encoding enabled."
            )
        if rotary is not None:
            expected = (padded_h_node.size(1) * self.n_heads, padded_h_node.size(0) + 1, self.head_dim)
            if tuple(rotary[0].shape) != expected:
                raise ValueError(
                    f"the rotation has shape {tuple(rotary[0].shape)}, but these tokens need {expected} "
                    "(graphs x heads, tokens + CLS, head width): it was computed for another batch."
                )

        if (queries is None) != (self.context_only == "none"):
            raise ValueError(
                "context_only is on, so forward() needs the queries common_step_local_to_global returned "
                "(GlobalInput.queries); without them there is no position to read from."
                if queries is None
                else "forward() got queries, but context_only is off."
            )
        if queries is not None and tuple(queries.shape) != tuple(padded_h_node.shape):
            raise ValueError(
                f"the queries have shape {tuple(queries.shape)}, but these tokens have {tuple(padded_h_node.shape)}: "
                "they were computed for another batch."
            )

        relevance = bool(register_hook)
        keep_maps = relevance or (register_hook is None and self.return_attention_maps)
        for encoder in self.transformer_encoder.layers:
            encoder.register_hook = relevance
            encoder.keep_maps = keep_maps

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
        # A StructuredMask (blocked pairs) is not a tensor the encoder could merge: it reaches the
        # layers as their per-batch context, and the encoder gets no dense mask. So does the hierarchy's context.
        hierarchical = isinstance(mask, HierarchicalContext)
        if hierarchical != (self.hierarchy is not None):
            raise ValueError(
                "this transformer is hierarchical; forward() needs the HierarchicalContext common_step_local_to_global returned."
                if self.hierarchy is not None
                else "forward() got a HierarchicalContext, but this transformer is not hierarchical."
            )
        structured = isinstance(mask, StructuredMask)
        if structured != (self.attention_spec.mask in ("pairs", "pattern")):
            raise ValueError(
                f"attention.kind {self.attention_kind!r} takes its mask as "
                + ("a dense tensor" if self.attention_spec.mask == "dense" else "a StructuredMask")
                + ", the form common_step_local_to_global builds for it."
            )
        if mask is not None and not structured and not hierarchical and mask.is_floating_point():
            # A float mask (an attention bias merged in) needs a float key-padding mask, or torch
            # warns on every layer that mixing the two types is deprecated. The boolean one is
            # still what this method returns, for the callers that drop padding by it.
            key_padding_mask = torch.zeros_like(src_padding_mask, dtype=mask.dtype).masked_fill(
                src_padding_mask, float("-inf")
            )
        # nn.TransformerEncoder passes its layers nothing but the masks, so the rotation reaches them
        # as an attribute, set for this call only: cleared even if a layer raises, so a later call
        # can never attend with a stale batch's positions.
        for encoder in self.transformer_encoder.layers:
            encoder.rotary = rotary
            encoder.attention_context = mask if structured else None
            if hierarchical:
                encoder.hier_context = mask
        try:
            if self.context_only == "none":
                transformer_out = self.transformer_encoder(
                    padded_h_node,
                    src_key_padding_mask=key_padding_mask,
                    mask=None if structured or hierarchical else mask,
                )  # (S, B, h_d)
            else:
                transformer_out = self._read_context(queries, padded_h_node, key_padding_mask, mask)  # (S, B, h_d)
        finally:
            for encoder in self.transformer_encoder.layers:
                encoder.rotary = None
                encoder.attention_context = None
                if hierarchical:
                    encoder.hier_context = None

        attn_matrices = []
        for encoder in self.transformer_encoder.layers:
            if keep_maps:
                attn_matrices.append(encoder.get_attn_output_weights())
            # The maps stay readable on the layers (`get_attn_output_weights`) until the next call.
            encoder.register_hook = False
            encoder.keep_maps = False

        if len(attn_matrices) > 0 and all(m is not None for m in attn_matrices):
            final_attn = torch.stack(attn_matrices)
        else:
            final_attn = None

        if self._attention_record:
            entry = self._attention_record[-1]
            entry["maps"] = final_attn.detach() if final_attn is not None else None
            references = [
                getattr(layer.self_attn, "reference_weights", None) for layer in self.transformer_encoder.layers
            ]
            entry["reference"] = torch.stack(references) if all(r is not None for r in references) else None
            cosines = [layer.token_cosine for layer in self.transformer_encoder.layers]
            entry["token_cosine"] = torch.stack(cosines) if all(c is not None for c in cosines) else None

        return transformer_out, src_padding_mask, final_attn

    def _read_context(self, queries, memory, key_padding_mask, mask) -> torch.Tensor:
        """The query stream (``context_only: query``): ``[S + 1, B, E]``, position-only queries read the cells' input tokens.

        Every cell starts as ``context_token + its node encoding`` (CLS as the CLS token), normalised like the tokens,
        and in each layer reads ``memory`` -- the input tokens, CLS included, the same at every layer -- through that
        layer's weights (:meth:`CustomTransformerEncoderLayer.read_context`). The attention mask is the one the
        tokens would have had: a cell never reads its own token, and under the long-range mask not the cells its
        local component already mixed in. The stream holds nothing of its cell's token, and the memory is not
        updated by it, so a cell's state is a function of the other cells' input tokens and its own position:
        what it says about the cell is what its context says. (``attention.center_qk`` subtracts the mean over
        all cells' tokens, the cell's own included at weight 1 / S, from queries and keys.)
        """
        cls = self.cls_embedding.expand(1, memory.size(1), -1)
        state = self.norm_input(torch.cat([self.context_token.view(1, 1, -1) + queries, cls], dim=0))
        for layer in self.transformer_encoder.layers:
            state = layer.read_context(state, memory, attn_mask=mask, key_padding_mask=key_padding_mask)
        return self.transformer_encoder.norm(state)

    def evaluate(self, batched_data, embedding):
        """Evaluates transformer encoder on a batch of data without masking and registering hook.

        Args:
            batched_data (_type_): _description_

        Returns
        -------
            I: torch.Tensor [N+1, N+1]
                Self-attention relevance matrix with CLS token.
        """
        if self.pooling is not None:
            raise NotImplementedError(
                "evaluate() reads the relevance of cell tokens, which a pooled model (model.global_component."
                "parameters.pooling) does not have: its transformer runs on patches. Use the masked-pass "
                "evaluation (_common_step) instead."
            )
        # evaluation on single graph
        batched_data.batch = torch.Tensor(len(batched_data.obs_names) * [0])
        given = self.common_step_local_to_global(batched_data, embedding, eval_step=True)
        transformer_in, src_padding_mask, pad_index_nodes, attn_mask, rotary = given

        transformer_out, src_padding_mask, _ = self.forward(
            transformer_in,
            src_padding_mask,
            attn_mask,
            register_hook=True,
            rotary=rotary,
            **self._query_kwargs(given),
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
            f"rotary_encoding: {sorted(self.rotary_encoding.encoders) if self.rotary_encoding else []}, \n"
            f"attention: {self.attention_kind}, head_dim: {self.head_dim}, \n"
            f"context_only: {self.context_only}, \n"
        )
        return summary
