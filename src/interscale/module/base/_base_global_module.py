from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from sklearn.decomposition import NMF, PCA

from interscale.tl.masking import MASK_VALUE

from ._base_module import BaseModule
from ._step_output import StepOutput, ViewOutput, gather_cls, gather_tokens

#: How a hidden cell's precomputed embedding is replaced (``precomputed_mask_fill``).
PRECOMPUTED_MASK_FILLS = ("learned", "minus_one")
#: What the transformer gets for a cell hidden whole (``remask``): its local token, or one learned token.
REMASK_KINDS = ("none", "hidden")


class GlobalModule(BaseModule):
    def __init__(
        self,
        precomputed_mask_fill: str = "learned",
        programme_masking: bool = False,
        pooling=None,
        remask: str = "none",
        **base_module_kwargs,
    ):
        """``precomputed_mask_fill``: what a hidden cell's ``Precomputed`` embedding is replaced by.

        A precomputed embedding was made from the cell's own expression, which is the
        reconstruction target, so a hidden cell must not hand it to the transformer.
        ``"learned"``: one trained vector shared by every hidden cell. ``"minus_one"``: every entry
        set to ``MASK_VALUE``, as masked expression is. See :meth:`hide_embeddings`.

        ``programme_masking``: a gene set (``dataset.mask_genes``) is hidden under ``mask_strategy
        gene``. A ``Precomputed`` embedding is then allowed, since one computed without those genes
        (``tl.check_embedding_genes`` verifies it when the graphs are built) holds none of the hidden
        values; no cell is replaced by the fill, as every cell keeps its visible genes.

        ``pooling``: a :class:`~interscale.module.global_modules.pooling.PatchPooling`, or None. With it the
        transformer runs on patches of cells (:meth:`step`); everything outside it still sees one row per cell.

        ``remask``: ``"hidden"`` hands the transformer one learned token for every cell hidden whole instead
        of the local component's token, which mixed in the cell's visible neighbours (:meth:`remask_tokens`).
        Only behind a local component: a model without one already tokenises a hidden cell from its masked
        input alone.
        """
        super().__init__(**base_module_kwargs)
        self.pooling = pooling
        if remask not in REMASK_KINDS:
            raise ValueError(f"remask must be one of {REMASK_KINDS}, got {remask!r}.")
        self.remask = remask
        if remask != "none":
            if self.type_gex_embedding is not None:
                raise ValueError(
                    "remask replaces the local component's token of a hidden cell, but this global module "
                    f"tokenises the expression itself (type_gex_embedding {self.type_gex_embedding!r}): a "
                    "GlobalModel's hidden tokens hold nothing of the cell already. Use it in a CombinedModel."
                )
            if pooling is not None:
                raise ValueError("remask: a pooled patch token already leaves out its hidden cells.")
            # From a private generator, so building it draws nothing from the global one: every other weight
            # is initialised as in the model without it.
            gen = torch.Generator().manual_seed(0)
            self.remask_token = torch.nn.Parameter(torch.randn(self.n_embed, generator=gen))

        self.registered_local_component = False
        self.registered_global_component = True

        if self.type_gex_embedding == "PCA":
            self.pca = PCA(n_components=self.n_embed)
            # A fitted sklearn estimator is not part of `module.state_dict()`, and
            # `BaseModel.save` persists nothing else -- so a GlobalModel reloaded for inference
            # would arrive with an UNFITTED pca and silently refit it on the first evaluation
            # batch. The transformer's weights were learned on the basis fitted to the first
            # TRAINING batch; a basis refitted elsewhere differs by rotation and by component
            # sign, so the reloaded model would decode a different space than it was trained on
            # and produce attention that means nothing. These buffers carry the fit through the
            # checkpoint.
            #
            # Registered unconditionally for the PCA branch (not lazily on first fit) because a
            # buffer that does not exist at construction time cannot receive a value from
            # `load_state_dict`, which is exactly when it is needed. Older checkpoints simply
            # have no entry for them; `BaseModel.load` uses strict=False, so they stay zeroed
            # and `pca_fitted_` stays False, reproducing the previous refit-on-load behaviour
            # rather than failing.
            self.register_buffer("pca_mean_", torch.zeros(self.n_input))
            self.register_buffer("pca_components_", torch.zeros(self.n_embed, self.n_input))
            self.register_buffer("pca_fitted_", torch.zeros(1, dtype=torch.bool))
        elif self.type_gex_embedding == "NMF":
            self.nmf = NMF(n_components=self.n_embed, init="random", random_state=0)
        elif self.type_gex_embedding == "Precomputed":
            if precomputed_mask_fill not in PRECOMPUTED_MASK_FILLS:
                raise ValueError(
                    f"precomputed_mask_fill must be one of {PRECOMPUTED_MASK_FILLS}, got {precomputed_mask_fill!r}."
                )
            if self.masked_nodes and self.mask_strategy == "gene" and not programme_masking:
                raise ValueError(
                    "Precomputed embeddings cannot hide single genes: a cell's embedding encodes all of its "
                    "expression. Use mask_strategy node (or spatial blocks), or hide a gene set "
                    "(dataset.mask_genes) with an embedding computed from the other genes."
                )
            self.precomputed_mask_fill = precomputed_mask_fill
            if precomputed_mask_fill == "learned":
                # From a private generator, so building it draws nothing from the global one: the
                # rest of the model is initialised identically under either fill.
                gen = torch.Generator().manual_seed(0)
                self.mask_embedding = torch.nn.Parameter(torch.randn(self.n_embed, generator=gen))
        elif self.type_gex_embedding == "Linear":
            # A learned map from the (masked) expression to the tokens, trained with the
            # transformer -- the only input of a global-only model that is not fixed in advance.
            # The transformer's own input LayerNorm normalises its output.
            self.input_embedding = torch.nn.Linear(self.n_input, self.n_embed)
        elif self.type_gex_embedding is None:
            # No GEX embedding needed when using CombinedModule (local module provides embeddings)
            pass
        else:
            raise ValueError(f"Invalid embedding type: {self.type_gex_embedding}")

    @abstractmethod
    def forward(self, embeddings: torch.Tensor):
        """
        Input:
            embeddings: torch.Tensor
                Size: [N, E], either output of local component or user-provided embeddings.
        """

    def gex_embedding(self, batch, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """``[N, n_embed]`` tokens of a model without a local component, from ``batch`` as given.

        ``Linear`` maps ``batch.x`` through the learned layer, on its device and in the graph;
        otherwise precomputed ``batch.embeddings``, or the PCA/NMF projection of ``batch.x``. The
        caller passes the masked batch in training and the clean one at inference. ``hidden``
        (``[N]`` boolean) marks the cells whose precomputed embeddings must not reach the model;
        the other inputs are computed from the masked ``batch.x`` and need no such marker.
        """
        if self.type_gex_embedding == "Linear":
            return self.input_embedding(batch.x.float())
        if hasattr(batch, "embeddings"):
            embedding = batch.embeddings
        else:
            embedding = self.create_gex_embedding(batch.x.cpu().numpy(), type=self.type_gex_embedding)
        embedding = torch.tensor(embedding, dtype=torch.float32, device=batch.x.device)
        if self.type_gex_embedding == "Precomputed" and hidden is not None:
            embedding = self.hide_embeddings(embedding, hidden)
        return embedding

    def hide_embeddings(self, embedding: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """``embedding`` with the rows of ``hidden`` cells replaced by the mask fill.

        Masking corrupts ``batch.x`` only, so without this a hidden cell would hand the transformer
        its precomputed embedding -- a compressed copy of the expression it is scored on. The fill
        goes through the transformer's input LayerNorm like any token. Note that a constant vector
        (``"minus_one"``) comes out of it as the LayerNorm's bias: the same token whatever the
        constant, and different from any real (non-constant) embedding.
        """
        if self.precomputed_mask_fill == "learned":
            fill = self.mask_embedding.to(embedding.dtype)
        else:
            fill = torch.full((self.n_embed,), MASK_VALUE, dtype=embedding.dtype, device=embedding.device)
        return torch.where(hidden.bool()[:, None], fill.expand_as(embedding), embedding)

    def create_gex_embedding(self, embeddings: torch.Tensor, type: Literal["PCA", "NMF", "scvi"]):
        """Generate embeddings for GEX if no local component is used.

        Parameters
        ----------
        batch.x: torch.Tensor
            Size: [N, F]
        type: Literal["PCA", "scvi"]
            Type of embedding to generate.

        Returns
        -------
        gex_embedding: torch.Tensor
            Size: [N, E]
        """
        if type == "PCA":
            # Fit PCA only once (on the first batch that arrives with no fit available), then
            # project every later batch through the stored basis. Two sources of a fit, in
            # order: the buffers restored from a checkpoint, then this process's own first
            # batch. Checking the buffers FIRST is what makes a reloaded model reproduce the
            # basis it was trained on instead of refitting on evaluation data.
            # `_common_step` hands this a numpy array but `GlobalModel.get_model_output` hands
            # it `batch.x` straight off the batch, which is a (possibly CUDA) tensor.
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            if not bool(self.pca_fitted_):
                self._fit_pca(embeddings)
            return self._pca_transform(embeddings)
        elif type == "NMF":
            if not hasattr(self.nmf, "components_"):
                return self.nmf.fit_transform(embeddings)
            else:
                return self.nmf.transform(embeddings)
        else:
            raise ValueError(f"Invalid embedding type: {type}")

    def _fit_pca(self, embeddings, deterministic: bool = False) -> None:
        """Fit the PCA basis on ``[n, n_input]`` ``embeddings`` and store it in the buffers.

        ``deterministic`` fixes the solver's random state: sklearn's ``auto`` solver is randomized
        above 500 rows and, with no state set, two fits of one matrix differ.
        """
        if deterministic:
            self.pca.set_params(random_state=0)
        self.pca.fit(embeddings)
        self.pca_mean_.copy_(torch.as_tensor(self.pca.mean_, dtype=self.pca_mean_.dtype))
        self.pca_components_.copy_(torch.as_tensor(self.pca.components_, dtype=self.pca_components_.dtype))
        self.pca_fitted_.fill_(True)

    def fit_pca_on_training(self, graphs) -> int:
        """Fit the PCA basis once on the masked inputs of the training ``graphs``; the number of rows used.

        ``model.global_component.parameters.pca_fit: train``. Each graph's input is corrupted as in
        training (:meth:`_common_step_masking`, from the graph's current masks), so a hidden gene is
        a constant column and adds no variance, and no hidden value enters the basis. A no-op for
        a module without PCA tokens or whose basis is already fitted (a restored checkpoint).
        """
        from torch_geometric.data import Batch

        if self.type_gex_embedding != "PCA" or bool(self.pca_fitted_):
            return 0
        rows = []
        for graph in graphs:
            masked, _, _ = self._common_step_masking(Batch.from_data_list([graph]))
            rows.append(masked.x.detach().cpu().numpy())
        x = np.concatenate(rows, axis=0)
        self._fit_pca(x, deterministic=True)
        return len(x)

    def wholly_hidden(self, batch_masked) -> torch.Tensor:
        """``[N]`` bool: the cells whose every entry is hidden (cell masking and discs; under gene masking, a fully hidden row)."""
        none = torch.zeros(batch_masked.x.shape[0], dtype=torch.bool, device=batch_masked.x.device)
        if not self.masked_nodes:
            return none
        if self.mask_strategy == "gene":
            gene_mask = getattr(batch_masked, "gene_mask", None)
            return none if gene_mask is None else gene_mask.bool().all(dim=1)
        return batch_masked.mask.bool()

    def remask_tokens(self, batch_masked, emb: torch.Tensor) -> torch.Tensor:
        """``emb`` ``[N, E]`` with the rows of the cells hidden whole replaced by the learned ``remask_token``.

        Masking corrupts the input once, before the local component, which then hands a hidden cell a token
        made of its visible neighbours: the transformer would read the cell's neighbourhood in the cell's own
        residual stream, and its head could copy the local one. Re-masked, a hidden cell enters the
        transformer as the same token everywhere (plus its position, added after), so its global embedding
        is what attention gathers from other cells -- under the long-range mask, from beyond the local reach.
        Visible cells keep their tokens. Unchanged when ``remask`` is ``"none"``.
        """
        if self.remask == "none":
            return emb
        hidden = self.wholly_hidden(batch_masked)
        return torch.where(hidden[:, None], self.remask_token.to(emb.dtype).expand_as(emb), emb)

    def step(self, batched_data, emb: torch.Tensor):
        """The transformer on the cell tokens ``emb`` ``[N, E]`` of a masked batch: ``(global_embedding, src_padding_mask, index_nodes, attention_mask, attn)``.

        Without pooling this is ``common_step_local_to_global`` followed by ``forward``. With it the cell
        tokens are first pooled into patch tokens, the transformer runs on the patches, and each cell gets
        its patch's output, laid out as the cell-level output is (``pooling.cell_layout``) -- so the
        decoders, the loss and every consumer read one row per cell either way. ``index_nodes`` then lists
        every cell of each graph, and ``attn`` holds the patches' attention maps. Under ``remask`` the tokens of
        the cells hidden whole are replaced first (:meth:`remask_tokens`).
        """
        emb = self.remask_tokens(batched_data, emb)
        if self.pooling is None:
            padded_emb, src_padding_mask, pad_index_nodes, attention_mask, rotary = self.common_step_local_to_global(
                batched_data, emb
            )
            assert not torch.any(torch.isnan(padded_emb)), "padded_emb contains NaN values"
            global_embedding, src_padding_mask, attn = self.forward(
                padded_emb, src_padding_mask, attention_mask, rotary=rotary
            )
            assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN values"
            return global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn

        from interscale.module.global_modules.pooling import cell_layout

        plan = self.pooling.plan(batched_data, max_patches=self.max_seq_len)
        tokens, visible = self.pooling.pool(emb, self.wholly_hidden(batched_data), plan)
        padded_emb, patch_padding, patch_index_nodes, attention_mask, rotary = self.common_step_local_to_global(
            self.pooling.batch(plan, visible), tokens
        )
        assert not torch.any(torch.isnan(padded_emb)), "padded_emb contains NaN values"
        out, patch_padding, attn = self.forward(padded_emb, patch_padding, attention_mask, rotary=rotary)
        assert not torch.any(torch.isnan(out)), "global_embedding contains NaN values"
        rows = gather_tokens(out, patch_padding)  # [P, E], patch 0 first: every patch was kept (max_seq_len checked)
        global_embedding, cell_padding, index_nodes = cell_layout(
            rows[plan.patch], out[-1], batched_data.batch.long(), int(batched_data.batch.max()) + 1
        )
        return global_embedding, cell_padding, index_nodes, attention_mask, attn

    def _pca_transform(self, embeddings):
        """Project onto the stored PCA basis: ``(X - mean_) @ components_.T``.

        Written out rather than delegated to ``self.pca.transform`` so that the projection
        depends only on the two buffers, which are the only part of the fit that survives a
        checkpoint round trip. ``sklearn``'s own ``transform`` would additionally require the
        estimator's private fitted attributes to be present, which after a reload they are not.
        Equivalent to it for ``whiten=False``, which is the default this module constructs.
        """
        mean = self.pca_mean_.detach().cpu().numpy()
        components = self.pca_components_.detach().cpu().numpy()
        return (np.asarray(embeddings, dtype=np.float64) - mean) @ components.T

    def _process_batch_for_metrics(self, batch, prediction_task, prediction_level, pad_index_nodes):
        """Gather the ground truth for every cell the transformer actually produced, in its order.

        ``pad_batch`` emits one row per kept cell, per graph, in the order named by
        ``pad_index_nodes`` -- which is a per-graph, graph-local index. This turns that into a
        single batch-global index so predictions, truth and masks can all be lined up by it.

        Parameters
        ----------
        batch
            Input batch
        prediction_task: str
            Type of prediction task ('classification' or 'regression')
        prediction_level: str
            Level of prediction ('node' or 'graph')
        pad_index_nodes: List[List[int]]
            Per-graph indices of the cells the padding kept: ``[B, S]``, or ``[B, N]`` when the
            graph is smaller than max_seq_len.

        Returns
        -------
        y_true: torch.Tensor [N_kept, C] (classification) or [N_kept, F] (regression)
            Ground truth for the kept cells, in transformer-output order.
        padded_node_idx: torch.Tensor [N_kept]
            The batch-global node index of each kept row. Index anything that lives in the
            batch's own node order (a local-decoder output, ``batch.mask``, ``batch.gene_mask``)
            with this to bring it into the same order.
        entry_mask: torch.Tensor [N_kept, F] | None
            Which entries the *loss* is scored on, in the same order. Under gene masking this is
            the batch's ``gene_mask``; under cell masking it is the per-cell mask broadcast over
            all F genes, so a masked cell is all-True and an unmasked cell all-False -- which is
            what lets ``masked_loss`` recognise a whole-row mask and keep the row structure that
            row-normalising criteria need. ``None`` for classification, where the target is a
            label per cell and there are no entries to mask.

        Notes
        -----
        This replaces an earlier version that returned ``adjusted_mask_idx`` -- the positions of
        the masked cells within the kept rows -- which every caller then used to subset y_pred and
        y_true down to masked cells only. Metrics are now computed over ALL kept cells and only
        the loss is restricted, so the subsetting is gone and with it the index arithmetic that
        matched ``mask_idx`` against ``pad_indices`` graph by graph.
        """
        assert prediction_level == "node", "Node specific retrieval only necessary for node-level prediction."

        nr_batches = int(batch.batch[-1]) + 1
        device = batch.x.device

        batch_sizes = torch.tensor([batch.batch.eq(i).sum().item() for i in range(nr_batches)], device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), batch_sizes.cumsum(0)[:-1]])

        # Graph-local kept indices -> batch-global node indices, concatenated in output order.
        padded_node_idx = torch.cat(
            [
                torch.tensor(pad_index_nodes[i], device=device, dtype=torch.long) + batch_starts[i]
                for i in range(nr_batches)
            ]
        )

        if "classification" in prediction_task:
            y_true = batch.y[padded_node_idx]
            entry_mask = None
        elif "regression" in prediction_task:
            y_true = batch.x[padded_node_idx]
            use_gene_mask = self.mask_strategy == "gene"
            gene_mask = getattr(batch, "gene_mask", None) if use_gene_mask else None
            if gene_mask is not None:
                entry_mask = gene_mask[padded_node_idx].bool()
            else:
                # Whole-cell masking expressed at entry level, so one mask covers both strategies.
                entry_mask = batch.mask[padded_node_idx].bool().unsqueeze(1).expand_as(y_true)
        else:
            raise Exception("Choose a valid prediction task (classification or regression).")

        if entry_mask is not None:
            assert entry_mask.shape == y_true.shape, (
                f"Mismatch: entry_mask.shape: {tuple(entry_mask.shape)}, y_true.shape: {tuple(y_true.shape)}"
            )

        return y_true, padded_node_idx, entry_mask

    # def _process_batch_for_metrics(self, batch, prediction_task, prediction_level, pad_index_nodes, mask_idx_tensor):
    #     """Process batch to extract y_true and adjusted_mask_idx for metrics calculation.

    #     mask_idx = torch.tensor([0, 2, 3, 7, 8])
    #     pad_index_nodes = [[0, 1, 2, 3], [0, 1], [0, 1, 2, 3]]

    #     Parameters
    #     ----------
    #     batch
    #         Input batch
    #     prediction_task: str
    #         Type of prediction task ('classification' or 'regression')
    #     prediction_level: str
    #         Level of prediction ('node' or 'graph')
    #     pad_index_nodes: List[List[int]]
    #         List of padded node indices: [B, S] or [B,N] if number of nodes in graph are smaller than max_seq_len (S)
    #     mask_idx_tensor: torch.Tensor
    #         Indices of masked nodes of shape [N_masked_nodes] with range [0, N_nodes-1]

    #     Returns
    #     -------
    #     y_true: torch.Tensor [N_included_nodes, C] (classification) or [N_included_nodes, F] (regression)
    #         Ground truth values
    #     adjusted_mask_idx: torch.Tensor [N_masked nodes]
    #         Adjusted indices for masked nodes
    #     """
    #     assert prediction_level == "node", "Node specific retrieval only necessary for node-level prediction."

    #     y_true = []
    #     adjusted_mask_idx = []  # Track new positions of masked nodes
    #     current_offset = 0
    #     start = 0
    #     mask_j = 0
    #     nr_batches = batch.batch[-1] + 1

    #     for i in range(nr_batches):
    #         mask = batch.batch.eq(i)
    #         pad_indices = torch.tensor(pad_index_nodes[i], device=batch.x.device) + start
    #         end = start + torch.sum(mask)

    #         # can not assume that pad_indices is a subset of mask_idx
    #         #TODO: use stack and pop instead
    #         for j, mask_idx in enumerate(mask_idx_tensor[mask_j:]):
    #             if mask_idx > end:
    #                 break
    #             if mask_idx in pad_indices:
    #                 new_idx = torch.where(pad_indices == mask_idx)[0].item()
    #                 adjusted_mask_idx.append(new_idx + current_offset)

    #         current_offset += len(pad_indices)
    #         start = end
    #         mask_j = j

    #         # only return y_true for included nodes
    #         if 'classification' in prediction_task:
    #             y_true += batch.y[mask][pad_index_nodes[i]].clone().detach()
    #         elif 'regression' in prediction_task:
    #             y_true += batch.x[mask][pad_index_nodes[i]].clone().detach()
    #         else:
    #             raise Exception('Choose a valid prediction tasks (graph or node).')
    #         assert len(mask) >= len(pad_indices) >= len(adjusted_mask_idx), "mask, pad_indices, adjusted_mask_idx are not consistent"

    #     y_true = torch.stack(y_true)
    #     adjusted_mask_idx = torch.tensor(adjusted_mask_idx, device=y_true.device)

    #     assert max(adjusted_mask_idx) < len(y_true), f"Mismatch: max(adjusted_mask_idx): {max(adjusted_mask_idx)}, len(y_true): {len(y_true)}"
    #     if nr_batches > 1:
    #         assert  max(adjusted_mask_idx) > len(pad_index_nodes[0]), f"No masked node included from all batches: first batch has {len(pad_index_nodes[0])} nodes, but {max(adjusted_mask_idx)} nodes were included"

    #     assert torch.equal(y_true_new, y_true), "y_true_new and y_true are not consistent"
    #     assert torch.equal(adjusted_mask_idx_new, adjusted_mask_idx), "adjusted_mask_idx_new and adjusted_mask_idx are not consistent"

    #     return y_true, adjusted_mask_idx

    def predict(self, global_embedding, src_padding_mask, prediction_level):
        """Predict with the decoder.

        Parameters
        ----------
        global_embedding: torch.Tensor
            Size: [N, E]
        prediction_level: Literal["node", "graph"]
            Level of prediction
        """
        ## Graph-level prediction: get cls_token from last position
        if "graph" in prediction_level:
            return self.decoder(gather_cls(global_embedding))
        ## Node-level prediction: remove cls_token from last position
        elif "node" in prediction_level:
            return self.decoder(gather_tokens(global_embedding, src_padding_mask))
        else:
            raise Exception("Choose a valid prediction tasks (graph or node).")

    def _common_step(self, batch, prediction_task: str, prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.

        Returns
        -------
        StepOutput
            ``y_pred``/``y_true`` are ``[N, C]`` (classification) or ``[N, F]`` (regression) over
            every cell the transformer produced, not just the masked ones -- see
            ``_process_batch_for_metrics``. ``entry_mask`` marks the entries the LOSS is scored
            on. The single view carries the padded ``global_embedding`` ``[S, B, E]``, its
            ``src_padding_mask``, the ``padded_node_idx`` that maps tokens back to batch node
            order, and the stacked per-layer attention.
        """
        # Mask nodes  - before GEX embedding because otherwise embedding contains information about masked nodes
        batch_masked, _, _ = self._common_step_masking(batch)
        # The cells whose own values the model must not see; a precomputed embedding of theirs is
        # replaced too (`hide_embeddings`). Under gene masking no cell is hidden whole: a
        # precomputed embedding is then one made without the hidden genes (programme masking), and
        # stays as it is.
        hidden = batch_masked.mask.bool() if self.masked_nodes and self.mask_strategy != "gene" else None
        embedding = self.gex_embedding(batch_masked, hidden=hidden)
        assert embedding.shape == (batch_masked.x.shape[0], self.n_embed), (
            f"Mismatch: embedding.shape: {embedding.shape}, batch_masked.x.shape: {batch_masked.x.shape}"
        )
        assert not torch.any(torch.isnan(embedding)), "embedding contains NaN values"

        global_embedding, src_padding_mask, pad_index_nodes, _, attn_matrix = self.step(batch_masked, embedding)

        y_pred = self.predict(global_embedding, src_padding_mask, prediction_level)

        if prediction_task == "classification" and prediction_level == "graph":
            y_true = batch.y[batch.ptr[:-1]]
            entry_mask = None
            padded_node_idx = None
        else:
            y_true, padded_node_idx, entry_mask = self._process_batch_for_metrics(
                batch, prediction_task, prediction_level, pad_index_nodes
            )
            if entry_mask is None:
                # Node-level classification: the masked cells ARE the supervision targets, so
                # both loss and metrics stay restricted to them. Only reconstruction moved to
                # scoring every cell.
                keep = batch.mask[padded_node_idx].bool()
                y_pred, y_true = y_pred[keep], y_true[keep]

        assert len(y_pred) == len(y_true), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true)), "y_true contains NaN values"

        return StepOutput(
            y_pred=y_pred,
            y_true=y_true,
            entry_mask=entry_mask,
            views=[
                ViewOutput(
                    global_embedding=global_embedding,
                    src_padding_mask=src_padding_mask,
                    padded_node_idx=padded_node_idx,
                    attn=attn_matrix,
                )
            ],
        )

    def get_global_embeddings(self, x, edge_index):
        return self.forward(x, edge_index)

    # acts as a factory method to create a module from a config
    @staticmethod
    def from_config(cfg, *, local_mask_hops: int, **kwargs):
        """Build the configured global module.

        ``local_mask_hops`` is how many hops the local component has already mixed into each
        embedding -- its ``receptive_field_hops``, or 0 for a model without one -- and is what the
        long-range mask blocks. It is a required argument rather than a config key because only
        the caller knows which local module it built; reading it from the config used a key that
        did not exist, and every model silently fell back to one hop.
        """
        module_name = cfg.model.global_component.name
        params = cfg.model.global_component.parameters.copy()  # Make a copy to avoid modifying the original

        if module_name == "self-attn-transformer":
            from interscale.module.global_modules import TransformerNodeEncoderHook
            from interscale.module.global_modules.attention import attention_head_dim
            from interscale.module.global_modules.pooling import build_pooling
            from interscale.module.global_modules.positional_encodings import build_positional_encodings
            from interscale.train.losses import LOSSES_READING_ATTENTION

            attention = params.get("attention", None)
            n_embed = kwargs.get("n_embed")
            head_dim = attention_head_dim(attention, n_embed, params["n_heads"]) if n_embed is not None else None
            return TransformerNodeEncoderHook(
                max_seq_len=params["max_seq_len"],
                n_heads=params["n_heads"],
                dropout_global=params["dropout_global"],
                act_func=params["activation_func"],
                num_layers=params["num_layers"],
                dim_feedforward=params["dim_feedforward"],
                long_range_attention=params["long_range_attention"],
                local_mask_hops=local_mask_hops,
                return_attention_maps=cfg.optim.get("loss", None) in LOSSES_READING_ATTENTION,
                attention=attention,
                attention_seed=int(cfg.optim.get("seed", 0) or 0),
                precomputed_mask_fill=params.get("precomputed_mask_fill", "learned"),
                programme_masking=bool(list(cfg.dataset.get("mask_genes", []) or [])),
                pooling=build_pooling(cfg, n_embed) if n_embed is not None else None,
                remask=params.get("remask", "none"),
                **build_positional_encodings(cfg, n_embed=n_embed, n_heads=params["n_heads"], head_dim=head_dim),
                **kwargs,
            )
        # Add more elifs for other modules
        else:
            raise ValueError(f"Unknown local module name: {module_name}")
