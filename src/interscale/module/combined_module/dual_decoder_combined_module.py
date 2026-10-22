from typing import Literal

import torch
from yacs.config import CfgNode as CN

from interscale.module.base import BaseModule, GlobalModule, LocalModule, StepOutput, ViewOutput
from interscale.tl.depth import RegimeLoss
from interscale.tl.masking import masked_loss


class DualDecoderCombinedModule(BaseModule):
    """Combined module with decoders for both local and global modules.

    This class uses decoders from both the local and global modules to predict
    from their respective embeddings. The loss function combines predictions
    from both decoders on masked tokens.
    """

    def __init__(self, cfg: CN, **base_module_kwargs):
        super().__init__(**base_module_kwargs)

        self.local_module_args = cfg.model.local_component
        self.global_module_args = cfg.model.global_component

        # module_name = cfg.model.local_component.name
        # local_class = MODULE_REGISTRY.get(module_name)

        # if local_class is None:
        #     raise ValueError(f"Module {module_name} not found in MODULE_REGISTRY")

        self.registered_local_component = True
        self.registered_global_component = True
        # `model.decoder.global_residual`: the global head's prediction is the local one (detached)
        # plus the global decoder's output -- see `combine_global`.
        self.global_residual = bool(cfg.model.decoder.get("global_residual", False))
        # `model.decoder.additive`: the global head's prediction is the sum of the two decoders' outputs, the local half
        # of the stacked output is the local decoder's alone -- see `combine_global`.
        self.additive = bool(cfg.model.decoder.get("additive", False))
        if self.additive and self.global_residual:
            raise ValueError(
                "model.decoder.additive and model.decoder.global_residual both make the global head a sum of the two "
                "decoders: choose one."
            )
        # `optim.regime_loss`: each head's loss on the hidden cells of its regime (`tl.depth`), or None.
        self.regime = RegimeLoss.from_config(cfg)

        # Local module with decoder
        self.local_module = LocalModule.from_config(
            cfg,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=cfg.model.decoder.type,
            dropout_decoder=cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=cfg.model.decoder.hidden_dims,
            mask_percentage=self.mask_percentage,
            mask_strategy=self.mask_strategy,
            block_masking=self.block_masking,
        )
        # Global module with decoder
        self.global_module = GlobalModule.from_config(
            cfg,
            # The long-range mask blocks exactly what the local module above already mixed.
            local_mask_hops=self.local_module.receptive_field_hops,
            n_input=self.n_input,
            n_output=self.n_output,
            n_embed=self.n_embed,
            decoder_type=cfg.model.decoder.type,
            dropout_decoder=cfg.model.decoder.dropout_decoder,
            decoder_hidden_dims=cfg.model.decoder.hidden_dims,
            mask_percentage=self.mask_percentage,
            mask_strategy=self.mask_strategy,
            block_masking=self.block_masking,
        )

        # Store split point for separating concatenated predictions
        # For node-level: first half is local, second half is global
        # For graph-level: only global predictions (no split)
        self._n_masked_nodes = None
        self._is_graph_level = False

    def predict_local(self, local_embedding, node_idx=None):
        """Predict with the local decoder, optionally reordered/subset to ``node_idx``.

        Parameters
        ----------
        local_embedding: torch.Tensor
            Size: [N, E]
        node_idx: torch.Tensor | None
            Batch-global node indices to gather, in the order wanted. Pass
            `_process_batch_for_metrics`'s ``padded_node_idx`` to line the local predictions up
            with the global branch's output. ``None`` returns all nodes in batch order.

        Returns
        -------
        y_pred_local: torch.Tensor
            Size: [len(node_idx), C] or [len(node_idx), F]

        Notes
        -----
        This used to take ``mask_idx`` -- the masked nodes in the *batch's* order -- while the
        global branch was subset by ``adjusted_mask_idx``, the masked nodes in the *padded* order.
        The two halves were then concatenated and compared row against row, which is only correct
        when the padding keeps every cell in its original order. Gathering both branches by
        ``padded_node_idx`` makes the pairing correct by construction.
        """
        y_pred_all = self.local_module.decoder.forward(local_embedding)
        return y_pred_all if node_idx is None else y_pred_all[node_idx]

    def predict_global(self, global_embedding, src_padding_mask, prediction_level):
        """Predict with the global decoder.

        Parameters
        ----------
        global_embedding: torch.Tensor
            Size: [S+1, B, E] where S is sequence length, B is batch size
        src_padding_mask: torch.Tensor
            Padding mask. Size: [B, S+1]
        prediction_level: Literal["node", "graph"]
            Level of prediction

        Returns
        -------
        y_pred_global: torch.Tensor
            Size: [N, C] or [N, F] where N depends on prediction_level
        """
        # Get predictions from global decoder
        y_pred = self.global_module.predict(global_embedding, src_padding_mask, prediction_level)
        return y_pred

    @property
    def global_is_sum(self) -> bool:
        """The global head is the local and the global decoder's outputs added (``additive`` or ``global_residual``)."""
        return self.additive or self.global_residual

    def combine_global(self, y_pred_local: torch.Tensor, y_pred_global: torch.Tensor) -> torch.Tensor:
        """The global head's prediction from the two decoders' outputs for the same cells, in the same order.

        Under ``model.decoder.global_residual`` it is ``y_pred_local.detach() + y_pred_global``: the
        global decoder models what the local prediction misses. The global loss does not reach the
        local prediction, so the local decoder answers to the local loss alone and cannot drift
        toward whatever makes the sum easy; the GCN still gets the global loss through the tokens,
        as under plain dual decoding. Under ``model.decoder.additive`` it is ``y_pred_local + y_pred_global``
        with the gradient reaching both decoders: the two outputs are parts of one prediction, and the loss on the sum
        is what each is trained for (decoder_plan.md, D3.1). Otherwise the global decoder's output as it is.
        """
        if self.additive:
            return y_pred_local + y_pred_global
        if not self.global_residual:
            return y_pred_global
        return y_pred_local.detach() + y_pred_global

    def forward(self, batch_masked):
        """Forward pass through the model"""
        local_out = self.local_module.forward(batch_masked.x, batch_masked.edge_index)

        if isinstance(local_out, dict):  # neede for scVI component
            local_embedding = local_out["embedding"]
            self._current_local_latent_params = local_out
        else:
            local_embedding = local_out
            self._current_local_latent_params = None

        global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn = self.global_module.step(
            batch_masked, local_embedding
        )

        return local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn

    def _common_step(self, batch, prediction_task, prediction_level: Literal["node", "graph"]):
        """Shared step between train, val and test.

        Returns
        -------
        StepOutput
            ``y_pred``/``y_true`` are the stacked ``[2N, ...]`` local-then-global form, over
            EVERY cell the transformer produced -- not only the masked ones.
            ``entry_mask`` marks the entries the loss is restricted to; the metrics use
            everything. The single view carries both embeddings and the ``padded_node_idx``
            that aligns them.
        """
        batch_masked, _, _ = self._common_step_masking(batch)

        local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn = self.forward(
            batch_masked
        )
        loss_mask = None
        parts = None

        # Predict from global embedding (one row per cell the padding kept)
        y_pred_global = self.predict_global(global_embedding, src_padding_mask, prediction_level)

        if prediction_task == "classification" and prediction_level == "graph":
            # One prediction per graph, so there is no local/global pairing to do.
            y_true = batch.y[batch.ptr[:-1]]
            y_pred_combined = y_pred_global
            y_true_combined = y_true
            self._n_masked_nodes = None
            self._is_graph_level = True
            entry_mask_combined = None
            padded_node_idx = None
        elif prediction_level == "node":
            y_true, padded_node_idx, entry_mask = self.global_module._process_batch_for_metrics(
                batch, prediction_task, prediction_level, pad_index_nodes
            )

            # Both branches gathered by the SAME index, so row i of each is the same cell.
            y_pred_local = self.predict_local(local_embedding, padded_node_idx)
            y_pred_global_decoded = y_pred_global
            y_pred_global = self.combine_global(y_pred_local, y_pred_global)
            if self.global_is_sum:
                # The global head is the local prediction plus what its decoder adds: two parts to attribute. Under
                # `additive` they are the sum's own tensors, gradient and all: the head `global_part` the evaluation adds
                # is read by the influence readouts, which differentiate it. A residual head's local part is the detached one
                # the sum used.
                local_part = y_pred_local if self.additive else y_pred_local.detach()
                parts = {"global": {"local": local_part, "global": y_pred_global_decoded}}

            if entry_mask is None:
                # Node-level classification: the masked cells are the supervision targets, so
                # both halves stay restricted to them. Only reconstruction scores every cell.
                keep = batch.mask[padded_node_idx].bool()
                y_pred_local, y_pred_global, y_true = y_pred_local[keep], y_pred_global[keep], y_true[keep]
                if parts is not None:
                    parts = {"global": {name: part[keep] for name, part in parts["global"].items()}}

            assert y_pred_local.shape == y_pred_global.shape, (
                f"Local and global predictions differ in shape: "
                f"{tuple(y_pred_local.shape)} vs {tuple(y_pred_global.shape)}"
            )
            assert len(y_true) == len(y_pred_local), (
                f"Ground truth and predictions differ in length: {len(y_true)} vs {len(y_pred_local)}"
            )

            self._n_masked_nodes = len(y_pred_local)
            self._is_graph_level = False

            # [N, C] + [N, C] -> [2N, C], so the loss can score both decoders.
            y_pred_combined = torch.cat([y_pred_local, y_pred_global], dim=0)
            y_true_combined = torch.cat([y_true, y_true], dim=0)
            # The entry mask is per-cell, and both halves are now the same cells in the same
            # order, so it is simply stacked twice.
            entry_mask_combined = None if entry_mask is None else torch.cat([entry_mask, entry_mask], dim=0)
            if self.regime is not None:
                # Each head's loss on the cells of its regime; the metrics keep every hidden entry.
                loss_mask = self.regime.loss_mask(
                    entry_mask,
                    self.global_module.wholly_hidden(batch_masked),
                    self._positions(batch),
                    batch.batch,
                    padded_node_idx,
                )
        else:
            raise ValueError(f"Invalid prediction level: {prediction_level}")

        assert len(y_pred_combined) == len(y_true_combined), "y_pred and y_true are not consistent"
        assert not torch.any(torch.isnan(y_pred_combined)), "y_pred contains NaN values"
        assert not torch.any(torch.isnan(y_true_combined)), "y_true contains NaN values"

        return StepOutput(
            y_pred=y_pred_combined,
            y_true=y_true_combined,
            entry_mask=entry_mask_combined,
            loss_mask=loss_mask,
            parts=parts,
            views=[
                ViewOutput(
                    local_embedding=local_embedding,
                    global_embedding=global_embedding,
                    src_padding_mask=src_padding_mask,
                    padded_node_idx=padded_node_idx,
                    attn=attn,
                )
            ],
        )

    @staticmethod
    def _positions(batch) -> torch.Tensor:
        """The batch's cell coordinates, which ``optim.regime_loss`` measures depth in."""
        pos = getattr(batch, "pos", None)
        if pos is None:
            raise ValueError(
                "optim.regime_loss measures how far a hidden cell lies from a visible one, but the graphs carry "
                "no coordinates (`pos`): set dataset.spatial_key."
            )
        return pos

    def _head_loss(self, loss_fn, loss_type, y_pred, y_true, entry_mask):
        """One head's masked loss; a zero that still reaches the head when its regime has no cell in the batch.

        Without ``optim.regime_loss`` every hidden cell counts for both heads and nothing is special-cased.
        Under it a head can have none (a batch without a deep cell, a slide hidden nowhere), where a mean over
        no cells is NaN.
        """
        if self.regime is not None and entry_mask is not None and not bool(entry_mask.any()):
            return y_pred.sum() * 0.0
        return masked_loss(loss_fn, loss_type, y_pred, y_true, entry_mask)

    def get_separate_predictions(self, y_pred_combined, y_true_combined):
        """Get separate predictions and ground truth for local and global decoders.

        Parameters
        ----------
        y_pred_combined: torch.Tensor
            Combined predictions from _common_step. For node-level: [2*N_masked, C],
            for graph-level: [B, C] where B is batch size.
        y_true_combined: torch.Tensor
            Combined ground truth from _common_step. Same shape as y_pred_combined.

        Returns
        -------
        dict: Dictionary with keys:
            - 'local': tuple of (y_pred_local, y_true_local) or None if not available
            - 'global': tuple of (y_pred_global, y_true_global) or None if not available
        """
        local = None
        global_pred = None

        if self._is_graph_level:
            # For graph level, only global predictions exist
            global_pred = (y_pred_combined, y_true_combined)
        elif self._n_masked_nodes is not None:
            # For node level, split the concatenated predictions
            # First half is local, second half is global
            y_pred_local = y_pred_combined[: self._n_masked_nodes]
            y_pred_global = y_pred_combined[self._n_masked_nodes :]
            y_true_local = y_true_combined[: self._n_masked_nodes]
            y_true_global = y_true_combined[self._n_masked_nodes :]

            local = (y_pred_local, y_true_local)
            global_pred = (y_pred_global, y_true_global)

        return {"local": local, "global": global_pred}

    def compute_separate_losses(
        self,
        loss_fn,
        loss_type: Literal["GaussianNLL", "MSELoss", "CrossEntropy", "WeightedCE"],
        y_pred_combined: torch.Tensor,
        y_true_combined: torch.Tensor,
        entry_mask_combined: torch.Tensor | None = None,
    ):
        """Compute separate losses for local and global predictions.

        Parameters
        ----------
        loss_fn
            Loss function that takes ``(y_pred, y_true)`` and returns a scalar loss.
            Should be compatible with the prediction task (classification or regression).
        loss_type
            Type of loss function to use. One of ``"GaussianNLL"``, ``"MSELoss"``,
            ``"CrossEntropy"``, ``"WeightedCE"``.
        y_pred_combined
            Combined predictions from ``_common_step``. For node-level:
            ``[2*N_masked, C]``; for graph-level: ``[B, C]`` where ``B`` is batch size.
        y_true_combined
            Combined ground truth from ``_common_step``. Same shape as ``y_pred_combined``.
        entry_mask_combined
            Optional ``[2*N_masked, F]`` boolean from ``_common_step``. When given, each half's
            loss is computed over that half's masked entries only -- which is the whole point of
            gene masking, since the unmasked entries were handed to the model as input and
            reconstructing them is the identity. Under ``optim.regime_loss`` the training plan passes the
            step's ``loss_mask`` here instead: the same entries, each half cut to its regime's cells.

        Returns
        -------
        Dictionary with keys ``"local_loss"``, ``"global_loss"``, ``"combined_loss"``
        (average of local and global), and ``"kl_loss"``. Any value may be ``None``.
        """
        losses = {"local_loss": None, "global_loss": None, "combined_loss": None, "kl_loss": None}

        local_loss = None
        global_loss = None

        if self._is_graph_level:
            # For graph level, only global predictions exist
            global_loss = loss_fn(y_pred_combined, y_true_combined)
            losses["global_loss"] = global_loss
        elif self._n_masked_nodes is not None:
            # For node level, split the concatenated predictions
            # First half is local, second half is global
            y_pred_local = y_pred_combined[: self._n_masked_nodes]
            y_pred_global = y_pred_combined[self._n_masked_nodes :]
            y_true_local = y_true_combined[: self._n_masked_nodes]
            y_true_global = y_true_combined[self._n_masked_nodes :]
            if entry_mask_combined is None:
                mask_local = mask_global = None
            else:
                mask_local = entry_mask_combined[: self._n_masked_nodes]
                mask_global = entry_mask_combined[self._n_masked_nodes :]
            local_loss = self._head_loss(loss_fn, loss_type, y_pred_local, y_true_local, mask_local)
            global_loss = self._head_loss(loss_fn, loss_type, y_pred_global, y_true_global, mask_global)
            losses["local_loss"] = local_loss
            losses["global_loss"] = global_loss

        if self._current_local_latent_params is not None:
            losses["kl_loss"] = self.local_module.loss_kl(self._current_local_latent_params)
            self._current_local_latent_params = None

        return losses

    def get_model_summary(self) -> str:
        """Returns a string containing the model's parameters summary.

        Returns
        -------
            str: Summary string with model parameters
        """
        summary = (
            f"Dual Decoder Combined Module: \n"
            f"Local Module: {self.local_module.get_model_summary()}\n"
            f"Global Module: {self.global_module.get_model_summary()}\n"
        )
        return summary
