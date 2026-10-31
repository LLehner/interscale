from abc import ABC, abstractmethod
from typing import Literal

import pytorch_lightning as L
import torch

from interscale.nn import LinearDecoder, LinearLSEDecoder, NonLinearDecoder
from interscale.tl.masking import MASK_STRATEGIES, SECTION_OFFSETS, apply_mask, section_visible_mean


class BaseModule(L.LightningModule, ABC):
    """Abstract base class for all models defining the common training interface."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_embed: int = 16,
        decoder_type: None | Literal["linear", "nonlinear"] = "linear",
        dropout_decoder: float = 0.2,
        decoder_hidden_dims: list[int] = [128, 128],
        dual_decoder: bool = False,
        mask_percentage: float = 0.0,
        mask_strategy: Literal["node", "gene"] = "node",
        type_gex_embedding: Literal["PCA", "NMF", "scvi"] | None = None,
        block_masking: bool = False,
        section_offset: str = "none",
    ):
        """
        Parameters
        ----------
        n_input: int
            Number of input features.
        n_output: int
            If classification, number of output features / classes.
            For example, number of cell types.
        n_embed: int
            Number of embedding dimensions.
        decoder_type: Literal["linear", "nonlinear", linear-lse"]
            Type of decoder to use. For combined module the submodules will potentially not have their own decoder (set to None).
        dropout_decoder: float
            Dropout rate for the decoder only if decoder_type is "nonlinear".
        decoder_hidden_dims: List[int]
            Hidden dimensions for the decoder only if decoder_type is "nonlinear".
        dual_decoder: bool
            If True, use dual decoder for combined module. Both local and global decoders are used.
        mask_percentage: float
            Bernoulli masking probability -- per cell under ``mask_strategy="node"``, per
            (cell, gene) entry under ``"gene"``.
        mask_strategy: Literal["node", "gene"]
            Granularity of the reconstruction corruption; see ``interscale.tl.masking``.
        type_gex_embedding: Literal["PCA", "NMF","scvi"] | None
            Type of GEX embedding to use.
        block_masking: bool
            Spatial blocks are hidden as well (``dataset.mask_block_fraction`` > 0). With a
            ``mask_percentage`` of 0 they are the only corruption, and still count as one.
        section_offset: Literal["none", "visible_mean"]
            ``"visible_mean"``: the prediction gets a part ``section`` = ``w * m``, ``m`` each gene's mean over the visible
            entries of the cell's graph (``tl.masking.section_visible_mean``), ``w`` one learned weight per gene starting at 1.
            The model then predicts each cell as its section's visible mean plus what the decoder adds, so a section's own
            offset (donor, condition, batch) is not what the embedding has to carry. Reconstruction only (``n_output`` is the
            genes); a LocalModel or GlobalModel puts it into its prediction, a combined one refuses it.
        """
        super().__init__()

        self.module_name = None

        self.n_input = n_input
        self.n_embed = n_embed
        self.n_output = n_output
        self.dropout_decoder = dropout_decoder
        self.decoder_type = decoder_type
        self.decoder_hidden_dims = decoder_hidden_dims
        self.dual_decoder = dual_decoder
        self.mask_percentage = mask_percentage
        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError(f"mask_strategy must be one of {MASK_STRATEGIES}, got {mask_strategy!r}.")
        self.mask_strategy = mask_strategy
        self.type_gex_embedding = type_gex_embedding
        # `masked_nodes` means "the reconstruction input is corrupted at all", not "whole cells
        # are blanked". Both strategies set it: under gene masking every cell is a supervision
        # target, so `pad_batch`'s keep_indices is all-True and it degenerates to the plain
        # subsampling branch, which is the correct behaviour.
        self.block_masking = bool(block_masking)
        self.masked_nodes = self.mask_percentage > 0 or self.block_masking

        if section_offset not in SECTION_OFFSETS:
            raise ValueError(f"section_offset must be one of {SECTION_OFFSETS}, got {section_offset!r}.")
        self.section_offset = section_offset
        if section_offset != "none":
            if n_output != n_input:
                raise ValueError(
                    f"section_offset adds each gene's section mean to its prediction: it needs a reconstruction (n_output "
                    f"{n_output} = n_input {n_input})."
                )
            # Ones, not an initialiser: no draw from the global RNG, so every other weight is initialised as without it.
            self.section_offset_weight = torch.nn.Parameter(torch.ones(n_output))

        # Define components
        self.local_component = None
        self.global_component = None

        if self.decoder_type == "linear-lse":
            self.decoder = LinearLSEDecoder(n_input=self.n_embed, n_output=self.n_output)
        elif self.decoder_type == "linear":
            self.decoder = LinearDecoder(n_input=self.n_embed, n_output=self.n_output)
        elif self.decoder_type == "nonlinear":
            self.decoder = NonLinearDecoder(
                n_input=self.n_embed,
                n_output=self.n_output,
                hidden_dims=self.decoder_hidden_dims,
                dropout=self.dropout_decoder,
            )
        elif self.decoder_type == None:  # If Local + Global model sequential and no decoder needed
            self.decoder = None
        else:
            raise ValueError(f"Decoder {self.decoder_type} not found.")

    def _common_step_masking(self, batch):
        """Corrupt the reconstruction input of the batch.

        Parameters
        ----------
        batch: Batch
            Batch of data.

        Returns
        -------
        batch_masked: Batch
            Batch of data with the masked entries set to MASK_VALUE.
        mask_idx: torch.Tensor
            Indices of the cells that are supervision targets. Size: [N_masked_nodes, ]
        entry_mask: torch.Tensor | None
            ``[N, G]`` boolean over the full node ordering when ``mask_strategy == "gene"``,
            marking the entries the loss must be restricted to; ``None`` under cell masking,
            where the whole row of every target cell is scored. Callers must subset it with the
            same row indices they apply to ``y_true``.
        """
        if self.masked_nodes:
            batch_masked, mask_idx, entry_mask = apply_mask(batch, self.mask_strategy)
        else:
            mask_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
            batch_masked = batch
            entry_mask = None
        return batch_masked, mask_idx, entry_mask

    def visible_entries(self, batch_masked) -> torch.Tensor:
        """``[N, G]`` bool: the entries of a masked batch the model was given (the complement of what the loss scores).

        ``gene`` strategy: everything but ``gene_mask`` (hidden genes, sets, channels, and the rows of spatial blocks); ``node``:
        everything but the rows of ``mask`` (random cells and blocks). All of them without masking.
        """
        x = batch_masked.x
        if not self.masked_nodes:
            return torch.ones(x.shape, dtype=torch.bool, device=x.device)
        if self.mask_strategy == "gene":
            gene_mask = getattr(batch_masked, "gene_mask", None)
            return torch.ones(x.shape, dtype=torch.bool, device=x.device) if gene_mask is None else ~gene_mask.bool()
        return ~batch_masked.mask.bool()[:, None].expand(x.shape)

    def section_input(self, batch, masked: bool = True) -> torch.Tensor | None:
        """``[N, G]`` in batch node order: the section part of each cell's prediction (``section_offset``), None when off.

        ``masked``: ``batch`` is the masked batch of a training-style pass, its hidden entries left out of the means; False: a
        clean batch (inference), every entry counts. The cells' graphs are ``batch.batch`` (one graph when absent).
        """
        if self.section_offset == "none":
            return None
        x = batch.x.float()
        visible = self.visible_entries(batch) if masked else torch.ones(x.shape, dtype=torch.bool, device=x.device)
        graph = getattr(batch, "batch", None)
        mean = section_visible_mean(x, visible, None if graph is None else graph.long())
        return mean * self.section_offset_weight.to(mean.dtype)

    @abstractmethod
    def _common_step(self, batch):
        """Shared step between train, val and test."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor | None = None,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
            mask: Node mask [N]

        Returns
        -------
            z: Embeddings [N, E]
            out: Model predictions
            index_nodes: Node indices [N]
        """
        pass

    # @abstractmethod
    # def loss(self, *args, **kwargs):
    #     """Compute the loss for a minibatch of data.

    #     This function uses the outputs of the inference and generative functions to compute
    #     a loss. This many optionally include other penalty terms, which should be computed here.
    #     """
