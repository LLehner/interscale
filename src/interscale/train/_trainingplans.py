from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

from interscale.module.base._base_module import BaseModule
from interscale.nn import CosineWarmupScheduler
from interscale.tl.masking import masked_loss

from .aux_losses import CompositeAuxLoss
from .losses import BalancedPearsonCorrelationLoss, SCE_EntropyATT_Loss, SCELoss


class RunningCosineSimilarity(torchmetrics.Metric):
    """Mean per-cell cosine similarity, with state that does not grow with the dataset.

    ``torchmetrics.CosineSimilarity`` is a *list-state* metric: it keeps every prediction and
    target it is shown and concatenates them at compute time. Every other regression metric here
    holds a few kilobytes of running sums, and this one holds ``n_cells x n_genes x 2`` floats --
    which ``MetricCollection.forward`` then duplicates via ``_copy_state_dict`` on every step.

    On legnini23 (43k cells, 88 genes) that is ~30 MB and invisible. On the CosMx pancreas
    (387k cells, 979 genes) one epoch is ~850 MB before the copy, and it OOMed a 20 GB card
    inside ``_regression_metrics`` on the very first trial, regardless of batch size -- the total
    per epoch is the same however the cells are batched.

    This computes the same quantity (the mean over cells of the per-cell cosine) from a running
    sum and count, so the state is two scalars.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate the summed per-cell cosine similarity and the number of cells."""
        cos = nn.functional.cosine_similarity(preds, target, dim=1)
        self.total = self.total + cos.sum()
        self.count = self.count + cos.numel()

    def compute(self) -> torch.Tensor:
        """Mean per-cell cosine similarity over everything seen since the last reset."""
        return self.total / self.count


def masked_regression_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, entry_mask: torch.Tensor, eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    """The regression metrics of ``_setup_regression_metrics``, restricted to the masked entries.

    Under gene masking the scored rows are mostly entries the model was *given*. Feeding the full
    rows to the ``MetricCollection`` would score the identity map on those and inflate every
    number -- including ``val_r2``, which drives early stopping and checkpoint selection. There
    is no way to express "these entries only" to a torchmetrics per-output metric (the surviving
    entries are ragged across genes), so the same quantities are computed here from masked sums.

    Every entry that is not masked is multiplied by zero before any sum is taken, and zeros
    contribute nothing to a sum, so each moment below is exactly the moment over the masked
    entries -- no approximation.

    Parameters
    ----------
    y_pred, y_true
        ``[N, G]`` predictions and targets for the scored rows.
    entry_mask
        ``[N, G]`` boolean marking the masked entries.
    eps
        Guard for degenerate (zero-variance) genes.

    Returns
    -------
    dict
        Unprefixed metric names mapped to scalar tensors, matching the keys that
        ``_setup_regression_metrics`` produces: ``mse``, ``r2`` (per-gene, uniform average),
        ``pearson_corr``, ``concordance_corr``, ``cosine_similarity`` (per cell).
    """
    m = entry_mask.to(y_pred.dtype)
    p_ = y_pred * m
    t_ = y_true * m

    n_gene = m.sum(dim=0)  # [G] masked cells per gene
    n_total = m.sum()

    mse = ((p_ - t_) ** 2).sum() / n_total.clamp(min=1)

    # Per-gene first and second moments over that gene's masked cells.
    ng = n_gene.clamp(min=1)
    mean_p = p_.sum(dim=0) / ng
    mean_t = t_.sum(dim=0) / ng
    var_p = (p_**2).sum(dim=0) / ng - mean_p**2
    var_t = (t_**2).sum(dim=0) / ng - mean_t**2
    cov = (p_ * t_).sum(dim=0) / ng - mean_p * mean_t

    # A gene with fewer than two masked cells, or with no spread in either vector, has no
    # correlation defined; NaN it out and let nanmean skip it, exactly as the unmasked path
    # already does for constant genes.
    nan = torch.tensor(float("nan"), device=y_pred.device, dtype=y_pred.dtype)
    usable = (n_gene >= 2) & (var_t > eps)

    pearson = torch.where(usable & (var_p > eps), cov / (var_p.clamp(min=eps) * var_t.clamp(min=eps)).sqrt(), nan)
    concordance = torch.where(usable, 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + eps), nan)

    # R2 per gene, then uniform average -- the same reduction torchmetrics'
    # R2Score(multioutput="uniform_average") applies.
    ss_res = ((p_ - t_) ** 2).sum(dim=0)
    ss_tot = var_t * ng
    r2 = torch.where(usable, 1 - ss_res / ss_tot.clamp(min=eps), nan)

    # Per-cell cosine over that cell's masked genes: the zeroed entries drop out of both the dot
    # product and the two norms, so this is the cosine on the masked coordinates.
    cosine = nn.functional.cosine_similarity(p_, t_, dim=1)

    return {
        "mse": mse,
        "r2": torch.nanmean(r2),
        "pearson_corr": torch.nanmean(pearson),
        "concordance_corr": torch.nanmean(concordance),
        "cosine_similarity": cosine.mean(),
    }


CLASSIFICATION_LOSSES = ["CrossEntropy", "WeightedCE"]
REGRESSION_LOSSES = [
    "MSELoss",
    "GaussianNLL",
    "SmoothL1",
    "BalancedPearsonCorrelationLoss",
    "SCELoss",
    "SCE_EntropyATT_Loss",
]


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/main/src/scvi/train/_trainingplans.py
# accessed on 28 April 2025
class TrainingPlan(pl.LightningModule):
    """Lightning module task to train scvi-tools modules.

    The training plan is a PyTorch Lightning Module that is initialized
    with a scvi-tools module object. It configures the optimizers, defines
    the training step and validation step, and computes metrics to be recorded
    during training. The training step and validation step are functions that
    take data, run it through the model and return the loss, which will then
    be used to optimize the model parameters in the Trainer. Overall, custom
    training plans can be used to develop complex inference schemes on top of
    modules.

    The following developer tutorial will familiarize you more with training plans
    and how to use them: :doc:`/tutorials/notebooks/dev/model_user_guide`.

    Parameters
    ----------
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.

    lr_scheduler: None | Literal["ReduceLROnPlateau", "CosineWarmupScheduler"] = None
        Learning rate scheduler to use. Default is None. CosineWarmupScheduler reduces LR at each step, ReduceLROnPlateau reduces LR with a patience if no improvement is seen.
    """

    def __init__(
        self,
        module: BaseModule,
        prediction_task: str,
        prediction_level: Literal["node", "graph"],
        loss: Literal[CLASSIFICATION_LOSSES, REGRESSION_LOSSES],
        cross_corr: Literal["gene", "cell"],
        batch_size: int,
        class_weights: np.ndarray | None = None,
        class_labels: list[str] | None = None,
        *,
        aux_losses: CompositeAuxLoss | None = None,
        lr_scheduler: None | Literal["ReduceLROnPlateau", "CosineWarmupScheduler"] = None,
        weight_decay: float = 1e-6,
        lr: float = 1e-3,
        lr_warmup: int = 0,
        lr_max_epochs: int = 100000,
        patience_in_steps: int = 100000,
        **kwargs,
    ):
        super().__init__()
        self.module = module
        # Auxiliary terms added beside the reconstruction criterion; empty unless some
        # `optim.aux_loss_weights` entry is non-zero. They are stored on the MODULE, not here, so
        # that their heads travel with `module.state_dict()` and `module.parameters()` -- see
        # `BaseModel._attach_aux_losses`. `self.aux_losses` is a read-only view of that; binding
        # them here as well would register the same parameters under two parents and hand the
        # optimiser each of them twice.
        if aux_losses is not None:
            self.module.aux_losses = aux_losses
        elif getattr(self.module, "aux_losses", None) is None:
            self.module.aux_losses = CompositeAuxLoss({}, {})
        self.prediction_task = prediction_task
        self.prediction_level = prediction_level
        self.loss_type = loss
        self.cross_corr = cross_corr
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.class_labels = class_labels
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.patience_in_steps = patience_in_steps
        self.lr_warmup = lr_warmup
        self.lr_max_epochs = lr_max_epochs
        self.lr = lr
        if self.prediction_task == "regression":
            if self.cross_corr == "gene":
                print("cross-gene per cell correlation metrics")
                self.AXIS = 1  # selecting rows / cells
            elif self.cross_corr == "cell":
                print("cross-cell per gene correlation metrics")
                self.AXIS = 0  # selecting columns / genes

        # setup metrics and loss
        if "classification" in self.prediction_task:
            metrics = self._setup_classification_metrics(self.module.n_output)
            self.loss = self._setup_classification_loss(self.loss_type, self.class_weights)
            # Must name a metric that is actually logged -- this is handed to Lightning as the
            # LR-scheduler monitor. "val_f1" never existed; only val_f1_micro/macro/<class> do.
            self.monitor_metric = "val_f1_macro"
        elif "regression" in self.prediction_task:
            metrics = self._setup_regression_metrics(self.module.n_output)
            self.loss = self._setup_regression_loss(self.loss_type)
            self.monitor_metric = "val_r2"
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    @staticmethod
    def _setup_classification_loss(
        loss: Literal["CrossEntropy", "WeightedCE"], class_weights: torch.Tensor | None = None
    ):
        """Setup loss function based on prediction task and configuration."""
        assert loss in CLASSIFICATION_LOSSES, "Classification must be run with CrossEntropy or WeightedCE loss."
        if loss == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif loss == "WeightedCE":
            assert class_weights is not None, "Class weights must be provided for WeightedCE loss."
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch tensor"
            # .float() guards against a float64 weight buffer meeting float32 logits.
            return nn.CrossEntropyLoss(weight=class_weights.float())

    def _setup_regression_loss(self, loss: Literal[REGRESSION_LOSSES]):
        """Setup loss function based on prediction task and configuration."""
        assert loss in REGRESSION_LOSSES, (
            f"{loss} not in {REGRESSION_LOSSES}"
        )  # "Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss."
        if loss == "MSELoss":
            return nn.MSELoss()
        elif loss == "GaussianNLL":
            return nn.GaussianNLLLoss()
        elif loss == "SmoothL1":
            return nn.SmoothL1Loss()
        elif loss == "BalancedPearsonCorrelationLoss":
            return BalancedPearsonCorrelationLoss(None)
        elif loss == "SCELoss":
            return SCELoss()
        elif loss == "SCE_EntropyATT_Loss":
            return SCE_EntropyATT_Loss()

    @staticmethod
    def _setup_classification_metrics(num_outputs: int):
        return MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs),
                "f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="micro"),
                "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="macro"),
                "f1_per_class": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average=None),
            }
        )

    @staticmethod
    def _setup_regression_metrics(num_outputs: int):
        return MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "r2": torchmetrics.R2Score(multioutput="uniform_average"),
                "pearson_corr": torchmetrics.PearsonCorrCoef(num_outputs=num_outputs),
                # Pearson is invariant to any per-gene affine rescaling of the predictions, so a
                # model whose outputs are (say) 11x too spread out still scores a high r while
                # its R2 goes to -113. Writing predictions as k times the true sd with offset d,
                # R2 = 2*r*k - k^2 - d^2/sigma^2, maximised at k = r -- so R2 <= r^2, and the gap
                # between them is purely calibration. Concordance correlation folds that penalty
                # back in, which makes it the metric to select on when both the co-variation
                # structure AND the expression scale have to be usable.
                "concordance_corr": torchmetrics.ConcordanceCorrCoef(num_outputs=num_outputs),
                # Not torchmetrics.CosineSimilarity: see RunningCosineSimilarity for why its
                # list state cannot be used on a dataset this size. Same value, O(1) memory.
                "cosine_similarity": RunningCosineSimilarity(),
            }
        )

    def _classification_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate classification metrics."""
        ## TODO: Currently mask_idx is applied in module._common_step. Maybe move to here?
        # if mask_idx is not None:
        #     y_pred = y_pred[mask_idx]
        #     y_true = y_true[mask_idx]

        loss = self.loss(y_pred, y_true)
        metrics = metrics(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        metrics[f"{mode}_loss"] = loss

        return loss, metrics

    def _regression_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: torch.Tensor | None = None,
        attn: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate regression metrics.

        Parameters
        ----------
        y_true : torch.Tensor
            True values of shape [N, G], where N is the number of cells and G is the number of genes
        y_pred : torch.Tensor
            Predicted values of shape [N, G], where N is the number of cells and G is the number of genes
            True and predicted values of shape [N, G], where N is the number of cells and G is the number of genes
        mode : str
            The mode of the metrics.
        metrics : MetricCollection
            The metrics to calculate.
        mask_idx : torch.Tensor | None
            The mask indices to apply to the metrics.
        attn : torch.Tensor | None
            The attention weights to apply to the metrics.
        entry_mask : torch.Tensor | None
            [N, G] boolean marking the entries that were actually masked.

            The LOSS is restricted to those entries -- that is the self-supervised objective, and
            training on entries the model was handed as input would make it the identity map.
            The METRICS are computed over every cell in ``y_pred``, masked or not, because a
            reconstruction is wanted for the whole tissue and not only for the hidden part.
            The masked-only versions are logged alongside under a ``masked_`` prefix; they are
            the honest held-out score and the two differ a lot, so read the prefix.
        """
        if self.loss_type == "SCE_EntropyATT_Loss":
            # Takes attention as a third argument, so it cannot go through masked_loss. Zeroing
            # the unmasked entries restricts its row-wise cosine to the masked coordinates,
            # which is what the row-structured branch of masked_loss does too.
            if entry_mask is None:
                loss = self.loss(y_pred, y_true, attn)
            else:
                m = entry_mask.to(y_pred.dtype)
                loss = self.loss(y_pred * m, y_true * m, attn)
        else:
            loss = masked_loss(self.loss, self.loss_type, y_pred, y_true, entry_mask)

        # Primary metrics: every cell the model produced, masked or not.
        metrics = metrics(y_pred, y_true)
        # Take mean across pearson correlation
        metrics[f"{mode}_pearson_corr"] = torch.nanmean(metrics[f"{mode}_pearson_corr"].contiguous())
        # Same reduction, same reason: both are per-gene vectors of length n_output, and a
        # constant gene yields NaN rather than a number.
        metrics[f"{mode}_concordance_corr"] = torch.nanmean(metrics[f"{mode}_concordance_corr"].contiguous())

        # Held-out companions on the masked entries only. Kept because the all-cell numbers above
        # include entries the model was given as input, which it can partly copy -- so they are
        # the reconstruction score for the tissue, not evidence the model generalises. These are.
        if entry_mask is not None:
            for name, value in masked_regression_metrics(y_pred, y_true, entry_mask).items():
                metrics[f"{mode}_masked_{name}"] = value

        metrics[f"{mode}_loss"] = loss
        return loss, metrics

    def forward(self, *args, **kwargs):
        """Passthrough to the module's forward method."""
        return self.module(
            *args,
            **kwargs,
        )

    # @torch.inference_mode() decorator disables gradient computation. TODO: enable again after calculating loss in module.
    def _compute_and_log_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        attn: torch.Tensor | None,
        entry_mask: torch.Tensor | None = None,
    ):
        """Helper method to log metrics for training, validation, or test steps.

        Parameters
        ----------
        y_true, y_pred: torch.Tensor
            True and predicted values of shape [N, G], where N is the number of cells and G is the number of genes
        mode
            One of 'train', 'val', or 'test'
        metrics: MetricCollection
            Metrics to log
        """
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        # TODO: where is the batch size?

        if "classification" in self.prediction_task:
            loss, metrics = self._classification_metrics(y_pred, y_true, mode, metrics)
            for class_idx, class_score in enumerate(metrics[f"{mode}_f1_per_class"]):
                metrics[f"{mode}_f1_{self.class_labels[class_idx]}"] = class_score
            metrics.pop(f"{mode}_f1_per_class")

        elif "regression" in self.prediction_task:
            loss, metrics = self._regression_metrics(y_pred, y_true, mode, metrics, attn=attn, entry_mask=entry_mask)

        # Set sync_dist=True only for test mode
        sync_dist = mode == "test"
        self.log_dict(metrics, batch_size=int(self.batch_size), on_step=False, on_epoch=True, sync_dist=sync_dist)

        return loss

    @property
    def aux_losses(self) -> CompositeAuxLoss:
        """The auxiliary terms, which live on the module -- see ``__init__``."""
        return self.module.aux_losses

    #: Per-mode differences between the three steps: the metric collection to update, and
    #: whether logging is synchronised across ranks.
    _MODE_METRICS = {"train": "train_metrics", "val": "valid_metrics", "test": "test_metrics"}

    def _log_scalar(self, name: str, value: torch.Tensor, sync_dist: bool) -> None:
        self.log(name, value, on_step=False, on_epoch=True, batch_size=int(self.batch_size), sync_dist=sync_dist)

    def _step(self, batch, mode: Literal["train", "val", "test"]):
        """The shared body of ``training_step`` / ``validation_step`` / ``test_step``.

        These were three near-identical copies. They are one so that a new loss term is wired in
        once, and so the three cannot drift apart -- which they already had: ``val`` never logged
        ``combined_loss`` although ``train`` and ``test`` did, and ``test`` logged ``kl_loss``
        without ``sync_dist`` while logging everything else with it. Both quirks are reproduced
        below rather than quietly fixed, so that this refactor changes no number; see the
        ``mode != "val"`` guard and the ``sync_dist=False`` on the KL log.

        Parameters
        ----------
        batch
            The batch as collated by the dataloader.
        mode
            Which split is running; selects the metric collection and the log prefix.

        Returns
        -------
        torch.Tensor
            The scalar to optimise (``train``) or report.
        """
        metrics = getattr(self, self._MODE_METRICS[mode])
        sync_dist = mode == "test"

        out = self._forward_views(batch)
        y_pred, y_true, entry_mask, attn = out.y_pred, out.y_true, out.entry_mask, out.attn

        # Modules that decode twice (DualDecoderCombinedModule) report each half separately.
        if not hasattr(self.module, "compute_separate_losses"):
            loss = self._compute_and_log_metrics(y_pred, y_true, mode, metrics, attn=attn, entry_mask=entry_mask)
            return self._add_aux_losses(loss, out, batch, mode, sync_dist)

        separate_losses = self.module.compute_separate_losses(self.loss, self.loss_type, y_pred, y_true, entry_mask)

        for key in ("local_loss", "global_loss", "combined_loss"):
            # `val` never logged combined_loss; preserved to keep this refactor number-identical.
            if key == "combined_loss" and mode == "val":
                continue
            if separate_losses.get(key) is not None:
                self._log_scalar(f"{mode}_{key}", separate_losses[key], sync_dist)

        # Metrics are computed from the combined predictions, as before.
        loss = self._compute_and_log_metrics(y_pred, y_true, mode, metrics, attn=attn, entry_mask=entry_mask)

        if separate_losses.get("kl_loss") is not None:
            kl_loss = separate_losses["kl_loss"]
            # KL annealing/weighting (beta): a fixed weight, or drive it from self.current_epoch.
            kl_weight = getattr(self.hparams, "kl_weight", 1.0)
            # sync_dist=False even under `test`, matching the previous code.
            self._log_scalar(f"{mode}_kl_loss", kl_loss, False)
            loss = loss + kl_weight * kl_loss

        loss = self._add_aux_losses(loss, out, batch, mode, sync_dist)
        assert not torch.isnan(loss), "loss is NaN"
        return loss

    def _forward_views(self, batch):
        """Run the module once per view an auxiliary term asked for, and merge them.

        How many views to run is read off the enabled terms (``CompositeAuxLoss.requires_views``),
        not from a separate config switch -- a switch would be one more thing to forget, and
        forgetting it trains the wrong objective without complaining.

        The extra passes go through the *same* ``_common_step``, so no module changes are needed:
        the views differ only by whatever stochasticity is already inside the encoder. Today that
        is dropout, which makes a two-view run the SimCSE floor -- the baseline everything richer
        has to beat. Independent corruption draws per view come with the view sampler in Stage 3.

        ``y_pred`` / ``y_true`` are taken from the first view, so the reconstruction term and
        every metric are computed exactly as in a single-view run.

        **Dropout-only views vanish under evaluation.** ``validation_step`` and ``test_step`` run
        with the module in ``eval()``, where the encoder is deterministic -- so the two views
        coincide exactly and any invariance term between them is identically zero. A
        ``val_vicreg_inv`` of 0.0 therefore means "no stochasticity in eval", not "the views
        agree", and ``val_loss`` under a VICReg-only objective reports its variance and
        covariance halves alone. Stage 3's view sampler corrupts the *input*, which does not
        depend on training mode and fixes this.
        """
        out = self.module._common_step(batch, self.prediction_task, self.prediction_level)
        n_views = self.aux_losses.requires_views
        if n_views <= 1:
            return out

        extra = [
            self.module._common_step(batch, self.prediction_task, self.prediction_level)
            for _ in range(n_views - 1)
        ]
        out.views = [out.view, *(o.view for o in extra)]
        return out

    def _add_aux_losses(self, loss, out, batch, mode: str, sync_dist: bool):
        """Add the weighted auxiliary terms to ``loss`` and log each one.

        Returns ``loss`` untouched -- the same tensor object, not an equal one -- when no term is
        enabled, which is the default. That is deliberate: an unconditional ``loss + 0`` would be
        numerically identical but would still make every existing run take a different code path,
        and the point of this scaffolding is that it is inert until something is switched on.

        Terms are logged *unweighted*, under ``<mode>_<name>``, because the unweighted value is
        what stays comparable across runs that weight the term differently. The weighted sum is
        logged separately as ``<mode>_aux_total``.
        """
        if not self.aux_losses:
            return loss

        aux_total, reported = self.aux_losses(out, batch)
        for name, value in reported.items():
            self._log_scalar(f"{mode}_{name}", value, sync_dist)
        self._log_scalar(f"{mode}_aux_total", aux_total, sync_dist)
        return loss + aux_total

    def training_step(self, batch):
        """Training step for the model.

        Returns
        -------
            loss: torch.nn.Module
        """
        return self._step(batch, "train")

    def validation_step(self, batch):
        """Validation step for the model."""
        return self._step(batch, "val")

    def test_step(self, batch):
        """Test step for the model."""
        return self._step(batch, "test")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        params = []
        # Includes any projection/expander heads: the auxiliary terms are a submodule of
        # `self.module`, so collecting them separately here would add the same parameters twice.
        params.extend(filter(lambda p: p.requires_grad, self.module.parameters()))
        # if self.model.local_component is not None:
        #     params.extend(filter(lambda p: p.requires_grad, self.module.local_component.parameters()))
        # if self.model.global_component is not None:
        #     params.extend(filter(lambda p: p.requires_grad, self.model.global_component.parameters()))
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=self.patience_in_steps, verbose=True
            )
        elif self.lr_scheduler == "CosineWarmupScheduler":
            lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.lr_warmup, max_epochs=self.lr_max_epochs)
        elif self.lr_scheduler is None:
            lr_scheduler = None
        else:
            raise ValueError(
                f"Invalid lr_scheduler: {self.lr_scheduler}. Must be either 'None', 'ReduceLROnPlateau' or 'CosineWarmupScheduler'."
            )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor": self.monitor_metric}]
