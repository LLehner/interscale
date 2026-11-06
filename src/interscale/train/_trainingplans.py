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

from ._schedule import DEFAULT_LOSS_WEIGHTS, frozen_modules, resolve_loss_weights
from .aux_losses import CompositeAuxLoss
from .losses import BalancedPearsonCorrelationLoss, SCE_EntropyATT_Loss, SCELoss, scaled_cosine_error


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


class RunningScaledCosineError(torchmetrics.Metric):
    """Mean per-cell scaled cosine error ``(1 - cos) ** gamma`` (GraphMAE), from a running sum.

    The metric twin of ``SCELoss``: with the same ``gamma`` and the same cells it is the same
    number, so a run trained on another criterion can still be read on the SCE scale. Two scalars
    of state, for the reason ``RunningCosineSimilarity`` gives.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, gamma: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate the summed per-cell scaled cosine error and the number of cells."""
        sce = scaled_cosine_error(preds, target, self.gamma)
        self.total = self.total + sce.sum()
        self.count = self.count + sce.numel()

    def compute(self) -> torch.Tensor:
        """Mean per-cell scaled cosine error over everything seen since the last reset."""
        return self.total / self.count


class MeanPearsonCorrCoef(torchmetrics.PearsonCorrCoef):
    """``PearsonCorrCoef`` over ``num_outputs`` genes, reduced to the NaN-skipping mean.

    Lightning can only log a metric object whose ``compute`` returns a scalar. A constant gene
    yields NaN rather than a number, hence ``nanmean``.
    """

    def compute(self) -> torch.Tensor:
        """Mean over genes of the per-gene correlation, skipping undefined genes."""
        return torch.nanmean(super().compute().contiguous())


class MeanConcordanceCorrCoef(torchmetrics.ConcordanceCorrCoef):
    """``ConcordanceCorrCoef`` reduced to the NaN-skipping mean over genes; see MeanPearsonCorrCoef."""

    def compute(self) -> torch.Tensor:
        """Mean over genes of the per-gene concordance, skipping undefined genes."""
        return torch.nanmean(super().compute().contiguous())


class SingleClassF1Score(torchmetrics.classification.MulticlassF1Score):
    """F1 of one class, so that each class can be logged as its own scalar metric object."""

    def __init__(self, num_classes: int, class_idx: int, **kwargs):
        super().__init__(num_classes=num_classes, average=None, **kwargs)
        self.class_idx = class_idx

    def compute(self) -> torch.Tensor:
        """F1 of ``class_idx`` over everything seen since the last reset."""
        return super().compute()[self.class_idx]


_MASKED_MOMENTS = ("n_gene", "sum_p", "sum_t", "sum_pp", "sum_tt", "sum_pt", "cos_total", "sce_total", "n_rows")


def _masked_moments(
    y_pred: torch.Tensor, y_true: torch.Tensor, entry_mask: torch.Tensor, sce_gamma: float = 3.0
) -> dict[str, torch.Tensor]:
    """Additive sufficient statistics for :func:`masked_regression_metrics`.

    Every entry that is not masked is multiplied by zero before any sum is taken, and zeros
    contribute nothing to a sum, so each moment is exactly the moment over the masked entries.
    Every quantity is a plain sum, so moments of several batches add up to the moments of their
    union -- which is what lets :class:`MaskedRegressionMetric` report an exact epoch-level value.
    """
    m = entry_mask.to(y_pred.dtype)
    p_ = y_pred * m
    t_ = y_true * m
    # Per-cell cosine over that cell's masked genes: the zeroed entries drop out of both the dot
    # product and the two norms, so this is the cosine on the masked coordinates.
    # Averaged over the cells that HAVE a masked gene only. A cell with none has no masked
    # coordinates, and its all-zero row would enter the mean as a cosine of 0 -- under cell masking
    # that is every unmasked cell, which scaled the reported cosine by the fraction of cells masked.
    scored = entry_mask.any(dim=1)
    cosine = nn.functional.cosine_similarity(p_[scored], t_[scored], dim=1)
    sce = scaled_cosine_error(p_[scored], t_[scored], sce_gamma)
    moments = {
        "n_gene": m.sum(dim=0),  # [G] masked cells per gene
        "sum_p": p_.sum(dim=0),
        "sum_t": t_.sum(dim=0),
        "sum_pp": (p_**2).sum(dim=0),
        "sum_tt": (t_**2).sum(dim=0),
        "sum_pt": (p_ * t_).sum(dim=0),
        "cos_total": cosine.sum(),
        "sce_total": sce.sum(),
        "n_rows": torch.tensor(float(cosine.numel()), device=y_pred.device),
    }
    # float64: the variances below are differences of sums, which loses precision in float32
    # once the sums run over a whole epoch of cells.
    return {k: v.double() for k, v in moments.items()}


def _masked_metrics_from_moments(moments: dict[str, torch.Tensor], eps: float = 1e-8) -> dict[str, torch.Tensor]:
    """The metrics of :func:`masked_regression_metrics`, from (possibly accumulated) moments."""
    n_gene = moments["n_gene"]
    ng = n_gene.clamp(min=1)

    # Per-gene first and second moments over that gene's masked cells.
    mean_p = moments["sum_p"] / ng
    mean_t = moments["sum_t"] / ng
    var_p = moments["sum_pp"] / ng - mean_p**2
    var_t = moments["sum_tt"] / ng - mean_t**2
    cov = moments["sum_pt"] / ng - mean_p * mean_t

    # Squared error per gene, sum((p - t)^2) expanded into the accumulated sums.
    ss_res = moments["sum_pp"] - 2 * moments["sum_pt"] + moments["sum_tt"]
    mse = ss_res.sum() / n_gene.sum().clamp(min=1)

    # A gene with fewer than two masked cells, or with no spread in either vector, has no
    # correlation defined; NaN it out and let nanmean skip it, exactly as the unmasked path
    # already does for constant genes.
    nan = torch.tensor(float("nan"), device=n_gene.device, dtype=n_gene.dtype)
    usable = (n_gene >= 2) & (var_t > eps)

    pearson = torch.where(usable & (var_p > eps), cov / (var_p.clamp(min=eps) * var_t.clamp(min=eps)).sqrt(), nan)
    concordance = torch.where(usable, 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + eps), nan)

    # R2 per gene, then uniform average -- the same reduction torchmetrics'
    # R2Score(multioutput="uniform_average") applies.
    ss_tot = var_t * ng
    r2 = torch.where(usable, 1 - ss_res / ss_tot.clamp(min=eps), nan)

    return {
        "mse": mse,
        "r2": torch.nanmean(r2),
        "pearson_corr": torch.nanmean(pearson),
        "concordance_corr": torch.nanmean(concordance),
        "cosine_similarity": moments["cos_total"] / moments["n_rows"].clamp(min=1),
        "scaled_cosine_error": moments["sce_total"] / moments["n_rows"].clamp(min=1),
    }


class MaskedRegressionMetric(torchmetrics.Metric):
    """One of the :func:`masked_regression_metrics` values, accumulated exactly over an epoch.

    Holds the per-gene moments of :func:`_masked_moments` as running sums (``O(n_genes)`` state),
    so ``compute`` gives the value over every masked entry seen since the last reset -- not a
    mean of per-batch values. One instance per metric name, because Lightning can only log a
    metric object that computes a scalar.

    Parameters
    ----------
    name
        One of ``mse``, ``r2``, ``pearson_corr``, ``concordance_corr``, ``cosine_similarity``,
        ``scaled_cosine_error``.
    num_outputs
        Number of genes ``G``.
    sce_gamma
        The exponent of the scaled cosine error.
    """

    is_differentiable = False
    full_state_update = False

    def __init__(self, name: str, num_outputs: int, sce_gamma: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.sce_gamma = float(sce_gamma)
        for key in _MASKED_MOMENTS:
            shape = (num_outputs,) if key.startswith(("n_gene", "sum_")) else ()
            self.add_state(key, default=torch.zeros(shape, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, entry_mask: torch.Tensor) -> None:
        """Add this batch's masked-entry moments to the running sums."""
        for key, value in _masked_moments(preds, target, entry_mask, self.sce_gamma).items():
            setattr(self, key, getattr(self, key) + value)

    def compute(self) -> torch.Tensor:
        """The metric over every masked entry seen since the last reset."""
        moments = {key: getattr(self, key) for key in _MASKED_MOMENTS}
        return _masked_metrics_from_moments(moments)[self.name].float()


def masked_regression_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, entry_mask: torch.Tensor, eps: float = 1e-8, sce_gamma: float = 3.0
) -> dict[str, torch.Tensor]:
    """The regression metrics of ``_setup_regression_metrics``, restricted to the masked entries.

    Under gene masking the scored rows are mostly entries the model was *given*. Feeding the full
    rows to the ``MetricCollection`` would score the identity map on those and inflate every
    number -- including ``val_r2``, which drives early stopping and checkpoint selection. There
    is no way to express "these entries only" to a torchmetrics per-output metric (the surviving
    entries are ragged across genes), so the same quantities are computed here from masked sums.

    Every entry that is not masked is multiplied by zero before any sum is taken, and zeros
    contribute nothing to a sum, so each moment is exactly the moment over the masked entries --
    no approximation. This is the single-batch form; training logs the epoch-level equivalent,
    :class:`MaskedRegressionMetric`, built on the same two helpers.

    Parameters
    ----------
    y_pred, y_true
        ``[N, G]`` predictions and targets for the scored rows.
    entry_mask
        ``[N, G]`` boolean marking the masked entries.
    eps
        Guard for degenerate (zero-variance) genes.
    sce_gamma
        The exponent of the scaled cosine error, as for ``SCELoss``.

    Returns
    -------
    dict
        Unprefixed metric names mapped to scalar tensors, matching the keys that
        ``_setup_regression_metrics`` produces: ``mse``, ``r2`` (per-gene, uniform average),
        ``pearson_corr``, ``concordance_corr``, and per cell ``cosine_similarity`` and
        ``scaled_cosine_error``.
    """
    metrics = _masked_metrics_from_moments(_masked_moments(y_pred, y_true, entry_mask, sce_gamma), eps=eps)
    return {k: v.to(y_pred.dtype) for k, v in metrics.items()}


#: ``optim.loss`` values that mean "no reconstruction criterion at all". The objective is then
#: whatever ``optim.aux_loss_weights`` provides -- VICReg on its own, say. Config validation
#: rejects the combination of no criterion and no auxiliary term, which would train on nothing.
NO_LOSS = (None, "none", "None")

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
        sce_gamma: float = 3.0,
        target_center: torch.Tensor | None = None,
        loss_weights: dict | None = None,
        frozen: tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__()
        self.module = module
        # Stage C of .claude/global_plan.md. `loss_weights`: w_local * L_local + w_global * L_global
        # for a dual decoder; None (or the default 0.5 / 0.5) keeps the stacked loss of every run before,
        # which is that mean, computed as it always was. `frozen`: components kept in eval mode at every
        # training step; making them untrainable (requires_grad) is `train._schedule.freezing`'s job.
        self.loss_weights = resolve_loss_weights(loss_weights)
        if self.loss_weights is not None:
            if not hasattr(module, "compute_separate_losses"):
                raise ValueError("loss_weights weight the two heads of a dual decoder; this module has one head.")
            if loss in NO_LOSS or loss == "SCE_EntropyATT_Loss":
                raise ValueError(f"loss_weights need each head's own loss, which loss {loss!r} does not give.")
        self.frozen = tuple(frozen)
        self._frozen_modules = frozen_modules(module, self.frozen)
        # `optim.center_targets`: each gene's training mean, subtracted from predictions and targets
        # before the reconstruction loss (never before the metrics). None leaves the loss as it was.
        self.register_buffer("target_center", None if target_center is None else torch.as_tensor(target_center).float())
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
        # The exponent of the scaled cosine error: used by SCELoss / SCE_EntropyATT_Loss when one
        # is the criterion, and by the `scaled_cosine_error` metrics whatever the criterion is.
        self.sce_gamma = float(sce_gamma)
        if self.prediction_task == "regression":
            if self.cross_corr == "gene":
                print("cross-gene per cell correlation metrics")
                self.AXIS = 1  # selecting rows / cells
            elif self.cross_corr == "cell":
                print("cross-cell per gene correlation metrics")
                self.AXIS = 0  # selecting columns / genes

        # setup metrics and loss
        #
        # Metrics are torchmetrics objects that are only UPDATED per batch and logged as objects,
        # so Lightning computes each one once over the whole epoch and resets it. Logging the
        # per-batch values instead (what this did before) made Lightning average them: for a
        # non-decomposable metric such as macro-F1, R2 or a correlation that is a different number
        # from the epoch value, and it depended on batch size and on how the (unshuffled) val
        # graphs fell into batches -- a batch holding one class turns macro-F1 into accuracy.
        masked_metrics = None
        if "classification" in self.prediction_task:
            metrics = self._setup_classification_metrics(self.module.n_output, self.class_labels)
            self.loss = self._setup_classification_loss(self.loss_type, self.class_weights)
            # Must name a metric that is actually logged -- this is handed to Lightning as the
            # LR-scheduler monitor. "val_f1" never existed; only val_f1_micro/macro/<class> do.
            self.monitor_metric = "val_f1_macro"
        elif "regression" in self.prediction_task:
            metrics = self._setup_regression_metrics(self.module.n_output, self.sce_gamma)
            masked_metrics = self._setup_masked_regression_metrics(self.module.n_output, self.sce_gamma)
            self.loss = self._setup_regression_loss(self.loss_type)
            self.monitor_metric = "val_r2"
        else:
            raise ValueError("Prediction task must define 'classification' or 'regression'.")

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        # Only updated (and therefore only logged) on batches that carry an entry_mask. Keyed
        # "<mode>_masked", not "<mode>": a ModuleDict key "train" collides with nn.Module.train().
        self.masked_metrics = (
            None
            if masked_metrics is None
            else nn.ModuleDict(
                {f"{mode}_masked": masked_metrics.clone(prefix=f"{mode}_masked_") for mode in ("train", "val", "test")}
            )
        )

    @staticmethod
    def _setup_classification_loss(
        loss: Literal["CrossEntropy", "WeightedCE"], class_weights: torch.Tensor | None = None
    ):
        """Setup loss function based on prediction task and configuration. None disables it."""
        if loss in NO_LOSS:
            return None
        assert loss in CLASSIFICATION_LOSSES, "Classification must be run with CrossEntropy or WeightedCE loss."
        if loss == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif loss == "WeightedCE":
            assert class_weights is not None, "Class weights must be provided for WeightedCE loss."
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch tensor"
            # .float() guards against a float64 weight buffer meeting float32 logits.
            return nn.CrossEntropyLoss(weight=class_weights.float())

    def _setup_regression_loss(self, loss: Literal[REGRESSION_LOSSES]):
        """Setup loss function based on prediction task and configuration. None disables it."""
        if loss in NO_LOSS:
            return None
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
            return SCELoss(gamma=self.sce_gamma)
        elif loss == "SCE_EntropyATT_Loss":
            return SCE_EntropyATT_Loss(gamma=self.sce_gamma)

    @staticmethod
    def _setup_classification_metrics(num_outputs: int, class_labels: list[str] | None = None):
        metrics = {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs),
            "f1_micro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="micro"),
            "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=num_outputs, average="macro"),
        }
        # One scalar metric per class (logged as <mode>_f1_<label>, the names used so far). A
        # ModuleDict key may not contain ".", so that one character is replaced.
        labels = class_labels if class_labels is not None else [str(i) for i in range(num_outputs)]
        for idx, label in enumerate(labels):
            metrics[f"f1_{str(label).replace('.', '_')}"] = SingleClassF1Score(num_classes=num_outputs, class_idx=idx)
        return MetricCollection(metrics)

    @staticmethod
    def _setup_regression_metrics(num_outputs: int, sce_gamma: float = 3.0):
        return MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "r2": torchmetrics.R2Score(multioutput="uniform_average"),
                "pearson_corr": MeanPearsonCorrCoef(num_outputs=num_outputs),
                # Pearson is invariant to any per-gene affine rescaling of the predictions, so a
                # model whose outputs are (say) 11x too spread out still scores a high r while
                # its R2 goes to -113. Writing predictions as k times the true sd with offset d,
                # R2 = 2*r*k - k^2 - d^2/sigma^2, maximised at k = r -- so R2 <= r^2, and the gap
                # between them is purely calibration. Concordance correlation folds that penalty
                # back in, which makes it the metric to select on when both the co-variation
                # structure AND the expression scale have to be usable.
                "concordance_corr": MeanConcordanceCorrCoef(num_outputs=num_outputs),
                # Not torchmetrics.CosineSimilarity: see RunningCosineSimilarity for why its
                # list state cannot be used on a dataset this size. Same value, O(1) memory.
                "cosine_similarity": RunningCosineSimilarity(),
                # GraphMAE's scaled cosine error with the configured gamma: the SCELoss value on
                # the same cells, tracked whatever the criterion. gamma = 1 is 1 - cosine.
                "scaled_cosine_error": RunningScaledCosineError(sce_gamma),
            }
        )

    @staticmethod
    def _setup_masked_regression_metrics(num_outputs: int, sce_gamma: float = 3.0):
        # compute_groups=False: the five share identical state, but letting MetricCollection merge
        # them would be an optimisation whose correctness depends on its state-equality check.
        return MetricCollection(
            {
                name: MaskedRegressionMetric(name, num_outputs, sce_gamma)
                for name in (
                    "mse",
                    "r2",
                    "pearson_corr",
                    "concordance_corr",
                    "cosine_similarity",
                    "scaled_cosine_error",
                )
            },
            compute_groups=False,
        )

    def _classification_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the classification loss and add this batch to ``metrics`` (no per-batch compute)."""
        ## TODO: Currently mask_idx is applied in module._common_step. Maybe move to here?
        # if mask_idx is not None:
        #     y_pred = y_pred[mask_idx]
        #     y_true = y_true[mask_idx]

        loss = self._zero(y_pred) if self.loss is None else self.loss(y_pred, y_true)
        metrics.update(y_pred.argmax(dim=1), y_true.argmax(dim=1))
        return loss

    def _regression_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mode: str,
        metrics: MetricCollection,
        mask_idx: torch.Tensor | None = None,
        attn: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
        target_center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the regression loss and add this batch to the metrics (no per-batch compute).

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
        # What the loss sees: the values minus each gene's training mean under optim.center_targets.
        y_pred_raw, y_true_raw = y_pred, y_true
        y_pred, y_true = self.centred(y_pred, y_true, target_center)
        if self.loss is None:
            # No reconstruction criterion: the objective comes entirely from the auxiliary terms.
            # Metrics are still computed and logged -- they say what the representation can do,
            # which is exactly the question when nothing is being reconstructed.
            loss = self._zero(y_pred)
        elif self.loss_type == "SCE_EntropyATT_Loss":
            # Takes attention as a third argument, so it cannot go through masked_loss. Zeroing
            # the unmasked entries restricts its row-wise cosine to the masked coordinates,
            # which is what the row-structured branch of masked_loss does too.
            if entry_mask is None:
                loss = self.loss(y_pred, y_true, attn)
            else:
                # Only the cells with a masked entry: a cell without one would enter the cosine
                # as an all-zero row, i.e. as an error of 1 that no prediction can change.
                rows = entry_mask.any(dim=1)
                m = entry_mask[rows].to(y_pred.dtype)
                loss = self.loss(y_pred[rows] * m, y_true[rows] * m, attn)
        else:
            loss = masked_loss(self.loss, self.loss_type, y_pred, y_true, entry_mask)
        y_pred, y_true = y_pred_raw, y_true_raw

        # Primary metrics: every cell the model produced, masked or not. The per-gene correlations
        # are reduced with nanmean inside MeanPearsonCorrCoef / MeanConcordanceCorrCoef.
        metrics.update(y_pred, y_true)

        # Held-out companions on the masked entries only. Kept because the all-cell numbers above
        # include entries the model was given as input, which it can partly copy -- so they are
        # the reconstruction score for the tissue, not evidence the model generalises. These are.
        if entry_mask is not None:
            self.masked_metrics[f"{mode}_masked"].update(y_pred, y_true, entry_mask)

        return loss

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        """A zero on the right device and dtype, so `loss + aux` needs no special case."""
        return torch.zeros((), device=reference.device, dtype=reference.dtype)

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
        extra_loss: torch.Tensor | None = None,
        loss_override: torch.Tensor | None = None,
        target_center: torch.Tensor | None = None,
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

        if "classification" in self.prediction_task:
            loss = self._classification_metrics(y_pred, y_true, mode, metrics)
        elif "regression" in self.prediction_task:
            loss = self._regression_metrics(
                y_pred, y_true, mode, metrics, attn=attn, entry_mask=entry_mask, target_center=target_center
            )
            if entry_mask is not None:
                self.log_dict(self.masked_metrics[f"{mode}_masked"], on_step=False, on_epoch=True)

        if loss_override is not None:
            # The weighted sum of the two heads' losses (`loss_weights`): what this run optimises,
            # so what `<mode>_loss` -- monitored by checkpointing and early stopping -- must report.
            loss = loss_override

        if extra_loss is not None:
            # `<mode>_loss` must be the quantity actually optimised, not the reconstruction half
            # of it. It is what EarlyStopping and ModelCheckpoint monitor for a regression run,
            # so under `optim.loss: none` -- where the reconstruction half is a constant zero --
            # logging that half would give a flat curve and stop every run at `patience`.
            loss = loss + extra_loss

        # Metric OBJECTS: Lightning calls compute() once at epoch end over every batch, then
        # reset(). Distributed sync is handled by torchmetrics itself.
        self.log_dict(metrics, on_step=False, on_epoch=True)

        # The loss is a per-batch mean, so its epoch value IS a weighted mean of batches -- weighted
        # by the batch's real size (graphs, or cells for node-level tasks), not the configured
        # batch_size, which over-weighted a short last batch.
        sync_dist = mode == "test"
        self.log(
            f"{mode}_loss", loss, batch_size=int(y_true.shape[0]), on_step=False, on_epoch=True, sync_dist=sync_dist
        )

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

        # Computed before the metrics so that `<mode>_loss` can be logged as the full objective.
        aux_total = self._aux_losses(out, batch, mode, sync_dist)

        # Modules that decode twice (DualDecoderCombinedModule) report each half separately --
        # but there are no halves to report when no reconstruction criterion is configured.
        if self.loss is None or not hasattr(self.module, "compute_separate_losses"):
            return self._compute_and_log_metrics(
                y_pred,
                y_true,
                mode,
                metrics,
                attn=attn,
                entry_mask=entry_mask,
                extra_loss=aux_total,
                target_center=out.target_center,
            )

        # `optim.regime_loss`: each head's loss on the hidden cells of its regime (`tl.depth`). The metrics keep
        # every hidden entry (`entry_mask`); only the loss is cut.
        regimes = out.loss_mask is not None
        loss_mask = out.loss_mask if regimes else entry_mask
        separate_losses = self.module.compute_separate_losses(
            self.loss, self.loss_type, *self.centred(y_pred, y_true), loss_mask
        )
        # The default stacked loss is one mean over the rows of both heads, which under regimes would weigh a
        # head by how many cells its regime has: weigh the heads instead, by `loss_weights` (0.5 / 0.5).
        weights = self.loss_weights if self.loss_weights is not None or not regimes else DEFAULT_LOSS_WEIGHTS
        weighted = None
        if weights is not None:
            if separate_losses.get("local_loss") is None or separate_losses.get("global_loss") is None:
                raise RuntimeError("loss_weights need both heads' losses; this step produced one (graph level?).")
            weighted = (
                weights["local"] * separate_losses["local_loss"] + weights["global"] * separate_losses["global_loss"]
            )
        if regimes:
            half = len(loss_mask) // 2
            for name, rows in (("local", loss_mask[:half]), ("global", loss_mask[half:])):
                self._log_scalar(f"{mode}_regime_cells_{name}", rows.any(dim=1).sum().float(), sync_dist)

        for key in ("local_loss", "global_loss", "combined_loss"):
            # `val` never logged combined_loss; preserved to keep this refactor number-identical.
            if key == "combined_loss" and mode == "val":
                continue
            if separate_losses.get(key) is not None:
                self._log_scalar(f"{mode}_{key}", separate_losses[key], sync_dist)

        self._log_head_sce(y_pred, y_true, entry_mask, mode, sync_dist)

        # Metrics are computed from the combined predictions, as before.
        loss = self._compute_and_log_metrics(
            y_pred,
            y_true,
            mode,
            metrics,
            attn=attn,
            entry_mask=entry_mask,
            extra_loss=aux_total,
            loss_override=weighted,
        )

        if separate_losses.get("kl_loss") is not None:
            kl_loss = separate_losses["kl_loss"]
            # KL annealing/weighting (beta): a fixed weight, or drive it from self.current_epoch.
            kl_weight = getattr(self.hparams, "kl_weight", 1.0)
            # sync_dist=False even under `test`, matching the previous code.
            self._log_scalar(f"{mode}_kl_loss", kl_loss, False)
            loss = loss + kl_weight * kl_loss

        assert not torch.isnan(loss), "loss is NaN"
        return loss

    def _log_head_sce(self, y_pred, y_true, entry_mask, mode: str, sync_dist: bool) -> None:
        """``<mode>_masked_scaled_cosine_error_{local,global}``: each head's SCE on the masked entries.

        A dual decoder's step output stacks the local and the global half, so the combined
        ``<mode>_masked_scaled_cosine_error`` is their mean: with one head frozen at a random
        initialisation (SCE ~ 1) it reads about half-way between the two and says nothing about
        the head that trains. These two are each head's own score, raw (not centred).
        """
        if entry_mask is None:
            return
        n = len(y_pred) // 2
        for name, rows in (("local", slice(0, n)), ("global", slice(n, None))):
            mask = entry_mask[rows].to(y_pred.dtype)
            scored = mask.any(dim=1)
            if not scored.any():
                continue
            sce = scaled_cosine_error((y_pred[rows] * mask)[scored], (y_true[rows] * mask)[scored], self.sce_gamma)
            self._log_scalar(f"{mode}_masked_scaled_cosine_error_{name}", sce.mean(), sync_dist)

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
            self.module._common_step(batch, self.prediction_task, self.prediction_level) for _ in range(n_views - 1)
        ]
        out.views = [out.view, *(o.view for o in extra)]
        return out

    def _aux_losses(self, out, batch, mode: str, sync_dist: bool):
        """Evaluate and log the auxiliary terms; return their weighted total, or None.

        ``None`` -- not a zero -- when no term is enabled, which is the default. The caller then
        skips the addition entirely, so an unweighted run takes exactly the code path it took
        before any of this existed rather than an arithmetically-equal one.

        Terms are logged *unweighted*, under ``<mode>_<name>``, because the unweighted value is
        what stays comparable across runs that weight the term differently. The weighted sum is
        logged as ``<mode>_aux_total``, and folded into ``<mode>_loss`` by the caller.
        """
        if not self.aux_losses:
            return None

        aux_total, reported = self.aux_losses(out, batch)
        for name, value in reported.items():
            self._log_scalar(f"{mode}_{name}", value, sync_dist)
        self._log_scalar(f"{mode}_aux_total", aux_total, sync_dist)
        return aux_total

    def training_step(self, batch):
        """Training step for the model.

        Frozen components go back to eval mode first: Lightning puts the whole model into training
        mode at every epoch, and a probe pass may have done so since the last step.

        Returns
        -------
            loss: torch.nn.Module
        """
        for module in self._frozen_modules:
            module.eval()
        return self._step(batch, "train")

    def validation_step(self, batch):
        """Validation step for the model."""
        return self._step(batch, "val")

    def test_step(self, batch):
        """Test step for the model."""
        return self._step(batch, "test")

    def centred(self, y_pred: torch.Tensor, y_true: torch.Tensor, center: torch.Tensor | None = None):
        """``(y_pred, y_true)`` minus ``center`` (per cell: ``StepOutput.target_center``), else minus each gene's training mean
        under ``optim.center_targets``, else as given.
        """
        center = self.target_center if center is None else center
        if center is None:
            return y_pred, y_true
        return y_pred - center, y_true - center

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
