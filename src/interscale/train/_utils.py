import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback

from interscale.tl.masking import block_masking_on

#: wandb section of the efficiency scalars (it splits panels on the text before the first "/").
EFFICIENCY_PREFIX = "efficiency"


def redraws_masks(cfg) -> bool:
    """Whether the training masks are redrawn every epoch: any Bernoulli rate or spatial blocks.

    Blocks with a rate of 0 used to fall through (the check read the rate alone), leaving one fixed
    set of discs for the whole run.
    """
    return cfg.dataset.mask_percentage > 0 or block_masking_on(cfg)


class NodeMaskResampleCallback(Callback):
    """Redraws the training node mask at the start of every epoch.

    Without this, `GraphAnnDataModule` only ever samples one fixed set of masked nodes for the
    whole training run (see `GraphAnnDataModule.resample_train_mask`), so nodes outside that
    initial sample never receive direct supervision. Validation/test masks are left untouched so
    monitored metrics (e.g. `val_loss`) stay comparable across epochs.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.resample_train_mask()


class EfficiencyMonitor(Callback):
    """What a training epoch costs: throughput and peak GPU memory, for comparing attention backends.

    Logged at the end of every training epoch, under ``efficiency/``:

    - ``train_steps_per_s`` and ``train_cells_per_s``: training steps (forward, backward, optimizer
      step) and graph nodes per second of step time. Data loading between steps is left out, so
      these measure the model.
    - ``train_epoch_s``: from the epoch's start to its last training step, data loading included and
      validation (and the probes it runs) left out.
    - ``peak_gpu_mem_gib``: the most memory allocated on the module's GPU during the epoch's training
      steps. Read at the end of each step: Lightning runs validation -- and the probes with it --
      before the epoch-end hook, so a peak read there would be theirs. Absent on CPU.

    Every boundary synchronises the GPU, so a step's queued work is counted in that step; the price
    is that the CPU no longer runs ahead between steps. Built only when ``optim.log_efficiency`` is
    set (:func:`build_efficiency_monitor`).
    """

    def __init__(self, clock=time.perf_counter):
        super().__init__()
        self._clock = clock
        self._epoch_start = self._step_start = self._last_step_end = None
        self._step_seconds, self._steps, self._cells = 0.0, 0, 0
        self._peak_bytes = 0

    @staticmethod
    def _cuda(pl_module):
        device = pl_module.device
        return device if device.type == "cuda" else None

    def _now(self, pl_module) -> float:
        device = self._cuda(pl_module)
        if device is not None:
            torch.cuda.synchronize(device)
        return self._clock()

    def on_train_epoch_start(self, trainer, pl_module):
        self._step_start = self._last_step_end = None
        self._step_seconds, self._steps, self._cells = 0.0, 0, 0
        self._peak_bytes = 0
        device = self._cuda(pl_module)
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
        self._epoch_start = self._now(pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_start = self._now(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._step_start is None:
            return
        self._last_step_end = self._now(pl_module)
        self._step_seconds += self._last_step_end - self._step_start
        self._steps += 1
        self._cells += int(getattr(batch, "num_nodes", 0) or 0)
        self._step_start = None
        device = self._cuda(pl_module)
        if device is not None:
            self._peak_bytes = max(self._peak_bytes, torch.cuda.max_memory_allocated(device))

    def epoch_values(self, pl_module) -> dict[str, float]:
        """This epoch's scalars, by logged name; empty before a training step has finished."""
        if self._epoch_start is None or self._last_step_end is None or self._step_seconds <= 0:
            return {}
        values = {
            f"{EFFICIENCY_PREFIX}/train_steps_per_s": self._steps / self._step_seconds,
            f"{EFFICIENCY_PREFIX}/train_cells_per_s": self._cells / self._step_seconds,
            f"{EFFICIENCY_PREFIX}/train_epoch_s": self._last_step_end - self._epoch_start,
        }
        if self._cuda(pl_module) is not None:
            values[f"{EFFICIENCY_PREFIX}/peak_gpu_mem_gib"] = self._peak_bytes / 2**30
        return values

    def on_train_epoch_end(self, trainer, pl_module):
        for name, value in self.epoch_values(pl_module).items():
            pl_module.log(name, value, on_step=False, on_epoch=True, batch_size=1)


def build_efficiency_monitor(cfg):
    """An :class:`EfficiencyMonitor`, or None unless ``optim.log_efficiency`` is set."""
    return EfficiencyMonitor() if cfg.optim.get("log_efficiency", False) else None


def train_gene_means(data_list) -> torch.Tensor:
    """``[G]``: each gene's mean over every spot of the training graphs, for ``optim.center_targets``.

    Read from ``data.x``, which the masking never touches (``apply_mask`` corrupts a clone).
    """
    return torch.cat([data.x.float() for data in data_list]).mean(0)


class MetricsHistory(Callback):
    def __init__(self):
        super().__init__()
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # Convert tensor values to float
        epoch_dict = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        epoch_dict["epoch"] = trainer.current_epoch
        self.history.append(epoch_dict)

    def on_test_epoch_end(self, trainer, pl_module):
        print("on_test_epoch_end", trainer.current_epoch)
        metrics = trainer.callback_metrics
        # Convert tensor values to float
        epoch_dict = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
        epoch_dict["epoch"] = trainer.current_epoch

        # Record metrics every 10 epochs or at the end of training
        if trainer.current_epoch % 5 == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            self.history.append(epoch_dict)

    def plot_history(self, subset_term=None):
        history = pd.DataFrame(self.history)
        if subset_term is not None:
            history = history[[c for c in history.columns if subset_term in c]]
        ax = history[[c for c in history.columns if "train" in c]].plot()
        plt.gca().set_prop_cycle(None)
        history[[c for c in history.columns if "val" in c]].plot(style="--", ax=ax)
        # plt.gca().set_prop_cycle(None)
        # history[[c for c in history.columns if 'test' in c]].plot(style=':', ax=ax)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()
