"""The efficiency monitor (``optim.log_efficiency``): what a training epoch costs, and nothing when off.

Throughput counts step time only (forward, backward, optimizer step), the epoch ends at its last
training step so validation and the probes are not billed to training, and on a GPU every boundary
synchronises first so a step's queued work lands in that step.
"""

import pytest

torch = pytest.importorskip("torch")

from interscale.train._utils import EFFICIENCY_PREFIX, EfficiencyMonitor, build_efficiency_monitor

STEPS = f"{EFFICIENCY_PREFIX}/train_steps_per_s"
CELLS = f"{EFFICIENCY_PREFIX}/train_cells_per_s"
EPOCH = f"{EFFICIENCY_PREFIX}/train_epoch_s"
PEAK = f"{EFFICIENCY_PREFIX}/peak_gpu_mem_gib"


class _Clock:
    """Hands out the given times in order, and fails if read once more."""

    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


class _Module:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = (value, kwargs)


class _Batch:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes


def _epoch(monitor, module, cells):
    monitor.on_train_epoch_start(None, module)
    for batch_idx, n in enumerate(cells):
        monitor.on_train_batch_start(None, module, _Batch(n), batch_idx)
        monitor.on_train_batch_end(None, module, None, _Batch(n), batch_idx)


def test_throughput_counts_step_time_and_the_epoch_ends_at_its_last_step():
    """Epoch starts at 0; steps run 1-3 (10 cells) and 4-5 (30 cells). The epoch end reads no
    clock, so whatever ran after the last step -- validation, the probes -- is not counted."""
    monitor, module = EfficiencyMonitor(clock=_Clock([0.0, 1.0, 3.0, 4.0, 5.0])), _Module()
    _epoch(monitor, module, (10, 30))
    monitor.on_train_epoch_end(None, module)

    assert {name: value for name, (value, _) in module.logged.items()} == pytest.approx(
        {STEPS: 2 / 3, CELLS: 40 / 3, EPOCH: 5.0}
    )
    assert all(kwargs == {"on_step": False, "on_epoch": True, "batch_size": 1} for _, kwargs in module.logged.values())


def test_every_epoch_starts_from_zero():
    monitor, module = EfficiencyMonitor(clock=_Clock([0.0, 1.0, 2.0, 10.0, 11.0, 15.0])), _Module()
    _epoch(monitor, module, (5,))
    _epoch(monitor, module, (8,))

    assert monitor.epoch_values(module) == pytest.approx({STEPS: 1 / 4, CELLS: 2.0, EPOCH: 5.0})


def test_an_epoch_without_a_finished_step_logs_nothing():
    monitor, module = EfficiencyMonitor(clock=_Clock([0.0])), _Module()
    monitor.on_train_epoch_start(None, module)
    monitor.on_train_epoch_end(None, module)

    assert module.logged == {}


def test_on_a_gpu_every_boundary_synchronises_and_the_peak_is_logged(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append(("reset", device)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 3 * 2**29)
    monitor, module = EfficiencyMonitor(clock=_Clock([0.0, 1.0, 2.0])), _Module("cuda:0")
    _epoch(monitor, module, (4,))

    device = torch.device("cuda:0")
    assert calls == [("reset", device), ("sync", device), ("sync", device), ("sync", device)]
    assert monitor.epoch_values(module)[PEAK] == pytest.approx(1.5)


def test_the_peak_is_the_training_steps_not_the_validation_after_them(monkeypatch):
    """Lightning validates -- probes included -- before the epoch-end hook. Here the steps peak at
    0.5 and 1.5 GiB, then validation allocates 6 GiB: the epoch's peak is 1.5."""
    allocated = [2**29, 3 * 2**29]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda device=None: allocated.pop(0) if allocated else 6 * 2**30
    )
    monitor, module = EfficiencyMonitor(clock=_Clock([0.0, 1.0, 2.0, 3.0, 4.0])), _Module("cuda:0")
    _epoch(monitor, module, (4, 4))
    monitor.on_train_epoch_end(None, module)

    assert module.logged[PEAK][0] == pytest.approx(1.5)


def test_it_is_built_only_when_asked_for():
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    assert build_efficiency_monitor(cfg) is None
    cfg.optim.log_efficiency = True
    assert isinstance(build_efficiency_monitor(cfg), EfficiencyMonitor)


def test_lightning_takes_its_scalars_from_the_epoch_end_hook():
    """Through a real trainer on a toy module, so the hook may log and the batch carries cells."""
    import lightning.pytorch as pl
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    class _Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 1)

        def training_step(self, batch, batch_idx):
            return self.lin(batch.x).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    gen = torch.Generator().manual_seed(0)
    graphs = [Data(x=torch.randn(n, 3, generator=gen)) for n in (5, 7, 6, 4)]
    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[EfficiencyMonitor()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    trainer.fit(_Toy(), DataLoader(graphs, batch_size=2))

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith(EFFICIENCY_PREFIX)}
    assert set(metrics) == {STEPS, CELLS, EPOCH}
    assert all(value > 0 for value in metrics.values())
