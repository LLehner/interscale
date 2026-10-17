"""Staged training of the local and global components (`train._schedule`; global_plan.md, Stage C).

Pinned here:
- **Schedules:** presets expand to the phases the plan describes, the epoch budget is split
  exactly, and the config refuses schedules that cannot run.
- **Loss weights:** the default keeps the stacked loss of every run before; other weights give
  `w_local * L_local + w_global * L_global`, which is also the logged `<mode>_loss`, and a head with
  weight 0 sends no gradient.
- **Freezing:** frozen components get no gradient, are not in the optimizer, stay in eval mode
  through every training step, and get their own `requires_grad` flags back afterwards.
- **The driver:** phases get their epochs, learning rates, warm-ups and cosines. An end-to-end run
  trains each component only in its phases, carries the weights over, redraws masks in every
  phase, and checkpoints and saves once, at the end.
- **The rest:** the W&B step continues across phases, a trained LocalModel can be copied in as
  phase 1, and checkpoint names tell schedules apart.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.train._schedule import (
    Phase,
    freezing,
    phase_epochs,
    phase_warmup,
    schedule_phases,
    validate_schedule,
)

G = 8


def _cfg(**optim):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder = True
    cfg.optim.loss = "SCELoss"
    for key, value in optim.items():
        cfg.optim[key] = value
    return cfg


# --------------------------------------------------------------------------- schedules


def test_the_presets_expand_to_the_planned_phases():
    l2g = schedule_phases(_cfg(schedule="local_then_global"))
    assert [(p.share, p.frozen, p.loss_weights) for p in l2g] == [
        (0.5, ("global",), {"local": 1.0, "global": 0.0}),
        (0.5, ("local",), {"local": 0.0, "global": 1.0}),
    ]
    ft = schedule_phases(_cfg(schedule="local_then_global_ft"))
    assert [p.share for p in ft] == [0.4, 0.4, 0.2]
    assert ft[2].frozen == () and ft[2].loss_weights is None and ft[2].lr_scale == pytest.approx(0.1)

    alt = schedule_phases(_cfg(schedule="alternating", schedule_cycles=3))
    assert [p.frozen for p in alt] == [("global",), ("local",)] * 3  # ends on a global phase
    assert all(p.loss_weights == {"local": 1.0, "global": 0.0} for p in alt[::2])
    coupled = schedule_phases(_cfg(schedule="alternating_coupled", schedule_cycles=3))
    assert all(p.loss_weights is None for p in coupled[::2])  # the GCN also learns from the global loss
    assert all(p.loss_weights == {"local": 0.0, "global": 1.0} for p in coupled[1::2])


def test_joint_is_one_phase_with_the_top_level_settings_and_custom_reads_its_list():
    assert schedule_phases(_cfg()) == [Phase(1.0)]
    cfg = _cfg(frozen=["local"])
    cfg.optim.loss_weights["global"] = 1.0
    assert schedule_phases(cfg) == [Phase(1.0, ("local",), {"local": 0.5, "global": 1.0})]

    phases = [
        {"share": 0.25, "frozen": ["global"], "loss_weights": {"local": 1.0, "global": 0.0}},
        {"share": 0.75, "lr_scale": 0.5},
    ]
    assert schedule_phases(_cfg(schedule="custom", schedule_phases=phases)) == [
        Phase(0.25, ("global",), {"local": 1.0, "global": 0.0}),
        Phase(0.75, (), None, 0.5),
    ]


@pytest.mark.parametrize(
    ("shares", "total", "expected"),
    [((0.5, 0.5), 200, [100, 100]), ((0.4, 0.4, 0.2), 200, [80, 80, 40]), ((1 / 3,) * 3, 10, [4, 3, 3])],
)
def test_the_epoch_budget_is_split_exactly(shares, total, expected):
    epochs = phase_epochs([Phase(s) for s in shares], total)
    assert epochs == expected and sum(epochs) == total


def test_every_phase_gets_an_epoch_and_a_warmup():
    assert phase_epochs([Phase(0.98), Phase(0.01), Phase(0.01)], 10) == [8, 1, 1]
    assert phase_epochs([Phase(0.125)] * 8, 2) == [1] * 8  # a 2-epoch smoke run of 8 phases
    assert phase_warmup(20, 100, 200) == 10
    assert phase_warmup(20, 1, 200) == 1  # never 0: the scheduler divides by it


@pytest.mark.parametrize(
    ("optim", "match"),
    [
        ({"schedule": "staged"}, "optim.schedule"),
        ({"schedule": "custom"}, "schedule_phases"),
        ({"schedule": "custom", "schedule_phases": [{"share": 0.5}, {"share": 0.4}]}, "sum to"),
        ({"schedule": "custom", "schedule_phases": [{"share": 1.0, "epochs": 3}]}, "unknown keys"),
        (
            {"schedule": "custom", "schedule_phases": [{"share": 1.0, "frozen": ["local", "global"]}]},
            "nothing to train",
        ),
        (
            {"schedule": "custom", "schedule_phases": [{"share": 1.0, "loss_weights": {"local": -1.0, "global": 1.0}}]},
            ">= 0",
        ),
        ({"schedule": "custom", "schedule_phases": [{"share": 1.0, "loss_weights": {"local": 1.0}}]}, "exactly"),
        ({"schedule": "local_then_global", "frozen": ["local"]}, "per phase"),
        ({"schedule": "alternating", "schedule_cycles": 0}, "schedule_cycles"),
        ({"frozen": ["decoder"]}, "unknown components"),
        ({"schedule": "local_then_global", "loss": "SCE_EntropyATT_Loss"}, "each head's own loss"),
    ],
)
def test_the_config_refuses_schedules_that_cannot_run(optim, match):
    with pytest.raises(ValueError, match=match):
        validate_schedule(_cfg(**optim))


def test_a_staged_schedule_needs_the_dual_decoder_and_config_loading_checks_it():
    from interscale.config import _validate

    for name in ("local_then_global", "local_then_global_ft", "alternating", "alternating_coupled"):
        validate_schedule(_cfg(schedule=name))
    cfg = _cfg(schedule="local_then_global")
    cfg.model.decoder.dual_decoder = False
    with pytest.raises(ValueError, match="dual_decoder"):
        _validate(cfg)


def test_the_checkpoint_name_tells_schedules_apart():
    from interscale.tl.utils import get_model_filename_prefix

    def name(**optim):
        cfg = _cfg(**optim)
        cfg.dataset.name = "d"
        return get_model_filename_prefix(cfg, True, True)

    assert "var-" not in name()  # joint with the defaults: unchanged
    tags = {
        name(schedule="local_then_global"),
        name(schedule="local_then_global_ft"),
        name(schedule="alternating", schedule_cycles=4),
        name(schedule="alternating", schedule_cycles=2),
        name(schedule="alternating_coupled", schedule_cycles=4),
        name(schedule="custom", schedule_phases=[{"share": 1.0, "lr_scale": 0.5}]),
        name(schedule="custom", schedule_phases=[{"share": 1.0, "lr_scale": 0.25}]),
        name(frozen=["local"]),
    }
    assert len(tags) == 8
    assert name(schedule="local_then_global").endswith("var-sch-l2g_")
    assert name(schedule="alternating_coupled", schedule_cycles=4).endswith("var-sch-altc4_")
    cfg = _cfg()
    cfg.dataset.name = "d"
    cfg.optim.loss_weights["local"] = 0.0
    assert get_model_filename_prefix(cfg, True, True).endswith("var-lw0-0.5_")


# --------------------------------------------------------------------------- loss weights and freezing


def _module():
    from interscale.module.combined_module import DualDecoderCombinedModule

    torch.manual_seed(0)
    return DualDecoderCombinedModule(
        cfg=_cfg(),
        n_input=G,
        n_output=G,
        n_embed=16,
        decoder_type=None,
        dropout_decoder=0.1,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="node",
    )


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    datas = []
    for n in (9, 12):
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        d = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=torch.tensor([src, dst]))
        d.mask = torch.zeros(n, dtype=torch.bool)
        d.mask[::3] = True
        datas.append(d)
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _plan(module, weights=None, frozen=()):
    from interscale.train._trainingplans import TrainingPlan

    plan = TrainingPlan(
        module,
        "regression",
        "node",
        "SCELoss",
        "gene",
        batch_size=2,
        sce_gamma=1.0,
        loss_weights=weights,
        frozen=frozen,
    )
    plan.logged = {}
    plan.log = lambda name, value, **kw: plan.logged.__setitem__(name, value)
    plan.log_dict = lambda metrics, **kw: plan.logged.update(metrics)
    return plan


def test_the_default_weights_keep_the_stacked_loss():
    from interscale.tl.masking import masked_loss
    from interscale.train.losses import SCELoss

    module = _module().eval()
    for weights in (None, {"local": 0.5, "global": 0.5}):
        assert _plan(module, weights).loss_weights is None  # the same code path as before Stage C
    plan = _plan(module)
    torch.manual_seed(1)
    loss = plan._step(_batch(), "val")
    torch.manual_seed(1)
    out = module._common_step(_batch(), "regression", "node")
    stacked = masked_loss(SCELoss(1.0), "SCELoss", out.y_pred, out.y_true, out.entry_mask)
    assert torch.equal(loss, stacked)
    assert loss.item() == pytest.approx(
        0.5 * (plan.logged["val_local_loss"] + plan.logged["val_global_loss"]).item(), abs=1e-6
    )


@pytest.mark.parametrize("weights", [(1.0, 0.0), (0.0, 1.0), (0.3, 0.7)])
def test_other_weights_give_the_weighted_sum_and_it_is_what_val_loss_reports(weights):
    plan = _plan(_module().eval(), {"local": weights[0], "global": weights[1]})
    loss = plan._step(_batch(), "val")
    expected = weights[0] * plan.logged["val_local_loss"] + weights[1] * plan.logged["val_global_loss"]
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert plan.logged["val_loss"].item() == pytest.approx(expected.item(), abs=1e-6)


def _grads(module):
    return {name: (p.grad if p.grad is not None else torch.zeros_like(p)) for name, p in module.named_parameters()}


def test_a_head_with_weight_zero_sends_no_gradient():
    module = _module().eval()
    _plan(module, {"local": 1.0, "global": 0.0})._step(_batch(), "train").backward()
    grads = _grads(module)
    assert all(torch.count_nonzero(g) == 0 for n, g in grads.items() if n.startswith("global_module."))
    assert any(torch.count_nonzero(g) > 0 for n, g in grads.items() if n.startswith("local_module.decoder"))

    module = _module().eval()
    _plan(module, {"local": 0.0, "global": 1.0})._step(_batch(), "train").backward()
    grads = _grads(module)
    assert all(torch.count_nonzero(g) == 0 for n, g in grads.items() if n.startswith("local_module.decoder"))
    # The global loss reaches the GCN through the tokens: what the alternating schedules decouple.
    gcn = [g for n, g in grads.items() if n.startswith("local_module.") and not n.startswith("local_module.decoder")]
    assert any(torch.count_nonzero(g) > 0 for g in gcn)


def test_a_frozen_component_does_not_move_stays_in_eval_and_gets_its_flags_back():
    module = _module()
    keep_fixed = next(module.local_module.parameters())
    keep_fixed.requires_grad_(False)  # a parameter that was never trainable must stay so
    before = {n: p.detach().clone() for n, p in module.named_parameters()}

    with freezing(module, ["local"]):
        plan = _plan(module, frozen=("local",))
        optimizer = plan.configure_optimizers()[0][0]
        trained = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert not any(id(p) in trained for p in module.local_module.parameters())
        for step in range(3):
            module.train()  # as Lightning does at every epoch
            loss = plan.training_step(_batch(step))
            assert not module.local_module.training and module.global_module.training
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    after = dict(module.named_parameters())
    assert all(torch.equal(before[n], after[n]) for n in before if n.startswith("local_module."))
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("global_module."))
    assert not keep_fixed.requires_grad
    assert all(p.requires_grad for p in module.parameters() if p is not keep_fixed)


def test_freezing_and_weights_refuse_what_the_model_cannot_do():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    single = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=16,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="node",
        type_gex_embedding="Linear",
    )
    with pytest.raises(ValueError, match="one head"):
        _plan(single, {"local": 1.0, "global": 0.0})
    with pytest.raises(ValueError, match="only"):
        _plan(single, frozen=("local",))
    with pytest.raises(ValueError, match="nothing to train"):
        _plan(single, frozen=("global",))
    with pytest.raises(ValueError, match="nothing to train"):
        _plan(_module(), frozen=("local", "global"))


def test_hidden_values_never_reach_a_prediction_with_a_component_frozen():
    module = _module()
    with freezing(module, ["local"]):
        plan = _plan(module, frozen=("local",))
        module.train()
        for m in plan._frozen_modules:
            m.eval()
        module.global_module.eval()  # dropout off on the rest, to compare two passes
        batch = _batch()
        with torch.no_grad():
            before = module._common_step(batch, "regression", "node").y_pred
            changed = batch.clone()
            changed.x = changed.x.clone()
            changed.x[changed.mask] += torch.rand(int(changed.mask.sum()), G) * 5
            after = module._common_step(changed, "regression", "node").y_pred
    assert torch.allclose(before, after)


# --------------------------------------------------------------------------- the driver


def _setup_model(tmp_path, schedule="joint", n_epochs=4, save=None, lr_warmup=1, min_epochs=2):
    import scanpy as sc
    from anndata import AnnData

    from interscale.config import load_config
    from interscale.model.combined_model import CombinedModel

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    n = 40 * len(samples)
    coords = rng.uniform(0, 100, size=(n, 2))
    latent = np.stack([np.sin(coords[:, 0] / 12), np.cos(coords[:, 1] / 9)], axis=1)
    X = np.abs(latent @ rng.normal(size=(2, G)) + rng.normal(scale=0.3, size=(n, G))).astype(np.float32) + 0.05
    adata = AnnData(X=X)
    adata.obsm["spatial"] = coords
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, 40))
    adata.obs["split"] = pd.Categorical(
        [{"s1": "train", "s2": "train", "s3": "val", "s4": "test"}[s] for s in adata.obs["sample"]]
    )
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "staged.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "staged.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  local_component:\n    name: GCN\n"
        "  global_component:\n    name: self-attn-transformer\n    parameters:\n      max_seq_len: 64\n"
        "      num_layers: 1\n      n_heads: 2\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n  save: {save if save else 'null'}\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n"
        f"  n_epochs: {n_epochs}\n  lr_warmup: {lr_warmup}\n  min_epochs: {min_epochs}\n  early_stopping: False\n"
        f"  schedule: {schedule}\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: stagedtest\n  prediction_task: regression\n  prediction_level: node\n"
        "  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n  mask_percentage: 0.3\n"
        "  batch_size: 2\n  evaluate_test: False\n  spatial_neigbors_kwargs:\n    radius: 25\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    CombinedModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    return CombinedModel(adata, cfg=cfg), adata, cfg


def test_the_driver_gives_each_phase_its_epochs_learning_rate_and_schedule(tmp_path, monkeypatch):
    model, _, cfg = _setup_model(tmp_path, schedule="local_then_global_ft", n_epochs=10, lr_warmup=4, min_epochs=5)
    calls = []
    monkeypatch.setattr(model, "_train_phase", lambda phase, index, n, **kw: calls.append((phase, index, n, kw)))
    model.train(max_epochs=10, datamodule=None, early_stopping=False, wandb_use=False)
    assert [c[1:3] for c in calls] == [(0, 3), (1, 3), (2, 3)]
    assert [c[3]["max_epochs"] for c in calls] == [4, 4, 2]
    assert [c[3]["lr_max_epochs"] for c in calls] == [4, 4, 2]  # each phase its own cosine
    assert [c[3]["lr"] for c in calls] == pytest.approx([0.01, 0.01, 0.001])
    assert [c[3]["lr_warmup"] for c in calls] == [2, 2, 1]  # 4 of 10 epochs, scaled to each phase
    assert [c[3]["min_epochs"] for c in calls] == [4, 4, 2]
    assert [c[0].frozen for c in calls] == [("global",), ("local",), ()]

    (tmp_path / "joint").mkdir()
    joint, _, cfg = _setup_model(tmp_path / "joint", schedule="joint", n_epochs=10)
    calls.clear()
    monkeypatch.setattr(joint, "_train_phase", lambda phase, index, n, **kw: calls.append((phase, index, n, kw)))
    joint.train(max_epochs=7, datamodule=None, early_stopping=False, wandb_use=False)
    ((phase, index, n, kw),) = calls
    assert (index, n, kw["max_epochs"]) == (0, 1, 7)
    # As before Stage C: the config's warm-up and cosine length, not ones derived from max_epochs.
    assert (kw["lr_warmup"], kw["lr_max_epochs"], kw["min_epochs"]) == (
        cfg.optim.lr_warmup,
        cfg.optim.n_epochs,
        cfg.optim.min_epochs,
    )


def test_a_staged_run_trains_each_component_only_in_its_phases_and_saves_once(tmp_path, monkeypatch):
    import interscale.train._training as training
    from interscale.train._utils import MetricsHistory

    records = []

    class Recorder(MetricsHistory):
        def on_train_start(self, trainer, pl_module):
            self.start = {k: v.detach().clone() for k, v in pl_module.module.state_dict().items()}
            self.frozen = pl_module.frozen
            self.max_epochs = trainer.max_epochs
            self.callbacks = [type(c).__name__ for c in trainer.callbacks]
            self.frozen_grad = [p.requires_grad for m in pl_module._frozen_modules for p in m.parameters()]
            self.eval_ok = []

        def on_before_backward(self, trainer, pl_module, loss):
            self.eval_ok.append(all(not m.training for m in pl_module._frozen_modules))

        def on_train_end(self, trainer, pl_module):
            self.end = {k: v.detach().clone() for k, v in pl_module.module.state_dict().items()}
            records.append(self)

    monkeypatch.setattr(training, "MetricsHistory", Recorder)
    save = tmp_path / "out"
    model, adata, cfg = _setup_model(tmp_path, schedule="local_then_global", n_epochs=4, save=f"{save}/")

    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.tl import prepare_geome_dataset

    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    saves = []
    original_save = model.save
    monkeypatch.setattr(model, "save", lambda *a, **k: saves.append(a) or original_save(*a, **k))
    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=False, wandb_use=False)

    assert len(saves) == 1  # once, after the last phase
    assert len(records) == 2 and [r.max_epochs for r in records] == [2, 2]
    assert [r.frozen for r in records] == [("global",), ("local",)]

    def moved(record, prefix):
        return [k for k in record.start if k.startswith(prefix) and not torch.equal(record.start[k], record.end[k])]

    assert not moved(records[0], "global_module.") and moved(records[0], "local_module.")
    assert not moved(records[1], "local_module.") and moved(records[1], "global_module.")
    assert all(torch.equal(records[1].start[k], records[0].end[k]) for k in records[0].end)  # carried over
    for record in records:
        assert record.eval_ok and all(record.eval_ok)  # frozen parts in eval at every step
        assert record.frozen_grad and not any(record.frozen_grad)
        assert "NodeMaskResampleCallback" in record.callbacks  # masks redrawn in every phase
    assert "ModelCheckpoint" not in records[0].callbacks and "ModelCheckpoint" in records[1].callbacks
    assert all(p.requires_grad for p in model.module.parameters())  # flags back after the run
    assert len(model.phase_histories_) == 2 and model.history_ is model.phase_histories_[-1]
    assert model.is_trained_

    files = sorted(p.name for p in save.iterdir())
    assert not any("-v1" in f for f in files), files
    assert sum(f.endswith(".ckpt") for f in files) == 2  # the best epoch, and the final trainer state
    assert all("var-sch-l2g_" in f for f in files)


def test_the_wandb_step_continues_across_phases(monkeypatch):
    from lightning.pytorch.loggers import WandbLogger

    from interscale.train._utils import StepOffsetWandbLogger

    seen = []
    monkeypatch.setattr(WandbLogger, "log_metrics", lambda self, metrics, step=None: seen.append(step))
    logger = StepOffsetWandbLogger(name="run")
    logger.log_metrics({"a": 1.0}, step=3)
    logger.step_offset = 40
    logger.log_metrics({"a": 1.0}, step=3)
    logger.log_metrics({"a": 1.0})
    assert seen == [3, 43, None]


def test_a_trained_local_model_can_be_copied_in_as_phase_one(tmp_path):
    from interscale.model.local_model import LocalModel

    combined, adata, cfg = _setup_model(tmp_path)
    LocalModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    torch.manual_seed(5)
    local = LocalModel(adata, cfg=cfg)
    assert not all(
        torch.equal(a, b)
        for a, b in zip(
            local.module.state_dict().values(), combined.module.local_module.state_dict().values(), strict=False
        )
    )
    combined.copy_local_from(local)
    source = {k: v for k, v in local.module.state_dict().items() if not k.startswith("aux_losses.")}
    target = combined.module.local_module.state_dict()
    assert set(source) == set(target) and all(torch.equal(source[k], target[k]) for k in source)
