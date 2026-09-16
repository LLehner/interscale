"""The auxiliary-loss scaffolding: opt-in, additive, and inert until switched on.

The contract these lock down is the one that makes every later stage cheap: a term with weight
0.0 is not constructed, the composite is empty, and the training step takes exactly the path it
took before any of this existed. Everything else here is about a term being able to reach what it
needs without a signature change.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from interscale.config import get_cfg_defaults
from interscale.module.base import StepOutput, ViewOutput
from interscale.train.aux_losses import (
    AUX_LOSSES,
    AuxLoss,
    CompositeAuxLoss,
    build_aux_losses,
    register_aux_loss,
)


@pytest.fixture
def registry():
    """Restore the global registry, so a test term cannot leak into another test."""
    original = dict(AUX_LOSSES)
    yield AUX_LOSSES
    AUX_LOSSES.clear()
    AUX_LOSSES.update(original)


@pytest.fixture
def cfg():
    return get_cfg_defaults()


def make_out():
    view = ViewOutput(
        global_embedding=torch.randn(5, 2, 4),
        src_padding_mask=torch.zeros(2, 5, dtype=torch.bool),
        padded_node_idx=torch.arange(8),
    )
    return StepOutput(y_pred=torch.randn(8, 3), y_true=torch.randn(8, 3), views=[view])


# --------------------------------------------------------------------------- inert by default


def test_no_weights_means_an_empty_composite(cfg):
    composite = build_aux_losses(cfg)

    assert not composite
    assert len(composite.terms) == 0
    assert composite.requires_views == 1


def test_empty_composite_reports_a_zero_tensor(cfg):
    total, reported = build_aux_losses(cfg)(make_out(), None)

    assert reported == {}
    assert total == 0.0
    assert isinstance(total, torch.Tensor), "a float here would break `loss + total` on device"


def test_a_zero_weight_term_is_never_constructed(cfg, registry):
    built = []

    @register_aux_loss("never_built")
    class NeverBuilt(AuxLoss):
        def __init__(self, cfg, module=None):
            super().__init__()
            built.append(1)

        def forward(self, out, batch):
            return {"never_built": torch.zeros(())}

    cfg.optim.aux_loss_weights.never_built = 0.0
    composite = build_aux_losses(cfg)

    assert built == [], "a zero-weighted term must cost nothing, not be built and multiplied by 0"
    assert not composite


# --------------------------------------------------------------------------- once switched on


def test_enabled_terms_are_weighted_and_reported_unweighted(cfg, registry):
    @register_aux_loss("constant")
    class Constant(AuxLoss):
        def __init__(self, cfg, module=None):
            super().__init__()

        def forward(self, out, batch):
            return {"constant": torch.tensor(2.0)}

    cfg.optim.aux_loss_weights.constant = 3.0
    total, reported = build_aux_losses(cfg)(make_out(), None)

    assert total == pytest.approx(6.0), "the weight applies to the total"
    assert reported["constant"] == pytest.approx(2.0), "reporting stays unweighted and comparable"


def test_required_views_is_the_max_over_enabled_terms(cfg, registry):
    @register_aux_loss("one_view")
    class OneView(AuxLoss):
        def __init__(self, cfg, module=None):
            super().__init__()

        def forward(self, out, batch):
            return {"one_view": torch.zeros(())}

    @register_aux_loss("two_view")
    class TwoView(AuxLoss):
        requires_views = 2

        def __init__(self, cfg, module=None):
            super().__init__()

        def forward(self, out, batch):
            return {"two_view": torch.zeros(())}

    cfg.optim.aux_loss_weights.one_view = 1.0
    cfg.optim.aux_loss_weights.two_view = 0.0
    assert build_aux_losses(cfg).requires_views == 1

    cfg.optim.aux_loss_weights.two_view = 1.0
    assert build_aux_losses(cfg).requires_views == 2, "enabling a two-view term is the only switch"


def test_term_parameters_are_reachable_for_the_optimiser(cfg, registry):
    """A projection head lives on the term; if it is not in .parameters() it never trains."""

    @register_aux_loss("with_head")
    class WithHead(AuxLoss):
        def __init__(self, cfg, module=None):
            super().__init__()
            self.head = nn.Linear(4, 2)

        def forward(self, out, batch):
            return {"with_head": self.head(out.view.tokens()).pow(2).mean()}

    cfg.optim.aux_loss_weights.with_head = 1.0
    composite = build_aux_losses(cfg)

    assert len(list(composite.parameters())) == 2  # weight + bias


def test_a_term_can_read_tokens_and_backpropagate(cfg, registry):
    @register_aux_loss("token_norm")
    class TokenNorm(AuxLoss):
        def __init__(self, cfg, module=None):
            super().__init__()
            self.head = nn.Linear(4, 2)

        def forward(self, out, batch):
            return {"token_norm": self.head(out.view.tokens()).pow(2).mean()}

    cfg.optim.aux_loss_weights.token_norm = 1.0
    composite = build_aux_losses(cfg)
    total, _ = composite(make_out(), None)
    total.backward()

    assert composite.terms["token_norm"].head.weight.grad is not None


# --------------------------------------------------------------------------- failure modes


def test_unregistered_name_with_a_weight_is_rejected(cfg):
    """A typo in a sweep must not silently train without the term it claims to measure."""
    cfg.optim.aux_loss_weights.definitely_not_a_loss = 1.0

    with pytest.raises(ValueError, match="unregistered auxiliary losses"):
        build_aux_losses(cfg)


def test_colliding_report_names_are_rejected(registry):
    class Same(AuxLoss):
        def __init__(self):
            super().__init__()

        def forward(self, out, batch):
            return {"same": torch.zeros(())}

    composite = CompositeAuxLoss({"a": Same(), "b": Same()}, {"a": 1.0, "b": 1.0})

    with pytest.raises(ValueError, match="names must be unique"):
        composite(make_out(), None)


def test_double_registration_is_rejected(registry):
    @register_aux_loss("dupe")
    class First(AuxLoss):
        def forward(self, out, batch):
            return {}

    with pytest.raises(ValueError, match="already registered"):

        @register_aux_loss("dupe")
        class Second(AuxLoss):
            def forward(self, out, batch):
                return {}


# --------------------------------------------------------------------------- through a real step


def _tiny_datamodule():
    from torch_geometric.data import Data

    from interscale.geome_dataloader import GraphAnnDataModule

    def graph(n, seed):
        g = torch.Generator().manual_seed(seed)
        src = list(range(n - 1)) + list(range(1, n))
        d = Data(
            x=torch.randn(n, 6, generator=g),
            edge_index=torch.tensor([src, list(range(1, n)) + list(range(n - 1))], dtype=torch.long),
        )
        d.embeddings = torch.randn(n, 4, generator=g)
        return d

    dm = GraphAnnDataModule(
        datas=[[graph(7, 0), graph(9, 1)], [graph(8, 2)], [graph(8, 3)]],
        batch_size=2,
        num_workers=0,
        mask_percentage=0.3,
        mask_strategy="node",
        learning_type="node",
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    return dm


def _tiny_module():
    from interscale.module.global_modules import TransformerNodeEncoderHook

    return TransformerNodeEncoderHook(
        max_seq_len=16,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=8,
        long_range_attention=False,
        local_mask_hops=1,
        n_input=6,
        n_output=6,
        n_embed=4,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[8],
        mask_percentage=0.3,
        mask_strategy="node",
        type_gex_embedding=None,
    )


def test_an_enabled_term_is_logged_and_reaches_the_optimiser(cfg, registry):
    """The whole scaffolding end to end: built from cfg, fed a real step, logged, and trained.

    The isolated tests above can all pass while the term never reaches a training run -- it is
    the wiring through `TrainingPlan` and `configure_optimizers` that decides that, and this is
    what checks it.
    """
    import lightning.pytorch as pl

    from interscale.train._trainingplans import TrainingPlan

    @register_aux_loss("e2e_probe")
    class Probe(AuxLoss):
        def __init__(self, cfg, module=None):
            super().__init__()
            self.head = nn.Linear(module.n_embed, 3)

        def forward(self, out, batch):
            # Reads both coordinate systems: tokens in padded order, slide id in batch order.
            z = self.head(out.view.tokens())
            slide = batch.batch[out.view.padded_node_idx]
            return {"e2e_probe": z.pow(2).mean(), "e2e_slides": torch.tensor(float(slide.unique().numel()))}

    module = _tiny_module()
    cfg.optim.aux_loss_weights.e2e_probe = 0.5
    composite = build_aux_losses(cfg, module)

    plan = TrainingPlan(
        module,
        "regression",
        "node",
        "MSELoss",
        "cell",
        batch_size=2,
        aux_losses=composite,
        # A scheduler is not optional in practice: `configure_optimizers` returns a scheduler dict
        # whose "scheduler" entry is None when lr_scheduler is None, which Lightning rejects.
        lr_scheduler="CosineWarmupScheduler",
        lr_warmup=1,
        lr_max_epochs=2,
    )
    before = composite.terms["e2e_probe"].head.weight.detach().clone()

    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(plan, datamodule=_tiny_datamodule())

    logged = trainer.logged_metrics
    assert "train_e2e_probe" in logged, "the term never reached the logs"
    assert "train_aux_total" in logged
    # Reported unweighted, summed weighted.
    expected = 0.5 * (logged["train_e2e_probe"] + logged["train_e2e_slides"])
    assert logged["train_aux_total"] == pytest.approx(float(expected), rel=1e-5)
    assert not torch.equal(before, composite.terms["e2e_probe"].head.weight), "the head never trained"


# --------------------------------------------------------------------------- persistence


def _module_with_head(cfg, registry_name="persisted"):
    """A module carrying one parameterised auxiliary term, built twice per test in places."""
    if registry_name not in AUX_LOSSES:

        @register_aux_loss(registry_name)
        class Persisted(AuxLoss):
            def __init__(self, cfg, module=None):
                super().__init__()
                self.head = nn.Linear(module.n_embed, 3)

            def forward(self, out, batch):
                return {registry_name: self.head(out.view.tokens()).pow(2).mean()}

    module = _tiny_module()
    setattr(cfg.optim.aux_loss_weights, registry_name, 1.0)
    module.aux_losses = build_aux_losses(cfg, module)
    return module


def test_heads_are_part_of_the_module_state_dict(cfg, registry):
    """`BaseModel.save` writes only `module.state_dict()`; a head outside it is lost on reload."""
    module = _module_with_head(cfg)

    keys = [k for k in module.state_dict() if "aux_losses" in k]

    assert keys == ["aux_losses.terms.persisted.head.weight", "aux_losses.terms.persisted.head.bias"]


def test_a_trained_head_survives_a_state_dict_round_trip(cfg, registry):
    """The point of the whole placement: resuming must not silently reinitialise the head."""
    trained = _module_with_head(cfg)
    with torch.no_grad():
        trained.aux_losses.terms["persisted"].head.weight.fill_(0.123)

    fresh = _module_with_head(cfg)
    before = fresh.aux_losses.terms["persisted"].head.weight.detach().clone()
    fresh.load_state_dict(trained.state_dict(), strict=False)
    after = fresh.aux_losses.terms["persisted"].head.weight

    assert not torch.equal(before, after), "load did not reach the head"
    assert torch.equal(after, trained.aux_losses.terms["persisted"].head.weight)


def test_a_checkpoint_without_heads_still_loads(cfg, registry):
    """Checkpoints predating auxiliary losses must keep loading; `strict=False` carries that."""
    old_state = _tiny_module().state_dict()
    module = _module_with_head(cfg)

    missing, unexpected = module.load_state_dict(old_state, strict=False)

    assert all("aux_losses" in k for k in missing), f"unexpected missing keys: {missing}"
    assert unexpected == []


def test_a_checkpoint_with_heads_loads_into_a_model_that_has_none(cfg, registry):
    """Turning a term off must not make earlier checkpoints unloadable."""
    state = _module_with_head(cfg).state_dict()
    plain = _tiny_module()
    plain.aux_losses = build_aux_losses(get_cfg_defaults(), plain)  # no weights -> no terms

    missing, unexpected = plain.load_state_dict(state, strict=False)

    assert all("aux_losses" in k for k in unexpected), f"unexpected extras: {unexpected}"
    assert missing == []


def test_the_optimiser_is_not_handed_the_same_parameter_twice(cfg, registry):
    """The terms hang off the module, so collecting them again separately would duplicate them."""
    from interscale.train._trainingplans import TrainingPlan

    module = _module_with_head(cfg)
    plan = TrainingPlan(
        module, "regression", "node", "MSELoss", "cell", batch_size=2,
        lr_scheduler="CosineWarmupScheduler", lr_warmup=1, lr_max_epochs=2,
    )
    [optimizer], _ = plan.configure_optimizers()

    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert any(p is module.aux_losses.terms["persisted"].head.weight for p in params)


def test_the_plan_exposes_the_modules_terms(cfg, registry):
    from interscale.train._trainingplans import TrainingPlan

    module = _module_with_head(cfg)
    plan = TrainingPlan(
        module, "regression", "node", "MSELoss", "cell", batch_size=2,
        lr_scheduler="CosineWarmupScheduler", lr_warmup=1, lr_max_epochs=2,
    )

    assert plan.aux_losses is module.aux_losses


# --------------------------------------------------------------------------- multiple views


def test_a_two_view_term_receives_two_row_aligned_views(cfg, registry):
    """`requires_views` is the only switch: asking for two is what makes two passes happen."""
    import lightning.pytorch as pl

    from interscale.train._trainingplans import TrainingPlan
    from interscale.train.aux_losses import aligned_view_tokens

    seen = {}

    @register_aux_loss("two_view_probe")
    class TwoViewProbe(AuxLoss):
        requires_views = 2

        def __init__(self, cfg, module=None):
            super().__init__()

        def forward(self, out, batch):
            views = aligned_view_tokens(out)
            mode = "train" if module.training else "eval"
            seen["n_views"] = len(views)
            seen["aligned"] = all(v.shape == views[0].shape for v in views)
            seen[f"differ_{mode}"] = not torch.equal(views[0], views[1])
            return {"two_view_probe": (views[0] - views[1]).pow(2).mean()}

    module = _tiny_module()
    module.transformer_encoder.layers[0].dropout1.p = 0.5  # make the two passes visibly differ
    cfg.optim.aux_loss_weights.two_view_probe = 1.0
    composite = build_aux_losses(cfg, module)
    assert composite.requires_views == 2

    plan = TrainingPlan(
        module, "regression", "node", "MSELoss", "cell", batch_size=2, aux_losses=composite,
        lr_scheduler="CosineWarmupScheduler", lr_warmup=1, lr_max_epochs=2,
    )
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(plan, datamodule=_tiny_datamodule())

    assert seen["n_views"] == 2
    assert seen["aligned"]
    assert seen["differ_train"], "two training passes with dropout on must not be identical"
    # And the caveat that comes with dropout-only views: in eval mode the encoder is
    # deterministic, so the two views coincide exactly and any invariance term is identically
    # zero on validation. See `_forward_views`.
    assert not seen["differ_eval"]


def test_a_single_view_term_still_runs_one_pass(cfg, registry):
    """The default path must be untouched: one view, one forward."""
    from interscale.train._trainingplans import TrainingPlan

    module = _tiny_module()
    plan = TrainingPlan(
        module, "regression", "node", "MSELoss", "cell", batch_size=2,
        lr_scheduler="CosineWarmupScheduler", lr_warmup=1, lr_max_epochs=2,
    )
    assert plan.aux_losses.requires_views == 1


def test_misaligned_views_are_rejected_with_the_cause():
    """pad_batch subsamples at random above max_seq_len, so two passes can keep different cells."""
    from interscale.module.base import StepOutput, ViewOutput
    from interscale.train.aux_losses import aligned_view_tokens

    def view(idx):
        return ViewOutput(
            global_embedding=torch.randn(4, 1, 3),
            src_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
            padded_node_idx=torch.tensor(idx),
        )

    out = StepOutput(
        y_pred=torch.zeros(3, 2), y_true=torch.zeros(3, 2),
        views=[view([0, 1, 2]), view([0, 1, 5])],
    )

    with pytest.raises(ValueError, match="max_seq_len"):
        aligned_view_tokens(out)
