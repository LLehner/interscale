"""VICReg through the training plan: beside reconstruction, and instead of it.

The arithmetic is tested in `test_vicreg.py`. What matters here is the wiring -- that the
coefficients decide how many forward passes run, that the term reaches the optimiser, and that
switching the reconstruction criterion off leaves a real objective rather than a constant zero.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

import lightning.pytorch as pl

from interscale.config import get_cfg_defaults
from interscale.train._trainingplans import TrainingPlan
from interscale.train.aux_losses import build_aux_losses
from tests.test_aux_losses import _tiny_datamodule, _tiny_module


def _cfg(**contrastive):
    cfg = get_cfg_defaults()
    cfg.optim.aux_loss_weights.vicreg = 1.0
    for key, value in contrastive.items():
        setattr(cfg.optim.contrastive, key, value)
    return cfg


def _plan(module, composite, loss="MSELoss"):
    return TrainingPlan(
        module, "regression", "node", loss, "cell", batch_size=2, aux_losses=composite,
        lr_scheduler="CosineWarmupScheduler", lr_warmup=1, lr_max_epochs=2,
    )


def _fit(plan):
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(plan, datamodule=_tiny_datamodule())
    return trainer.logged_metrics


# --------------------------------------------------------------------------- how many views


def test_the_invariance_coefficient_is_the_two_view_switch():
    module = _tiny_module()

    with_inv = build_aux_losses(_cfg(vicreg_lambda=25.0), module)
    without = build_aux_losses(_cfg(vicreg_lambda=0.0), module)

    assert with_inv.requires_views == 2
    assert without.requires_views == 1


def test_variance_and_covariance_alone_need_one_pass(tmp_path):
    """Stage 1 as planned: a collapse regulariser beside reconstruction, no pairing at all."""
    module = _tiny_module()
    composite = build_aux_losses(_cfg(vicreg_lambda=0.0), module)

    logged = _fit(_plan(module, composite))

    assert "train_vicreg_var" in logged and "train_vicreg_cov" in logged
    assert "train_vicreg_inv" not in logged


def test_all_three_terms_are_logged_when_two_views_run():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)

    logged = _fit(_plan(module, composite))

    for name in ("train_vicreg_var", "train_vicreg_cov", "train_vicreg_inv"):
        assert name in logged, f"{name} missing"


def test_the_collapse_diagnostic_is_logged_but_not_optimised():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(vicreg_lambda=0.0, vicreg_nu=0.0), module)

    logged = _fit(_plan(module, composite))

    assert "train__vicreg_std" in logged
    # aux_total is the variance term alone; the std diagnostic must not have been added to it.
    assert logged["train_aux_total"] == pytest.approx(float(logged["train_vicreg_var"]), rel=1e-4)


# --------------------------------------------------------------------------- instead of recon


def test_vicreg_can_be_the_whole_objective():
    """`optim.loss: none` plus a vicreg weight: no reconstruction criterion anywhere."""
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)
    plan = _plan(module, composite, loss="none")
    assert plan.loss is None

    before = [p.detach().clone() for p in module.parameters()]
    logged = _fit(plan)

    assert logged["train_loss"] == pytest.approx(float(logged["train_aux_total"]), rel=1e-4)
    assert any(not torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True))


def test_reconstruction_metrics_are_still_reported_without_a_criterion():
    """Nothing is being reconstructed, but what the representation can do is still the question."""
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)

    logged = _fit(_plan(module, composite, loss="none"))

    assert "train_r2" in logged and "train_mse" in logged


def test_the_expander_parameters_are_trained():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(expander_dims=[32, 32]), module)
    module.aux_losses = composite
    expander = composite.terms["vicreg"].expander
    before = expander[0].weight.detach().clone()

    _fit(_plan(module, composite, loss="none"))

    assert not torch.equal(before, expander[0].weight)


# --------------------------------------------------------------------------- refusals


def test_all_zero_coefficients_are_rejected():
    with pytest.raises(ValueError, match="contributes nothing"):
        build_aux_losses(_cfg(vicreg_lambda=0.0, vicreg_mu=0.0, vicreg_nu=0.0), _tiny_module())


def test_invariance_without_variance_warns_about_collapse():
    """Mapping every cell to one point satisfies invariance exactly."""
    with pytest.warns(UserWarning, match="collapses"):
        build_aux_losses(_cfg(vicreg_mu=0.0), _tiny_module())


def test_an_empty_expander_is_allowed_as_the_ablation():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(expander_dims=[]), module)

    assert isinstance(composite.terms["vicreg"].expander, torch.nn.Identity)
    _fit(_plan(module, composite))


# --------------------------------------------------------------------------- which module


def _gcn_module():
    from interscale.module.local_modules.GCN import GCN

    return GCN(
        n_layers=2, hidden_dim=16, dropout_local=0.3,
        n_input=6, n_output=6, n_embed=8, decoder_type="linear",
        dropout_decoder=0.0, decoder_hidden_dims=[8],
        mask_percentage=0.3, mask_strategy="node",
    )


def _local_batch():
    from torch_geometric.data import Batch, Data

    def graph(n, seed):
        g = torch.Generator().manual_seed(seed)
        src = list(range(n - 1)) + list(range(1, n))
        data = Data(
            x=torch.randn(n, 6, generator=g),
            edge_index=torch.tensor([src, list(range(1, n)) + list(range(n - 1))], dtype=torch.long),
        )
        data.mask = torch.rand(n, generator=g) < 0.4
        data.mask[0] = True
        data.obs_names = torch.arange(n)
        return data

    return Batch.from_data_list([graph(7, 0), graph(9, 1)])


def _two_view_out(module, batch):
    out = module._common_step(batch, "regression", "node")
    out.views = [out.view, module._common_step(batch, "regression", "node").view]
    return out


@pytest.mark.parametrize("embedding", ["auto", "local"])
def test_vicreg_runs_on_a_local_only_module(embedding):
    """A GCN has no transformer tokens; `auto` must fall back to the local embedding.

    This is what lets one config block serve LocalModel, GlobalModel and CombinedModel: the
    alternative is a `contrastive` section that has to be rewritten per model type.
    """
    module = _gcn_module().train()
    cfg = _cfg(embedding=embedding, expander_dims=[32])
    composite = build_aux_losses(cfg, module)

    total, reported = composite(_two_view_out(module, _local_batch()), _local_batch())

    assert {"vicreg_var", "vicreg_cov", "vicreg_inv"} <= set(reported)
    total.backward()
    assert module.input_proj.weight.grad is not None, "the term never reached the encoder"


def test_asking_for_global_on_a_local_module_says_what_to_do():
    module = _gcn_module().train()
    batch = _local_batch()
    composite = build_aux_losses(_cfg(embedding="global", expander_dims=[32]), module)

    with pytest.raises(ValueError, match="'global' but this module produced none"):
        composite(_two_view_out(module, batch), batch)


def test_local_and_global_are_separately_selectable_on_a_combined_run():
    """On a model with both, `embedding` chooses which scale the regulariser acts on."""
    module = _tiny_module()  # transformer: has global tokens, no local embedding
    assert build_aux_losses(_cfg(embedding="global"), module).terms["vicreg"].embedding == "global"
    assert build_aux_losses(_cfg(embedding="local"), module).terms["vicreg"].embedding == "local"


def test_an_invariance_term_with_no_encoder_stochasticity_warns():
    """Dropout is currently the ONLY thing making two passes differ; at 0 there is no task.

    Confirmed on a real run: with every dropout at 0, `train_vicreg_inv` is exactly 0.0 even in
    training mode, so VICReg silently degenerates to variance+covariance -- which its own Table 7
    reports as collapse.
    """
    cfg = _cfg()
    cfg.model.local_component.parameters = type(cfg.model)()
    cfg.model.local_component.parameters.dropout_local = 0.0
    cfg.model.global_component.parameters = type(cfg.model)()
    cfg.model.global_component.parameters.dropout_global = 0.0

    with pytest.warns(UserWarning, match="every .*dropout is 0"):
        build_aux_losses(cfg, _tiny_module())


def test_no_warning_when_some_dropout_is_configured():
    cfg = _cfg()
    cfg.model.global_component.parameters = type(cfg.model)()
    cfg.model.global_component.parameters.dropout_global = 0.1
    module = _tiny_module()  # built outside the filter: torch emits its own unrelated warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_aux_losses(cfg, module)

    assert not [w for w in record if "dropout is 0" in str(w.message)]
