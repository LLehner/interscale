"""Ring NCE through the training plan, and the config states that would misplace the annulus.

The mechanics are tested in `test_ring_nce.py`. What matters here is that the inner radius is
read off the module the term trains -- the number the attention mask was built with -- rather than
guessed, that the term runs in one pass, and that the combinations which quietly contradict the
architecture say so.
"""

import pytest

torch = pytest.importorskip("torch")

import lightning.pytorch as pl

from interscale.config import get_cfg_defaults
from interscale.train._trainingplans import TrainingPlan
from interscale.train.aux_losses import build_aux_losses
from tests.test_aux_losses import _tiny_datamodule, _tiny_module


def _cfg(**contrastive):
    cfg = get_cfg_defaults()
    cfg.optim.aux_loss_weights.ring_nce = 1.0
    cfg.optim.contrastive.ring_outer_hops = 3
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


# --------------------------------------------------------------------------- the inner radius


def test_the_inner_radius_is_the_modules_local_reach():
    """The point of the term is to contrast what the local component saw against what it did not,
    so the ring has to start exactly where the mask's reach ends -- and that reach is decided by
    the module, not by the config."""
    term = build_aux_losses(_cfg(), _tiny_module(local_mask_hops=2)).terms["ring_nce"]

    assert term.inner_hops == 2
    assert term.derived_inner


def test_no_local_reach_makes_the_core_the_anchor_alone():
    """A GlobalModel has no local component, so its reach is 0 and the ring starts at the anchor's
    direct neighbours. Reading 0 as "unset" would invent a neighbourhood nothing aggregated."""
    term = build_aux_losses(_cfg(), _tiny_module(local_mask_hops=0)).terms["ring_nce"]

    assert term.inner_hops == 0


def test_a_derived_inner_radius_needs_the_module():
    with pytest.raises(ValueError, match="must be built with the module"):
        build_aux_losses(_cfg())


def test_an_inner_radius_disagreeing_with_the_mask_warns():
    """Below the mask the positive contains cells the transformer cannot attend to; above it a
    band is left out of the objective. Neither raises and both train."""
    with pytest.warns(UserWarning, match="does not start where"):
        build_aux_losses(_cfg(ring_inner_hops=2), _tiny_module(local_mask_hops=1))


def test_an_outer_radius_inside_the_inner_one_is_rejected_at_construction():
    with pytest.raises(ValueError, match="must exceed the inner radius"):
        build_aux_losses(_cfg(ring_outer_hops=2), _tiny_module(local_mask_hops=2))


def test_unmasked_anchors_warn_because_the_negative_becomes_self_detectable():
    with pytest.warns(UserWarning, match="self-detection"):
        build_aux_losses(_cfg(anchors_masked_only=False), _tiny_module())


# --------------------------------------------------------------------------- wiring


def test_the_term_needs_a_single_forward_pass():
    assert build_aux_losses(_cfg(), _tiny_module()).requires_views == 1


def test_it_runs_beside_the_reconstruction_criterion_and_is_logged():
    module = _tiny_module()

    logged = _fit(_plan(module, build_aux_losses(_cfg(), module)))

    assert "train_ring_nce" in logged
    assert torch.isfinite(logged["train_ring_nce"])


def test_the_hybrid_folds_both_terms_into_the_optimised_loss():
    module = _tiny_module()

    logged = _fit(_plan(module, build_aux_losses(_cfg(), module)))

    assert logged["train_loss"] > logged["train_aux_total"] > 0


def test_the_projector_is_trained():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)
    first = next(p for p in composite.terms["ring_nce"].projector.parameters())
    before = first.detach().clone()

    _fit(_plan(module, composite))

    assert not torch.allclose(before, first.detach())


def test_it_is_a_real_number_in_validation():
    module = _tiny_module()

    logged = _fit(_plan(module, build_aux_losses(_cfg(), module)))

    assert "val_ring_nce" in logged
    assert logged["val_ring_nce"] > 0


def test_the_diagnostics_are_reported_but_not_optimised():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)
    batch = next(iter(_tiny_datamodule().train_dataloader()))
    out = module._common_step(batch, "regression", "node")

    total, reported = composite(out, batch)

    assert "_ring_nce_gap" in reported and "_ring_nce_kept" in reported
    assert total.item() == pytest.approx(reported["ring_nce"].item())


def test_one_negative_is_the_default_and_more_are_available():
    """`ring_n_negatives: 1` is the anchor's own core alone -- a two-way softmax. Above 1 the rest
    come from other anchors' cores, so every negative is still a GNN-covered neighbourhood."""
    module = _tiny_module()

    one = build_aux_losses(_cfg(), module).terms["ring_nce"]
    many = build_aux_losses(_cfg(ring_n_negatives=8), module).terms["ring_nce"]

    assert one.n_negatives == 1 and many.n_negatives == 8
    for term in (one, many):
        batch = next(iter(_tiny_datamodule().train_dataloader()))
        out = module._common_step(batch, "regression", "node")
        assert torch.isfinite(term(out, batch)["ring_nce"])


def test_it_refuses_an_embedding_the_module_does_not_produce():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(embedding="local"), module)
    batch = next(iter(_tiny_datamodule().train_dataloader()))
    out = module._common_step(batch, "regression", "node")

    with pytest.raises(ValueError, match="produced none"):
        composite(out, batch)
