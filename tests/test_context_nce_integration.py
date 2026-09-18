"""Context NCE through the training plan: one pass, real gradients, and the guards that refuse.

The mechanics are tested in `test_context_nce.py`. What matters here is the wiring -- that the
term runs on a real step without a second forward pass, that its projector reaches the optimiser,
that it composes with the reconstruction criterion (the hybrid), and that the configurations which
would quietly measure the wrong thing fail at construction instead.
"""

import pytest

torch = pytest.importorskip("torch")

import lightning.pytorch as pl

from interscale.config import get_cfg_defaults
from interscale.train._trainingplans import TrainingPlan
from interscale.train.aux_losses import build_aux_losses
from tests.test_aux_losses import _tiny_module


def _cfg(**contrastive):
    cfg = get_cfg_defaults()
    cfg.optim.aux_loss_weights.context_nce = 1.0
    cfg.optim.contrastive.n_candidates = 6
    cfg.optim.contrastive.n_negatives = 2
    cfg.optim.contrastive.context_hops = 1
    for key, value in contrastive.items():
        setattr(cfg.optim.contrastive, key, value)
    return cfg


def _datamodule(with_celltype=True):
    """The tiny datamodule, plus the cell-type annotation the composition match needs."""
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
        if with_celltype:
            # One-hot over 3 types, as `prepare_geome_dataset` attaches a categorical obs column.
            codes = torch.arange(n) % 3
            d.celltype = torch.nn.functional.one_hot(codes, num_classes=3).float()
        return d

    dm = GraphAnnDataModule(
        datas=[[graph(9, 0), graph(11, 1)], [graph(10, 2)], [graph(10, 3)]],
        batch_size=2,
        num_workers=0,
        mask_percentage=0.5,
        mask_strategy="node",
        learning_type="node",
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    return dm


def _plan(module, composite, loss="MSELoss"):
    return TrainingPlan(
        module, "regression", "node", loss, "cell", batch_size=2, aux_losses=composite,
        lr_scheduler="CosineWarmupScheduler", lr_warmup=1, lr_max_epochs=2,
    )


def _fit(plan, dm=None):
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(plan, datamodule=dm or _datamodule())
    return trainer.logged_metrics


# --------------------------------------------------------------------------- one pass


def test_the_term_needs_a_single_forward_pass():
    """The whole point of Stage 2 against Stage 3: the pairing is structural, not augmented."""
    composite = build_aux_losses(_cfg(), _tiny_module())

    assert composite.requires_views == 1


def test_it_runs_beside_the_reconstruction_criterion_and_is_logged():
    module = _tiny_module()

    logged = _fit(_plan(module, build_aux_losses(_cfg(), module)))

    assert "train_context_nce" in logged
    assert torch.isfinite(logged["train_context_nce"])


def test_the_hybrid_folds_both_terms_into_the_optimised_loss():
    """`train_loss` must be the thing actually optimised -- reconstruction alone would make the
    contrastive term invisible to EarlyStopping and to every sweep that monitors it."""
    module = _tiny_module()

    logged = _fit(_plan(module, build_aux_losses(_cfg(), module)))

    assert logged["train_loss"] > logged["train_aux_total"] > 0


def test_the_projector_is_trained():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)
    first = next(p for p in composite.terms["context_nce"].projector.parameters())
    before = first.detach().clone()

    _fit(_plan(module, composite))

    assert not torch.allclose(before, first.detach())


def test_the_term_is_a_real_number_in_validation():
    """Unlike VICReg's invariance term, which is identically 0.0 under eval() because dropout-only
    views coincide there. A structural pairing does not depend on training mode."""
    module = _tiny_module()

    logged = _fit(_plan(module, build_aux_losses(_cfg(), module)))

    assert "val_context_nce" in logged
    assert logged["val_context_nce"] > 0


def test_the_diagnostics_are_reported_but_not_optimised():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)
    dm = _datamodule()
    batch = next(iter(dm.train_dataloader()))
    out = module._common_step(batch, "regression", "node")

    total, reported = composite(out, batch)

    assert "_context_nce_clean" in reported and "_context_nce_acc" in reported
    assert total.item() == pytest.approx(reported["context_nce"].item())


# --------------------------------------------------------------------------- the ablations


@pytest.mark.parametrize("negative_context", ["neighbourhood", "scattered"])
def test_both_negative_policies_produce_a_finite_loss(negative_context):
    module = _tiny_module()
    composite = build_aux_losses(_cfg(negative_context=negative_context), module)

    logged = _fit(_plan(module, composite))

    assert torch.isfinite(logged["train_context_nce"])


def test_unmatched_negatives_need_no_celltype_annotation():
    """`match_composition: False` is the ablation, and it must stay runnable on a dataset with no
    annotation at all -- otherwise the ablation is unavailable exactly where it is cheapest."""
    module = _tiny_module()
    composite = build_aux_losses(_cfg(match_composition=False), module)

    logged = _fit(_plan(module, composite), dm=_datamodule(with_celltype=False))

    assert torch.isfinite(logged["train_context_nce"])


# --------------------------------------------------------------------------- guards


def test_matching_without_an_annotation_fails_loudly():
    """Falling back to a single pseudo-type would leave composition free to separate positive from
    negative -- the term would run, log, and measure the thing it exists to rule out."""
    module = _tiny_module()
    composite = build_aux_losses(_cfg(), module)
    dm = _datamodule(with_celltype=False)
    batch = next(iter(dm.train_dataloader()))
    out = module._common_step(batch, "regression", "node")

    with pytest.raises(ValueError, match="dataset.celltype_key"):
        composite(out, batch)


def test_a_bank_no_larger_than_the_negative_count_is_rejected():
    """With no surplus every candidate is taken regardless of histogram, so the matching silently
    does nothing while the config still says `match_composition: True`."""
    with pytest.raises(ValueError, match="must exceed n_negatives"):
        build_aux_losses(_cfg(n_candidates=2, n_negatives=2), _tiny_module())


def test_an_unknown_negative_policy_is_rejected():
    with pytest.raises(ValueError, match="negative_context"):
        build_aux_losses(_cfg(negative_context="random"), _tiny_module())


def test_it_refuses_an_embedding_the_module_does_not_produce():
    module = _tiny_module()
    composite = build_aux_losses(_cfg(embedding="local"), module)
    dm = _datamodule()
    batch = next(iter(dm.train_dataloader()))
    out = module._common_step(batch, "regression", "node")

    with pytest.raises(ValueError, match="produced none"):
        composite(out, batch)
