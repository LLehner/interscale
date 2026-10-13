"""The reconstruction loss on gene-centred values (`optim.center_targets`; attention_plan.md, round 3).

On raw expression most of every spot's vector is the profile all spots share, so a raw cosine is
nearly satisfied by the mean profile and barely rewards context. Pinned here: the loss is the SCE of
the values minus each gene's training mean, over the same masked entries as before; predicting the
mean profile scores the worst cosine instead of the best one; the metrics stay on the raw values;
the dual decoder's split losses are centred too; the means come from the training graphs; and the
config rejects losses the shift cannot change.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from interscale.module.base import StepOutput, ViewOutput
from interscale.tl.masking import masked_loss, sample_gene_mask
from interscale.train._trainingplans import TrainingPlan
from interscale.train._utils import train_gene_means
from interscale.train.losses import SCELoss, scaled_cosine_error

G = 20


class _Stub(nn.Module):
    n_output = G


def _plan(center=None, loss="SCELoss"):
    return TrainingPlan(_Stub(), "regression", "node", loss, "gene", batch_size=2, sce_gamma=1.0, target_center=center)


def _data(n=50, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mean = torch.rand(G, generator=gen) * 3 + 1.0  # a shared profile that dominates every spot
    y_true = mean + 0.3 * torch.randn(n, G, generator=gen)
    entry_mask = sample_gene_mask(n, G, 0.3, generator=gen)
    entry_mask[:, 0] = True
    return mean, y_true, entry_mask


def test_the_loss_is_the_sce_of_the_centred_values_over_the_masked_entries():
    mean, y_true, entry_mask = _data()
    y_pred = y_true + 0.2 * torch.randn(y_true.shape, generator=torch.Generator().manual_seed(5))
    plan = _plan(mean)

    loss, metrics = plan._regression_metrics(y_pred, y_true, "val", plan.valid_metrics, entry_mask=entry_mask)

    expected = masked_loss(SCELoss(1.0), "SCELoss", y_pred - mean, y_true - mean, entry_mask)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    # The metrics are the raw ones: the logged SCE is the uncentred cosine error.
    assert metrics["val_scaled_cosine_error"].item() == pytest.approx(
        scaled_cosine_error(y_pred, y_true).mean().item(), abs=1e-6
    )


def test_predicting_the_mean_profile_scores_worst_instead_of_best():
    mean, y_true, entry_mask = _data()
    constant = mean.expand_as(y_true).clone()
    raw_plan, centred_plan = _plan(None), _plan(mean)

    raw, _ = raw_plan._regression_metrics(constant, y_true, "val", raw_plan.valid_metrics, entry_mask=entry_mask)
    centred, _ = centred_plan._regression_metrics(
        constant, y_true, "val", centred_plan.valid_metrics, entry_mask=entry_mask
    )
    halfway = mean + 0.5 * (y_true - mean)  # the right deviations at half their size
    better, _ = centred_plan._regression_metrics(
        halfway, y_true, "val", centred_plan.valid_metrics, entry_mask=entry_mask
    )

    assert raw.item() < 0.05  # raw: the mean profile alone is nearly a perfect cosine
    assert centred.item() == pytest.approx(1.0)  # centred: it carries no direction at all
    assert better.item() == pytest.approx(0.0, abs=1e-5)


def test_the_dual_decoders_split_losses_are_centred_too():
    mean, y_true, entry_mask = _data()
    y_pred = y_true + 0.1
    seen = {}

    class _DualStub(_Stub):
        def compute_separate_losses(self, loss_fn, loss_type, y_pred, y_true, entry_mask):
            seen["y_pred"], seen["y_true"] = y_pred, y_true
            return {"local_loss": None, "global_loss": None, "combined_loss": None, "kl_loss": None}

    plan = TrainingPlan(
        _DualStub(), "regression", "node", "SCELoss", "gene", batch_size=2, sce_gamma=1.0, target_center=mean
    )
    out = StepOutput(y_pred=y_pred, y_true=y_true, entry_mask=entry_mask, views=[ViewOutput()])
    plan._forward_views = lambda batch: out
    plan._aux_losses = lambda *args: None
    plan.log = lambda *args, **kwargs: None
    plan.log_dict = lambda *args, **kwargs: None
    plan._step(types.SimpleNamespace(), "val")

    assert torch.allclose(seen["y_pred"], y_pred - mean) and torch.allclose(seen["y_true"], y_true - mean)


def test_the_means_come_from_the_training_graphs():
    graphs = [types.SimpleNamespace(x=torch.full((3, G), 1.0)), types.SimpleNamespace(x=torch.full((1, G), 5.0))]
    assert torch.allclose(train_gene_means(graphs), torch.full((G,), 2.0))


def test_without_centring_the_plan_is_unchanged():
    mean, y_true, entry_mask = _data()
    y_pred = y_true * 1.1
    plan = _plan(None)
    loss, _ = plan._regression_metrics(y_pred, y_true, "val", plan.valid_metrics, entry_mask=entry_mask)
    assert loss.item() == pytest.approx(
        masked_loss(SCELoss(1.0), "SCELoss", y_pred, y_true, entry_mask).item(), abs=1e-7
    )
    assert "target_center" not in plan.state_dict()


@pytest.mark.parametrize(
    ("task", "loss", "match"),
    [("regression", "MSELoss", "row-normalising"), ("classification", "CrossEntropy", "regression")],
)
def test_the_config_rejects_centring_where_it_changes_nothing(task, loss, match):
    from interscale.config import _validate_center_targets, get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.prediction_task, cfg.optim.loss, cfg.optim.center_targets = task, loss, True
    with pytest.raises(ValueError, match=match):
        _validate_center_targets(cfg)
    cfg.dataset.prediction_task, cfg.optim.loss = "regression", "SCELoss"
    _validate_center_targets(cfg)


def test_the_checkpoint_name_carries_the_centring():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.tl.utils import get_model_filename_prefix

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.dataset.name = "d"
    assert "var-" not in get_model_filename_prefix(cfg, True, True)
    cfg.optim.center_targets = True
    for components in ((True, True), (False, True), (True, False)):
        assert get_model_filename_prefix(cfg, *components).endswith("var-ctr_")


def test_config_loading_runs_the_check():
    from interscale.config import _validate, get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.prediction_task, cfg.optim.loss, cfg.optim.center_targets = "regression", "MSELoss", True
    with pytest.raises(ValueError, match="row-normalising"):
        _validate(cfg)
