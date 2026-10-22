"""One prediction from two parts: the additive decoder of the dual model (`model.decoder.additive`; decoder_plan.md, D3.1).

``y = W_l h_local + W_g h_global + b``. The stacked output keeps its two halves: the local half is the local decoder's
output alone, the global half is the sum (what the model reports, as under ``global_residual``). Pinned here:

* **the sum** -- the global half is the local decoder's output plus the global decoder's, for the same cells in the same
  order; the local half is the local decoder alone; the option adds no state and every weight starts as without it;
* **the gradient** -- unlike the residual head, the loss on the sum reaches both decoders and both embeddings; the local
  half alone trains the local side only;
* **the parts** -- the two outputs are handed over (the sum's own tensors, so the influence readouts can differentiate the
  head ``global_part``) and add up to the global half;
* **no leakage** -- a hidden entry reaches neither half, a visible one does;
* **inference** -- ``get_model_output`` reports the same sum, and the local decoder's output as the local prediction;
* **the evaluation** -- ``collect_head_predictions`` returns ``local``, ``global`` and ``global_part`` (the global
  decoder's output alone), carries the parts, and the attribution shares add up to 1;
* **the config** -- it needs the dual decoder at node level, excludes ``global_residual``, and makes the loss weights a
  choice (the default mean also trains the local part to predict the whole target alone);
* **pooled tokens** -- a patch's cells share the global part, the local part tells them apart (the un-pooling of T1b);
* **the trainer** -- a dual model trains through ``train()`` with it, the local decoder learning from the loss on the sum,
  also under a staged schedule (the local part alone first, then the global part on the sum with the local one frozen);
* **the name** -- the checkpoint carries it.
"""

import types

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation.masked_probing import (
    attribution_shares,
    blend_heads,
    collect_head_predictions,
    score_heads,
)
from interscale.tl.utils import get_model_filename_prefix

G, E = 8, 16


def _cfg(additive=True, residual=False, weights=(0.0, 1.0)):
    """A dual-decoder config; ``weights`` are ``optim.loss_weights`` (local, global): the sum only, by default."""
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
    cfg.model.decoder.additive = additive
    cfg.model.decoder.global_residual = residual
    cfg.optim.loss_weights["local"], cfg.optim.loss_weights["global"] = weights
    cfg.dataset.name = "d"
    return cfg


def _module(additive=True, residual=False, seed=0, strategy="gene", rate=0.3):
    from interscale.module.combined_module import DualDecoderCombinedModule

    torch.manual_seed(seed)
    return DualDecoderCombinedModule(
        cfg=_cfg(additive, residual),
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=rate,
        mask_strategy=strategy,
    ).eval()


def _batch(sizes=(9, 12), seed=0):
    rng = np.random.default_rng(seed)
    datas = []
    for n in sizes:
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        d = Data(x=torch.tensor(rng.random((n, G)), dtype=torch.float32), edge_index=torch.tensor([src, dst]))
        d.pos = torch.stack([torch.arange(n), torch.zeros(n)], dim=1).float() * 100.0
        d.gene_mask = torch.tensor(rng.random((n, G)) < 0.3)
        d.gene_mask[:, 0] = True
        d.mask = d.gene_mask.any(dim=1)
        datas.append(d)
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


# --------------------------------------------------------------------------- the sum


def test_the_option_adds_no_state_and_starts_as_without_it():
    plain, additive = _module(False), _module(True)
    assert plain.additive is False and additive.additive is True
    assert set(plain.state_dict()) == set(additive.state_dict())
    assert all(torch.equal(plain.state_dict()[k], additive.state_dict()[k]) for k in plain.state_dict())
    assert _cfg(additive=False).model.decoder.additive is False  # off by default in the config too


def test_the_global_half_is_the_sum_of_the_two_decoders_and_the_local_half_the_local_one_alone():
    plain, additive = _module(False), _module(True)  # the same weights at one seed
    batch = _batch()
    with torch.no_grad():
        off = plain._common_step(batch, "regression", "node")
        on = additive._common_step(batch, "regression", "node")
    n = len(off.y_pred) // 2
    local_off, global_off = off.y_pred[:n], off.y_pred[n:]
    local_on, global_on = on.y_pred[:n], on.y_pred[n:]
    assert torch.allclose(local_on, local_off, atol=1e-6)  # the local half is the local decoder's output, as before
    assert torch.allclose(global_on, local_off + global_off, atol=1e-5)
    assert (global_on - global_off).abs().max() > 1e-3  # and the sum is not the global decoder alone
    assert torch.equal(on.y_true, off.y_true) and torch.equal(on.view.padded_node_idx, off.view.padded_node_idx)


def test_the_combination_adds_the_two_outputs_without_cutting_the_gradient():
    additive, residual, plain = _module(True), _module(False, True), _module(False)
    local = torch.randn(5, G, requires_grad=True)
    raw = torch.randn(5, G, requires_grad=True)
    out = additive.combine_global(local, raw)
    assert torch.allclose(out, local + raw)
    out.sum().backward()
    assert torch.equal(local.grad, torch.ones_like(local)) and torch.equal(raw.grad, torch.ones_like(raw))
    # the residual head cuts the local side, the plain head is the global decoder alone
    cut = torch.randn(5, G, requires_grad=True)
    residual.combine_global(cut, raw).sum().backward()
    assert cut.grad is None
    assert torch.equal(plain.combine_global(local, raw), raw)
    assert additive.global_is_sum and residual.global_is_sum and not plain.global_is_sum


def test_the_additive_and_residual_heads_are_not_combined():
    from interscale.module.combined_module import DualDecoderCombinedModule

    with pytest.raises(ValueError, match="choose one"):
        DualDecoderCombinedModule(
            cfg=_cfg(True, True),
            n_input=G,
            n_output=G,
            n_embed=E,
            decoder_type=None,
            mask_percentage=0.3,
            mask_strategy="gene",
        )


# --------------------------------------------------------------------------- the gradient


def _grads(module):
    return {
        "local_decoder": [p.grad for p in module.local_module.decoder.parameters()],
        "global_decoder": [p.grad for p in module.global_module.decoder.parameters()],
        "gcn": [p.grad for name, p in module.local_module.named_parameters() if not name.startswith("decoder")],
    }


def _nonzero(grads):
    return bool(grads) and all(g is not None for g in grads) and any(torch.count_nonzero(g) > 0 for g in grads)


def _zero(grads):
    return all(g is None or torch.count_nonzero(g) == 0 for g in grads)


def test_the_loss_on_the_sum_reaches_both_decoders_and_the_local_embedding():
    module = _module(True).train()
    out = module._common_step(_batch(), "regression", "node")
    n = len(out.y_pred) // 2
    out.y_pred[n:].pow(2).mean().backward()  # the global half's loss alone: the sum
    grads = _grads(module)
    assert _nonzero(grads["local_decoder"]) and _nonzero(grads["global_decoder"])  # not so under global_residual
    assert _nonzero(grads["gcn"])

    residual = _module(False, True).train()
    out = residual._common_step(_batch(), "regression", "node")
    out.y_pred[n:].pow(2).mean().backward()
    assert _zero(_grads(residual)["local_decoder"])  # the residual head cuts it: the contrast this option exists for


def test_the_local_half_alone_trains_the_local_side_only_through_its_own_decoder():
    module = _module(True).train()
    out = module._common_step(_batch(), "regression", "node")
    n = len(out.y_pred) // 2
    out.y_pred[:n].pow(2).mean().backward()  # the local half's loss alone
    grads = _grads(module)
    assert _nonzero(grads["local_decoder"])
    assert _zero(grads["global_decoder"])  # the local half is the local decoder alone, not the sum


# --------------------------------------------------------------------------- the parts


def test_the_two_outputs_are_handed_over_and_add_up_to_the_global_half():
    module = _module(True)
    out = module._common_step(_batch(), "regression", "node")  # with gradients: a detached part shows as such
    n = len(out.y_pred) // 2
    parts = out.parts["global"]
    assert set(parts) == {"local", "global"}
    assert torch.allclose(parts["local"] + parts["global"], out.y_pred[n:], atol=1e-6)
    assert torch.allclose(parts["local"], out.y_pred[:n], atol=1e-6)
    assert parts["local"].requires_grad and parts["global"].requires_grad  # the sum's own tensors
    assert out.y_pred[n:].requires_grad
    assert "local" not in out.parts
    assert _module(False)._common_step(_batch(), "regression", "node").parts is None


# --------------------------------------------------------------------------- no leakage


def test_no_hidden_entry_reaches_either_half_and_a_visible_one_does():
    module = _module(True)
    batch = _batch()
    with torch.no_grad():
        before = module._common_step(batch, "regression", "node").y_pred
        hidden = batch.clone()
        hidden.x = batch.x.clone()
        hidden.x[batch.gene_mask] = torch.rand(int(batch.gene_mask.sum())) * 50.0
        assert torch.equal(module._common_step(hidden, "regression", "node").y_pred, before)
        visible = batch.clone()
        visible.x = batch.x.clone()
        col = int((~batch.gene_mask).any(dim=0).nonzero()[0])
        row = int((~batch.gene_mask[:, col]).nonzero()[0])
        visible.x[row, col] += 50.0
        after = module._common_step(visible, "regression", "node").y_pred
    assert not torch.allclose(after, before)


# --------------------------------------------------------------------------- inference


def test_inference_reports_the_sum_and_the_local_decoders_output_as_the_local_prediction(monkeypatch):
    import random

    from anndata import AnnData

    import interscale.model.combined_model as combined_model
    from interscale.model.combined_model import CombinedModel

    n, genes = 12, 4
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(n, genes)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.random((n, 2))
    CombinedModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    cfg = _cfg()
    cfg.model.global_component.parameters.max_seq_len = 8  # fewer tokens than cells: a subsample
    torch.manual_seed(0)
    model = CombinedModel(adata, cfg=cfg)
    model.module.eval()
    model.is_trained_ = True

    graph = Data(
        x=torch.tensor(adata.X),
        edge_index=torch.tensor([list(range(n - 1)), list(range(1, n))]),
        obs_names=torch.arange(1, n + 1),
    )
    monkeypatch.setattr(combined_model, "prepare_a2d_dataset", lambda cfg: lambda adata: [[graph], None])
    monkeypatch.setattr(combined_model, "attach_positional_inputs", lambda pyg, cfg: None)
    saved = {}
    monkeypatch.setattr(model, "save_evaluation_results", lambda adata, prefix, **out: saved.update(out) or adata)

    preds = {}
    for additive in (False, True):
        model.module.additive = additive
        random.seed(0)  # the same subsample both times
        model.get_model_output(adata.copy())
        preds[additive] = saved["y_pred_global_df"].to_numpy(dtype=float)
    local = saved["y_pred_local_df"].to_numpy(dtype=float)

    kept = ~np.isnan(preds[True]).any(axis=1)
    assert kept.sum() == 8
    assert np.allclose(preds[True][kept], local[kept] + preds[False][kept], atol=1e-5)
    assert not np.allclose(preds[True][kept], preds[False][kept], atol=1e-3)


# --------------------------------------------------------------------------- the evaluation


def test_collect_head_predictions_returns_the_global_part_and_the_parts_and_the_shares_follow():
    module = _module(True)
    graphs = [_batch(sizes=(14,), seed=0).get_example(0), _batch(sizes=(14,), seed=1).get_example(0)]
    pred = collect_head_predictions(
        module,
        graphs,
        prediction_task="regression",
        prediction_level="node",
        mask_strategy="gene",
        batch_size=1,
        device="cpu",
        spatial_unit_um=1.0,
    )
    assert set(pred.heads) == {"local", "global", "global_part"}
    assert set(pred.parts["global"]) == {"local", "global"}
    assert np.allclose(pred.heads["local"] + pred.heads["global_part"], pred.heads["global"], atol=1e-5)
    assert np.allclose(pred.parts["global"]["global"], pred.heads["global_part"])
    assert np.allclose(pred.parts["global"]["local"], pred.heads["local"])

    table = attribution_shares(pred, [f"g{k}" for k in range(G)])
    shares = table[(table["group"] == "all") & (table["depth_um"] == "all")].set_index("part")["cov_share"]
    assert set(shares.index) == {"local", "global"}
    assert shares.sum() == pytest.approx(1.0)

    # The third head flows through the scores, the blends and the report like any other: columns are `<head>_<metric>`.
    pred.heads["ref_own+context"] = pred.true * 0.0 + pred.true.mean(0)
    blended = blend_heads(pred, "ref_own+context")
    assert {"blend_local", "blend_global", "blend_global_part"} <= set(blended.heads)
    scores = score_heads(
        blended, [f"g{k}" for k in range(G)], center=pred.true.mean(0), gene_groups={"all": [f"g{k}" for k in range(G)]}
    )
    by_group = scores["by_group"]
    assert {"local_cosine", "global_cosine", "global_part_cosine", "blend_global_part_cosine"} <= set(by_group.columns)
    assert "global - local_cosine" in by_group.columns

    plain = collect_head_predictions(
        _module(False),
        graphs,
        prediction_task="regression",
        prediction_level="node",
        mask_strategy="gene",
        batch_size=1,
        device="cpu",
        spatial_unit_um=1.0,
    )
    assert set(plain.heads) == {"local", "global"} and plain.parts is None  # the default model's heads are unchanged


# --------------------------------------------------------------------------- pooled tokens


def test_pooled_tokens_share_the_global_part_within_a_patch_and_the_local_part_tells_the_cells_apart():
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = _cfg()
    params = cfg.model.global_component.parameters
    params.pooling.kind, params.pooling.size_um = "grid", 300.0
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", 1.0
    torch.manual_seed(0)
    module = DualDecoderCombinedModule(
        cfg=cfg,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="gene",
    ).eval()
    batch = _batch(sizes=(12,))  # 12 cells 100 um apart on a line: three of 300 um fall in a patch
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    n = len(out.y_pred) // 2
    parts = out.parts["global"]
    assert torch.allclose(parts["local"] + parts["global"], out.y_pred[n:], atol=1e-6)
    shared = [i for i in range(11) if torch.equal(parts["global"][i], parts["global"][i + 1])]
    assert shared, "no two neighbouring cells share a patch"  # the global part is the patch's
    assert all(
        not torch.equal(parts["local"][i], parts["local"][i + 1]) for i in shared
    )  # the local part is the cell's
    assert not torch.equal(out.y_pred[n + shared[0]], out.y_pred[n + shared[0] + 1])  # so the sum tells them apart


def test_the_influence_readouts_attribute_every_head_of_an_additive_model_including_the_global_part():
    """``global_part`` is a head the influence readouts differentiate: a detached one would raise there."""
    from interscale.evaluation.masked_probing import draw_masks
    from interscale.evaluation.receptive_field import response_influence

    side, response = 6, 5
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    gen = torch.Generator().manual_seed(0)
    data = Data(
        x=torch.rand(side * side, G, generator=gen) * 3.0,
        edge_index=(torch.cdist(pos, pos) == 1).nonzero().t(),
        pos=pos,
    )
    data.obs_names = torch.arange(side * side, dtype=torch.float32) + 1
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[response])
    centre = side * side // 2 + side // 2
    edges = pd.DataFrame({"sender": [centre + 1], "receiver": [centre], "range_class": "near", "program": "P>R"})
    found = response_influence(
        _module(True, strategy="gene", rate=1.0),
        [data],
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        spatial_unit_um=10.0,
        edges=edges,
        edge_genes=lambda program: [response],
        tolerance_um=0.5,
        n_projections=2,
        steps=8,
    )
    assert set(found["edges"]["head"]) == {"local", "global", "global_part"}
    check = found["check"].set_index(["head", "program"])["value"]
    # the library flags a gap over 0.1 (too few integration steps); the head alone lands near 0.06 with 8 steps
    assert (check < 0.1).all() and set(check.index.get_level_values("head")) == {"local", "global", "global_part"}


# --------------------------------------------------------------------------- the config


def test_the_config_needs_the_dual_decoder_at_node_level_and_excludes_the_residual_head():
    from interscale.config import _validate_additive

    _validate_additive(_cfg())
    _validate_additive(_cfg(additive=False, weights=(0.5, 0.5)))  # off: nothing is checked
    cfg = _cfg()
    cfg.model.decoder.dual_decoder = False
    with pytest.raises(ValueError, match="dual_decoder"):
        _validate_additive(cfg)
    cfg = _cfg()
    cfg.dataset.prediction_level = "graph"
    with pytest.raises(ValueError, match="prediction_level"):
        _validate_additive(cfg)
    with pytest.raises(ValueError, match="choose one"):
        _validate_additive(_cfg(True, True))


@pytest.mark.parametrize("weights", [(0.0, 1.0), (0.1, 0.9), (1.0, 0.0)])
def test_the_loss_weights_must_be_chosen_and_any_other_than_the_default_mean_will_do(weights):
    from interscale.config import _validate_additive

    _validate_additive(_cfg(weights=weights))
    with pytest.raises(ValueError, match="loss_weights"):
        _validate_additive(_cfg(weights=(0.5, 0.5)))


def test_a_staged_schedule_carries_its_own_weights_and_needs_none_from_the_config():
    from interscale.config import _validate_additive

    cfg = _cfg(weights=(0.5, 0.5))
    cfg.optim.schedule = "local_then_global"
    _validate_additive(cfg)


def test_config_loading_runs_the_check():
    from interscale.config import _validate

    cfg = _cfg()
    cfg.model.decoder.dual_decoder = False
    with pytest.raises(ValueError, match="dual_decoder"):
        _validate(cfg)


# --------------------------------------------------------------------------- refusals


def test_a_model_without_two_decoders_refuses_to_train_with_the_option():
    from interscale.train._training import NodeMaskingTrainingPlan

    local_only = types.SimpleNamespace(_cfg=_cfg(), module=types.SimpleNamespace())
    with pytest.raises(ValueError, match="additive"):
        NodeMaskingTrainingPlan._refuse_unsupported_options(local_only)
    has_it = types.SimpleNamespace(_cfg=_cfg(), module=types.SimpleNamespace(additive=True, regime=None))
    NodeMaskingTrainingPlan._refuse_unsupported_options(has_it)
    plain = types.SimpleNamespace(_cfg=_cfg(additive=False), module=types.SimpleNamespace())
    NodeMaskingTrainingPlan._refuse_unsupported_options(plain)


# --------------------------------------------------------------------------- the name


def test_the_checkpoint_name_carries_it():
    names = {
        (additive, residual): get_model_filename_prefix(_cfg(additive, residual, (0.0, 1.0)), True, True)
        for additive, residual in ((False, False), (True, False), (False, True))
    }
    assert len(set(names.values())) == 3
    assert "var-add+lw0-1_" in names[(True, False)] and "var-res+lw0-1_" in names[(False, True)]
    plain = _cfg(additive=False, weights=(0.5, 0.5))
    assert "add" not in get_model_filename_prefix(plain, True, True)  # the default adds nothing
    cfg = _cfg()
    cfg.optim.center_targets = True
    assert get_model_filename_prefix(cfg, True, True).endswith("var-ctr+add+lw0-1_")


# --------------------------------------------------------------------------- through the trainer


def _trained(tmp_path, schedule="joint"):
    """A dual additive model trained through ``train()`` on four toy slides; the model and the initial local decoder.

    A joint run takes the loss on the sum only (weights 0 / 1); a staged schedule sets the weights per phase.
    """
    import scanpy as sc
    from anndata import AnnData

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.model.combined_model import CombinedModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    per = 30
    n = per * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, per))
    split = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "additive.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "additive.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  local_component:\n    name: GCN\n  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 64\n      num_layers: 1\n"
        "      n_heads: 2\n      long_range_attention: False\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n    additive: True\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 4\n"
        f"  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n  schedule: {schedule}\n"
        f"  loss_weights:\n    local: {0.0 if schedule == 'joint' else 0.5}\n    global: {1.0 if schedule == 'joint' else 0.5}\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: additivetest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
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
    net = CombinedModel(adata, cfg=cfg)
    assert net.module.additive
    before = {k: v.clone() for k, v in net.module.local_module.decoder.state_dict().items()}
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    net.train(max_epochs=4, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    return net, before


def test_a_dual_model_trains_through_the_trainer_with_the_loss_on_the_sum(tmp_path):
    net, before = _trained(tmp_path)
    rows = [row for row in net.history_.history if "val_loss" in row]
    assert rows and all(np.isfinite(row["val_loss"]) for row in rows)
    # the weights 0 / 1: the objective is the loss on the sum, whatever the local half scores alone
    assert all(row["val_loss"] == pytest.approx(row["val_global_loss"], rel=1e-5) for row in rows)
    assert all(np.isfinite(row["val_local_loss"]) for row in rows)
    after = net.module.local_module.decoder.state_dict()
    # the local half has weight 0, so the local decoder learns only through the loss on the sum
    assert any(not torch.equal(before[k], after[k].cpu()) for k in before)
    assert any("var-add+lw0-1_" in p.name for p in (tmp_path / "out").iterdir())


def test_the_additive_decoder_composes_with_a_staged_schedule(tmp_path):
    """The local part alone first (the global component frozen), then the global part on the sum (the local frozen)."""
    net, _ = _trained(tmp_path, schedule="local_then_global")
    first, second = net.phase_summaries_
    assert first["drift_global"] == 0.0 and first["drift_local"] > 0.0  # phase 1 trains the local side alone
    assert second["drift_local"] == 0.0 and second["drift_global"] > 0.0  # phase 2 the global side, on the sum
    assert all(np.isfinite(v) for phase in net.phase_summaries_ for v in phase.values() if isinstance(v, float))
