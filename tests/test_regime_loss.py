"""Regime-assigned losses (``optim.regime_loss``; decoder_plan.md, D2.2).

Both heads of a dual decoder predict the full target of every hidden cell, so the global head can do no better
than copy the local one where the local component can see. ``regime_loss`` gives each head the hidden cells of
its regime: *near* (a visible cell within ``depth_um``) for the local head, *deep* for the global one.

Pinned here, on a lattice with a square hole whose depths are known in closed form:

* **depth** -- ``hidden_depth`` is the distance to the nearest visible cell of the same graph (chunked or not,
  graphs apart, ``inf`` when nothing is visible); ``regime_rows`` cuts it at the threshold;
* **the masks** -- each head's loss mask is the hidden entries of its regime's cells, in token order, the metric
  mask (``entry_mask``) untouched;
* **the loss** -- each head's loss is the mean over its own cells and the heads are weighed 0.5 / 0.5 whatever
  the cells' numbers; a head's loss does not read the other regime's targets; an empty regime is a zero, not
  a NaN; ``all`` / ``all`` is the loss as before;
* **the switch** -- off by default and then nothing changes; the config refuses what cannot work; checkpoint
  names tell the regimes apart; a model without two heads refuses to train with it;
* **through the trainer** -- a dual-decoder CombinedModel trains on discs, finite, and logs the cells of each loss.
"""

import types

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

import interscale.tl.depth as depth_module
from interscale.tl.depth import RegimeLoss, hidden_depth, regime_rows
from interscale.tl.masking import masked_loss
from interscale.train.losses import SCELoss

G, SIDE, SPACING = 8, 11, 100.0
LO, HI = 2, 9  # the hidden square: rows and columns 2..8, so 7 x 7 cells with visible cells all around


def _lattice(block=(LO, HI), seed=0, shift=0.0):
    """A SIDE x SIDE lattice with the square ``block`` hidden whole; returns the Data."""
    ii, jj = torch.meshgrid(torch.arange(SIDE), torch.arange(SIDE), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING + shift
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    gen = torch.Generator().manual_seed(seed)
    data = Data(x=torch.rand(SIDE * SIDE, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    lo, hi = block
    data.mask = ((ii >= lo) & (ii < hi) & (jj >= lo) & (jj < hi)).flatten()
    data.ii, data.jj = ii.flatten(), jj.flatten()
    return data


def _closed_form_depth(block=(LO, HI)):
    """The depth of each hidden cell in um: the steps to the nearest visible row or column, times the spacing."""
    data = _lattice(block)
    lo, hi = block
    steps = torch.stack([data.ii - (lo - 1), hi - data.ii, data.jj - (lo - 1), hi - data.jj]).min(dim=0).values
    return torch.where(data.mask, steps.float() * SPACING, torch.zeros(()))


def _batch(block=(LO, HI), seed=0):
    data = _lattice(block, seed)
    del data.ii, data.jj
    batch = Batch.from_data_list([data])
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


# --------------------------------------------------------------------------- depth


def test_the_depth_of_a_hidden_cell_is_its_distance_to_the_nearest_visible_one():
    data = _lattice()
    depth = hidden_depth(data.pos, data.mask, torch.zeros(data.num_nodes, dtype=torch.long))
    assert torch.allclose(depth.float(), _closed_form_depth())
    hidden = data.mask
    assert sorted(depth[hidden].unique().tolist()) == [100.0, 200.0, 300.0, 400.0]
    assert [(depth[hidden] == d).sum().item() for d in (100.0, 200.0, 300.0, 400.0)] == [24, 16, 8, 1]
    assert not depth[~hidden].any()  # a visible cell has depth 0


def test_chunking_the_distances_changes_no_depth(monkeypatch):
    data = _lattice()
    graph = torch.zeros(data.num_nodes, dtype=torch.long)
    whole = hidden_depth(data.pos, data.mask, graph)
    monkeypatch.setattr(depth_module, "DEPTH_CHUNK", 5)
    assert torch.equal(hidden_depth(data.pos, data.mask, graph), whole)


def test_graphs_are_measured_apart_and_a_graph_hidden_completely_is_infinitely_deep():
    one, two = _lattice(), _lattice(block=(0, SIDE), shift=0.0)  # the second is hidden completely
    pos = torch.cat([one.pos, two.pos])  # the same places: a visible cell of one must not serve the other
    hidden = torch.cat([one.mask, two.mask])
    graph = torch.cat([torch.zeros(len(one.pos), dtype=torch.long), torch.ones(len(two.pos), dtype=torch.long)])
    depth = hidden_depth(pos, hidden, graph)
    assert torch.allclose(depth[: len(one.pos)].float(), _closed_form_depth())
    assert torch.isinf(depth[len(one.pos) :]).all()
    assert not hidden_depth(one.pos, torch.zeros_like(one.mask), torch.zeros(len(one.pos), dtype=torch.long)).any()


def test_the_evaluation_modules_read_the_same_function():
    from interscale.evaluation import online_probes

    assert online_probes.hidden_depth is hidden_depth


def test_the_regimes_cut_the_depth_at_the_threshold():
    depth = torch.tensor([0.0, 100.0, 290.0, 300.0, float("inf")], dtype=torch.float64)
    assert regime_rows(depth, 290.0, "near").tolist() == [True, True, True, False, False]
    assert regime_rows(depth, 290.0, "deep").tolist() == [False, False, False, True, True]
    assert regime_rows(depth, 290.0, "all").all()
    with pytest.raises(ValueError, match="near"):
        regime_rows(depth, 290.0, "middle")


def test_the_threshold_is_in_the_units_of_the_coordinates():
    cfg = _cfg(regime="depth")
    cfg.dataset.spatial_unit_um = 2.0  # one unit of `pos` is 2 um
    assert RegimeLoss.from_config(cfg).threshold == 145.0
    assert RegimeLoss.from_config(_cfg()) is None


# --------------------------------------------------------------------------- the masks


def _cfg(regime="none", local="near", glob="deep", depth_um=290.0):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 200
    cfg.model.decoder.dual_decoder = True
    cfg.optim.loss = "SCELoss"
    cfg.optim.regime_loss.kind = regime
    cfg.optim.regime_loss.depth_um = depth_um
    cfg.optim.regime_loss.local_cells, cfg.optim.regime_loss.global_cells = local, glob
    cfg.dataset.prediction_task, cfg.dataset.prediction_level = "regression", "node"
    cfg.dataset.mask_strategy, cfg.dataset.mask_percentage = "node", 0.3
    cfg.dataset.mask_block_fraction, cfg.dataset.mask_block_radius_um = 0.3, 400.0
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", 1.0
    cfg.dataset.name = "d"
    return cfg


def _module(regime="depth", local="near", glob="deep", seed=0):
    from interscale.module.combined_module import DualDecoderCombinedModule

    torch.manual_seed(seed)
    return DualDecoderCombinedModule(
        cfg=_cfg(regime, local, glob),
        n_input=G,
        n_output=G,
        n_embed=16,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="node",
    ).eval()


def _regimes(block=(LO, HI), threshold=290.0):
    depth = _closed_form_depth(block)
    hidden = _lattice(block).mask
    return hidden & (depth <= threshold), hidden & (depth > threshold)


def test_each_head_gets_the_hidden_entries_of_its_regimes_cells_and_the_metrics_keep_all_of_them():
    batch = _batch()
    off, on = _module(regime="none"), _module()
    assert off.regime is None and on.regime is not None
    with torch.no_grad():
        base = off._common_step(batch, "regression", "node")
        out = on._common_step(batch, "regression", "node")
    assert base.loss_mask is None  # off: the loss uses the metric mask, as before
    assert torch.equal(out.entry_mask, base.entry_mask)
    assert torch.equal(out.y_pred, base.y_pred) and torch.equal(out.y_true, base.y_true)

    near, deep = _regimes()
    assert near.sum() == 40 and deep.sum() == 9
    order = out.view.padded_node_idx
    n = len(order)
    assert out.loss_mask.shape == out.entry_mask.shape == (2 * n, G)
    local_rows, global_rows = out.loss_mask[:n].any(dim=1), out.loss_mask[n:].any(dim=1)
    assert torch.equal(local_rows, near[order]) and torch.equal(global_rows, deep[order])
    # Whole rows, as cell masking gives: a row of the loss mask is its entry mask row or nothing.
    assert torch.equal(out.loss_mask[:n][local_rows], out.entry_mask[:n][local_rows])
    assert not out.loss_mask[:n][~local_rows].any() and not out.loss_mask[n:][~global_rows].any()


def test_the_regimes_can_be_swapped_or_both_taken():
    batch = _batch()
    near, deep = _regimes()
    for local, glob, want_local, want_global in (
        ("deep", "near", deep, near),
        ("all", "deep", near | deep, deep),
        ("all", "all", near | deep, near | deep),
    ):
        module = _module(local=local, glob=glob)
        with torch.no_grad():
            out = module._common_step(batch, "regression", "node")
        n = len(out.view.padded_node_idx)
        order = out.view.padded_node_idx
        assert torch.equal(out.loss_mask[:n].any(dim=1), want_local[order])
        assert torch.equal(out.loss_mask[n:].any(dim=1), want_global[order])


def test_a_slide_the_transformer_subsamples_is_still_measured_over_all_its_cells():
    """Depth is the distance to a visible cell of the slide, also one that `pad_batch` left out of the transformer."""
    module = _module()
    module.global_module.max_seq_len = 60  # fewer tokens than the 121 cells
    batch = _batch()
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    order = out.view.padded_node_idx
    assert len(order) == 60
    near, deep = _regimes()
    n = len(order)
    assert torch.equal(out.loss_mask[:n].any(dim=1), near[order])
    assert torch.equal(out.loss_mask[n:].any(dim=1), deep[order])


# --------------------------------------------------------------------------- the loss


def _plan(module, weights=None):
    from interscale.train._trainingplans import TrainingPlan

    plan = TrainingPlan(
        module, "regression", "node", "SCELoss", "gene", batch_size=1, sce_gamma=1.0, loss_weights=weights
    )
    plan.logged = {}
    plan.log = lambda name, value, **kw: plan.logged.__setitem__(name, value)
    plan.log_dict = lambda metrics, **kw: plan.logged.update(metrics)
    return plan


def test_each_heads_loss_is_the_mean_over_its_own_cells_and_the_heads_are_weighed_alike():
    module = _module()
    plan = _plan(module)
    batch = _batch()
    loss = plan._step(batch, "val")
    out = module._common_step(batch, "regression", "node")
    n = len(out.view.padded_node_idx)
    order = out.view.padded_node_idx
    near, deep = _regimes()
    sce = SCELoss(1.0)
    local = masked_loss(sce, "SCELoss", out.y_pred[:n][near[order]], out.y_true[:n][near[order]])
    glob = masked_loss(sce, "SCELoss", out.y_pred[n:][deep[order]], out.y_true[n:][deep[order]])
    assert plan.logged["val_local_loss"].item() == pytest.approx(local.item(), abs=1e-6)
    assert plan.logged["val_global_loss"].item() == pytest.approx(glob.item(), abs=1e-6)
    assert loss.item() == pytest.approx(0.5 * (local + glob).item(), abs=1e-6)
    assert plan.logged["val_loss"].item() == pytest.approx(loss.item(), abs=1e-6)
    # 40 near cells and 9 deep ones: one mean over the rows of both heads would weigh the heads 40 : 9.
    stacked = masked_loss(sce, "SCELoss", out.y_pred, out.y_true, out.loss_mask)
    assert abs(stacked.item() - loss.item()) > 1e-3
    assert plan.logged["val_regime_cells_local"].item() == 40 and plan.logged["val_regime_cells_global"].item() == 9


def test_the_metrics_do_not_change_with_the_regimes():
    batch = _batch()
    off_plan, on_plan = _plan(_module(regime="none")), _plan(_module())
    off_plan._step(batch, "val")
    on_plan._step(batch, "val")
    for key in (
        "val_masked_scaled_cosine_error_local",
        "val_masked_scaled_cosine_error_global",
        "val_masked_scaled_cosine_error",
        "val_scaled_cosine_error",
    ):
        assert on_plan.logged[key].item() == pytest.approx(off_plan.logged[key].item(), abs=1e-6), key
    assert "val_regime_cells_local" not in off_plan.logged


def test_a_head_does_not_read_the_other_regimes_targets():
    """Hidden values are masked in the input, so changing a target moves no prediction, only the loss that scores it."""
    batch = _batch()
    near, deep = _regimes()
    base = _plan(_module())
    base._step(batch, "val")
    cases = ((deep, "val_local_loss", "val_global_loss"), (near, "val_global_loss", "val_local_loss"))
    for changed, same, other in cases:
        moved = batch.clone()
        moved.x = batch.x.clone()
        moved.x[changed] += 3.0
        plan = _plan(_module())
        plan._step(moved, "val")
        assert plan.logged[same].item() == base.logged[same].item()
        assert plan.logged[other].item() != base.logged[other].item()


def test_all_all_is_the_loss_as_before():
    batch = _batch()
    stacked = _plan(_module(regime="none"))._step(batch, "val")
    weighed = _plan(_module(local="all", glob="all"))._step(batch, "val")
    assert weighed.item() == pytest.approx(stacked.item(), abs=1e-6)


def test_a_head_without_a_cell_in_its_regime_gets_a_zero_and_no_gradient():
    module = _module().train()
    batch = _batch(block=(4, 7))  # a 3 x 3 hole, 200 um deep at most: nothing is deep
    near, deep = _regimes(block=(4, 7))
    assert near.sum() == 9 and deep.sum() == 0
    plan = _plan(module)
    loss = plan._step(batch, "train")
    assert torch.isfinite(loss)
    assert plan.logged["train_global_loss"].item() == 0.0
    loss.backward()
    grads = {n: (p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in module.named_parameters()}
    assert all(torch.count_nonzero(g) == 0 for n, g in grads.items() if n.startswith("global_module."))
    assert any(torch.count_nonzero(g) > 0 for n, g in grads.items() if n.startswith("local_module."))
    assert plan.logged["train_regime_cells_global"].item() == 0


def test_the_global_head_trains_on_deep_cells_only_and_the_local_head_on_near_cells_only():
    near, deep = _regimes()
    batch = _batch()
    for weights, trained, silent in (
        ({"local": 0.0, "global": 1.0}, "global_module.decoder", "local_module.decoder"),
        ({"local": 1.0, "global": 0.0}, "local_module.decoder", "global_module.decoder"),
    ):
        module = _module().train()
        _plan(module, weights)._step(batch, "train").backward()
        grads = {n: p.grad for n, p in module.named_parameters() if p.grad is not None}
        assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith(trained))
        assert all(g.abs().sum() == 0 for n, g in grads.items() if n.startswith(silent))
    # The decoder gradient of the global head is the one of a loss on the deep cells alone.
    module = _module().train()
    plan = _plan(module, {"local": 0.0, "global": 1.0})
    plan._step(batch, "train").backward()
    got = module.global_module.decoder.decoder.weight.grad.clone()
    reference = _module(local="all", glob="all").train()
    out = reference._common_step(batch, "regression", "node")
    n = len(out.view.padded_node_idx)
    order = out.view.padded_node_idx
    rows = deep[order]
    masked_loss(SCELoss(1.0), "SCELoss", out.y_pred[n:][rows], out.y_true[n:][rows]).backward()
    assert torch.allclose(got, reference.global_module.decoder.decoder.weight.grad, atol=1e-6)


# --------------------------------------------------------------------------- the switch


def test_off_by_default():
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    assert cfg.optim.regime_loss.kind == "none"
    assert (cfg.optim.regime_loss.local_cells, cfg.optim.regime_loss.global_cells) == ("near", "deep")
    assert cfg.optim.regime_loss.depth_um == 290.0
    assert not depth_module.regime_loss_on(cfg)
    assert depth_module.regime_loss_on(_cfg(regime="depth"))


def _validate(cfg):
    from interscale.config import _validate_regime_loss

    _validate_regime_loss(cfg)


def test_the_config_takes_discs_wider_than_the_threshold_and_cell_masking_beside_them():
    _validate(_cfg(regime="depth"))
    _validate(_cfg(regime="depth", local="all", glob="all"))
    cfg = _cfg(regime="depth")
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 0.0
    cfg.dataset.mask_genes = []
    cfg.dataset.mask_percentage = 0.3
    _validate(cfg)  # discs hide whole rows under gene masking too
    _validate(_cfg(regime="none"))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: setattr(c.optim.regime_loss, "kind", "length"), "kind must be one of"),
        (lambda c: setattr(c.optim.regime_loss, "depth_um", 0.0), "depth_um must be > 0"),
        (lambda c: setattr(c.optim.regime_loss, "local_cells", "middle"), "local_cells must be one of"),
        (lambda c: setattr(c.optim.regime_loss, "global_cells", "far"), "global_cells must be one of"),
        (lambda c: setattr(c.model.decoder, "dual_decoder", False), "dual_decoder"),
        (lambda c: setattr(c.dataset, "prediction_level", "graph"), "regression"),
        (lambda c: setattr(c.optim, "loss", "none"), "own loss"),
        (lambda c: setattr(c.dataset, "mask_genes", ["g0"]), "no cell whole"),
        (lambda c: setattr(c.dataset, "spatial_key", None), "spatial_key"),
        (lambda c: setattr(c.dataset, "mask_block_radius_um", 290.0), "nothing to learn from"),
        (lambda c: setattr(c.dataset, "mask_block_fraction", 0.0), "no spatial blocks"),
    ],
)
def test_the_config_refuses_what_the_regimes_cannot_do(change, message):
    cfg = _cfg(regime="depth")
    change(cfg)
    with pytest.raises(ValueError, match=message):
        _validate(cfg)


def test_the_config_refuses_gene_masking_without_discs():
    cfg = _cfg(regime="depth", local="all", glob="all")
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_block_fraction, cfg.dataset.mask_block_radius_um = 0.0, 0.0
    with pytest.raises(ValueError, match="only spatial blocks"):
        _validate(cfg)


def test_a_regime_that_never_needs_deep_cells_needs_no_wide_discs():
    cfg = _cfg(regime="depth", local="near", glob="near")
    cfg.dataset.mask_block_radius_um = 200.0
    _validate(cfg)


def test_checkpoint_names_tell_the_regimes_and_the_threshold_apart():
    from interscale.tl.utils import get_model_filename_prefix

    names = {
        key: get_model_filename_prefix(_cfg(*key[:1], local=key[1], glob=key[2], depth_um=key[3]), True, True)
        for key in (
            ("none", "near", "deep", 290.0),
            ("depth", "near", "deep", 290.0),
            ("depth", "all", "deep", 290.0),
            ("depth", "near", "deep", 500.0),
        )
    }
    assert len(set(names.values())) == 4
    assert "rgm" not in names[("none", "near", "deep", 290.0)]
    assert "rgm290-near-deep" in names[("depth", "near", "deep", 290.0)]


def test_a_model_without_two_heads_refuses_to_train_with_regimes():
    from interscale.train._training import NodeMaskingTrainingPlan

    cfg = _cfg(regime="depth")
    refuse = NodeMaskingTrainingPlan._refuse_unsupported_options
    with pytest.raises(ValueError, match="regime_loss"):
        refuse(types.SimpleNamespace(_cfg=cfg, module=types.SimpleNamespace()))
    with pytest.raises(ValueError, match="regime_loss"):
        refuse(types.SimpleNamespace(_cfg=cfg, module=types.SimpleNamespace(regime=None)))
    refuse(types.SimpleNamespace(_cfg=cfg, module=types.SimpleNamespace(regime=object())))
    refuse(types.SimpleNamespace(_cfg=_cfg(), module=types.SimpleNamespace()))


def test_train_itself_refuses_a_global_only_model_with_regimes():
    from anndata import AnnData

    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    n = 12
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.random((n, 2))
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    cfg = _cfg(regime="depth")
    cfg.model.global_component.parameters.type_gex_embedding = "Linear"
    model = GlobalModel(adata, cfg=cfg)
    with pytest.raises(ValueError, match="regime_loss"):
        model.train(max_epochs=1, wandb_use=False)


# --------------------------------------------------------------------------- through the trainer


def test_a_dual_decoder_model_trains_on_discs_with_the_regimes(tmp_path):
    import scanpy as sc
    from anndata import AnnData

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, block_mask_kwargs
    from interscale.model.combined_model import CombinedModel
    from interscale.tl import prepare_geome_dataset

    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    lattice = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.float32) * SPACING
    samples = ["s1", "s2", "s3", "s4"]
    rng = np.random.default_rng(0)
    n = len(lattice) * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = np.tile(lattice, (len(samples), 1))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, len(lattice)))
    split = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "regime.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "regime.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  local_component:\n    name: GCN\n  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 200\n      num_layers: 1\n"
        "      n_heads: 2\n      long_range_attention: True\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 4\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        "  regime_loss:\n    kind: depth\n    depth_um: 290.0\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: regimetest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.0\n  mask_block_radius_um: 400.0\n  mask_block_fraction: 0.3\n"
        "  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 150\n    library_key: sample\n"
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
    assert net.module.regime is not None and net.module.regime.threshold == 290.0
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg,
        num_workers=0,
        batch_size=2,
        mask_percentage=0.0,
        mask_strategy="node",
        learning_type="node",
        **block_mask_kwargs(cfg),
    )
    net.train(max_epochs=4, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    rows = [row for row in net.history_.history if "val_loss" in row]
    assert rows and all(np.isfinite(row["val_loss"]) for row in rows)
    logged = {key for row in net.history_.history for key in row}
    assert {"train_regime_cells_local", "train_regime_cells_global", "val_regime_cells_global"} <= logged
    cells = [row["train_regime_cells_global"] for row in net.history_.history if "train_regime_cells_global" in row]
    assert max(cells) > 0  # discs of 400 um leave cells deeper than 290 um
    assert any("rgm290-near-deep" in p.name for p in (tmp_path / "out").iterdir())
