"""``optim.dual_mask`` (2.1, ``.claude/comparison.md``): a second pass per training step with the other kind of mask, its loss added.

The second batch carries the other mask (cells hidden whole under channel training, one shared channel set per graph under cell training) from a
private generator; the module's strategy is switched for that pass only; the run trains, logs the term, tags the checkpoint; validators refuse the
same kind twice and a dual-decoder model.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

G = 6


def _plan(strategy, module=None):
    from interscale.train._trainingplans import TrainingPlan

    module = module if module is not None else torch.nn.Linear(2, 2)
    module.n_output = G
    module.mask_strategy = "gene" if strategy == "node" else "node"
    return TrainingPlan(module, "regression", "node", "SCELoss", False, 1, None, None,
                        dual_mask={"weight": 1.0, "strategy": strategy, "rate": 0.4, "channel_fraction": 0.5, "seed": 3})  # fmt: skip


def _batch():
    graphs = [Data(x=torch.rand(30, G), pos=torch.rand(30, 2)) for _ in range(2)]
    for g in graphs:
        g.mask = torch.zeros(30, dtype=torch.bool)
        g.gene_mask = torch.zeros(30, G, dtype=torch.bool)
    return Batch.from_data_list(graphs)


def test_the_second_batch_carries_the_other_kind_of_mask_from_a_private_generator():
    batch = _batch()
    node_plan, channel_plan = _plan("node"), _plan("channel")
    state = torch.random.get_rng_state()
    node, kind = node_plan._dual_masked(batch)
    assert kind == "node" and not hasattr(node, "gene_mask") and 0 < int(node.mask.sum()) < 60
    chan, kind = channel_plan._dual_masked(batch)
    assert kind == "gene" and chan.gene_mask.shape == (60, G)
    for b in range(2):
        rows = chan.gene_mask[chan.batch == b]
        assert torch.equal(rows, rows[:1].expand_as(rows)) and int(rows[0].sum()) == 3  # one shared set per graph, half the genes
        assert bool(chan.mask[chan.batch == b].all())
    assert torch.equal(state, torch.random.get_rng_state())  # the global generator is not touched
    assert not batch.mask.any() and not batch.gene_mask.any()  # the original batch keeps its masks


def test_a_channel_trained_model_with_a_cell_mask_pass_trains_and_says_so(tmp_path):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    per = 50
    adata = AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 400, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "dm.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "dm.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 256\n      num_layers: 1\n      n_heads: 2\n      long_range_attention: False\n      type_gex_embedding: Linear\n"
        f"  decoder:\n    type: linear\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  center_targets: True\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n  dual_mask:\n    weight: 1.0\n    strategy: node\n    rate: 0.3\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: dmtest\n  prediction_task: regression\n  prediction_level: node\n  layer_key: log1p_norm\n"
        "  sample_key: ['sample']\n  mask_strategy: gene\n  mask_percentage: 1.0\n  mask_channel_fraction: 0.3\n"
        "  spatial_key: spatial\n  batch_size: 1\n  evaluate_test: False\n  spatial_neigbors_kwargs:\n    radius: 120\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    data = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(adata=data, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None)
    net = GlobalModel(data, cfg=cfg)
    pyg, _ = prepare_geome_dataset(data, cfg)
    dm = GraphAnnDataModule(datas=pyg, num_workers=0, batch_size=1, mask_percentage=1.0, mask_strategy="gene", learning_type="node",
                            **gene_set_mask_kwargs(cfg, data))  # fmt: skip
    net.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.module.mask_strategy == "gene"
    logged = [row for row in net.history_.history if "train_dual_mask_loss" in row]
    assert logged and all(np.isfinite(row["train_dual_mask_loss"]) for row in logged)
    assert any("dmn1" in p.name for p in (tmp_path / "out").iterdir())


def test_the_validator_wants_the_other_kind_and_a_single_head_model():
    from interscale.config import _validate_dual_mask, get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.prediction_task, cfg.dataset.prediction_level = "regression", "node"
    cfg.optim.dual_mask.weight = 1.0
    cfg.dataset.mask_strategy, cfg.optim.dual_mask.strategy = "node", "node"
    with pytest.raises(ValueError, match="other kind"):
        _validate_dual_mask(cfg)
    cfg.optim.dual_mask.strategy = "channel"
    _validate_dual_mask(cfg)
    cfg.dataset.mask_strategy = "gene"
    with pytest.raises(ValueError, match="other kind"):
        _validate_dual_mask(cfg)
    cfg.optim.dual_mask.strategy = "blocks"
    with pytest.raises(ValueError, match="'node' or 'channel'"):
        _validate_dual_mask(cfg)
