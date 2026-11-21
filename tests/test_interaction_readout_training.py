"""The interaction read-out trains end to end (``.claude/ccc_model_plan.md``): one arm per training mode, every new option on.

* ``programme``: synth_spot's task (a gene set hidden in every cell), query from the position and the visible genes, additive
  read-out, the raw own-cell skip, the content-dependent rate, both penalties with a ramp.
* ``all_cells_keys``: mode (c), every cell a target and 20 % of cells + discs removed as keys only, the label in the query and
  the target (label_mean), the same-type block, key dropout.
* ``all_cells_none``: mode (b), nothing hidden, keys from the label alone.

Each arm trains two epochs on CPU, its weights move, the loss is finite, and the checkpoint name carries its options.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData

G = 6

ARMS = {
    "programme": (
        "  mask_strategy: gene\n  mask_percentage: 1.0\n  mask_genes: [g0, g1]\n",
        "    own_cell_skip: True\n    own_cell_skip_input: raw\n",
        "      context_query:\n        inputs: [position, own]\n"
        "      pe:\n        node: ['sinusoidal']\n        bias: ['distance']\n"
        "        distance:\n          kind: scales\n          scale_min_um: 50.0\n          scale_max_um: 500.0\n          shape: exponential\n"
        "      attention:\n        kind: readout\n        readout:\n          null_key: True\n          content_rate: True\n",
        "  aux_loss_weights:\n    attn_entropy: 0.05\n    value_l1: 0.01\n  attn_entropy:\n    ramp_epochs: 2\n",
        ("ro-additive", "q-po", "attn-readout-n-r", "aux-ent0.05-vl10.01"),
    ),
    "all_cells_keys": (
        "  mask_strategy: node\n  mask_percentage: 0.2\n  mask_block_radius_um: 80.0\n  mask_block_fraction: 0.1\n"
        "  score_all_cells: True\n  extra_obs_keys: ['cell_type']\n",
        "    section_offset: label_mean\n    label_offset_key: cell_type\n",
        "      context_query:\n        inputs: [type]\n        type_key: cell_type\n"
        "      pe:\n        bias: ['distance']\n"
        "        distance:\n          kind: scales\n          scale_min_um: 50.0\n          scale_max_um: 500.0\n"
        "      attention:\n        kind: readout\n        visible_keys_only: True\n"
        "        readout:\n          same_type_keys: block\n          key_dropout: 0.3\n",
        "",
        ("ro-additive", "q-t", "attn-readout-n-st-kd0.3-vk", "all"),
    ),
    "all_cells_none": (
        "  mask_strategy: node\n  mask_percentage: 0.0\n  score_all_cells: True\n  extra_obs_keys: ['cell_type']\n",
        "    section_offset: label_mean\n    label_offset_key: cell_type\n",
        "      context_query:\n        inputs: [type]\n        type_key: cell_type\n        key_inputs: type\n"
        "      pe:\n        bias: ['distance']\n"
        "        distance:\n          kind: scales\n          scale_min_um: 50.0\n          scale_max_um: 500.0\n"
        "      attention:\n        kind: readout\n",
        "",
        ("ro-additive", "q-t", "k-type", "all"),
    ),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_each_mode_trains(tmp_path, arm):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, block_mask_kwargs, gene_set_mask_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset
    from interscale.tl.utils import get_model_filename_prefix

    masking, decoder, params, optim, tags = ARMS[arm]
    rng = np.random.default_rng(0)
    per = 60
    adata = AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["a", "b", "c"], size=per * 3))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "arm.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "arm.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 256\n      num_layers: 2\n      n_heads: 2\n      long_range_attention: False\n"
        "      type_gex_embedding: Linear\n      context_only: query\n      context_readout: additive\n"
        f"{params}"
        f"  decoder:\n    type: linear\n{decoder}  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  center_targets: True\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        f"  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n{optim}"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: irtest\n  prediction_task: regression\n"
        f"  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n{masking}"
        "  spatial_key: spatial\n  batch_size: 1\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    name = get_model_filename_prefix(cfg, False, True)
    for tag in tags:
        assert tag in name, (tag, name)
    data = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=data, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    net = GlobalModel(data, cfg=cfg)
    pyg, _ = prepare_geome_dataset(data, cfg)
    dm = GraphAnnDataModule(
        datas=pyg,
        num_workers=0,
        batch_size=1,
        mask_percentage=float(cfg.dataset.mask_percentage),
        mask_strategy=str(cfg.dataset.mask_strategy),
        learning_type="node",
        **gene_set_mask_kwargs(cfg, data),
        **block_mask_kwargs(cfg),
    )
    before = {n: p.detach().clone() for n, p in net.module.named_parameters()}
    net.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    moved = [n for n, p in net.module.named_parameters() if not torch.equal(p.detach().cpu(), before[n].cpu())]
    assert any("in_proj_weight" in n for n in moved)
    if arm == "programme":
        assert any("null_logit" in n for n in moved) and any("rate_key" in n for n in moved)
        assert any("query_own" in n for n in moved)
    else:
        assert any("query_type" in n for n in moved)
    # The step's output: every cell scored under score_all_cells.
    batch = next(iter(dm.train_dataloader()))
    net.module.eval()
    with torch.no_grad():
        out = net.module._common_step(batch, "regression", "node")
    assert torch.isfinite(out.y_pred).all()
    if "all" in tags:
        assert out.entry_mask.all()
