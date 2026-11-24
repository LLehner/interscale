"""``optim.aux_loss_weights.field_corr`` and ``evaluation.field_correlation`` (2.3, ``.claude/comparison.md``).

The term by hand (a sender field that is a linear function of a hidden gene gives r^2 = 1, the term near -1, the gradient reaches the fields); a
dense context-query model trains with it, hands out fields that carry the attention's gradient, and tags its checkpoint; the readout pools the
correlations over graphs and leaves unread pairs NaN, and its kernel bar finds a planted neighbour dependence; the validator refuses a map-less
backend and gene sets that are not channels.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

G = 6


def _term():
    from interscale.config import get_cfg_defaults
    from interscale.train.aux_losses import FieldCorrelation

    return FieldCorrelation(get_cfg_defaults())


def test_the_term_by_hand():
    from interscale.module.base import StepOutput, ViewOutput

    rng = np.random.default_rng(0)
    y = torch.tensor(rng.normal(size=(40, 4)))
    entry = torch.zeros(40, 4, dtype=torch.bool)
    entry[:, [2, 3]] = True  # the hidden genes: targets
    fields = torch.stack([2.0 * y[:, 2] + 1.0, torch.tensor(rng.normal(size=40)), y[:, 0], y[:, 1]], dim=1).requires_grad_(True)
    out = StepOutput(y_pred=y, y_true=y, entry_mask=entry, fields=fields, views=[ViewOutput(padded_node_idx=None)])
    terms = _term()(out, None)
    assert terms["_field_corr_max"].item() > 0.5  # target 2 has a perfect sender, target 3 a weak one
    assert -1.0 <= terms["field_corr"].item() < -0.5
    terms["field_corr"].backward()
    assert fields.grad[:, :2].abs().sum() > 0 and fields.grad[:, 2:].abs().sum() == 0  # only the senders' fields (genes 0, 1) are read


def _model(tmp_path, weight):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    per = 60
    adata = AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 400, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "fc.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "fc.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 256\n      num_layers: 2\n      n_heads: 2\n      long_range_attention: False\n      type_gex_embedding: Linear\n"
        "      context_only: query\n      context_query:\n        inputs: [own]\n"
        f"  decoder:\n    type: linear\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  center_targets: True\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        f"  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n  aux_loss_weights:\n    field_corr: {weight}\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: fctest\n  prediction_task: regression\n  prediction_level: node\n  layer_key: log1p_norm\n"
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
    return net, dm, pyg


def test_a_dense_context_query_model_trains_with_the_term_and_the_fields_carry_the_attention(tmp_path):
    from interscale.evaluation.field_correlation import field_pairs, pair_table
    from interscale.evaluation.gene_ranking import hide_genes

    net, dm, pyg = _model(tmp_path, 1.0)
    graph = hide_genes([pyg[2][0]], [1, 4])[0]
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    module = net.module.train()
    out = module._common_step(graph, "regression", "node")
    assert out.fields is not None and out.fields.shape == out.y_pred.shape
    from interscale.train.aux_losses import FieldCorrelation

    loss = FieldCorrelation(net._cfg)(out, graph)["field_corr"]
    loss.backward()
    last = module.transformer_encoder.layers[-1]
    grads = [p.grad for p in last.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)  # the term trains the attention
    net.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    logged = [row for row in net.history_.history if "train_field_corr" in row]
    assert logged and all(np.isfinite(row["train_field_corr"]) for row in logged)
    assert any("fc1" in p.name for p in (tmp_path / "out").iterdir())
    tests = [g for g in pyg[2]]
    r = field_pairs(net.module, tests, G, [np.array([0, 1]), np.array([2, 3, 4, 5])], kernel_um=80.0)
    assert r["model"].shape == (G, G) and np.isnan(r["model"][0, 1]) and np.isfinite(r["model"][2, 0])  # 0 and 1 share a fold: unread
    assert np.all(np.abs(r["model"][np.isfinite(r["model"])]) <= 1 + 1e-9)
    assert len(pair_table(r["model"], [f"g{k}" for k in range(G)])) == 2 * 4 + 4 * 2


def test_the_kernel_bar_finds_a_planted_neighbour_dependence():
    from interscale.evaluation.field_correlation import field_pairs

    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 500, size=(300, 2))
    x = rng.random((300, G))
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    k = np.exp(-d2 / (2 * 40.0**2))
    np.fill_diagonal(k, 0)
    x[:, 3] = (k / k.sum(1, keepdims=True)) @ x[:, 0] * 5 + 0.05 * rng.random(300)  # gene 3 = neighbours' gene 0
    graph = Data(x=torch.tensor(x, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32))
    r = field_pairs(None, [graph], G, [np.array([3])], kernel_um=40.0, model=False)
    assert r["kernel"][0, 3] > 0.8 and np.nanmax(np.abs(np.delete(r["kernel"][:, 3], [0, 3]))) < 0.4


def test_the_validator_wants_a_dense_map_and_channels():
    from interscale.config import _validate_field_corr, get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.prediction_task, cfg.dataset.prediction_level = "regression", "node"
    cfg.optim.aux_loss_weights.field_corr = 1.0
    from yacs.config import CfgNode as CN

    cfg.model.global_component.name = "self-attn-transformer"
    cfg.model.global_component.parameters = CN({"attention": CN({"kind": "readout"})})
    with pytest.raises(ValueError, match="attention.kind dense"):
        _validate_field_corr(cfg)
    cfg.model.global_component.parameters.attention.kind = "dense"
    cfg.dataset.mask_strategy = "node"
    with pytest.raises(ValueError, match="channel masking"):
        _validate_field_corr(cfg)
    cfg.dataset.mask_strategy, cfg.dataset.mask_channel_fraction = "gene", 0.2
    _validate_field_corr(cfg)
