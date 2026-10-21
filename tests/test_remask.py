"""Re-masking before the transformer (``model.global_component.parameters.remask``; roadmap.md, T2).

Cells are masked once, in the input; the GCN then hands a hidden cell a token made of its visible
neighbours. ``remask: hidden`` replaces that token by one learned token before the transformer.

Pinned here:

* **the switch** -- off by default, then no state is added and the tokens pass through unchanged; on, the
  model differs by the one token and every other weight is initialised as before;
* **what the transformer reads** -- never a hidden cell's own local token (its gradient is exactly zero),
  visible cells' tokens unchanged; with the long-range mask, nothing within the local reach of a hidden
  cell reaches its global embedding, while cells beyond it do; under discs read as gene masks too;
* **through the real model and trainer** -- the dual-decoder CombinedModel trains, the token learns, the
  checkpoint name carries ``rmk-hidden``;
* **refusals** -- unknown kinds, masks that hide no cell whole, pooling, a GlobalModel.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.module.base import gather_tokens

E, G = 8, 6
SPACING = 100.0
SIDE = 7


def _lattice(side=SIDE, seed=0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    n = side * side
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    return data


def _batch(graphs):
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _config(remask="hidden", *, lrm=False, strategy="node", model="CombinedModel"):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    if model == "CombinedModel":
        cfg.model.local_component.name = "GCN"
        cfg = get_local_component_cfg(cfg, "GCN")
        cfg.model.decoder.dual_decoder = True
    cfg.model.n_embed = E
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 64, 2, 1, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.node = ["sinusoidal"]
    params.long_range_attention = lrm
    params.remask = remask
    cfg.dataset.mask_strategy = strategy
    cfg.dataset.mask_percentage = 0.3
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    if strategy == "gene":
        cfg.dataset.mask_block_radius_um = 150.0
        cfg.dataset.mask_block_fraction = 0.3
    return cfg


def _build(cfg, model="CombinedModel", seed=0):
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    n = SIDE * SIDE
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    cls = {"GlobalModel": GlobalModel, "CombinedModel": CombinedModel}[model]
    cls._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return cls(adata, cfg=cfg)


def _hide(batch, hidden, strategy="node"):
    """Hide the cells ``hidden`` whole, as the dataloader (``node``) or the disc evaluation (``gene``) writes it."""
    batch.mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    batch.mask[hidden] = True
    if strategy == "gene":
        batch.gene_mask = batch.mask[:, None].expand(-1, G).clone()
    return batch


def _token_gradients(module, batch, query: int) -> torch.Tensor:
    """``[N]``: how much each cell's token entering the global module moves ``query``'s global embedding."""
    module.eval()
    batch_masked, _, _ = module._common_step_masking(batch)
    with torch.no_grad():
        local = module.local_module.forward(batch_masked.x, batch_masked.edge_index)
    tokens = local.clone().requires_grad_(True)
    out, padding, nodes, _, _ = module.global_module.step(batch_masked, tokens)
    rows = gather_tokens(out, padding)
    order = torch.as_tensor(nodes[0], dtype=torch.long)
    row = rows[(order == query).nonzero().item()]
    direction = torch.randn(row.shape[-1], generator=torch.Generator().manual_seed(1))
    (row * direction).sum().backward()  # a random projection: a LayerNorm zeroes a plain sum
    return tokens.grad.abs().sum(1)


CENTRE = (SIDE // 2) * SIDE + SIDE // 2


def _hops_from_centre() -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(SIDE), torch.arange(SIDE), indexing="ij")
    return ((ii - SIDE // 2).abs() + (jj - SIDE // 2).abs()).flatten()


# --------------------------------------------------------------------------- the switch


def test_off_by_default_adds_no_state_and_passes_the_tokens_through():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_global_component_cfg(get_cfg_defaults(), "self-attn-transformer")
    assert cfg.model.global_component.parameters.remask == "none"
    module = _build(_config("none")).module
    assert not any("remask" in key for key in module.state_dict())
    batch = _hide(_batch([_lattice()]), [CENTRE])
    tokens = torch.randn(batch.num_nodes, E)
    assert module.global_module.remask_tokens(batch, tokens) is tokens


def test_on_the_model_differs_by_one_token_and_every_other_weight_is_initialised_as_before():
    plain = _build(_config("none")).module.state_dict()
    remasked = _build(_config("hidden")).module.state_dict()
    assert set(remasked) - set(plain) == {"global_module.remask_token"}
    assert set(plain) <= set(remasked)
    assert all(torch.equal(plain[k], remasked[k]) for k in plain)


def test_visible_tokens_stay_and_every_hidden_cell_gets_the_same_token():
    module = _build(_config("hidden")).module
    batch = _hide(_batch([_lattice()]), [0, 5, CENTRE])
    tokens = torch.randn(batch.num_nodes, E)
    out = module.global_module.remask_tokens(batch, tokens)
    hidden = batch.mask
    assert torch.equal(out[~hidden], tokens[~hidden])
    assert torch.equal(out[hidden], module.global_module.remask_token.detach().expand(3, E))


# --------------------------------------------------------------------------- what the transformer reads


@pytest.mark.parametrize("strategy", ["node", "gene"])
def test_the_transformer_never_reads_a_hidden_cells_own_local_token(strategy):
    batch = _hide(_batch([_lattice()]), [CENTRE, 3], strategy)
    remasked = _token_gradients(_build(_config("hidden", strategy=strategy)).module, batch, CENTRE)
    plain = _token_gradients(_build(_config("none", strategy=strategy)).module, batch, CENTRE)
    assert remasked[CENTRE] == 0 and remasked[3] == 0  # no hidden cell's own token, its or another's
    assert plain[CENTRE] > 0  # without re-masking its token is its residual stream
    assert (remasked > 0).sum() > 0  # the visible cells still reach it


def test_with_the_long_range_mask_a_hidden_cell_reads_only_beyond_the_local_reach():
    """One transformer layer, a 2-hop GCN: a hidden cell's global embedding depends on no token within 2 hops."""
    batch = _hide(_batch([_lattice()]), [CENTRE])
    hops = _hops_from_centre()
    module = _build(_config("hidden", lrm=True)).module
    assert module.local_module.receptive_field_hops == 2
    grads = _token_gradients(module, batch, CENTRE)
    assert torch.all(grads[hops <= 2] == 0)
    assert torch.all(grads[hops > 2] > 0)
    # Without re-masking the own token is the way in: the cell reads its neighbourhood through the GCN.
    plain = _token_gradients(_build(_config("none", lrm=True)).module, batch, CENTRE)
    assert plain[CENTRE] > 0 and torch.all(plain[(hops > 0) & (hops <= 2)] == 0)


def test_no_hidden_value_reaches_a_prediction():
    batch = _hide(_batch([_lattice()]), [CENTRE, 10, 11])
    module = _build(_config("hidden")).module.eval()
    with torch.no_grad():
        before = module._common_step(batch, "regression", "node").y_pred
        changed = batch.clone()
        changed.x = batch.x.clone()
        changed.x[[CENTRE, 10, 11]] += 50.0
        after = module._common_step(changed, "regression", "node").y_pred
    # The truth of the hidden rows changed; no prediction did, of either head.
    assert torch.equal(after, before)


# --------------------------------------------------------------------------- through the trainer


def test_the_token_learns_and_the_dual_decoder_model_trains_end_to_end(tmp_path):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.model.combined_model import CombinedModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    n = 40 * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, 40))
    split = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "remask.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "remask.yaml"
    path.write_text(
        f"model:\n  n_embed: {E}\n  local_component:\n    name: GCN\n  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 64\n      num_layers: 1\n"
        "      n_heads: 2\n      long_range_attention: True\n      remask: hidden\n"
        "      pe:\n        node: [sinusoidal]\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 3\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: remasktest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 25\n    library_key: sample\n"
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
    start = net.module.global_module.remask_token.detach().clone()
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    net.train(max_epochs=3, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    losses = [row["val_loss"] for row in net.history_.history if "val_loss" in row]
    assert losses and all(np.isfinite(losses))
    assert not torch.equal(net.module.global_module.remask_token.detach().cpu(), start)
    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert any("rmk-hidden" in name for name in saved), saved


# --------------------------------------------------------------------------- refusals


def _validate(cfg):
    from interscale.config import _validate_remask

    _validate_remask(cfg)


def test_the_config_takes_remask_with_cell_masking_and_with_discs():
    _validate(_config("hidden"))
    _validate(_config("hidden", strategy="gene"))  # discs under gene masking hide whole rows
    _validate(_config("none"))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: setattr(c.model.global_component.parameters, "remask", "visible"), "must be one of"),
        (lambda c: setattr(c.dataset, "mask_genes", ["g0"]), "hides no cell whole"),
        (lambda c: setattr(c.dataset, "mask_block_fraction", 0.0), "only spatial blocks"),
        (lambda c: setattr(c.model.global_component.parameters.pooling, "kind", "grid"), "pooled"),
    ],
)
def test_the_config_refuses_what_remask_cannot_do(change, message):
    cfg = _config("hidden", strategy="gene")
    change(cfg)
    with pytest.raises(ValueError, match=message):
        _validate(cfg)


def test_the_config_refuses_remask_when_nothing_is_masked():
    cfg = _config("hidden")
    cfg.dataset.mask_percentage = 0.0
    with pytest.raises(ValueError, match="nothing is masked"):
        _validate(cfg)


def test_a_global_model_refuses_remask():
    with pytest.raises(ValueError, match="CombinedModel"):
        _build(_config("hidden", model="GlobalModel"), model="GlobalModel")


def test_the_checkpoint_name_carries_remask():
    from interscale.tl.utils import get_model_filename_prefix

    on = get_model_filename_prefix(_config("hidden"), True, True)
    off = get_model_filename_prefix(_config("none"), True, True)
    assert "rmk-hidden" in on and "rmk" not in off
