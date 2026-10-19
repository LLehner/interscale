"""A precomputed embedding under programme masking (roadmap.md, T0; global_plan.md, Stage A).

Gene masking cannot hide a gene from an embedding computed from it, so ``Precomputed`` was refused
under ``mask_strategy gene``. A gene set hidden in every cell (``dataset.mask_genes``) changes that:
an embedding computed from the *other* genes holds none of the hidden values. Pinned here:

* the config and the module accept ``Precomputed`` with a hidden gene set, and still refuse single
  hidden entries;
* through a GlobalModel built from a config, no cell's embedding is replaced by the fill (every cell
  keeps its visible genes), while the hidden genes' values never reach a prediction;
* the guard: ``record_embedding_genes`` / ``check_embedding_genes`` refuse an embedding that saw a
  hidden gene, and one whose genes are unknown, when the graphs are built.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

E, G, N = 8, 6, 12
HIDDEN = [0, 1]


def _cfg(hide=True):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads = 64, 2
    params.type_gex_embedding = "Precomputed"
    cfg.model.global_component.latent_obsm_key = "X_emb"
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    if hide:
        cfg.dataset.mask_genes = ["g0", "g1"]
    return cfg


def _model(cfg):
    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(N, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * N)
    adata.obs["split"] = pd.Categorical(["train"] * (N - 2) + ["val"] * 2)
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(0)
    return GlobalModel(adata, cfg=cfg)


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    d = Data(x=torch.rand(N, G, generator=gen) + 1.0, edge_index=torch.zeros((2, 0), dtype=torch.long))
    d.embeddings = torch.randn(N, E, generator=gen)
    d.mask = torch.ones(N, dtype=torch.bool)  # programme masking selects every cell
    d.gene_mask = torch.zeros(N, G, dtype=torch.bool)
    d.gene_mask[:, HIDDEN] = True
    batch = Batch.from_data_list([d])
    batch.obs_names = torch.arange(N)
    return batch


def _predict(module, batch):
    with torch.no_grad():
        return module._common_step(batch, "regression", "node").y_pred


def test_the_config_accepts_a_hidden_gene_set_and_refuses_single_entries():
    from interscale.config import _validate

    _validate(_cfg(hide=True))
    with pytest.raises(ValueError, match="mask_strategy gene"):
        _validate(_cfg(hide=False))


def test_the_module_is_built_with_a_gene_set_and_refuses_single_entries():
    assert _model(_cfg(hide=True)).module.type_gex_embedding == "Precomputed"
    with pytest.raises(ValueError, match="single genes"):
        _model(_cfg(hide=False))


def test_no_cell_is_replaced_by_the_fill_and_hidden_values_reach_nothing():
    module = _model(_cfg()).module.eval()
    batch = _batch()
    before = _predict(module, batch)

    # The embedding of ANY cell reaches the prediction: every cell is selected by the mask, and
    # none of them may be swapped for the fill (that would leave the model no information at all).
    for cell in (0, 5, 11):
        shifted = batch.clone()
        shifted.embeddings = shifted.embeddings.clone()
        shifted.embeddings[cell] += torch.randn(E, generator=torch.Generator().manual_seed(cell)) * 3.0
        assert not torch.allclose(_predict(module, shifted), before), f"cell {cell}'s embedding was replaced"

    # The values of the hidden genes are the targets, not inputs: changing them moves no prediction.
    other = batch.clone()
    other.x = other.x.clone()
    other.x[:, HIDDEN] = torch.rand(N, len(HIDDEN)) * 50.0
    assert torch.allclose(_predict(module, other), before)


def test_the_node_strategy_still_replaces_hidden_cells():
    """The programme path must not have switched off the whole-cell guard of ``precomputed_mask_fill``."""
    cfg = _cfg(hide=False)
    cfg.dataset.mask_strategy, cfg.dataset.mask_percentage = "node", 0.3
    module = _model(cfg).module.eval()
    batch = _batch()
    batch.mask = torch.zeros(N, dtype=torch.bool)
    batch.mask[:4] = True
    before = _predict(module, batch)
    shifted = batch.clone()
    shifted.embeddings = shifted.embeddings.clone()
    shifted.embeddings[:4] += torch.randn(4, E, generator=torch.Generator().manual_seed(1)) * 3.0
    assert torch.allclose(_predict(module, shifted), before)


# --------------------------------------------------------------------------- the guard


def _adata(n=16, genes=6):
    rng = np.random.default_rng(0)
    adata = AnnData(X=np.abs(rng.normal(size=(n, genes))).astype(np.float32) + 0.1)
    adata.var_names = [f"g{k}" for k in range(genes)]
    adata.obsm["spatial"] = rng.uniform(0, 50, size=(n, 2))
    adata.obsm["X_emb"] = rng.normal(size=(n, E)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(["s1"] * (n // 2) + ["s2"] * (n // 2))
    adata.obs["split"] = pd.Categorical(["train"] * (n // 2) + ["val"] * (n // 2))
    return adata


def test_the_record_round_trips_and_names_only_known_genes(tmp_path):
    import anndata as ad

    from interscale.tl import embedding_genes, record_embedding_genes

    adata = _adata()
    assert embedding_genes(adata, "X_emb") is None
    record_embedding_genes(adata, "X_emb", ["g2", "g3", "g4"])
    assert embedding_genes(adata, "X_emb") == ["g2", "g3", "g4"]
    adata.write_h5ad(tmp_path / "a.h5ad")  # the record survives the file format
    assert embedding_genes(ad.read_h5ad(tmp_path / "a.h5ad"), "X_emb") == ["g2", "g3", "g4"]
    with pytest.raises(ValueError, match="not in adata.var_names"):
        record_embedding_genes(adata, "X_emb", ["g2", "nope"])


@pytest.mark.parametrize(
    ("seen", "match"),
    [(None, "does not say"), (["g0", "g3"], "computed from 1 of the genes"), (["g2", "g3", "g4", "g5"], None)],
)
def test_the_guard_refuses_an_embedding_that_saw_a_hidden_gene_or_whose_genes_are_unknown(seen, match):
    from interscale.tl import check_embedding_genes, record_embedding_genes

    adata, cfg = _adata(), _cfg()
    if seen is not None:
        record_embedding_genes(adata, "X_emb", seen)
    if match is None:
        check_embedding_genes(adata, cfg)
    else:
        with pytest.raises(ValueError, match=match):
            check_embedding_genes(adata, cfg)


def test_the_guard_is_a_no_op_without_a_hidden_gene_set_or_another_token_input():
    from interscale.tl import check_embedding_genes

    check_embedding_genes(_adata(), _cfg(hide=False))  # no gene set: nothing to verify
    cfg = _cfg()
    cfg.model.global_component.parameters.type_gex_embedding = "Linear"
    check_embedding_genes(_adata(), cfg)


def test_the_graphs_are_not_built_from_an_embedding_that_saw_a_hidden_gene():
    from interscale.tl import prepare_geome_dataset, record_embedding_genes

    def build(seen):
        adata, cfg = _adata(), _cfg()
        cfg.dataset.sample_key = ["sample"]
        cfg.dataset.prediction_task = "regression"
        cfg.dataset.spatial_key = "spatial"
        cfg.dataset.spatial_neigbors_kwargs.library_key = "sample"
        cfg.dataset.spatial_neigbors_kwargs.radius = 40
        if seen is not None:
            record_embedding_genes(adata, "X_emb", seen)
        return prepare_geome_dataset(adata, cfg)[0]

    datas = build(["g2", "g3"])
    assert datas[0][0].embeddings.shape[1] == E
    with pytest.raises(ValueError, match="computed from"):
        build(["g1", "g2"])
    with pytest.raises(ValueError, match="does not say"):
        build(None)


def test_a_real_run_trains_on_a_precomputed_embedding_under_programme_masking(tmp_path):
    """Through ``train()``: the guard passes for an embedding of the other genes, the loss is on the hidden set only."""
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset, record_embedding_genes

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    n = 30 * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, 30))
    adata.obs["split"] = pd.Categorical([{"s1": "train", "s2": "train", "s3": "val", "s4": "test"}[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    visible = [k for k in range(G) if k not in (2, 5)]  # the genes the config hides: g2 and g5
    adata.obsm["X_emb"] = (adata.layers["log1p_norm"][:, visible] @ rng.normal(size=(len(visible), E))).astype(np.float32)
    record_embedding_genes(adata, "X_emb", [f"g{k}" for k in visible])
    h5ad = tmp_path / "pre.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "pre.yaml"
    path.write_text(
        f"model:\n  n_embed: {E}\n  global_component:\n    name: self-attn-transformer\n    latent_obsm_key: X_emb\n"
        "    parameters:\n      max_seq_len: 64\n      num_layers: 1\n      n_heads: 2\n      type_gex_embedding: Precomputed\n"
        "  decoder:\n    type: linear\n  save: null\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: pretest\n  prediction_task: regression\n  prediction_level: node\n"
        "  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: gene\n  mask_percentage: 1.0\n"
        "  mask_genes: ['g2', 'g5']\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 25\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=adata, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    model = GlobalModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=1.0, mask_strategy="gene", learning_type="node",
        **gene_set_mask_kwargs(cfg, adata),
    )
    model.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert model.is_trained_
    losses = [row["val_loss"] for row in model.history_.history if "val_loss" in row]
    assert losses and all(np.isfinite(losses))
