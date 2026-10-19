"""``pca_fit: train``: a GlobalModel's PCA tokens fitted on the training slides (roadmap.md, T0).

By default the basis is fitted on the first batch the module sees, which in a real run is the first
(masked) validation slide, because Lightning's sanity check runs before training. That basis misses
slide- and condition-level structure of the training slides (global_plan.md, first read: batch 0.02
and condition 0.30 against 0.985 for an embedding fitted on the training slides). ``train`` fits it
once, before training, on the masked inputs of every training slide. Pinned: the basis equals a
deterministic PCA of those inputs, a hidden gene adds nothing to it, an already-fitted module
(a restored checkpoint) is left alone, the default is unchanged, and the config and the checkpoint
name know the switch.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

E, G = 6, 10
HIDDEN = [2, 5]


def _cfg(fit="train"):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads = 64, 2
    params.type_gex_embedding = "PCA"
    params.pca_fit = fit
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    cfg.dataset.mask_genes = ["g2", "g5"]
    return cfg


def _model(cfg):
    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(12, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 12)
    adata.obs["split"] = pd.Categorical(["train"] * 10 + ["val"] * 2)
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(0)
    return GlobalModel(adata, cfg=cfg)


def _graphs(n_graphs=3, n=40, seed=0):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for k in range(n_graphs):
        # A slide-level shift: what a basis fitted on one slide alone would not know.
        x = torch.rand(n, G, generator=gen) * 4.0 + k * 3.0
        out.append(Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), pos=torch.rand(n, 2, generator=gen) * 50))
    return out


def _datamodule(graphs, cfg):
    from interscale.geome_dataloader import GraphAnnDataModule

    return GraphAnnDataModule(
        datas=[graphs, graphs[:1]],
        mask_percentage=1.0,
        mask_strategy="gene",
        learning_type="node",
        mask_gene_columns=HIDDEN,
    )


def test_the_basis_is_a_deterministic_pca_of_the_masked_training_inputs():
    from sklearn.decomposition import PCA

    model = _model(_cfg())
    graphs = _graphs()
    dm = _datamodule(graphs, model._cfg)
    model._fit_input_pca(dm)

    masked = torch.cat([g.x.clone() for g in graphs])
    masked[:, HIDDEN] = -1.0  # programme masking: the gene set is MASK_VALUE in every cell
    expected = PCA(n_components=E, random_state=0).fit(masked.numpy())
    module = model.module
    assert bool(module.pca_fitted_)
    np.testing.assert_allclose(module.pca_mean_.numpy(), expected.mean_, atol=1e-5)
    # The components agree up to sign.
    got = module.pca_components_.numpy()
    for k in range(E):
        assert min(np.abs(got[k] - expected.components_[k]).max(), np.abs(got[k] + expected.components_[k]).max()) < 1e-4


def test_a_hidden_gene_adds_nothing_to_the_basis():
    model = _model(_cfg())
    model._fit_input_pca(_datamodule(_graphs(), model._cfg))
    loadings = model.module.pca_components_.numpy()
    assert np.abs(loadings[:, HIDDEN]).max() < 1e-6, "a constant hidden column has no variance to load on"
    assert np.abs(np.delete(loadings, HIDDEN, axis=1)).max() > 0.1


def test_the_basis_knows_every_training_slide_not_just_the_first_batch():
    graphs = _graphs()
    train = _model(_cfg("train"))
    train._fit_input_pca(_datamodule(graphs, train._cfg))
    # The default fits lazily on whatever batch arrives first -- here one slide.
    first = _model(_cfg("first_batch"))
    first._fit_input_pca(_datamodule(graphs, first._cfg))
    assert not bool(first.module.pca_fitted_), "the default does not fit before training"
    first.module.create_gex_embedding(graphs[0].x.numpy(), type="PCA")
    assert not np.allclose(first.module.pca_mean_.numpy(), train.module.pca_mean_.numpy(), atol=0.5)
    # The training mean lies between the slides' own means (they are shifted by 3 per slide).
    assert train.module.pca_mean_.numpy()[0] > graphs[0].x.mean() + 1.0


def test_a_fitted_module_is_left_alone_and_other_tokens_are_untouched():
    model = _model(_cfg())
    graphs = _graphs()
    model.module.create_gex_embedding(graphs[1].x.numpy(), type="PCA")
    before = model.module.pca_components_.clone()
    assert model.module.fit_pca_on_training(graphs) == 0
    assert torch.equal(model.module.pca_components_, before)

    other = _cfg()
    other.model.global_component.parameters.type_gex_embedding = "Linear"
    other.model.global_component.parameters.pca_fit = "first_batch"
    linear = _model(other)
    linear._fit_input_pca(_datamodule(graphs, linear._cfg))  # not PCA: nothing to fit
    assert linear.module.fit_pca_on_training(graphs) == 0


def test_the_config_and_the_checkpoint_name_know_the_switch():
    from interscale.config import _validate
    from interscale.tl.utils import get_model_filename_prefix

    _validate(_cfg("train"))
    bad = _cfg("later")
    with pytest.raises(ValueError, match="pca_fit"):
        _validate(bad)
    wrong = _cfg("train")
    wrong.model.global_component.parameters.type_gex_embedding = "Linear"
    with pytest.raises(ValueError, match="fits the PCA tokens"):
        _validate(wrong)

    names = {}
    for fit in ("first_batch", "train"):
        cfg = _cfg(fit)
        cfg.dataset.name = "d"
        names[fit] = get_model_filename_prefix(cfg, False, True)
    assert "emb-pca-train" in names["train"] and "emb-pca" not in names["first_batch"]


def test_a_real_run_fits_the_basis_on_the_training_slides_before_the_first_step(tmp_path, monkeypatch):
    """Through ``train()``: the hook runs once, on the training graphs, with the masks of the datamodule."""
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.model.global_model import GlobalModel
    from interscale.module.base import GlobalModule
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    n = 30 * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, 30))
    adata.obs["split"] = pd.Categorical([{"s1": "train", "s2": "train", "s3": "val", "s4": "test"}[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "pca.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "pca.yaml"
    path.write_text(
        "model:\n  n_embed: 4\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 64\n      num_layers: 1\n      n_heads: 2\n      type_gex_embedding: PCA\n      pca_fit: train\n"
        "  decoder:\n    type: linear\n  save: null\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: pcatest\n  prediction_task: regression\n  prediction_level: node\n"
        "  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n  mask_percentage: 0.3\n"
        "  batch_size: 2\n  evaluate_test: False\n  spatial_neigbors_kwargs:\n    radius: 25\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=adata, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    model = GlobalModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )

    calls = []
    original = GlobalModule.fit_pca_on_training
    monkeypatch.setattr(
        GlobalModule,
        "fit_pca_on_training",
        lambda self, graphs: calls.append((len(graphs), bool(self.pca_fitted_))) or original(self, graphs),
    )
    model.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert calls == [(2, False)], "once, on the two training slides, before anything was fitted"
    assert bool(model.module.pca_fitted_)
