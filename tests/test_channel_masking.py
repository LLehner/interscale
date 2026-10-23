"""Channel masking: a random share of the genes hidden in every cell, drawn again each epoch (`dataset.mask_channel_fraction`; roadmap.md, T3).

The label-free form of programme masking. A fixed gene set needs to be known; a random one does not. Pinned here:

* **the draw** -- exactly ``round(fraction * G)`` genes, never none and never all, every gene equally likely, reproducible
  from a generator, and the same set in every selected cell (so no neighbour shows a hidden gene);
* **the cells** -- selected as for a gene set: all of them, a share, or the discs;
* **the datamodule** -- the training graphs get a fresh set at every redraw, the validation graphs keep theirs, 30% and
  50% are the same code with another number, and it refuses node masking and a fixed gene set beside it;
* **the config and the name** -- the fraction is a share in (0, 1), needs gene masking, excludes ``mask_genes``, and is
  part of the checkpoint name; re-masking, the regime loss, visible-keys-only attention and a precomputed embedding
  refuse it, since no cell is hidden whole and every gene is hidden at some step;
* **no leakage** -- through the real model a hidden gene reaches no prediction and a visible one does, and a model
  trained on channels is evaluated on a fixed programme set like any other;
* **through the trainer** -- a model trains on it, the training masks change from epoch to epoch.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation.masked_probing import draw_masks
from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
from interscale.tl.masking import hides_gene_set, sample_channel_columns, sample_channel_mask

G = 10


def _lattice(side, seed=0, genes=G):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * 100.0
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= 101.0).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, genes, generator=gen) * 3.0, edge_index=edges, pos=pos)
    data.obs_names = torch.arange(n, dtype=torch.float32) + 1
    return data


# --------------------------------------------------------------------------- the draw


@pytest.mark.parametrize(("genes", "fraction", "count"), [(20, 0.3, 6), (20, 0.5, 10), (10, 0.3, 3), (7, 0.5, 4)])
def test_a_draw_hides_exactly_the_rounded_share_of_the_genes(genes, fraction, count):
    columns = sample_channel_columns(genes, fraction, torch.Generator().manual_seed(1))
    assert len(columns) == count == len(set(columns))
    assert columns == sorted(columns) and all(0 <= c < genes for c in columns)


@pytest.mark.parametrize(("genes", "fraction"), [(10, 0.001), (10, 0.999), (2, 0.01), (2, 0.99), (3, 0.9)])
def test_a_draw_is_never_empty_and_never_everything(genes, fraction):
    for seed in range(20):
        columns = sample_channel_columns(genes, fraction, torch.Generator().manual_seed(seed))
        assert 1 <= len(columns) <= genes - 1


def test_every_gene_is_equally_likely_to_be_hidden():
    generator = torch.Generator().manual_seed(0)
    counts = torch.zeros(G)
    draws = 3000
    for _ in range(draws):
        counts[sample_channel_columns(G, 0.3, generator)] += 1
    assert (counts / draws - 0.3).abs().max() < 0.03


def test_a_draw_is_reproducible_from_a_generator_and_differs_without_one():
    a = sample_channel_columns(40, 0.3, torch.Generator().manual_seed(5))
    assert a == sample_channel_columns(40, 0.3, torch.Generator().manual_seed(5))
    assert a != sample_channel_columns(40, 0.3, torch.Generator().manual_seed(6))
    assert len({tuple(sample_channel_columns(40, 0.3)) for _ in range(5)}) > 1


def test_a_draw_refuses_what_cannot_be_split_or_shared_out():
    with pytest.raises(ValueError, match="cannot be split"):
        sample_channel_columns(1, 0.3)
    for fraction in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            sample_channel_columns(10, fraction)


# --------------------------------------------------------------------------- the cells


def test_the_same_genes_are_hidden_in_every_selected_cell():
    every = sample_channel_mask(40, G, 0.3, 1.0, generator=torch.Generator().manual_seed(2))
    assert every.shape == (40, G)
    assert (every == every[0]).all()  # one set for the whole graph
    assert int(every[0].sum()) == 3
    other = sample_channel_mask(40, G, 0.3, 1.0, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(every, other)  # and another set on the next draw


def test_the_cells_are_selected_as_for_a_gene_set():
    some = sample_channel_mask(400, G, 0.5, 0.25, generator=torch.Generator().manual_seed(0))
    rows = some.any(dim=1)
    assert 0.18 < rows.float().mean() < 0.32
    assert (some[rows] == some[rows][0]).all() and not some[
        ~rows
    ].any()  # selected cells hide the set, the rest nothing
    blocks = torch.zeros(30, dtype=torch.bool)
    blocks[:5] = True
    only_blocks = sample_channel_mask(30, G, 0.3, 0.0, blocks=blocks, generator=torch.Generator().manual_seed(0))
    assert torch.equal(only_blocks.any(dim=1), blocks)
    assert sample_channel_mask(30, G, 0.3, 0.0, generator=torch.Generator().manual_seed(0)).any(), (
        "a graph keeps a target"
    )


# --------------------------------------------------------------------------- the datamodule


def _module(fraction=0.3, **kwargs):
    datas = [[_lattice(6, s) for s in range(3)], [_lattice(5, 9), _lattice(5, 10)]]
    options = {"num_workers": 0, "mask_strategy": "gene", "mask_percentage": 1.0, "learning_type": "node"}
    dm = GraphAnnDataModule(datas=datas, mask_channel_fraction=fraction, **(options | kwargs))
    dm.setup(stage="fit")
    return dm


@pytest.mark.parametrize(("fraction", "count"), [(0.3, 3), (0.5, 5)])
def test_the_datamodule_hides_a_shared_random_set_of_the_asked_share(fraction, count):
    dm = _module(fraction)
    for graphs in (dm.train_data, dm.val_data):
        for data in graphs:
            assert (data.gene_mask == data.gene_mask[0]).all() and int(data.gene_mask[0].sum()) == count
            assert torch.equal(data.mask, data.gene_mask.any(1)) and data.mask.all()


def test_the_training_graphs_get_a_new_set_at_every_redraw_and_the_validation_graphs_keep_theirs():
    dm = _module(0.3)
    train = [d.gene_mask[0].clone() for d in dm.train_data]
    val = [d.gene_mask[0].clone() for d in dm.val_data]
    seen = {tuple(t.nonzero().flatten().tolist()) for t in train}
    for _ in range(8):
        dm.resample_train_mask()
        seen |= {tuple(d.gene_mask[0].nonzero().flatten().tolist()) for d in dm.train_data}
    assert len(seen) > 3, "the sets do not change from draw to draw"
    assert all(torch.equal(v, d.gene_mask[0]) for v, d in zip(val, dm.val_data, strict=True))  # drawn once at setup
    assert any(not torch.equal(t, d.gene_mask[0]) for t, d in zip(train, dm.train_data, strict=True))


def test_every_gene_is_hidden_at_some_redraw_of_a_training_graph():
    dm = _module(0.3, mask_percentage=1.0)
    hidden = torch.zeros(G, dtype=torch.bool)
    for _ in range(40):
        dm.resample_train_mask()
        hidden |= dm.train_data[0].gene_mask[0]
    assert hidden.all()


def test_the_datamodule_selects_cells_by_the_rate_and_by_discs():
    dm = _module(0.3, mask_percentage=0.5)
    for data in dm.train_data:
        rows = data.gene_mask.any(1)
        assert 0 < int(rows.sum()) < data.num_nodes
        assert (data.gene_mask[rows] == data.gene_mask[rows][0]).all() and not data.gene_mask[~rows].any()
    discs = _module(0.3, mask_percentage=0.0, mask_block_radius=150.0, mask_block_fraction=0.3)
    for data in discs.train_data:
        rows = data.gene_mask.any(1)
        assert not rows.all() and int(data.gene_mask[rows][0].sum()) == 3, "discs pick cells, not rows"
        # the discs cover at least their share of the cells; without them the one cell every graph keeps would be all
        assert int(rows.sum()) >= 0.3 * data.num_nodes


def test_the_datamodule_refuses_node_masking_a_fixed_set_beside_it_and_a_bad_share():
    datas = [[_lattice(5, 0)], [_lattice(5, 1)]]
    with pytest.raises(ValueError, match="mask_strategy 'gene'"):
        GraphAnnDataModule(datas=datas, num_workers=0, mask_strategy="node", mask_channel_fraction=0.3)
    with pytest.raises(ValueError, match="give one"):
        GraphAnnDataModule(
            datas=datas, num_workers=0, mask_strategy="gene", mask_channel_fraction=0.3, mask_gene_columns=[1, 2]
        )
    for fraction in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="share of the genes"):
            GraphAnnDataModule(datas=datas, num_workers=0, mask_strategy="gene", mask_channel_fraction=fraction)


def test_off_the_datamodule_draws_no_set_and_no_random_number():
    datas = [[_lattice(5, 0)], [_lattice(5, 1)]]
    torch.manual_seed(3)
    GraphAnnDataModule(datas=datas, num_workers=0, mask_strategy="gene", mask_percentage=1.0, mask_gene_columns=[1, 2])
    state = torch.get_rng_state()
    torch.manual_seed(3)
    GraphAnnDataModule(
        datas=datas,
        num_workers=0,
        mask_strategy="gene",
        mask_percentage=1.0,
        mask_gene_columns=[1, 2],
        mask_channel_fraction=0,
    )
    assert torch.equal(torch.get_rng_state(), state)


# --------------------------------------------------------------------------- the config and the name


def _cfg(**dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    for key, value in dataset.items():
        setattr(cfg.dataset, key, value)
    return cfg


def test_the_fraction_is_off_by_default_a_share_in_zero_one_and_needs_gene_masking():
    from interscale.config import _validate_masking

    assert _cfg().dataset.mask_channel_fraction == 0.0 and not hides_gene_set(_cfg())
    for fraction in (0.3, 0.5, 0.01):
        _validate_masking(_cfg(mask_channel_fraction=fraction))
        assert hides_gene_set(_cfg(mask_channel_fraction=fraction))
    for fraction in (1.0, 1.2, -0.3):
        with pytest.raises(ValueError, match="share of genes to hide"):
            _validate_masking(_cfg(mask_channel_fraction=fraction))
    with pytest.raises(ValueError, match="needs mask_strategy 'gene'"):
        _validate_masking(_cfg(mask_strategy="node", mask_percentage=0.5, mask_channel_fraction=0.3))
    with pytest.raises(ValueError, match="give one"):
        _validate_masking(_cfg(mask_channel_fraction=0.3, mask_genes=["role=response"]))


def test_a_partial_cell_rate_warns_that_the_other_cells_show_every_gene():
    import warnings

    from interscale.config import _validate_masking

    with pytest.warns(UserWarning, match="neighbour can give away a hidden one"):
        _validate_masking(_cfg(mask_channel_fraction=0.3, mask_percentage=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_masking(_cfg(mask_channel_fraction=0.3, mask_percentage=1.0))  # every cell: nothing to say
        _validate_masking(_cfg(mask_strategy="gene", mask_percentage=0.5))  # and random entries say nothing either


def test_discs_may_select_the_cells_alone_as_for_a_gene_set():
    from interscale.config import _validate_block_mask, _validate_masking

    discs = _cfg(mask_channel_fraction=0.3, mask_percentage=0.0, mask_block_radius_um=500.0, mask_block_fraction=0.3)
    _validate_masking(discs)
    _validate_block_mask(discs)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        _validate_masking(_cfg(mask_channel_fraction=0.3, mask_percentage=0.0))


def test_the_kwargs_carry_the_fraction_and_read_no_var_for_it():
    var = pd.DataFrame({"role": ["a", "b"]}, index=["A", "B"])
    assert gene_set_mask_kwargs(_cfg(mask_channel_fraction=0.3), object()) == {"mask_channel_fraction": 0.3}
    both = gene_set_mask_kwargs(_cfg(mask_genes=["role=b"]), type("A", (), {"var": var})())
    assert both == {"mask_gene_columns": [1]}
    assert gene_set_mask_kwargs(_cfg(), object()) == {}


def test_the_checkpoint_name_carries_the_fraction_and_the_cell_rate():
    from interscale.tl.utils import _training_tag

    assert _training_tag(_cfg()) == ""
    assert _training_tag(_cfg(mask_channel_fraction=0.3)) == "var-ch0.3c1_"
    assert _training_tag(_cfg(mask_channel_fraction=0.5)) == "var-ch0.5c1_"
    assert _training_tag(_cfg(mask_channel_fraction=0.3, mask_percentage=0.5)) == "var-ch0.3c0.5_"
    assert _training_tag(_cfg(mask_genes=["role=response"])) == "var-gs1-role-response_"  # unchanged


def test_options_that_need_cells_hidden_whole_refuse_channels():
    from interscale.config import _validate_regime_loss, _validate_remask, _validate_visible_keys_only

    cfg = _cfg(mask_channel_fraction=0.3)
    cfg.model.global_component.name = "self-attn-transformer"
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.remask = "hidden"
    with pytest.raises(ValueError, match="channel masking"):
        _validate_remask(cfg)
    params = cfg.model.global_component.parameters
    with pytest.raises(ValueError, match="channel masking"):
        _validate_visible_keys_only(cfg, params, "dense", type("S", (), {"mask": "dense"})())
    cfg.optim.regime_loss.kind = "depth"
    cfg.model.decoder.dual_decoder = True
    cfg.dataset.mask_block_fraction = 0.3
    cfg.dataset.mask_block_radius_um = 500.0
    with pytest.raises(ValueError, match="channel masking"):
        _validate_regime_loss(cfg)


def test_a_precomputed_embedding_is_refused_for_channels_but_not_for_a_fixed_set():
    from interscale.config import _validate_precomputed_embedding
    from interscale.config.global_component_config import get_global_component_cfg

    def precomputed(**dataset):
        cfg = get_global_component_cfg(_cfg(**dataset), "self-attn-transformer")
        cfg.model.global_component.parameters.type_gex_embedding = "Precomputed"
        cfg.model.global_component.latent_obsm_key = "X_pca"
        return cfg

    _validate_precomputed_embedding(precomputed(mask_genes=["role=response"]))  # computed from the other genes
    with pytest.raises(ValueError, match="every gene at some step"):
        _validate_precomputed_embedding(precomputed(mask_channel_fraction=0.3))


# --------------------------------------------------------------------------- no leakage


def _global_module():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.global_component.parameters.n_heads = 2
    cfg.dataset.spatial_key = "spatial"
    torch.manual_seed(0)
    common = {"n_input": G, "n_output": G, "n_embed": 8, "decoder_type": "linear", "dropout_decoder": 0.0}
    common |= {"decoder_hidden_dims": [8], "mask_percentage": 1.0, "mask_strategy": "gene"}
    return GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **common).eval()


def test_no_hidden_gene_reaches_a_prediction_and_a_visible_one_does():
    module = _global_module()
    dm = _module(0.3)
    batch = Batch.from_data_list(list(dm.train_data[:2]))
    hidden = batch.gene_mask
    assert 0 < int(hidden.sum()) < hidden.numel()
    with torch.no_grad():
        base = module._common_step(batch.clone(), "regression", "node").y_pred
        changed = batch.clone()
        changed.x = batch.x.clone()
        changed.x[hidden] = torch.rand(int(hidden.sum())) * 9.0
        assert torch.equal(module._common_step(changed, "regression", "node").y_pred, base)
        visible = batch.clone()
        visible.x = batch.x.clone()
        column = int((~hidden[0]).nonzero()[0])
        visible.x[:, column] += 1.0
        assert not torch.allclose(module._common_step(visible, "regression", "node").y_pred, base)


def test_a_model_trained_on_channels_is_evaluated_on_a_fixed_programme_set_like_any_other():
    """The evaluation draws its own masks (``draw_masks``): a gene set hidden in every cell, whatever the model trained on."""
    module = _global_module()
    datas = [_lattice(6, 0), _lattice(5, 1)]
    draw_masks(datas, "gene_set", 1.0, seed=0, columns=[4, 5])
    batch = Batch.from_data_list(datas)
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    assert torch.equal(out.entry_mask.any(0).nonzero().flatten(), torch.tensor([4, 5]))


# --------------------------------------------------------------------------- through the trainer


def test_a_model_trains_on_channels_and_the_masks_change_from_epoch_to_epoch(tmp_path):
    import scanpy as sc
    from anndata import AnnData

    from interscale.config import load_config
    from interscale.model.global_model import GlobalModel
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
    h5ad = tmp_path / "channels.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "channels.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 64\n      num_layers: 1\n      n_heads: 2\n      type_gex_embedding: Linear\n"
        f"  decoder:\n    type: linear\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 4\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: channeltest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: gene\n"
        "  mask_percentage: 1.0\n  mask_channel_fraction: 0.3\n  spatial_key: spatial\n  batch_size: 2\n"
        "  evaluate_test: False\n  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    net = GlobalModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg,
        num_workers=0,
        batch_size=2,
        mask_percentage=1.0,
        mask_strategy="gene",
        learning_type="node",
        **gene_set_mask_kwargs(cfg, adata),
    )
    assert dm.mask_channel_fraction == 0.3
    sets = []
    resample = dm.resample_train_mask

    def recording():
        resample()
        sets.append(tuple(dm.train_data[0].gene_mask[0].nonzero().flatten().tolist()))

    dm.resample_train_mask = recording
    net.train(max_epochs=4, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    rows = [row for row in net.history_.history if "val_loss" in row]
    assert rows and all(np.isfinite(row["val_loss"]) for row in rows)
    assert len(sets) >= 3 and len(set(sets)) > 1, f"the training masks never changed: {sets}"
    assert all(len(s) == 3 for s in sets)
    assert any("ch0.3c1" in p.name for p in (tmp_path / "out").iterdir())
