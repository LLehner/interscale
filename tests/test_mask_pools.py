"""Randomized vs pre-selected gene sets: channels from a pool, and whole sets drawn from a pool (masking_plan.md, M1).

``dataset.mask_channel_genes`` (the pool channel masking draws from) and ``dataset.mask_gene_sets`` /
``mask_sets_per_draw`` (some whole sets of a pool, hidden together). Pinned here:

* **the draws** -- a pooled channel draw takes exactly ``round(fraction * pool)`` columns of the pool and never one outside
  it, every pool column equally likely; without a pool the draw is the old one, random number for random number; a set
  draw is the union of ``k`` distinct sets, every set equally likely; both reproducible from a generator, both refuse what
  cannot be drawn;
* **the datamodule** -- one draw per graph shared by its selected cells, the training graphs drawn again at every
  redraw, the validation graphs kept; the refusals (no fraction for a pool, sets beside another hidden set, node
  masking, a draw size the pool cannot give); off draws no random number;
* **the config, the kwargs and the name** -- off by default; the validator's refusals; selectors resolved on ``var``;
  distinct checkpoint names; the options that need cells hidden whole refuse a set pool;
* **no leakage** -- through the real model a hidden gene of a drawn set reaches no prediction;
* **through the trainer** -- a model trains on a set pool and the sets change from epoch to epoch.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
from interscale.tl.masking import (
    hides_gene_set,
    sample_channel_columns,
    sample_channel_mask,
    sample_set_columns,
    sample_sets_mask,
)

G = 12
POOL = [1, 3, 4, 7, 8, 10]
SETS = [[0, 1], [2], [3, 4, 5], [9, 10]]


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


# --------------------------------------------------------------------------- the draws


@pytest.mark.parametrize(("fraction", "count"), [(0.3, 2), (0.5, 3), (0.01, 1), (0.99, 5)])
def test_a_pooled_draw_takes_the_rounded_share_of_the_pool_and_nothing_else(fraction, count):
    for seed in range(10):
        columns = sample_channel_columns(G, fraction, torch.Generator().manual_seed(seed), pool=POOL)
        assert len(columns) == count == len(set(columns)) and columns == sorted(columns)
        assert set(columns) <= set(POOL)


def test_every_pool_gene_is_equally_likely_and_the_others_never():
    generator = torch.Generator().manual_seed(0)
    counts = torch.zeros(G)
    for _ in range(3000):
        counts[sample_channel_columns(G, 0.5, generator, pool=POOL)] += 1
    assert (counts[POOL] / 3000 - 0.5).abs().max() < 0.04
    assert counts[[c for c in range(G) if c not in POOL]].sum() == 0


def test_without_a_pool_the_draw_is_the_old_one():
    for seed in range(5):
        old = sorted(torch.randperm(G, generator=torch.Generator().manual_seed(seed))[: round(0.3 * G)].tolist())
        assert sample_channel_columns(G, 0.3, torch.Generator().manual_seed(seed)) == old
        assert sample_channel_columns(G, 0.3, torch.Generator().manual_seed(seed), pool=None) == old


def test_a_pooled_draw_refuses_a_pool_it_cannot_split():
    with pytest.raises(ValueError, match="cannot be split"):
        sample_channel_columns(G, 0.3, pool=[4])
    with pytest.raises(ValueError, match="outside"):
        sample_channel_columns(G, 0.3, pool=[1, G])


def test_a_set_draw_is_the_union_of_k_distinct_sets():
    for seed in range(20):
        columns = sample_set_columns(SETS, 2, torch.Generator().manual_seed(seed))
        hits = [s for s in SETS if set(s) <= set(columns)]
        assert len(hits) == 2 and set(columns) == set().union(*map(set, hits))
    assert sample_set_columns(SETS, 4) == sorted(set().union(*map(set, SETS)))
    assert sample_set_columns([[2, 1], [1, 3]], 2) == [1, 2, 3]  # overlapping sets: their union


def test_every_set_is_equally_likely():
    generator = torch.Generator().manual_seed(1)
    counts = np.zeros(len(SETS))
    for _ in range(4000):
        columns = sample_set_columns(SETS, 1, generator)
        counts[[k for k, s in enumerate(SETS) if s == columns]] += 1
    assert counts.sum() == 4000 and np.abs(counts / 4000 - 0.25).max() < 0.03


def test_a_set_draw_is_reproducible_and_refuses_what_cannot_be_drawn():
    a = [sample_set_columns(SETS, 1, torch.Generator().manual_seed(s)) for s in range(8)]
    assert a == [sample_set_columns(SETS, 1, torch.Generator().manual_seed(s)) for s in range(8)]
    assert len({tuple(x) for x in a}) > 1
    for k in (0, 5):
        with pytest.raises(ValueError, match="between 1 and"):
            sample_set_columns(SETS, k)
    with pytest.raises(ValueError, match="no empty one"):
        sample_set_columns([[1], []], 1)


def test_the_masks_share_one_draw_over_the_selected_cells():
    pooled = sample_channel_mask(30, G, 0.5, 1.0, generator=torch.Generator().manual_seed(2), pool=POOL)
    assert (pooled == pooled[0]).all() and set(pooled[0].nonzero().flatten().tolist()) <= set(POOL)
    drawn = sample_sets_mask(30, G, SETS, 1, 1.0, generator=torch.Generator().manual_seed(2))
    assert (drawn == drawn[0]).all() and drawn[0].nonzero().flatten().tolist() in SETS
    blocks = torch.zeros(30, dtype=torch.bool)
    blocks[:4] = True
    only = sample_sets_mask(30, G, SETS, 1, 0.0, blocks=blocks, generator=torch.Generator().manual_seed(0))
    assert torch.equal(only.any(1), blocks)


# --------------------------------------------------------------------------- the datamodule


def _module(**kwargs):
    datas = [[_lattice(6, s) for s in range(3)], [_lattice(5, 9), _lattice(5, 10)]]
    options = {"num_workers": 0, "mask_strategy": "gene", "mask_percentage": 1.0, "learning_type": "node"}
    dm = GraphAnnDataModule(datas=datas, **(options | kwargs))
    dm.setup(stage="fit")
    return dm


def _hidden(data) -> tuple[int, ...]:
    return tuple(data.gene_mask[0].nonzero().flatten().tolist())


@pytest.mark.parametrize(
    "options",
    [{"mask_channel_fraction": 0.5, "mask_channel_pool": POOL}, {"mask_gene_sets": SETS, "mask_sets_per_draw": 1}],
)
def test_training_graphs_are_drawn_again_and_validation_graphs_keep_theirs(options):
    dm = _module(**options)
    allowed = (lambda h: set(h) <= set(POOL)) if "mask_channel_pool" in options else (lambda h: list(h) in SETS)
    for data in [*dm.train_data, *dm.val_data]:
        assert (data.gene_mask == data.gene_mask[0]).all() and data.mask.all() and allowed(_hidden(data))
    val = [_hidden(d) for d in dm.val_data]
    seen = {_hidden(d) for d in dm.train_data}
    for _ in range(10):
        dm.resample_train_mask()
        seen |= {_hidden(d) for d in dm.train_data}
        assert all(allowed(_hidden(d)) for d in dm.train_data)
    assert len(seen) > 2 and [_hidden(d) for d in dm.val_data] == val


def test_the_datamodule_hides_as_many_whole_sets_as_asked():
    dm = _module(mask_gene_sets=SETS, mask_sets_per_draw=2)
    for _ in range(5):
        for data in [*dm.train_data, *dm.val_data]:
            hidden = set(_hidden(data))
            inside = [s for s in SETS if set(s) <= hidden]
            assert len(inside) == 2 and hidden == set().union(*map(set, inside))
        dm.resample_train_mask()


def test_the_datamodule_refuses_what_the_pools_cannot_do():
    datas = [[_lattice(5, 0)], [_lattice(5, 1)]]
    base = {"datas": datas, "num_workers": 0, "mask_strategy": "gene"}
    with pytest.raises(ValueError, match="needs mask_channel_fraction"):
        GraphAnnDataModule(**base, mask_channel_pool=POOL)
    with pytest.raises(ValueError, match="at least 2"):
        GraphAnnDataModule(**base, mask_channel_fraction=0.3, mask_channel_pool=[3])
    with pytest.raises(ValueError, match="give one"):
        GraphAnnDataModule(**base, mask_gene_sets=SETS, mask_gene_columns=[1])
    with pytest.raises(ValueError, match="give one"):
        GraphAnnDataModule(**base, mask_gene_sets=SETS, mask_channel_fraction=0.3)
    with pytest.raises(ValueError, match="mask_strategy 'gene'"):
        GraphAnnDataModule(**(base | {"mask_strategy": "node"}), mask_gene_sets=SETS)
    with pytest.raises(ValueError, match="between 1 and"):
        GraphAnnDataModule(**base, mask_gene_sets=SETS, mask_sets_per_draw=5)
    with pytest.raises(ValueError, match="at least one gene"):
        GraphAnnDataModule(**base, mask_gene_sets=[[1], []])


def test_off_the_datamodule_draws_what_it_drew_before():
    torch.manual_seed(3)
    before = _module(mask_channel_fraction=0.3)
    torch.manual_seed(3)
    after = _module(mask_channel_fraction=0.3, mask_channel_pool=None, mask_gene_sets=None, mask_sets_per_draw=1)
    assert [_hidden(d) for d in before.train_data] == [_hidden(d) for d in after.train_data]


# --------------------------------------------------------------------------- the config, the kwargs and the name


def _cfg(**dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    for key, value in dataset.items():
        setattr(cfg.dataset, key, value)
    return cfg


SET_SELECTORS = [["MC7", "MC8"], ["role=receiver"]]


def test_off_by_default_and_the_validator_refuses_what_cannot_run():
    from interscale.config import _validate_masking

    cfg = _cfg()
    assert cfg.dataset.mask_channel_genes == [] and cfg.dataset.mask_gene_sets == []
    assert cfg.dataset.mask_sets_per_draw == 1 and not hides_gene_set(cfg)
    _validate_masking(_cfg(mask_channel_fraction=0.3, mask_channel_genes=["is_spatial=True"]))
    _validate_masking(_cfg(mask_gene_sets=SET_SELECTORS, mask_sets_per_draw=2))
    assert hides_gene_set(_cfg(mask_gene_sets=SET_SELECTORS))
    refused = [
        ({"mask_channel_genes": ["is_spatial=True"]}, "needs dataset.mask_channel_fraction"),
        ({"mask_gene_sets": ["role=response"]}, "each entry is a set"),
        ({"mask_gene_sets": [["MC7"], []]}, "at least one selector"),
        ({"mask_gene_sets": SET_SELECTORS, "mask_sets_per_draw": 3}, "between 1 and the 2 sets"),
        ({"mask_sets_per_draw": 2}, "which is empty"),
        ({"mask_gene_sets": SET_SELECTORS, "mask_genes": ["role=response"]}, "give one"),
        ({"mask_gene_sets": SET_SELECTORS, "mask_channel_fraction": 0.3}, "give one"),
        ({"mask_gene_sets": SET_SELECTORS, "mask_strategy": "node", "mask_percentage": 0.5}, "mask_strategy 'gene'"),
    ]
    for options, message in refused:
        with pytest.raises(ValueError, match=message):
            _validate_masking(_cfg(**options))
    with pytest.warns(UserWarning, match="give away a hidden one"):
        _validate_masking(_cfg(mask_gene_sets=SET_SELECTORS, mask_percentage=0.5))


def test_the_kwargs_resolve_the_selectors_on_var():
    var = pd.DataFrame({"role": ["receiver", "x", "receiver", "y"], "is_spatial": ["True", "True", "False", "True"]})
    var.index = ["R1", "MC7", "R2", "MC8"]
    adata = type("A", (), {"var": var})()
    pooled = gene_set_mask_kwargs(_cfg(mask_channel_fraction=0.5, mask_channel_genes=["is_spatial=True"]), adata)
    assert pooled == {"mask_channel_fraction": 0.5, "mask_channel_pool": [0, 1, 3]}
    sets = gene_set_mask_kwargs(_cfg(mask_gene_sets=SET_SELECTORS, mask_sets_per_draw=2), adata)
    assert sets == {"mask_gene_sets": [[1, 3], [0, 2]], "mask_sets_per_draw": 2}
    with pytest.raises(ValueError, match="matches no gene"):
        gene_set_mask_kwargs(_cfg(mask_gene_sets=[["MC9"]]), adata)


def test_the_checkpoint_name_tells_every_pool_apart():
    from interscale.tl.utils import _training_tag

    assert _training_tag(_cfg(mask_channel_fraction=0.3)) == "var-ch0.3c1_"  # unchanged
    pooled = _training_tag(_cfg(mask_channel_fraction=0.3, mask_channel_genes=["is_spatial=True"]))
    assert pooled == "var-ch0.3c1-pis-spatial-True_"
    one = _training_tag(_cfg(mask_gene_sets=SET_SELECTORS))
    two = _training_tag(_cfg(mask_gene_sets=SET_SELECTORS, mask_sets_per_draw=2))
    other = _training_tag(_cfg(mask_gene_sets=[["MC7"], ["role=receiver"]]))
    assert one.startswith("var-sets1of2c1-") and two.startswith("var-sets2of2c1-")
    assert len({one, two, other, pooled}) == 4


def test_options_that_need_cells_hidden_whole_refuse_a_set_pool():
    from interscale.config import _validate_remask
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_global_component_cfg(_cfg(mask_gene_sets=SET_SELECTORS), "self-attn-transformer")
    cfg.model.global_component.parameters.remask = "hidden"
    with pytest.raises(ValueError, match="pool of gene sets"):
        _validate_remask(cfg)


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


def test_no_hidden_gene_of_a_drawn_set_reaches_a_prediction():
    module = _global_module()
    dm = _module(mask_gene_sets=SETS, mask_sets_per_draw=2)
    batch = Batch.from_data_list(list(dm.train_data[:2]))
    hidden = batch.gene_mask
    assert 0 < int(hidden.sum()) < hidden.numel()
    with torch.no_grad():
        base = module._common_step(batch.clone(), "regression", "node").y_pred
        changed = batch.clone()
        changed.x = batch.x.clone()
        changed.x[hidden] = torch.rand(int(hidden.sum())) * 9.0
        assert torch.equal(module._common_step(changed, "regression", "node").y_pred, base)


# --------------------------------------------------------------------------- through the trainer


def test_a_model_trains_on_a_set_pool_and_the_sets_change_from_epoch_to_epoch(tmp_path):
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
    h5ad = tmp_path / "sets.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "sets.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 64\n      num_layers: 1\n      n_heads: 2\n      type_gex_embedding: Linear\n"
        f"  decoder:\n    type: linear\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 4\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: setstest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: gene\n"
        "  mask_percentage: 1.0\n  mask_gene_sets: [[g0, g1], [g2], [g3, g4, g5], [g9, g10]]\n"
        "  spatial_key: spatial\n  batch_size: 2\n"
        "  evaluate_test: False\n  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=adata, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
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
    assert dm.mask_gene_sets == SETS
    seen = []
    resample = dm.resample_train_mask

    def recording():
        resample()
        seen.append(_hidden(dm.train_data[0]))

    dm.resample_train_mask = recording
    net.train(max_epochs=4, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    rows = [row for row in net.history_.history if "val_loss" in row]
    assert rows and all(np.isfinite(row["val_loss"]) for row in rows)
    assert len(seen) >= 3 and len(set(seen)) > 1 and all(list(s) in SETS for s in seen)
    assert any("sets1of4c1-" in p.name for p in (tmp_path / "out").iterdir())
