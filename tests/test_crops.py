"""Training on crops (``dataset.crop``, ``tl/crops.py``; roadmap Tier 3, K18).

Pinned here:

* **the cut** -- a square / disc window keeps exactly the cells inside it; the crop slices every per-cell attribute (expression,
  masks, positions, names, precomputed encodings) and keeps per-graph ones, with the edges among the kept cells renumbered;
  a window under ``min_cells`` is drawn again;
* **the draws** -- item ``i`` crops graph ``i // per_graph``; every read draws a new crop, from a private generator (the global
  ones untouched) or, in a loader worker, from the worker's per-epoch seed; the same seed gives the same crops;
* **the masks** -- a crop slices the section's current masks, so an in-place redraw reaches the next crops;
* **the wiring** -- the datamodule trains on crops and validates on whole graphs; ``crop_kwargs`` converts µm; the validator;
  a GlobalModel trains end to end on crops.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from interscale.tl.crops import CropDataset, crop_graph, crop_window

G = 4


def _section(side=20, spacing=10.0, seed=0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * spacing
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= spacing * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen), edge_index=edges, pos=pos)
    data.mask = torch.rand(n, generator=gen) < 0.3
    data.gene_mask = data.mask[:, None].expand(n, G).clone()
    data.obs_names = torch.arange(n) + 1000 * seed
    data.density_radius = torch.rand(n, generator=gen)
    data.spectral_eigval = torch.rand(1, 5, generator=gen)  # per graph
    return data


# --------------------------------------------------------------------------- the cut


def test_the_window_keeps_exactly_the_cells_inside_it():
    pos = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [6.0, 0.0], [0.0, -5.1]])
    centre = torch.tensor([0.0, 0.0])
    assert crop_window(pos, centre, 10.0, "square").tolist() == [True, True, True, False, False]
    assert crop_window(pos, centre, 10.0, "disc").tolist() == [True, True, False, False, False]
    with pytest.raises(ValueError, match="crop shape"):
        crop_window(pos, centre, 10.0, "hexagon")


def test_a_crop_slices_every_per_cell_attribute_and_renumbers_the_edges():
    data = _section()
    crop = crop_graph(data, 55.0, generator=torch.Generator().manual_seed(1))
    kept = crop.obs_names  # seed 0: a cell's name is its index in the section
    assert 20 <= crop.num_nodes <= 36 and crop.num_nodes < data.num_nodes
    for key in ("x", "pos", "mask", "gene_mask", "density_radius"):
        assert torch.equal(getattr(crop, key), getattr(data, key)[kept]), key
    assert torch.equal(crop.spectral_eigval, data.spectral_eigval)  # per graph: kept whole
    # every edge of the crop is an edge of the section between two kept cells, and every such edge is kept
    old = {(int(a), int(b)) for a, b in data.edge_index.t()}
    new = {(int(kept[a]), int(kept[b])) for a, b in crop.edge_index.t()}
    inside = set(kept.tolist())
    assert new == {(a, b) for a, b in old if a in inside and b in inside}
    span = crop.pos.max(0).values - crop.pos.min(0).values
    assert (span <= 55.0).all()


def test_a_window_too_small_is_drawn_again_and_the_largest_kept():
    # 40 cells far apart, each alone in any window, and one cluster of 12: most first draws are a lone cell.
    lone = torch.arange(40, dtype=torch.float32)[:, None] * torch.tensor([[1000.0, 0.0]]) + 5000.0
    cluster = torch.rand(12, 2, generator=torch.Generator().manual_seed(0)) * 20.0
    pos = torch.cat([lone, cluster])
    data = Data(x=torch.zeros(52, G), pos=pos, edge_index=torch.zeros((2, 0), dtype=torch.long))
    for seed in range(5):
        found = crop_graph(data, 50.0, generator=torch.Generator().manual_seed(seed), min_cells=9, tries=40)
        assert found.num_nodes == 12, seed  # redrawn until the window holds enough cells
        best = crop_graph(data, 50.0, generator=torch.Generator().manual_seed(seed), min_cells=100, tries=40)
        assert best.num_nodes == 12, seed  # never enough: the largest window seen


# --------------------------------------------------------------------------- the draws


def test_items_crop_their_graph_each_read_is_new_and_the_seed_repeats_them():
    graphs = [_section(seed=0), _section(seed=1)]
    state = torch.random.get_rng_state()
    data = CropDataset(graphs, size=60.0, shape="square", per_graph=3, min_cells=5, seed=7)
    assert len(data) == 6
    assert set(data[4].obs_names.tolist()) <= set(graphs[1].obs_names.tolist())
    assert set(data[2].obs_names.tolist()) <= set(graphs[0].obs_names.tolist())
    first = [data[i].obs_names for i in range(6)]
    second = [data[i].obs_names for i in range(6)]
    assert any(not torch.equal(a, b) for a, b in zip(first, second, strict=True))  # new crops every read
    again = CropDataset(graphs, size=60.0, shape="square", per_graph=3, min_cells=5, seed=7)
    again[4], again[2]  # the same reads as above, in the same order
    assert all(torch.equal(a, again[i].obs_names) for i, a in enumerate(first))  # the seed repeats them
    assert torch.equal(state, torch.random.get_rng_state())  # the global generator untouched


def test_in_a_worker_each_epoch_draws_new_crops_and_a_seeded_run_repeats_them():
    def epochs(seed):
        torch.manual_seed(seed)
        data = CropDataset([_section()], size=60.0, shape="disc", per_graph=4, min_cells=5, seed=0)
        loader = DataLoader(data, batch_size=1, num_workers=1, shuffle=False)
        return [[batch.obs_names.clone() for batch in loader] for _ in range(2)]

    a = epochs(3)
    assert any(not torch.equal(x, y) for x, y in zip(a[0], a[1], strict=True))  # the second epoch is new
    b = epochs(3)
    assert all(torch.equal(x, y) for e in range(2) for x, y in zip(a[e], b[e], strict=True))


def test_a_crop_slices_the_sections_current_masks():
    section = _section()
    data = CropDataset([section], size=200.0, shape="square", per_graph=1, min_cells=5, seed=0)  # the whole section
    section.mask = torch.zeros(section.num_nodes, dtype=torch.bool)
    section.mask[:7] = True  # redrawn in place, as resample_train_mask does
    crop = data[0]
    assert int(crop.mask.sum()) == 7 and torch.equal(crop.mask, section.mask[crop.obs_names])


# --------------------------------------------------------------------------- the wiring


def _cfg(size=60.0, **dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.crop.size_um = size
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", 0.5
    for key, value in dataset.items():
        cfg.dataset[key] = value
    return cfg


def test_the_datamodule_trains_on_crops_and_validates_on_whole_graphs():
    from interscale.geome_dataloader import GraphAnnDataModule, crop_kwargs

    assert crop_kwargs(_cfg(size=0.0)) == {}
    kwargs = crop_kwargs(_cfg())
    assert kwargs["crop"]["size"] == 120.0  # 60 µm at 0.5 µm per unit
    train, val = [_section(seed=0), _section(seed=1)], [_section(seed=2)]
    dm = GraphAnnDataModule(datas=[train, val], batch_size=2, num_workers=0, mask_percentage=0.3, **kwargs)
    dm.setup("fit")
    assert isinstance(dm.train_dataloader().dataset, CropDataset) and len(dm.train_dataloader()) == 8  # 2 graphs x 8 / 2
    assert dm.val_dataloader().dataset is val
    batch = next(iter(dm.train_dataloader()))
    assert batch.num_graphs == 2 and batch.num_nodes < 2 * 400
    dm.resample_train_mask()  # the masks are the sections', redrawn in place
    plain = GraphAnnDataModule(datas=[train, val], batch_size=2, num_workers=0, mask_percentage=0.3)
    plain.setup("fit")
    assert plain.train_dataloader().dataset is train


def test_the_validator_wants_coordinates_node_level_and_warns_about_per_sequence_quantities():
    from interscale.config import _validate_crop
    from interscale.config.global_component_config import get_global_component_cfg

    _validate_crop(_cfg(size=0.0, spatial_key=None))
    for change, match in (
        ({"spatial_key": None}, "spatial_key"),
        ({"prediction_level": "graph"}, "prediction_level node"),
    ):
        with pytest.raises(ValueError, match=match):
            _validate_crop(_cfg(**change))
    bad = _cfg()
    bad.dataset.crop.shape = "hexagon"
    with pytest.raises(ValueError, match="shape"):
        _validate_crop(bad)
    bad = _cfg(size=-1.0)
    with pytest.raises(ValueError, match=">= 0"):
        _validate_crop(bad)
    centred = _cfg()
    centred.model.global_component.name = "self-attn-transformer"
    centred = get_global_component_cfg(centred, "self-attn-transformer")
    centred.model.global_component.parameters.attention.center_qk = True
    with pytest.warns(UserWarning, match="center_qk"):
        _validate_crop(centred)


def test_a_global_model_trains_end_to_end_on_crops(tmp_path):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, crop_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    per, samples = 120, ["s1", "s2", "s3"]
    adata = sc.AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "crops.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "crops.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n      max_seq_len: 200\n"
        "      num_layers: 1\n      n_heads: 2\n      long_range_attention: False\n      type_gex_embedding: Linear\n"
        f"  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: croptest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  crop:\n    size_um: 250.0\n    per_graph: 6\n    min_cells: 10\n"
        "  spatial_neigbors_kwargs:\n    radius: 80\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(h5ad)
    GlobalModel._setup_anndata(
        adata=adata, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    net = GlobalModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node", **crop_kwargs(cfg)
    )
    net.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    rows = [row for row in net.history_.history if "val_loss" in row]
    assert rows and all(np.isfinite(row["val_loss"]) for row in rows)
    assert len(dm.train_dataloader()) == 3  # one training section x 6 crops / batch 2: three steps an epoch, not one
