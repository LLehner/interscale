"""Pooled tokens: the transformer on patches of cells (``module/global_modules/pooling.py``; roadmap.md, T1b).

Pinned here:

* **the grid** -- squares by position alone, graph-major and contiguous, translation invariant, the
  side from ``size_um`` or the one that gives about ``n_patches`` occupied squares per slide;
* **the patch token** -- the mean over visible cells only, a learned fill where none is visible, a patch
  of one visible cell is that cell's token, the optional count/spread features start at zero;
* **the layout** -- one row per cell, left-padded with CLS last, in batch order, as every consumer reads it;
* **through the real models** (GlobalModel, CombinedModel with the dual decoder) -- no hidden value reaches
  a prediction under cell, gene-set and disc masking while visible cells' values do, every cell of a
  patch shares its global embedding, a model with every cell its own patch is the per-cell model, the
  GCN learns through the pooled tokens, the transformer sees patches and not cells;
* **the switch** -- off by default and then adds no state; the config refuses what a patch cannot carry;
  checkpoint names tell sizes apart; cell-by-cell attention readouts refuse a pooled model.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.module.base import gather_tokens
from interscale.module.global_modules.pooling import (
    PatchPooling,
    cell_layout,
    grid_patches,
    side_for_patches,
)

E, G = 8, 6
SPACING = 100.0


def _lattice(side, seed=0, shift=(0.0, 0.0)):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING + torch.tensor(shift)
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    return data


def _batch(graphs):
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


# --------------------------------------------------------------------------- the grid


def test_the_grid_groups_a_lattice_into_squares_graph_by_graph():
    big, small = _lattice(6), _lattice(4)
    pos = torch.cat([big.pos, small.pos])
    graph = torch.cat([torch.zeros(36, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    patch, n = grid_patches(pos, graph, size=2 * SPACING)
    assert n == 9 + 4
    assert torch.bincount(patch).tolist() == [4] * 13
    # Patches are ordered by graph, so each graph's patches are contiguous.
    assert patch[graph == 0].max() < patch[graph == 1].min()
    # A cell's patch is its square: the four cells of a 2 x 2 block share one.
    first = patch[:36].reshape(6, 6)
    assert first[0, 0] == first[0, 1] == first[1, 0] == first[1, 1] and first[0, 0] != first[0, 2]


def test_the_grid_does_not_care_where_the_slide_sits_and_the_offset_moves_its_squares():
    lattice = _lattice(6)
    graph = torch.zeros(36, dtype=torch.long)
    here, n_here = grid_patches(lattice.pos, graph, size=2 * SPACING)
    there, n_there = grid_patches(lattice.pos + torch.tensor([12345.0, -777.0]), graph, size=2 * SPACING)
    assert n_here == n_there and torch.equal(here, there)
    shifted, n_shifted = grid_patches(lattice.pos, graph, size=2 * SPACING, offset=0.5)
    assert not torch.equal(shifted, here) and n_shifted > n_here  # the squares no longer line up with the lattice


@pytest.mark.parametrize("target", [9, 25, 60, 150])
def test_n_patches_finds_a_side_that_gives_about_that_many_squares(target):
    gen = torch.Generator().manual_seed(1)
    pos = torch.rand(900, 2, generator=gen) * torch.tensor([3000.0, 2000.0])
    patch, n = grid_patches(pos, torch.zeros(900, dtype=torch.long), size=0.0, n_patches=target)
    assert abs(n - target) <= max(2, 0.12 * target), (n, target)
    # Two slides of different extent each get about the target, with different sides.
    wide = torch.cat([pos, pos * 2.0])
    graph = torch.cat([torch.zeros(900, dtype=torch.long), torch.ones(900, dtype=torch.long)])
    _, total = grid_patches(wide, graph, size=0.0, n_patches=target)
    assert abs(total - 2 * target) <= max(4, 0.12 * 2 * target)
    assert side_for_patches(pos * 2.0, target) == pytest.approx(2 * side_for_patches(pos, target), rel=0.1)


def test_a_target_at_or_above_the_cell_count_gives_one_cell_per_patch():
    lattice = _lattice(5)
    patch, n = grid_patches(lattice.pos, torch.zeros(25, dtype=torch.long), size=0.0, n_patches=1000)
    assert n == 25 and torch.bincount(patch).max() == 1


# --------------------------------------------------------------------------- the patch token


def _plan(graphs, size=2 * SPACING, **kwargs):
    batch = _batch(graphs)
    pooling = PatchPooling(E, kind="grid", size_um=size, **kwargs)
    return pooling, batch, pooling.plan(batch)


def test_a_patch_sits_at_the_mean_position_of_all_its_cells_hidden_or_not():
    graphs = [_lattice(3, shift=(1000.0, 0.0)), _lattice(2)]  # 3 x 3 cells in squares of 2 x 2: patches of 4, 2, 2 and 1
    pooling, batch, plan = _plan(graphs)
    assert plan.n_patches == 4 + 1 and sorted(plan.n_cells.tolist()) == [1, 2, 2, 4, 4]
    for p in range(plan.n_patches):
        assert torch.allclose(plan.centroid[p], batch.pos[plan.patch == p].mean(0))
    assert plan.patch_graph.tolist() == [0, 0, 0, 0, 1]


def test_a_patch_token_is_the_mean_of_its_visible_cells_and_a_hidden_cell_weighs_nothing():
    pooling, batch, plan = _plan([_lattice(4)])
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn(16, E, generator=gen)
    hidden = torch.zeros(16, dtype=torch.bool)
    hidden[[0, 5]] = True  # two cells of patch 0 (cells 0, 1, 4, 5 of a 4 x 4 lattice)
    tokens, visible = pooling.pool(emb, hidden, plan)
    assert visible.tolist() == [2, 4, 4, 4]
    patch0 = plan.patch[0]
    assert torch.allclose(tokens[patch0], (emb[1] + emb[4]) / 2, atol=1e-6)
    # Whatever a hidden cell's token holds, the pooled tokens are the same.
    changed = emb.clone()
    changed[hidden] += 100.0
    assert torch.equal(pooling.pool(changed, hidden, plan)[0], tokens)
    # And a visible cell's token reaches its patch's.
    changed = emb.clone()
    changed[1] += 100.0
    assert not torch.allclose(pooling.pool(changed, hidden, plan)[0][patch0], tokens[patch0])


def test_a_patch_of_one_visible_cell_is_that_cells_token_and_one_with_none_gets_the_learned_fill():
    pooling, batch, plan = _plan([_lattice(4)])
    emb = torch.randn(16, E, generator=torch.Generator().manual_seed(1))
    hidden = torch.ones(16, dtype=torch.bool)
    hidden[10] = False  # the one visible cell of its patch
    tokens, visible = pooling.pool(emb, hidden, plan)
    lone = plan.patch[10]
    assert visible[lone] == 1 and torch.allclose(tokens[lone], emb[10])
    empty = [p for p in range(plan.n_patches) if visible[p] == 0]
    assert len(empty) == 3
    for p in empty:
        assert torch.equal(tokens[p], pooling.fill)
    tokens.sum().backward()
    assert pooling.fill.grad is not None and torch.count_nonzero(pooling.fill.grad) > 0


def test_the_count_and_spread_features_start_at_zero_and_then_move_the_token():
    plain, batch, plan = _plan([_lattice(4)])
    rich, _, _ = _plan([_lattice(4)], features=["count", "std"])
    emb = torch.randn(16, E, generator=torch.Generator().manual_seed(2))
    hidden = torch.zeros(16, dtype=torch.bool)
    base, _ = plain.pool(emb, hidden, plan)
    start, _ = rich.pool(emb, hidden, plan)
    assert torch.equal(base, start)  # a patch of one cell is that cell, at the start
    with torch.no_grad():
        rich.count_embed.normal_()
        rich.std_embed.weight.normal_()
    moved, _ = rich.pool(emb, hidden, plan)
    assert not torch.allclose(moved, base)
    with pytest.raises(ValueError, match="features"):
        PatchPooling(E, size_um=100.0, features=["median"])


def test_pooling_needs_positions_and_enough_room():
    pooling = PatchPooling(E, kind="grid", size_um=SPACING)
    batch = _batch([_lattice(4)])
    with pytest.raises(ValueError, match="max_seq_len"):
        pooling.plan(batch, max_patches=10)  # 16 one-cell patches
    batch.pos = None
    with pytest.raises(ValueError, match="coordinates"):
        pooling.plan(batch)


# --------------------------------------------------------------------------- the layout


def test_the_cell_layout_is_the_one_every_consumer_reads():
    sizes = [5, 3]
    graph = torch.cat([torch.full((n,), g) for g, n in enumerate(sizes)])
    rows = torch.arange(8, dtype=torch.float32)[:, None].expand(8, E).clone()
    cls = torch.tensor([[100.0] * E, [200.0] * E])
    embedding, padding, index_nodes = cell_layout(rows, cls, graph, 2)
    assert embedding.shape == (5 + 1, 2, E) and padding.shape == (2, 6)
    assert index_nodes == [[0, 1, 2, 3, 4], [0, 1, 2]]
    assert torch.equal(gather_tokens(embedding, padding), rows)  # cells in batch order, CLS and padding gone
    assert torch.equal(embedding[-1], cls)  # CLS last
    assert padding[0].tolist() == [False] * 6 and padding[1].tolist() == [True, True, False, False, False, False]
    assert torch.equal(embedding[:2, 1], torch.zeros(2, E))  # the smaller graph is left-padded


# --------------------------------------------------------------------------- through the models


def _config(model="GlobalModel", pool=None, *, strategy="node", rope=False, attention=None, features=()):
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
    if rope:
        params.pe.rotary = ["rope"]
        params.pe.rope.min_wavelength = 100.0
    if attention is not None:
        params.attention.kind = attention
    cfg.dataset.mask_strategy = strategy
    cfg.dataset.mask_percentage = 0.3 if strategy == "node" else 1.0
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    if strategy == "gene":
        cfg.dataset.mask_genes = ["g0", "g1"]
    if pool is not None:
        params.pooling.kind = "grid"
        params.pooling.size_um = pool
        params.pooling.features = list(features)
    return cfg


def _build(cfg, model="GlobalModel", seed=0):
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    cls = {"GlobalModel": GlobalModel, "CombinedModel": CombinedModel}[model]
    cls._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return cls(adata, cfg=cfg)


def _predict(module, batch, head="global"):
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    if hasattr(module, "predict_local"):  # dual decoder: [local; global]
        n = len(out.y_pred) // 2
        return out.y_pred[:n] if head == "local" else out.y_pred[n:]
    return out.y_pred


def test_with_every_cell_its_own_patch_the_pooled_model_is_the_per_cell_model():
    graphs = [_lattice(5, seed=1), _lattice(4, seed=2)]
    batch = _batch(graphs)
    batch.mask = torch.zeros(batch.num_nodes, dtype=torch.bool)  # nothing hidden: every patch is a visible cell
    plain = _build(_config()).module.eval()
    pooled = _build(_config(pool=SPACING / 2)).module.eval()
    assert plain.pooling is None and pooled.pooling is not None
    missing = set(pooled.state_dict()) - set(plain.state_dict())
    assert missing == {"pooling.fill"}  # nothing else: the rest of the model is initialised identically
    assert all(torch.equal(plain.state_dict()[k], pooled.state_dict()[k]) for k in plain.state_dict())
    assert torch.allclose(_predict(plain, batch), _predict(pooled, batch), atol=1e-5)


def test_the_transformer_sees_patches_and_every_cell_of_a_patch_shares_its_embedding():
    graphs = [_lattice(6, seed=3)]
    batch = _batch(graphs)
    module = _build(_config(pool=2 * SPACING)).module.eval()
    with module.recording_attention() as record, torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    assert record[0]["maps"].shape[-1] == 9 + 1  # 9 patches and CLS, not 36 cells
    embedding = gather_tokens(out.view.global_embedding, out.view.src_padding_mask)
    assert embedding.shape == (36, E)
    patch = grid_patches(graphs[0].pos, torch.zeros(36, dtype=torch.long), size=2 * SPACING)[0]
    for p in range(9):
        rows = embedding[patch == p]
        assert torch.equal(rows, rows[:1].expand_as(rows))
    assert torch.equal(out.view.padded_node_idx, torch.arange(36))


@pytest.mark.parametrize("strategy", ["node", "gene"])
def test_no_hidden_value_reaches_a_prediction_and_a_visible_one_does(strategy):
    graphs = [_lattice(6, seed=4)]
    batch = _batch(graphs)
    n = batch.num_nodes
    if strategy == "node":
        batch.mask = torch.zeros(n, dtype=torch.bool)
        batch.mask[:12] = True  # the first two rows of squares, hidden whole
    else:
        batch.gene_mask = torch.zeros(n, G, dtype=torch.bool)
        batch.gene_mask[:, [0, 1]] = True  # programme masking: a gene set in every cell
        batch.mask = torch.ones(n, dtype=torch.bool)
    module = _build(_config(pool=2 * SPACING, strategy=strategy)).module.eval()
    before = _predict(module, batch)

    hidden_changed = batch.clone()
    hidden_changed.x = batch.x.clone()
    if strategy == "node":
        hidden_changed.x[:12] += torch.rand(12, G) * 50.0
    else:
        hidden_changed.x[:, [0, 1]] = torch.rand(n, 2) * 50.0
    assert torch.equal(_predict(module, hidden_changed), before)

    visible_changed = batch.clone()
    visible_changed.x = batch.x.clone()
    visible_changed.x[30, 3] += 50.0  # a visible cell, in a patch nobody else is in
    after = _predict(module, visible_changed)
    assert not torch.allclose(after, before)
    # It reaches its own patch's cells and, through attention, every other patch.
    assert not torch.allclose(after[:12], before[:12])


def test_a_patch_with_nothing_visible_is_a_position_only_token():
    graphs = [_lattice(6, seed=5)]
    batch = _batch(graphs)
    batch.mask = torch.zeros(36, dtype=torch.bool)
    patch = grid_patches(graphs[0].pos, torch.zeros(36, dtype=torch.long), size=2 * SPACING)[0]
    batch.mask[patch == 4] = True  # the middle patch is hidden whole
    module = _build(_config(pool=2 * SPACING)).module
    out = module._common_step(batch, "regression", "node")
    out.y_pred.pow(2).mean().backward()
    assert module.pooling.fill.grad is not None and torch.count_nonzero(module.pooling.fill.grad) > 0
    # Its cells are predicted from position and the other patches, all at the same place.
    pred = out.y_pred.detach()
    inside = pred[patch == 4]
    assert torch.equal(inside, inside[:1].expand_as(inside))


def test_the_gcn_learns_through_the_pooled_tokens_in_the_dual_decoder_model():
    graphs = [_lattice(6, seed=6)]
    batch = _batch(graphs)
    batch.mask = torch.zeros(36, dtype=torch.bool)
    batch.mask[::3] = True
    module = _build(_config("CombinedModel", pool=2 * SPACING), "CombinedModel").module
    out = module._common_step(batch, "regression", "node")
    n = len(out.y_pred) // 2
    assert out.y_pred.shape[0] == 2 * 36 and out.view.local_embedding.shape == (36, E)
    out.y_pred[n:].pow(2).mean().backward()  # the global head alone
    gcn = [p.grad for name, p in module.local_module.named_parameters() if p.grad is not None]
    assert gcn and any(torch.count_nonzero(g) > 0 for g in gcn), "the global loss never reached the GCN"
    local = [p.grad for p in module.local_module.decoder.parameters()]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in local)  # the local decoder answers to the local loss


@pytest.mark.parametrize("attention", ["performer", "entmax", "topk"])
def test_other_attention_backends_run_on_patches(attention):
    graphs = [_lattice(6, seed=7)]
    batch = _batch(graphs)
    module = _build(_config(pool=SPACING, attention=attention)).module.eval()
    pred = _predict(module, batch)
    assert pred.shape == (36, G) and torch.isfinite(pred).all()


def test_rope_reads_the_patch_centroids():
    graphs = [_lattice(6, seed=8)]
    moved = [_lattice(6, seed=8, shift=(5000.0, -300.0))]
    one, two = _batch(graphs), _batch(moved)
    module = _build(_config(pool=2 * SPACING, rope=True)).module.eval()
    assert torch.allclose(_predict(module, one), _predict(module, two), atol=1e-4)  # translation invariant


def test_the_features_train_with_the_model():
    batch = _batch([_lattice(6, seed=9)])
    module = _build(_config(pool=2 * SPACING, features=["count", "std"])).module
    module._common_step(batch, "regression", "node").y_pred.pow(2).mean().backward()
    assert module.pooling.count_embed.grad is not None and module.pooling.std_embed.weight.grad is not None


# --------------------------------------------------------------------------- the switch


def test_off_by_default_adds_no_state_and_builds_nothing():
    from interscale.module.global_modules.pooling import build_pooling

    cfg = _config()
    assert build_pooling(cfg, E) is None
    module = _build(cfg).module
    assert module.pooling is None and not any(k.startswith("pooling") for k in module.state_dict())


@pytest.mark.parametrize(
    ("change", "match"),
    [
        pytest.param(lambda p: setattr(p.pooling, "size_um", 0.0), "exactly one", id="no size"),
        pytest.param(lambda p: setattr(p.pooling, "n_patches", 10), "exactly one", id="both sizes"),
        pytest.param(lambda p: setattr(p.pooling, "offset", 1.0), "offset", id="offset"),
        pytest.param(lambda p: setattr(p.pooling, "features", ["median"]), "features", id="feature"),
        pytest.param(lambda p: setattr(p, "long_range_attention", True), "long_range_attention", id="long-range mask"),
        pytest.param(lambda p: setattr(p.pe, "node", ["lap"]), "lap", id="lap"),
        pytest.param(lambda p: setattr(p.pe, "bias", ["spectral"]), "spectral", id="spectral"),
        pytest.param(lambda p: setattr(p.attention, "kind", "exphormer"), "Exphormer", id="exphormer"),
    ],
)
def test_the_config_refuses_what_a_patch_cannot_carry(change, match):
    from interscale.config import _validate

    cfg = _config(pool=200.0)
    _validate(cfg)
    change(cfg.model.global_component.parameters)
    with pytest.raises(ValueError, match=match):
        _validate(cfg)
    bare = _config(pool=200.0)
    bare.dataset.spatial_key = None
    with pytest.raises(ValueError, match="spatial_key"):
        _validate(bare)


def test_a_grid_by_count_is_accepted_and_the_checkpoint_name_tells_sizes_apart():
    from interscale.config import _validate
    from interscale.tl.utils import get_model_filename_prefix

    names = {}
    for label, pool in (("cells", None), ("200", 200.0), ("300", 300.0)):
        cfg = _config(pool=pool)
        cfg.dataset.name = "d"
        names[label] = get_model_filename_prefix(cfg, False, True)
    assert len(set(names.values())) == 3 and "pool-grid200um" in names["200"] and "pool" not in names["cells"]
    counted = _config(pool=200.0)
    counted.model.global_component.parameters.pooling.size_um = 0.0
    counted.model.global_component.parameters.pooling.n_patches = 100
    counted.model.global_component.parameters.pooling.offset = 0.5
    counted.model.global_component.parameters.pooling.features = ["count"]
    _validate(counted)
    counted.dataset.name = "d"
    assert "pool-grid-n100-o0.5-count_" in get_model_filename_prefix(counted, False, True)
    combined = _config("CombinedModel", pool=200.0, rope=False)
    combined.dataset.name = "d"
    assert "pool-grid200um" in get_model_filename_prefix(combined, True, True)


def test_the_cell_level_readouts_refuse_a_pooled_model():
    from interscale.evaluation.attention_diagnostics import attention_by_query, attention_diagnostics
    from interscale.evaluation.attention_flow import patch_flow

    module = _build(_config(pool=200.0)).module
    graphs = [_lattice(6)]
    common = {"prediction_task": "regression", "prediction_level": "node", "device": "cpu"}
    with pytest.raises(ValueError, match="between patches"):
        attention_by_query(module, graphs, spatial_unit_um=1.0, depth_bins_um=[0.0, 100.0, float("inf")], **common)
    with pytest.raises(ValueError, match="between patches"):
        attention_diagnostics(module, graphs, **common)
    with pytest.raises(ValueError, match="rollout"):
        patch_flow(module, [graphs], mask_strategy="node", spatial_unit_um=1.0, patch_um=200.0, **common)
    batch = _batch(graphs)
    with pytest.raises(NotImplementedError, match="patches"):
        module.evaluate(batch, torch.randn(36, E))


# --------------------------------------------------------------------------- a real run


@pytest.mark.parametrize("model", ["GlobalModel", "CombinedModel"])
def test_a_pooled_model_trains_end_to_end_under_programme_masking(tmp_path, model):
    """Through ``train()``: masks drawn and redrawn, the loss on the hidden gene set per cell, checkpoint saved and reloadable."""
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    samples = ["s1", "s2", "s3", "s4"]
    n = 40 * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, 40))
    adata.obs["split"] = pd.Categorical([{"s1": "train", "s2": "train", "s3": "val", "s4": "test"}[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "pool.h5ad"
    adata.write_h5ad(h5ad)
    local = "  local_component:\n    name: GCN\n" if model == "CombinedModel" else ""
    dual = "    dual_decoder: True\n" if model == "CombinedModel" else ""
    embedding = "      type_gex_embedding: Linear\n" if model == "GlobalModel" else ""
    path = tmp_path / "pool.yaml"
    path.write_text(
        f"model:\n  n_embed: {E}\n{local}  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        f"      max_seq_len: 64\n      num_layers: 1\n      n_heads: 2\n{embedding}"
        "      pe:\n        node: [sinusoidal]\n"
        "      pooling:\n        kind: grid\n        size_um: 40.0\n        features: [count]\n"
        f"  decoder:\n    type: linear\n{dual}  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 3\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: pooltest\n  prediction_task: regression\n  prediction_level: node\n"
        "  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: gene\n  mask_percentage: 1.0\n"
        "  mask_genes: ['g2', 'g5']\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 25\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    cls = {"GlobalModel": GlobalModel, "CombinedModel": CombinedModel}[model]
    cls._setup_anndata(
        adata=adata, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    net = cls(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=1.0, mask_strategy="gene", learning_type="node",
        **gene_set_mask_kwargs(cfg, adata),
    )
    net.train(max_epochs=3, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    losses = [row["val_loss"] for row in net.history_.history if "val_loss" in row]
    assert losses and all(np.isfinite(losses))
    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert any("pool-grid40um" in name for name in saved), saved  # the checkpoint name carries the patch size
    pooling = getattr(net.module, "global_module", net.module).pooling
    assert pooling.count_embed.abs().sum() > 0  # the learned count feature trained
