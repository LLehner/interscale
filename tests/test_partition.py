"""V1, unique and shared R² per gene (yardsticks_plan.md): ``evaluation.partition``.

Pinned here:

* **the near block** -- per entry the mean of the visible values over the reach, the graph's visible mean where no neighbour
  shows the gene, 0 where the graph shows none;
* **the folds** -- by graph no graph is split; spatial blocks stay within a graph; reproducible;
* **the partition** -- on planted features the unique and shared fractions are what was planted; a block of pure noise with as
  many features as rows earns nothing out of fold; Shapley shares add up to the total; a per-graph offset of the target is not
  counted; refusals;
* **the collector** -- through a global-only and a dual toy model: the rows, truth and hidden entries are the step's, the
  global block is what the decoder reads, ``own`` is the visible input (0 where hidden), ``near`` the hand-computed visible mean
  over the hop reach; folds hide every cell once and give the masks back; no global generator is touched.
"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from interscale.evaluation.partition import (
    FeatureBlocks,
    collect_feature_blocks,
    partition_r2,
    spatial_folds,
    visible_neighbour_mean,
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------- the near block


def test_visible_neighbour_mean_by_hand():
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 40.0], [9.0, 50.0]])
    visible = np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0]], bool)
    graph = np.array([0, 0, 0, 1, 1])
    reach = sp.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],  # no neighbour: graph 1's visible mean
                [0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
    )
    out = visible_neighbour_mean(x, visible, reach, graph)
    assert np.allclose(out[0], [3.0, 30.0])  # gene 0: only cell 1 visible; gene 1: only cell 2
    assert np.allclose(out[1], [1.0, 10.0])
    assert np.allclose(out[2], [2.0, 10.0])
    assert np.allclose(out[3], [8.0, 40.0])  # graph 1: gene 0 visible at both, gene 1 at cell 3 only
    assert np.allclose(out[4], [7.0, 40.0])
    hidden_all = visible_neighbour_mean(x, np.zeros_like(visible), reach, graph)
    assert np.all(hidden_all == 0.0)


# ------------------------------------------------------------------------------- folds


def test_folds_by_graph_and_by_block():
    rng = np.random.default_rng(0)
    graph = np.repeat(np.arange(6), 50)
    pos = rng.uniform(0, 100, size=(300, 2))
    f = spatial_folds(graph, n_folds=3, seed=1)
    for g in range(6):
        assert len(np.unique(f[graph == g])) == 1
    assert sorted(np.bincount(f)) == [100, 100, 100]
    assert np.array_equal(f, spatial_folds(graph, n_folds=3, seed=1))
    b = spatial_folds(graph, pos, n_folds=4, block=25.0, seed=0)
    cell = np.floor(pos / 25.0).astype(int)
    for g in range(6):
        for key in {tuple(c) for c in cell[graph == g]}:
            here = (graph == g) & (cell[:, 0] == key[0]) & (cell[:, 1] == key[1])
            assert len(np.unique(b[here])) == 1
    assert len(np.unique(b)) == 4
    coarse = np.floor(pos / 50.0).astype(int)  # the 25-blocks are dealt one by one: a 50-block holds several folds
    varies = [
        len(np.unique(b[(graph == g) & (coarse[:, 0] == c0) & (coarse[:, 1] == c1)])) > 1
        for g in range(6)
        for c0, c1 in {tuple(c) for c in coarse[graph == g]}
    ]
    assert np.mean(varies) > 0.5
    with pytest.raises(ValueError, match="pass `block`"):
        spatial_folds(np.zeros(10, int), n_folds=3)
    with pytest.raises(ValueError, match="positions"):
        spatial_folds(graph, None, n_folds=3, block=10.0)


# ------------------------------------------------------------------------------- the partition


def _planted(n=3000, seed=0):
    """Gene 0: half from a signal both blocks hold, half from A's own; gene 1: B's own; gene 2: noise."""
    rng = np.random.default_rng(seed)
    common, own_a, own_b = rng.normal(size=(3, n))
    a = np.column_stack([common, own_a, rng.normal(size=(n, 3))]) * np.array([10.0, 0.1, 3.0, 1.0, 5.0]) + 7.0
    b = np.column_stack([common, own_b, rng.normal(size=(n, 3))]) * np.array([0.2, 30.0, 1.0, 2.0, 0.5]) - 4.0
    y = np.column_stack([common + own_a, own_b, np.zeros(n)]) + 0.1 * rng.normal(size=(n, 3))
    graph = np.repeat(np.arange(6), n // 6)
    return FeatureBlocks(true=y, hidden=np.ones_like(y, bool), graph=graph, pos=None, blocks={"A": a, "B": b})


def test_the_planted_fractions_come_back():
    feats = _planted()
    p = partition_r2(feats, ["g0", "g1", "g2"], folds=spatial_folds(feats.graph, n_folds=3))
    t = p.table
    assert np.isclose(t.loc["g0", "unique_A"], 0.5, atol=0.03) and np.isclose(t.loc["g0", "shared"], 0.5, atol=0.03)
    assert abs(t.loc["g0", "unique_B"]) < 0.02
    assert np.isclose(t.loc["g1", "unique_B"], 0.99, atol=0.02) and abs(t.loc["g1", "shared"]) < 0.02
    assert abs(t.loc["g2", "total"]) < 0.01
    assert np.allclose(t["share_A"] + t["share_B"], t["total"])
    assert np.allclose(t["share_A"], t["unique_A"] + t["shared"] / 2)
    assert np.allclose(t["r2_A+B"], t["total"])


def test_noise_with_many_features_earns_nothing_out_of_fold():
    rng = np.random.default_rng(1)
    n = 300
    feats = FeatureBlocks(
        true=rng.normal(size=(n, 2)),
        hidden=np.ones((n, 2), bool),
        graph=np.repeat(np.arange(3), 100),
        pos=None,
        blocks={"noise": rng.normal(size=(n, 250))},
    )
    p = partition_r2(feats, ["a", "b"], folds=spatial_folds(feats.graph, n_folds=3))
    assert (p.table["total"] < 0.02).all()


def test_a_graph_offset_is_not_counted_within_graphs():
    feats = _planted()
    shifted = FeatureBlocks(
        true=feats.true + 7.0 * feats.graph[:, None],
        hidden=feats.hidden,
        graph=feats.graph,
        pos=None,
        blocks=feats.blocks,
    )
    folds = spatial_folds(feats.graph, n_folds=3)
    a = partition_r2(feats, ["g0", "g1", "g2"], folds=folds).table
    b = partition_r2(shifted, ["g0", "g1", "g2"], folds=folds).table
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-8)


def test_groups_join_blocks_and_only_genes_hidden_everywhere_are_scored():
    feats = _planted()
    feats.blocks["C"] = feats.blocks["B"].copy()
    feats.hidden[:5, 2] = False  # gene 2 visible in a few rows: not scored
    p = partition_r2(
        feats,
        ["g0", "g1", "g2"],
        groups={"local": ["A"], "global": ["B", "C"]},
        folds=spatial_folds(feats.graph, n_folds=3),
    )
    assert list(p.table.index) == ["g0", "g1"]
    assert {"unique_local", "unique_global", "shared", "r2_local+global"} <= set(p.table.columns)
    assert p.settings["n_genes_scored"] == 2


def test_partition_refusals():
    feats = _planted()
    folds = spatial_folds(feats.graph, n_folds=3)
    with pytest.raises(ValueError, match="does not hold|do not hold"):
        partition_r2(feats, ["g0", "g1", "g2"], groups={"x": ["Z"]}, folds=folds)
    with pytest.raises(ValueError, match="takes 1 to"):
        partition_r2(feats, ["g0", "g1", "g2"], groups={str(i): ["A"] for i in range(5)}, folds=folds)
    with pytest.raises(ValueError, match="folds for"):
        partition_r2(feats, ["g0", "g1", "g2"], folds=folds[:-1])
    feats.hidden[0] = False
    with pytest.raises(ValueError, match="hidden in every row"):
        partition_r2(feats, ["g0", "g1", "g2"], folds=folds)


# ------------------------------------------------------------------------------- the collector through models


def _global_setup():
    vk = _load("test_visible_keys_backends")
    module = vk._stir(vk._build(vk._config("dense", visible=False))).eval()
    graphs = [vk._lattice(6, seed=6, hidden=vk.HIDDEN), vk._lattice(5, seed=5, hidden=[0, 1, 2, 7])]
    return module, graphs


def _reach_by_hand(edge_index, n, hops):
    a = np.zeros((n, n), bool)
    a[edge_index[0], edge_index[1]] = True
    a = a | a.T
    within = a.copy()
    for _ in range(hops - 1):
        within = within | ((within.astype(int) @ a.astype(int)) > 0)
    np.fill_diagonal(within, False)
    return within


def test_the_collector_reads_the_steps_rows_through_a_global_model():

    from interscale.evaluation.dimensions import collect_dimensions

    module, graphs = _global_setup()
    state = torch.random.get_rng_state()
    f = collect_feature_blocks(
        module, graphs, batch_size=2, device=torch.device("cpu"), blocks=("own", "near", "global"), hops=2
    )
    assert torch.equal(state, torch.random.get_rng_state())
    r = collect_dimensions(module, graphs, batch_size=2, device=torch.device("cpu"))
    kept = r.hidden.any(axis=1)  # the collector keeps the rows with a hidden entry
    assert np.allclose(f.blocks["global"], r.z[kept]) and np.allclose(f.true, r.true[kept])
    assert np.array_equal(f.hidden, r.hidden[kept])
    assert np.all(f.blocks["own"] == 0.0)  # every kept row is a cell hidden whole
    # near, by hand, for graph 0
    g0 = graphs[0]
    within = _reach_by_hand(g0.edge_index.numpy(), g0.num_nodes, 2)
    vis = ~g0.mask.numpy()
    x = g0.x.double().numpy()
    hidden_cells = np.flatnonzero(g0.mask.numpy())
    rows0 = np.flatnonzero(f.graph == 0)
    assert len(rows0) == len(hidden_cells)
    for row, cell in zip(rows0, hidden_cells, strict=True):
        nb = np.flatnonzero(within[cell] & vis)
        assert np.allclose(f.blocks["near"][row], x[nb].mean(axis=0))
    with pytest.raises(ValueError, match="no local embedding"):
        collect_feature_blocks(module, graphs, batch_size=2, device=torch.device("cpu"), blocks=("local",))


def test_with_folds_every_cell_once_and_the_masks_come_back():
    module, graphs = _global_setup()
    masks = [g.mask.clone() for g in graphs]
    f = collect_feature_blocks(
        module, graphs, batch_size=2, device=torch.device("cpu"), blocks=("own", "near", "global"), folds=3
    )
    assert len(f.true) == sum(g.num_nodes for g in graphs) and f.hidden.all()
    assert all(torch.equal(g.mask, m) for g, m in zip(graphs, masks, strict=True))
    truth = np.concatenate([g.x.double().numpy() for g in graphs])
    assert sorted(map(tuple, np.round(f.true, 5))) == sorted(map(tuple, np.round(truth, 5)))


def test_the_dual_model_hands_both_embeddings():
    gs = _load("test_gene_scales")
    module = gs._module()
    graphs = gs._graphs()
    f = collect_feature_blocks(
        module, graphs, batch_size=2, device=torch.device("cpu"), blocks=("local", "global", "near"), folds=3
    )
    assert f.blocks["local"].shape[0] == 32
    assert f.blocks["global"].shape[0] == 32 and np.isfinite(f.blocks["global"]).all()
    assert np.abs(f.blocks["local"]).max() > 1e-3 and np.abs(f.blocks["global"]).max() > 1e-3
    p = partition_r2(
        f,
        [f"g{k}" for k in range(gs.G)],
        groups={"local": ["local"], "global": ["global"]},
        folds=spatial_folds(f.graph, f.pos, n_folds=2, block=150.0),
    )
    assert {"unique_local", "unique_global", "shared"} <= set(p.table.columns) and len(p.table) == gs.G


def test_own_is_the_visible_input_under_gene_masking():
    vk = _load("test_visible_keys_backends")
    cfg = vk._config("dense", visible=False)
    cfg.dataset.mask_strategy = "gene"
    module = vk._stir(vk._build(cfg)).eval()
    graphs = [vk._lattice(6, seed=6), vk._lattice(5, seed=5)]
    for g in graphs:
        gm = torch.zeros(g.num_nodes, vk.G, dtype=torch.bool)
        gm[:, [1, 4]] = True  # a programme: genes 1 and 4 hidden in every cell
        g.gene_mask = gm
    f = collect_feature_blocks(
        module, graphs, batch_size=2, device=torch.device("cpu"), blocks=("own", "near", "global")
    )
    x = np.concatenate([g.x.double().numpy() for g in graphs])
    assert len(f.true) == len(x)
    want = x.copy()
    want[:, [1, 4]] = 0.0
    assert np.allclose(f.blocks["own"], want)
    assert np.all(f.blocks["near"][:, [1, 4]] == 0.0)  # no cell shows them
    p = partition_r2(
        f,
        [f"g{k}" for k in range(vk.G)],
        groups={"local": ["own", "near"], "global": ["global"]},
        folds=spatial_folds(f.graph, f.pos, n_folds=2, block=250.0),
    )
    assert list(p.table.index) == ["g1", "g4"]


def test_without_graph_centring_a_featureless_fit_predicts_the_training_mean():
    rng = np.random.default_rng(4)
    n = 600
    graph = np.repeat(np.arange(3), 200)
    y = (graph * 3.0)[:, None] + 0.1 * rng.normal(size=(n, 1))
    feats = FeatureBlocks(
        true=y,
        hidden=np.ones_like(y, bool),
        graph=graph,
        pos=None,
        blocks={"const": np.ones((n, 2)), "noise": rng.normal(size=(n, 3)) * 5.0 + 2.0},
    )
    folds = spatial_folds(graph, n_folds=3)
    p = partition_r2(feats, ["g"], folds=folds, within_graph=False).table
    # the held-out graph's offset is unseen: both predict the training mean, far from it
    assert p.loc["g", "r2_const"] < -0.5
    assert abs(p.loc["g", "r2_noise"] - p.loc["g", "r2_const"]) < 0.02


def test_the_local_rows_follow_the_tokens_when_the_model_subsamples():

    from torch_geometric.data import Batch

    gs = _load("test_gene_scales")
    module = gs._module()
    graphs = gs._graphs(sizes=(80, 90))  # more cells than max_seq_len (64): pad_batch subsamples
    for g in graphs:
        g.mask = torch.zeros(g.num_nodes, dtype=torch.bool)
        g.mask[::3] = True
    random.seed(0)
    f = collect_feature_blocks(module, graphs, batch_size=2, device=torch.device("cpu"), blocks=("local", "global"))
    random.seed(0)
    with torch.no_grad():
        out = module._common_step(Batch.from_data_list(graphs), "regression", "node")
    idx = out.view.padded_node_idx
    assert len(idx) < sum(g.num_nodes for g in graphs)
    n = len(idx)
    kept = out.entry_mask[:n].any(dim=1).numpy()
    want = out.view.local_embedding[idx].double().numpy()[kept]
    assert np.allclose(f.blocks["local"], want, atol=1e-6)
