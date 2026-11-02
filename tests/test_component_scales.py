"""V3, the per-gene split of one prediction by its components' scales (yardsticks_plan.md): ``evaluation.component_scales``.

Pinned here:

* **the components** -- scores times loadings give back the prediction centred within each graph; a gene's variance is its
  summed ``S_k² U_gk²``; a per-graph offset changes nothing; the signs are fixed;
* **the split** -- on planted maps (a gradient wider than the reach, a pattern on the scale of a cell) the global component is
  global and the local one local; genes loading on one map get that share, a gene on both splits between them, an unstructured
  gene lands in ``unstructured``/``rest``; the gene's length scale follows its map; private generators;
* **the reach** -- measured on a lattice by hand; refusals.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

from interscale.evaluation.component_scales import (
    GLOBAL,
    LOCAL,
    UNSTRUCTURED,
    component_scales,
    prediction_components,
    reach_from_graph,
)

SIDE, SPACING = 40, 10.0


def _tissue(n_graphs=3, seed=0):
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    pos1 = np.column_stack([ii.ravel(), jj.ravel()]).astype(float) * SPACING
    pos, graph, wide, fine, noise = [], [], [], [], []
    for g in range(n_graphs):
        phase = rng.uniform(0, 2 * np.pi)
        wide.append(np.sin(pos1[:, 0] / 70.0 + phase))  # wavelength ~440 µm
        f = gaussian_filter(rng.normal(size=(SIDE, SIDE)), 0.8).ravel()
        fine.append(f / f.std())
        noise.append(rng.normal(size=len(pos1)))
        pos.append(pos1)
        graph.append(np.full(len(pos1), g))
    wide, fine, noise = (np.concatenate(v) for v in (wide, fine, noise))
    pred = np.column_stack(
        [
            2.0 * wide,
            1.6 * wide,
            -1.8 * wide,  # g0-g2: the wide map
            0.9 * fine,
            0.8 * fine,
            0.7 * fine,  # g3-g5: the fine map
            1.0 * wide + 0.6 * fine,  # g6: both
            0.05 * noise,  # g7: no structure
        ]
    )
    return pred + 3.0, np.concatenate(graph), np.vstack(pos)


GENES = [f"g{k}" for k in range(8)]


def test_the_components_give_back_the_centred_prediction():
    pred, graph, _ = _tissue()
    scores, loadings, s, variance = prediction_components(pred, graph, n_components=50)
    centred = pred.copy()
    for g in np.unique(graph):
        centred[graph == g] -= pred[graph == g].mean(axis=0)
    assert np.allclose(scores @ loadings.T, centred, atol=1e-8)
    assert np.allclose((loadings**2 * s**2).sum(axis=1), variance)
    assert np.all(np.abs(loadings).max(axis=0) == loadings.max(axis=0))  # largest entry positive
    shifted = pred + 10.0 * graph[:, None]
    again = prediction_components(shifted, graph, n_components=50)
    assert np.allclose(again[0] @ again[1].T, scores @ loadings.T, atol=1e-8)


def test_the_planted_maps_split_the_genes():
    pred, graph, pos = _tissue()
    state = np.random.get_state()[1].copy()
    res = component_scales(
        pred, graph, pos, GENES, reach_um=30.0, n_components=3, n_permutations=5, ks=(4, 8, 16, 32, 64, 128, 256)
    )
    assert np.array_equal(np.random.get_state()[1], state)
    comp = res.components
    assert comp["class"].iloc[0] == GLOBAL and comp["class"].iloc[1] == LOCAL
    t = res.genes
    for g in ("g0", "g1", "g2"):
        assert t.loc[g, "share_global"] > 0.97
    for g in ("g3", "g4", "g5"):
        assert t.loc[g, "share_local"] > 0.97
    assert 0.6 < t.loc["g6", "share_global"] < 0.85 and 0.15 < t.loc["g6", "share_local"] < 0.4
    assert t.loc["g7", "share_global"] + t.loc["g7", "share_local"] < 0.2
    assert np.allclose(t[["share_global", "share_local", "share_unstructured", "share_rest"]].sum(axis=1), 1.0)
    assert (
        t.loc["g3", "ell_um"] < 30.0 < t.loc["g0", "ell_um"]
    )  # inf counts: the wide map never halves within the sizes
    assert t.loc["g6", "ell_um"] == t.loc["g0", "ell_um"]  # mostly wide: the median sits on the wide component
    assert res.settings["n_components"] == 3
    narrow = component_scales(
        pred, graph, pos, GENES, reach_um=5.0, n_components=3, n_permutations=5, ks=(4, 8, 16, 32, 64, 128, 256)
    )
    assert narrow.components["class"].iloc[1] == GLOBAL  # the fine map's scale (~16 µm) is wider than a 5 µm reach
    two = component_scales(
        pred, graph, pos, GENES, reach_um=30.0, n_components=2, n_permutations=5, ks=(4, 8, 16, 32, 64, 128, 256)
    )
    assert two.genes.loc["g7", "share_rest"] > 0.9 and two.genes.loc["g0", "share_rest"] < 0.01


def test_reach_by_hand_and_refusals():
    ii, jj = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    pos = np.column_stack([ii.ravel(), jj.ravel()]).astype(float) * SPACING
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    adjacency = sp.csr_matrix((np.abs(d - SPACING) < 1e-9).astype(float))
    graph = np.zeros(36, int)
    assert reach_from_graph(adjacency, graph, pos, hops=2, unit_um=1.0) == pytest.approx(20.0)
    assert reach_from_graph(adjacency, graph, pos, hops=1, unit_um=2.0) == pytest.approx(20.0)
    pred, graph, pos = _tissue(n_graphs=1)
    with pytest.raises(ValueError, match="rows"):
        component_scales(pred[:-1], graph, pos, GENES, reach_um=30.0)
    pred[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        component_scales(pred, graph, pos, GENES, reach_um=30.0)
    assert UNSTRUCTURED == "unstructured"


def test_fold_predictions_cover_every_cell_once_with_the_steps_values():
    import importlib.util
    import random
    from pathlib import Path

    import torch
    from torch_geometric.data import Batch

    from interscale.evaluation.component_scales import collect_fold_predictions

    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    vk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vk)
    module = vk._stir(vk._build(vk._config("dense", visible=False))).eval()
    graphs = [vk._lattice(6, seed=6, hidden=vk.HIDDEN), vk._lattice(5, seed=5, hidden=[0, 1, 2, 7])]
    masks = [g.mask.clone() for g in graphs]
    state = torch.random.get_rng_state()
    f = collect_fold_predictions(module, graphs, batch_size=2, device=torch.device("cpu"), folds=3)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(g.mask, m) for g, m in zip(graphs, masks, strict=True))
    assert len(f.true) == 61 and f.hidden.all() and set(f.heads) == {"global"}
    truth = np.concatenate([g.x.double().numpy() for g in graphs])
    order = np.lexsort(np.column_stack([f.graph, f.pos]).T[::-1])
    want = np.lexsort(
        np.column_stack([np.repeat([0, 1], [36, 25]), np.concatenate([g.pos.numpy() for g in graphs])]).T[::-1]
    )
    assert np.allclose(f.true[order], truth[want])
    # without folds: the graphs' own masks, rows with a hidden entry, the step's prediction
    g = collect_fold_predictions(module, graphs, batch_size=2, device=torch.device("cpu"))
    random.seed(0)
    with torch.no_grad():
        out = module._common_step(Batch.from_data_list(graphs), "regression", "node")
    kept = out.entry_mask.any(dim=1).numpy()
    assert np.allclose(g.heads["global"], out.y_pred.double().numpy()[kept], atol=1e-6)
    assert len(g.true) == len(vk.HIDDEN) + 4
