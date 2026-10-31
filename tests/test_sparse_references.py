"""The model-free references without a distance matrix (``reference_predictions(method="sparse")``; roadmap Tier 3, item 8).

The dense path is pinned by ``test_reference_predictions.py``; here the sparse one must give the same numbers -- slide mean, label
mean, nearest cells exactly, kernels to their truncation -- on random positions (no ties), with cells hidden whole and in part, a gene
hidden everywhere, far-away cells (the dense form's nearest-cell normalisation), two graphs, chunks of any size, and graphs with
one or no open cell; ``auto`` switches by size; no ``N x N`` tensor is formed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interscale.evaluation.masked_probing import reference_predictions

G = 5


def _data(n=(70, 45), seed=0, far=False):
    gen = torch.Generator().manual_seed(seed)
    total = sum(n)
    x = torch.rand(total, G, generator=gen) * 3.0
    pos = torch.rand(total, 2, generator=gen) * 400.0
    if far:
        pos[:5] += 20000.0  # far from every other cell: the kernels underflow without the nearest-cell shift
    graph = torch.cat([torch.full((k,), i) for i, k in enumerate(n)])
    hidden = torch.zeros(total, G, dtype=torch.bool)
    hidden[torch.rand(total, generator=gen) < 0.3] = True  # cells hidden whole
    hidden |= torch.rand(total, G, generator=gen) < 0.1  # entries
    hidden[:, 4] = True  # a gene hidden everywhere
    labels = torch.randint(0, 4, (total,), generator=gen)
    return x, hidden, pos, graph, labels


ARGS = {"spatial_unit_um": 0.5, "kernels_um": (15.0, 60.0, 400.0), "k_nearest": (1, 6, 30)}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("far", [False, True], ids=["spread", "far cells"])
@pytest.mark.parametrize("chunk_pairs", [2**24, 7], ids=["one chunk", "tiny chunks"])
def test_the_sparse_references_are_the_dense_ones(far, chunk_pairs, seed):
    x, hidden, pos, graph, labels = _data(far=far, seed=seed)
    dense = reference_predictions(x, hidden, pos, graph, labels=labels, method="dense", **ARGS)
    sparse = reference_predictions(x, hidden, pos, graph, labels=labels, method="sparse", chunk_pairs=chunk_pairs, **ARGS)
    assert set(dense) == set(sparse)
    for name, value in dense.items():
        assert torch.allclose(sparse[name], value, atol=1e-10), name  # the kernels' cut at exp(-40) moves nothing


def test_a_graph_with_one_or_no_open_cell_and_auto_switches_by_size():
    x, hidden, pos, graph, _ = _data(n=(6, 4))
    hidden[:6] = True
    hidden[0, 1] = False  # graph 0: one open cell
    hidden[6:] = True  # graph 1: none
    dense = reference_predictions(x, hidden, pos, graph, method="dense", **ARGS)
    sparse = reference_predictions(x, hidden, pos, graph, method="sparse", **ARGS)
    for name, value in dense.items():
        assert torch.allclose(sparse[name], value, atol=1e-10), name
    x, hidden, pos, graph, labels = _data()
    calls = []
    original = torch.cdist
    torch.cdist = lambda *a, **k: calls.append(a[0].shape[0]) or original(*a, **k)
    try:
        auto = reference_predictions(x, hidden, pos, graph, max_dense_cells=50, **ARGS)  # graph 0 sparse, graph 1 dense
    finally:
        torch.cdist = original
    assert calls == [45]  # one distance matrix: the 45-cell graph's; the 70-cell one took the neighbour lists
    want = reference_predictions(x, hidden, pos, graph, method="dense", **ARGS)
    for name, value in want.items():
        assert torch.allclose(auto[name], value, atol=1e-10), name
    with pytest.raises(ValueError, match="method must be one of"):
        reference_predictions(x, hidden, pos, graph, method="tree", **ARGS)


def test_the_sparse_path_forms_no_matrix_over_the_cells_pairs(monkeypatch):
    x, hidden, pos, graph, _ = _data(n=(300,))
    calls = []
    original = torch.cdist
    monkeypatch.setattr(torch, "cdist", lambda *a, **k: calls.append(1) or original(*a, **k))
    reference_predictions(x, hidden, pos, graph, method="sparse", chunk_pairs=500, **ARGS)
    assert not calls  # no distance matrix
    reference_predictions(x, hidden, pos, graph, method="dense", **ARGS)
    assert calls
