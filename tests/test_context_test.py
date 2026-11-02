"""V2, the per-gene context test (yardsticks_plan.md): ``evaluation.context_test``.

Pinned here:

* **the shift** -- on a full lattice a bijection, a move of at least ``min_shift`` of the period, reproducible, private RNG;
* **the receivers** -- candidates only, more than ``2 r`` apart within a batch, zones covering at most ``max_cover``, never in
  two batches;
* **the surrogate** -- ``far`` keeps the zones and replaces the rest, ``near`` replaces the zones but not the receivers; whole
  rows move (``x`` and masks), positions and edges stay; the input graphs are not changed; an expression-derived attribute the
  caller did not place is refused;
* **the test** -- on a module with known dependence (gene 0 read 400 µm away, gene 1 from neighbours within 150 µm, gene 2
  constant) the far test flags gene 0 only and the near test gene 1 only; a constant prediction gets p = 1; labels;
  Benjamini-Hochberg by hand; a real toy global model runs end to end and leaves every generator alone.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.context_test import (
    _bh,
    apply_sources,
    choose_receivers,
    context_labels,
    context_test,
    shift_sources,
    surrogate_sources,
    zones,
)
from interscale.module.base._step_output import StepOutput, ViewOutput

SIDE, SPACING = 20, 100.0


def _lattice_pos(side=SIDE):
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float) * SPACING


# ------------------------------------------------------------------------------- the shift and the receivers


def test_the_shift_is_a_bijection_on_a_lattice_and_moves_far_enough():
    pos = _lattice_pos()
    state = np.random.get_state()[1].copy()
    src = shift_sources(pos, rng=np.random.default_rng(0), min_shift=0.25)
    assert np.array_equal(np.random.get_state()[1], state)
    assert sorted(src.tolist()) == list(range(len(pos)))
    period = SIDE * SPACING
    move = np.mod(pos[src] - pos, period)
    assert np.all((move >= 0.25 * period - 1e-9) & (move <= 0.75 * period + 1e-9))
    assert len(np.unique(move, axis=0)) == 1  # one translation for the whole graph
    assert np.array_equal(src, shift_sources(pos, rng=np.random.default_rng(0)))
    with pytest.raises(ValueError, match="min_shift"):
        shift_sources(pos, rng=np.random.default_rng(0), min_shift=0.5)


def test_receivers_are_spaced_cover_little_and_never_repeat():
    pos = _lattice_pos()
    rng = np.random.default_rng(1)
    candidates = rng.random(len(pos)) < 0.3
    batches = choose_receivers(pos, candidates, 250.0, rng=rng, max_cover=0.15, n_batches=3)
    assert len(batches) == 3
    single = choose_receivers(pos, candidates, 250.0, rng=np.random.default_rng(1), n_batches=4, per_batch=1)
    assert [len(b) for b in single] == [1, 1, 1, 1]
    seen = np.concatenate(batches)
    assert len(seen) == len(set(seen.tolist())) and candidates[seen].all()
    for b in batches:
        d = np.linalg.norm(pos[b][:, None] - pos[b][None], axis=-1)
        assert np.all(d[np.triu_indices(len(b), 1)] > 500.0)
        covered = np.unique(np.concatenate(zones(pos, b, 250.0)))
        assert len(covered) <= 0.15 * len(pos)
    assert choose_receivers(pos, np.zeros(len(pos), bool), 250.0, rng=rng) == []


def test_far_keeps_the_zones_and_near_replaces_them_but_not_the_receivers():
    pos = _lattice_pos()
    receivers = np.array([0, 210])
    zone = np.zeros(len(pos), bool)
    for z in zones(pos, receivers, 250.0):
        zone[z] = True
    far = surrogate_sources(pos, receivers, 250.0, "far", rng=np.random.default_rng(2))
    assert np.array_equal(far[zone], np.flatnonzero(zone)) and np.all(far[~zone] != np.flatnonzero(~zone))
    near = surrogate_sources(pos, receivers, 250.0, "near", rng=np.random.default_rng(2))
    ring = zone.copy()
    ring[receivers] = False
    assert np.array_equal(near[receivers], receivers)
    assert np.array_equal(near[~zone], np.flatnonzero(~zone)) and np.all(near[ring] != np.flatnonzero(ring))
    with pytest.raises(ValueError, match="region"):
        surrogate_sources(pos, receivers, 250.0, "middle", rng=np.random.default_rng(2))


def test_whole_rows_move_and_the_input_stays():
    pos = _lattice_pos(4)
    data = Data(x=torch.arange(32.0).view(16, 2), pos=torch.tensor(pos), edge_index=torch.tensor([[0, 1], [1, 0]]))
    data.mask = torch.arange(16) % 3 == 0
    data.obs_names = torch.arange(16.0)
    source = np.roll(np.arange(16), 5)
    out = apply_sources(data, source, ("x", "mask", "gene_mask"))
    assert torch.equal(out.x, data.x[source]) and torch.equal(out.mask, data.mask[source])
    assert torch.equal(out.pos, data.pos) and torch.equal(out.obs_names, data.obs_names)
    assert torch.equal(out.edge_index, data.edge_index)
    assert torch.equal(data.x, torch.arange(32.0).view(16, 2))


# ------------------------------------------------------------------------------- the test on a module of known dependence


class _Known(torch.nn.Module):
    """Gene 0 at a cell: gene 3 of the cell 400 µm along x (wrapping); gene 1: the mean of gene 4 over the visible cells
    within 150 µm; gene 2: 0.5. Hidden cells (``mask``) show nothing: their values are never read."""

    registered_global_component = True

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def _common_step(self, batch, task, level):
        x, pos, mask = batch.x.double(), batch.pos.double(), batch.mask.bool()
        pred = torch.full_like(x, 0.5)
        for g in range(int(batch.num_graphs)):
            rows = torch.nonzero(batch.batch == g).flatten()
            p, xv, vis = pos[rows], x[rows], ~mask[rows]
            period = SIDE * SPACING
            target = torch.stack([torch.remainder(p[:, 0] + 400.0, period), p[:, 1]], dim=1)
            partner = torch.cdist(target, p).argmin(dim=1)
            pred[rows, 0] = torch.where(vis[partner], xv[partner, 3], torch.tensor(0.5, dtype=x.dtype))
            d = torch.cdist(p, p)
            near = (d <= 150.0) & (d > 0) & vis[None, :]
            count = near.sum(dim=1)
            pred[rows, 1] = torch.where(
                count > 0, (near.double() @ xv[:, 4]) / count.clamp_min(1), torch.tensor(0.5, dtype=x.dtype)
            )
        entry = mask[:, None].expand_as(x)
        return StepOutput(y_pred=pred.float(), y_true=batch.x, entry_mask=entry, views=[ViewOutput()])


def _known_graphs(n_graphs=2, seed=0, rate=0.15):
    rng = np.random.default_rng(seed)
    pos = _lattice_pos()
    period = SIDE * SPACING
    graphs = []
    for _ in range(n_graphs):
        x = rng.random((len(pos), 5))
        target = np.column_stack([np.mod(pos[:, 0] + 400.0, period), pos[:, 1]])
        partner = np.argmin(np.linalg.norm(target[:, None] - pos[None], axis=-1), axis=1)
        x[:, 0] = x[partner, 3] + 0.05 * rng.normal(size=len(pos))
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        near = (d <= 150.0) & (d > 0)
        x[:, 1] = (near @ x[:, 4]) / near.sum(1) + 0.05 * rng.normal(size=len(pos))
        data = Data(x=torch.tensor(x, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32))
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.mask = torch.tensor(rng.random(len(pos)) < rate)
        graphs.append(data)
    return graphs


def test_far_flags_the_far_gene_and_near_the_near_gene():
    module, graphs = _Known(), _known_graphs(rate=0.1)
    before = [g.x.clone() for g in graphs]
    kwargs = dict(radius=250.0, mask_strategy="node", n_shifts=19, max_cover=0.2, seed=3, batch_size=2)
    far = context_test(module, graphs, ["g0", "g1", "g2", "g3", "g4"], region="far", n_batches=2, **kwargs)
    near = context_test(module, graphs, ["g0", "g1", "g2", "g3", "g4"], region="near", n_batches=25, **kwargs)
    assert all(torch.equal(g.x, b) for g, b in zip(graphs, before, strict=True))
    f, n = far.table, near.table
    assert f.loc["g0", "p_rank"] == pytest.approx(1 / 20) and f.loc["g0", "p"] < 1e-3 and f.loc["g0", "delta_r2"] > 0.5
    assert (
        f.loc["g1", "delta"] == pytest.approx(0.0, abs=1e-12)
        and f.loc["g1", "p"] == 1.0
        and f.loc["g1", "p_rank"] == 1.0
    )  # a tie
    assert n.loc["g1", "p_rank"] == pytest.approx(1 / 20) and n.loc["g1", "p"] < 1e-3 and n.loc["g1", "delta_r2"] > 0.5
    assert (
        n.loc["g0", "delta"] == pytest.approx(0.0, abs=1e-12) and n.loc["g0", "p"] == 1.0
    )  # the partner beyond r: untouched
    assert f.loc["g2", "p"] == 1.0 and n.loc["g2", "p"] == 1.0  # a constant prediction
    labels = context_labels(far, near, q=0.05)
    assert (
        labels.loc["g0", "label"] == "global"
        and labels.loc["g1", "label"] == "local"
        and labels.loc["g2", "label"] == "none"
    )
    assert far.losses.shape == (19, 5) and far.settings["passes"] == 1 + 2 * 19
    assert (
        near.settings["per_batch"] == 1
        and near.settings["n_receivers"] == sum(min(25, int(g.mask.sum())) for g in graphs)
        and near.settings["passes"] == 1 + 25 * 19
    )


def test_several_near_receivers_per_pass_leak_far_context():
    """Why ``near`` takes one receiver per pass: receiver j's replaced zone holds receiver i's far partner."""
    module = _Known()
    graphs = _known_graphs(n_graphs=1, rate=0.0)
    pos = graphs[0].pos.numpy()
    i = int(np.flatnonzero((pos[:, 0] == 500.0) & (pos[:, 1] == 1000.0))[0])
    j = int(
        np.flatnonzero((pos[:, 0] == 1100.0) & (pos[:, 1] == 1000.0))[0]
    )  # 600 µm from i; i's partner (900, 1000) 200 µm from j
    graphs[0].mask[[i, j]] = True
    kwargs = dict(radius=250.0, region="near", mask_strategy="node", n_shifts=9, n_batches=2, max_cover=1.0, seed=3)
    many = context_test(module, graphs, ["g0", "g1", "g2", "g3", "g4"], per_batch=None, **kwargs)
    assert many.settings["n_receivers"] == 2 and many.table.loc["g0", "delta"] > 0
    one = context_test(module, graphs, ["g0", "g1", "g2", "g3", "g4"], **kwargs)
    assert one.table.loc["g0", "delta"] == pytest.approx(0.0, abs=1e-12) and one.table.loc["g0", "p"] == 1.0


def test_bh_by_hand():
    p = np.array([0.01, 0.04, 0.03, 0.5, np.nan])
    q = _bh(p)
    assert np.allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.5]) and np.isnan(q[4])


def test_refusals():
    module, graphs = _Known(), _known_graphs(n_graphs=1)
    graphs[0].celltype = torch.zeros(len(graphs[0].x), 3)
    with pytest.raises(ValueError, match="neither moved"):
        context_test(module, graphs, list("abcde"), radius=250.0, mask_strategy="node", n_shifts=2)
    out = context_test(
        module, graphs, list("abcde"), radius=250.0, mask_strategy="node", n_shifts=2, fields=("x", "mask", "celltype")
    )
    assert len(out.table) == 5
    graphs[0].mask[:] = False
    with pytest.raises(ValueError, match="no receiver"):
        context_test(
            module,
            graphs,
            list("abcde"),
            radius=250.0,
            mask_strategy="node",
            n_shifts=2,
            fields=("x", "mask", "celltype"),
        )


def test_a_real_toy_global_model_runs_end_to_end():
    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    vk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vk)
    module = vk._stir(vk._build(vk._config("dense", visible=False))).eval()
    graphs = [vk._lattice(6, seed=6, hidden=[0, 14, 35]), vk._lattice(5, seed=5, hidden=[0, 24])]
    torch_state, np_state = torch.random.get_rng_state(), np.random.get_state()[1].copy()
    res = context_test(
        module, graphs, [f"g{k}" for k in range(vk.G)], radius=150.0, mask_strategy="node", n_shifts=4, max_cover=0.5
    )
    assert torch.equal(torch_state, torch.random.get_rng_state()) and np.array_equal(np_state, np.random.get_state()[1])
    assert len(res.table) == vk.G and res.table["p"].between(0, 1).all() and np.isfinite(res.table["loss_true"]).all()
    assert res.settings["n_receivers"] >= 2


def test_the_normaliser_is_the_graphs_variance_even_with_one_receiver_per_graph():
    module, graphs = _Known(), _known_graphs(rate=0.1)
    near = context_test(
        module,
        graphs,
        list("abcde"),
        radius=250.0,
        region="near",
        mask_strategy="node",
        n_shifts=3,
        n_batches=1,
        seed=0,
    )
    assert near.settings["n_receivers"] == 2
    want = np.mean([g.x.double().numpy().var(axis=0, ddof=1) for g in graphs], axis=0)  # one entry per graph per gene
    assert np.allclose(near.table["variance"].to_numpy(), want)
    assert np.isfinite(near.table["delta_r2"]).all()
