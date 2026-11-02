"""R2, sender genes by in-silico deletion (ranking_plan.md): ``evaluation.deletion``.

Pinned here:

* **the conditional** -- one precision matrix gives every gene's (and every block's) conditional mean, equal to the regression on
  the other genes;
* **the deletion** -- on a module of known dependence (target 0 reads sender 3 of a cell 400 µm away, target 1 the neighbours' sender
  4, target 2 the cell's own sender 5): ``context`` finds 3 -> 0 and 4 -> 1 and gives exactly 0 to 5 -> 2 (the receiver's own row is
  kept), ``all`` finds 5 -> 2 as well; an unread gene gets exactly 0; planted senders rank first;
* **conditional against permute** -- with a near copy of sender 3 in the same cell, the conditional deletion of 3 hardly moves target 0
  (no unique information), the permutation does; deleting the pair as a group does;
* the graphs and every global generator are left alone; refusals; a real toy model runs end to end.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.deletion import (
    conditional_mean,
    conditional_parts,
    planted_recovery,
    sender_deletion,
)
from interscale.module.base._step_output import StepOutput, ViewOutput

SIDE, SPACING, G = 16, 100.0, 8
GENES = [f"g{k}" for k in range(G)]
HIDDEN = [0, 1, 2]  # the programme: hidden in every cell


def test_the_conditional_mean_is_the_regression_on_the_other_genes():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5))
    x = rng.normal(size=(4000, 5)) @ a + 3.0
    mean, precision = conditional_parts(x, ridge=1e-10)
    for j in range(5):
        others = [k for k in range(5) if k != j]
        design = np.column_stack([np.ones(len(x)), x[:, others]])
        coef, *_ = np.linalg.lstsq(design, x[:, j], rcond=None)
        assert np.allclose(conditional_mean(x, [j], mean, precision)[:, 0], design @ coef, atol=1e-6)
    block = [1, 3]
    design = np.column_stack([np.ones(len(x)), x[:, [0, 2, 4]]])
    coef, *_ = np.linalg.lstsq(design, x[:, block], rcond=None)
    assert np.allclose(conditional_mean(x, block, mean, precision), design @ coef, atol=1e-6)


def _pos():
    ii, jj = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    return np.column_stack([ii.ravel(), jj.ravel()]).astype(float) * SPACING


def _partner(pos):
    period = SIDE * SPACING
    target = np.column_stack([np.mod(pos[:, 0] + 400.0, period), pos[:, 1]])
    return np.argmin(np.linalg.norm(target[:, None] - pos[None], axis=-1), axis=1)


def _near(pos):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return (d <= 150.0) & (d > 0)


class _Known(torch.nn.Module):
    """Target 0: sender 3 of the partner 400 µm along x, plus noise from the cell's own gene 7; target 1: the mean of sender 4 over
    the cells within 150 µm; target 2: the cell's own sender 5. Every other prediction 0. Reads visible genes only (3, 4, 5, 7)."""

    registered_global_component = True

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def _common_step(self, batch, task, level):
        x, pos = batch.x.double(), batch.pos.double().numpy()
        pred = torch.zeros_like(x)
        for g in range(int(batch.num_graphs)):
            rows = torch.nonzero(batch.batch == g).flatten()
            p = pos[rows.numpy()]
            xv = x[rows]
            partner = torch.as_tensor(_partner(p))
            near = torch.as_tensor(_near(p), dtype=torch.float64)
            pred[rows, 0] = xv[partner, 3] + 0.5 * (
                xv[:, 7] - 0.5
            )  # plus noise from the cell's own gene 7 (unrelated to the truth)
            pred[rows, 1] = (near @ xv[:, 4]) / near.sum(1)
            pred[rows, 2] = xv[:, 5]
        return StepOutput(y_pred=pred.float(), y_true=batch.x, entry_mask=batch.gene_mask.bool(), views=[ViewOutput()])


def _graphs(n_graphs=3, seed=0, copy=False):
    rng = np.random.default_rng(seed)
    pos = _pos()
    partner, near = _partner(pos), _near(pos)
    out = []
    for _ in range(n_graphs):
        x = rng.random((len(pos), G))
        if copy:
            x[:, 6] = x[:, 3] + 0.02 * rng.normal(size=len(pos))  # a near copy of sender 3 in the same cell
        x[:, 0] = x[partner, 3] + 0.02 * rng.normal(size=len(pos))
        x[:, 1] = (near @ x[:, 4]) / near.sum(1) + 0.02 * rng.normal(size=len(pos))
        x[:, 2] = x[:, 5] + 0.02 * rng.normal(size=len(pos))
        d = Data(x=torch.tensor(x, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32))
        d.edge_index = torch.zeros((2, 0), dtype=torch.long)
        gm = torch.zeros(len(pos), G, dtype=torch.bool)
        gm[:, HIDDEN] = True
        d.gene_mask = gm
        out.append(d)
    return out


SENDERS = ["g3", "g4", "g5", "g6", "g7"]


def test_context_finds_the_context_senders_and_all_finds_the_own_one_too():
    module, graphs = _Known(), _graphs()
    before = [g.x.clone() for g in graphs]
    torch_state, np_state = torch.random.get_rng_state(), np.random.get_state()[1].copy()
    kw = dict(senders=SENDERS, mask_strategy="gene", n_draws=2, max_cover=0.2, seed=1)
    ctx = sender_deletion(module, graphs, GENES, region="context", **kw)
    assert torch.equal(torch_state, torch.random.get_rng_state()) and np.array_equal(np_state, np.random.get_state()[1])
    assert all(torch.equal(g.x, b) for g, b in zip(graphs, before, strict=True))
    d = ctx.delta_r2
    assert list(d.columns) == ["g0", "g1", "g2"]  # the hidden genes are the targets
    assert d.loc["g3", "g0"] > 0.5 and d.loc["g4", "g1"] > 0.3
    assert d.loc["g5", "g2"] == pytest.approx(0.0, abs=1e-12)  # the receiver's own row is kept
    assert (d.loc[["g6", "g7"]].abs() < 1e-12).all().all()  # genes the model never reads
    assert ctx.z.loc["g3", "g0"] > 3 and ctx.senders.index[0] in ("g3", "g4")
    assert set(ctx.senders.loc[ctx.senders["n_targets"] > 0].index) == {"g3", "g4"}
    rec = planted_recovery(ctx, [("g3", "g0"), ("g4", "g1")])
    assert rec["first"].all() and np.allclose(rec["reciprocal_rank"], 1.0)
    every = sender_deletion(module, graphs, GENES, region="all", **kw)
    assert every.delta_r2.loc["g5", "g2"] > 0.5 and every.settings["n_receivers"] == 3 * SIDE * SIDE
    assert ctx.settings["passes"] == 1 + 5 * 2


def test_conditional_deletion_reads_unique_information_and_groups_restore_it():
    module, graphs = _Known(), _graphs(copy=True)
    kw = dict(mask_strategy="gene", n_draws=2, region="all", seed=2)
    cond = sender_deletion(module, graphs, GENES, senders=["g3"], method="conditional", **kw)
    perm = sender_deletion(module, graphs, GENES, senders=["g3"], method="permute", **kw)
    pair = sender_deletion(module, graphs, GENES, senders={"g3+g6": ["g3", "g6"]}, method="conditional", **kw)
    assert perm.delta_r2.loc["g3", "g0"] > 0.5
    assert cond.delta_r2.loc["g3", "g0"] < 0.1 * perm.delta_r2.loc["g3", "g0"]
    assert pair.delta_r2.loc["g3+g6", "g0"] > 0.5


def test_mean_replacement_and_refusals():
    module, graphs = _Known(), _graphs(n_graphs=1)
    m = sender_deletion(
        module, graphs, GENES, senders=["g4"], mask_strategy="gene", method="mean", region="all", n_draws=1
    )
    assert m.delta_r2.loc["g4", "g1"] > 0.3
    with pytest.raises(ValueError, match="hidden in graph"):
        sender_deletion(module, graphs, GENES, senders=["g0"], mask_strategy="gene")
    with pytest.raises(ValueError, match="not among the genes"):
        sender_deletion(module, graphs, GENES, senders=["nope"], mask_strategy="gene")
    with pytest.raises(ValueError, match="region"):
        sender_deletion(module, graphs, GENES, senders=["g3"], mask_strategy="gene", region="middle")
    with pytest.raises(ValueError, match="method"):
        sender_deletion(module, graphs, GENES, senders=["g3"], mask_strategy="gene", method="zero")
    graphs[0].embeddings = torch.zeros(len(graphs[0].x), 4)
    with pytest.raises(ValueError, match="precomputed embeddings"):
        sender_deletion(module, graphs, GENES, senders=["g3"], mask_strategy="gene")


def test_far_keeps_the_zone_and_the_twin_column_beside():
    module, graphs = _Known(), _graphs()
    far = sender_deletion(
        module, graphs, GENES, senders=["g3", "g4"], mask_strategy="gene", region="far", radius=250.0, max_cover=0.2
    )
    assert far.delta_r2.loc["g4", "g1"] == pytest.approx(0.0, abs=1e-12)  # the neighbours lie within 250 µm: kept
    assert far.delta_r2.loc["g3", "g0"] > 0.5  # the partner lies 400 µm away: replaced
    twin = sender_deletion(
        module, graphs, GENES, senders=["g3", "g4"], mask_strategy="gene", region="far", radius=500.0, max_cover=0.5
    )
    rec = planted_recovery(far, [("g3", "g0")], null=twin)
    assert rec.loc["g0", "rank"] == 1 and "null_rank" in rec.columns


def test_a_real_toy_global_model_runs_end_to_end():
    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    vk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vk)
    module = vk._stir(vk._build(vk._config("dense", visible=False))).eval()
    graphs = [vk._lattice(6, seed=6, hidden=[0, 14, 35]), vk._lattice(5, seed=5, hidden=[0, 24])]
    genes = [f"g{k}" for k in range(vk.G)]
    res = sender_deletion(module, graphs, genes, senders=genes, mask_strategy="node", region="all", n_draws=1)
    assert res.delta_r2.shape == (vk.G, vk.G) and np.isfinite(res.delta_r2.to_numpy()).all()
    assert res.settings["n_receivers"] == 5


def test_exact_sums_draws_and_negative_effects():
    from interscale.evaluation.deletion import _graph_model, _replaced_values

    module, graphs = _Known(), _graphs()
    kw = dict(senders=SENDERS, mask_strategy="gene", region="all", method="mean")
    one = sender_deletion(module, graphs, GENES, n_draws=1, **kw)
    three = sender_deletion(module, graphs, GENES, n_draws=3, **kw)
    assert np.allclose(one.delta_r2.to_numpy(), three.delta_r2.to_numpy())  # the mean replacement draws nothing
    # replacing gene 7 by its mean removes the noise the model adds: a negative effect, left out of the importance
    assert one.delta_r2.loc["g7", "g0"] < -0.05
    assert np.allclose(one.senders["importance"], one.delta_r2.clip(lower=0).sum(axis=1).reindex(one.senders.index))
    assert one.senders.loc["g7", "importance"] == 0.0
    loose = sender_deletion(module, graphs, GENES, n_draws=1, min_effect=-1.0, **kw)
    zeros = (loose.delta_r2.loc["g6"].abs() < 1e-12).sum()
    assert zeros == 3 and loose.senders.loc["g6", "n_targets"] == 0  # no effect, z 0: never counted
    # the context region ignores the radius: the receiver alone is kept
    ctx = sender_deletion(
        module, graphs, GENES, senders=["g4"], mask_strategy="gene", region="context", radius=250.0, max_cover=0.2
    )
    assert ctx.delta_r2.loc["g4", "g1"] > 0.3
    # a conditional draw keeps the gene's spread (the conditional mean alone would shrink it)
    x = graphs[0].x.double().numpy()
    vis = ~graphs[0].gene_mask.numpy()
    model = _graph_model(x, vis, method="conditional", ridge=1e-2)
    values = _replaced_values(x, model, [3], model.rows, method="conditional", rng=np.random.default_rng(0))
    assert values.std() == pytest.approx(x[:, 3].std(), rel=0.15)
