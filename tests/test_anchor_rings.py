"""The flipped anchored-SVG task (``dataset.anchor_rings``; ``.claude/svg_targets_plan.md``, round 3).

Pinned here:

* **geometry** -- the grid discs are the bridge script's (bounding-box fractions, snapped to the nearest cell, dropped off the
  tissue), disc and rings are disjoint, the ring index has the bridge's (``np.digitize``) boundaries;
* **the draw** -- the gene set is hidden in the ring cells and nowhere else, the disc's cells are the keys (``key_mask``);
* **the keys** -- with ``key_mask`` no token outside it reaches any other cell's prediction (dense and fused), without it it does;
* **the switch** -- the validator, the checkpoint tag, the unit conversion of ``block_mask_kwargs``.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_visible_keys_backends import E, _build, _config, _lattice, _stir
from torch_geometric.data import Batch

from interscale.tl.masking import anchor_grid_discs, anchor_ring_masks, sample_anchor_rings


def _bridge_discs(pos, grid):
    """svg_targets.grid_discs, verbatim in numpy (the bridge runs in another env)."""
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    step = (hi - lo) / grid
    out = []
    for i in range(grid):
        for j in range(grid):
            target = lo + step * np.array([i + 0.5, j + 0.5])
            d = np.hypot(*(pos - target).T)
            k = int(np.argmin(d))
            if d[k] <= 0.5 * float(step.min()):
                out.append((i * grid + j, k))
    return out


def test_grid_discs_and_rings_are_the_bridges():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 1000, (400, 2))
    pos = pos[~((pos[:, 0] > 600) & (pos[:, 1] > 600))]  # a hole: the corner disc must go
    discs = anchor_grid_discs(torch.as_tensor(pos), 3)
    assert discs == _bridge_discs(pos, 3)
    assert 8 not in [d for d, _ in discs]
    r_disc, r_max = 120.0, 300.0
    for _, centre in discs:
        disc, rings, ring = anchor_ring_masks(torch.as_tensor(pos), centre, r_disc, r_max)
        assert not (disc & rings).any() and disc[centre]
        edge = np.hypot(*(pos - pos[centre]).T) - r_disc
        expect = np.where((edge > 0) & (edge <= r_max), np.digitize(edge, np.linspace(0, r_max, 4)[1:-1]), -1)
        assert np.array_equal(ring.numpy(), expect)


def test_the_draw_hides_the_gene_set_in_the_rings_only():
    from interscale.geome_dataloader import GraphAnnDataModule

    data = _lattice(12, seed=1)
    cols = [1, 3]
    dm = GraphAnnDataModule(
        datas=[[data], [data.clone()]], mask_strategy="gene", mask_percentage=1.0, mask_gene_columns=cols,
        anchor_rings={"grid": 3, "r_disc": 150.0, "r_max": 400.0},
    )
    torch.manual_seed(0)
    dm._assign_random_mask(data)
    disc, rings, _ = sample_anchor_rings(data.pos, 3, 150.0, 400.0, disc_id=int(data.anchor_disc))
    assert torch.equal(data.key_mask, disc) and torch.equal(data.mask, rings)
    assert torch.equal(data.gene_mask[:, cols], rings[:, None].expand(-1, 2))
    assert not data.gene_mask[:, [0, 2, 4, 5]].any()
    seen = set()
    for seed in range(30):
        torch.manual_seed(seed)
        dm._assign_random_mask(data)
        seen.add(int(data.anchor_disc))
    assert len(seen) > 4  # discs are drawn, not fixed


@pytest.mark.parametrize("kind", ["dense", "fused"])
def test_only_the_key_mask_cells_reach_other_predictions(kind):
    graphs = [_lattice(6, seed=6), _lattice(5, seed=5)]
    for g in graphs:
        g.mask = torch.zeros(g.num_nodes, dtype=torch.bool)
        g.mask[[0, 1]] = True
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.arange(batch.num_nodes)
    keys = torch.zeros(batch.num_nodes, dtype=torch.bool)
    keys[[14, 15, 20, 21, 40, 41, 42]] = True  # a few cells of each graph
    shifted = torch.zeros(batch.num_nodes, dtype=torch.bool)
    shifted[[2, 3, 30, 31, 50]] = True  # cells that are no keys
    module = _stir(_build(_config(kind, visible=False, max_seq_len=200, center_qk=False))).eval()

    def rows(with_keys):
        b = batch.clone()
        if with_keys:
            b.key_mask = keys.clone()
        shift = torch.zeros(b.num_nodes, E)
        shift[shifted] = 3.0 * torch.randn(int(shifted.sum()), E, generator=torch.Generator().manual_seed(7))
        original = module.gex_embedding
        outs = []
        for s in (0.0, 1.0):
            module.gex_embedding = lambda bb, hidden=None, s=s: original(bb, hidden=hidden) + s * shift
            with torch.no_grad():
                random.seed(0)
                out = module._common_step(b, "regression", "node")
            del module.gex_embedding
            others = ~shifted[out.view.padded_node_idx]
            outs.append(out.y_pred[others])
        return outs

    before, after = rows(with_keys=False)
    assert not torch.allclose(before, after, atol=1e-6)
    before, after = rows(with_keys=True)
    assert torch.allclose(before, after, atol=1e-6)


def test_validator_tag_and_units():
    from interscale.config import _validate
    from interscale.geome_dataloader import block_mask_kwargs
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _config("dense", visible=False)
    cfg.dataset.mask_strategy, cfg.dataset.mask_percentage = "gene", 1.0
    cfg.dataset.mask_genes = ["svg_role=target"]
    cfg.dataset.spatial_unit_um = 0.5
    cfg.dataset.anchor_rings.grid, cfg.dataset.anchor_rings.r_disc_um, cfg.dataset.anchor_rings.r_max_um = 3, 100.0, 300.0
    _validate(cfg)
    assert block_mask_kwargs(cfg) == {"anchor_rings": {"grid": 3, "r_disc": 200.0, "r_max": 600.0}}
    assert "ar3-100-300" in get_model_filename_prefix(cfg, False, True)
    cfg.dataset.mask_percentage = 0.5
    with pytest.raises(ValueError, match="mask_percentage"):
        _validate(cfg)
    cfg.dataset.mask_percentage = 1.0
    cfg.dataset.mask_genes = []
    with pytest.raises(ValueError, match="mask_genes"):
        _validate(cfg)
