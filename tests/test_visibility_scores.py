"""Hidden against visible cells: how much of a reconstruction score is the cell's own input (masking_plan.md, M0).

``masked_probing.visibility_scores``, ``visible_view``, ``HeadPredictions.near_hidden_um`` and the ``"cells+blocks"`` mask.
Pinned here:

* **the mask** -- cells at a rate and discs together, as cell masking with discs trains, from a private generator; a rate
  of 0 leaves the discs alone; the refusals;
* **the distance** -- each visible cell's distance to the nearest hidden cell of its graph, row for row (also on graphs
  longer than the sequence), 0 for the hidden cells;
* **the view and the scores** -- the view scores every entry of the visible cells and nothing else; an identity head scores
  1 on visible cells and its fill on hidden ones, the training mean 0 on both, a head that only knows the context ties the
  leave-one-out reference; the per-gene, gap and distance tables; no visible cell under programme masking;
* **through a real model** -- the references at a visible cell never read the cell itself.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.masked_probing import (
    HIDDEN_CELLS,
    VISIBLE_CELLS,
    HeadPredictions,
    collect_head_predictions,
    draw_masks,
    visibility_scores,
    visible_view,
)

G = 6


def _graph(side, seed=0):
    """A lattice graph with 4-neighbour edges, positions in units of 10 µm."""
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    d = torch.cdist(pos, pos)
    edges = ((d > 0) & (d <= 1.0)).nonzero().t()
    gen = torch.Generator().manual_seed(seed)
    data = Data(x=torch.rand(side * side, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.obs_names = torch.arange(side * side)
    return data


def _module(max_seq_len=256):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    cfg.model.decoder.dual_decoder = True
    torch.manual_seed(0)
    return DualDecoderCombinedModule(
        cfg=cfg, n_input=G, n_output=G, n_embed=16, decoder_type=None, dropout_decoder=0.1,
        decoder_hidden_dims=[16], mask_percentage=0.3, mask_strategy="node",
    )


def _collect(module, datas, references=None):
    return collect_head_predictions(
        module, datas, prediction_task="regression", prediction_level="node", mask_strategy="gene", batch_size=2,
        device=torch.device("cpu"), spatial_unit_um=10.0, references=references,
    )


# --------------------------------------------------------------------------- the mask


def test_cells_and_blocks_hide_random_cells_and_discs_together():
    datas = [_graph(16, 0), _graph(12, 1)]
    state = torch.get_rng_state()
    draw_masks(datas, "cells+blocks", 0.2, seed=4, radius=2.0, block_rate=0.2)
    assert torch.equal(state, torch.get_rng_state())
    for d in datas:
        assert (d.gene_mask.all(1) | ~d.gene_mask.any(1)).all() and torch.equal(d.mask, d.gene_mask.any(1))
        dist = torch.cdist(d.pos, d.pos)
        whole_discs = torch.stack([dist[c] <= 2.0 for c in range(d.num_nodes) if bool(d.mask[dist[c] <= 2.0].all())])
        in_discs = whole_discs.any(0)
        assert float(in_discs.float().mean()) >= 0.2  # the discs cover their share
        outside = d.mask & ~in_discs
        assert int(outside.sum()) > 0  # and random cells besides them
    discs_only = [_graph(16, 0)]
    draw_masks(discs_only, "cells+blocks", 0.0, seed=4, radius=2.0, block_rate=0.2)
    alone = [_graph(16, 0)]
    gen = torch.Generator().manual_seed(4)
    torch.rand(256, generator=gen)  # the cell draw, at rate 0, still takes its numbers first
    from interscale.tl.masking import sample_block_mask

    assert torch.equal(discs_only[0].mask, sample_block_mask(alone[0].pos, 2.0, 0.2, generator=gen))
    again = [_graph(16, 0), _graph(12, 1)]
    draw_masks(again, "cells+blocks", 0.2, seed=4, radius=2.0, block_rate=0.2)
    assert all(torch.equal(a.gene_mask, b.gene_mask) for a, b in zip(again, datas, strict=True))


@pytest.mark.parametrize(("rate", "radius", "block_rate"), [(1.0, 2.0, 0.2), (0.2, None, 0.2), (0.2, 2.0, None), (0.2, 2.0, 1.0)])
def test_cells_and_blocks_refuse_what_cannot_be_drawn(rate, radius, block_rate):
    with pytest.raises(ValueError, match="cells\\+blocks"):
        draw_masks([_graph(5)], "cells+blocks", rate, seed=0, radius=radius, block_rate=block_rate)


# --------------------------------------------------------------------------- the distance


def test_each_visible_cell_knows_its_distance_to_the_nearest_hidden_cell():
    import random

    datas = [_graph(10, 0), _graph(7, 1)]
    draw_masks(datas, "cells+blocks", 0.1, seed=0, radius=2.0, block_rate=0.2)
    random.seed(0)
    pred = _collect(_module(max_seq_len=40), datas)  # both graphs subsampled to 40 tokens
    for graph, d in enumerate(datas):
        hid = d.mask
        near = torch.zeros(d.num_nodes, dtype=torch.float64)
        near[~hid] = torch.cdist(d.pos, d.pos)[~hid][:, hid].min(dim=1).values.double() * 10.0
        rows = slice(40 * graph, 40 * graph + 40)
        node = [int(torch.where((d.x == torch.tensor(t, dtype=d.x.dtype)).all(1))[0]) for t in pred.true[rows]]
        assert np.allclose(pred.near_hidden_um[rows], near[node].numpy())
    assert (pred.near_hidden_um[pred.hidden.any(1)] == 0).all()
    assert (pred.near_hidden_um[~pred.hidden.any(1)] >= 10.0).all()


# --------------------------------------------------------------------------- the view and the scores


def _planted(n=300, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, G)) + 3.0
    hidden_cells = rng.random(n) < 0.3
    center = np.full(G, 3.0)
    context = true + rng.normal(scale=1.5, size=true.shape)  # a context smoother: right in part, about the same everywhere
    heads = {
        "identity": np.where(hidden_cells[:, None], 0.0, true),  # copies its input, fills hidden cells with a constant
        "mean": np.tile(center, (n, 1)) + 1e-3 * rng.normal(size=true.shape),  # the mean (a zero vector has no cosine)
        "context": context,
        "ref_context": context.copy(),
    }
    pred = HeadPredictions(
        heads=heads,
        true=true,
        hidden=np.repeat(hidden_cells[:, None], G, axis=1),
        depth_um=np.where(hidden_cells, rng.uniform(10, 50, n), 0.0),
        near_hidden_um=np.where(hidden_cells, 0.0, rng.uniform(10, 50, n)),
    )
    return pred, center, hidden_cells


def test_the_view_scores_every_entry_of_the_visible_cells_and_nothing_else():
    pred, _, hidden_cells = _planted()
    view = visible_view(pred)
    assert np.array_equal(view.hidden, np.repeat(~hidden_cells[:, None], G, axis=1))
    assert view.heads is pred.heads and view.true is pred.true
    assert np.array_equal(view.depth_um, np.where(hidden_cells, 0.0, pred.near_hidden_um))


def test_an_identity_head_scores_one_on_visible_cells_and_the_mean_scores_zero_on_both():
    pred, center, _ = _planted()
    tables = visibility_scores(pred, [f"g{k}" for k in range(G)], center=center, near_bins_um=[0, 25, np.inf])
    groups = tables["groups"].set_index("group")
    assert groups.loc[VISIBLE_CELLS, "identity_cosine_centred"] == pytest.approx(1.0)
    assert abs(groups.loc[HIDDEN_CELLS, "identity_cosine_centred"]) < 0.1  # a constant fill knows nothing of the cell
    assert groups.loc[[HIDDEN_CELLS, VISIBLE_CELLS], "mean_cosine_centred"].abs().max() < 0.1
    # a head that only knows the context reads both groups alike, and ties a reference made the same way
    assert abs(groups.loc[VISIBLE_CELLS, "context_cosine_centred"] - groups.loc[HIDDEN_CELLS, "context_cosine_centred"]) < 0.1
    assert groups.loc[VISIBLE_CELLS, "context_cosine_centred"] == groups.loc[VISIBLE_CELLS, "ref_context_cosine_centred"]
    assert groups["n_cells"].sum() == len(pred.true)
    gap = tables["gap"].set_index("head")
    assert gap.loc["identity", "cosine_centred_visible - hidden"] > 0.9
    per_gene = tables["per_gene"].set_index("gene")
    assert np.allclose(per_gene["identity_visible"], 1.0)
    assert np.allclose(per_gene["identity_visible - hidden"], per_gene["identity_visible"] - per_gene["identity_hidden"])
    assert (per_gene["n_hidden_cells"] + per_gene["n_visible_cells"] == len(pred.true)).all()
    by_distance = tables["visible_by_distance"]
    assert by_distance["n_cells"].sum() == int((~pred.hidden.any(1)).sum())
    assert list(by_distance["distance_to_hidden_um"]) == ["(0.0, 25.0]", "(25.0, inf]"]
    assert "hidden_by_depth" not in tables  # no depth bins asked for


def test_under_programme_masking_there_is_no_visible_cell():
    pred, center, _ = _planted()
    programme = HeadPredictions(
        heads=pred.heads, true=pred.true, hidden=np.zeros_like(pred.hidden), near_hidden_um=pred.near_hidden_um
    )
    programme.hidden[:, 2] = True  # one gene hidden in every cell
    tables = visibility_scores(programme, [f"g{k}" for k in range(G)], center=center)
    assert list(tables) == ["groups"]
    groups = tables["groups"].set_index("group")
    assert groups.loc[VISIBLE_CELLS, "n_cells"] == 0 and groups.loc[HIDDEN_CELLS, "n_cells"] == len(pred.true)


# --------------------------------------------------------------------------- through a real model


def test_the_references_at_a_visible_cell_never_read_the_cell_itself():
    datas = [_graph(9, 0)]
    draw_masks(datas, "cells", 0.3, seed=1)
    pred = _collect(_module(), datas, references={"kernels_um": (15.0,), "k_nearest": (3,)})
    d = datas[0]
    visible = ~d.mask
    x, pos = d.x.double(), d.pos.double()
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))
    dist[:, d.mask] = float("inf")  # only visible cells are averaged
    rows = np.flatnonzero(visible.numpy())
    for row in rows[:10]:
        nearest = torch.topk(dist[row], 3, largest=False).indices
        assert np.allclose(pred.heads["ref_nearest_3"][row], x[nearest].mean(0).numpy())
        assert not np.allclose(pred.heads["ref_nearest_3"][row], x[row].numpy())
        others = visible.clone()
        others[row] = False
        assert np.allclose(pred.heads["ref_slide_mean"][row], x[others].mean(0).numpy())  # the slide without the cell
    hidden_row = int(np.flatnonzero(d.mask.numpy())[0])
    assert np.allclose(pred.heads["ref_slide_mean"][hidden_row], x[visible].mean(0).numpy())  # as before
    tables = visibility_scores(pred, [f"g{k}" for k in range(G)], center=x.mean(0).numpy(), near_bins_um=[0, 15, np.inf])
    assert {"local", "global", "ref_nearest_3", "ref_kernel_15um"} <= set(tables["gap"]["head"])


def test_a_blend_keeps_the_distances():
    from interscale.evaluation.masked_probing import blend_heads

    pred, _, _ = _planted()
    blended = blend_heads(pred, "ref_context")
    assert blended.near_hidden_um is pred.near_hidden_um and "blend_identity" in blended.heads
