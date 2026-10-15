"""The decoder heads scored on hidden cells (`masked_probing`, round 3b of attention_plan.md).

`get_model_output` runs without masking, so a per-gene comparison of the two heads read from it
scores how well each head passes the cell's own values through. Pinned here: disc masks for the
evaluation passes come from a private generator and hide whole discs; the head predictions are the
step's own two halves, row-aligned with a per-cell depth into the hidden area that equals the
distance to the nearest visible cell; and the scores read planted predictions correctly per gene,
per group and per depth.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.masked_probing import (
    HeadPredictions,
    collect_head_predictions,
    draw_masks,
    score_heads,
)

G = 6


def _lattice(side):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=1).float()


def _graph(side, seed=0):
    """A lattice graph with 4-neighbour edges, positions in units of 10 µm."""
    n = side * side
    pos = _lattice(side)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and float((pos[i] - pos[j]).abs().sum()) == 1.0:
                src.append(i)
                dst.append(j)
    gen = torch.Generator().manual_seed(seed)
    return Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=torch.tensor([src, dst]), pos=pos)


# --------------------------------------------------------------------------- disc masks


def test_disc_masks_hide_whole_discs_from_their_own_generator():
    datas = [_graph(12, 0), _graph(9, 1)]
    state = torch.get_rng_state()
    draw_masks(datas, "blocks", 0.3, seed=3, radius=2.0)
    assert torch.equal(state, torch.get_rng_state()), "drew from the global torch RNG"
    first = [d.gene_mask.clone() for d in datas]
    for d in datas:
        assert (d.gene_mask.all(1) | ~d.gene_mask.any(1)).all(), "a cell is hidden whole or not at all"
        assert torch.equal(d.mask, d.gene_mask.any(1))
        covered = float(d.mask.float().mean())
        assert 0.3 <= covered <= 0.3 + 13 / d.num_nodes  # at most one disc of radius 2 (13 cells) over
        # Every hidden cell lies within the radius of a hidden cell that has its whole disc hidden.
        dist = torch.cdist(d.pos, d.pos)
        centres = [c for c in torch.where(d.mask)[0] if bool(d.mask[dist[c] <= 2.0].all())]
        covered_by_discs = torch.stack([dist[c] <= 2.0 for c in centres]).any(0)
        assert torch.equal(covered_by_discs, d.mask)
    draw_masks(datas, "blocks", 0.3, seed=3, radius=2.0)
    assert all(torch.equal(a, d.gene_mask) for a, d in zip(first, datas, strict=True)), "same seed, same discs"


@pytest.mark.parametrize(("radius", "rate", "match"), [(None, 0.3, "radius"), (2.0, 1.0, "rate"), (0.0, 0.3, "radius")])
def test_disc_masks_need_a_radius_and_a_rate(radius, rate, match):
    with pytest.raises(ValueError, match=match):
        draw_masks([_graph(5)], "blocks", rate, seed=0, radius=radius)


def test_disc_masks_need_coordinates():
    graph = _graph(5)
    del graph.pos
    with pytest.raises(ValueError, match="pos"):
        draw_masks([graph], "blocks", 0.3, seed=0, radius=2.0)


# --------------------------------------------------------------------------- the predictions


def _module(residual=False, strategy="node", max_seq_len=256):
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
    cfg.model.decoder.global_residual = residual
    torch.manual_seed(0)
    return DualDecoderCombinedModule(
        cfg=cfg,
        n_input=G,
        n_output=G,
        n_embed=16,
        decoder_type=None,
        dropout_decoder=0.1,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy=strategy,
    )


def _datas():
    datas = [_graph(10, 0), _graph(7, 1)]
    for d in datas:
        d.obs_names = torch.arange(d.num_nodes)
    draw_masks(datas, "blocks", 0.3, seed=0, radius=2.0)
    return datas


def _collect(module, datas, strategy="node", batch_size=2):
    return collect_head_predictions(
        module,
        datas,
        prediction_task="regression",
        prediction_level="node",
        mask_strategy=strategy,
        batch_size=batch_size,
        device=torch.device("cpu"),
        spatial_unit_um=10.0,
    )


def test_the_predictions_are_the_steps_two_halves_with_the_mask_it_used():
    from torch_geometric.data import Batch

    module, datas = _module(residual=True), _datas()
    pred = _collect(module, datas)

    module.eval()
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.cat([d.obs_names for d in datas])
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    n = len(out.y_pred) // 2
    assert np.allclose(pred.heads["local"], out.y_pred[:n].numpy(), atol=1e-6)
    assert np.allclose(pred.heads["global"], out.y_pred[n:].numpy(), atol=1e-6)
    assert np.allclose(pred.true, out.y_true[:n].numpy())
    hidden_cells = torch.cat([d.mask for d in datas]).numpy()
    assert np.array_equal(pred.hidden, np.repeat(hidden_cells[:, None], G, axis=1))


def test_the_depth_is_the_distance_to_the_nearest_visible_cell_of_the_same_graph():
    """On graphs longer than the sequence, so each graph keeps a subset of its cells as tokens:
    every row's depth must still be its own cell's, found here by matching its true values."""
    import random

    datas = _datas()
    random.seed(0)
    pred = _collect(_module(max_seq_len=40), datas, batch_size=2)
    assert len(pred.true) == 40 + 40  # both graphs (100 and 49 cells) subsampled to 40

    rows_seen = 0
    for graph, d in enumerate(datas):
        hid = d.mask
        depth = torch.zeros(d.num_nodes, dtype=torch.float64)
        depth[hid] = torch.cdist(d.pos, d.pos)[hid][:, ~hid].min(dim=1).values.double() * 10.0
        rows = slice(40 * graph, 40 * graph + 40)
        node = [int(torch.where((d.x == torch.tensor(t, dtype=d.x.dtype)).all(1))[0]) for t in pred.true[rows]]
        assert np.allclose(pred.depth_um[rows], depth[node].numpy())
        rows_seen += len(node)
    assert rows_seen == len(pred.depth_um)
    assert (pred.depth_um[pred.hidden.all(1)] >= 10.0).all()


def test_under_gene_masking_only_wholly_hidden_cells_have_a_depth():
    datas = _datas()
    for d in datas:  # a few extra hidden entries outside the discs
        d.gene_mask[~d.mask, 0] = True
        d.mask = d.gene_mask.any(1)
    pred = _collect(_module(strategy="gene"), datas, strategy="gene")
    wholly = np.concatenate([d.gene_mask.all(1).numpy() for d in datas])
    assert ((pred.depth_um > 0) == wholly).all()
    assert pred.hidden.any(1).all()


def test_the_pass_leaves_the_module_and_the_random_state_as_they_were():
    module, datas = _module(), _datas()
    module.train()
    state = torch.get_rng_state()
    _collect(module, datas)
    assert module.training
    assert torch.equal(state, torch.get_rng_state())


def _single_head_module(kind, max_seq_len=256):
    """The module of a GlobalModel (learned linear input) or a LocalModel (GCN), at rate 0 with discs."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.module.base import GlobalModule, LocalModule

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    common = {"n_input": G, "n_output": G, "n_embed": 16, "decoder_type": "linear", "dropout_decoder": 0.1}
    common |= {"decoder_hidden_dims": [16], "mask_percentage": 0.0, "mask_strategy": "node", "block_masking": True}
    torch.manual_seed(0)
    if kind == "global":
        return GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **common)
    return LocalModule.from_config(cfg, **common)


@pytest.mark.parametrize("kind", ["global", "local"])
def test_a_single_head_model_gives_its_one_head_on_the_cells_it_hid(kind):
    """GlobalModel and LocalModel modules: one head, named for the component, the step's own output,
    and none of the hidden cells' values reach it."""
    from torch_geometric.data import Batch

    module, datas = _single_head_module(kind), _datas()
    pred = _collect(module, datas)
    assert set(pred.heads) == {kind}

    module.eval()
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.cat([d.obs_names for d in datas])
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    assert np.allclose(pred.heads[kind], out.y_pred.numpy(), atol=1e-6)
    hidden_cells = torch.cat([d.mask for d in datas]).numpy()
    assert np.array_equal(pred.hidden, np.repeat(hidden_cells[:, None], G, axis=1))

    changed = [d.clone() for d in datas]
    for d in changed:
        d.x = d.x.clone()
        d.x[d.mask] = d.x[d.mask] + 5.0
    again = _collect(module, changed)
    assert np.allclose(again.heads[kind], pred.heads[kind], atol=1e-6)


def test_a_single_head_model_scores_without_global_minus_local_columns():
    pred = _collect(_single_head_module("global"), _datas())
    tables = score_heads(pred, [f"g{j}" for j in range(G)], center=pred.true.mean(0), depth_bins_um=[0, 15, 100])
    assert "global_sce" in tables["cells"] and not any("global - local" in c for c in tables["cells"].columns)
    assert not any("global - local" in c for c in tables["per_gene"].columns)


# --------------------------------------------------------------------------- the scores


def _planted(n=400, seed=0):
    """Global predicts the truth; local predicts each gene's mean over its hidden entries."""
    rng = np.random.default_rng(seed)
    true = rng.random((n, G)) * 2 + 3
    hidden = np.zeros((n, G), bool)
    hidden[: n // 2] = True  # half the cells hidden whole
    hidden[n // 2 :, 0] = rng.random(n - n // 2) < 0.3  # a few single entries in the rest
    local = np.tile(np.nanmean(np.where(hidden, true, np.nan), axis=0), (n, 1))
    depth = np.zeros(n)
    depth[: n // 2] = np.linspace(10, 400, n // 2)
    return HeadPredictions(
        heads={"local": local, "global": true.copy()}, true=true, hidden=hidden, depth_um=depth
    ), true


def test_the_scores_read_planted_heads_per_gene_group_and_depth():
    pred, true = _planted()
    genes = [f"g{j}" for j in range(G)]
    center = true.mean(0)
    tables = score_heads(
        pred, genes, center=center, gene_groups={"a": genes[:3], "b": genes[3:]}, depth_bins_um=[0, 100, 500]
    )

    per_gene = tables["per_gene"].set_index("gene")
    assert np.allclose(per_gene["global_r2"], 1.0) and np.allclose(per_gene["global_pearson"], 1.0)
    assert np.allclose(per_gene["local_r2"], 0.0, atol=1e-12)  # the mean of the scored entries
    assert np.allclose(per_gene["global - local_r2"], 1.0)
    assert per_gene.loc["g0", "n_hidden"] > per_gene.loc["g1", "n_hidden"] == 200

    cells = tables["cells"].iloc[0]
    assert cells["global_sce"] == pytest.approx(0.0, abs=1e-12)
    assert cells["local_sce"] > 0 and cells["global - local_sce"] < 0
    assert cells["global_cosine_centred"] == pytest.approx(1.0)
    assert cells["n_cells"] == int(pred.hidden.any(1).sum())

    groups = tables["by_group"].set_index("group")
    assert np.allclose(groups["global_cosine"], 1.0) and (groups["global - local_cosine"] > 0).all()
    # Pooled over the deviations from the centre: a mean prediction carries no direction (raw: ~0.97).
    assert (groups["local_cosine"].abs() < 0.3).all()

    depth = tables["by_depth"]
    assert depth["n_cells"].tolist() == [
        int(((pred.depth_um > 0) & (pred.depth_um <= 100)).sum()),
        200 - depth["n_cells"].iloc[0],
    ]
    assert np.allclose(depth["global_sce"], 0.0)
    # Bins reaching below 0 still hold only wholly hidden cells, not the visible ones at depth 0.
    below = score_heads(pred, genes, depth_bins_um=[-1, 100, 500])["by_depth"]
    assert below["n_cells"].tolist() == depth["n_cells"].tolist()


def test_without_a_center_or_a_depth_the_optional_parts_are_left_out():
    pred, _ = _planted()
    pred.depth_um = None
    tables = score_heads(pred, [f"g{j}" for j in range(G)])
    assert set(tables) == {"cells", "per_gene"}
    assert not any("centred" in c for c in tables["cells"].columns)
    with pytest.raises(ValueError, match="depth"):
        score_heads(pred, [f"g{j}" for j in range(G)], depth_bins_um=[0, 100])
