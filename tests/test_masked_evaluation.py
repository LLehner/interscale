"""The masked-pass evaluation (global_plan.md, evaluation rework of 2026-10-17).

Pinned here, on synthetic arrays and toy modules:

* the shared row statistics read 1 for uniform weights (span, effective cells, distance profile),
  and a planted sender ranks 1 among distance-matched cells while a distance kernel ranks 0.5;
* the probe pass gathers each cell's depth and any extra feature set in probe row order; the
  smoother set is the references side by side and never reads a hidden value; the position set is
  the same wherever a slide lies, one patch per scale, reads no expression and gives a probe what
  position alone says; the label and covariate probes score by depth, their pooled rows unchanged;
  the heads score genes by cosine;
* ``attention_by_query`` reads the maps a masked step returns, gives exactly uniform attention 1
  against every uniform reference and 0.5 on planted pairs, and touches no generator;
* ``receptive_field`` estimates the Jacobian norms it claims, reads no influence where a GCN cannot
  reach and some where a transformer can, and leaves the module as it was;
* CKA is 1 under rotation and scaling, the participation ratio counts directions, and the table
  groups cells by depth.
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation import _spatial_weights as sw
from interscale.evaluation.masked_probing import (
    HeadPredictions,
    combined_features,
    feature_sets,
    position_features,
    probe_covariates,
    probe_labels,
    reference_predictions,
    score_heads,
    smoother_features,
)
from interscale.evaluation.online_probes import ProbeBatchFeatures, collect_features, hidden_depth

G, E, UNIT = 6, 16, 10.0  # genes, embedding width, µm per lattice step


def _lattice_graph(side, seed=0):
    """A ``side x side`` lattice, 4-neighbour edges, unit spacing (10 µm), named by position + 1."""
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    n = side * side
    near = (torch.cdist(pos, pos, p=1) == 1).nonzero().t()
    gen = torch.Generator().manual_seed(seed)
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=near, pos=pos)
    data.obs_names = torch.arange(n, dtype=torch.float32) + 1
    return data


def _disc(data, centre, radius):
    """Hide the cells within ``radius`` of ``centre``: ``mask`` and ``gene_mask``, as draw_masks writes them."""
    hidden = (data.pos - torch.tensor(centre, dtype=torch.float32)).norm(dim=1) <= radius
    data.mask = hidden
    data.gene_mask = hidden[:, None].expand(-1, G).clone()
    return data


def _cfg():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 256
    return cfg


_COMMON = {"n_input": G, "n_output": G, "n_embed": E, "decoder_type": "linear", "dropout_decoder": 0.0}
_COMMON |= {"decoder_hidden_dims": [16], "mask_percentage": 0.0, "mask_strategy": "node", "block_masking": True}


def _global_module(uniform=False, max_seq_len=256):
    """A GlobalModel's module, learned linear input, dense attention; ``uniform`` zeroes queries and keys.

    Below the graph size, ``max_seq_len`` makes ``pad_batch`` subsample, so tokens and cells differ in order."""
    from interscale.module.base import GlobalModule

    cfg = _cfg()
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    torch.manual_seed(0)
    module = GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **_COMMON)
    if uniform:
        with torch.no_grad():
            for layer in module.transformer_encoder.layers:
                layer.self_attn.in_proj_weight[: 2 * E] = 0.0
                layer.self_attn.in_proj_bias[: 2 * E] = 0.0
    return module


def _local_module():
    from interscale.module.base import LocalModule

    torch.manual_seed(0)
    return LocalModule.from_config(_cfg(), **_COMMON)


def _dual_module():
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = _cfg()
    cfg.model.decoder.dual_decoder = True
    torch.manual_seed(0)
    common = dict(_COMMON)
    common.pop("block_masking")
    common["decoder_type"] = None
    module = DualDecoderCombinedModule(cfg=cfg, **common)
    module.masked_nodes = True  # discs only, as block_masking makes it
    return module


# --------------------------------------------------------------------------- shared statistics


def test_uniform_weights_read_one_against_every_uniform_reference():
    dist = torch.tensor([[0.0, 100.0, 200.0, 400.0], [100.0, 0.0, 100.0, 300.0]], dtype=torch.float64)
    keep = torch.tensor([[True, True, True, False], [True, True, True, True]])
    stats = sw.row_statistics(keep.double(), dist, keep, radii_um=(150.0,))
    assert torch.allclose(stats["span_rel"], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(stats["effective_rel"], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(stats["beyond_150um"], torch.tensor([1 / 3, 1 / 4], dtype=torch.float64))

    one_hot = torch.tensor([[0.0, 0.0, 1.0, 5.0], [0.0, 0.0, 0.0, 2.0]], dtype=torch.float64)  # 5.0 not kept
    stats = sw.row_statistics(one_hot, dist, keep)
    assert torch.allclose(stats["span_um"], torch.tensor([200.0, 300.0], dtype=torch.float64))
    assert torch.allclose(stats["effective_rel"], torch.tensor([1 / 3, 1 / 4], dtype=torch.float64))

    empty = sw.row_statistics(torch.zeros(1, 4, dtype=torch.float64), dist[:1], keep[:1])
    assert all(torch.isnan(v).all() for v in empty.values())


def test_the_distance_profile_of_uniform_weights_lifts_one_in_every_bin():
    gen = torch.Generator().manual_seed(0)
    dist = torch.rand(5, 40, generator=gen, dtype=torch.float64) * 1000
    keep = torch.rand(5, 40, generator=gen) < 0.7
    bins = [0, 250, 500, 1000]
    weight, uniform = sw.distance_profile(keep.double() * 3.0, dist, keep, bins)
    assert torch.allclose(weight.sum(0) / uniform.sum(0), torch.ones(3, dtype=torch.float64))

    near = torch.where(dist <= 250, 1.0, 0.0).double()
    weight, uniform = sw.distance_profile(near, dist, keep, bins)
    lift = weight.sum(0) / uniform.sum(0)
    assert lift[0] > 1 and torch.allclose(lift[1:], torch.zeros(2, dtype=torch.float64))


def test_a_planted_sender_ranks_one_and_a_distance_kernel_one_half():
    """Receiver 0; cells 1-4 at 100 µm, 5-6 at 200 µm; sender 1, and 2 another sender of 0."""
    dist = torch.tensor([[0.0, 100, 100, 100, 100, 200, 200]], dtype=torch.float64)
    keep = torch.ones(1, 7, dtype=torch.bool)
    pairs = pd.DataFrame({"sender": [1, 2], "receiver": [0, 0]})
    exclude = sw.pair_exclusions(pairs, 7, "cpu")
    rows, senders = torch.tensor([0, 0]), torch.tensor([1, 2])

    planted = torch.tensor([[0.0, 0.9, 0.95, 0.1, 0.1, 0.5, 0.5]], dtype=torch.float64)
    ranks = sw.matched_ranks(planted, dist, keep, rows, senders, exclude, tolerance_um=10.0)
    assert ranks.tolist() == [1.0, 1.0]  # compared with cells 3 and 4 only: never with 2, 5 or 6

    kernel = torch.exp(-dist / 100.0)
    assert sw.matched_ranks(kernel, dist, keep, rows, senders, exclude, 10.0).tolist() == [0.5, 0.5]

    blocked = keep.clone()
    blocked[0, 1] = False
    ranks = sw.matched_ranks(planted, dist, blocked, rows, senders, exclude, 10.0)
    assert np.isnan(ranks[0].item()) and ranks[1].item() == 1.0


def test_the_default_tolerance_keeps_a_lattice_neighbour_from_ranking_against_the_diagonals():
    """On the 100 um lattice a +-50 um tolerance compares a 100 um sender with the 141 um diagonals, which a
    distance kernel ranks below it: the planted pair read 1.0 for weights that know only distance."""
    dist = torch.tensor([[0.0, 100, 100, 100, 100, 141, 141, 141, 141]], dtype=torch.float64)
    keep = torch.ones(1, 9, dtype=torch.bool)
    kernel = torch.exp(-dist / 100.0)
    exclude = torch.zeros(1, 9, dtype=torch.bool)
    rows, senders = torch.tensor([0]), torch.tensor([1])
    assert sw.matched_ranks(kernel, dist, keep, rows, senders, exclude, 50.0).item() > 0.7
    assert sw.matched_ranks(kernel, dist, keep, rows, senders, exclude, 10.0).item() == 0.5
    import inspect

    from interscale.evaluation.attention_diagnostics import attention_by_query
    from interscale.evaluation.receptive_field import receptive_field

    for function in (attention_by_query, receptive_field):
        assert inspect.signature(function).parameters["tolerance_um"].default == 10.0


def test_planted_edges_are_found_by_their_positions_in_a_graph_of_any_order():
    edges = pd.DataFrame({"sender": [10, 11, 99], "receiver": [12, 10, 10], "range_class": ["short", "long", "long"]})
    names = torch.tensor([13.0, 11.0, 12.0])  # positions 12, 10 and 11, plus one
    found = sw.graph_edges(edges, names)
    assert found.to_dict("list") == {"sender": [1, 2], "receiver": [0, 1], "range_class": ["short", "long"]}


def test_query_groups_put_visible_cells_apart_and_hidden_ones_in_their_depth_bin():
    depth = np.array([0.0, 50.0, 200.0, 900.0])
    groups = sw.query_groups(depth, np.array([False, True, True, True]), [0, 100, 500])
    assert groups.tolist() == [sw.VISIBLE, "(0.0, 100.0]", "(100.0, 500.0]", ""]
    assert sw.bin_labels([0, 100, 500]) == ["(0.0, 100.0]", "(100.0, 500.0]"]


# --------------------------------------------------------------------------- probes


def test_the_probe_pass_gathers_depth_and_extra_features_in_probe_row_order():
    datas = [_disc(_lattice_graph(7, 0), (3, 3), 2.0), _disc(_lattice_graph(6, 1), (0, 0), 1.5)]

    def ids(batch, hidden):
        return {"id": torch.stack([batch.batch, torch.arange(batch.num_nodes)], dim=1).float()}

    features = collect_features(
        _global_module(max_seq_len=30),  # both graphs subsampled: probe rows are not batch order
        datas,
        prediction_task="regression",
        prediction_level="node",
        embeddings=("global",),
        gene_index={},
        categorical_targets=(),
        obs_targets=(),
        mask_strategy="node",
        batch_size=2,
        device=torch.device("cpu"),
        spatial_unit_um=UNIT,
        extra_features=ids,
    )
    offsets = np.cumsum([0] + [d.num_nodes for d in datas])
    expected = np.concatenate(
        [hidden_depth(d.pos, d.mask, torch.zeros(d.num_nodes, dtype=torch.long)).numpy() * UNIT for d in datas]
    )
    rows = features.embeddings["id"][:, 1].astype(int)  # batch node of each probe row
    assert len(rows) == 60 and len(set(rows)) == 60  # 30 of each graph's cells, not all 85
    assert np.array_equal(features.embeddings["id"][:, 0], np.searchsorted(offsets, rows, side="right") - 1)
    assert np.allclose(features.depth_um, expected[rows])
    assert np.array_equal(features.cell_masked, np.concatenate([d.mask.numpy() for d in datas])[rows])
    assert (features.depth_um[features.cell_masked] > 0).all() and (features.depth_um[~features.cell_masked] == 0).all()
    picked = features.select(np.arange(5))
    assert np.array_equal(picked.depth_um, features.depth_um[:5])


def test_the_smoother_is_the_references_side_by_side_and_never_reads_a_hidden_value():
    data = _disc(_lattice_graph(8), (4, 4), 2.0)
    batch = Batch.from_data_list([data])
    hidden = batch.gene_mask.bool()
    smoother = smoother_features(spatial_unit_um=UNIT, kernels_um=(15.0,), k_nearest=(3,))
    found = smoother(batch, hidden)["smoother"]
    refs = reference_predictions(
        batch.x, hidden, batch.pos, batch.batch, spatial_unit_um=UNIT, kernels_um=(15.0,), k_nearest=(3,)
    )
    assert found.shape == (batch.num_nodes, 3 * G)
    assert torch.allclose(found, torch.cat(list(refs.values()), dim=1).float())

    changed = batch.clone()
    changed.x = batch.x.clone()
    changed.x[hidden] += 7.0
    assert torch.equal(smoother(changed, hidden)["smoother"], found)


def test_feature_sets_add_the_smoother_to_each_model_embedding():
    features = ProbeBatchFeatures(embeddings={k: np.zeros((3, 2)) for k in ("local", "global", "pe", "smoother")})
    sets = feature_sets(features)
    assert {"local+global", "smoother+local", "smoother+global", "smoother+local+global"} <= set(sets)
    assert "smoother+pe" not in sets and sets["smoother+local+global"].shape == (3, 6)
    assert set(feature_sets(ProbeBatchFeatures(embeddings={"global": np.zeros((3, 2))}))) == {"global"}


def test_the_position_is_the_same_wherever_a_slide_lies_one_patch_per_scale_and_no_expression():
    first, moved = _lattice_graph(6, 0), _lattice_graph(6, 1)
    moved.pos = moved.pos + torch.tensor([300.0, -70.0])  # the same lattice elsewhere, other expression
    batch = Batch.from_data_list([first, moved])
    position = position_features(spatial_unit_um=UNIT, patch_um=(20.0, 60.0), max_offset_um=25.0)
    found = position(batch, torch.zeros(batch.num_nodes, G, dtype=torch.bool))["position"]
    # offset (2) + distance (1) + 2 * ceil(25 / 20) = 4 patches a side + 2 * ceil(25 / 60) = 2 a side
    assert found.shape == (72, 3 + 16 + 4)
    assert torch.equal(found[:36], found[36:]), "two copies of a slide, placed apart, differ"
    offset = found[:36, :2]
    assert torch.allclose(offset.mean(0), torch.zeros(2), atol=1e-5)  # relative to the centroid
    assert torch.allclose(found[:36, 2], offset.norm(dim=1))
    assert torch.allclose(offset[1] - offset[0], torch.tensor([0.0, UNIT]))  # in µm
    assert (found[:, 3:19].sum(1) == 1).all() and (found[:, 19:].sum(1) == 1).all()
    # The 60 µm grid has 2 x 2 patches around the centroid: the quadrants.
    quadrant = found[:36, 19:].argmax(1)
    expected = (offset[:, 0] >= 0).long() * 2 + (offset[:, 1] >= 0).long()
    assert torch.equal(quadrant, expected)

    changed = batch.clone()
    changed.x = torch.rand_like(batch.x)
    assert torch.equal(position(changed, torch.ones_like(batch.x, dtype=torch.bool))["position"], found)

    # Beyond max_offset_um a cell falls into the border patch rather than out of the grid.
    narrow = position_features(spatial_unit_um=UNIT, patch_um=(20.0,), max_offset_um=5.0)
    clipped = narrow(batch, torch.zeros(batch.num_nodes, G, dtype=torch.bool))["position"]
    assert clipped.shape == (72, 3 + 4) and (clipped[:, 3:].sum(1) == 1).all()


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"patch_um": (0.0,)}, "positive"), ({"max_offset_um": 0.0}, "max_offset")]
)
def test_the_position_needs_positive_sizes(kwargs, match):
    options = {"spatial_unit_um": UNIT, "patch_um": (20.0,), "max_offset_um": 25.0} | kwargs
    with pytest.raises(ValueError, match=match):
        position_features(**options)


def test_a_probe_on_the_position_reads_a_label_fixed_by_place_but_not_one_drawn_per_slide():
    def slides(seed):
        datas = []
        rng = np.random.default_rng(seed)
        for k in range(6):
            data = _disc(_lattice_graph(8, seed * 10 + k), (3, 3), 2.5)
            data.pos = data.pos + torch.tensor(rng.uniform(-500, 500, 2), dtype=torch.float32)
            # One-hot, as annotations reach the graphs: the upper half of every slide, and one
            # class per slide drawn at random.
            data.fixed = torch.nn.functional.one_hot((torch.arange(64) % 8 >= 4).long(), 2).float()
            data.drawn = torch.nn.functional.one_hot(torch.full((64,), int(rng.integers(0, 2))), 2).float()
            datas.append(data)
        return datas

    position = position_features(spatial_unit_um=UNIT, patch_um=(40.0,), max_offset_um=40.0)
    common = {
        "prediction_task": "regression",
        "prediction_level": "node",
        "embeddings": ("global",),
        "gene_index": {},
        "categorical_targets": ("fixed", "drawn"),
        "obs_targets": (),
        "mask_strategy": "node",
        "batch_size": 3,
        "device": torch.device("cpu"),
        "extra_features": combined_features(position, smoother_features(spatial_unit_um=UNIT, kernels_um=(20.0,))),
    }
    module = _global_module()
    train, test = collect_features(module, slides(0), **common), collect_features(module, slides(1), **common)
    assert {"position", "smoother"} <= set(train.embeddings)
    labels = probe_labels(train, test, ["fixed", "drawn"])
    accuracy = labels[labels["metric"] == "balanced_accuracy"].set_index(["target", "feature_set"])["value"]
    assert accuracy[("fixed", "position")] > 0.95
    assert accuracy[("drawn", "position")] < 0.7
    assert {"smoother+position", "smoother+position+global"} <= set(accuracy.loc["fixed"].index)


def test_feature_sets_put_the_position_beside_the_smoother_as_one_bar():
    features = ProbeBatchFeatures(embeddings={k: np.zeros((3, 2)) for k in ("global", "smoother", "position")})
    sets = feature_sets(features)
    assert {"smoother+global", "smoother+position", "smoother+position+global"} <= set(sets)
    assert "position+global" not in sets and sets["smoother+position+global"].shape == (3, 6)
    alone = feature_sets(ProbeBatchFeatures(embeddings={k: np.zeros((3, 2)) for k in ("global", "position")}))
    assert set(alone) == {"global", "position", "position+global"}


def _depth_features(n, seed):
    """Labels readable from ``global`` on shallow cells only; every cell hidden, half shallow, half deep."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    depth = np.where(np.arange(n) % 2 == 0, 50.0, 400.0)
    signal = np.where(depth < 100, y, rng.integers(0, 2, n)).astype(float)
    x = np.stack([signal + 0.05 * rng.normal(size=n), rng.normal(size=n)], axis=1)
    return ProbeBatchFeatures(
        embeddings={"global": x},
        categorical={"label": y},
        continuous={"c1": y + 0.01 * rng.normal(size=n), "c2": -y + 0.01 * rng.normal(size=n)},
        cell_masked=np.ones(n, bool),
        depth_um=depth,
    )


def test_label_and_covariate_probes_score_by_depth_and_keep_their_pooled_rows():
    train, test = _depth_features(800, 0), _depth_features(800, 1)
    bins = [0, 100, 500]
    labels = probe_labels(train, test, ["label"], depth_bins_um=bins)
    acc = labels[(labels["feature_set"] == "global") & (labels["metric"] == "balanced_accuracy")].set_index("depth_um")
    assert acc.loc["(0.0, 100.0]", "value"] > 0.95
    assert abs(acc.loc["(100.0, 500.0]", "value"] - 0.5) < 0.1
    assert acc.loc["(0.0, 100.0]", "n"] == 400
    pooled = probe_labels(train, test, ["label"])
    assert "depth_um" not in pooled
    merged = labels[labels["depth_um"] == "all"].drop(columns="depth_um").reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, pooled)

    covariates = probe_covariates(train, test, ["c1", "c2"], depth_bins_um=bins)
    cos = covariates[covariates["feature_set"] == "global"].set_index("depth_um")["value"]
    assert cos["(0.0, 100.0]"] > 0.9 > cos["(100.0, 500.0]"]
    pooled = probe_covariates(train, test, ["c1", "c2"])
    assert np.isclose(cos["all"], pooled.loc[pooled["feature_set"] == "global", "value"].item())


def test_a_depth_bin_holding_one_class_scores_nan():
    train, test = _depth_features(400, 0), _depth_features(400, 1)
    test.categorical["label"] = np.where(test.depth_um > 100, 1, test.categorical["label"])  # deep: one class
    labels = probe_labels(train, test, ["label"], depth_bins_um=[0, 100, 500])
    deep = labels[labels["depth_um"] == "(100.0, 500.0]"]
    assert deep["value"].isna().all() and (deep["n"] == 200).all()
    assert labels.loc[labels["depth_um"] != "(100.0, 500.0]", "value"].notna().all()


def test_probes_by_depth_need_the_depth():
    train = _depth_features(50, 0)
    train.depth_um = None
    with pytest.raises(ValueError, match="spatial_unit_um"):
        probe_labels(train, train, ["label"], depth_bins_um=[0, 100])


def test_the_heads_can_score_genes_by_centred_cosine_alone():
    rng = np.random.default_rng(0)
    true = rng.random((300, G)) + 2.0
    hidden = np.zeros((300, G), bool)
    hidden[:150] = True
    mean = true.mean(0)
    scaled = mean + 2.0 * (true - mean)  # the deviations doubled: centred cosine 1, raw cosine below
    pred = HeadPredictions(heads={"global": scaled, "local": true[::-1].copy()}, true=true, hidden=hidden)
    genes = [f"g{j}" for j in range(G)]
    tables = score_heads(pred, genes, center=mean, gene_groups={"a": genes[:3]}, gene_metrics=("cosine",))
    per_gene = tables["per_gene"]
    assert np.allclose(per_gene["global_cosine"], 1.0)
    assert {"local_cosine", "global - local_cosine"} <= set(per_gene) and not any(
        "r2" in c or "pearson" in c for c in per_gene
    )
    assert not any("r2" in c for c in tables["by_group"])
    assert "global_r2" in score_heads(pred, genes, center=true.mean(0))["per_gene"]  # the default is unchanged
    with pytest.raises(ValueError, match="center"):
        score_heads(pred, genes, gene_metrics=("cosine",))
    with pytest.raises(ValueError, match="gene_metrics"):
        score_heads(pred, genes, center=true.mean(0), gene_metrics=("mse",))


# --------------------------------------------------------------------------- attention by query


def _attention(module, datas, **kwargs):
    from interscale.evaluation.attention_diagnostics import attention_by_query

    kwargs.setdefault("depth_bins_um", [0, 15, 100])
    return attention_by_query(
        module,
        datas,
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        spatial_unit_um=UNIT,
        **kwargs,
    )


def _value(table, **where):
    for key, value in where.items():
        table = table[table[key] == value]
    assert len(table) == 1, where
    return float(table["value"].iloc[0])


def test_attention_by_query_reads_the_maps_a_masked_step_returns():
    data = _disc(_lattice_graph(8), (4, 4), 2.5)
    module = _global_module().eval()
    found = _attention(module, [data])["by_query"]

    module.return_attention_maps = True
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        out = module._common_step(batch, "regression", "node")
    module.return_attention_maps = False
    padding = out.view.src_padding_mask[0, :-1]
    spots = (~padding).nonzero().squeeze(1)
    nodes = out.view.padded_node_idx
    hid = data.mask[nodes]
    depth = hidden_depth(data.pos, data.mask, torch.zeros(data.num_nodes, dtype=torch.long))[nodes] * UNIT
    deep = hid & (depth > 15)
    attn = out.attn[1, 2].double()  # layer 1, head 2 (one graph: row = head)
    cells = attn[spots][:, spots]
    keys = cells > 0  # what a query can attend to: not itself, the diagonal is blocked
    visible_mass = cells[:, ~hid].sum(1) / cells.sum(1)
    visible_share = (keys & ~hid[None, :]).sum(1) / keys.sum(1)
    expected = (visible_mass[deep] / visible_share[deep]).mean().item()
    assert _value(found, layer=1, head="h2", query="(15.0, 100.0]", metric="visible_lift") == pytest.approx(expected)
    expected = attn[spots][:, -1][~hid].mean().item()
    assert _value(found, layer=1, head="h2", query=sw.VISIBLE, metric="cls_mass") == pytest.approx(expected)
    heads = found[(found["layer"] == 0) & (found["query"] == sw.VISIBLE) & (found["metric"] == "span_rel")]
    mean = heads[heads["head"] == "mean"]["value"].item()
    assert mean == pytest.approx(heads[heads["head"] != "mean"]["value"].mean())

    changed = data.clone()
    changed.x = data.x.clone()
    changed.x[data.mask] += 5.0
    again = _attention(module, [changed])["by_query"]
    pd.testing.assert_frame_equal(again, found)  # what was hidden never reached attention


def test_uniform_attention_reads_one_against_every_reference_and_one_half_on_planted_pairs():
    data = _disc(_lattice_graph(8), (4, 4), 2.5)
    centre = int(((data.pos - torch.tensor([4.0, 4.0])).norm(dim=1) == 0).nonzero())
    edges = pd.DataFrame(
        {"sender": [centre - 3 * 8, 0], "receiver": [centre, centre], "range_class": ["long", "short"]}
    )
    module = _global_module(uniform=True).eval()
    module.return_attention_maps = True
    with torch.no_grad():
        keys = int((module._common_step(Batch.from_data_list([data]), "regression", "node").attn[0, 0, 0] > 0).sum())
    module.return_attention_maps = False
    found = _attention(module, [data], distance_bins_um=[0, 20, 50, 200], edges=edges)
    table = found["by_query"]
    means = table[table["head"] == "mean"].set_index(["layer", "query", "metric"])["value"]
    for metric in ("span_rel", "effective_rel", "visible_lift"):
        assert np.allclose(means.xs(metric, level="metric"), 1.0)
    assert np.allclose(means.xs("cls_mass", level="metric"), 1.0 / keys)  # CLS is one of the keys
    assert np.allclose(found["profile"]["lift"].dropna(), 1.0)
    edge = found["edges"].set_index(["layer", "range_class", "metric"])["value"]
    assert np.allclose(edge.xs("matched_rank", level="metric"), 0.5)
    assert np.allclose(edge.xs("coverage", level="metric"), 1.0)


def test_attention_by_query_touches_no_generator_and_restores_the_module():
    module = _global_module()
    module.train()
    states = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    _attention(module, [_disc(_lattice_graph(6), (2, 2), 1.5)])
    assert module.training
    assert torch.equal(states[0], torch.get_rng_state())
    assert np.array_equal(states[1], np.random.get_state()[1]) and states[2] == random.getstate()
    assert _attention(_local_module(), [_lattice_graph(4)]) == {}


# --------------------------------------------------------------------------- receptive field


def _field(module, datas, **kwargs):
    from interscale.evaluation.receptive_field import receptive_field

    kwargs.setdefault("depth_bins_um", [0, 15, 25, 100])
    kwargs.setdefault("queries_per_bin", 50)
    return receptive_field(
        module,
        datas,
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        spatial_unit_um=UNIT,
        **kwargs,
    )


def test_the_influence_estimates_the_squared_jacobian_norm():
    from interscale.evaluation.masked_probing import _model_heads
    from interscale.evaluation.receptive_field import _differentiable_inputs, _influence

    data = _disc(_lattice_graph(5), (2, 2), 1.0)
    module = _global_module(max_seq_len=20).eval()  # 25 cells: pad_batch keeps 20, in its own order

    batch = Batch.from_data_list([data])
    random.seed(1)
    with torch.enable_grad(), _differentiable_inputs(module, batch) as captured:
        out = module._common_step(batch, "regression", "node")
        pred = _model_heads(module, out)[0]["global"]
        kept = out.view.padded_node_idx
        query = int(kept[data.mask[kept]][-1])  # a hidden cell the transformer kept
        row = int((kept == query).nonzero())
        assert row != query, "the token order should differ from the cell order here"
        exact = torch.zeros(data.num_nodes, dtype=torch.float64)
        for g in range(G):
            (grad,) = torch.autograd.grad(pred[row, g], captured["inputs"], retain_graph=True)
            exact += grad.double().pow(2).sum(1)

    left_out = int(np.setdiff1d(np.arange(data.num_nodes), kept.numpy())[0])
    random.seed(1)  # the same subsample
    found = _influence(
        module,
        Batch.from_data_list([data]),
        np.array([query, left_out]),
        "regression",
        600,
        torch.Generator().manual_seed(0),
        _model_heads,
    )["global"]
    estimate = found[0]
    assert estimate.sum().item() == pytest.approx(exact.sum().item(), rel=0.1)
    assert np.corrcoef(estimate.numpy(), exact.numpy())[0, 1] > 0.95
    assert torch.isnan(found[1]).all()  # a cell the transformer never saw has no row


def test_a_gcn_reads_no_influence_beyond_its_hops_and_a_transformer_does():
    """A disc of radius 4.5 steps: its centre lies 5 steps from the nearest visible cell, beyond the
    2-layer GCN's reach, while cells near the rim are inside it."""
    data = _disc(_lattice_graph(13), (6, 6), 4.5)
    local = _field(_local_module(), [data])["by_query"].set_index(["head", "query", "metric"])["value"]
    assert local[("local", "(25.0, 100.0]", "influence")] == 0.0
    assert local[("local", "(0.0, 15.0]", "influence")] > 0
    assert ("local", "(25.0, 100.0]", "span_rel") not in local.index  # undefined without influence

    found = _field(_global_module(), [data], radii_um=(25.0,))["by_query"].set_index(["head", "query", "metric"])[
        "value"
    ]
    assert found[("global", "(25.0, 100.0]", "influence")] > 0
    assert found[("global", "(0.0, 15.0]", "beyond_25um")] > 0  # past the GCN's reach


def test_a_dual_decoder_reads_both_heads_and_only_the_global_one_reaches_far():
    data = _disc(_lattice_graph(13), (6, 6), 4.5)
    found = _field(_dual_module(), [data])["by_query"].set_index(["head", "query", "metric"])["value"]
    assert found[("local", "(25.0, 100.0]", "influence")] == 0.0
    assert found[("global", "(25.0, 100.0]", "influence")] > 0


def test_planted_edges_are_ranked_on_the_influence_of_their_hidden_receivers():
    data = _disc(_lattice_graph(9), (4, 4), 2.5)
    centre = int(((data.pos - torch.tensor([4.0, 4.0])).norm(dim=1) == 0).nonzero())
    edges = pd.DataFrame(
        {"sender": [centre - 3 * 9, centre - 3], "receiver": [centre, centre], "range_class": ["a", "b"]}
    )
    found = _field(_global_module(), [data], edges=edges)["edges"].set_index(["range_class"])
    assert set(found.index) == {"a", "b", "any"}
    assert found.loc["any", "n_pairs"] == 2
    assert ((found["value"] >= 0) & (found["value"] <= 1)).all()


def test_the_receptive_field_touches_no_generator_and_restores_the_module():
    module = _global_module()
    module.train()
    flags = [p.requires_grad for p in module.parameters()]
    state = torch.get_rng_state()
    data = _disc(_lattice_graph(7), (3, 3), 2.0)
    first = _field(module, [data], seed=3)["by_query"]
    assert module.training and flags == [p.requires_grad for p in module.parameters()]
    assert torch.equal(state, torch.get_rng_state())
    assert "gex_embedding" not in vars(module)
    pd.testing.assert_frame_equal(first, _field(module, [data], seed=3)["by_query"])


# --------------------------------------------------------------------------- representations


def test_cka_is_one_under_rotation_and_scaling_and_low_for_unrelated_sets():
    from interscale.evaluation.representations import linear_cka

    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)) + 5.0  # off-centre: CKA centres each set first
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0)
    assert linear_cka(x, rng.normal(size=(500, 8)) - 4.0) < 0.1
    assert np.isnan(linear_cka(x, np.ones((500, 3))))


def test_the_participation_ratio_counts_the_directions_cells_spread_along():
    from interscale.evaluation.representations import participation_ratio

    rng = np.random.default_rng(0)
    assert participation_ratio(rng.normal(size=(20000, 6))) == pytest.approx(6.0, rel=0.05)
    line = rng.normal(size=(500, 1)) @ rng.normal(size=(1, 6))
    assert participation_ratio(line) == pytest.approx(1.0)


def test_the_representation_table_groups_cells_by_depth_and_reduces_wide_sets():
    from interscale.evaluation.representations import representation_table

    rng = np.random.default_rng(0)
    n = 600
    latent = rng.normal(size=(n, 2))
    local = latent @ rng.normal(size=(2, 8))
    wide = np.concatenate([local, rng.normal(size=(n, 192))], axis=1)  # the structure, then noise
    features = ProbeBatchFeatures(
        embeddings={"local": local, "global": 2.0 * local, "pe": rng.normal(size=(n, 8)), "smoother": wide},
        depth_um=np.where(np.arange(n) < n // 2, 0.0, 200.0),
    )
    table = representation_table(features, depth_bins_um=[0, 100, 500], max_dims=4)
    cka = table[table["metric"] == "cka"].set_index(["query", "a", "b"])["value"]
    assert set(cka.index.get_level_values("query")) == {sw.VISIBLE, "hidden", "(100.0, 500.0]"}
    assert np.allclose(cka.xs(("local", "global"), level=("a", "b")), 1.0)
    assert (cka.xs(("local", "pe"), level=("a", "b")) < 0.2).all()
    unreduced = representation_table(features, depth_bins_um=[0, 100, 500], max_dims=0)
    unreduced = unreduced[unreduced["metric"] == "cka"].set_index(["query", "a", "b"])["value"]
    assert (
        cka.xs(("local", "smoother"), level=("a", "b")) > unreduced.xs(("local", "smoother"), level=("a", "b")) + 0.3
    ).all()
    ratio = table[table["metric"] == "participation_ratio"].set_index(["query", "a"])["value"]
    assert ratio[("hidden", "local")] < 2.5  # two latent directions
