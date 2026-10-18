"""Patch and label flow of attention on masked passes (``evaluation.attention_flow``).

Pinned here: patches are numbered on each graph's own grid; uniform attention reads as the uniform
expectation and as the distance expectation (both divergences 0); attention that is a function of
distance alone is explained by the distance expectation but not the uniform one; label flow sums to
1 per query label; the graphs' own masks survive several disc draws; influence rows come for hidden
queries; no generator is touched; and a sparse backend's zeros are not read as blocked keys, here
or in ``attention_by_query``.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.attention_flow import interesting_sources, patch_flow, patch_index, plot_patch_flow

H, E, G = 2, 8, 6
UNIT = 10.0  # µm per lattice step


def _graphs(sides=(8, 7), seed=0):
    gen = torch.Generator().manual_seed(seed)
    datas = []
    for side in sides:
        ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
        n = side * side
        data = Data(
            x=torch.rand(n, G, generator=gen) * 3.0, edge_index=(torch.cdist(pos, pos) == 1).nonzero().t(), pos=pos
        )
        data.mask = (pos - pos.mean(0)).norm(dim=1) <= 1.6
        data.gene_mask = data.mask[:, None].expand(-1, G).clone()
        data.half = torch.nn.functional.one_hot((pos[:, 0] >= side / 2).long(), 2).float()
        datas.append(data)
    return datas


def _module(kind="dense", bias=(), uniform=False, slope=None):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len = 128
    params.n_heads = H
    params.attention.kind = kind
    params.attention.topk.k = 3
    params.pe.bias = list(bias)
    params.pe.distance.kind = "linear"
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = UNIT
    torch.manual_seed(0)
    common = {"n_input": G, "n_output": G, "n_embed": E, "decoder_type": "linear", "dropout_decoder": 0.0}
    common |= {"decoder_hidden_dims": [8], "mask_percentage": 0.0, "mask_strategy": "node", "block_masking": True}
    module = GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **common)
    with torch.no_grad():
        if uniform:
            for layer in module.transformer_encoder.layers:
                inner = layer.self_attn.in_proj_weight.shape[0] // 3
                layer.self_attn.in_proj_weight[: 2 * inner] = 0.0
                layer.self_attn.in_proj_bias[: 2 * inner] = 0.0
        if slope is not None:
            module.attention_bias.encoders["distance"].slope.fill_(slope)
    return module


def _flow(module, datas, **kwargs):
    options = {
        "prediction_task": "regression",
        "prediction_level": "node",
        "device": torch.device("cpu"),
        "spatial_unit_um": UNIT,
        "patch_um": 20.0,
        "distance_bins_um": list(np.arange(0.0, 130.0, 5.0)),
        "min_queries": 1,
    }
    return patch_flow(module, datas, **(options | kwargs))


def _summary(found, layer="l0", group="hidden"):
    table = found["summary"]
    return table[(table["layer"] == layer) & (table["group"] == group)].set_index("metric")["value"]


def test_patches_are_numbered_on_each_graphs_own_grid():
    pos = np.array([[100.0, 50.0], [119.0, 50.0], [121.0, 75.0], [100.0, 89.0]])
    patch, centres = patch_index(pos, 20.0)
    # From the lower corner (100, 50): x cells 0, 0, 1, 0; y cells 0, 0, 1, 1; 2 x 2 grid.
    assert patch.tolist() == [0, 0, 3, 1]
    assert np.allclose(centres, [[110, 60], [110, 80], [130, 60], [130, 80]])


def test_uniform_attention_is_explained_by_both_expectations():
    found = _flow(_module(uniform=True), _graphs())
    for group in ("hidden", "visible"):
        summary = _summary(found, group=group)
        assert summary["kl_uniform"] == pytest.approx(0.0, abs=1e-9)
        assert summary["kl_distance"] == pytest.approx(0.0, abs=1e-9)
        assert summary["self_mass"] == pytest.approx(summary["self_uniform"])
    patches = found["patches"]
    assert np.allclose(patches["observed"], patches["uniform"])
    assert set(patches["layer"]) == {"l0", "l1"} and set(patches["graph"]) == {0, 1}
    # Exactly: each query spreads over every other cell of its graph (the diagonal is blocked), and
    # a group's sources hold only its own queries.
    for g, data in enumerate(_graphs()):
        patch, _ = patch_index(data.pos.numpy() * UNIT, 20.0)
        n = data.num_nodes
        for group, cells in (("hidden", data.mask.numpy()), ("visible", ~data.mask.numpy())):
            part = patches[(patches["graph"] == g) & (patches["layer"] == "l0") & (patches["group"] == group)]
            queries = part.groupby("source")["n_queries"].first()
            assert queries.to_dict() == pd.Series(patch[cells]).value_counts().to_dict()
            for (source, target), value in part.set_index(["source", "target"])["uniform"].items():
                in_source = np.flatnonzero(cells & (patch == source))
                expected = sum(((patch == target).sum() - (patch[i] == target)) / (n - 1) for i in in_source)
                assert value == pytest.approx(expected)


def test_attention_by_distance_alone_is_explained_by_the_distance_expectation():
    found = _flow(_module(bias=("distance",), uniform=True, slope=-200.0), _graphs())
    summary = _summary(found)
    assert summary["kl_uniform"] > 0.3
    assert summary["kl_distance"] < 0.05 * summary["kl_uniform"]
    assert summary["self_mass"] > summary["self_uniform"]
    assert summary["self_mass"] == pytest.approx(summary["self_distance"], abs=0.05)
    sources = interesting_sources(found["patches"], graph=0, layer="l0", n=2)
    assert len(sources) == 2


def test_labels_flow_sums_to_one_per_query_label_and_the_graphs_keep_their_masks():
    datas = _graphs()
    before = [d.mask.clone() for d in datas]
    state = torch.get_rng_state()
    found = _flow(_module(), datas, label_key="half", mask_seeds=[1, 2, 3], block_radius_um=15.0, block_fraction=0.3)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a, d.mask) for a, d in zip(before, datas, strict=True))
    labels = found["labels"]
    totals = labels.groupby(["layer", "group", "query_label"])["mass"].sum()
    assert np.allclose(totals, 1.0)
    uniform = _flow(_module(uniform=True), _graphs(), label_key="half")["labels"]
    assert np.allclose(uniform["lift_uniform"], 1.0) and np.allclose(uniform["lift_distance"], 1.0)


def test_influence_rows_come_for_hidden_queries_over_visible_keys():
    found = _flow(_module(), _graphs(), influence_queries_per_patch=1)
    rows = found["patches"]
    influence = rows[rows["layer"] == "influence"]
    assert len(influence) and set(influence["group"]) == {"hidden"}
    assert np.isfinite(_summary(found, layer="influence")["kl_uniform"])


def test_a_sparse_backends_zeros_are_not_blocked_keys():
    from interscale.evaluation.attention_diagnostics import attention_by_query

    datas = _graphs()
    found = _flow(_module(kind="topk"), datas)
    patches = found["patches"]
    # Uniform expectation over every open key: the whole slide, not the 3 kept keys.
    # The uniform expectation depends on the mask only, so it is the one uniform attention gets.
    reference = _flow(_module(uniform=True), datas)["patches"]
    keys = ["graph", "layer", "group", "source", "target"]
    merged = patches.merge(reference, on=keys, suffixes=("", "_ref"), how="outer").fillna(0.0)
    assert len(merged) == len(reference)
    assert np.allclose(merged["uniform"], merged["uniform_ref"])
    by_query = attention_by_query(
        _module(kind="topk"),
        datas,
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
        spatial_unit_um=UNIT,
        depth_bins_um=[0.0, 15.0, 100.0],
    )["by_query"]
    effective = by_query[(by_query["metric"] == "effective_rel") & (by_query["head"] == "mean")]["value"]
    assert (effective < 0.2).all(), "3 keys of ~50 open: effective keys far below the open ones"


def test_the_flow_map_plots_its_sources():
    import matplotlib

    matplotlib.use("Agg")
    found = _flow(_module(), _graphs())
    fig = plot_patch_flow(found["patches"], found["centres"], graph=0, layer="l1", sources=[0, 5], patch_um=20.0)
    assert len(fig.axes) == 3  # two panels and the colour bar
    assert isinstance(found["centres"], pd.DataFrame)
