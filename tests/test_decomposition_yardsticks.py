"""The yardsticks that read how well two components separate (decoder_plan.md, D0; roadmap.md, T2).

* **redundancy** (``representations.redundancy_table``) -- the ridge R² of one representation from another, fitted on the
  training split and scored on the test split, per query group. Pinned on constructed embeddings where the answer is known:
  a linear copy reads ~1, independent noise ~0 on held-out cells (not on the cells fitted), each group is fitted on its own
  cells, a wide source is reduced with the training cells' components, missing sets and small groups are left out;
* **attribution shares** (``masked_probing.attribution_shares``) -- how much of a head's variance each named part supplies:
  the shares add up to 1, a part that works against the others is negative, a constant offset and the visible entries
  change nothing, the groups and the depth bins select their own entries;
* **the parts** -- the modules hand them over (a global-only model's read-out and own-cell skip, a residual dual decoder's
  local and global part), they add up to the head's prediction, and ``collect_head_predictions`` carries them;
* **the report** -- the three tables take their compact form and reach the summary.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation import report
from interscale.evaluation.masked_probing import (
    HeadPredictions,
    attribution_shares,
    blend_heads,
    collect_head_predictions,
)
from interscale.evaluation.online_probes import ProbeBatchFeatures
from interscale.evaluation.representations import REDUNDANCY_PAIRS, redundancy_table

BINS = [0.0, 150.0, 290.0, 500.0, float("inf")]


# --------------------------------------------------------------------------- redundancy


def _features(n=600, dims=8, *, copy_deep=True, seed=0, noise=0.05):
    """local ~ N(0, 1); global = a rotation of local plus noise, except in the deepest bin when ``copy_deep`` is False."""
    rng = np.random.default_rng(seed)
    depth = np.concatenate([np.zeros(n // 2), np.full(n // 4, 100.0), np.full(n - n // 2 - n // 4, 400.0)])
    local = rng.normal(size=(n, dims))
    rotation = np.linalg.qr(np.random.default_rng(123).normal(size=(dims, dims)))[0]  # the same map in every split
    glob = local @ rotation + noise * rng.normal(size=(n, dims))
    if not copy_deep:
        deep = depth > 290
        glob[deep] = rng.normal(size=(int(deep.sum()), dims))
    return ProbeBatchFeatures(embeddings={"local": local, "global": glob}, depth_um=depth)


def _r2(table, target="global", source="local"):
    rows = table[(table["target"] == target) & (table["source"] == source)]
    return rows.set_index("query")["r2"]


def test_a_linear_copy_is_read_in_full_and_independent_noise_not_at_all_on_held_out_cells():
    copy = redundancy_table(_features(seed=0), _features(seed=1), depth_bins_um=BINS)
    assert (_r2(copy) > 0.95).all()
    assert (_r2(copy, "local", "global") > 0.95).all()  # the rotation reads back too
    rng = np.random.default_rng(2)
    train, test = _features(seed=0), _features(seed=1)
    train.embeddings["global"] = rng.normal(size=train.embeddings["global"].shape)
    test.embeddings["global"] = rng.normal(size=test.embeddings["global"].shape)
    independent = _r2(redundancy_table(train, test, depth_bins_um=BINS))
    assert (independent < 0.05).all()


def test_each_group_is_fitted_on_its_own_cells():
    """The global tokens copy the local ones except deep inside: a map learnt on every cell would not say so."""
    found = _r2(
        redundancy_table(_features(copy_deep=False, seed=0), _features(copy_deep=False, seed=1), depth_bins_um=BINS)
    )
    assert found["visible"] > 0.95 and found["(0.0, 150.0]"] > 0.95
    assert found["(290.0, 500.0]"] < 0.05
    assert found["hidden"] < found["visible"]  # the hidden cells pool a copying group and a non-copying one


def test_it_scores_on_the_test_split_so_overfitting_does_not_pass_for_redundancy():
    rng = np.random.default_rng(3)
    n = 40
    make = lambda: ProbeBatchFeatures(
        embeddings={"local": rng.normal(size=(n, 30)), "global": rng.normal(size=(n, 4))}, depth_um=np.zeros(n)
    )
    # An almost unpenalised ridge fits 40 noise cells with 30 dimensions closely; held out it says nothing.
    table = redundancy_table(make(), make(), depth_bins_um=BINS, min_cells=30, alphas=(1e-6,))
    assert _r2(table)["visible"] < 0.1


def test_a_source_wider_than_the_limit_is_reduced_to_its_leading_components_of_the_training_cells():
    """The signal sits in the last 30 of 60 columns, among 30 of pure noise: keeping the first columns would lose it."""

    def wide(seed):
        r = np.random.default_rng(seed)
        latent = r.normal(size=(500, 3))
        columns = r.normal(size=(500, 60))
        columns[:, 30:] = latent @ np.random.default_rng(9).normal(size=(3, 30)) + 0.1 * r.normal(size=(500, 30))
        return ProbeBatchFeatures(
            embeddings={"smoother": columns, "global": latent @ np.random.default_rng(10).normal(size=(3, 5))},
            depth_um=np.zeros(500),
        )

    table = redundancy_table(wide(0), wide(1), depth_bins_um=BINS, max_dims=6, pairs=(("global", "smoother"),))
    assert _r2(table, "global", "smoother")["visible"] > 0.9


def test_missing_sets_small_groups_and_missing_depth_are_handled():
    features = _features()
    table = redundancy_table(features, _features(seed=1), depth_bins_um=BINS)  # no `smoother`, no `pe` in these
    assert set(zip(table["target"], table["source"], strict=True)) == {("global", "local"), ("local", "global")}
    assert set(REDUNDANCY_PAIRS) >= {("global", "local"), ("global", "smoother"), ("global", "pe")}
    few = redundancy_table(features, _features(seed=1), depth_bins_um=BINS, min_cells=200)
    assert "(290.0, 500.0]" not in set(few["query"])  # 150 deep cells: too few
    assert {"n_train", "n_test"} <= set(few.columns) and (few["n_train"] >= 200).all()
    no_depth = ProbeBatchFeatures(embeddings=features.embeddings)
    with pytest.raises(ValueError, match="spatial_unit_um"):
        redundancy_table(no_depth, features, depth_bins_um=BINS)


# --------------------------------------------------------------------------- attribution shares


def _prediction(parts, hidden=None, depth=None, true=None):
    whole = sum(parts.values())
    n, g = whole.shape
    hidden = np.ones((n, g), dtype=bool) if hidden is None else hidden
    return HeadPredictions(
        heads={"global": whole},
        true=np.zeros((n, g)) if true is None else true,
        hidden=hidden,
        depth_um=depth,
        parts={"global": parts},
    )


def _shares(table, column="cov_share", group="all", depth="all"):
    rows = table[(table["group"] == group) & (table["depth_um"] == depth)]
    return rows.set_index("part")[column]


def test_the_shares_add_up_to_one_and_a_part_against_the_others_is_negative():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 5))
    genes = list("abcde")
    small = attribution_shares(_prediction({"a": a, "b": 0.05 * rng.normal(size=a.shape)}), genes)
    shares = _shares(small)
    assert shares.sum() == pytest.approx(1.0) and shares["a"] > 0.99
    against = attribution_shares(_prediction({"a": a, "b": -0.5 * a}), genes)  # the whole is half of a
    assert _shares(against)["a"] == pytest.approx(2.0) and _shares(against)["b"] == pytest.approx(-1.0)
    assert _shares(against, "norm_ratio")["a"] == pytest.approx(2.0)  # parts that cancel sum to more than 1 in norm
    assert _shares(against, "norm_ratio")["b"] == pytest.approx(1.0)


def test_a_constant_offset_and_the_visible_entries_change_nothing():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(300, 4)), rng.normal(size=(300, 4))
    hidden = np.ones((300, 4), dtype=bool)
    hidden[:100] = False  # a third of the cells shows everything
    base = _shares(attribution_shares(_prediction({"a": a, "b": b}, hidden), list("abcd")))
    shifted = a + 100.0
    shifted[:100] += 1e6 * rng.normal(size=(100, 4))  # wild values where nothing is hidden
    again = _shares(attribution_shares(_prediction({"a": shifted, "b": b}, hidden), list("abcd")))
    assert again.to_dict() == pytest.approx(base.to_dict())


def test_the_visible_entries_of_a_cell_with_a_hidden_entry_are_not_part_of_the_share():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(300, 4)), rng.normal(size=(300, 4))
    hidden = np.zeros((300, 4), dtype=bool)
    hidden[:, :2] = True  # genes c and d are visible in every cell
    genes = list("abcd")
    base = _shares(attribution_shares(_prediction({"a": a, "b": b}, hidden), genes))
    wild_a = a.copy()
    wild_a[:, 2:] += 1e3 * rng.normal(size=(300, 2))
    assert _shares(attribution_shares(_prediction({"a": wild_a, "b": b}, hidden), genes)).to_dict() == pytest.approx(
        base.to_dict()
    )


def test_groups_and_depth_bins_pick_their_own_entries():
    rng = np.random.default_rng(2)
    n, genes = 300, list("abcd")
    a, b = np.zeros((n, 4)), np.zeros((n, 4))
    a[:, :2] = rng.normal(size=(n, 2))  # part a speaks for genes a, b only
    b[:, 2:] = rng.normal(size=(n, 2))  # part b for genes c, d
    depth = np.where(np.arange(n) < n // 2, 100.0, 400.0)
    table = attribution_shares(
        _prediction({"a": a, "b": b}, depth=depth),
        genes,
        gene_groups={"first": ["a", "b"], "second": ["c", "d"]},
        depth_bins_um=BINS,
    )
    assert _shares(table, group="first")["a"] == pytest.approx(1.0) and _shares(table, group="first")[
        "b"
    ] == pytest.approx(0.0)
    assert _shares(table, group="second")["b"] == pytest.approx(1.0)
    assert 0.2 < _shares(table)["a"] < 0.8  # over all genes the two parts split the variance
    assert set(table["depth_um"]) == {"all", "(0.0, 150.0]", "(290.0, 500.0]"}
    assert {"head", "part", "group", "depth_um", "n_hidden", "norm_ratio", "cov_share"} == set(table.columns)


def test_a_head_without_parts_has_no_shares_and_blending_keeps_the_parts():
    rng = np.random.default_rng(3)
    pred = _prediction({"a": rng.normal(size=(50, 3)), "b": rng.normal(size=(50, 3))})
    bare = HeadPredictions(heads=pred.heads, true=pred.true, hidden=pred.hidden)
    assert attribution_shares(bare, list("abc")).empty
    blended = blend_heads(
        HeadPredictions(
            heads=pred.heads | {"ref_x": pred.heads["global"]}, true=pred.true, hidden=pred.hidden, parts=pred.parts
        ),
        "ref_x",
    )
    assert blended.parts is pred.parts


def test_depth_bins_need_the_depth():
    pred = _prediction({"a": np.ones((10, 2)), "b": np.arange(20.0).reshape(10, 2)})
    with pytest.raises(ValueError, match="depth"):
        attribution_shares(pred, ["a", "b"], depth_bins_um=BINS)


# --------------------------------------------------------------------------- the parts, through the modules

G, E = 6, 16


def _global_module(skip, seed=0):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.global_component.parameters.type_gex_embedding = "Linear"
    cfg.model.decoder.own_cell_skip = skip
    torch.manual_seed(seed)
    module = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=1.0,
        mask_strategy="gene",
        type_gex_embedding="Linear",
    ).eval()
    if skip:
        module.own_skip_weight.data = torch.randn(G, E, generator=torch.Generator().manual_seed(1))
    return module


def _batch(n=14, seed=0):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.stack([torch.arange(n) % 4, torch.arange(n) // 4], dim=1).float() * 100.0
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=torch.zeros((2, 0), dtype=torch.long), pos=pos)
    data.gene_mask = torch.zeros(n, G, dtype=torch.bool)
    data.gene_mask[:, [0, 1]] = True
    data.mask = torch.ones(n, dtype=torch.bool)
    batch = Batch.from_data_list([data])
    batch.obs_names = torch.arange(n)
    return batch


def test_a_global_only_model_hands_over_its_read_out_and_its_skip_and_they_add_up():
    skip, plain = _global_module(True), _global_module(False)
    batch = _batch()
    with torch.no_grad():
        out = skip._common_step(batch, "regression", "node")
        assert plain._common_step(batch, "regression", "node").parts is None
    parts = out.parts["global"]
    assert set(parts) == {"context", "own_cell"}
    assert torch.equal(parts["context"] + parts["own_cell"], out.y_pred)
    with torch.no_grad():
        assert torch.equal(
            skip.predict(
                out.view.global_embedding,
                out.view.src_padding_mask,
                "node",
                _own(skip, batch),
                out.view.padded_node_idx,
            ),
            out.y_pred,
        )


def _own(module, batch):
    masked, _, _ = module._common_step_masking(batch)
    return module.gex_embedding(masked)


def test_collect_head_predictions_carries_the_parts_and_the_shares_follow():
    skip = _global_module(True)
    pred = collect_head_predictions(
        skip,
        [_batch(seed=0).get_example(0), _batch(seed=1).get_example(0)],
        prediction_task="regression",
        prediction_level="node",
        mask_strategy="gene",
        batch_size=1,
        device="cpu",
        spatial_unit_um=1.0,
    )
    assert set(pred.parts["global"]) == {"context", "own_cell"}
    assert np.allclose(
        pred.parts["global"]["context"] + pred.parts["global"]["own_cell"], pred.heads["global"], atol=1e-6
    )
    table = attribution_shares(pred, [f"g{k}" for k in range(G)])
    shares = _shares(table)
    assert (
        shares.sum() == pytest.approx(1.0) and shares["own_cell"] > 0.5
    )  # a random skip dominates an untrained read-out


def _dual(residual, seed=0):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder = True
    cfg.model.decoder.global_residual = residual
    torch.manual_seed(seed)
    return DualDecoderCombinedModule(
        cfg=cfg,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=1.0,
        mask_strategy="gene",
    ).eval()


def test_a_residual_dual_decoder_hands_over_its_local_and_its_global_part():
    batch = _batch()
    batch.edge_index = torch.tensor([list(range(13)), list(range(1, 14))])
    module = _dual(True)
    out = module._common_step(batch, "regression", "node")  # with gradients: a detached part shows as such
    with torch.no_grad():
        assert _dual(False)._common_step(batch, "regression", "node").parts is None
    parts = out.parts["global"]
    n = len(out.y_pred) // 2
    assert set(parts) == {"local", "global"}
    assert torch.allclose(parts["local"] + parts["global"], out.y_pred[n:], atol=1e-6)
    assert not parts["local"].requires_grad  # the residual head does not train the local prediction
    assert parts["global"].requires_grad and out.y_pred[:n].requires_grad
    assert "local" not in out.parts  # the local head is one part: nothing to attribute


# --------------------------------------------------------------------------- the report


def test_the_three_tables_take_their_compact_form_and_reach_the_summary():
    redundancy = pd.DataFrame(
        {
            "query": ["visible", "visible", "(0.0, 150.0]"],
            "target": ["global", "global", "global"],
            "source": ["local", "smoother", "local"],
            "r2": [0.9, 0.5, 0.7],
            "n_train": 10,
            "n_test": 10,
        }
    )
    shares = pd.DataFrame(
        {
            "head": "global",
            "part": ["context", "own_cell", "context", "own_cell"],
            "group": "all",
            "depth_um": ["all", "all", "(0.0, 150.0]", "(0.0, 150.0]"],
            "n_hidden": 10,
            "norm_ratio": [1.0, 0.3, 0.9, 0.5],
            "cov_share": [0.8, 0.2, 0.6, 0.4],
        }
    )
    compact = report.compact_tables({"redundancy": redundancy, "attribution_shares": shares})
    assert compact["redundancy"].loc["global <- local", "visible"] == 0.9
    assert list(compact["redundancy"].index) == ["global <- local", "global <- smoother"]
    assert compact["attribution_shares"].loc[("global", "own_cell", "all"), "(0.0, 150.0]"] == 0.4
    assert compact["attribution_norms"].loc[("global", "own_cell", "all"), "all"] == 0.3
    sections = {(arm, name): table for arm in ("a", "b") for name, table in compact.items()}
    titles = [title for title, _ in report.summary_tables(sections, ["a", "b"])]
    assert any(t.startswith("redundancy:") for t in titles) and any(t.startswith("attribution:") for t in titles)
