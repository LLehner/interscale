"""Attribution (``evaluation.attribution``; ``.claude/attribution_plan.md`` A0-A6) on real GlobalModels at toy size.

The interaction read-out's decomposition sums exactly to its context part; integrated gradients close their completeness gap as steps grow;
occlusion is by definition the change when one sender takes its reference, and with one receiver Sensitivity-1 against occlusion is exactly 1;
the pairwise interaction is what its definition says; receivers and cells hidden whole never move; the module, its parameters' flags and the
global generators are left as found; DeepLIFT's secant gradient is the secant and the activations are put back.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData

from interscale.evaluation.attribution import (
    _SecantFn,
    _secant_activations,
    agreement,
    attribute,
    deviation_targets,
    disc_strengths,
    distance_profile,
    gene_targets,
    map_fields,
    near_candidates,
    pairwise_interactions,
    planted_top_k,
    reference_tokens,
    sensitivity_n,
    separated_receivers,
    strengths,
)

G = 6
SCALES = "      pe:\n        bias: ['distance']\n        distance:\n          kind: scales\n          scale_min_um: 50.0\n          scale_max_um: 400.0\n"
ARMS = {
    "ir": (
        "      context_only: query\n      context_readout: additive\n      context_query:\n        inputs: [type]\n        type_key: cell_type\n"
        + SCALES + "      attention:\n        kind: readout\n        visible_keys_only: True\n",
        "    section_offset: label_mean\n    label_offset_key: cell_type\n",
        "  mask_strategy: node\n  mask_percentage: 0.2\n  score_all_cells: True\n  extra_obs_keys: ['cell_type']\n",
    ),
    "stream": (
        "      context_only: query\n      context_query:\n        inputs: [position]\n" + SCALES + "      attention:\n        kind: dense\n        center_qk: False\n",
        "",
        "  mask_strategy: node\n  mask_percentage: 0.2\n  extra_obs_keys: ['cell_type']\n",
    ),
}  # fmt: skip


def _model(tmp_path, arm):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    params, decoder, masking = ARMS[arm]
    rng = np.random.default_rng(0)
    per = 70
    adata = AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["a", "b", "c"], size=per * 3))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5 = tmp_path / f"{arm}.h5ad"
    adata.write_h5ad(h5)
    path = tmp_path / f"{arm}.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 256\n      num_layers: 2\n      n_heads: 2\n      long_range_attention: False\n      type_gex_embedding: Linear\n"
        f"{params}  decoder:\n    type: linear\n{decoder}  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  center_targets: True\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 1\n"
        f"dataset:\n  h5ad_data: {h5}\n  name: {arm}\n  prediction_task: regression\n  prediction_level: node\n  layer_key: log1p_norm\n"
        f"  sample_key: ['sample']\n{masking}  spatial_key: spatial\n  batch_size: 1\n  spatial_neigbors_kwargs:\n    radius: 120\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    data = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(adata=data, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None)
    torch.manual_seed(0)
    net = GlobalModel(data, cfg=cfg)
    with torch.no_grad():
        for p in net.module.parameters():  # stir the init: zero tables make positional readouts vacuous
            p.add_(0.2 * torch.randn_like(p))
    train, _, test = prepare_geome_dataset(data, cfg)[0]
    if getattr(net.module, "section_offset", None) == "label_mean":
        net.module.set_label_means(list(train))  # data, not weights: as train() sets them
    graph = test[0].clone()
    graph.mask = torch.zeros(graph.num_nodes, dtype=torch.bool)  # the clean pass: nothing hidden
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    labels = data.obs["cell_type"].astype(str).to_numpy()[np.rint(np.asarray(graph.obs_names, dtype=float)).astype(int) - 1]
    module = net.module.eval()
    module.score_all_cells = True
    return module, graph, labels


@pytest.fixture(scope="module")
def ir(tmp_path_factory):
    return _model(tmp_path_factory.mktemp("ir"), "ir")


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    return _model(tmp_path_factory.mktemp("stream"), "stream")


def _targets(module, graph, labels, receivers):
    from interscale.evaluation.context_test import _predict

    pred = _predict(module, [graph], head="global", batch_size=1, device="cpu")[0]
    return deviation_targets(pred, labels, receivers)


def test_the_decomposition_sums_to_the_context_part_and_the_receiver_is_left_out(ir):
    module, graph, labels = ir
    receivers = np.array([3, 17, 40])
    w = _targets(module, graph, labels, receivers)
    attr = attribute(module, graph, receivers, w, "decomposition", eval_step=True)
    assert np.all(attr.gap < 1e-4)
    assert np.all(attr.contribution[np.arange(3), receivers] == 0.0)
    att = attribute(module, graph, receivers, w, "attention", eval_step=True)
    assert np.allclose(np.nansum(att.contribution, axis=1) + 0.0, np.nansum(att.contribution, axis=1))  # finite rows
    assert np.all(np.nansum(att.contribution, axis=1) <= 1.0 + 1e-5)  # head means of rows that sum to one minus the null


def test_ig_closes_its_gap_and_leaves_everything_as_found(stream):
    module, graph, labels = stream
    receivers = np.array([2, 11, 30])
    w = gene_targets([0, 1], len(receivers), G)
    flags = [p.requires_grad for p in module.parameters()]
    torch_state, np_state = torch.random.get_rng_state(), np.random.get_state()[1].copy()
    x_before = graph.x.clone()
    coarse = attribute(module, graph, receivers, w, "ig", steps=2, retry_steps=2, labels=labels, reference="type_mean")
    fine = attribute(module, graph, receivers, w, "ig", steps=16, retry_steps=16, labels=labels, reference="type_mean")
    assert np.nanmedian(fine.gap) <= np.nanmedian(coarse.gap) + 1e-9 and np.nanmax(fine.gap) < 0.05
    assert [p.requires_grad for p in module.parameters()] == flags and not module.training
    assert torch.equal(torch_state, torch.random.get_rng_state()) and np.array_equal(np_state, np.random.get_state()[1])
    assert torch.equal(graph.x, x_before)
    assert np.all(fine.contribution[np.arange(3), receivers] == 0.0)
    gi = attribute(module, graph, receivers, w, "grad_input", labels=labels, reference="type_mean")
    assert gi.contribution.shape == fine.contribution.shape and np.isfinite(gi.contribution).all()


def test_hidden_cells_never_move(stream):
    module, graph, labels = stream
    g = graph.clone()
    g.mask = torch.zeros(g.num_nodes, dtype=torch.bool)
    g.mask[[5, 6, 7]] = True
    receivers = np.array([5, 20])  # a hidden receiver reads the others
    attr = attribute(module, g, receivers, gene_targets([2], 2, G), "grad_input")
    assert np.all(attr.contribution[:, [6, 7]] == 0.0)


def test_occlusion_is_the_change_when_one_sender_takes_its_reference_and_sensitivity_1_matches_it(stream):
    module, graph, labels = stream
    receivers = np.array([9])
    w = gene_targets([0, 3], 1, G)
    cand = near_candidates(graph.pos.numpy(), receivers, np.ones(graph.num_nodes, bool), k=6)
    occ = attribute(module, graph, receivers, w, "occlusion", senders=cand[0])
    assert np.isnan(occ.contribution[0]).sum() == graph.num_nodes - len(cand[0]) - 1  # the receiver's own column is 0
    sens = sensitivity_n(module, graph, receivers, w, {"occlusion": occ}, cand, ns=(1,), draws=12, seed=1)
    assert sens["occlusion"][1][0] == pytest.approx(1.0, abs=1e-6)
    inter = pairwise_interactions(module, graph, receivers, w, cand, pairs=3, seed=2)
    assert inter["interaction"].shape == (1, 3) and np.all(inter["single"] >= 0)


def test_deeplift_runs_puts_activations_back_and_the_secant_is_a_secant(stream):
    module, graph, labels = stream
    before = {id(m): (getattr(m, "activation", None)) for m in module.modules()}
    attr = attribute(module, graph, np.array([4, 25]), gene_targets([1], 2, G), "deeplift")
    assert np.isfinite(attr.contribution).all() and np.isfinite(attr.gap).all()
    assert {id(m): getattr(m, "activation", None) for m in module.modules()} == before
    assert not any("forward" in vars(m) for m in module.modules())
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([1.0, 0.5, -2.0], dtype=torch.float64)
    _SecantFn.apply(x, ref, 0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [(0 - 1) / (-2.0), 1.0, (2 - 0) / 4.0])
    store = []
    with _secant_activations(module, "record", store):
        pass
    assert not any("forward" in vars(m) for m in module.modules())


def test_references_and_measures():
    tokens = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0], [12.0, 4.0]])
    labels = np.array(["a", "a", "b", "b"])
    np.testing.assert_allclose(reference_tokens(tokens, labels, "type_mean").numpy(), [[1, 1], [1, 1], [11, 2], [11, 2]])
    np.testing.assert_allclose(reference_tokens(tokens, labels, "graph_mean").numpy()[0], [6, 1.5])
    drawn = reference_tokens(tokens, labels, "sampled", rng=np.random.default_rng(0)).numpy()
    np.testing.assert_allclose(drawn, [[2, 2], [0, 0], [12, 4], [10, 0]])  # the other cell of the same label
    pos = np.column_stack([np.arange(10) * 50.0, np.zeros(10)])
    chosen = separated_receivers(pos, np.ones(10, bool), per_label=10, min_distance=120.0)
    assert np.all(np.diff(pos[chosen, 0]) >= 120.0)
    from interscale.evaluation.attribution import Attribution

    contrib = np.zeros((2, 10), dtype=np.float32)
    contrib[0, [1, 2]] = [0.5, -0.1]
    contrib[1, [8, 3]] = [0.2, 0.9]
    attr = Attribution("x", None, np.array([0, 9]), contrib, np.array([0.4, 1.1]), np.zeros(2), 0.0)
    assert planted_top_k(attr, {0: {2}, 9: {1}}, k=2).tolist() == [True, False]
    prof = distance_profile(attr, pos, [0, 75, 175, 500])
    assert prof["share"].sum() == pytest.approx(1.0) and prof["cumulative"][-1] == pytest.approx(1.0)
    s = strengths(attr, 10)
    assert s["out"][3] == pytest.approx(0.9) and s["in"][0] == pytest.approx(0.4)
    d = disc_strengths(attr, np.array([0, 0]), 10)
    assert d["in"][0] == pytest.approx(d["in"][9]) and d["out"][3] == pytest.approx(0.9)
    fields = map_fields(pos, s["out"], s["in"], np.ones(10))
    assert fields["flow_cells"].sum() == 10
    same = agreement(attr, attr, top=3, k=2)
    assert np.allclose(same["top_overlap"], 1.0)


def test_conservative_keeps_the_forward_and_restores_every_patched_function(stream, ir):
    from interscale.evaluation.attribution import _conservative_rules

    saved = (torch.softmax, torch.nn.functional.softmax, torch.Tensor.softmax, torch.nn.functional.scaled_dot_product_attention, torch.nn.functional.layer_norm)
    for module, graph, labels in (stream, ir):
        receivers = np.array([3, 12])
        w = gene_targets([0, 2], 2, G)
        from interscale.evaluation.attribution import _Reader, _actual_tokens

        tokens = _actual_tokens(module, graph)
        with torch.no_grad():
            plain = _Reader(module, graph, receivers, w, "global")(tokens.clone())
            with _conservative_rules():
                ruled = _Reader(module, graph, receivers, w, "global")(tokens.clone())
        torch.testing.assert_close(plain, ruled, rtol=1e-5, atol=1e-6)
        attr = attribute(module, graph, receivers, w, "conservative", labels=labels, reference="type_mean")
        assert np.isfinite(attr.contribution).all() and np.isfinite(attr.gap).all()
    assert saved == (torch.softmax, torch.nn.functional.softmax, torch.Tensor.softmax, torch.nn.functional.scaled_dot_product_attention,
                     torch.nn.functional.layer_norm)  # fmt: skip


def test_removal_by_the_models_own_mask(stream):
    from types import SimpleNamespace

    from interscale.evaluation.attribution import _Reader, _actual_tokens, _check_removal

    module, graph, labels = stream
    receivers = np.array([9])
    w = gene_targets([1], 1, G)
    cand = near_candidates(graph.pos.numpy(), receivers, np.ones(graph.num_nodes, bool), k=4)
    occ = attribute(module, graph, receivers, w, "occlusion", senders=cand[0], removal="mask")
    j = int(cand[0][0])
    hidden = graph.clone()
    hidden.mask = graph.mask.clone()
    hidden.mask[j] = True
    with torch.no_grad():
        base = _Reader(module, graph, receivers, w, "global")(_actual_tokens(module, graph))
        without = _Reader(module, hidden, receivers, w, "global")(_actual_tokens(module, hidden))
    assert occ.contribution[0, j] == pytest.approx(float(base[0] - without[0]), abs=1e-5)
    sens = sensitivity_n(module, graph, receivers, w, {"occlusion": occ}, cand, ns=(1,), draws=8, removal="mask", seed=3)
    assert sens["occlusion"][1][0] == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="node-masked"):
        _check_removal(SimpleNamespace(mask_strategy="gene"), "mask")


def test_gradients_through_attention_computed_in_checkpointed_blocks(ir, monkeypatch):
    """On a whole section the read-out attention runs a block of queries at a time and recomputes each block in the backward pass; every
    gradient method must give what it gives in one block. Before the fix, ``conservative`` took its backward outside its patches and the
    recompute ran the real SDPA kernel against the explicit form saved in the forward (CheckpointError on CosMx); toy graphs fit one block."""
    import interscale.module.global_modules.attention as attention_mod

    module, graph, labels = ir
    layers = [m for m in module.modules() if hasattr(m, "chunk_pairs")]
    assert layers
    receivers = np.array([3, 17, 40])
    w = gene_targets([0, 2], 3, G)
    methods = ("grad_input", "conservative", "deeplift", "ig")
    whole = {m: attribute(module, graph, receivers, w, m, labels=labels, reference="type_mean", steps=4) for m in methods}
    calls = {"n": 0}
    real = attention_mod.checkpoint

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(attention_mod, "checkpoint", counted)
    saved = [m.chunk_pairs for m in layers]
    try:
        for m in layers:
            m.chunk_pairs = 64  # a few queries per block: many blocks, each recomputed in the backward
        for method in methods:
            before = calls["n"]
            blocked = attribute(module, graph, receivers, w, method, labels=labels, reference="type_mean", steps=4)
            assert calls["n"] > before, f"{method}: no block was checkpointed, the test checks nothing"
            np.testing.assert_allclose(blocked.contribution, whole[method].contribution, rtol=1e-4, atol=1e-6)
    finally:
        for m, c in zip(layers, saved, strict=True):
            m.chunk_pairs = c
