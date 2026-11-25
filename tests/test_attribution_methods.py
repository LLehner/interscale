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
    decomposition_by_distance,
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


def test_a6_measures_run_on_a_single_graph_without_a_batch_vector(stream):
    """The synth attribution job (2026-11-24) handed ``sensitivity_n`` a graph without ``batch``: ``attribute`` set it on its own
    device copy only, so ``pad_batch`` read ``None``. Every entry point fills it in, the same numbers as with it."""
    module, graph, labels = stream
    bare = graph.clone()
    del bare.batch
    receivers = np.array([9])
    w = gene_targets([0, 3], 1, G)
    cand = near_candidates(graph.pos.numpy(), receivers, np.ones(graph.num_nodes, bool), k=6)
    occ = attribute(module, graph, receivers, w, "occlusion", senders=cand[0])
    with_batch = sensitivity_n(module, graph, receivers, w, {"occlusion": occ}, cand, ns=(1,), draws=6, seed=1)
    without = sensitivity_n(module, bare.clone(), receivers, w, {"occlusion": occ}, cand, ns=(1,), draws=6, seed=1)
    np.testing.assert_allclose(without["occlusion"][1], with_batch["occlusion"][1])
    inter = pairwise_interactions(module, bare.clone(), receivers, w, cand, pairs=2, seed=2)
    assert inter["interaction"].shape == (1, 2)


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


def test_the_expression_path_equals_the_token_path_for_a_linear_embedding_and_splits_over_genes(stream):
    """With a linear expression embedding the token is W x + b, so the straight path from the type-mean expression to x is the token path from
    the type-mean token: integrated gradients agree cell by cell. The per-gene parts add up to each cell's contribution (near senders) and to
    the total over cells (gene totals)."""
    module, graph, labels = stream
    receivers = np.array([3, 12, 30])
    w = gene_targets([0, 2], 3, G)
    tok = attribute(module, graph, receivers, w, "ig", labels=labels, reference="type_mean", steps=6, retry_steps=6)
    cand = near_candidates(graph.pos.numpy(), receivers, np.ones(graph.num_nodes, bool), k=5)
    ex = attribute(module, graph, receivers, w, "ig", labels=labels, reference="type_mean", steps=6, retry_steps=6, space="expression",
                   gene_senders=cand)  # fmt: skip
    np.testing.assert_allclose(ex.contribution, tok.contribution, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(ex.change, tok.change, rtol=1e-4, atol=1e-6)  # float32 forward passes
    np.testing.assert_allclose(ex.extra["gene_total"].sum(1), np.nansum(ex.contribution, axis=1)
                               + ex.contribution[np.arange(3), receivers], rtol=1e-4, atol=1e-6)  # fmt: skip
    for k in range(3):
        for j, c in enumerate(cand[k]):
            if c >= 0 and c != receivers[k]:
                assert ex.extra["gene_near"][k, j].sum() == pytest.approx(float(ex.contribution[k, c]), rel=1e-4, abs=1e-6)
    with pytest.raises(ValueError, match="expression path"):
        attribute(module, graph, receivers, w, "deeplift", labels=labels, space="expression")


def test_reference_expression_reads_visible_entries_only():
    from interscale.evaluation.attribution import reference_expression

    x = torch.tensor([[1.0, 10.0], [3.0, -1.0], [5.0, 30.0], [7.0, 40.0]], dtype=torch.float64)
    visible = np.array([[True, True], [True, False], [True, True], [True, True]])
    labels = np.array(["a|n1", "a|n1", "a|n1", "b|n1"])
    ref = reference_expression(x, visible, labels, "type_mean")
    np.testing.assert_allclose(ref[:3].numpy(), [[3.0, 20.0]] * 3)  # the hidden -1 is not in the mean
    np.testing.assert_allclose(ref[3].numpy(), [7.0, 40.0])
    sampled = reference_expression(x, visible, labels, "sampled", rng=np.random.default_rng(0))
    assert not torch.equal(sampled[0], x[0])
    # a donor's hidden entry is never handed on: where it is hidden the cell keeps its own value
    for i in range(3):
        assert sampled[i, 1].item() != -1.0 or i == 1


def _prototype(tmp_path):
    """A toy prototype: CombinedModel (GCN -> dense transformer, dual decoder), gene masking, two genes hidden in every cell of the graph."""
    import scanpy as sc

    from interscale.config import load_config
    from interscale.model.combined_model import CombinedModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(1)
    per = 60
    adata = AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 300, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5 = tmp_path / "proto.h5ad"
    adata.write_h5ad(h5)
    path = tmp_path / "proto.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  local_component:\n    name: GCN\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 256\n      num_layers: 2\n      n_heads: 2\n      long_range_attention: False\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 1\n"
        f"dataset:\n  h5ad_data: {h5}\n  name: proto\n  prediction_task: regression\n  prediction_level: node\n  layer_key: log1p_norm\n"
        "  sample_key: ['sample']\n  mask_strategy: gene\n  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 1\n"
        "  spatial_neigbors_kwargs:\n    radius: 60\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    data = sc.read_h5ad(cfg.dataset.h5ad_data)
    CombinedModel._setup_anndata(adata=data, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None)
    torch.manual_seed(0)
    net = CombinedModel(data, cfg=cfg)
    with torch.no_grad():
        for p in net.module.parameters():
            p.add_(0.2 * torch.randn_like(p))
    _, _, test = prepare_geome_dataset(data, cfg)[0]
    graph = test[0].clone()
    gene_mask = torch.zeros(graph.num_nodes, G, dtype=torch.bool)
    gene_mask[:, [1, 4]] = True
    graph.gene_mask, graph.mask = gene_mask, gene_mask.any(dim=1)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return net.module.eval(), graph


def test_the_expression_path_reads_the_prototype_and_never_moves_a_hidden_entry(tmp_path):
    module, graph = _prototype(tmp_path)
    receivers = np.array([2, 20, 41])
    w = gene_targets([1, 4], 3, G)  # the hidden genes: the prototype's targets
    labels = np.array(["a"] * graph.num_nodes)
    coarse = attribute(module, graph, receivers, w, "ig", labels=labels, reference="type_mean", steps=2, retry_steps=2, space="expression")
    fine = attribute(module, graph, receivers, w, "ig", labels=labels, reference="type_mean", steps=24, retry_steps=24, space="expression")
    assert np.nanmedian(fine.gap) < 0.05 and np.nanmedian(fine.gap) <= np.nanmedian(coarse.gap) + 1e-9
    assert np.all(fine.extra["gene_total"][:, [1, 4]] == 0.0)  # hidden in every cell: no sender reads them
    assert np.all(fine.contribution[np.arange(3), receivers] == 0.0)
    before = graph.x.clone()
    attribute(module, graph, receivers, w, "grad_input", labels=labels, reference="graph_mean", space="expression")
    assert torch.equal(graph.x, before)


# ----------------------------------------------------------------------------------------------- evaluation.attention_rows


def test_readout_rows_equal_the_attention_method(ir):
    module, graph, labels = ir
    receivers = np.array([3, 17, 40])
    from interscale.evaluation.attention_rows import attention_rows

    rows = attention_rows(module, graph, receivers, layer="mean", eval_step=True)
    att = attribute(module, graph, receivers, gene_targets([0], 3, 6), "attention", eval_step=True)
    np.testing.assert_allclose(rows, np.nan_to_num(att.contribution), rtol=1e-5, atol=1e-6)
    last = attention_rows(module, graph, receivers, layer=-1)
    assert np.all(last >= 0) and np.all(last.sum(1) <= 1 + 1e-5)


def test_dense_context_query_rows_never_weigh_the_own_cell(stream):
    module, graph, _ = stream
    receivers = np.array([0, 9, 33])
    from interscale.evaluation.attention_rows import attention_rows

    rows = attention_rows(module, graph, receivers, layer=-1)
    assert np.all(rows >= 0) and np.all(rows.sum(1) <= 1 + 1e-5) and np.all(rows.sum(1) > 0.5)
    assert np.all(rows[np.arange(3), receivers] < 1e-7)


def test_prototype_attention_rows(tmp_path):
    module, graph = _prototype(tmp_path)
    from interscale.evaluation.attention_rows import attention_rows

    rows = attention_rows(module, graph, np.array([1, 5]), layer="mean", eval_step=True)
    assert rows.shape == (2, graph.num_nodes) and np.all(rows >= 0) and np.all(rows.sum(1) <= 1 + 1e-5)


def test_attention_row_summary_and_agreement_by_hand():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    uniform = np.array([[0.0, 0.25, 0.25, 0.25, 0.25]])
    from interscale.evaluation.attention_rows import row_agreement, row_summary

    s = row_summary(uniform, pos, [0]).iloc[0]
    assert s.entropy == pytest.approx(1.0) and s.eff_keys == pytest.approx(4.0) and s.mean_dist_um == pytest.approx(2.5)
    peaked = np.array([[0.0, 0.9, 0.05, 0.05, 0.0]])
    assert row_summary(peaked, pos, [0]).iloc[0].max_share == pytest.approx(0.9)
    same = row_agreement(peaked, peaked, [0], top=3).iloc[0]
    assert same.spearman == pytest.approx(1.0) and same.js == pytest.approx(0.0, abs=1e-12) and same.top_overlap == pytest.approx(1.0)
    assert row_agreement(peaked, uniform, [0], top=3).iloc[0].js > 0.05


def test_subtract_fields_cancels_what_the_twin_shares():
    from interscale.evaluation.attribution import map_fields, subtract_fields

    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 300, size=(200, 2))
    layout = np.where(np.arange(200) < 20, -1.0, 0.05)  # the discs: sinks in both maps
    signal = np.where(np.arange(200) % 7 == 0, 1.0, 0.0)  # what training added
    trained = map_fields(pos, 100.0 * (layout + signal), np.zeros(200), np.ones(200, np.int8))
    twin = map_fields(pos, 0.01 * layout, np.zeros(200), np.ones(200, np.int8))
    diff = subtract_fields(trained, twin)
    net = diff["flow_out"] - diff["flow_in"]
    assert np.corrcoef(net, signal)[0, 1] > 0.9 and abs(np.corrcoef(net, layout)[0, 1]) < np.corrcoef(trained["flow_out"], layout)[0, 1]
    with pytest.raises(ValueError, match="same cells"):
        subtract_fields(trained, map_fields(pos[::-1], layout, np.zeros(200), np.ones(200, np.int8)))


def test_the_distance_split_of_the_decomposition_adds_up_to_the_one_hot_decomposition_of_every_gene(ir):
    module, graph, labels = ir
    receivers = np.array([3, 17, 40, 55])
    pos = graph.pos[:, :2].double().numpy()
    radii = [0.0, 60.0, 150.0, 1e9]
    x_before = graph.x.clone()
    split = decomposition_by_distance(module, graph, receivers, radii, eval_step=True)
    assert split["within"].shape == (4, 4, G) and split["total"].shape == (4, G)
    for g in range(G):
        w = gene_targets([g], len(receivers), G)
        attr = attribute(module, graph, receivers, w, "decomposition", eval_step=True)
        np.testing.assert_allclose(split["total"][:, g], attr.contribution.sum(axis=1), rtol=1e-4, atol=1e-5)
        for k, r in enumerate(radii):  # the senders within each radius, by hand from the per-sender shares
            near = np.array([attr.contribution[i][np.linalg.norm(pos - pos[q], axis=1) <= r].sum() for i, q in enumerate(receivers)])
            np.testing.assert_allclose(split["within"][:, k, g], near, rtol=1e-4, atol=1e-5)
    # radius 0 holds only the receiver itself, which is left out; an infinite radius holds every sender
    np.testing.assert_allclose(split["within"][:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(split["within"][:, -1], split["total"], rtol=1e-6, atol=1e-7)
    assert torch.equal(graph.x, x_before) and not module.training
    with pytest.raises(ValueError, match="ascending"):
        decomposition_by_distance(module, graph, receivers, [100.0, 50.0])


def test_distance_flows_read_the_same_in_batches_as_one_copy_at_a_time(ir):
    from interscale.evaluation.gene_ranking import distance_flows, hide_genes

    module, graph, labels = ir
    hidden = hide_genes([graph], [0, 1])[0]
    hidden.batch = graph.batch
    saved, module.mask_strategy = module.mask_strategy, "gene"  # folds are read as gene masks, as the cluster readouts do
    try:
        one = distance_flows(module, [hidden], [labels], [f"g{k}" for k in range(G)], [0.0, 80.0], per_type=6, seed=1, batch_size=1)
        many = distance_flows(module, [hidden], [labels], [f"g{k}" for k in range(G)], [0.0, 80.0], per_type=6, seed=1, batch_size=4)
    finally:
        module.mask_strategy = saved
    np.testing.assert_allclose(np.nan_to_num(one.drop), np.nan_to_num(many.drop), rtol=1e-5, atol=1e-7)
    assert np.isfinite(one.drop[0][:, :2]).any() and np.isnan(one.drop[0][:, 2:]).all()  # only the hidden genes are read
