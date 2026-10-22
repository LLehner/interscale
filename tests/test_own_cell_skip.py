"""The own-cell skip and the decoder types of a global-only model (``model.decoder``; roadmap.md, T1b / T2).

``own_cell_skip``: ``y = decoder(h_global) + W_o e_own``, where ``e_own`` is the cell's own token before the
transformer, from the masked input. Pinned here:

* **the switch** -- off by default, then no state; on, the model differs by the one parameter, which starts at
  zero, so every other weight is initialised as before and the first prediction is the model without it;
* **what it adds** -- exactly ``W_o e_own`` for the cells the transformer kept, in token order, also with pooled
  tokens (where the transformer's output is shared by a patch and the skip is the only per-cell part);
* **no leakage** -- a hidden entry never reaches a prediction through it, a visible one does;
* **inference** -- ``get_model_output`` decodes with it too;
* **refusals** -- a module without tokens of its own, graph-level prediction, a model that has no such part;
* **the decoder types** -- ``nonlinear`` and ``linear-lse`` train finite through a global-only model, and each
  decoder and the skip have a checkpoint name of their own;
* **the raw read** (``own_cell_skip_input: raw``) -- the skip reads the cell's visible genes instead of its token: the
  weight is ``[n_output, n_input]`` and the contribution has full rank where the token's has at most ``n_embed``;
  the same switch, leakage, inference, refusals and name checks as the token read.
"""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from interscale.module.base import gather_tokens
from interscale.tl.utils import get_model_filename_prefix

G, E = 6, 16
SPACING = 100.0


def _cfg(skip=False, decoder="linear", embedding="Linear", pool=None, read="embedding"):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.type_gex_embedding = 64, embedding
    params.pe.node = ["sinusoidal"]
    cfg.model.decoder.type, cfg.model.decoder.own_cell_skip = decoder, skip
    cfg.model.decoder.own_cell_skip_input = read
    cfg.dataset.name = "d"
    if pool is not None:
        params.pooling.kind, params.pooling.size_um = "grid", pool
        cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", 1.0
    return cfg


def _module(skip=False, decoder="linear", strategy="node", rate=0.3, pool=None, seed=0, read="embedding", embed=E):
    from interscale.module.base import GlobalModule

    torch.manual_seed(seed)
    return GlobalModule.from_config(
        _cfg(skip, decoder, pool=pool, read=read),
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=embed,
        decoder_type=decoder,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=rate,
        mask_strategy=strategy,
        type_gex_embedding="Linear",
    ).eval()


def _batch(n=12, seed=0, strategy="node"):
    gen = torch.Generator().manual_seed(seed)
    side = int(np.ceil(np.sqrt(n)))
    pos = torch.stack([torch.arange(n) % side, torch.arange(n) // side], dim=1).float() * SPACING
    d = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=torch.zeros((2, 0), dtype=torch.long), pos=pos)
    if strategy == "node":
        d.mask = torch.zeros(n, dtype=torch.bool)
        d.mask[:4] = True
    else:  # programme masking: genes 0 and 1 hidden in every cell
        d.gene_mask = torch.zeros(n, G, dtype=torch.bool)
        d.gene_mask[:, [0, 1]] = True
        d.mask = torch.ones(n, dtype=torch.bool)
    batch = Batch.from_data_list([d])
    batch.obs_names = torch.arange(n)
    return batch


def _own_tokens(module, batch):
    """The cells' own tokens as ``_common_step`` hands them to ``predict``, from the masked batch."""
    batch_masked, _, _ = module._common_step_masking(batch)
    hidden = batch_masked.mask.bool() if module.mask_strategy != "gene" else None
    return module.gex_embedding(batch_masked, hidden=hidden)


def _with_random_skip(seed=1, **kwargs):
    on = _module(skip=True, **kwargs)
    on.own_skip_weight.data = torch.randn(on.own_skip_weight.shape, generator=torch.Generator().manual_seed(seed))
    return on


# --------------------------------------------------------------------------- the switch


def test_off_by_default_adds_no_state():
    assert _cfg().model.decoder.own_cell_skip is False
    module = _module()
    assert module.own_cell_skip is False
    assert not any("own_skip" in key for key in module.state_dict())


def test_on_the_model_differs_by_one_zero_parameter_and_starts_as_without_it():
    off, on = _module(), _module(skip=True)
    extra = set(on.state_dict()) - set(off.state_dict())
    assert extra == {"own_skip_weight"} and set(off.state_dict()) <= set(on.state_dict())
    assert all(torch.equal(off.state_dict()[k], on.state_dict()[k]) for k in off.state_dict())  # same initial weights
    assert not on.own_skip_weight.any()
    batch = _batch()
    with torch.no_grad():
        first_off = off._common_step(batch, "regression", "node").y_pred
        first_on = on._common_step(batch, "regression", "node").y_pred
    assert torch.equal(first_off, first_on)


# --------------------------------------------------------------------------- what it adds


@pytest.mark.parametrize("strategy", ["node", "gene"])
def test_the_prediction_adds_the_cells_own_token_through_the_weight(strategy):
    rate = 0.3 if strategy == "node" else 1.0
    off = _module(strategy=strategy, rate=rate)
    on = _with_random_skip(strategy=strategy, rate=rate)
    batch = _batch(strategy=strategy)
    with torch.no_grad():
        out_off = off._common_step(batch, "regression", "node")
        out_on = on._common_step(batch, "regression", "node")
        own = _own_tokens(on, batch)
    index = out_on.view.padded_node_idx
    assert torch.equal(index, out_off.view.padded_node_idx)
    assert torch.allclose(out_on.y_pred - out_off.y_pred, F.linear(own[index], on.own_skip_weight), atol=1e-5)
    assert (out_on.y_pred - out_off.y_pred).abs().max() > 1e-3  # and it is not nothing


def test_the_weight_gets_a_gradient_and_the_tokens_still_train():
    on = _module(skip=True).train()
    out = on._common_step(_batch(), "regression", "node")
    (out.y_pred - out.y_true).pow(2).mean().backward()
    assert on.own_skip_weight.grad is not None and on.own_skip_weight.grad.abs().sum() > 0
    assert on.input_embedding.weight.grad.abs().sum() > 0  # the learned input layer reaches the prediction both ways


def test_pooled_tokens_keep_the_skip_per_cell():
    """A patch's cells share the transformer's output; the skip is what tells them apart."""
    off = _module(pool=3 * SPACING)
    on = _with_random_skip(pool=3 * SPACING)
    batch = _batch(n=16)
    with torch.no_grad():
        out_off = off._common_step(batch, "regression", "node")
        out_on = on._common_step(batch, "regression", "node")
        own = _own_tokens(on, batch)
    delta = out_on.y_pred - out_off.y_pred
    assert torch.allclose(delta, F.linear(own[out_on.view.padded_node_idx], on.own_skip_weight), atol=1e-5)
    patch = out_off.y_pred  # without the skip, the cells of a patch are one row
    pairs = [(i, j) for i in range(16) for j in range(i)]
    assert any(torch.equal(patch[i], patch[j]) and not torch.equal(delta[i], delta[j]) for i, j in pairs)


# --------------------------------------------------------------------------- no leakage


def test_no_hidden_entry_reaches_a_prediction_through_the_skip_and_a_visible_one_does():
    on = _with_random_skip(strategy="gene", rate=1.0)
    batch = _batch(strategy="gene")
    with torch.no_grad():
        before = on._common_step(batch, "regression", "node").y_pred
        hidden = batch.clone()
        hidden.x = batch.x.clone()
        hidden.x[:, [0, 1]] = torch.rand(batch.num_nodes, 2) * 50.0
        assert torch.equal(on._common_step(hidden, "regression", "node").y_pred, before)
        visible = batch.clone()
        visible.x = batch.x.clone()
        visible.x[5, 4] += 50.0
        assert not torch.allclose(on._common_step(visible, "regression", "node").y_pred[5], before[5])


# --------------------------------------------------------------------------- inference


@pytest.mark.parametrize("read", ["embedding", "raw"])
def test_inference_decodes_with_the_skip_too(monkeypatch, read):
    import pandas as pd
    from anndata import AnnData

    import interscale.model.global_model as global_model
    from interscale.model.global_model import GlobalModel

    n = 12
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.random((n, 2))
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    cfg = _cfg(skip=True, read=read)
    cfg.model.n_embed = E
    cfg.dataset.spatial_key = "spatial"
    cfg.model.global_component.parameters.max_seq_len = 8  # fewer tokens than cells: a subsample
    torch.manual_seed(0)
    model = GlobalModel(adata, cfg=cfg)
    model.module.eval()
    model.is_trained_ = True
    width = G if read == "raw" else E
    model.module.own_skip_weight.data = torch.randn(G, width, generator=torch.Generator().manual_seed(2))

    graph = Data(
        x=torch.tensor(adata.X),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        pos=torch.tensor(adata.obsm["spatial"], dtype=torch.float32) * SPACING,
        obs_names=torch.arange(1, n + 1),
    )
    monkeypatch.setattr(global_model, "prepare_a2d_dataset", lambda cfg: lambda adata: [[graph], None])
    monkeypatch.setattr(global_model, "attach_positional_inputs", lambda pyg, cfg: None)
    saved = {}
    monkeypatch.setattr(model, "save_evaluation_results", lambda adata, prefix, **out: saved.update(out) or adata)
    import random

    random.seed(0)
    model.get_model_output(adata.copy())
    found = saved["y_pred_global_df"].to_numpy(dtype=float)

    module = model.module
    tokens = module.input_embedding(graph.x)
    random.seed(0)  # the same subsample
    _, emb, padding, kept, _ = module.evaluate(graph.clone(), tokens)
    base = module.decoder(gather_tokens(emb, padding))
    own = graph.x.float() if read == "raw" else tokens  # what the skip reads: the genes, or the cell's token
    expected = (base + F.linear(own[torch.as_tensor(kept[0])], module.own_skip_weight)).detach().numpy()
    rows = ~np.isnan(found).any(axis=1)
    assert rows.sum() == 8
    assert np.allclose(found[rows], expected, atol=1e-5)
    assert not np.allclose(found[rows], base.detach().numpy(), atol=1e-3)  # the skip is in it


# --------------------------------------------------------------------------- refusals


def test_a_module_without_tokens_of_its_own_refuses_the_skip():
    from interscale.module.base import GlobalModule

    cfg = _cfg(skip=True)
    with pytest.raises(ValueError, match="GlobalModel"):
        GlobalModule.from_config(
            cfg, local_mask_hops=2, n_input=G, n_output=G, n_embed=E, decoder_type="linear", mask_percentage=0.3
        )


def test_predict_refuses_to_run_without_the_tokens_rather_than_predict_as_if_without_the_skip():
    on = _module(skip=True)
    with torch.no_grad():
        out = on._common_step(_batch(), "regression", "node")
    with pytest.raises(ValueError, match="own_embedding"):
        on.predict(out.view.global_embedding, out.view.src_padding_mask, "node")


def test_a_graph_level_prediction_has_no_cell_to_add():
    on = _module(skip=True)
    with torch.no_grad():
        out = on._common_step(_batch(), "regression", "node")
    with pytest.raises(ValueError, match="per-cell"):
        on.predict(out.view.global_embedding, out.view.src_padding_mask, "graph")


def test_the_config_wants_a_node_level_prediction():
    from interscale.config import _validate_own_cell_skip

    cfg = _cfg(skip=True)
    cfg.dataset.prediction_level = "node"
    _validate_own_cell_skip(cfg)
    cfg.dataset.prediction_level = "graph"
    with pytest.raises(ValueError, match="node"):
        _validate_own_cell_skip(cfg)


def test_a_model_without_such_a_part_refuses_to_train_with_the_option():
    from interscale.train._training import NodeMaskingTrainingPlan

    local_only = types.SimpleNamespace(_cfg=_cfg(skip=True), module=types.SimpleNamespace())
    with pytest.raises(ValueError, match="own_cell_skip"):
        NodeMaskingTrainingPlan._refuse_unsupported_options(local_only)
    has_it = types.SimpleNamespace(_cfg=_cfg(skip=True), module=types.SimpleNamespace(own_cell_skip=True))
    NodeMaskingTrainingPlan._refuse_unsupported_options(has_it)
    plain = types.SimpleNamespace(_cfg=_cfg(), module=types.SimpleNamespace())
    NodeMaskingTrainingPlan._refuse_unsupported_options(plain)


def test_train_itself_refuses_a_local_model_with_the_skip():
    import pandas as pd
    from anndata import AnnData

    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.model.local_model import LocalModel

    rng = np.random.default_rng(0)
    n = 12
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.random((n, 2))
    LocalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    cfg = _cfg(skip=True)
    cfg.model.local_component.name = "GCN"
    cfg = get_local_component_cfg(cfg, "GCN")
    model = LocalModel(adata, cfg=cfg)
    with pytest.raises(ValueError, match="own_cell_skip"):
        model.train(max_epochs=1, wandb_use=False)


# --------------------------------------------------------------------------- decoder types


@pytest.mark.parametrize("decoder", ["linear", "nonlinear", "linear-lse"])
@pytest.mark.parametrize("skip", [False, True])
def test_every_decoder_trains_finite_through_a_global_only_model(decoder, skip):
    module = _module(skip=skip, decoder=decoder).train()
    out = module._common_step(_batch(), "regression", "node")
    assert torch.isfinite(out.y_pred).all()
    (out.y_pred - out.y_true).pow(2).mean().backward()
    grads = [p.grad for p in module.decoder.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_each_decoder_and_the_skip_have_a_checkpoint_name_of_their_own():
    names = {
        (decoder, skip): get_model_filename_prefix(_cfg(skip, decoder), False, True)
        for decoder in ("linear", "nonlinear", "linear-lse")
        for skip in (False, True)
    }
    assert len(set(names.values())) == 6
    assert "dec-" not in names[("linear", False)]  # the default adds nothing: every earlier name is unchanged
    assert "dec-linear-skip_" in names[("linear", True)] and "dec-nonlinear_" in names[("nonlinear", False)]
    assert "dec-linear-lse-skip_" in names[("linear-lse", True)]
    # The combined and the local-only names tell them apart too.
    for local, glob in ((True, True), (True, False)):
        nonlinear = get_model_filename_prefix(_cfg(False, "nonlinear"), local, glob)
        assert nonlinear != get_model_filename_prefix(_cfg(), local, glob)


# --------------------------------------------------------------------------- the raw read


def _raw_inputs(module, batch):
    """The genes the raw skip reads, as ``_common_step`` hands them to ``predict``: the masked batch's expression."""
    batch_masked, _, _ = module._common_step_masking(batch)
    return module.own_cell_input(batch_masked, module.gex_embedding(batch_masked)), batch_masked


def test_the_raw_read_is_off_by_default_and_gives_the_weight_the_width_of_the_genes():
    assert _cfg().model.decoder.own_cell_skip_input == "embedding"
    assert _module(skip=True).own_skip_weight.shape == (G, E)  # the token's width
    raw = _module(skip=True, read="raw")
    assert raw.own_skip_weight.shape == (G, G) and raw.own_cell_skip_input == "raw"
    assert _module().own_cell_skip_input == "embedding"


def test_the_raw_read_differs_by_one_zero_parameter_and_starts_as_without_it():
    off, on = _module(), _module(skip=True, read="raw")
    assert set(on.state_dict()) - set(off.state_dict()) == {"own_skip_weight"}
    assert all(torch.equal(off.state_dict()[k], on.state_dict()[k]) for k in off.state_dict())  # same initial weights
    assert not on.own_skip_weight.any()
    batch = _batch()
    with torch.no_grad():
        assert torch.equal(
            off._common_step(batch, "regression", "node").y_pred, on._common_step(batch, "regression", "node").y_pred
        )


@pytest.mark.parametrize("strategy", ["node", "gene"])
def test_the_raw_read_adds_exactly_the_weight_times_the_masked_genes(strategy):
    rate = 0.3 if strategy == "node" else 1.0
    off = _module(strategy=strategy, rate=rate)
    on = _with_random_skip(strategy=strategy, rate=rate, read="raw")
    batch = _batch(strategy=strategy)
    with torch.no_grad():
        out_off = off._common_step(batch, "regression", "node")
        out_on = on._common_step(batch, "regression", "node")
        genes, masked = _raw_inputs(on, batch)
    index = out_on.view.padded_node_idx
    assert torch.equal(genes, masked.x.float())  # what the model is given, hidden entries filled
    assert torch.allclose(out_on.y_pred - out_off.y_pred, F.linear(genes[index], on.own_skip_weight), atol=1e-5)
    assert (out_on.y_pred - out_off.y_pred).abs().max() > 1e-3
    assert torch.equal(out_on.parts["global"]["context"], out_off.y_pred)  # the transformer's part is untouched
    assert torch.allclose(out_on.parts["global"]["own_cell"], F.linear(genes[index], on.own_skip_weight), atol=1e-5)


def test_the_raw_read_has_full_rank_where_the_tokens_read_has_at_most_the_embedding_width():
    """Why the option exists: the token's width bounds the rank of what the skip can read off the genes."""
    narrow = 4  # fewer than the 6 genes; the transformer needs a width its 4 heads divide
    batch = _batch(n=24)
    ranks = {}
    for read in ("embedding", "raw"):
        on = _with_random_skip(read=read, embed=narrow)
        with torch.no_grad():
            out = on._common_step(batch, "regression", "node")
        part = out.parts["global"]["own_cell"]
        ranks[read] = int(torch.linalg.matrix_rank(part - part.mean(0), atol=1e-4))
    assert ranks["embedding"] <= narrow
    assert ranks["raw"] == G > narrow


def test_the_raw_weight_gets_a_gradient_and_the_tokens_still_train():
    on = _module(skip=True, read="raw").train()
    out = on._common_step(_batch(), "regression", "node")
    (out.y_pred - out.y_true).pow(2).mean().backward()
    assert on.own_skip_weight.grad is not None and on.own_skip_weight.grad.abs().sum() > 0
    assert on.input_embedding.weight.grad.abs().sum() > 0


def test_pooled_tokens_keep_the_raw_skip_per_cell():
    off = _module(pool=3 * SPACING)
    on = _with_random_skip(pool=3 * SPACING, read="raw")
    batch = _batch(n=16)
    with torch.no_grad():
        out_off = off._common_step(batch, "regression", "node")
        out_on = on._common_step(batch, "regression", "node")
        genes, _ = _raw_inputs(on, batch)
    delta = out_on.y_pred - out_off.y_pred
    assert torch.allclose(delta, F.linear(genes[out_on.view.padded_node_idx], on.own_skip_weight), atol=1e-5)
    patch = out_off.y_pred
    pairs = [(i, j) for i in range(16) for j in range(i)]
    assert any(torch.equal(patch[i], patch[j]) and not torch.equal(delta[i], delta[j]) for i, j in pairs)


def test_no_hidden_entry_reaches_a_prediction_through_the_raw_skip_and_a_visible_one_does():
    on = _with_random_skip(strategy="gene", rate=1.0, read="raw")
    batch = _batch(strategy="gene")
    with torch.no_grad():
        before = on._common_step(batch, "regression", "node").y_pred
        hidden = batch.clone()
        hidden.x = batch.x.clone()
        hidden.x[:, [0, 1]] = torch.rand(batch.num_nodes, 2) * 50.0
        assert torch.equal(on._common_step(hidden, "regression", "node").y_pred, before)
        visible = batch.clone()
        visible.x = batch.x.clone()
        visible.x[5, 4] += 50.0
        assert not torch.allclose(on._common_step(visible, "regression", "node").y_pred[5], before[5])


def test_a_cell_hidden_whole_gives_the_raw_skip_nothing_of_its_own_row():
    """Under cell masking the hidden cell's row is the fill: changing its true expression moves no prediction."""
    on = _with_random_skip(read="raw")
    batch = _batch()
    assert batch.mask[:4].all() and not batch.mask[4:].any()  # cells 0-3 are hidden by `_batch`
    with torch.no_grad():
        before = on._common_step(batch, "regression", "node")
        changed = batch.clone()
        changed.x = batch.x.clone()
        changed.x[:4] += 50.0
        after = on._common_step(changed, "regression", "node")
    index = before.view.padded_node_idx
    assert torch.equal(index, after.view.padded_node_idx)
    assert torch.equal(after.parts["global"]["own_cell"], before.parts["global"]["own_cell"])
    assert torch.equal(after.y_pred, before.y_pred)


def test_the_raw_read_needs_the_skip_and_a_known_name():
    from interscale.config import _validate_own_cell_skip
    from interscale.module.base import GlobalModule

    _validate_own_cell_skip(_cfg(skip=True, read="raw"))
    _validate_own_cell_skip(_cfg(skip=True))
    _validate_own_cell_skip(_cfg())
    with pytest.raises(ValueError, match="own_cell_skip_input"):
        _validate_own_cell_skip(_cfg(skip=True, read="genes"))
    with pytest.raises(ValueError, match="is off"):
        _validate_own_cell_skip(_cfg(skip=False, read="raw"))
    with pytest.raises(ValueError, match="is off"):
        GlobalModule.from_config(
            _cfg(skip=False, read="raw"), local_mask_hops=0, n_input=G, n_output=G, n_embed=E, decoder_type="linear"
        )


def test_the_raw_read_has_a_checkpoint_name_of_its_own():
    token, raw = _cfg(True), _cfg(True, read="raw")
    names = {get_model_filename_prefix(c, False, True) for c in (_cfg(), token, raw)}
    assert len(names) == 3
    assert "dec-linear-skipraw_" in get_model_filename_prefix(raw, False, True)
    assert "dec-linear-skip_" in get_model_filename_prefix(token, False, True)  # unchanged
    assert get_model_filename_prefix(_cfg(read="raw"), False, True) == get_model_filename_prefix(_cfg(), False, True)
