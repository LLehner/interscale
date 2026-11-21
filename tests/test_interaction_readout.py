"""The interaction read-out (``.claude/ccc_model_plan.md``, IR): the read-out attention, the receiver-aware query, the additive
read-out, key inputs, the same-type block, key dropout, every cell as a target, and the two penalties.

Pinned here:

* **off is the fused model** -- the read-out attention without null key and content rate gives the fused attention's read-out
  (one block and many);
* **the null key** -- rows over the cells sum to one minus the null mass, and the null value is zero;
* **content rate** -- at its zero initialisation the model is the scales prior's exactly; stirred, it changes only the decay;
* **the same-type block** -- a query puts no weight on a key of its own label;
* **blindness** -- a cell's context embedding does not move with its own expression (type query, additive read-out, blocks);
* **key dropout** -- only in training, from its own generator (the global one untouched), never on CLS;
* **entropy** -- in [0, 1], the same in one block and in many;
* **validation** -- what every cell as a target refuses, the entropy penalty without a null key, the content rate without the
  distance prior.

Every parameter is stirred off its initialisation (tests-stir-the-init) unless a test is about the initialisation.
"""

import random

import numpy as np
import pandas as pd

import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.module.base import GlobalModule
from interscale.module.base._step_output import gather_tokens

E, G = 8, 5
SPACING = 100.0
SIDE = 6
N_TYPES = 3


def _cfg(
    kind="readout",
    *,
    chunk_pairs=2**22,
    null_key=True,
    content_rate=False,
    same_type="open",
    key_dropout=0.0,
    inputs=("position",),
    type_key="",
    key_inputs="expression",
    readout_mode="stream",
    bias=("distance",),
    node=("sinusoidal",),
):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 128, 2, 2, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.node, params.pe.bias, params.pe.rotary = list(node), list(bias), []
    params.pe.fourier.max_wavelength = params.pe.rope.max_wavelength = params.pe.distance.max_dist = 2000.0
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    params.pe.distance.shape = "exponential"
    params.attention.kind = kind
    params.attention.fused.chunk_pairs = chunk_pairs
    params.attention.readout.chunk_pairs = chunk_pairs
    params.attention.readout.null_key = null_key
    params.attention.readout.content_rate = content_rate
    params.attention.readout.same_type_keys = same_type
    params.attention.readout.key_dropout = key_dropout
    params.attention.readout.type_vocab = 8
    params.attention.center_qk = False
    params.long_range_attention = False
    params.context_only = "query"
    params.context_query.inputs = list(inputs)
    params.context_query.type_key = type_key
    params.context_query.type_vocab = 8
    params.context_query.key_inputs = key_inputs
    params.context_readout = readout_mode
    cfg.model.n_embed = E
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um, cfg.dataset.name = "spatial", 1.0, "d"
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.extra_obs_keys = [type_key] if type_key else []
    return cfg


def _stir(module, scale=0.1, seed=7):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=gen))
    return module


def _module(seed=0, **kwargs):
    torch.manual_seed(seed)
    return GlobalModule.from_config(
        _cfg(**kwargs),
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.0,
        mask_strategy="node",
        type_gex_embedding="Linear",
    ).eval()


def _graph(side=SIDE, seed=0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    codes = torch.arange(n) % N_TYPES
    data.ctype = torch.nn.functional.one_hot(codes, N_TYPES).float()
    batch = Batch.from_data_list([data])
    batch.obs_names = torch.arange(n)
    return batch


def _rows(module, batch):
    """``[N, E]`` the global embeddings in cell order (one pass, no gradient)."""
    random.seed(0)
    tokens = module.input_embedding(batch.x.float())
    with torch.no_grad():
        out, padding, index, _, _ = module.step(batch, tokens)
    rows = gather_tokens(out, padding)
    cells = torch.full((batch.num_nodes, rows.shape[1]), float("nan"))
    cells[torch.as_tensor(index[0], dtype=torch.long)] = rows
    return cells


def _changed(batch, cell, amount=3.0):
    other = batch.clone()
    other.x = batch.x.clone()
    other.x[cell] = other.x[cell] + amount
    return other


# --------------------------------------------------------------------------- off is the fused model


@pytest.mark.parametrize("chunk_pairs", [2**22, 64])
def test_readout_without_null_is_the_fused_read_out(chunk_pairs):
    fused = _stir(_module(kind="fused", chunk_pairs=chunk_pairs))
    readout = _module(kind="readout", chunk_pairs=chunk_pairs, null_key=False)
    readout.load_state_dict(fused.state_dict(), strict=True)
    batch = _graph()
    torch.testing.assert_close(_rows(readout, batch), _rows(fused, batch), rtol=1e-5, atol=1e-5)


def test_null_key_rows_sum_to_one_minus_the_null_mass_and_hold_no_cls():
    module = _stir(_module(null_key=True))
    batch = _graph()
    tokens = module.input_embedding(batch.x.float())
    attn = module.transformer_encoder.layers[0].self_attn
    attn.recording = []
    try:
        with torch.no_grad():
            module.step(batch, tokens)
        record = attn.recording[0]
        seq_len = record["q"].shape[2]
        weights = attn.weights_rows(record, 0, seq_len)
    finally:
        attn.recording = None
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert torch.all(weights[..., seq_len - 1] == 0)  # CLS is no key under the null key
    assert torch.all(weights[..., -1] > 0)  # the null key is always open


# --------------------------------------------------------------------------- content rate


def test_content_rate_at_initialisation_is_the_scales_prior():
    base = _stir(_module(content_rate=False, type_key="ctype", inputs=("position", "type")))
    rated = _module(content_rate=True, type_key="ctype", inputs=("position", "type"))
    state = {**base.state_dict(), **{k: v for k, v in rated.state_dict().items() if "rate_" in k}}
    rated.load_state_dict(state, strict=True)
    batch = _graph()
    torch.testing.assert_close(_rows(rated, batch), _rows(base, batch))
    _stir(rated, seed=11)  # the rate factors move: the output does too
    with torch.no_grad():
        for layer in rated.transformer_encoder.layers:
            layer.self_attn.rate_type.add_(1.0)
    assert not torch.allclose(_rows(rated, batch), _rows(base, batch))


# --------------------------------------------------------------------------- same-type block


def test_same_type_block_gives_no_weight_to_own_label():
    module = _stir(_module(same_type="block", type_key="ctype", inputs=("type",)))
    batch = _graph()
    tokens = module.input_embedding(batch.x.float())
    attn = module.transformer_encoder.layers[1].self_attn
    attn.recording = []
    try:
        with torch.no_grad():
            module.step(batch, tokens)
        record = attn.recording[0]
        seq_len = record["q"].shape[2]
        weights = attn.weights_rows(record, 0, seq_len)[0, :, : seq_len - 1, : seq_len - 1]  # cells over cells
    finally:
        attn.recording = None
    codes = record["context"].token_codes[0, : seq_len - 1]
    same = codes[:, None] == codes[None, :]
    assert torch.all(weights[:, same] == 0)
    assert torch.all(weights[:, ~same].sum() > 0)


# --------------------------------------------------------------------------- blindness


@pytest.mark.parametrize("chunk_pairs", [2**22, 64])
@pytest.mark.parametrize("readout_mode", ["stream", "additive"])
def test_a_cell_is_blind_to_its_own_expression(chunk_pairs, readout_mode):
    module = _stir(_module(chunk_pairs=chunk_pairs, readout_mode=readout_mode, type_key="ctype", inputs=("position", "type")))
    batch = _graph()
    cell = 14
    before, after = _rows(module, batch), _rows(module, _changed(batch, cell))
    torch.testing.assert_close(before[cell], after[cell], rtol=0, atol=0)
    others = [i for i in range(batch.num_nodes) if i != cell]
    assert not torch.allclose(before[others], after[others])


def test_key_inputs_none_makes_weights_independent_of_expression():
    module = _stir(_module(key_inputs="none", readout_mode="additive"))
    batch = _graph()
    attn = module.transformer_encoder.layers[0].self_attn

    def first_layer_weights(b):
        attn.recording = []
        with torch.no_grad():
            module.step(b, module.input_embedding(b.x.float()))
        record = attn.recording[0]
        attn.recording = None
        return attn.weights_rows(record, 0, record["q"].shape[2])

    torch.testing.assert_close(first_layer_weights(batch), first_layer_weights(_changed(batch, 3)))


# --------------------------------------------------------------------------- key dropout


def test_key_dropout_acts_in_training_only_and_leaves_the_global_generator():
    module = _stir(_module(key_dropout=0.5))
    batch = _graph()
    tokens = module.input_embedding(batch.x.float())
    module.eval()
    given = module.common_step_local_to_global(batch, tokens)
    assert given.attention_mask.hidden_keys is None
    module.train()
    state = torch.random.get_rng_state()
    given = module.common_step_local_to_global(batch, tokens)
    assert torch.equal(state, torch.random.get_rng_state())
    keys = given.attention_mask.hidden_keys
    assert keys is not None and not keys[:, -1].any()


# --------------------------------------------------------------------------- entropy


def test_entropy_is_normalised_and_block_invariant():
    values = []
    for chunk_pairs in (2**22, 64):
        module = _stir(_module(chunk_pairs=chunk_pairs))
        module.track_attn_entropy = True
        batch = _graph()
        _rows(module, batch)
        values.append(float(module.last_attn_entropy))
    assert 0.0 < values[0] < 1.0
    assert values[0] == pytest.approx(values[1], rel=1e-5)


# --------------------------------------------------------------------------- validation


def _validate(cfg):
    from interscale.config import _validate_interaction_readout

    _validate_interaction_readout(cfg)


def test_validator_refuses_what_every_cell_as_target_cannot_have():
    cfg = _cfg(type_key="ctype", inputs=("type",))
    cfg.dataset.score_all_cells = True
    cfg.dataset.mask_percentage = 0.0
    _validate(cfg)
    bad = cfg.clone()
    bad.model.global_component.parameters.context_query.inputs = ["type", "own"]
    with pytest.raises(ValueError, match="own"):
        _validate(bad)
    bad = cfg.clone()
    bad.model.decoder.own_cell_skip = True
    with pytest.raises(ValueError, match="own_cell_skip"):
        _validate(bad)
    bad = cfg.clone()
    bad.dataset.mask_percentage = 0.2
    with pytest.raises(ValueError, match="visible_keys_only"):
        _validate(bad)


def test_validator_refuses_entropy_without_null_and_rate_without_the_prior():
    cfg = _cfg(null_key=False)
    cfg.optim.aux_loss_weights.attn_entropy = 0.1
    with pytest.raises(ValueError, match="null"):
        _validate(cfg)
    cfg = _cfg(content_rate=True, bias=("distance", "fourier"))
    with pytest.raises(ValueError, match="content_rate"):
        _validate(cfg)


def test_every_cell_is_scored_under_score_all_cells():
    module = _stir(_module(type_key="ctype", inputs=("type",)))
    module.score_all_cells = True
    batch = _graph()
    y_true, idx, entry = module._process_batch_for_metrics(batch, "regression", "node", [list(range(batch.num_nodes))])
    assert entry.all() and entry.shape == y_true.shape


# --------------------------------------------------------------------------- the exact influence (evaluation.interaction_readout)


def _decomposition_check(module, batch):
    from interscale.evaluation.interaction_readout import _block_weights, _values, readout_pass

    rp = readout_pass(module, batch, eval_step=True)
    y = _values(module, rp).float()  # [X, L, E]
    w_dec = module.decoder.decoder.weight.detach().float()
    intercept = module.decoder.decoder.bias.detach().float().clone()
    for layer in module.transformer_encoder.layers:
        intercept = intercept + w_dec @ layer.self_attn.out_proj.bias.detach().float()
    with torch.no_grad():
        weights, _ = _block_weights(module, rp, rp.offset, rp.seq_len - 1)  # [cells, X, L]
    contributions = torch.einsum("rxl,xle->rle", weights.float(), y) @ w_dec.T  # [cells, L, G]
    rebuilt = contributions.sum(1) + intercept
    nodes = torch.as_tensor(rp.index_nodes)
    return rp, rebuilt, torch.as_tensor(rp.context)[nodes]


@pytest.mark.parametrize("null_key", [True, False])
@pytest.mark.parametrize("chunk_pairs", [2**22, 64])
def test_contributions_sum_to_the_context_part(null_key, chunk_pairs):
    module = _stir(_module(null_key=null_key, chunk_pairs=chunk_pairs, readout_mode="additive", type_key="ctype", inputs=("position", "type")))
    _, rebuilt, context = _decomposition_check(module, _graph())
    torch.testing.assert_close(rebuilt, context, rtol=1e-4, atol=1e-5)


def test_influence_agrees_with_the_direct_decomposition():
    from interscale.evaluation.interaction_readout import _block_weights, _values, influence

    module = _stir(_module(readout_mode="additive"))
    batch = _graph()
    rp, _, _ = _decomposition_check(module, batch)
    receivers = np.array([3, 17, 30])
    target = torch.zeros(1, G)
    target[0, 1] = 1.0
    inf = influence(module, rp, receivers, targets=target, receiver_target=np.zeros(3, dtype=int), top_k=5, gene_top=2)
    y = _values(module, rp).float()
    w_dec = module.decoder.decoder.weight.detach().float()
    for row, node in enumerate(receivers):
        pos = int(rp.position_of([node])[0])
        weights, _ = _block_weights(module, rp, pos, pos + 1)
        c = (torch.einsum("xl,xle->le", weights[0].float().detach(), y) @ w_dec.T).numpy()  # [L, G]
        cells = c[rp.offset : rp.seq_len - 1]
        mags = np.linalg.norm(cells, axis=1)
        best = np.argsort(-mags)[:5]
        np.testing.assert_array_equal(inf["top_nodes"][row], np.asarray(rp.index_nodes)[best])
        np.testing.assert_allclose(inf["top_mag"][row], mags[best], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(inf["top_genes"][row, 0], cells[best[0]], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(inf["scores"][row][np.asarray(rp.index_nodes)], cells[:, 1], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(inf["total"][row], mags.sum() + np.linalg.norm(c[-1]) * 0, rtol=1e-4)


def test_planted_scores_and_maps_run_and_types_decompose():
    from interscale.evaluation.interaction_readout import (
        TypeTestAccumulator,
        cell_maps,
        head_windows,
        planted_scores,
        readout_pass,
        segmentation_flags,
        type_contributions,
    )

    module = _stir(_module(readout_mode="additive", content_rate=True, type_key="ctype", inputs=("type",)))
    batch = _graph()
    rp = readout_pass(module, batch, eval_step=True)
    n = batch.num_nodes
    var = pd.DataFrame({"interaction": ["P1", "P1", "", "", ""], "role": ["response", "response", "noise", "noise", "noise"]})
    edges = pd.DataFrame({"sender": [0, 1, 2], "receiver": [7, 8, 9], "program": ["P1"] * 3, "range_class": ["short"] * 3})
    table = planted_scores(module, rp, edges, var, receivers_per_programme=3)
    assert set(table["program"]) == {"P1"} and 0 <= table["top5"].iloc[0] <= 1
    assert np.isfinite(table["receiver_auroc"].iloc[0])
    # negatives restricted (the receiver type's cells on typed data): none eligible -> no receiver score, the sender scores stay
    none = planted_scores(module, rp, edges, var, receivers_per_programme=3, negatives={"P1": np.zeros(n, dtype=bool)})
    assert np.isnan(none["receiver_auroc"].iloc[0]) and np.isnan(none["receiver_ap"].iloc[0]) and np.isnan(none["null_others"].iloc[0])
    assert none["sender_ap"].iloc[0] == table["sender_ap"].iloc[0]
    some = np.zeros(n, dtype=bool)
    some[[3, 4, 5]] = True
    few = planted_scores(module, rp, edges, var, receivers_per_programme=3, negatives={"P1": some})
    assert np.isfinite(few["receiver_auroc"].iloc[0])
    windows = head_windows(module)
    assert {"scale_um", "d10_um", "null_logit", "rate_t0"} <= set(windows.columns)
    maps = cell_maps(module, rp, top_k=4)
    assert np.isfinite(maps["mass"]).all() and (maps["mass"] <= 1).all()
    codes = (np.arange(n) % N_TYPES).astype(int)
    out = type_contributions(module, batch, rp, codes, N_TYPES, eval_step=True)
    # the decomposition by type sums to the cells' summed contributions (no CLS under the null key)
    rp2, rebuilt, context = _decomposition_check(module, batch)
    intercept = context - torch.as_tensor(np.nansum(out["decomposition"], axis=1)[np.asarray(rp2.index_nodes)])
    assert torch.allclose(intercept, intercept[:1].expand_as(intercept), atol=1e-4)
    acc = TypeTestAccumulator(N_TYPES, G)
    acc.add(out["ablation"], np.random.default_rng(0).normal(size=(n, G)), codes)
    assert len(acc.table([f"t{k}" for k in range(N_TYPES)], [f"g{k}" for k in range(G)]))
    flags = segmentation_flags(
        np.random.default_rng(1).random((n, G)), codes, rp.pos_um, np.zeros(n, dtype=int),
        [f"t{k}" for k in range(N_TYPES)], [f"g{k}" for k in range(G)], near_um=150.0, far_um=100.0,
    )
    assert {"flagged", "q"} <= set(flags.columns)
