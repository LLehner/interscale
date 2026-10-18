"""Sparse attention and the expression bias (attention_plan.md, the legnini attention sweep).

Pinned here: the ``entmax`` backend is alpha-entmax of the dense logits, softmax-like at alpha
near 1 and sparse above, with a learnable alpha per head; the ``topk`` backend is softmax over each
query's k best open keys and dense softmax when k covers them all; neither ever puts weight on a
blocked key. The ``expression`` bias ranks keys by expression distance over the genes visible in
both cells, gives every pair with a wholly hidden cell one learned value, ignores padding, and --
through a real GlobalModule -- lets no hidden value reach a prediction. Checkpoint names and the
validator know both backends and the bias.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
entmax = pytest.importorskip("entmax")

from torch_geometric.data import Batch, Data

from interscale.module.global_modules.attention import DenseAttention, EntmaxAttention, TopKAttention
from interscale.module.global_modules.positional_encodings import ExpressionBias
from interscale.tl.masking import MASK_VALUE

H, HD, E = 2, 4, 8  # heads, head width, embedding width
G = 6


def _scores(n_rows=H * 2, length=9, seed=0, blocked=True):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(n_rows, length, length, generator=gen) * 3.0
    if blocked:
        scores[:, :, 2] = float("-inf")  # one key blocked for every query
    return scores


def _layer(cls, **kwargs):
    torch.manual_seed(0)
    return cls(E, H, HD, **kwargs)


# --------------------------------------------------------------------------- entmax


def test_entmax_is_alpha_entmax_of_the_logits_per_head_and_never_weights_a_blocked_key():
    layer = _layer(EntmaxAttention, alpha=1.5, learn_alpha=True)
    scores = _scores()
    weights = layer._normalise(scores)
    assert torch.allclose(weights.sum(-1), torch.ones(weights.shape[:-1]), atol=1e-5)
    assert (weights[:, :, 2] == 0).all()
    assert (weights == 0).any(), "alpha 1.5 on spread logits gives exact zeros"
    assert torch.allclose(weights, entmax.entmax_bisect(scores, alpha=1.5, dim=-1), atol=1e-6)
    # Per head: rows are graph-major over heads, so row r belongs to head r % H.
    with torch.no_grad():
        layer.alpha_logit.copy_(torch.tensor([-4.0, 4.0]))  # alpha ~1.02 and ~1.98
    weights = layer._normalise(scores)
    alpha = layer.alpha()
    for row in range(scores.shape[0]):
        expected = entmax.entmax_bisect(scores[row], alpha=float(alpha[row % H]), dim=-1)
        assert torch.allclose(weights[row], expected, atol=1e-5)
    near_softmax = weights[0::H]
    assert torch.allclose(near_softmax, torch.softmax(scores[0::H], dim=-1), atol=0.05)
    assert (weights[1::H] == 0).float().mean() > (near_softmax == 0).float().mean()


def test_entmax_learns_alpha_per_head_or_keeps_it_fixed():
    layer = _layer(EntmaxAttention, alpha=1.3, learn_alpha=True)
    assert torch.allclose(layer.alpha(), torch.full((H,), 1.3))
    layer._normalise(_scores())[:, 0, 1].sum().backward()
    assert layer.alpha_logit.grad is not None and (layer.alpha_logit.grad != 0).all()
    fixed = _layer(EntmaxAttention, alpha=2.0, learn_alpha=False)
    assert not any("alpha" in name for name, _ in fixed.named_parameters())
    assert torch.allclose(fixed.alpha(), torch.full((H,), 2.0))


def test_entmax_starts_from_the_dense_projections():
    assert torch.equal(_layer(EntmaxAttention).in_proj_weight, _layer(DenseAttention).in_proj_weight)


# --------------------------------------------------------------------------- top-k


def test_topk_is_softmax_over_each_querys_k_best_open_keys():
    layer = _layer(TopKAttention, k=3)
    scores = _scores()
    weights = layer._normalise(scores)
    assert ((weights > 0).sum(-1) == 3).all()
    assert (weights[:, :, 2] == 0).all()
    best = scores.topk(3, dim=-1).indices
    kept = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, best, True)
    expected = torch.softmax(scores.masked_fill(~kept, float("-inf")), dim=-1)
    assert torch.allclose(weights, expected)
    # k covering every key is dense softmax; blocked keys stay at 0 even when k exceeds the open ones.
    dense = _layer(TopKAttention, k=9)._normalise(scores)
    assert torch.allclose(dense, torch.softmax(scores, dim=-1))
    # Ties at the cut do not stretch the budget.
    tied = torch.zeros(1, 4, 6)
    assert ((_layer(TopKAttention, k=2)._normalise(tied) > 0).sum(-1) == 2).all()


def test_topk_passes_gradients_to_the_kept_logits_only():
    scores = _scores(blocked=False).requires_grad_()
    weights = _layer(TopKAttention, k=2)._normalise(scores)
    (weights * torch.arange(9.0)).sum().backward()
    kept = weights.detach() > 0
    assert (scores.grad[~kept] == 0).all() and (scores.grad[kept] != 0).any()


# --------------------------------------------------------------------------- the expression bias


def _padded(x, visible):
    return torch.as_tensor(x, dtype=torch.float32)[None], torch.as_tensor(visible, dtype=torch.float32)[None]


def test_keys_are_ranked_by_expression_distance_over_the_genes_both_cells_show():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 9.0], [0.0, 0.0]])
    visible = torch.ones_like(x)
    visible[3, 1] = 0.0  # cell 3's second gene hidden: compared on the first only, where it equals cell 0
    visible[4] = 0.0  # cell 4 hidden whole
    bias = ExpressionBias(H, num_kernels=4, max_rank=8)
    rank, known = bias.ranks(*_padded(x, visible))
    rank, known = rank[0], known[0]
    assert not known[4].any() and not known[:, 4].any()
    assert known[:4, :4].all()
    # From cell 0: itself and cell 3 (equal on the one gene both show, so tied), then cell 1, cell 2.
    assert set(rank[0, [0, 3]].tolist()) == {0, 1} and rank[0, 1] == 2 and rank[0, 2] == 3
    rms, _ = ExpressionBias.distances(*_padded(x, visible))
    assert rms[0, 1, 2] == pytest.approx(math.sqrt(2.0**2 / 2))  # (2, 0) apart, rms over the 2 shared genes
    assert rms[0, 0, 3] == pytest.approx(0.0, abs=1e-12)
    assert rms[0, 1, 3] == pytest.approx(1.0)  # only the first gene is shared: |1 - 0|


def test_the_bias_reads_its_profile_and_one_learned_value_for_hidden_pairs():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 5.0]])
    visible = torch.ones_like(x)
    visible[3] = 0.0
    bias = ExpressionBias(H, num_kernels=3, max_rank=3)  # knots at log2(1 + rank) = 0, 1, 2
    with torch.no_grad():
        bias.table.copy_(torch.tensor([[0.0, 1.0, 2.0], [5.0, 4.0, 3.0]]))
        bias.unknown.copy_(torch.tensor([-7.0, 7.0]))
    out = bias(*_padded(x, visible))[0]  # [heads, S, S]
    loss = out[:, :3, :3].sum()
    out = out.detach()
    assert out.shape == (H, 4, 4)
    assert torch.equal(out[:, 3], torch.tensor([[-7.0] * 4, [7.0] * 4]))
    assert torch.equal(out[:, :, 3], torch.tensor([[-7.0] * 4, [7.0] * 4]))
    # From cell 0: cell 1 has rank 1 (knot 1), cell 2 rank 2 (log2 3 = 1.585 knots).
    assert out[0, 0, 1] == pytest.approx(1.0) and out[1, 0, 1] == pytest.approx(4.0)
    assert out[0, 0, 2] == pytest.approx(math.log2(3.0))
    loss.backward()
    assert (bias.table.grad != 0).any() and bias.unknown.grad.abs().sum() == 0


def test_padding_takes_no_rank_and_a_fresh_bias_is_zero():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    bias = ExpressionBias(H, num_kernels=4, max_rank=8)
    rank, _ = bias.ranks(*_padded(x, torch.ones_like(x)))
    padded_x = torch.cat([torch.zeros(2, 2), x])  # left padding: zero rows, invisible
    padded_v = torch.cat([torch.zeros(2, 2), torch.ones_like(x)])
    rank_padded, known = bias.ranks(*_padded(padded_x, padded_v))
    assert torch.equal(rank_padded[0, 2:, 2:], rank[0]) and not known[0, :2].any()
    assert torch.equal(bias(*_padded(x, torch.ones_like(x))), torch.zeros(1, H, 3, 3))


# --------------------------------------------------------------------------- through a GlobalModule


def _module(kind="dense", bias=()):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len = 64
    params.n_heads = H
    params.attention.kind = kind
    params.attention.topk.k = 4
    params.pe.bias = list(bias)
    params.pe.distance.max_dist = 100.0
    cfg.dataset.spatial_key = "spatial"
    torch.manual_seed(0)
    common = {"n_input": G, "n_output": G, "n_embed": E, "decoder_type": "linear", "dropout_decoder": 0.0}
    common |= {"decoder_hidden_dims": [8], "mask_percentage": 0.0, "mask_strategy": "node", "block_masking": True}
    module = GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **common)
    if bias:
        # Non-zero biases, so they shape attention and a leak would show.
        gen = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for encoder in module.attention_bias.encoders.values():
                for parameter in encoder.parameters():
                    parameter.copy_(torch.randn(parameter.shape, generator=gen) * 3.0)
    return module.eval()


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    datas = []
    for side in (5, 6):
        ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * 10.0
        n = side * side
        edges = (torch.cdist(pos, pos) == 10.0).nonzero().t()
        data = Data(x=torch.rand(n, G, generator=gen) * 3.0, edge_index=edges, pos=pos)
        data.mask = (pos - pos.mean(0)).norm(dim=1) <= 12.0
        datas.append(data)
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


@pytest.mark.parametrize("kind", ["entmax", "topk"])
@pytest.mark.parametrize("bias", [(), ("distance", "expression")])
def test_no_hidden_value_reaches_a_prediction_and_visible_ones_do(kind, bias):
    module = _module(kind, bias)
    batch = _batch()
    with torch.no_grad():
        base = module._common_step(batch.clone(), "regression", "node").y_pred
        changed = batch.clone()
        changed.x = batch.x.clone()
        changed.x[batch.mask] = torch.rand(int(batch.mask.sum()), G) * 9.0
        assert torch.equal(module._common_step(changed, "regression", "node").y_pred, base)
        visible = batch.clone()
        visible.x = batch.x.clone()
        visible.x[~batch.mask] += 1.0
        assert not torch.allclose(module._common_step(visible, "regression", "node").y_pred, base)


@pytest.mark.parametrize("kind", ["entmax", "topk"])
def test_a_training_step_is_finite_and_the_maps_are_sparse(kind):
    module = _module(kind, ("distance", "expression")).train()
    out = module._common_step(_batch(), "regression", "node")
    loss = torch.nn.functional.mse_loss(out.y_pred, out.y_true)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in module.attention_bias.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    module.eval()
    with torch.no_grad(), module.recording_attention() as record:
        module._common_step(_batch(), "regression", "node")
    maps = record[0]["maps"]
    assert torch.allclose(maps.sum(-1)[maps.sum(-1) > 0], torch.ones(1), atol=1e-4)
    assert (maps == 0).float().mean() > 0.5
    if kind == "topk":
        assert ((maps > 0).sum(-1) <= 4).all()


def test_the_validator_and_the_checkpoint_name_know_the_new_attention():
    from interscale.config import _validate_attention, get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.tl.utils import _attention_tag, _positional_encoding_tag

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.dataset.spatial_key = "spatial"
    params = cfg.model.global_component.parameters
    params.attention.kind = "entmax"
    params.pe.bias = ["distance", "expression"]
    _validate_attention(cfg)
    assert _attention_tag(cfg) == "attn-entmax-a1.5l_"
    assert _positional_encoding_tag(cfg) == "pe-distance+expression_"
    params.attention.entmax.learn_alpha = False
    params.attention.entmax.alpha = 2.0
    assert _attention_tag(cfg) == "attn-entmax-a2_"
    params.attention.entmax.learn_alpha = True
    with pytest.raises(ValueError, match="alpha"):
        _validate_attention(cfg)
    params.attention.kind = "topk"
    params.attention.topk.k = 16
    assert _attention_tag(cfg) == "attn-topk-k16_"
    _validate_attention(cfg)
    params.attention.topk.k = 0
    with pytest.raises(ValueError, match="topk.k"):
        _validate_attention(cfg)
    params.attention.topk.k = 16
    params.attention.kind = "exphormer"  # attends along pairs; the expression bias is dense only
    params.pe.bias = ["expression"]
    with pytest.raises(ValueError, match="expression"):
        _validate_attention(cfg)


def test_the_expression_bias_validates_its_settings():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.global_modules.positional_encodings import _validate_expression

    cfg = get_global_component_cfg(get_cfg_defaults(), "self-attn-transformer")
    pe = cfg.model.global_component.parameters.pe
    _validate_expression(pe)
    pe.expression.num_kernels = 1
    with pytest.raises(ValueError, match="num_kernels"):
        _validate_expression(pe)


def test_mask_value_is_what_the_bias_treats_as_hidden():
    assert MASK_VALUE < 0  # a log1p value never equals it, so visibility is read off the input
    assert np.isfinite(MASK_VALUE)


def test_the_attention_bias_reads_hidden_cells_off_the_masked_input():
    from types import SimpleNamespace

    from interscale.module.global_modules.positional_encodings import AttentionBias

    expression = ExpressionBias(H, num_kernels=4, max_rank=8)
    with torch.no_grad():
        expression.unknown.copy_(torch.tensor([-3.0, 3.0]))
        expression.table.copy_(torch.arange(8.0).view(H, 4))
    bias = AttentionBias({"expression": expression}, n_heads=H, spatial_unit_um=1.0)
    x = torch.rand(5, G) * 2.0
    x[1] = MASK_VALUE  # a hidden cell, as apply_mask leaves it
    x[3, :2] = MASK_VALUE  # two hidden entries
    data = SimpleNamespace(x=x, batch=torch.zeros(5, dtype=torch.long))
    out = bias(data, [torch.arange(5)], torch.zeros(1, 5, dtype=torch.bool), dtype=torch.float32, device="cpu")
    out = out.view(H, 6, 6)[:, :5, :5]  # CLS last
    assert torch.equal(out[:, 1], torch.tensor([[-3.0] * 5, [3.0] * 5]))
    assert torch.equal(out[:, :, 1], torch.tensor([[-3.0] * 5, [3.0] * 5]))
    others = [0, 2, 3, 4]
    assert (out[:, others][:, :, others] >= 0).all()  # read off the profile, not the unknown value
