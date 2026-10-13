"""Queries and keys from mean-centred tokens (`attention.center_qk`; attention_plan.md, round 3).

Most of every spot's expression is the profile all spots share, so the tokens entering attention
are nearly identical, their dot products barely tell keys apart, and attention stays uniform with
no gradient to leave it. Pinned here: centring removes whatever all cell tokens of a sequence share
(padding and CLS left out of the mean), only queries and keys see it (values and the residual keep
the tokens), near-identical tokens give uniform attention without it and not with it in every
backend, and the diagnostics read the tokens' similarity.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.module.global_modules.attention import DenseAttention, PerformerAttention
from interscale.module.global_modules.transformer_encoder_layer import (
    CustomTransformerEncoderLayer,
    cell_positions,
    center_tokens,
    mean_pairwise_cosine,
)

E, H = 8, 2


def _tokens(seq_len=6, n_graphs=2, seed=0):
    """``[L, N, E]`` tokens, CLS last; graph 0 has its first two positions padded."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(seq_len, n_graphs, E, generator=gen)
    padding = torch.zeros(n_graphs, seq_len, dtype=torch.bool)
    padding[0, :2] = True
    return x, padding


def _layer(center_qk, factory=None):
    torch.manual_seed(0)
    layer = CustomTransformerEncoderLayer(E, H, 16, 0.0, center_qk=center_qk, attention_factory=factory)
    layer.keep_maps = True
    return layer.eval()


def test_the_centring_ignores_padding_and_cls():
    x, padding = _tokens()
    cells = cell_positions(x, padding)
    assert cells[0].tolist() == [False, False, True, True, True, False]
    assert cells[1].tolist() == [True, True, True, True, True, False]

    changed = x.clone()
    changed[:2, 0] += 100.0  # graph 0's padding
    changed[-1] += 100.0  # CLS of both graphs
    centred, again = center_tokens(x, cells), center_tokens(changed, cells)
    assert torch.allclose(centred[cells.T], again[cells.T], atol=1e-5)
    # A float key-padding mask (-inf at padding), as the encoder hands it on, reads the same.
    float_mask = torch.zeros(padding.shape).masked_fill(padding, float("-inf"))
    assert torch.equal(cell_positions(x, float_mask), cells)


@pytest.mark.parametrize("backend", ["torch", "dense_core"])
def test_a_shared_shift_of_every_token_leaves_centred_attention_unchanged(backend):
    factory = None if backend == "torch" else (lambda: DenseAttention(E, H, 6))
    x, padding = _tokens()
    shift = torch.randn(E) * 5.0

    weights = {}
    for center in (True, False):
        layer = _layer(center, factory)
        layer._sa_block(x, None, padding, False)
        base = layer.get_attn_output_weights().clone()
        layer._sa_block(x + shift, None, padding, False)
        weights[center] = (base, layer.get_attn_output_weights())
    assert torch.allclose(*weights[True], atol=1e-5)
    assert not torch.allclose(*weights[False], atol=1e-3)


@pytest.mark.parametrize("backend", ["torch", "dense_core"])
def test_values_and_the_residual_keep_the_tokens_as_they_are(backend):
    """With the query and key projections zero, attention is uniform over the open keys, and the
    output is the projected mean of the RAW tokens' values -- not of the centred ones."""
    factory = None if backend == "torch" else (lambda: DenseAttention(E, H, E // H))
    layer = _layer(True, factory)
    inner = layer.self_attn.in_proj_weight.shape[0] // 3
    with torch.no_grad():
        layer.self_attn.in_proj_weight[: 2 * inner] = 0.0
    x, padding = _tokens()
    out = layer._sa_block(x, None, padding, False)

    w_v, b_v = layer.self_attn.in_proj_weight[2 * inner :], layer.self_attn.in_proj_bias[2 * inner :]
    for graph in range(x.shape[1]):
        keys = ~padding[graph]  # every non-padded position, CLS included
        mean_value = (x[keys, graph] @ w_v.T + b_v).mean(0)
        expected = layer.self_attn.out_proj(mean_value)
        assert torch.allclose(out[keys, graph], expected.expand(int(keys.sum()), -1), atol=1e-5)


def test_a_backend_still_refuses_queries_and_keys_from_different_tensors():
    attention = PerformerAttention(E, H, E // H, num_features=16)
    x, _ = _tokens()
    with pytest.raises(ValueError, match="self-attention only"):
        attention(x, x.clone(), x)


def test_the_token_cosine_reads_how_alike_the_cell_tokens_are():
    x, padding = _tokens(seq_len=40)
    cells = cell_positions(x, padding)
    alike = x * 0.01 + torch.randn(E) * 3.0  # one shared vector plus a little noise
    assert mean_pairwise_cosine(alike, cells).min() > 0.99
    assert mean_pairwise_cosine(x, cells).abs().max() < 0.2


# --------------------------------------------------------------------------- through a module


def _module(kind, center_qk):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = 16
    params = cfg.model.global_component.parameters
    params.n_heads, params.max_seq_len = 4, 64
    params.attention.kind = kind
    params.attention.center_qk = center_qk
    params.attention.performer.num_features = 512
    params.attention.exphormer.degree = 16
    torch.manual_seed(0)
    module = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=8,
        n_output=8,
        n_embed=16,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )
    return module.eval()


def _alike_graph(n=40, seed=0):
    """A graph whose embeddings are one shared vector plus small spot-specific deviations."""
    rng = np.random.default_rng(seed)
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    d = Data(x=torch.tensor(rng.random((n, 8)), dtype=torch.float32), edge_index=torch.tensor([src, dst]))
    shared = rng.normal(size=16) * 5.0
    d.embeddings = torch.tensor(shared + 0.1 * rng.normal(size=(n, 16)), dtype=torch.float32)
    d.mask = torch.zeros(n, dtype=torch.bool)
    batch = Batch.from_data_list([d])
    batch.obs_names = torch.arange(n)
    return batch


def _effective_keys(module, batch):
    """Mean over cell queries and heads of exp(entropy) of each row over the cell keys, and of the
    number of cell keys a row can attend to at all."""
    module.return_attention_maps = True
    out = module._common_step(batch, "regression", "node")
    module.return_attention_maps = False
    maps = out.attn[0].detach()[:, :-1, :-1]  # first layer, cells only
    rows = maps / maps.sum(-1, keepdim=True)
    return float(torch.special.entr(rows).sum(-1).exp().mean()), float((maps > 0).sum(-1).double().mean())


@pytest.mark.parametrize("kind", ["dense", "performer", "exphormer"])
def test_near_identical_tokens_leave_attention_uniform_unless_centred(kind):
    batch = _alike_graph()
    plain, open_keys = _effective_keys(_module(kind, False), batch)
    centred, _ = _effective_keys(_module(kind, True), batch)

    assert plain > 0.97 * open_keys  # uniform over whatever the backend lets a cell attend to
    assert centred < 0.9 * plain


def test_the_diagnostics_read_the_token_cosine_before_centring():
    from interscale.evaluation.attention_diagnostics import attention_diagnostics, attention_metric_name

    graph = _alike_graph().to_data_list()[0]
    graph.pos = torch.rand(graph.num_nodes, 2) * 100
    results = attention_diagnostics(
        _module("dense", True),
        [graph],
        prediction_task="regression",
        prediction_level="node",
        device=torch.device("cpu"),
    )
    # The first layer's input: one shared vector plus noise, even though queries and keys are centred.
    assert results[attention_metric_name("token_cosine", 0)] > 0.9


def test_the_checkpoint_name_carries_the_centring():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.tl.utils import get_model_filename_prefix

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.dataset.name = "d"
    assert get_model_filename_prefix(cfg, False, True).endswith("self-attn-transformer_")  # no tag by default
    cfg.model.global_component.parameters.attention.center_qk = True
    assert get_model_filename_prefix(cfg, False, True).endswith("attn-dense-cqk_")
    cfg.model.global_component.parameters.attention.kind = "performer"
    assert get_model_filename_prefix(cfg, False, True).endswith("attn-performer-cqk_")
