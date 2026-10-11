"""Attention backends of the global transformer (attention_plan.md).

Pinned here: the default attention is the model from before backends existed; maps are kept only
where something reads them; each backend computes what its definition says, keeps the long-range
mask's semantics, and has an explicit path that produces the same output together with its maps.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interscale.module.global_modules import TransformerNodeEncoderHook

N_HEADS = 2
N_EMBED = 8


class _Batch:
    """Minimal stand-in for the PyG batch the global module reads."""

    def __init__(self, edge_index, batch, n_nodes, pos=None):
        self.edge_index = edge_index
        self.batch = batch
        self.obs_names = torch.arange(n_nodes)
        self.num_nodes = n_nodes
        self.mask = torch.zeros(n_nodes, dtype=torch.bool)
        if pos is not None:
            self.pos = pos


def _path_graph(n, offset=0):
    src = list(range(offset, offset + n - 1)) + list(range(offset + 1, offset + n))
    dst = list(range(offset + 1, offset + n)) + list(range(offset, offset + n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _module(long_range=False, hops=1, max_seq_len=32, **kwargs):
    return TransformerNodeEncoderHook(
        max_seq_len=max_seq_len,
        n_heads=N_HEADS,
        dropout_global=0.0,
        act_func="relu",
        num_layers=2,
        dim_feedforward=16,
        long_range_attention=long_range,
        local_mask_hops=hops,
        n_input=8,
        n_output=8,
        n_embed=N_EMBED,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
        **kwargs,
    )


def _inputs(n=7, seed=0):
    rng = np.random.default_rng(seed)
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    emb = torch.tensor(rng.normal(size=(n, N_EMBED)), dtype=torch.float32)
    return batch, emb


# --------------------------------------------------------------------------- maps only when read


def test_a_training_step_keeps_no_map_and_hooks_nothing():
    batch, emb = _inputs()
    module = _module()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb)
    out, _, attn = module.forward(padded, padding, mask, rotary=rotary)
    out.sum().backward()

    assert attn is None
    for layer in module.transformer_encoder.layers:
        assert layer.get_attn_output_weights() is None
        assert layer.get_attn_gradients() is None


def test_a_loss_that_reads_attention_gets_every_layers_map():
    batch, emb = _inputs()
    module = _module(return_attention_maps=True)
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb)
    _, _, attn = module.forward(padded, padding, mask, rotary=rotary)

    n_tokens = padded.shape[0] + 1
    assert attn.shape == (2, N_HEADS, n_tokens, n_tokens)
    assert attn.requires_grad  # the entropy loss backpropagates through them


def test_relevance_gets_maps_and_their_gradients():
    batch, emb = _inputs()
    module = _module().eval()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
    out, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    out.sum().backward()

    assert attn is not None
    for layer in module.transformer_encoder.layers:
        assert layer.get_attn_output_weights() is not None
        assert layer.get_attn_gradients() is not None
        # Flags are per call: the next training step does not inherit them.
        assert not layer.register_hook and not layer.keep_maps


def test_keeping_maps_changes_no_number():
    batch, emb = _inputs()
    torch.manual_seed(0)
    plain = _module()
    torch.manual_seed(0)
    keeping = _module(return_attention_maps=True)
    gi = plain.common_step_local_to_global(batch, emb, eval_step=True)
    out_plain = plain.forward(gi[0], gi[1], gi[3], rotary=gi[4])[0]
    out_keep = keeping.forward(gi[0], gi[1], gi[3], rotary=gi[4])[0]
    assert torch.equal(out_plain, out_keep)


def test_from_config_returns_maps_only_for_the_attention_entropy_loss():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    def build(loss):
        cfg = get_cfg_defaults()
        cfg.model.global_component.name = "self-attn-transformer"
        cfg = get_global_component_cfg(cfg, "self-attn-transformer")
        cfg.optim.loss = loss
        return GlobalModule.from_config(
            cfg,
            local_mask_hops=0,
            n_input=8,
            n_output=8,
            n_embed=N_EMBED,
            decoder_type="linear",
            dropout_decoder=0.0,
            decoder_hidden_dims=[16],
            mask_percentage=0.1,
            mask_strategy="node",
        )

    assert build("SCE_EntropyATT_Loss").return_attention_maps
    assert not build("MSELoss").return_attention_maps


# --------------------------------------------------------------------------- registry, head width


def _cfg(kind="dense", head_dim=0, *, bias=(), rotary=(), loss="MSELoss", n_embed=16, n_heads=4):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = n_embed
    params = cfg.model.global_component.parameters
    params.n_heads = n_heads
    params.attention.kind = kind
    params.attention.head_dim = head_dim
    params.pe.bias = list(bias)
    params.pe.rotary = list(rotary)
    params.pe.distance.max_dist = 1000.0
    params.pe.rope.max_wavelength = 1000.0
    cfg.dataset.spatial_key = "spatial"
    cfg.optim.loss = loss
    return cfg


def _from_config(cfg):
    from interscale.module.base import GlobalModule

    return GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=8,
        n_output=8,
        n_embed=cfg.model.n_embed,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )


@pytest.fixture
def registered():
    """Register test-only backends, removed again afterwards."""
    from interscale.module.global_modules.attention import ATTENTION_REGISTRY, register_attention

    added = []

    def register(name, **spec):
        spec.setdefault("validate", lambda attention_cfg: None)
        spec.setdefault("mask", "dense")
        register_attention(name, **spec)(lambda *a, **k: None)
        added.append(name)

    yield register
    for name in added:
        ATTENTION_REGISTRY.pop(name)


def test_the_default_attention_is_torchs_and_keeps_its_state_dict_keys():
    from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits

    module = _from_config(_cfg())
    for layer in module.transformer_encoder.layers:
        assert type(layer.self_attn) is MultiHeadAttentionWithEdits
    assert module.head_dim == 4
    assert {k for k in module.state_dict() if "self_attn" in k} == {
        f"transformer_encoder.layers.{i}.self_attn.{name}"
        for i in range(2)
        for name in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias")
    }


def test_the_explicit_default_head_width_is_still_torchs_attention():
    from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits

    module = _from_config(_cfg(head_dim=4))
    assert type(module.transformer_encoder.layers[0].self_attn) is MultiHeadAttentionWithEdits


def test_a_wider_head_gets_its_own_projections():
    from interscale.module.global_modules.attention import DenseAttention

    module = _from_config(_cfg(head_dim=8))
    attn = module.transformer_encoder.layers[0].self_attn
    assert isinstance(attn, DenseAttention)
    assert attn.in_proj_weight.shape == (3 * 4 * 8, 16)
    assert attn.out_proj.weight.shape == (16, 4 * 8)


def _dense_pair(seed=0):
    from interscale.module.global_modules.attention import DenseAttention
    from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits

    torch.manual_seed(seed)
    reference = MultiHeadAttentionWithEdits(N_EMBED, N_HEADS, dropout=0.0)
    torch.manual_seed(seed)
    skeleton = DenseAttention(N_EMBED, N_HEADS, N_EMBED // N_HEADS)
    return reference, skeleton


def test_the_skeleton_draws_torchs_initial_weights():
    """At the default head width a backend starts from the dense model's projections at one seed,
    so a comparison between backends is paired in its initial weights too."""
    reference, skeleton = _dense_pair()
    assert torch.equal(reference.in_proj_weight, skeleton.in_proj_weight)
    assert torch.equal(reference.out_proj.weight, skeleton.out_proj.weight)
    assert torch.equal(reference.in_proj_bias, skeleton.in_proj_bias)


def test_the_skeletons_dense_core_reproduces_torchs_attention():
    reference, skeleton = _dense_pair()
    n, b = 6, 2
    x = torch.randn(n, b, N_EMBED)
    blocked = torch.zeros(b * N_HEADS, n, n, dtype=torch.bool)
    blocked[:, torch.arange(n), torch.arange(n)] = True
    padding = torch.zeros(b, n, dtype=torch.bool)
    padding[1, :2] = True
    angles = torch.rand(b * N_HEADS, n, N_EMBED // N_HEADS // 2) * 6
    cos, sin = angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)

    for mask in (blocked, torch.zeros_like(blocked, dtype=torch.float32).masked_fill(blocked, float("-inf"))):
        out_ref, w_ref = reference(
            x, x, x, key_padding_mask=padding, attn_mask=mask, average_attn_weights=False, rotary=(cos, sin)
        )
        out, w = skeleton(x, x, x, key_padding_mask=padding, attn_mask=mask, rotary=(cos, sin))
        assert torch.allclose(out, out_ref, atol=1e-6)
        assert torch.allclose(w, w_ref, atol=1e-6)


def test_a_backend_changes_no_weight_built_after_it():
    """Built in a forked RNG with the default attention's draws replayed: the FFN, norms, CLS token
    and decoder of a wide-head model equal the dense model's at one seed."""
    torch.manual_seed(0)
    dense = _from_config(_cfg())
    torch.manual_seed(0)
    wide = _from_config(_cfg(head_dim=8))
    wide_state = wide.state_dict()
    for key, value in dense.state_dict().items():
        if "self_attn" not in key:
            assert torch.equal(value, wide_state[key]), key


def test_wide_heads_train_finite_with_rope_and_the_long_range_mask():
    from interscale.module.global_modules.positional_encodings import build_positional_encodings

    cfg = _cfg(head_dim=8, rotary=["rope"], n_embed=N_EMBED, n_heads=N_HEADS)
    n = 9
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n, pos=torch.rand(n, 2) * 500)
    module = _module(
        long_range=True,
        hops=2,
        attention=cfg.model.global_component.parameters.attention,
        **{
            k: v
            for k, v in build_positional_encodings(cfg, n_embed=N_EMBED, n_heads=N_HEADS, head_dim=8).items()
            if v is not None
        },
    )
    emb = torch.randn(n, N_EMBED)
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
    out, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    out.sum().backward()

    assert rotary[0].shape[-1] == 8
    assert torch.isfinite(out).all() and torch.isfinite(attn).all()
    blocked = mask[0]
    assert torch.allclose(attn[0, 0][blocked], torch.zeros(int(blocked.sum())), atol=1e-6)
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


# --------------------------------------------------------------------------- the validator


def test_an_unknown_backend_is_rejected():
    from interscale.config import _validate_attention

    with pytest.raises(ValueError, match="not a registered backend"):
        _validate_attention(_cfg(kind="nope"))


def test_a_negative_or_indivisible_head_width_is_rejected():
    from interscale.config import _validate_attention

    with pytest.raises(ValueError, match="head_dim must be >= 0"):
        _validate_attention(_cfg(head_dim=-1))
    with pytest.raises(ValueError, match="not divisible"):
        _validate_attention(_cfg(n_embed=18, n_heads=4))
    _validate_attention(_cfg(n_embed=18, n_heads=4, head_dim=8))  # a set head width needs no division


def test_a_bias_the_backend_cannot_take_is_rejected_with_what_would_fit(registered):
    from interscale.config import _validate_attention

    registered("_test_no_bias", bias_forms=frozenset())
    with pytest.raises(ValueError, match="cannot carry the attention bias 'distance'") as err:
        _validate_attention(_cfg(kind="_test_no_bias", bias=["distance"]))
    assert "dense" in str(err.value)


def test_rope_on_a_backend_without_rotation_is_rejected(registered):
    from interscale.config import _validate_attention

    registered("_test_no_rope", rotary=False)
    with pytest.raises(ValueError, match="cannot apply pe.rotary"):
        _validate_attention(_cfg(kind="_test_no_rope", rotary=["rope"]))


def test_an_attention_reading_loss_needs_maps_in_training(registered):
    from interscale.config import _validate_attention

    registered("_test_no_maps", training_maps=False)
    with pytest.raises(ValueError, match="reads the attention maps in training"):
        _validate_attention(_cfg(kind="_test_no_maps", loss="SCE_EntropyATT_Loss"))
    _validate_attention(_cfg(kind="_test_no_maps"))


def test_rope_reads_the_attention_head_width():
    from interscale.config import _validate_pe

    with pytest.raises(ValueError, match="multiple of 4"):
        _validate_pe(_cfg(rotary=["rope"], head_dim=6))
    with pytest.raises(ValueError, match="multiple of 4"):
        _validate_pe(_cfg(rotary=["rope"], n_embed=24, n_heads=8))
    _validate_pe(_cfg(rotary=["rope"], n_embed=24, n_heads=8, head_dim=8))


def test_the_checkpoint_name_carries_a_non_default_attention():
    from interscale.tl.utils import get_model_filename_prefix

    def prefix(**kwargs):
        cfg = _cfg(**kwargs)
        cfg.dataset.name = "d"
        return get_model_filename_prefix(cfg, local_component=False, global_component=True)

    # (The component name, self-attn-transformer, contains "attn-" itself; the tag comes after it.)
    assert prefix().endswith("self-attn-transformer_")
    assert prefix(head_dim=0) == prefix()
    assert prefix(head_dim=8) == prefix() + "attn-dense-hd8_"


# --------------------------------------------------------------------------- performer


PERFORMER = {"kind": "performer", "head_dim": 0, "performer": {"num_features": 64, "redraw_interval": 0}}


class _Att(dict):
    """A dict that reads like the yacs attention block (attribute and .get access)."""

    __getattr__ = dict.__getitem__


def _performer_cfg(num_features=64, redraw_interval=0, head_dim=0):
    return _Att(
        kind="performer",
        head_dim=head_dim,
        performer=_Att(num_features=num_features, redraw_interval=redraw_interval),
    )


def _performer(num_features=64, seed=0):
    from interscale.module.global_modules.attention import PerformerAttention

    torch.manual_seed(seed)
    module = PerformerAttention(N_EMBED, N_HEADS, N_EMBED // N_HEADS, num_features=num_features)
    module.draw_features(torch.Generator().manual_seed(seed))
    return module


def _qkv(n_graphs=2, seq_len=7, head_dim=N_EMBED // N_HEADS, scale=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    shape = (n_graphs * N_HEADS, seq_len, head_dim)
    return tuple(torch.randn(shape, generator=gen) * scale for _ in range(3))


def _diagonal_pairs(n_graphs, seq_len):
    from interscale.tl.masking import blocked_pairs

    return blocked_pairs(None, None, [list(range(seq_len))] * n_graphs, n_hops=0, device=torch.device("cpu"))


def _exact_attention(q, k, v, blocked, padding):
    """Masked softmax attention, the reference the performer estimates."""
    logits = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])
    graph, query, key = blocked
    heads = torch.arange(N_HEADS)
    rows = (graph[:, None] * N_HEADS + heads).reshape(-1)
    logits[rows, query.repeat_interleave(N_HEADS), key.repeat_interleave(N_HEADS)] = float("-inf")
    if padding is not None:
        logits = logits.masked_fill(padding.repeat_interleave(N_HEADS, 0)[:, None, :], float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    return weights @ v, weights


import math

from interscale.module.global_modules.attention import StructuredMask


def test_orthogonal_random_features_estimate_the_softmax_kernel_without_bias():
    """E[phi(x) . phi(y)] = exp(x . y) needs every row's direction uniform on the sphere. Plain QR of
    a Gaussian matrix is not uniform over the orthogonal group; this caught it at ~7% too high."""
    from interscale.module.global_modules.attention import orthogonal_gaussian

    gen = torch.Generator().manual_seed(5)
    x, y = (
        torch.randn(4, generator=gen, dtype=torch.float64) * 0.6,
        torch.randn(4, generator=gen, dtype=torch.float64) * 0.6,
    )
    w = torch.cat([orthogonal_gaussian(4096, 4, torch.Generator().manual_seed(s)) for s in range(50)]).double()
    estimates = torch.exp(w @ x - 0.5 * x @ x) * torch.exp(w @ y - 0.5 * y @ y)
    stderr = estimates.std() / math.sqrt(len(estimates))
    assert abs(estimates.mean() - torch.exp(x @ y)) < 4 * stderr


def test_the_performer_estimates_softmax_attention_and_improves_with_features():
    q, k, v = _qkv(scale=0.6)
    blocked = _diagonal_pairs(2, 7)
    _, exact = _exact_attention(q, k, v, blocked, None)

    def error(m, seed):
        module = _performer(m, seed)
        _, weights = module._attend(q, k, v, None, None, True, StructuredMask(blocked))
        return (weights - exact).abs().sum(-1).mean().item() / 2  # mean total-variation distance

    few = np.mean([error(16, s) for s in range(8)])
    many = np.mean([error(4096, s) for s in range(8)])
    assert many < 0.02
    assert many < few / 4  # error falls like 1/sqrt(m): 256x the features, ~16x smaller


def test_the_factorised_and_explicit_paths_agree_with_mask_and_padding():
    # 8 token positions and the CLS slot last, two graphs, the second left-padded by 3.
    q, k, v = _qkv(n_graphs=2, seq_len=9)
    padding = torch.zeros(2, 9, dtype=torch.bool)
    padding[1, :3] = True
    from interscale.tl.masking import blocked_pairs

    edge_index = torch.cat([_path_graph(8), _path_graph(5, offset=8)], dim=1)
    batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    blocked = blocked_pairs(edge_index, batch, [list(range(8)), list(range(5))], n_hops=2)
    module = _performer()
    out_fact, none = module._attend(q, k, v, padding, None, False, StructuredMask(blocked))
    out_expl, weights = module._attend(q, k, v, padding, None, True, StructuredMask(blocked))

    assert none is None
    assert torch.allclose(out_fact, out_expl, atol=1e-5)
    graph, query, key = blocked
    rows = (graph[:, None] * N_HEADS + torch.arange(N_HEADS)).reshape(-1)
    assert torch.all(weights[rows, query.repeat_interleave(N_HEADS), key.repeat_interleave(N_HEADS)] == 0)
    assert torch.all(weights[2:][:, :, :3] == 0)  # padded keys of the second graph
    assert torch.allclose(weights.sum(-1), torch.ones(weights.shape[:2]), atol=1e-5)


def test_subtracting_the_blocked_terms_keeps_precision_when_they_dominate():
    """A query whose kernel mass sits almost entirely on its blocked keys (35 of 40 keys aligned
    with it; ~0.3% of the mass left open). The factorised path subtracts that mass from the full
    sums: in float32 the cancellation costs ~1e-5 here, in float64 nothing measurable."""
    from interscale.module.global_modules.attention import PerformerAttention

    gen = torch.Generator().manual_seed(0)
    n_rows, seq_len, n_blocked, head_dim = N_HEADS, 40, 35, N_EMBED // N_HEADS
    u = torch.randn(head_dim, generator=gen)
    u = u / u.norm()
    q = torch.randn(n_rows, seq_len, head_dim, generator=gen) * 0.3
    q[:, 0] = u * 4
    k = torch.randn(n_rows, seq_len, head_dim, generator=gen) * 0.3 - u
    k[:, :n_blocked] = u * 4 + torch.randn(n_rows, n_blocked, head_dim, generator=gen) * 0.05
    v = torch.randn(n_rows, seq_len, head_dim, generator=gen)
    blocked = torch.stack(
        [torch.zeros(n_blocked, dtype=torch.long), torch.zeros(n_blocked, dtype=torch.long), torch.arange(n_blocked)]
    )
    module = PerformerAttention(N_EMBED, N_HEADS, head_dim, num_features=256)
    module.draw_features(torch.Generator().manual_seed(1))
    out_fact, _ = module._attend(q, k, v, None, None, False, StructuredMask(blocked))
    out_expl, _ = module._attend(q, k, v, None, None, True, StructuredMask(blocked))
    assert torch.allclose(out_fact[:, 0], out_expl[:, 0], atol=2e-6, rtol=0)


def _module_with(attention, long_range=False, hops=1, **kwargs):
    return _module(long_range=long_range, hops=hops, attention=attention, **kwargs)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_the_performer_is_finite_on_a_dense_graph_under_the_long_range_mask(hops):
    """The NaN regression test of the long-range mask, for the performer: most of each row is blocked."""
    n = 12
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    module = _module_with(_performer_cfg(), long_range=True, hops=hops).eval()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, torch.randn(n, N_EMBED), True)
    out, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    out.sum().backward()

    assert isinstance(mask, StructuredMask)
    assert torch.isfinite(out).all() and torch.isfinite(attn).all()
    # Every cell is inside every other's neighbourhood: each cell row attends to CLS alone.
    assert torch.allclose(attn[0, :, :n, -1], torch.ones(N_HEADS, n), atol=1e-5)
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_the_performer_blocks_exactly_what_the_dense_mask_blocks():
    n = 9
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    emb = torch.randn(n, N_EMBED)
    dense = _module(long_range=True, hops=2).eval()
    performer = _module_with(_performer_cfg(), long_range=True, hops=2).eval()
    _, _, _, dense_mask, _ = dense.common_step_local_to_global(batch, emb, eval_step=True)
    padded, padding, _, mask, rotary = performer.common_step_local_to_global(batch, emb, eval_step=True)
    performer.forward(padded, padding, mask, register_hook=True, rotary=rotary)

    weights = performer.transformer_encoder.layers[0].get_attn_output_weights()[0]  # graph 0, head 0
    assert torch.equal(weights == 0, dense_mask[0])
    assert torch.allclose(weights.sum(-1), torch.ones(n + 1), atol=1e-5)


def test_the_factorised_training_path_matches_the_explicit_evaluation_path():
    n = 8
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    module = _module_with(_performer_cfg(), long_range=True, hops=2).eval()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, torch.randn(n, N_EMBED), True)
    fast = module.forward(padded, padding, mask, rotary=rotary)[0]
    explicit = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)[0]
    assert torch.allclose(fast, explicit, atol=1e-5)


def test_performer_training_steps_stay_finite():
    n = 10
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    emb = torch.randn(n, N_EMBED)
    target = torch.randn(n + 1, 1, N_EMBED)
    module = _module_with(_performer_cfg(), long_range=True, hops=2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
        loss = torch.nn.functional.mse_loss(module.forward(padded, padding, mask, rotary=rotary)[0], target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()


def test_relevance_runs_on_the_performer():
    n = 7
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    module = _module_with(_performer_cfg(), long_range=True, hops=1).eval()
    _, _, _, _, relevance = module.evaluate(batch, torch.randn(n, N_EMBED))
    assert relevance.shape == (n + 1, n + 1) and torch.isfinite(relevance).all()


def test_the_performer_starts_from_the_dense_models_weights_and_shifts_no_other():
    """At one seed and the default head width a performer model and a dense one share every
    initial weight -- projections included; only the random features are new."""
    torch.manual_seed(0)
    dense = _from_config(_cfg())
    torch.manual_seed(0)
    performer = _from_config(_cfg(kind="performer"))
    perf_state = performer.state_dict()
    for key, value in dense.state_dict().items():
        assert torch.equal(value, perf_state[key]), key
    assert {k for k in perf_state if k not in dense.state_dict()} == {
        f"transformer_encoder.layers.{i}.self_attn.favor_projection" for i in range(2)
    }


def test_random_features_come_from_their_own_seeded_generator():
    torch.manual_seed(0)
    a = _from_config(_cfg(kind="performer"))
    rng_after_a = torch.get_rng_state()
    torch.manual_seed(0)
    _from_config(_cfg())
    assert torch.equal(rng_after_a, torch.get_rng_state())  # the global RNG moved as for dense

    torch.manual_seed(123)  # a different global state: the features do not depend on it
    b = _from_config(_cfg(kind="performer"))
    feats = [layer.self_attn.favor_projection for layer in a.transformer_encoder.layers]
    assert torch.equal(feats[0], b.transformer_encoder.layers[0].self_attn.favor_projection)
    assert not torch.equal(feats[0], feats[1])  # each layer its own draw


def test_features_are_redrawn_only_in_training_and_only_when_asked():
    n = 6
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    emb = torch.randn(n, N_EMBED)

    def run(module, steps):
        layer = module.transformer_encoder.layers[0].self_attn
        before = layer.favor_projection.clone()
        for _ in range(steps):
            gi = module.common_step_local_to_global(batch, emb, eval_step=True)
            module.forward(gi[0], gi[1], gi[3], rotary=gi[4])
        return before, layer.favor_projection

    before, after = run(_module_with(_performer_cfg(redraw_interval=2)).train(), 3)
    assert not torch.equal(before, after)
    before, after = run(_module_with(_performer_cfg(redraw_interval=2)).eval(), 3)
    assert torch.equal(before, after)
    before, after = run(_module_with(_performer_cfg(redraw_interval=0)).train(), 3)
    assert torch.equal(before, after)


def _data_batch(sizes=(6, 9), seed=0):
    """A PyG batch of graphs carrying precomputed embeddings (as in test_step_output)."""
    from torch_geometric.data import Batch, Data

    rng = np.random.default_rng(seed)
    datas = []
    for n in sizes:
        d = Data(x=torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32), edge_index=_path_graph(n))
        d.embeddings = torch.tensor(rng.normal(size=(n, N_EMBED)), dtype=torch.float32)
        d.mask = torch.tensor(rng.random(n) < 0.4)
        d.mask[0] = True
        datas.append(d)
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def test_the_performer_trains_through_a_full_step():
    """End to end through _common_step, the path training takes."""
    module = TransformerNodeEncoderHook(
        max_seq_len=32,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=16,
        long_range_attention=False,
        local_mask_hops=0,
        n_input=8,
        n_output=8,
        n_embed=8,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="node",
        type_gex_embedding=None,
        attention=_performer_cfg(),
    )
    out = module._common_step(_data_batch(), "regression", "node")
    loss = torch.nn.functional.mse_loss(out.y_pred, out.y_true)
    loss.backward()
    assert torch.isfinite(loss) and out.attn is None


def test_the_validator_takes_rope_and_rejects_dense_only_biases_and_the_entropy_loss_for_the_performer():
    from interscale.config import _validate_attention

    _validate_attention(_cfg(kind="performer", rotary=["rope"]))
    _validate_attention(_cfg(kind="performer", bias=["spectral"]))  # low rank: folded into q and k
    with pytest.raises(ValueError, match="cannot carry the attention bias 'distance'"):
        _validate_attention(_cfg(kind="performer", bias=["distance"]))
    with pytest.raises(ValueError, match="reads the attention maps in training"):
        _validate_attention(_cfg(kind="performer", loss="SCE_EntropyATT_Loss"))
    cfg = _cfg(kind="performer")
    cfg.model.global_component.parameters.attention.performer.num_features = 0
    with pytest.raises(ValueError, match="num_features must be > 0"):
        _validate_attention(cfg)


def test_the_checkpoint_name_carries_the_performer():
    from interscale.tl.utils import _attention_tag

    assert _attention_tag(_cfg(kind="performer")) == "attn-performer_"


# --------------------------------------------------------------------------- low-rank biases


def _spectral_batch(sizes=(9, 6), k=4, seed=0):
    """Two path graphs with precomputed spectral inputs, as attach_positional_inputs leaves them."""
    from interscale.tl.positional import laplacian_spectrum

    vecs, vals, edges, offset = [], [], [], 0
    for n in sizes:
        edge_index = _path_graph(n)
        v, lam = laplacian_spectrum(edge_index, n, k)
        vecs.append(v * math.sqrt(n))
        vals.append(lam.unsqueeze(0))
        edges.append(edge_index + offset)
        offset += n
    n_total = sum(sizes)
    batch = _Batch(
        torch.cat(edges, dim=1),
        torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)]),
        n_total,
        pos=torch.rand(n_total, 2, generator=torch.Generator().manual_seed(seed)) * 1000,
    )
    batch.spectral_pe = torch.cat(vecs).float()
    batch.spectral_eigval = torch.cat(vals).float()
    return batch


def _spectral_bias(k=4, seed=1, n_heads=N_HEADS, scale=1.0):
    from interscale.module.global_modules.positional_encodings import AttentionBias, SpectralBias

    encoder = SpectralBias(n_heads, 8, 1e-3, 2.0, k=k)
    with torch.no_grad():
        encoder.table.copy_(torch.randn(encoder.table.shape, generator=torch.Generator().manual_seed(seed)) * scale)
    return AttentionBias({"spectral": encoder}, n_heads=n_heads, spatial_unit_um=1.0)


def _padded_layout(batch, max_seq_len):
    from interscale.tl.padding import pad_batch

    _, padding, index_nodes, *_ = pad_batch(torch.zeros(batch.num_nodes, 1), batch.batch, max_seq_len)
    return padding, index_nodes


def test_the_spectral_bias_in_factors_is_the_dense_bias():
    """U W^T equals the dense bias entry for entry -- CLS and padding included (both zero) -- on two
    graphs of different size, one of them subsampled by pad_batch."""
    batch = _spectral_batch()
    bias = _spectral_bias()
    padding, index_nodes = _padded_layout(batch, max_seq_len=7)
    dense = bias(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))
    left, right = bias.factors(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))

    assert left.shape == (2 * N_HEADS, 8, 4)
    assert torch.allclose(left @ right.transpose(1, 2), dense, atol=1e-5)
    assert torch.all(left[:, -1] == 0) and torch.all(right[:, -1] == 0)  # CLS
    assert torch.all(left[N_HEADS:, :1] == 0)  # the second graph's padding


def test_a_zero_bias_has_a_zero_side_and_a_small_one():
    """Every bias starts at zero; split by balanced_factors, one side is 0 and the other small, so
    folding it into a Performer's queries and keys adds almost no variance at the first step."""
    batch = _spectral_batch()
    bias = _spectral_bias(scale=0.0)
    padding, index_nodes = _padded_layout(batch, max_seq_len=16)
    left, right = bias.factors(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))
    assert torch.all(left == 0)
    assert right.norm(dim=-1).max() < 0.5


def test_the_performer_folds_the_spectral_bias_into_the_kernel_it_estimates():
    """A performer's map converges to dense softmax attention with the same bias as the features
    grow -- the bias reached the kernel, not just the shapes. (Its error falls like 1/sqrt(m); a
    bias of a few units inflates the estimator's variance, so it needs more features: measured
    TV 0.014 at bias +-1.2 and 0.05 at +-3, both at m = 16384.)"""
    from interscale.module.global_modules.attention import PerformerAttention

    batch = _spectral_batch(sizes=(9,))
    bias = _spectral_bias(scale=0.3)
    padding, index_nodes = _padded_layout(batch, max_seq_len=16)
    dense_bias = bias(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))
    factors = bias.factors(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))
    q, k, v = _qkv(n_graphs=1, seq_len=10, scale=0.6)
    blocked = _diagonal_pairs(1, 9)

    graph, query, key = blocked
    logits = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1]) + dense_bias
    rows = (graph[:, None] * N_HEADS + torch.arange(N_HEADS)).reshape(-1)
    logits[rows, query.repeat_interleave(N_HEADS), key.repeat_interleave(N_HEADS)] = float("-inf")
    exact = torch.softmax(logits, dim=-1)

    def error(m):
        errs = []
        for seed in range(4):
            module = PerformerAttention(N_EMBED, N_HEADS, q.shape[-1], num_features=m, bias_factor_dim=4)
            module.draw_features(torch.Generator().manual_seed(seed))
            _, weights = module._attend(q, k, v, None, None, True, StructuredMask(blocked, factors))
            errs.append((weights - exact).abs().sum(-1).mean().item() / 2)
        return np.mean(errs)

    without = torch.softmax(logits - dense_bias, dim=-1)  # the same attention without the bias
    few, many = error(1024), error(16384)
    assert many < 0.025
    assert many < few / 2
    assert many < (without - exact).abs().sum(-1).mean().item() / 2 / 3


def test_a_performer_with_the_spectral_bias_runs_end_to_end():
    from interscale.module.global_modules.positional_encodings import AttentionBias, SpectralBias

    batch = _spectral_batch()
    encoder = SpectralBias(N_HEADS, 8, 1e-3, 2.0, k=4)
    module = _module_with(
        _performer_cfg(),
        long_range=True,
        hops=1,
        attention_bias=AttentionBias({"spectral": encoder}, n_heads=N_HEADS, spatial_unit_um=1.0),
    ).eval()
    emb = torch.randn(batch.num_nodes, N_EMBED)
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
    assert mask.bias_factors is not None and mask.bias_factors[0].shape[-1] == 4
    fast = module.forward(padded, padding, mask, rotary=rotary)[0]
    out, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    out.sum().backward()
    assert torch.allclose(fast, out, atol=1e-5)
    assert torch.isfinite(attn).all() and encoder.table.grad is not None


# --------------------------------------------------------------------------- the Fourier bias


def _fourier(isotropic=True, seed=2, scale=0.5, directions=8, scales=16, n_heads=N_HEADS):
    from interscale.module.global_modules.positional_encodings import FourierBias

    bias = FourierBias(n_heads, directions, scales, 200.0, 12000.0, isotropic=isotropic)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in bias.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)
    return bias


def _fourier_attention_bias(**kwargs):
    from interscale.module.global_modules.positional_encodings import AttentionBias

    return AttentionBias({"fourier": _fourier(**kwargs)}, n_heads=N_HEADS, spatial_unit_um=1.0)


def test_the_fourier_bias_starts_at_zero():
    from interscale.module.global_modules.positional_encodings import FourierBias

    for isotropic in (True, False):
        bias = FourierBias(N_HEADS, 8, 16, 200.0, 12000.0, isotropic=isotropic)
        pos = torch.rand(1, 6, 2) * 3000
        assert torch.all(bias(pos) == 0)


def test_the_fourier_bias_in_factors_is_its_dense_bias_laid_out_like_the_tokens():
    """Through AttentionBias: the factors' product equals the dense bias on two graphs of different
    size, one subsampled, with CLS and padding zero -- and each entry is the learned function of the
    two cells' offset."""
    batch = _spectral_batch()
    bias = _fourier_attention_bias(isotropic=False)
    padding, index_nodes = _padded_layout(batch, max_seq_len=7)
    dense = bias(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))
    left, right = bias.factors(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))

    assert left.shape[-1] == 2 * 8 * 16
    assert torch.allclose(left @ right.transpose(1, 2), dense, atol=1e-4)
    assert torch.all(dense[:, -1] == 0) and torch.all(dense[:, :, -1] == 0)  # CLS
    # graph 0 keeps 7 of its 9 cells: entry (i, j) is b(p_j - p_i) for the kept cells.
    pos = batch.pos[:9][torch.as_tensor(index_nodes[0])]
    offsets = (pos[None, :, :] - pos[:, None, :]).reshape(-1, 2)
    expected = bias.encoders["fourier"].bias_at(offsets).reshape(N_HEADS, 7, 7)
    assert torch.allclose(dense[:N_HEADS, :7, :7], expected, atol=1e-4)


def test_the_fourier_bias_depends_on_offsets_only():
    bias = _fourier(isotropic=False)
    pos = torch.rand(1, 8, 2, generator=torch.Generator().manual_seed(0)) * 3000
    shifted = pos + torch.tensor([4000.0, -2500.0])
    assert torch.allclose(bias(pos), bias(shifted), atol=1e-4)


def test_the_isotropic_fourier_bias_is_symmetric_and_turns_with_its_directions():
    """Isotropic: no sines, so b(Δ) = b(-Δ); and the direction grid maps onto itself under a turn
    by 180° / directions, so the bias does too. The anisotropic one is directional and asymmetric."""
    offsets = torch.rand(50, 2, generator=torch.Generator().manual_seed(1)) * 4000 - 2000
    iso = _fourier(isotropic=True)
    assert torch.allclose(iso.bias_at(offsets), iso.bias_at(-offsets), atol=1e-5)
    turn = math.pi / 8
    rotation = torch.tensor([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    assert torch.allclose(iso.bias_at(offsets), iso.bias_at(offsets @ rotation.T), atol=1e-4)

    aniso = _fourier(isotropic=False)
    assert not torch.allclose(aniso.bias_at(offsets), aniso.bias_at(-offsets), atol=1e-2)


def test_the_isotropic_fourier_bias_can_single_out_a_distance():
    """Its coefficients fitted by (ridge) least squares, the default grid reproduces a ring at 1 mm
    within 3 mm -- the distance profile's job, in a form a Performer can carry. The ridge keeps the
    coefficients at the scale a model would learn (< 1): plain least squares reaches ~1e9 on the
    nearly constant longest waves, a cancellation no float32 bias could hold."""
    from interscale.module.global_modules.positional_encodings import FourierBias

    bias = FourierBias(1, 8, 16, 200.0, 12000.0, isotropic=True)
    gen = torch.Generator().manual_seed(0)
    radius = torch.rand(3000, generator=gen, dtype=torch.float64) * 3000
    angle = torch.rand(3000, generator=gen, dtype=torch.float64) * 2 * math.pi
    offsets = torch.stack([radius * angle.cos(), radius * angle.sin()], dim=-1)
    target = torch.exp(-(((radius - 1000) / 300) ** 2))

    phase = offsets @ bias.frequencies.double().T  # [n, scales * directions]
    per_scale = phase.cos().reshape(len(offsets), 16, 8).mean(-1)  # what cos_coef multiplies
    design = per_scale - per_scale.mean(0)  # a constant shift is irrelevant to a softmax
    gram = design.T @ design + 1e-4 * len(design) * torch.eye(16, dtype=torch.float64)
    coef = torch.linalg.solve(gram, design.T @ (target - target.mean())[:, None])
    assert coef.abs().max() < 1
    with torch.no_grad():
        bias.cos_coef.copy_(coef.T.float())
    fitted = bias.bias_at(offsets.float())[0].double()
    residual = (target - target.mean()) - (fitted - fitted.mean())
    assert 1 - residual.var() / target.var() > 0.9


def test_a_performer_carries_the_fourier_bias_end_to_end():
    batch = _spectral_batch()
    attention_bias = _fourier_attention_bias(scale=0.05)
    module = _module_with(_performer_cfg(), long_range=True, hops=1, attention_bias=attention_bias).eval()
    emb = torch.randn(batch.num_nodes, N_EMBED)
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
    assert mask.bias_factors[0].shape[-1] == 256
    fast = module.forward(padded, padding, mask, rotary=rotary)[0]
    out, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    out.sum().backward()
    assert torch.allclose(fast, out, atol=1e-5)
    assert torch.isfinite(attn).all()
    assert attention_bias.encoders["fourier"].cos_coef.grad.abs().sum() > 0


def test_a_dense_attention_carries_the_fourier_bias_too():
    batch = _spectral_batch()
    attention_bias = _fourier_attention_bias()
    module = _module(long_range=True, hops=1, attention_bias=attention_bias).eval()
    emb = torch.randn(batch.num_nodes, N_EMBED)
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
    assert mask.is_floating_point()  # merged into the float mask, like the other biases
    out = module.forward(padded, padding, mask, rotary=rotary)[0]
    assert torch.isfinite(out).all()


def test_the_validator_and_the_config_take_the_fourier_bias():
    from interscale.config import _validate_attention, _validate_pe

    cfg = _cfg(kind="performer", bias=["fourier"])
    _validate_attention(cfg)
    _validate_pe(cfg)
    cfg.dataset.spatial_key = None
    with pytest.raises(ValueError, match="read cell coordinates"):
        _validate_pe(cfg)
    cfg = _cfg(bias=["fourier"])
    cfg.model.global_component.parameters.pe.fourier.num_scales = 0
    with pytest.raises(ValueError, match="fourier.num_scales must be > 0"):
        _validate_pe(cfg)


def test_fourier_max_wavelength_zero_is_twice_the_largest_slide_diameter():
    import pandas as pd
    from anndata import AnnData

    from interscale.module.global_modules.positional_encodings import resolve_fourier_range

    coords = np.array([[0, 0], [30, 0], [0, 40], [30, 40], [1000, 0], [1006, 8]], dtype=np.float32)
    adata = AnnData(X=np.zeros((6, 3), dtype=np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["sample"] = pd.Categorical(["a"] * 4 + ["b"] * 2)
    cfg = _cfg(bias=["fourier"])
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.spatial_unit_um = 2.0
    cfg.model.global_component.parameters.pe.fourier.min_wavelength = 10.0
    cfg.freeze()

    resolved = resolve_fourier_range(cfg, adata)
    assert resolved.model.global_component.parameters.pe.fourier.max_wavelength == pytest.approx(200.0)
    assert cfg.model.global_component.parameters.pe.fourier.max_wavelength == 0.0


def test_the_checkpoint_name_carries_the_fourier_bias():
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _cfg(kind="performer", bias=["fourier"])
    cfg.dataset.name = "d"
    assert get_model_filename_prefix(cfg, False, True).endswith("pe-fourier_attn-performer_")


# --------------------------------------------------------------------------- linear attention


def _linear_cfg(head_dim=0):
    return _Att(kind="linear", head_dim=head_dim)


def _linear(head_dim=N_EMBED // N_HEADS, seed=0):
    from interscale.module.global_modules.attention import LinearAttention

    torch.manual_seed(seed)
    return LinearAttention(N_EMBED, N_HEADS, head_dim)


def _no_pairs():
    return torch.zeros(3, 0, dtype=torch.long)


def test_linear_attention_is_its_definition():
    """weights_ij = phi(q_i) . phi(k_j) / sum_l phi(q_i) . phi(k_l) with phi = elu + 1, blocked pairs zero."""
    q, k, v = _qkv(seq_len=9)
    blocked = _diagonal_pairs(2, 8)
    out, weights = _linear()._attend(q, k, v, None, None, True, StructuredMask(blocked))

    kernel = (torch.nn.functional.elu(q) + 1) @ (torch.nn.functional.elu(k) + 1).transpose(1, 2)
    graph, query, key = blocked
    rows = (graph[:, None] * N_HEADS + torch.arange(N_HEADS)).reshape(-1)
    kernel[rows, query.repeat_interleave(N_HEADS), key.repeat_interleave(N_HEADS)] = 0
    expected = kernel / kernel.sum(-1, keepdim=True)
    assert torch.allclose(weights, expected, atol=1e-6)
    assert torch.allclose(out, expected @ v, atol=1e-5)


def test_linear_attention_maps_have_rank_at_most_the_head_width():
    """The bound behind the head-width warning: with phi as wide as the head, each head's map is a
    mix of head_dim key profiles. (The mask's zeroed pairs are a separate, sparse correction; this
    is the map before it.) The maps are float32, so the rank counts singular values above 1e-5 of
    the largest: the linear map has head_dim of them at ~3e-2 or more and the rest at ~1e-8."""
    seq_len = 40
    for head_dim in (4, 8):
        q, k, v = _qkv(n_graphs=1, seq_len=seq_len, head_dim=head_dim)
        _, weights = _linear(head_dim)._attend(q, k, v, None, None, True, StructuredMask(_no_pairs()))
        ranks = [int(torch.linalg.matrix_rank(weights[row].double(), rtol=1e-5)) for row in range(N_HEADS)]
        assert ranks == [head_dim] * N_HEADS

    # Dense softmax at the same width is not bound: exp of a rank-4 logit matrix is (near) full rank.
    q, k, _ = _qkv(n_graphs=1, seq_len=seq_len, head_dim=4)
    dense = torch.softmax(q @ k.transpose(1, 2) / 2, dim=-1)
    assert int(torch.linalg.matrix_rank(dense[0].double(), rtol=1e-5)) > 30


def test_linear_attention_factorised_and_explicit_paths_agree_with_mask_and_padding():
    from interscale.tl.masking import blocked_pairs

    q, k, v = _qkv(n_graphs=2, seq_len=9)
    padding = torch.zeros(2, 9, dtype=torch.bool)
    padding[1, :3] = True
    edge_index = torch.cat([_path_graph(8), _path_graph(5, offset=8)], dim=1)
    batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    blocked = blocked_pairs(edge_index, batch, [list(range(8)), list(range(5))], n_hops=2)
    module = _linear()
    out_fact, none = module._attend(q, k, v, padding, None, False, StructuredMask(blocked))
    out_expl, weights = module._attend(q, k, v, padding, None, True, StructuredMask(blocked))
    assert none is None
    assert torch.allclose(out_fact, out_expl, atol=1e-5)
    assert torch.all(weights[N_HEADS:][:, :, :3] == 0)  # padded keys of the second graph


def test_linear_attention_blocks_exactly_what_the_dense_mask_blocks():
    n = 9
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    emb = torch.randn(n, N_EMBED)
    dense = _module(long_range=True, hops=2).eval()
    linear = _module_with(_linear_cfg(), long_range=True, hops=2).eval()
    _, _, _, dense_mask, _ = dense.common_step_local_to_global(batch, emb, eval_step=True)
    padded, padding, _, mask, rotary = linear.common_step_local_to_global(batch, emb, eval_step=True)
    linear.forward(padded, padding, mask, register_hook=True, rotary=rotary)

    weights = linear.transformer_encoder.layers[0].get_attn_output_weights()[0]
    assert torch.equal(weights == 0, dense_mask[0])
    assert torch.allclose(weights.sum(-1), torch.ones(n + 1), atol=1e-5)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_linear_attention_is_finite_on_a_dense_graph_under_the_long_range_mask(hops):
    n = 12
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    module = _module_with(_linear_cfg(), long_range=True, hops=hops).eval()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, torch.randn(n, N_EMBED), True)
    out, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    out.sum().backward()

    assert torch.isfinite(out).all() and torch.isfinite(attn).all()
    assert torch.allclose(attn[0, :, :n, -1], torch.ones(N_HEADS, n), atol=1e-5)  # CLS alone is open
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_linear_attention_trains_and_feeds_relevance():
    n = 10
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    emb = torch.randn(n, N_EMBED)
    target = torch.randn(n + 1, 1, N_EMBED)
    module = _module_with(_linear_cfg(), long_range=True, hops=2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
        loss = torch.nn.functional.mse_loss(module.forward(padded, padding, mask, rotary=rotary)[0], target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    module.eval()
    path = _Batch(_path_graph(7), torch.zeros(7, dtype=torch.long), 7)
    _, _, _, _, relevance = module.evaluate(path, torch.randn(7, N_EMBED))
    assert relevance.shape == (8, 8) and torch.isfinite(relevance).all()


def test_linear_attention_trains_through_a_full_step():
    module = TransformerNodeEncoderHook(
        max_seq_len=32,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=16,
        long_range_attention=False,
        local_mask_hops=0,
        n_input=8,
        n_output=8,
        n_embed=8,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="node",
        type_gex_embedding=None,
        attention=_linear_cfg(),
    )
    out = module._common_step(_data_batch(), "regression", "node")
    loss = torch.nn.functional.mse_loss(out.y_pred, out.y_true)
    loss.backward()
    assert torch.isfinite(loss) and out.attn is None


def test_linear_attention_adds_no_state_and_starts_from_the_dense_weights():
    """No features, no buffers: the state dict is the dense model's, weight for weight at one seed,
    and the global RNG moves as for dense -- a dense checkpoint loads into it strictly."""
    torch.manual_seed(0)
    dense = _from_config(_cfg())
    rng_dense = torch.get_rng_state()
    torch.manual_seed(0)
    linear = _from_config(_cfg(kind="linear"))
    assert torch.equal(rng_dense, torch.get_rng_state())

    dense_state, linear_state = dense.state_dict(), linear.state_dict()
    assert dense_state.keys() == linear_state.keys()
    for key, value in dense_state.items():
        assert torch.equal(value, linear_state[key]), key
    linear.load_state_dict(dense_state, strict=True)


def test_linear_attention_refuses_a_rotation_and_a_bias():
    q, k, v = _qkv(n_graphs=1, seq_len=5)
    module = _linear()
    x = torch.randn(5, 1, N_EMBED)
    cos, sin = torch.ones(N_HEADS, 5, N_EMBED // N_HEADS), torch.zeros(N_HEADS, 5, N_EMBED // N_HEADS)
    with pytest.raises(ValueError, match="does not apply RoPE"):
        module(x, x, x, rotary=(cos, sin), context=StructuredMask(_no_pairs()))
    factors = (torch.zeros(N_HEADS, 5, 2), torch.zeros(N_HEADS, 5, 2))
    with pytest.raises(ValueError, match="carries no attention bias"):
        module._attend(q, k, v, None, None, False, StructuredMask(_no_pairs(), factors))


def test_the_validator_explains_what_linear_attention_cannot_carry():
    from interscale.config import _validate_attention

    _validate_attention(_cfg(kind="linear"))
    for bias in ("distance", "spectral", "fourier"):
        with pytest.raises(ValueError, match="has no exponential") as err:
            _validate_attention(_cfg(kind="linear", bias=[bias]))
        assert "performer" in str(err.value)
    with pytest.raises(ValueError, match="cannot apply pe.rotary") as err:
        _validate_attention(_cfg(kind="linear", rotary=["rope"]))
    assert "has no exponential" in str(err.value)
    with pytest.raises(ValueError, match="reads the attention maps in training"):
        _validate_attention(_cfg(kind="linear", loss="SCE_EntropyATT_Loss"))


def test_narrow_linear_heads_warn_and_wide_ones_do_not():
    import warnings

    with pytest.warns(UserWarning, match="rank <= 4"):
        _module_with(_linear_cfg())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _module_with(_linear_cfg(head_dim=32))
    assert not [w for w in caught if "rank <=" in str(w.message)]


def test_wide_linear_heads_run_under_the_long_range_mask():
    n = 9
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n)
    module = _module_with(_linear_cfg(head_dim=32), long_range=True, hops=2).eval()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, torch.randn(n, N_EMBED), True)
    fast = module.forward(padded, padding, mask, rotary=rotary)[0]
    explicit, _, attn = module.forward(padded, padding, mask, register_hook=True, rotary=rotary)
    assert module.transformer_encoder.layers[0].self_attn.head_dim == 32
    assert torch.allclose(fast, explicit, atol=1e-5) and torch.isfinite(attn).all()


def test_the_checkpoint_name_carries_linear_attention():
    from interscale.tl.utils import _attention_tag

    assert _attention_tag(_cfg(kind="linear")) == "attn-linear_"
    assert _attention_tag(_cfg(kind="linear", head_dim=32)) == "attn-linear-hd32_"


# --------------------------------------------------------------------------- the pair form of the biases


def _every_pair(n_graphs, seq_len):
    """``[3, P]`` every (graph, query, key) of the padded layout, the CLS slot included."""
    g, q, k = torch.meshgrid(
        torch.arange(n_graphs), torch.arange(seq_len + 1), torch.arange(seq_len + 1), indexing="ij"
    )
    return torch.stack([g.reshape(-1), q.reshape(-1), k.reshape(-1)])


def _bias_encoders():
    from interscale.module.global_modules.positional_encodings import (
        DistanceBias,
        LinearDistanceBias,
        SpectralBias,
    )

    gen = torch.Generator().manual_seed(4)
    encoders = {
        "distance": DistanceBias(N_HEADS, 8, 1500.0),
        "linear": LinearDistanceBias(N_HEADS),
        "spectral": SpectralBias(N_HEADS, 8, 1e-3, 2.0, k=4),
        "fourier": _fourier(isotropic=False, scale=0.0),
    }
    for encoder in encoders.values():
        with torch.no_grad():
            for p in encoder.parameters():
                p.copy_(torch.randn(p.shape, generator=gen))
    return encoders


@pytest.mark.parametrize("name", ["distance", "linear", "spectral", "fourier"])
def test_the_pair_form_is_the_dense_bias_at_every_pair(name):
    """Each bias read off a list of pairs equals the dense bias at those pairs -- CLS and padding
    included (both 0) -- on two graphs of different size, one subsampled by pad_batch."""
    from interscale.module.global_modules.positional_encodings import AttentionBias

    registry_name = {"linear": "distance"}.get(name, name)
    bias = AttentionBias({registry_name: _bias_encoders()[name]}, n_heads=N_HEADS, spatial_unit_um=1.0)
    batch = _spectral_batch()
    padding, index_nodes = _padded_layout(batch, max_seq_len=7)
    dense = bias(batch, index_nodes, padding, dtype=torch.float32, device=torch.device("cpu"))
    pairs = _every_pair(2, 7)
    values = bias.pairs(batch, index_nodes, padding, pairs, dtype=torch.float32, device=torch.device("cpu"))

    expected = dense.reshape(2, N_HEADS, 8, 8)[pairs[0], :, pairs[1], pairs[2]]  # [P, heads]
    assert values.shape == (pairs.shape[1], N_HEADS)
    assert torch.allclose(values, expected, atol=1e-4)
    assert values.abs().sum() > 0  # the comparison is not between two zero tensors


def test_the_pair_form_carries_the_gradient():
    from interscale.module.global_modules.positional_encodings import AttentionBias

    encoders = _bias_encoders()
    bias = AttentionBias(
        {"distance": encoders["distance"], "spectral": encoders["spectral"], "fourier": encoders["fourier"]},
        n_heads=N_HEADS,
        spatial_unit_um=1.0,
    )
    batch = _spectral_batch()
    padding, index_nodes = _padded_layout(batch, max_seq_len=16)
    values = bias.pairs(batch, index_nodes, padding, _every_pair(2, 9), dtype=torch.float32, device=torch.device("cpu"))
    values.sum().backward()
    for name in ("distance", "spectral"):
        assert bias.encoders[name].table.grad.abs().sum() > 0
    assert bias.encoders["fourier"].cos_coef.grad.abs().sum() > 0
