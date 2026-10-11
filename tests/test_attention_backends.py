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
