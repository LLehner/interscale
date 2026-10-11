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
