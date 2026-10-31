"""A bias for a set of heads (``pe.bias_heads``) and given window scales (``pe.distance.scales_um``); attention_plan.md, stage 10a.

Pinned here:

* **placement** -- a head a bias does not serve gets exactly 0 from it, in the dense, pair, row and factor forms and in the
  hierarchy's ``offset_bias``; the served heads get what a bias built for them alone gives;
* **dealing** -- the scales prior spaces its windows over the heads it serves, or takes ``scales_um``; each encoder names the
  heads it serves (``served_heads``);
* **through the model** -- with a free head, the fused model is the dense one;
* **the switch** -- off by default (every name and number unchanged), validator, checkpoint tag;
* **scales from the data** -- ``gene_scales.window_scales`` reads the windows off a measured length-scale table.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.module.global_modules.positional_encodings import ScaledDistanceBias, build_attention_bias
from interscale.tl import pad_batch

H = 4


def _stir(module, seed=0, scale=0.3):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _cfg(bias=("distance", "fourier"), heads=None, scales_um=(), n_heads=H):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.n_heads = n_heads
    pe = params.pe
    pe.bias = list(bias)
    pe.distance.kind = "scales"
    pe.distance.scale_min_um, pe.distance.scale_max_um = 50.0, 800.0
    pe.distance.scales_um = list(scales_um)
    pe.fourier.max_wavelength = 2000.0
    pe.fourier.isotropic = False
    for name, chosen in (heads or {}).items():
        pe.bias_heads[name] = list(chosen)
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", 1.0
    return cfg


def _graphs():
    gen = torch.Generator().manual_seed(0)
    graphs = [Data(pos=torch.rand(n, 2, generator=gen) * 600.0, x=torch.zeros(n, 1)) for n in (7, 5)]
    batch = Batch.from_data_list(graphs)
    _, padding, index, *_ = pad_batch(torch.zeros(batch.num_nodes, 2), batch.batch, 64)
    return batch, index, padding


def _layout(bias):
    batch, index, padding = _graphs()
    return bias(batch, index, padding, torch.float32, torch.device("cpu")), (batch, index, padding)


# --------------------------------------------------------------------------- placement


def test_a_head_a_bias_does_not_serve_gets_zero_and_the_served_heads_get_the_bias_alone():
    torch.manual_seed(0)
    split = _stir(build_attention_bias(_cfg(heads={"distance": [0, 2], "fourier": [3]}), H))
    dense, (batch, index, padding) = _layout(split)
    full = dense.view(2, H, *dense.shape[1:])
    assert (full[:, 1] == 0).all()  # head 1: no bias reaches it -- a free head
    # Head 0 and 2 get the distance prior built for two heads; head 3 the Fourier bias built for one.
    alone = build_attention_bias(_cfg(bias=("distance",), n_heads=2), 2)
    alone.encoders["distance"].load_state_dict(split.encoders["distance"].state_dict())
    want = alone(batch, index, padding, torch.float32, torch.device("cpu")).view(2, 2, *dense.shape[1:])
    assert torch.allclose(full[:, [0, 2]], want, atol=1e-5)
    fourier = build_attention_bias(_cfg(bias=("fourier",), n_heads=1), 1)
    fourier.encoders["fourier"].load_state_dict(split.encoders["fourier"].state_dict())
    want = fourier(batch, index, padding, torch.float32, torch.device("cpu")).view(2, 1, *dense.shape[1:])
    assert torch.allclose(full[:, [3]], want, atol=1e-5)
    assert split.encoders["distance"].served_heads == [0, 2] and split.encoders["fourier"].served_heads == [3]


def test_every_form_is_the_dense_bias():
    torch.manual_seed(0)
    bias = _stir(build_attention_bias(_cfg(heads={"distance": [1, 2], "fourier": [0, 3]}), H))
    dense, (batch, index, padding) = _layout(bias)
    full = dense.view(2, H, *dense.shape[1:])
    seq = padding.shape[1]
    # pairs: every (graph, query, key) of the padded layout
    g, q, k = torch.meshgrid(torch.arange(2), torch.arange(seq), torch.arange(seq), indexing="ij")
    pairs = torch.stack([g.reshape(-1), q.reshape(-1), k.reshape(-1)])
    values = bias.pairs(batch, index, padding, pairs, torch.float32, torch.device("cpu"))
    assert torch.allclose(values.view(2, seq, seq, H).permute(0, 3, 1, 2), full[:, :, :seq, :seq], atol=1e-5)
    # rows, a block at a time
    read = bias.rows(batch, index, padding, torch.float32, torch.device("cpu"))
    rows = torch.cat([read(0, 3), read(3, seq + 1)], dim=2)
    assert torch.allclose(rows, full, atol=1e-5)
    # factors (the Fourier bias alone has them)
    fourier = _stir(build_attention_bias(_cfg(bias=("fourier",), heads={"fourier": [0, 3]}), H))
    left, right = fourier.factors(batch, index, padding, torch.float32, torch.device("cpu"))
    dense_f, _ = _layout(fourier)
    assert torch.allclose(left @ right.transpose(-1, -2), dense_f, atol=1e-4)
    assert (left.view(2, H, *left.shape[1:])[:, [1, 2]] == 0).all()


def test_the_hierarchy_places_its_offsets_alike():
    from interscale.module.global_modules.hierarchical_attention import offset_bias

    bias = _stir(build_attention_bias(_cfg(heads={"distance": [0, 2], "fourier": [3]}), H))
    offsets = torch.randn(5, 6, 2) * 200.0
    out = offset_bias(bias, offsets)
    assert out.shape == (5, 6, H) and (out[..., 1] == 0).all()
    alone = build_attention_bias(_cfg(bias=("distance",), n_heads=2), 2)
    alone.encoders["distance"].load_state_dict(bias.encoders["distance"].state_dict())
    distance_only = offset_bias(alone, offsets)
    fourier_only = bias.encoders["fourier"]
    phase = offsets.double() @ fourier_only.frequencies.double().T
    a, c = fourier_only.coefficients()
    assert torch.allclose(out[..., [0, 2]], distance_only, atol=1e-5)
    assert torch.allclose(out[..., 3], (phase.cos().float() @ a.T + phase.sin().float() @ c.T)[..., 0], atol=1e-4)


class _PairsOnly(torch.nn.Module):
    """A distance profile with its pair form alone, read from coordinates: the generic route of ``rows``."""

    reads_coordinates = True

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, pos_um):
        return self.inner(torch.cdist(pos_um, pos_um, compute_mode="donot_use_mm_for_euclid_dist"))

    def pairs(self, pos_um, graph, query, key):
        return self.inner.pairs(pos_um, graph, query, key)


def test_a_bias_with_a_pair_form_alone_is_placed_by_rows_too():
    from interscale.module.global_modules.positional_encodings import AttentionBias, DistanceBias

    encoder = _stir(_PairsOnly(DistanceBias(2, 8, 1500.0)))
    bias = AttentionBias({"distance": encoder}, n_heads=H, spatial_unit_um=1.0, heads={"distance": [1, 3]})
    dense, (batch, index, padding) = _layout(bias)
    read = bias.rows(batch, index, padding, torch.float32, torch.device("cpu"))
    rows = torch.cat([read(0, 4), read(4, padding.shape[1] + 1)], dim=2)
    full = dense.view(2, H, *dense.shape[1:])
    assert torch.allclose(rows, full, atol=1e-5) and (full[:, [0, 2]] == 0).all() and full[:, [1, 3]].abs().sum() > 0


# --------------------------------------------------------------------------- dealing


def test_the_windows_are_spaced_over_the_served_heads_or_given():
    bias = build_attention_bias(_cfg(bias=("distance",), heads={"distance": [0, 1, 3]}), H)
    assert torch.allclose(bias.encoders["distance"].scales(), torch.tensor([50.0, 200.0, 800.0]), rtol=1e-5)
    given = build_attention_bias(_cfg(bias=("distance",), heads={"distance": [1, 3]}, scales_um=[15.0, 400.0]), H)
    assert torch.allclose(given.encoders["distance"].scales(), torch.tensor([15.0, 400.0]), rtol=1e-5)
    every = build_attention_bias(_cfg(bias=("distance",), scales_um=[10.0, 20.0, 40.0, 80.0]), H)
    assert torch.allclose(every.encoders["distance"].scales(), torch.tensor([10.0, 20.0, 40.0, 80.0]), rtol=1e-5)
    with pytest.raises(ValueError, match="one positive scale per head"):
        ScaledDistanceBias(3, 10.0, 100.0, "gaussian", 1.0, False, scales_um=[1.0, 2.0])


# --------------------------------------------------------------------------- through the model


@pytest.mark.parametrize("chunk_pairs", [2**24, 37 * 2], ids=["one block", "blocks"])
def test_with_a_free_head_the_fused_model_is_the_dense_one(chunk_pairs):
    import importlib.util
    from pathlib import Path

    # The windowed GlobalModel builder of the visible-keys tests (tests run with --import-mode=importlib).
    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    vk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vk)

    def cfg(kind):
        c = vk._config(kind, visible=False, chunk_pairs=chunk_pairs)
        c.model.global_component.parameters.pe.bias = ["distance", "fourier"]
        c.model.global_component.parameters.pe.fourier.max_wavelength = 2000.0
        c.model.global_component.parameters.pe.bias_heads.distance = [0]
        c.model.global_component.parameters.pe.bias_heads.fourier = [1]
        return c

    dense, fused = vk._build(cfg("dense")), vk._build(cfg("fused"))
    vk._same(dense, fused)


# --------------------------------------------------------------------------- the switch


def test_off_by_default_nothing_changes_and_the_name_says_which_heads():
    from interscale.config import _validate_pe
    from interscale.tl.utils import _positional_encoding_tag

    plain = _cfg()
    assert all(not plain.model.global_component.parameters.pe.bias_heads[k] for k in ("distance", "fourier"))
    torch.manual_seed(0)
    a = build_attention_bias(plain, H)
    assert a.heads == {"distance": None, "fourier": None}
    assert _positional_encoding_tag(plain) == "pe-distance-scales-g50-800+fourier-aniso_"
    split = _cfg(heads={"distance": [2, 0], "fourier": [3]}, scales_um=[30.0, 300.0])
    _validate_pe(split)
    assert _positional_encoding_tag(split) == "pe-distance-scales-gl30.300@0-2+fourier-aniso@3_"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"heads": {"distance": [0, 4]}}, r"distinct heads in \[0, 4\)"),
        ({"heads": {"distance": [1, 1]}}, "distinct heads"),
        ({"heads": {"spectral": [0]}}, "does not switch 'spectral' on"),
        ({"scales_um": [10.0, 20.0]}, "one positive window per head the prior serves \\(4\\)"),
        ({"heads": {"distance": [0, 1]}, "scales_um": [10.0, -1.0]}, "one positive window"),
    ],
)
def test_the_validator_refuses_what_cannot_be_placed(change, match):
    from interscale.config import _validate_pe

    with pytest.raises(ValueError, match=match):
        _validate_pe(_cfg(**change))


# --------------------------------------------------------------------------- scales from the data


def test_the_windows_come_from_the_measured_length_scales_of_the_variable_genes():
    from interscale.evaluation.gene_scales import window_scales

    table = pd.DataFrame(
        {
            "ell_um": [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 5.0, np.nan, np.inf],
            "pattern": ["variable"] * 6 + ["flat", "variable", "variable"],
        }
    )
    scales = window_scales(table, 3)
    assert len(scales) == 3 and scales == sorted(scales)
    assert np.allclose(scales, np.quantile([10.0, 20.0, 40.0, 80.0, 160.0, 320.0], [1 / 6, 0.5, 5 / 6]))
    assert window_scales(table, 3, min_um=60.0)[0] == 60.0  # floored at a reach
    with pytest.raises(ValueError, match="no variable gene"):
        window_scales(table.assign(pattern="flat"), 2)
