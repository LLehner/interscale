"""The context query on the fused attention (``context_only: query`` with ``attention.kind: fused``; decoder_plan.md D1.2,
comparison_plan.md "Whole sections", route a; roadmap Tier 1, item 3).

Pinned here:

* **it is the dense read-out** -- at the same weights the fused query stream gives the dense one's global embeddings, maps and
  gradients: with node encodings, the Fourier bias, the distance window, RoPE, centred queries and keys, in one block and in
  many, on a subsampled slide;
* **blindness** -- a cell's embedding does not move, bit for bit, with its own expression, also in blocks and behind a GCN
  under the long-range mask;
* **the switch** -- the validator takes fused, still refuses the self-attention-only backends.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.module.base import GlobalModule
from interscale.module.base._step_output import gather_tokens

E, G = 8, 5
SPACING = 100.0
SIDE = 6
CELL = (SIDE // 2) * SIDE + SIDE // 2


def _cfg(kind, *, chunk_pairs=2**24, center=False, bias=("fourier",), rotary=(), node=("sinusoidal",), lrm=False, max_seq_len=128):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = max_seq_len, 2, 2, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.node, params.pe.bias, params.pe.rotary = list(node), list(bias), list(rotary)
    params.pe.fourier.max_wavelength = params.pe.rope.max_wavelength = params.pe.distance.max_dist = 2000.0
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    params.attention.kind = kind
    params.attention.fused.chunk_pairs = chunk_pairs
    params.attention.center_qk = center
    params.long_range_attention = lrm
    params.context_only = "query"
    cfg.model.n_embed = E
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um, cfg.dataset.name = "spatial", 1.0, "d"
    cfg.dataset.sample_key = ["sample"]
    return cfg


def _stir(module, scale=0.1, seed=7):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=gen))
    return module


def _module(kind, seed=0, **kwargs):
    torch.manual_seed(seed)
    return GlobalModule.from_config(
        _cfg(kind, **kwargs),
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
    batch = Batch.from_data_list([data])
    batch.obs_names = torch.arange(n)
    return batch


def _rows(module, batch, grad=False):
    """``(embeddings [N, E] in cell order, maps, parameter gradients)``: one pass, the maps recorded, a fixed weighting backed."""
    random.seed(0)
    module.zero_grad(set_to_none=True)
    tokens = module.input_embedding(batch.x.float())
    with torch.set_grad_enabled(grad), module.recording_attention() as record:
        out, padding, index, _, _ = module.step(batch, tokens)
    rows = gather_tokens(out, padding)
    order = torch.as_tensor(index[0], dtype=torch.long)
    grads = None
    if grad:
        weight = torch.randn(rows.shape, generator=torch.Generator().manual_seed(3))
        (rows * weight).sum().backward()
        grads = {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}
    cells = torch.full((batch.num_nodes, rows.shape[1]), float("nan"))
    cells[order] = rows.detach()
    return cells, record[0]["maps"], grads


def _changed(batch, cell, amount=3.0):
    other = batch.clone()
    other.x = batch.x.clone()
    other.x[cell] += amount
    return other


SETTINGS = [
    {},
    {"rotary": ("rope",)},
    {"bias": ("distance", "fourier"), "center": True},
    {"bias": ("distance",), "rotary": ("rope",), "node": ()},
]
IDS = ["node+fourier", "+rope", "window+fourier, centred", "window+rope, no node encoding"]


@pytest.mark.parametrize("settings", SETTINGS, ids=IDS)
@pytest.mark.parametrize(
    ("chunk_pairs", "max_seq_len"), [(2**24, 128), (37 * 3, 128), (25, 25)], ids=["one block", "blocks", "subsampled"]
)
def test_the_fused_read_out_is_the_dense_one(settings, chunk_pairs, max_seq_len):
    batch = _graph()
    dense = _stir(_module("dense", max_seq_len=max_seq_len, **settings))
    fused = _module("fused", chunk_pairs=chunk_pairs, max_seq_len=max_seq_len, **settings)
    missing, unexpected = fused.load_state_dict(dense.state_dict(), strict=False)
    assert not missing and not unexpected
    want, maps, grads_dense = _rows(dense, batch, grad=True)
    got, _, grads_fused = _rows(fused, batch, grad=True)
    kept = ~torch.isnan(want[:, 0])
    assert torch.equal(kept, ~torch.isnan(got[:, 0]))
    assert torch.allclose(got[kept], want[kept], atol=1e-5)
    assert set(grads_fused) == set(grads_dense)
    for name, value in grads_dense.items():
        assert torch.allclose(grads_fused[name], value, atol=1e-5, rtol=1e-4), name
    # The maps the read-out returns (the explicit path): the dense read-out's.
    with torch.no_grad():
        _, fused_maps, _ = _rows(fused, batch)
    assert fused_maps is not None and torch.allclose(fused_maps, maps, atol=1e-6)


@pytest.mark.parametrize("chunk_pairs", [2**24, 37 * 3], ids=["one block", "blocks"])
@pytest.mark.parametrize("settings", SETTINGS[:2] + SETTINGS[3:], ids=IDS[:2] + IDS[3:])
def test_a_cell_is_blind_to_itself_on_the_fused_attention(settings, chunk_pairs):
    batch = _graph()
    module = _stir(_module("fused", chunk_pairs=chunk_pairs, **settings))
    before, _, _ = _rows(module, batch)
    for cell in (0, CELL, batch.num_nodes - 1):
        after, _, _ = _rows(module, _changed(batch, cell, amount=-0.7))
        assert torch.equal(after[cell], before[cell]), cell
        others = [i for i in range(batch.num_nodes) if i != cell]
        assert (after[others] - before[others]).abs().max() > 0


# --------------------------------------------------------------------------- behind a GCN


def _combined(kind, *, chunk_pairs=2**24):
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.model.combined_model import CombinedModel

    cfg = _cfg(kind, chunk_pairs=chunk_pairs, lrm=True)
    cfg.model.local_component.name = "GCN"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg.model.decoder.dual_decoder = True
    cfg.model.decoder.additive = True
    cfg.optim.loss_weights["local"], cfg.optim.loss_weights["global"] = 0.1, 0.9
    cfg.dataset.mask_percentage = 0.0
    rng = np.random.default_rng(0)
    n = SIDE * SIDE
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(n, 2))
    CombinedModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(0)
    return CombinedModel(adata, cfg=cfg).module.eval()


def _global_rows(module, batch):
    with torch.no_grad():
        tokens = module.local_module.forward(batch.x, batch.edge_index)
        out, padding, index, _, _ = module.global_module.step(batch, tokens)
    rows = gather_tokens(out, padding)
    cells = torch.empty_like(rows)
    cells[torch.as_tensor(index[0], dtype=torch.long)] = rows
    return cells


@pytest.mark.parametrize("chunk_pairs", [2**24, 37 * 3], ids=["one block", "blocks"])
def test_behind_a_gcn_under_the_long_range_mask_the_fused_read_out_is_the_dense_one_and_blind(chunk_pairs):
    batch = _graph()
    dense = _stir(_combined("dense"))
    fused = _combined("fused", chunk_pairs=chunk_pairs)
    fused.load_state_dict(dense.state_dict())
    want = _global_rows(dense, batch)
    got = _global_rows(fused, batch)
    assert torch.allclose(got, want, atol=1e-5)
    after = _global_rows(fused, _changed(batch, CELL))
    assert torch.equal(after[CELL], got[CELL])  # the GCN tokens that hold the cell are blocked keys
    assert (after - got).abs().max() > 0


# --------------------------------------------------------------------------- the switch


def test_the_validator_takes_the_fused_attention_and_refuses_the_self_attention_only_backends():
    from interscale.config import _validate_context_only

    cfg = _cfg("fused")
    cfg.dataset.prediction_level = "node"
    _validate_context_only(cfg)
    for kind in ("performer", "entmax", "topk", "linear", "exphormer"):
        backend = _cfg(kind)
        backend.dataset.prediction_level = "node"
        with pytest.raises(ValueError, match="and the fused attention take"):
            _validate_context_only(backend)
