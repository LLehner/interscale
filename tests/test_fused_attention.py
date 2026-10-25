"""The fused attention (``attention.kind: fused``; attention_plan.md, stage 6).

Pinned here:

* **the biases by rows** -- ``AttentionBias.rows`` gives rows of the dense bias for every kind with a pair form
  (CLS and padding 0), on two graphs of different size, one subsampled, for blocks of any size, with its gradient;
* **the definition** -- the fused attention is dense attention: outputs, maps and gradients, with the blocked
  diagonal, the long-range mask, padding, every bias and RoPE, in one block and in many;
* **through the real model** -- a GlobalModel with the windowed settings (distance window, RoPE, centred queries
  and keys, the raw own-cell skip, a hidden gene set), per cell, subsampled and pooled, computes what the dense
  one does at the same weights, gradients included;
* **memory** -- with more than one block no op of a pass, backward included, forms a cells x cells tensor;
* **dropout** -- the blocks recomputed in the backward pass drop what the forward pass dropped;
* **the switch** -- a fused model starts from the dense model's weights and RNG; validator, config, checkpoint name.

Every parameter is stirred off its initialisation (biases, tables and projections start at zero or identity), so no
comparison is between two things that are zero by construction.
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch.utils._python_dispatch import TorchDispatchMode
from torch_geometric.data import Batch, Data

from interscale.module.global_modules import TransformerNodeEncoderHook
from interscale.module.global_modules.attention import DenseAttention, FusedAttention, StructuredMask

N_HEADS = 2
N_EMBED = 8
G = 6
CPU = torch.device("cpu")


def _stir(module, seed=0, scale=0.1):
    """Every parameter moved off its initialisation by seeded noise (tests-stir-the-init)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


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


def _path_graph(n):
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _two_graphs(sizes=(9, 6), k=4, seed=0):
    """Two path graphs with coordinates (µm) and the spectral inputs attach_positional_inputs leaves."""
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


def _padded_layout(batch, max_seq_len):
    from interscale.tl.padding import pad_batch

    _, padding, index_nodes, *_ = pad_batch(torch.zeros(batch.num_nodes, 1), batch.batch, max_seq_len)
    return padding, index_nodes


class _PairsOnly(torch.nn.Module):
    """The distance profile with its pair form alone, read from coordinates: the generic route of ``rows``."""

    reads_coordinates = True

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, pos_um):
        return self.inner(torch.cdist(pos_um, pos_um, compute_mode="donot_use_mm_for_euclid_dist"))

    def pairs(self, pos_um, graph, query, key):
        return self.inner.pairs(pos_um, graph, query, key)


def _encoder(name):
    """``(registry name, encoder)``, its parameters stirred."""
    from interscale.module.global_modules.positional_encodings import (
        DistanceBias,
        FourierBias,
        LinearDistanceBias,
        ScaledDistanceBias,
        SpectralBias,
    )

    encoders = {
        "profile": lambda: ("distance", DistanceBias(N_HEADS, 8, 1500.0)),
        "linear": lambda: ("distance", LinearDistanceBias(N_HEADS)),
        "gaussian": lambda: ("distance", ScaledDistanceBias(N_HEADS, 100.0, 800.0, "gaussian", 1.0, False)),
        "exponential": lambda: ("distance", ScaledDistanceBias(N_HEADS, 100.0, 800.0, "exponential", 1.0, False)),
        "spectral": lambda: ("spectral", SpectralBias(N_HEADS, 8, 1e-3, 2.0, k=4)),
        "fourier": lambda: ("fourier", FourierBias(N_HEADS, 4, 6, 200.0, 4000.0, isotropic=False)),
        "pairs_only": lambda: ("distance", _PairsOnly(DistanceBias(N_HEADS, 8, 1500.0))),
    }
    registry_name, encoder = encoders[name]()
    return registry_name, _stir(encoder, seed=len(name), scale=0.5)


def _bias(*names):
    from interscale.module.global_modules.positional_encodings import AttentionBias

    encoders = dict(_encoder(name) for name in names)
    return AttentionBias(encoders, n_heads=N_HEADS, spatial_unit_um=1.0)


# --------------------------------------------------------------------------- the biases by rows


@pytest.mark.parametrize("name", ["profile", "linear", "gaussian", "exponential", "spectral", "fourier", "pairs_only"])
def test_the_rows_of_every_bias_are_the_dense_bias(name):
    """Any block of rows is that block of the dense bias -- CLS and padding 0 -- on two graphs of different size,
    the first subsampled by pad_batch, the second padded."""
    bias = _bias(name)
    batch = _two_graphs()
    padding, index_nodes = _padded_layout(batch, max_seq_len=7)
    assert padding.shape == (2, 7) and padding[1].any() and not padding[0].any()
    dense = bias(batch, index_nodes, padding, dtype=torch.float32, device=CPU).view(2, N_HEADS, 8, 8)
    read = bias.rows(batch, index_nodes, padding, dtype=torch.float32, device=CPU)
    for start, stop in [(0, 8), (0, 3), (3, 6), (6, 8), (7, 8), (2, 3)]:
        block = read(start, stop)
        assert block.shape == (2, N_HEADS, stop - start, 8)
        assert torch.allclose(block, dense[:, :, start:stop], atol=1e-5), (start, stop)
    assert dense[:, :, :7, :7].abs().sum() > 0  # the comparison is not between two zero tensors


def test_the_rows_keep_exact_distances_far_from_the_origin():
    """Coordinates as a section's global ones (here ~50 mm from the origin) and a window of 20 µm: the rows' distances
    are differences then norms, as the dense form's, never the matrix-product shortcut, which loses µm there."""
    from interscale.module.global_modules.positional_encodings import AttentionBias, ScaledDistanceBias

    encoder = ScaledDistanceBias(N_HEADS, 20.0, 40.0, "gaussian", 1.0, False)
    bias = AttentionBias({"distance": encoder}, n_heads=N_HEADS, spatial_unit_um=1.0)
    batch = _two_graphs()
    batch.pos = batch.pos * 0.05 + 5e4  # cells ~10 µm apart, 50 mm from the origin
    padding, index_nodes = _padded_layout(batch, max_seq_len=7)
    dense = bias(batch, index_nodes, padding, dtype=torch.float32, device=CPU).view(2, N_HEADS, 8, 8)
    read = bias.rows(batch, index_nodes, padding, dtype=torch.float32, device=CPU)
    assert torch.allclose(torch.cat([read(0, 4), read(4, 8)], dim=2), dense, atol=1e-4)
    assert dense[:, :, :6, :6].abs().max() > 0.1


def test_the_rows_carry_the_gradient_of_the_dense_bias():
    bias = _bias("gaussian", "spectral", "fourier")
    batch = _two_graphs()
    padding, index_nodes = _padded_layout(batch, max_seq_len=7)
    weight = torch.randn(2, N_HEADS, 8, 8, generator=torch.Generator().manual_seed(1))

    def grads(loss):
        bias.zero_grad(set_to_none=True)
        loss.backward()
        return {n: p.grad.clone() for n, p in bias.named_parameters() if p.grad is not None}

    dense = bias(batch, index_nodes, padding, dtype=torch.float32, device=CPU).view(2, N_HEADS, 8, 8)
    expected = grads((dense * weight).sum())
    read = bias.rows(batch, index_nodes, padding, dtype=torch.float32, device=CPU)
    blocks = torch.cat([read(0, 3), read(3, 8)], dim=2)
    got = grads((blocks * weight).sum())
    assert set(got) == set(expected) and len(got) == 5  # log_scale, strength, two tables of coefficients, table
    for name, value in expected.items():
        assert value.abs().sum() > 0, name
        assert torch.allclose(got[name], value, atol=1e-5, rtol=1e-4), name


# --------------------------------------------------------------------------- the definition


def _attention_pair(seed=0, chunk_pairs=2**24, dropout=0.0):
    """A dense skeleton and a fused layer with the same, stirred, projections."""
    torch.manual_seed(seed)
    dense = DenseAttention(N_EMBED, N_HEADS, N_EMBED // N_HEADS, dropout=dropout)
    torch.manual_seed(seed)
    fused = FusedAttention(N_EMBED, N_HEADS, N_EMBED // N_HEADS, dropout=dropout, chunk_pairs=chunk_pairs)
    _stir(dense)
    fused.load_state_dict(dense.state_dict())
    return dense, fused


def _layout(hops, max_seq_len=7):
    """Two graphs (9 cells subsampled to 7, and 6) with their blocked pairs, padding (CLS open) and dense mask."""
    from interscale.tl.masking import blocked_pairs

    batch = _two_graphs()
    padding, index_nodes = _padded_layout(batch, max_seq_len=max_seq_len)
    seq_len = padding.shape[1]
    blocked = blocked_pairs(batch.edge_index, batch.batch, index_nodes, n_hops=hops)
    dense = torch.zeros(2, seq_len + 1, seq_len + 1, dtype=torch.bool)
    dense[blocked[0], blocked[1], blocked[2]] = True
    padding_cls = torch.cat([padding, torch.zeros(2, 1, dtype=torch.bool)], dim=1)
    return batch, index_nodes, padding, padding_cls, blocked, dense.repeat_interleave(N_HEADS, dim=0)


def _rotary(seq_len, seed=0):
    gen = torch.Generator().manual_seed(seed)
    angles = torch.rand(2 * N_HEADS, seq_len, N_EMBED // N_HEADS // 2, generator=gen) * 6
    return angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)


@pytest.mark.parametrize("hops", [0, 1, 2])
@pytest.mark.parametrize("chunk_pairs", [2**24, 48, 1])  # one block; blocks of 3 rows (2 graphs x 8 tokens); 1 row
def test_the_fused_attention_is_dense_attention(hops, chunk_pairs):
    """Outputs, maps and gradients -- of the tokens, the projections and every bias parameter -- equal dense
    attention's: with the blocked diagonal (hops 0) or the long-range mask, padding, a subsampled graph, the
    biases (window, spectral, Fourier) and RoPE, in one block and in many (recomputed in the backward pass)."""
    batch, index_nodes, padding, padding_cls, blocked, dense_mask = _layout(hops)
    seq_len = padding_cls.shape[1]
    rotary = _rotary(seq_len)
    bias = _bias("gaussian", "spectral", "fourier")
    dense, fused = _attention_pair(chunk_pairs=chunk_pairs)
    g = torch.randn(seq_len, 2, N_EMBED, generator=torch.Generator().manual_seed(5))
    x0 = torch.randn(seq_len, 2, N_EMBED, generator=torch.Generator().manual_seed(6))

    for with_bias in (False, True):

        def run(attention, need_weights, with_bias):
            attention.zero_grad(set_to_none=True)
            bias.zero_grad(set_to_none=True)
            x = x0.clone().requires_grad_(True)
            if attention is dense:
                mask = dense_mask
                if with_bias:
                    mask = bias(batch, index_nodes, padding, torch.float32, CPU).masked_fill(dense_mask, float("-inf"))
                out, weights = dense(x, x, x, key_padding_mask=padding_cls, attn_mask=mask, rotary=rotary)
            else:
                rows = bias.rows(batch, index_nodes, padding, torch.float32, CPU) if with_bias else None
                context = StructuredMask(blocked, bias_rows=rows)
                out, weights = fused(
                    x, x, x, key_padding_mask=padding_cls, need_weights=need_weights, rotary=rotary, context=context
                )
            (out * g).sum().backward()
            grads = {"x": x.grad} | {n: p.grad for n, p in attention.named_parameters()}
            grads |= {n: p.grad for n, p in bias.named_parameters() if p.grad is not None}
            return out.detach(), weights, grads

        out_dense, w_dense, g_dense = run(dense, True, with_bias)
        for need_weights in (False, True):
            out_fused, w_fused, g_fused = run(fused, need_weights, with_bias)
            assert torch.allclose(out_fused, out_dense, atol=1e-5)
            if need_weights:
                assert torch.allclose(w_fused, w_dense, atol=1e-5)
            else:
                assert w_fused is None
            assert set(g_fused) == set(g_dense) and len(g_fused) == (10 if with_bias else 5)
            for name, value in g_dense.items():
                assert torch.allclose(g_fused[name], value, atol=1e-5, rtol=1e-4), (name, need_weights, with_bias)

    # The mask is the one the dense model uses: a cell never reads its own key (hops 0), CLS always open.
    if hops == 0:
        cells = ~padding_cls[:, :-1]
        diagonal = w_fused.view(2, N_HEADS, seq_len, seq_len).diagonal(dim1=-2, dim2=-1)[..., :-1]
        assert (diagonal.permute(1, 0, 2)[:, cells] == 0).all()


def test_each_blocks_mask_is_built_when_it_runs_and_again_for_the_backward_pass():
    """A block's biases are computed when the block runs; with more than one block and a gradient to take they are
    computed again in the backward pass rather than kept (activation checkpointing). With one block, or without a
    gradient, once."""
    batch, index_nodes, padding, padding_cls, blocked, _ = _layout(hops=0)
    seq_len = padding_cls.shape[1]
    bias = _bias("gaussian")
    x0 = torch.randn(seq_len, 2, N_EMBED, generator=torch.Generator().manual_seed(8))

    def calls(chunk_pairs, grad):
        _, fused = _attention_pair(chunk_pairs=chunk_pairs)
        read = bias.rows(batch, index_nodes, padding, torch.float32, CPU)
        count = []

        def counted(start, stop):
            count.append((start, stop))
            return read(start, stop)

        x = x0.clone().requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            out, _ = fused(
                x,
                x,
                x,
                key_padding_mask=padding_cls,
                need_weights=False,
                context=StructuredMask(blocked, bias_rows=counted),
            )
        if grad:
            out.sum().backward()
        return count, fused.block_rows(2, seq_len)

    count, rows = calls(48, grad=True)
    blocks = [(start, min(start + rows, seq_len)) for start in range(0, seq_len, rows)]
    assert rows == 3 and len(blocks) == 3
    assert sorted(count) == sorted(blocks * 2)  # each block twice: the forward pass and its recomputation
    assert sorted(calls(48, grad=False)[0]) == blocks
    assert calls(2**24, grad=True)[0] == [(0, seq_len)]


def test_a_block_holds_chunk_pairs_query_key_pairs_and_at_least_one_query():
    fused = FusedAttention(N_EMBED, N_HEADS, N_EMBED // N_HEADS, chunk_pairs=100)
    assert fused.block_rows(n_graphs=2, seq_len=8) == 6
    assert fused.block_rows(n_graphs=2, seq_len=200) == 1
    assert FusedAttention(N_EMBED, N_HEADS, 4).block_rows(1, 2501) == 2**24 // 2501  # synth_spot: one block
    with pytest.raises(ValueError, match="chunk_pairs must be >= 1"):
        FusedAttention(N_EMBED, N_HEADS, 4, chunk_pairs=0)


def test_the_fused_attention_refuses_a_dense_mask():
    _, fused = _attention_pair()
    x = torch.randn(5, 1, N_EMBED)
    with pytest.raises(ValueError, match="takes its mask as a StructuredMask"):
        fused(x, x, x, attn_mask=torch.zeros(N_HEADS, 5, 5, dtype=torch.bool))


def _module(attention=None, long_range=False, hops=0, attention_bias=None):
    return TransformerNodeEncoderHook(
        max_seq_len=32,
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
        attention=attention,
        attention_bias=attention_bias,
    )


class _Att(dict):
    """A dict that reads like the yacs attention block (attribute and .get access)."""

    __getattr__ = dict.__getitem__


def _fused_cfg(chunk_pairs=2**24, center_qk=False):
    return _Att(kind="fused", head_dim=0, center_qk=center_qk, fused=_Att(chunk_pairs=chunk_pairs))


@pytest.mark.parametrize("chunk_pairs", [2**24, 40])
def test_a_transformer_under_the_long_range_mask_is_the_dense_one(chunk_pairs):
    """Two layers, the long-range mask over 2 hops, a distance window: the fused transformer's outputs, maps and
    gradients are the dense transformer's at the same (stirred) weights."""
    n = 11
    batch = _Batch(
        _path_graph(n),
        torch.zeros(n, dtype=torch.long),
        n,
        pos=torch.arange(n).float()[:, None] * 90 * torch.ones(1, 2),
    )
    emb = torch.randn(n, N_EMBED, generator=torch.Generator().manual_seed(2))

    def build(attention):
        torch.manual_seed(0)
        module = _module(attention, long_range=True, hops=2, attention_bias=_bias("gaussian"))
        return module

    dense, fused = build(None), build(_fused_cfg(chunk_pairs))
    _stir(dense)
    fused.load_state_dict(dense.state_dict())

    def run(module, maps):
        module.zero_grad(set_to_none=True)
        padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
        out, _, attn = module.forward(padded, padding, mask, register_hook=maps, rotary=rotary)
        out.pow(2).sum().backward()
        return out.detach(), attn, {k: p.grad for k, p in module.named_parameters() if p.grad is not None}

    out_dense, attn_dense, g_dense = run(dense, True)
    for maps in (False, True):
        out_fused, attn_fused, g_fused = run(fused, maps)
        assert torch.allclose(out_fused, out_dense, atol=1e-5)
        assert (attn_fused is None) != maps
        if maps:
            assert torch.allclose(attn_fused, attn_dense, atol=1e-5)
        assert set(g_fused) == set(g_dense)
        for name, value in g_dense.items():
            assert torch.allclose(g_fused[name], value, atol=1e-5, rtol=1e-4), name


# --------------------------------------------------------------------------- through the real model


def _lattice(side, seed, spacing=100.0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * spacing
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= spacing * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.ones(n, dtype=torch.bool)
    data.gene_mask = torch.zeros(n, G, dtype=torch.bool)
    data.gene_mask[:, [0, 1]] = True  # programme masking: a gene set hidden in every cell
    return data


def _lattices(sides=(6, 5)):
    batch = Batch.from_data_list([_lattice(side, seed=side) for side in sides])
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _window_config(kind, *, chunk_pairs=2**24, max_seq_len=64, pool=None):
    """The windowed global-only model of the window sweep, at toy width: GlobalModel, learned linear token,
    distance window, RoPE, centred queries and keys, the raw own-cell skip, a gene set hidden in every cell."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = N_EMBED
    cfg.model.decoder.dropout_decoder = 0.0
    cfg.model.decoder.own_cell_skip = True
    cfg.model.decoder.own_cell_skip_input = "raw"
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = max_seq_len, N_HEADS, 2, 0.0
    params.type_gex_embedding = "Linear"
    params.attention.kind = kind
    params.attention.center_qk = True
    params.attention.fused.chunk_pairs = chunk_pairs
    params.pe.bias = ["distance"]
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    params.pe.rotary = ["rope"]
    params.pe.rope.min_wavelength = 100.0
    if pool is not None:
        params.pooling.kind = "grid"
        params.pooling.size_um = pool
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    cfg.dataset.mask_genes = ["g0", "g1"]
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    return cfg


def _build(cfg, seed=0):
    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return GlobalModel(adata, cfg=cfg).module


def _run(module, batch, weight=None):
    """One masked pass of the model's own step, then the backward of a fixed weighting of its prediction."""
    random.seed(0)  # pad_batch subsamples with Python's random
    module.zero_grad(set_to_none=True)
    out = module._common_step(batch, "regression", "node")
    weight = torch.randn(out.y_pred.shape, generator=torch.Generator().manual_seed(3)) if weight is None else weight
    (out.y_pred * weight).sum().backward()
    return out.y_pred.detach(), {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


@pytest.mark.parametrize(
    ("chunk_pairs", "max_seq_len", "pool"),
    [(2**24, 64, None), (2 * 37 * 4, 64, None), (60, 30, None), (2 * 10 * 3, 64, 200.0)],
    ids=["one block", "blocks of 4 queries", "one query per block, subsampled", "pooled patches, blocks"],
)
def test_the_windowed_global_model_is_the_dense_one(chunk_pairs, max_seq_len, pool):
    """The model the window sweep reads, built twice at one seed -- dense and fused -- holds the same weights and,
    after stirring them, predicts the same and gets the same gradient in every parameter."""
    batch = _lattices()
    dense = _build(_window_config("dense", max_seq_len=max_seq_len, pool=pool))
    fused = _build(_window_config("fused", chunk_pairs=chunk_pairs, max_seq_len=max_seq_len, pool=pool))
    assert set(dense.state_dict()) == set(fused.state_dict())
    assert all(torch.equal(v, fused.state_dict()[k]) for k, v in dense.state_dict().items())
    _stir(dense)
    fused.load_state_dict(dense.state_dict())
    dense.eval(), fused.eval()

    pred_dense, grads_dense = _run(dense, batch)
    pred_fused, grads_fused = _run(fused, batch)
    assert torch.allclose(pred_fused, pred_dense, atol=1e-5)
    assert set(grads_fused) == set(grads_dense)
    window = [n for n in grads_dense if "log_scale" in n or "strength" in n]
    assert window and all(grads_dense[n].abs().sum() > 0 for n in window)
    for name, value in grads_dense.items():
        assert torch.allclose(grads_fused[name], value, atol=1e-5, rtol=1e-4), name


# --------------------------------------------------------------------------- memory


class _Shapes(TorchDispatchMode):
    """Every tensor any op returns, forward and backward, by shape."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                self.shapes.append(tuple(t.shape))
        return out


def _cells_by_cells(shapes, n_cells):
    """The shapes with two axes over the sequence (``n_cells`` cells, or with CLS): a matrix over the cells' pairs."""
    return [s for s in shapes if sum(d in (n_cells, n_cells + 1) for d in s) >= 2]


def test_a_pass_in_blocks_forms_no_matrix_over_the_cells_pairs():
    """With more than one block, nothing in the windowed model's step -- masks, biases, attention, and the backward
    pass through the recomputed blocks -- is a cells x cells tensor; the dense model's is, the check's control."""
    batch = _lattices()
    n_cells = 36  # the larger graph; 37 tokens with CLS
    for kind, chunk_pairs in (("fused", 2 * 37 * 4), ("dense", 2**24)):
        module = _stir(_build(_window_config(kind, chunk_pairs=chunk_pairs))).eval()
        with _Shapes() as record:
            _run(module, batch)
        found = _cells_by_cells(record.shapes, n_cells)
        assert (not found) if kind == "fused" else found, (kind, found[:5])


# --------------------------------------------------------------------------- dropout


def test_the_recomputed_blocks_drop_what_the_forward_pass_dropped(monkeypatch):
    """In training, with dropout and several blocks, the gradient through the recomputed blocks is the gradient of
    the forward pass that ran: the same as without recomputation, from the same random state."""
    import interscale.module.global_modules.attention as attention_module

    _, fused = _attention_pair(chunk_pairs=48, dropout=0.5)
    fused.train()
    batch, index_nodes, padding, padding_cls, blocked, _ = _layout(hops=0)
    seq_len = padding_cls.shape[1]
    bias = _bias("gaussian")
    x0 = torch.randn(seq_len, 2, N_EMBED, generator=torch.Generator().manual_seed(7))

    def run():
        torch.manual_seed(11)
        fused.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        context = StructuredMask(blocked, bias_rows=bias.rows(batch, index_nodes, padding, torch.float32, CPU))
        out, _ = fused(x, x, x, key_padding_mask=padding_cls, need_weights=False, context=context)
        out.pow(2).sum().backward()
        return out.detach(), x.grad, fused.in_proj_weight.grad.clone()

    recomputed = run()
    monkeypatch.setattr(attention_module, "checkpoint", lambda fn, *args, **kwargs: fn(*args))
    kept = run()
    for a, b in zip(recomputed, kept, strict=True):
        assert torch.allclose(a, b, atol=1e-6)

    context = StructuredMask(blocked, bias_rows=bias.rows(batch, index_nodes, padding, torch.float32, CPU))
    with torch.no_grad():
        _, maps = fused(x0, x0, x0, key_padding_mask=padding_cls, need_weights=True, context=context)
    assert (maps.sum(-1) - 1).abs().max() > 0.1  # the explicit path drops too, in training
    fused.eval()
    with torch.no_grad():
        plain, _ = fused(x0, x0, x0, key_padding_mask=padding_cls, need_weights=False, context=context)
        again, _ = fused(x0, x0, x0, key_padding_mask=padding_cls, need_weights=False, context=context)
    # Dropout did act in training: by far more than the rounding between the kernels taken with and without a gradient.
    assert (plain - recomputed[0]).abs().max() > 1e-2
    assert torch.equal(plain, again)  # and none in evaluation

    dense, _ = _attention_pair(dropout=0.5)
    dense.load_state_dict(fused.state_dict())
    dense.eval()
    mask = bias(batch, index_nodes, padding, torch.float32, CPU).masked_fill(_layout(hops=0)[5], float("-inf"))
    with torch.no_grad():
        reference, _ = dense(x0, x0, x0, key_padding_mask=padding_cls, attn_mask=mask)
    assert torch.allclose(plain, reference, atol=1e-5)


# --------------------------------------------------------------------------- the switch


def _cfg(kind="dense", *, bias=(), rotary=(), loss="MSELoss"):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = 16
    params = cfg.model.global_component.parameters
    params.n_heads = 4
    params.attention.kind = kind
    params.pe.bias = list(bias)
    params.pe.rotary = list(rotary)
    params.pe.distance.max_dist = 1000.0
    params.pe.rope.max_wavelength = 1000.0
    cfg.dataset.spatial_key = "spatial"
    cfg.optim.loss = loss
    return cfg


def test_a_fused_model_starts_from_the_dense_models_weights_and_rng():
    from interscale.module.base import GlobalModule

    def build(kind):
        torch.manual_seed(0)
        module = GlobalModule.from_config(
            _cfg(kind),
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
        return module, torch.get_rng_state()

    dense, rng_dense = build("dense")
    fused, rng_fused = build("fused")
    assert torch.equal(rng_dense, rng_fused)
    assert set(dense.state_dict()) == set(fused.state_dict())
    for key, value in dense.state_dict().items():
        assert torch.equal(value, fused.state_dict()[key]), key
    assert isinstance(fused.transformer_encoder.layers[0].self_attn, FusedAttention)


def test_the_validator_takes_every_bias_with_a_pair_form_and_rope_for_the_fused_attention():
    from interscale.config import _validate_attention

    _validate_attention(_cfg(kind="fused", bias=["distance", "spectral", "fourier"], rotary=["rope"]))
    for shape in ("gaussian", "exponential"):
        cfg = _cfg(kind="fused", bias=["distance"])
        cfg.model.global_component.parameters.pe.distance.kind = "scales"
        cfg.model.global_component.parameters.pe.distance.shape = shape
        _validate_attention(cfg)
    with pytest.raises(ValueError, match="cannot carry the attention bias 'expression'.*pair form"):
        _validate_attention(_cfg(kind="fused", bias=["expression"]))
    with pytest.raises(ValueError, match="reads the attention maps in training"):
        _validate_attention(_cfg(kind="fused", loss="SCE_EntropyATT_Loss"))
    cfg = _cfg(kind="fused")
    cfg.model.global_component.parameters.attention.visible_keys_only = True
    with pytest.raises(ValueError, match="visible_keys_only edits the dense attention mask"):
        _validate_attention(cfg)
    cfg = _cfg(kind="fused")
    cfg.model.global_component.parameters.attention.fused.chunk_pairs = 0
    with pytest.raises(ValueError, match="chunk_pairs .* must be >= 1"):
        _validate_attention(cfg)


def test_the_config_builds_it_with_its_chunk_and_the_checkpoint_name_carries_it():
    from interscale.module.base import GlobalModule
    from interscale.tl.utils import _attention_tag

    cfg = _cfg(kind="fused")
    assert cfg.model.global_component.parameters.attention.fused.chunk_pairs == 2**24
    cfg.model.global_component.parameters.attention.fused.chunk_pairs = 1234
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
    assert module.transformer_encoder.layers[1].self_attn.chunk_pairs == 1234
    assert _attention_tag(_cfg(kind="fused")) == "attn-fused_"
    cfg = _cfg(kind="fused")
    cfg.model.global_component.parameters.attention.center_qk = True
    assert _attention_tag(cfg) == "attn-fused-cqk_"


def test_a_fused_model_trains_feeds_relevance_and_runs_a_full_step():
    n = 10
    batch = _Batch(_path_graph(n), torch.zeros(n, dtype=torch.long), n, pos=torch.rand(n, 2) * 500)
    emb = torch.randn(n, N_EMBED)
    target = torch.randn(n + 1, 1, N_EMBED)
    module = _module(_fused_cfg(chunk_pairs=33), attention_bias=_bias("gaussian"))
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        padded, padding, _, mask, rotary = module.common_step_local_to_global(batch, emb, eval_step=True)
        loss = torch.nn.functional.mse_loss(module.forward(padded, padding, mask, rotary=rotary)[0], target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]

    module.eval()
    _, _, _, _, relevance = module.evaluate(batch, emb)
    assert relevance.shape == (n + 1, n + 1) and torch.isfinite(relevance).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_on_the_gpu_the_fused_kernel_is_the_dense_model():
    """The GPU's fused kernel (memory-efficient attention, its additive mask's gradient included) against dense
    attention, through the windowed model, in blocks -- with SDPA held to that kernel, which raises rather than
    falling back to the math kernel if it refuses the head width or the mask."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    batch = _lattices().to("cuda")
    dense = _build(_window_config("dense"))
    fused = _build(_window_config("fused", chunk_pairs=2 * 37 * 4))
    _stir(dense)
    fused.load_state_dict(dense.state_dict())
    dense, fused = dense.cuda().eval(), fused.cuda().eval()
    weight = torch.randn(batch.num_nodes, G, generator=torch.Generator().manual_seed(3)).cuda()
    pred_dense, grads_dense = _run(dense, batch, weight)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        pred_fused, grads_fused = _run(fused, batch, weight)
    assert torch.allclose(pred_fused, pred_dense, atol=1e-4)
    for name, value in grads_dense.items():
        assert torch.allclose(grads_fused[name], value, atol=1e-4, rtol=1e-3), name
