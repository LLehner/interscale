"""The `_common_step` contract: what every module type must return, and what it must mean.

`StepOutput` exists so that an auxiliary loss can read a named field instead of a tuple position.
That is only worth anything if the fields agree with each other, and the one that is easy to get
wrong is `padded_node_idx`: it is the only bridge between the transformer's padded token order and
the batch's own node order. If it is off, a loss will happily pair a cell's embedding with another
cell's neighbours and still train.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.module.base import StepOutput, ViewOutput
from interscale.module.global_modules import TransformerNodeEncoderHook

N_GENES = 8
N_EMBED = 8


def build_module(max_seq_len=32, mask_percentage=0.3, mask_strategy="node"):
    return TransformerNodeEncoderHook(
        max_seq_len=max_seq_len,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=16,
        long_range_attention=False,
        local_mask_hops=1,
        n_input=N_GENES,
        n_output=N_GENES,
        n_embed=N_EMBED,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=mask_percentage,
        mask_strategy=mask_strategy,
        type_gex_embedding=None,
    )


def make_batch(sizes=(6, 9), seed=0):
    """A batch of graphs carrying precomputed embeddings, so no PCA fit is involved."""
    rng = np.random.default_rng(seed)
    datas = []
    for n in sizes:
        x = torch.tensor(rng.normal(size=(n, N_GENES)), dtype=torch.float32)
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        d = Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long))
        d.embeddings = torch.tensor(rng.normal(size=(n, N_EMBED)), dtype=torch.float32)
        d.mask = torch.tensor(rng.random(n) < 0.4)
        d.mask[0] = True  # never an all-unmasked graph
        datas.append(d)
    return Batch.from_data_list(datas)


# --------------------------------------------------------------------------- shape of the contract


def test_step_output_exposes_a_single_view_by_default():
    out = build_module()._common_step(make_batch(), "regression", "node")

    assert isinstance(out, StepOutput)
    assert len(out.views) == 1
    assert out.view is out.views[0]


def test_shorthand_properties_read_the_reconstruction_view():
    v = ViewOutput(local_embedding=torch.zeros(2, 3), global_embedding=torch.zeros(4, 1, 3))
    out = StepOutput(y_pred=torch.zeros(2, 3), y_true=torch.zeros(2, 3), views=[v])

    assert out.local_embedding is v.local_embedding
    assert out.global_embedding is v.global_embedding
    assert out.attn is v.attn


def test_global_module_fills_the_fields_a_contrastive_loss_needs():
    batch = make_batch()
    out = build_module()._common_step(batch, "regression", "node")
    v = out.view

    assert v.global_embedding is not None
    assert v.src_padding_mask is not None
    assert v.padded_node_idx is not None
    assert v.attn is not None
    # No local component in a GlobalModule.
    assert v.local_embedding is None


# --------------------------------------------------------------------------- what the bridge means


def test_padded_node_idx_maps_tokens_back_to_batch_node_order():
    """`y_true` is gathered by this index, so it must reproduce `batch.x` row for row.

    This is the property every auxiliary loss leans on: index anything in batch node order with
    `padded_node_idx` and it lands in token order.
    """
    batch = make_batch()
    out = build_module()._common_step(batch, "regression", "node")

    assert torch.equal(out.y_true, batch.x[out.view.padded_node_idx])


def test_padded_node_idx_is_a_valid_index_into_the_batch():
    batch = make_batch(sizes=(6, 9, 4))
    out = build_module()._common_step(batch, "regression", "node")
    idx = out.view.padded_node_idx

    assert idx.dtype == torch.long
    assert idx.min() >= 0
    assert idx.max() < batch.num_nodes
    assert len(idx.unique()) == len(idx), "a cell must not be emitted as two tokens"


def test_token_count_matches_the_unpadded_positions():
    """The number of kept tokens is what `src_padding_mask` says it is, CLS excluded.

    `pad_batch` LEFT-pads and the CLS token sits at the LAST sequence position; a helper that
    drops the wrong end silently contrasts against CLS or against padding.
    """
    batch = make_batch(sizes=(6, 9, 4))
    out = build_module()._common_step(batch, "regression", "node")
    v = out.view

    n_real = int((~v.src_padding_mask[:, :-1]).sum())
    assert len(v.padded_node_idx) == n_real
    assert len(out.y_pred) == n_real


def test_entry_mask_is_carried_for_gene_masking():
    batch = make_batch()
    n, g = batch.x.shape
    gen = torch.Generator().manual_seed(0)
    batch.gene_mask = torch.rand(n, g, generator=gen) < 0.4
    batch.mask = batch.gene_mask.any(dim=1)

    out = build_module(mask_strategy="gene")._common_step(batch, "regression", "node")

    assert out.entry_mask is not None
    assert out.entry_mask.shape == out.y_true.shape
    assert torch.equal(out.entry_mask, batch.gene_mask[out.view.padded_node_idx])
