"""``attention.visible_keys_only`` beyond the dense mask: the fused attention and the hierarchical attention
(``.claude/attention_plan.md``, stage 4 / K5; roadmap Tier 1, item 2).

Pinned here:

* **fused** -- the hidden keys ride on the ``StructuredMask`` (``hidden_keys``) and every block blocks them: the fused
  model with the option is the dense model with it, in one block and in many, predictions and gradients;
* **hierarchy** -- every level blocks the tokens that hold no visible cell: the singleton hierarchy (lambda 0) with the
  option is the dense model with it;
* **what it removes** -- with the option, nothing of a hidden cell's token reaches any prediction (without it, it does),
  for the dense, fused and hierarchical attention;
* **the switch** -- a clean pass blocks nothing; the validator takes fused and the hierarchy, still refuses the backends
  that cannot block a key.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.module.global_modules.attention import DenseAttention, FusedAttention, StructuredMask

E, G, H = 8, 6, 2
SPACING = 100.0
HIDDEN = list(range(0, 12)) + [20, 21]  # two rows of the 6 x 6 lattice and two more cells, hidden whole


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _lattice(side, seed, hidden=()):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    data.mask[list(hidden)] = True
    return data


def _batch():
    graphs = [_lattice(6, seed=6, hidden=HIDDEN), _lattice(5, seed=5, hidden=[0, 1, 2, 7])]
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _config(kind="dense", *, visible=True, hierarchy=None, chunk_pairs=2**24, max_seq_len=64, center_qk=True):
    """The windowed global-only model at toy width under cell masking (what CosMx trains with)."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = max_seq_len, H, 2, 0.0
    params.type_gex_embedding = "Linear"
    params.attention.kind = kind
    params.attention.center_qk = center_qk
    params.attention.visible_keys_only = visible
    params.attention.fused.chunk_pairs = chunk_pairs
    params.pe.bias = ["distance"]
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    if hierarchy is not None:
        params.hierarchy.kind = "coarse_to_fine"
        for key, value in hierarchy.items():
            node = params.hierarchy
            *path, last = key.split(".")
            for part in path:
                node = node[part]
            node[last] = value
    cfg.dataset.mask_strategy = "node"
    cfg.dataset.mask_percentage = 0.3
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    return cfg


def _build(cfg, seed=0):
    from interscale.config import _validate
    from interscale.model.global_model import GlobalModel

    _validate(cfg)
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


def _run(module, batch):
    random.seed(0)
    module.zero_grad(set_to_none=True)
    out = module._common_step(batch, "regression", "node")
    weight = torch.randn(out.y_pred.shape, generator=torch.Generator().manual_seed(3))
    (out.y_pred * weight).sum().backward()
    return out.y_pred.detach(), {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}


def _same(reference, other, atol=1e-5):
    """Build both, stir ``reference``, copy it into ``other``, compare a pass and its gradients."""
    batch = _batch()
    reference = _stir(reference).eval()
    missing, unexpected = other.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected
    other.eval()
    pred_ref, grads_ref = _run(reference, batch)
    pred_other, grads_other = _run(other, batch)
    assert torch.allclose(pred_other, pred_ref, atol=atol)
    for name, value in grads_ref.items():
        assert torch.allclose(grads_other[name], value, atol=atol, rtol=1e-4), name
    return pred_ref


# --------------------------------------------------------------------------- the fused attention


def test_the_fused_layer_gives_a_hidden_key_no_weight_and_equals_dense_attention_with_it_blocked():
    gen = torch.Generator().manual_seed(1)
    n_graphs, seq_len, d = 2, 9, 4
    dense, fused = DenseAttention(E, H, d), FusedAttention(E, H, d, chunk_pairs=n_graphs * seq_len * 2)
    _stir(dense)
    fused.load_state_dict(dense.state_dict())
    x = torch.randn(seq_len, n_graphs, E, generator=gen)
    hidden = torch.zeros(n_graphs, seq_len, dtype=torch.bool)
    hidden[0, [1, 4]] = True
    hidden[1, [0, 2, 3]] = True
    diag = torch.arange(seq_len - 1)
    blocked = torch.stack([torch.arange(n_graphs).repeat_interleave(seq_len - 1), diag.repeat(n_graphs), diag.repeat(n_graphs)])
    context = StructuredMask(blocked, hidden_keys=hidden)
    dense_mask = torch.zeros(n_graphs, seq_len, seq_len, dtype=torch.bool)
    dense_mask[blocked[0], blocked[1], blocked[2]] = True
    dense_mask |= hidden[:, None, :]
    dense_mask = dense_mask.repeat_interleave(H, 0)
    want, maps = dense(x, x, x, attn_mask=dense_mask)
    got, _ = fused(x, x, x, need_weights=False, context=context)
    assert torch.allclose(got, want, atol=1e-6)
    _, fused_maps = fused(x, x, x, need_weights=True, context=context)
    keys = hidden.repeat_interleave(H, 0)[:, None, :].expand_as(fused_maps)
    assert (fused_maps[keys] == 0).all() and torch.allclose(fused_maps, maps, atol=1e-6)


@pytest.mark.parametrize(
    ("chunk_pairs", "max_seq_len"),
    [(2**24, 64), (2 * 37 * 4, 64), (60, 30)],
    ids=["one block", "blocks of 4 queries", "one query per block, subsampled"],
)
def test_the_fused_model_with_visible_keys_only_is_the_dense_one(chunk_pairs, max_seq_len):
    dense = _build(_config("dense", max_seq_len=max_seq_len))
    fused = _build(_config("fused", chunk_pairs=chunk_pairs, max_seq_len=max_seq_len))
    _same(dense, fused)


def test_a_clean_pass_blocks_no_key():
    module = _build(_config("fused"))
    batch = _batch()
    given = module.common_step_local_to_global(batch, module.gex_embedding(batch), eval_step=True)
    assert given.attention_mask.hidden_keys is None
    random.seed(0)
    given = module.common_step_local_to_global(batch, module.gex_embedding(batch))
    keys = given.attention_mask.hidden_keys
    assert keys.shape == (2, given.padded_emb.shape[0] + 1) and int(keys.sum()) == len(HIDDEN) + 4
    assert not keys[:, -1].any()  # CLS open


# --------------------------------------------------------------------------- the hierarchy

SOFT_ZERO = {"prior_mode": "soft", "lambda_init": 0.0, "lambda_learnable": "fixed"}


@pytest.mark.parametrize(
    "hierarchy",
    [
        {**SOFT_ZERO, "cells_per_cluster": [1.0]},
        {
            **SOFT_ZERO,
            "cells_per_cluster": [1.0],
            "candidates.mode": "routed",
            "candidates.topk": [100],
            "candidates.cap_per_query": 4096,
        },
        {"prior_mode": "none"},
    ],
    ids=["singletons, dense", "singletons, routed over every parent", "no coarse level"],
)
def test_the_singleton_hierarchy_with_visible_keys_only_is_the_dense_one(hierarchy):
    """Under cell masking the singleton hierarchy is the dense model only with the option on: off, its top level hands CLS
    the learned ``pool_fill`` of every hidden singleton where the dense model hands it the hidden cell's token (2-4e-2
    apart, measured 2026-10-31). ``center_qk`` stays off: the top level centres on a mean that holds ``pool_fill`` too
    (1.8e-4 apart with it)."""
    dense = _build(_config("dense", max_seq_len=200, center_qk=False))
    hier = _build(_config("dense", hierarchy=hierarchy, max_seq_len=200, center_qk=False))
    _same(dense, hier)


# --------------------------------------------------------------------------- what it removes


@pytest.mark.parametrize(
    ("kind", "hierarchy"),
    [("dense", None), ("fused", None), ("dense", {"prior_mode": "soft", "levels": [2, 6], "readout": "concat"})],
    ids=["dense", "fused", "hierarchy"],
)
def test_with_it_no_hidden_token_reaches_another_cells_prediction(kind, hierarchy):
    """Without ``center_qk``: that centres every query and key on the mean of all cell tokens, the hidden ones included (at
    weight 1 / S), the one path the option leaves open -- the same exception as the context query's."""
    batch = _batch()
    off = _build(_config(kind, visible=False, hierarchy=hierarchy, max_seq_len=200, center_qk=False))
    on = _build(_config(kind, visible=True, hierarchy=hierarchy, max_seq_len=200, center_qk=False))
    # The hidden cells' own predictions read their own residual stream; the visible cells' read only keys.
    visible = ~batch.mask.bool()

    def visible_rows(module):
        module = _stir(module).eval()
        # A random shift per hidden cell: a constant one would vanish in the input LayerNorm.
        shift = torch.zeros(batch.num_nodes, E)
        hidden = batch.mask.bool()
        shift[hidden] = 3.0 * torch.randn(int(hidden.sum()), E, generator=torch.Generator().manual_seed(7))
        original = module.gex_embedding
        outs = []
        for s in (0.0, 1.0):
            module.gex_embedding = lambda b, hidden=None, s=s: original(b, hidden=hidden) + s * shift
            with torch.no_grad():
                random.seed(0)
                out = module._common_step(batch, "regression", "node")
            del module.gex_embedding
            outs.append(out.y_pred[visible[out.view.padded_node_idx]])
        return outs

    before, after = visible_rows(off)
    assert not torch.allclose(before, after, atol=1e-6)  # without the option the hidden tokens are keys
    before, after = visible_rows(on)
    assert torch.allclose(before, after, atol=1e-6)
    # With centring the hidden tokens reach the others through the mean only.
    before, after = visible_rows(_build(_config(kind, visible=True, hierarchy=hierarchy, max_seq_len=200)))
    assert not torch.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------- the switch


def test_the_validator_takes_fused_and_the_hierarchy_and_refuses_what_cannot_block_a_key():
    from interscale.config import _validate

    _validate(_config("fused"))
    _validate(_config("dense", hierarchy={"prior_mode": "soft", "levels": [4]}, max_seq_len=200))
    for kind, block in (("performer", "performer"), ("exphormer", "exphormer")):
        cfg = _config(kind)
        cfg.model.global_component.parameters.pe.bias = []
        with pytest.raises(ValueError, match="cannot. Use one of .*fused"):
            _validate(cfg)
