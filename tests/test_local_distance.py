"""Distances in local units (``pe.distance.units: local``; roadmap Tier 2): each query's distances scaled by its density,
``d * density_ref / r_q``, ``r_q`` its distance to its ``density_k``-th nearest cell.

Pinned here:

* **the radius** -- ``knn_radius`` is the k-th neighbour distance within a graph, from positions alone; it is precomputed onto
  the graphs only under ``local``;
* **the bias** -- the dense form is the encoder at the scaled distances, row by row (asymmetric), and the pair and row forms
  are the dense one; a query in dense tissue reads a narrower window in µm; at the reference density nothing changes;
* **through the models** -- fused and dense agree, and so do the singleton hierarchy (dense and routed) and the dense model;
* **the switch** -- the reference is derived from the data, the tag says ``-loc<k>``, the validator and the builder refuse what
  is missing.
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.module.global_modules.positional_encodings import (
    attach_positional_inputs,
    build_attention_bias,
    density_factor,
    knn_radius,
)
from interscale.tl import pad_batch

H = 2


def _stir(module, seed=0, scale=0.2):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _cfg(units="local", kind="scales", ref=20.0, k=3, unit=1.0):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.n_heads = H
    pe = params.pe
    pe.bias = ["distance"]
    pe.distance.kind = kind
    pe.distance.scale_min_um, pe.distance.scale_max_um = 30.0, 300.0
    pe.distance.max_dist = 1000.0
    pe.distance.units, pe.distance.density_k, pe.distance.density_ref_um = units, k, ref
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", unit
    return cfg


def _graphs(k=3):
    """Two graphs: a dense cluster next to sparse cells, and a small sparse one."""
    gen = torch.Generator().manual_seed(1)
    dense = torch.rand(8, 2, generator=gen) * 40.0
    sparse = torch.rand(5, 2, generator=gen) * 600.0 + 100.0
    graphs = [Data(pos=torch.cat([dense, sparse]), x=torch.zeros(13, 1)), Data(pos=torch.rand(6, 2, generator=gen) * 500.0, x=torch.zeros(6, 1))]
    for g in graphs:
        g.density_radius = knn_radius(g.pos, k)
    batch = Batch.from_data_list(graphs)
    _, padding, index, *_ = pad_batch(torch.zeros(batch.num_nodes, 2), batch.batch, 64)
    return batch, index, padding


# --------------------------------------------------------------------------- the radius


def test_the_radius_is_the_kth_neighbour_distance_and_is_precomputed_only_in_local_units():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
    assert torch.allclose(knn_radius(pos, 1), torch.tensor([1.0, 1.0, 2.0, 4.0]))
    assert torch.allclose(knn_radius(pos, 2), torch.tensor([3.0, 2.0, 3.0, 6.0]))
    assert torch.allclose(knn_radius(pos, 9), torch.tensor([7.0, 6.0, 4.0, 7.0]))  # capped at the other cells
    assert knn_radius(pos[:1], 3).tolist() == [0.0]
    graphs = [Data(pos=pos.clone(), x=torch.zeros(4, 1))]
    attach_positional_inputs(graphs, _cfg(units="um"))
    assert not hasattr(graphs[0], "density_radius")
    attach_positional_inputs(graphs, _cfg(k=2))
    assert torch.allclose(graphs[0].density_radius, knn_radius(pos, 2))


# --------------------------------------------------------------------------- the bias


@pytest.mark.parametrize(("kind", "unit"), [("scales", 1.0), ("profile", 1.0), ("linear", 1.0), ("scales", 0.5)])
def test_the_dense_bias_is_the_encoder_at_each_query_rows_scaled_distances(kind, unit):
    batch, index, padding = _graphs()
    bias = _stir(build_attention_bias(_cfg(kind=kind, unit=unit), H))
    encoder = bias.encoders["distance"]
    got = bias(batch, index, padding, torch.float32, torch.device("cpu")).view(2, H, padding.shape[1] + 1, -1)
    for g, kept in enumerate(index):
        cells = batch.batch == g
        pos, radius = batch.pos[cells][kept] * unit, batch.density_radius[cells][kept] * unit
        dist = torch.cdist(pos, pos) * density_factor(encoder, radius)[:, None]
        want = encoder._lookup(dist).permute(2, 0, 1)  # [H, n, n]
        start = padding.shape[1] - len(kept)
        assert torch.allclose(got[g, :, start:-1, start:-1], want, atol=1e-5)
    assert not torch.allclose(got, got.transpose(-1, -2), atol=1e-4)  # scaled by the query: asymmetric


def test_every_form_is_the_dense_bias_and_at_the_reference_density_nothing_changes():
    batch, index, padding = _graphs()
    bias = _stir(build_attention_bias(_cfg(), H))
    dense = bias(batch, index, padding, torch.float32, torch.device("cpu")).view(2, H, padding.shape[1] + 1, -1)
    seq = padding.shape[1]
    g, q, k = torch.meshgrid(torch.arange(2), torch.arange(seq), torch.arange(seq), indexing="ij")
    values = bias.pairs(batch, index, padding, torch.stack([g.reshape(-1), q.reshape(-1), k.reshape(-1)]), torch.float32, torch.device("cpu"))
    assert torch.allclose(values.view(2, seq, seq, H).permute(0, 3, 1, 2), dense[:, :, :seq, :seq], atol=1e-5)
    read = bias.rows(batch, index, padding, torch.float32, torch.device("cpu"))
    assert torch.allclose(torch.cat([read(0, 5), read(5, seq + 1)], dim=2), dense, atol=1e-5)
    # Every cell at the reference density: the local units are the µm.
    flat = batch.clone()
    flat.density_radius = torch.full_like(batch.density_radius, 20.0)
    um = build_attention_bias(_cfg(units="um"), H)
    um.encoders["distance"].load_state_dict(bias.encoders["distance"].state_dict(), strict=False)
    assert torch.allclose(
        bias(flat, index, padding, torch.float32, torch.device("cpu")), um(flat, index, padding, torch.float32, torch.device("cpu")), atol=1e-5
    )


def test_a_query_in_dense_tissue_reads_a_narrower_window_in_micrometres():
    batch, index, padding = _graphs()
    bias = build_attention_bias(_cfg(), H)
    encoder = bias.encoders["distance"]
    radius = batch.density_radius[:13]
    dense_q, sparse_q = int(radius[:8].argmin()), 8 + int(radius[8:].argmax())
    assert radius[dense_q] < radius[sparse_q]
    at = torch.tensor([50.0])
    b_dense = encoder._lookup(at * density_factor(encoder, radius[dense_q : dense_q + 1]))
    b_sparse = encoder._lookup(at * density_factor(encoder, radius[sparse_q : sparse_q + 1]))
    assert (b_dense < b_sparse).all()  # 50 µm is far for a query in dense tissue, near for one in sparse tissue


# --------------------------------------------------------------------------- through the models


def _vk():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _local(cfg):
    cfg.model.global_component.parameters.pe.distance.units = "local"
    cfg.model.global_component.parameters.pe.distance.density_k = 3
    return cfg


def _same_with_radius(vk, reference, other):
    """vk._same with jittered positions (a regular lattice gives almost every cell one radius) and the density radius on them."""
    original = vk._batch

    def batch():
        b = original()
        b.pos = b.pos + torch.rand(b.pos.shape, generator=torch.Generator().manual_seed(5)) * 60.0
        b.density_radius = torch.cat([knn_radius(b.pos[b.batch == g], 3) for g in range(int(b.batch.max()) + 1)])
        return b

    vk._batch = batch
    try:
        return vk._same(reference, other)
    finally:
        vk._batch = original


@pytest.mark.parametrize("chunk_pairs", [2**24, 37 * 3], ids=["one block", "blocks"])
def test_the_fused_model_in_local_units_is_the_dense_one(chunk_pairs):
    vk = _vk()
    dense = vk._build(_local(vk._config("dense", visible=False)))
    fused = vk._build(_local(vk._config("fused", visible=False, chunk_pairs=chunk_pairs)))
    assert float(dense.attention_bias.encoders["distance"].density_ref) > 0  # derived from the data
    _same_with_radius(vk, dense, fused)


@pytest.mark.parametrize(
    "hierarchy",
    [
        {"prior_mode": "soft", "lambda_init": 0.0, "lambda_learnable": "fixed", "cells_per_cluster": [1.0]},
        {
            "prior_mode": "soft",
            "lambda_init": 0.0,
            "lambda_learnable": "fixed",
            "cells_per_cluster": [1.0],
            "candidates.mode": "routed",
            "candidates.topk": [100],
            "candidates.cap_per_query": 4096,
        },
    ],
    ids=["singletons, dense", "singletons, routed"],
)
def test_the_singleton_hierarchy_in_local_units_is_the_dense_one(hierarchy):
    vk = _vk()
    dense = vk._build(_local(vk._config("dense", max_seq_len=200, center_qk=False)))
    hier = vk._build(_local(vk._config("dense", hierarchy=hierarchy, max_seq_len=200, center_qk=False)))
    _same_with_radius(vk, dense, hier)


def test_a_coarse_level_reads_its_clusters_mean_radius_and_the_window_readout_takes_the_radii():
    from interscale.evaluation.hierarchy_readouts import window_rows
    from interscale.module.global_modules.hierarchical_attention import offset_bias

    vk = _vk()
    module = vk._build(_local(vk._config("dense", hierarchy={"prior_mode": "soft", "levels": [2, 6]}, max_seq_len=200)))
    module = _stir(module).eval()
    batch = vk._batch()
    batch.density_radius = torch.cat([knn_radius(batch.pos[batch.batch == g], 3) for g in range(2)])
    with torch.no_grad(), module.recording_hierarchy() as record:
        random.seed(0)
        module._common_step(batch, "regression", "node")
    entry = record[-1]
    assert entry["radius_um"] is not None and len(entry["radius_um"]) == len(entry["pos_um"])
    pos, radius = entry["pos_um"].float(), entry["radius_um"].float()
    keys = torch.arange(len(pos)).expand(len(pos), -1)
    rows = window_rows(keys, pos, module.attention_bias, radius=radius, block=7)
    logits = offset_bias(module.attention_bias, pos[None, :, :] - pos[:, None, :], radius[:, None])
    logits = logits.masked_fill(torch.eye(len(pos), dtype=torch.bool)[..., None], float("-inf"))
    assert torch.allclose(rows, torch.softmax(logits, dim=1).permute(2, 0, 1), atol=1e-5)
    with pytest.raises(ValueError, match="needs each query's radius"):
        window_rows(keys, pos, module.attention_bias)


def test_a_clusters_radius_is_the_mean_of_its_cells():
    from interscale.module.global_modules.hierarchical_attention import level_radii
    from interscale.module.global_modules.hierarchy import build_hierarchy

    pos = torch.rand(40, 2, generator=torch.Generator().manual_seed(2)) * 500.0
    hier = build_hierarchy(pos, levels=(3, 9))
    radius = torch.rand(40, generator=torch.Generator().manual_seed(3)) * 30.0 + 1.0
    found = level_radii(hier, radius)
    assert torch.equal(found[hier.n_levels], radius)
    for level in range(hier.n_levels):
        cells = hier.assign[level]  # each cell's cluster at this level
        want = torch.zeros(hier.counts[level]).index_add(0, cells, radius) / torch.bincount(cells, minlength=hier.counts[level])
        assert torch.allclose(found[level], want, atol=1e-5)


# --------------------------------------------------------------------------- the switch


def test_the_reference_is_derived_the_tag_says_it_and_what_is_missing_is_refused():
    from anndata import AnnData

    from interscale.config import _validate_pe
    from interscale.module.global_modules.positional_encodings import resolve_density_reference
    from interscale.tl.utils import _positional_encoding_tag

    rng = np.random.default_rng(0)
    adata = AnnData(X=np.zeros((60, 2), dtype=np.float32))
    adata.obsm["spatial"] = np.concatenate([rng.uniform(0, 100, (30, 2)), rng.uniform(0, 1000, (30, 2))])
    adata.obs["sample"] = pd.Categorical(["a"] * 30 + ["b"] * 30)
    cfg = _cfg(ref=0.0)
    cfg.dataset.sample_key = ["sample"]
    resolved = resolve_density_reference(cfg, adata)
    radii = torch.cat([knn_radius(torch.as_tensor(adata.obsm["spatial"][rows]), 3) for rows in (slice(0, 30), slice(30, 60))])
    assert np.isclose(resolved.model.global_component.parameters.pe.distance.density_ref_um, float(radii.median()), rtol=1e-5)
    assert resolve_density_reference(_cfg(units="um"), adata).model.global_component.parameters.pe.distance.density_ref_um == 20.0
    assert _positional_encoding_tag(_cfg()) == "pe-distance-scales-g30-300-loc3_"
    assert _positional_encoding_tag(_cfg(units="um")) == "pe-distance-scales-g30-300_"
    bad = _cfg()
    bad.model.global_component.parameters.pe.distance.units = "neighbours"
    with pytest.raises(ValueError, match="units must be one of"):
        _validate_pe(bad)
    with pytest.raises(ValueError, match="density_ref_um is 0"):
        build_attention_bias(_cfg(ref=0.0), H)
    batch, index, padding = _graphs()
    del batch.density_radius
    with pytest.raises(ValueError, match="density_radius"):
        build_attention_bias(_cfg(), H)(batch, index, padding, torch.float32, torch.device("cpu"))
