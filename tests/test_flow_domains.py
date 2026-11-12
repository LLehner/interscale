"""The flow-domain pipeline (``evaluation/flow_domains.py``; roadmap.md "Vector-field context", 2026-11-12): attention -> net flow
-> cell vectors (the chapter's sender-anchored construction and our midpoint one) -> gridded fields -> k-means domains -> roles,
and the planted-truth scores.

Pinned here:

* **the relevance** -- ReLU of the layer / head mean of grad ⊙ attn, zero diagonal; the head mean of the last layer alike;
* **the vectors** -- a sender's vector points at its receivers, a non-sender's is zero; the exact divergence ``out - in`` sums to 0 and is
  positive at planted senders, negative at receivers;
* **the pipeline** -- on a lattice with two planted senders read by their surroundings, both constructions put the senders in a source
  domain and rank them first by divergence (AUROC 1 for the exact divergence), and the net balance of the senders' domain is negative;
* **the agreement** -- ARI 1 for a relabelled partition, ~0 for a random one;
* **the model hook** -- ``attention_with_gradients`` returns maps and gradients of the masked loss over the kept cells, and leaves the
  module's mode and gradients as they were.
"""

import numpy as np
import pandas as pd
import pytest

from interscale.evaluation import flow_domains as fd


def _lattice(side=12, spacing=100.0):
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float) * spacing


def _planted(pos, senders, radius=250.0, weight=0.5):
    """``A``: every cell within ``radius`` of a sender reads it with ``weight`` (plus a little uniform reading), no reciprocity."""
    n = len(pos)
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 0.002, size=(n, n))
    np.fill_diagonal(A, 0.0)
    pairs = []
    for s in senders:
        d = np.linalg.norm(pos - pos[s], axis=1)
        for i in np.flatnonzero((d > 0) & (d <= radius)):
            A[i, s] += weight
            pairs.append((s, i))
    return A, pd.DataFrame(pairs, columns=["sender", "receiver"]).assign(program="planted")


def test_relevance_and_head_mean():
    rng = np.random.default_rng(1)
    maps = rng.uniform(size=(2, 3, 5, 5))
    grads = rng.normal(size=(2, 3, 5, 5))
    rel = fd.relevance_matrix(maps, grads)
    want = np.maximum((maps * grads).mean(axis=(0, 1)), 0.0)
    np.fill_diagonal(want, 0.0)
    assert np.allclose(rel, want) and (rel >= 0).all() and np.allclose(np.diag(rel), 0)
    hm = fd.head_mean_matrix(maps)
    assert np.allclose(hm, maps[-1].mean(0) - np.diag(np.diag(maps[-1].mean(0))))
    with pytest.raises(ValueError, match="differ in shape"):
        fd.relevance_matrix(maps, grads[:1])


def test_vectors_and_exact_divergence_point_from_senders_to_receivers():
    pos = _lattice()
    corner, interior = 0, 78  # a sender at the lattice corner (receivers on one side) and one inside (receivers all around)
    A, pairs = _planted(pos, [corner, interior])
    source, receiver, flow = fd.normalised_net_flow(A)
    assert flow.max() == pytest.approx(1.0)
    v = fd.sender_vectors(pos, source, receiver, flow, 400.0)
    non_senders = np.setdiff1d(np.arange(len(pos)), [corner, interior])
    noise = np.median(np.linalg.norm(v[non_senders], axis=1))
    # the corner sender's vector points into the lattice (towards its receivers), far above the noise ...
    assert v[corner, 0] > 0 and v[corner, 1] > 0 and np.linalg.norm(v[corner]) > 10 * noise
    # ... while the interior sender's receivers surround it and its vector cancels: the chapter's construction cannot see a
    # symmetrically read source (its own stated limitation); the exact divergence can.
    assert np.linalg.norm(v[interior]) < 2 * noise
    div = fd.exact_divergence(len(pos), source, receiver, flow)
    assert abs(div.sum()) < 1e-5
    assert div[corner] > 0 and div[interior] > 0
    assert div[[corner, interior]].min() > div[non_senders].max()
    receivers = np.unique(pairs["receiver"])
    assert np.median(div[receivers]) < 0
    points, arrows = fd.midpoint_vectors(pos, source, receiver, flow)
    assert points.shape == arrows.shape == (len(flow), 2)
    assert np.allclose(np.linalg.norm(arrows, axis=1), flow)


def _pipeline(A, pos, construction, **over):
    settings = dict(d_max_um=400.0, resolution=32, sigma_grid_um=120.0, sigma_smooth_um=60.0, k=4, kappa=0.5, seed=0,
                    tissue_radius_um=150.0)
    settings.update(over)
    return fd.run_pipeline(A, pos, construction=construction, **settings)


def test_the_midpoint_pipeline_finds_planted_senders_as_sources():
    pos = _lattice(side=16)
    senders = [40, 200]
    A, pairs = _planted(pos, senders)
    result = _pipeline(A, pos, "midpoint")
    assert result["domains"].shape == (32, 32) and (result["domains"] >= 0).any()
    assert result["cell_domain"].shape == (len(pos),)
    tables = fd.pipeline_tables(result, pairs)
    scores = tables["scores"].set_index(["program", "role"])
    assert scores.loc[("planted", "sender"), "auroc_exact"] == pytest.approx(1.0)
    assert scores.loc[("planted", "receiver"), "auroc_exact"] > 0.9
    assert scores.loc[("planted", "sender"), "auroc_field"] > 0.9
    assert scores.loc[("planted", "sender"), "in_role_domain"] == 1.0
    balance = tables["balance"].set_index("domain")["net_balance"]
    assert balance.sum() == pytest.approx(0.0, abs=1e-9)
    for s in senders:
        assert balance.loc[result["cell_domain"][s]] < 0  # the senders' domain is a net sender
    assert {"channel", "domain", "mean_divergence", "role", "n_grid"} <= set(tables["roles"].columns)
    assert set(tables["roles"]["role"]) <= set(fd.ROLES)
    with pytest.raises(ValueError, match="construction"):
        fd.run_pipeline(A, pos, construction="other", d_max_um=1.0, sigma_grid_um=1.0, sigma_smooth_um=0.0)


def test_the_sender_construction_misses_symmetric_sources_where_the_midpoint_one_finds_them():
    pos = _lattice(side=16)
    A, pairs = _planted(pos, [40, 200])  # both inside the lattice: read from all sides
    by_construction = {}
    for construction in ("sender", "midpoint"):
        tables = fd.pipeline_tables(_pipeline(A, pos, construction), pairs)
        by_construction[construction] = tables["scores"].set_index(["program", "role"]).loc[("planted", "sender")]
    assert by_construction["midpoint"]["auroc_field"] > 0.9
    assert by_construction["sender"]["auroc_field"] < by_construction["midpoint"]["auroc_field"]
    # the exact divergence is construction-free
    assert by_construction["sender"]["auroc_exact"] == by_construction["midpoint"]["auroc_exact"] == pytest.approx(1.0)


def test_channels_split_the_fields_by_the_senders_label():
    pos = _lattice(side=10)
    A, pairs = _planted(pos, [12, 77])
    labels = (pos[:, 0] > pos[:, 0].mean()).astype(int)  # two halves; one sender in each
    result = fd.run_pipeline(A, pos, labels=labels, construction="sender", d_max_um=400.0, resolution=20, sigma_grid_um=120.0,
                             sigma_smooth_um=0.0, k=3, seed=1)
    assert result["fields"]["V"].shape[0] == 2
    # channel 0's field has mass only on its half (the sender at 12 is in the low-x half)
    V = result["fields"]["V"]
    low, high = V[0, :10].__abs__().sum(), V[0, 10:].__abs__().sum()
    assert low > high
    assert set(result["roles"]["channel"]) == {0, 1}


def test_partition_agreement():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=(16, 16))
    b = (a + 1) % 4
    a[0, 0] = -1  # off tissue in one partition is left out
    assert fd.partition_agreement(a, b) == pytest.approx(1.0)
    c = rng.integers(0, 4, size=(16, 16))
    assert abs(fd.partition_agreement(a, c)) < 0.1


def test_planted_flags_by_programme():
    pairs = pd.DataFrame({"sender": [0, 0, 3], "receiver": [1, 2, 4], "program": ["p", "p", "q"]})
    senders, receivers = fd.planted_flags(pairs, 6)
    assert senders["p"].tolist() == [True, False, False, False, False, False]
    assert receivers["q"].tolist() == [False, False, False, False, True, False]
    assert senders["all"].sum() == 2 and receivers["all"].sum() == 3


def test_attention_with_gradients_on_a_small_model():
    torch = pytest.importorskip("torch")
    from anndata import AnnData
    from torch_geometric.data import Batch, Data

    from interscale.config import _validate, get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.model.global_model import GlobalModel

    G, E = 6, 8
    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 64, 2, 2, 0.0
    params.type_gex_embedding = "Linear"
    cfg.optim.loss = "SCELoss"
    cfg.dataset.mask_strategy = "node"
    cfg.dataset.mask_percentage = 0.25
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    _validate(cfg)
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(0)
    module = GlobalModel(adata, cfg=cfg).module
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=gen))
    n = 16
    pos = torch.tensor(_lattice(side=4), dtype=torch.float32)
    data = Data(x=torch.rand(n, G, generator=gen) * 3 + 1.0, pos=pos, edge_index=torch.zeros(2, 0, dtype=torch.long))
    data.mask = torch.arange(n) % 4 == 0
    data.obs_names = torch.arange(n)
    module.train()
    maps, grads, nodes, hidden = fd.attention_with_gradients(module, Batch.from_data_list([data]), "regression", "node")
    assert maps.shape == grads.shape == (2, 2, n, n)
    assert len(nodes) == n and hidden.sum() == 4
    assert np.abs(grads).sum() > 0
    assert module.training  # the mode is restored
    assert all(p.grad is None for p in module.parameters())  # and the gradients cleared
    rel = fd.relevance_matrix(maps, grads)
    assert rel.shape == (n, n) and (rel >= 0).all()
