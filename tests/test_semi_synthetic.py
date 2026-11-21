"""Planted interactions on real cells (``evaluation.semi_synthetic``; ``.claude/ccc_model_plan.md``, E2).

Pinned: the kernels are what they say (step within the length, exp of the nearest sender's distance, a saturating field); the
spike adds counts only to the chosen genes of the receiver type, more where the kernel is larger; a control pair adds nothing;
no gene is used twice; the truth is in synth_spot's format; the model-free check sees a spiked programme and not a control.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from interscale.evaluation import semi_synthetic as ss


def _tissue(n=3000, g=40, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1500, size=(n, 2))
    types = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.2, 0.2])
    counts = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(counts))
    adata.var_names = [f"g{i}" for i in range(g)]
    adata.obs["type"] = pd.Categorical(types)
    adata.obs["section"] = pd.Categorical(np.where(pos[:, 0] < 750, "s1", "s2"))
    adata.obsm["spatial"] = pos
    adata.layers["raw"] = sp.csr_matrix(counts)
    norm = counts / counts.sum(1, keepdims=True) * 1e4
    adata.layers["log1p_norm"] = sp.csr_matrix(np.log1p(norm))
    return adata


def test_kernels():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 0.0], [500.0, 0.0]])
    codes = np.array([0, 1, 1, 1])
    graph = np.zeros(4, dtype=int)
    f, edges = ss.kernel_values(pos, graph, codes, ss.PlantedPair("S", "R", "step", 20.0), 0, 1)
    np.testing.assert_array_equal(f, [0, 1, 0, 0])
    assert set(edges["receiver"]) == {1}
    f, _ = ss.kernel_values(pos, graph, codes, ss.PlantedPair("S", "R", "exp", 50.0), 0, 1)
    np.testing.assert_allclose(f[1:], np.exp(-np.array([10.0, 50.0, 500.0]) / 50.0))
    f, edges = ss.kernel_values(pos, graph, codes, ss.PlantedPair("S", "R", "field", 100.0), 0, 1)
    assert f[1] > f[2] > f[3] >= 0 and f[3] == 0  # 500 um is beyond 3 l


def test_field_follows_sender_density_instead_of_saturating():
    # dense senders: 20 within reach of every receiver would put 1 - exp(-sum w) at ~1 everywhere; the scaled field keeps the median
    # reached receiver at 0.5 and ranks receivers by how many senders are near
    rng = np.random.default_rng(0)
    n = 3000
    pos = rng.uniform(0, 3000, size=(n, 2))
    senders = rng.random(n) < np.clip(pos[:, 0] / 3000, 0.05, 1.0) * 0.5  # sender density rises along x
    codes = np.where(senders, 0, 1)
    f, _ = ss.kernel_values(pos, np.zeros(n, dtype=int), codes, ss.PlantedPair("S", "R", "field", 500.0), 0, 1)
    recv = codes == 1
    assert abs(np.median(f[recv & (f > 0)]) - 0.5) < 0.01
    q1, q3 = np.quantile(f[recv], [0.25, 0.75])
    assert q3 - q1 > 0.1
    from scipy.stats import spearmanr

    assert spearmanr(pos[recv, 0], f[recv]).statistic > 0.3


def test_spike_in_adds_counts_only_where_planned_and_writes_the_truth():
    adata = _tissue()
    before = adata.layers["raw"].toarray().copy()
    pairs = [ss.PlantedPair("B", "A", "exp", 100.0, fold=3.0, n_genes=4), ss.PlantedPair("C", "A", "step", 30.0, n_genes=4, spike=False)]
    excluded = list(adata.var_names[:5])
    table = ss.spike_in(adata, pairs, label_key="type", graph_key="section", seed=0, exclude_genes=excluded)
    chosen = {g.strip() for genes in table["genes"] for g in genes.split(",")}
    assert not chosen & set(excluded)
    after = adata.layers["raw"].toarray()
    diff = after - before
    spiked = [adata.var_names.get_loc(g.strip()) for g in table.loc[0, "genes"].split(",")]
    control = [adata.var_names.get_loc(g.strip()) for g in table.loc[1, "genes"].split(",")]
    assert not set(spiked) & set(control)
    others = np.setdiff1d(np.arange(adata.n_vars), spiked)
    assert diff[:, others].sum() == 0 and diff.min() >= 0
    is_a = adata.obs["type"].to_numpy() == "A"
    assert diff[~is_a].sum() == 0 and diff[is_a][:, spiked].sum() > 0
    f = adata.obs[f"spike_{pairs[0].name}"].to_numpy()
    q1, q3 = np.quantile(f[is_a], [0.25, 0.75])
    high, low = is_a & (f >= q3), is_a & (f <= q1)
    assert diff[high][:, spiked].mean() > diff[low][:, spiked].mean()
    edges = adata.uns["synthetic"]["interaction_edges"]
    assert {"sender", "receiver", "program", "range_class", "slide", "weight"} <= set(edges.columns)
    assert (adata.var.loc[adata.var_names[spiked], "role"] == "response").all()
    assert (adata.var.loc[adata.var_names[control], "role"] == "control").all()
    # the model-free check after normalising as the inputs do
    counts = adata.layers["raw"].toarray()
    adata.layers["log1p_norm"] = sp.csr_matrix(np.log1p(counts / counts.sum(1, keepdims=True) * 1e4))
    check = ss.signal_check(adata, table, label_key="type", graph_key="section").set_index("program")
    assert check.loc[pairs[0].name, "spearman_resid_kernel"] > 0.2
    assert abs(check.loc[pairs[1].name, "spearman_resid_kernel"]) < 0.15
