"""SAFFRON-style probes of embeddings (``evaluation.sae_probe``; ``.claude/ccc_model_plan.md``, E1).

Pinned: the Matryoshka BatchTopK SAE reconstructs a sparse code and keeps its features alive; a feature that tracks a variable
within every domain scores near 1 and far above its within-domain permutation null, and noise scores at the null; OMP needs one
feature for a variable that is one feature; the spatial variables (signed boundary distance, density, density near a structure)
are what they say; the whole probe runs and finds a planted gradient.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from interscale.evaluation import sae_probe


def _sparse_data(n=3000, d=16, m=32, active=3, seed=0):
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(m, d))
    codes = np.zeros((n, m))
    for i in range(n):
        codes[i, rng.choice(m, size=active, replace=False)] = rng.exponential(size=active)
    return codes @ atoms, codes


def test_the_sae_reconstructs_a_sparse_code_and_keeps_its_features():
    z, _ = _sparse_data()
    fitted = sae_probe.train_sae(z, n_features=64, k_fraction=0.08, epochs=30, seed=0, device="cpu")
    assert fitted.r2 > 0.7
    assert fitted.dead < 0.5
    feats = fitted.features(z[:100])
    assert feats.shape == (100, 64) and (feats >= 0).all()


def test_best_features_finds_a_tracking_feature_against_the_null():
    rng = np.random.default_rng(1)
    n, m = 2000, 40
    domains = np.repeat(np.arange(4), n // 4)
    tau = rng.normal(size=n)
    features = rng.exponential(size=(n, m))
    features[:, 7] = np.exp(tau) + 0.01 * rng.normal(size=n)  # monotone in tau: Spearman ~1
    table = sae_probe.best_features(features, pd.DataFrame({"tau": tau, "noise": rng.normal(size=n)}), domains, prefixes=(10, 20, 40), n_permutations=50)
    row = table.set_index("variable")
    assert row.loc["tau", "score"] > 0.95 and row.loc["tau", "feature"] == 7
    assert row.loc["tau", "block_0"] > 0.95  # feature 7 is in the first prefix block
    assert row.loc["noise", "score"] < row.loc["noise", "null95"] + 0.05


def test_omp_needs_one_feature_for_a_single_feature_variable():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(600, 20))
    tau = 2.0 * features[:, 5]
    groups = np.repeat(np.arange(3), 200)
    curve = sae_probe.omp_curve(features, tau, groups, max_features=4)
    assert curve["omp"][0] > 0.99 and curve["ridge"] > 0.99


def test_spatial_variables_are_what_they_say():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [100.0, 0.0], [110.0, 0.0]])
    obs = pd.DataFrame({"type": ["islet", "islet", "acinar", "acinar", "immune"], "fov": [1, 1, 2, 2, 2]})
    graph = np.zeros(5, dtype=int)
    specs = [
        {"name": "islet_signed", "kind": "distance_to_label", "column": "type", "values": ["islet"], "signed": True},
        {"name": "immune_density", "kind": "density_of_label", "column": "type", "values": ["immune"], "radius_um": 15.0},
        {"name": "immune_near_islet", "kind": "density_near_label", "column": "type", "values": ["immune"], "radius_um": 15.0,
         "near_values": ["islet"], "near_um": 50.0},
        {"name": "fov", "kind": "obs", "column": "fov"},
    ]
    out = sae_probe.spatial_variables(pos, graph, specs, obs=obs)
    np.testing.assert_allclose(out["islet_signed"], [-20.0, -10.0, 10.0, 90.0, 100.0])
    np.testing.assert_allclose(out["immune_density"], [0, 0, 0, 0.5, 0.5])
    assert np.isnan(out["immune_near_islet"][3]) and out["immune_near_islet"][0] == 0
    np.testing.assert_array_equal(out["fov"], [1, 1, 2, 2, 2])


def test_probe_embedding_finds_a_planted_gradient():
    rng = np.random.default_rng(3)
    n, d = 1600, 8
    pos = rng.uniform(0, 1000, size=(n, 2))
    graph = np.repeat(np.arange(4), n // 4)
    gradient = pos[:, 0] / 1000.0
    z = rng.normal(size=(n, d)) * 0.3
    z[:, 0] += 3 * gradient
    variables = pd.DataFrame({"x_axis": gradient, "noise": rng.normal(size=n)})
    out = sae_probe.probe_embedding(
        "toy", z, z, variables, graph, graph, pos, graph, n_features=32, epochs=20, n_permutations=30, scales=True,
        decoder_weight=rng.normal(size=(5, d)), gene_names=[f"g{i}" for i in range(5)],
    )
    best = out["sae_best"].set_index("variable")
    assert best.loc["x_axis", "score"] > best.loc["x_axis", "null95"] + 0.2
    assert best.loc["x_axis", "score"] > best.loc["noise", "score"]
    assert {"sae_fit", "sae_scales", "sae_genes"} <= set(out)
