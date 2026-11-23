"""``tl._preprocessing.neighbourhood_niches``: niches from neighbourhood composition, fitted on training cells, existing columns kept."""

import anndata as ad
import numpy as np
import pandas as pd

from interscale.tl._preprocessing import neighbourhood_niches


def _adata():
    rng = np.random.default_rng(0)
    ii, jj = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    pos = np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
    left = pos[:, 0] < 15
    types = np.where(left, rng.choice(["A", "B"], size=len(pos), p=[0.9, 0.1]), rng.choice(["A", "B"], size=len(pos), p=[0.1, 0.9]))
    a = ad.AnnData(rng.random((len(pos), 5)), obs=pd.DataFrame({"type": types, "sample": "s0"}, index=[f"c{i}" for i in range(len(pos))]))
    a.obsm["spatial"] = pos
    return a, left


def test_composition_niches_follow_the_neighbourhood_and_respect_existing_columns():
    a, left = _adata()
    train = np.ones(a.n_obs, bool)
    neighbourhood_niches(a, "niche", train=train, sample_key="sample", label_key="type", k=15, n_niches=2)
    niche = a.obs["niche"].astype(str).to_numpy()
    agree = max((niche[left] == niche[left][0]).mean(), 0) * 0.5 + (niche[~left] != niche[left][0]).mean() * 0.5
    assert agree > 0.9  # the two halves are two niches, though both hold both types
    before = a.obs["niche"].copy()
    neighbourhood_niches(a, "niche", train=train, sample_key="sample", label_key="type", n_niches=5)
    assert a.obs["niche"].equals(before)
    neighbourhood_niches(a, "expr_niche", train=train, sample_key="sample", k=10, n_niches=3, n_pcs=3)
    assert set(a.obs["expr_niche"].astype(str)) <= {"n0", "n1", "n2"}
