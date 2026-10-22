"""What the label and distance probes cost, and the two switches that cut it (`masked_probing`; roadmap.md, T2 / T0).

A disc sweep spent about 25 of the 30 minutes of an arm's evaluation on `probe_labels`: one logistic regression per
target and feature set, each on every hidden training cell, the sets holding the smoother the widest. Pinned here:

* **`max_train_rows`** -- each probe is fitted on one seeded random subset of the training rows, the same rows for every
  feature set and target (so the sets stay paired), never more than the cap, none cut when they are fewer, and the test
  rows are all scored: a capped call equals the uncapped call on the subset taken by hand;
* **`joint_embeddings=False`** -- leaves out `local+global` and the sets built on it, and nothing else changes;
* **the defaults** -- fit everything, as before;
* **a cap changes the cost, not the finding** -- a label readable from the embedding is still read.
"""

import numpy as np
import pandas as pd
import pytest

from interscale.evaluation.masked_probing import _fit_rows, feature_sets, probe_covariates, probe_labels
from interscale.evaluation.online_probes import ProbeBatchFeatures


def _features(n, seed, hidden_share=0.8):
    """Dual-model features with a smoother and a position, a label readable from `global`, two covariates."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    depth = np.where(np.arange(n) % 2 == 0, 50.0, 400.0)

    def block(width, signal):
        return np.concatenate([(y * signal + rng.normal(size=n))[:, None], rng.normal(size=(n, width - 1))], axis=1)

    return ProbeBatchFeatures(
        embeddings={
            "local": block(3, 0.0),
            "global": block(3, 2.0),
            "smoother": block(6, 0.3),
            "position": block(2, 0.0),
        },
        categorical={"label": y, "other": rng.integers(0, 3, n)},
        continuous={"c1": y + 0.1 * rng.normal(size=n), "c2": -y + 0.1 * rng.normal(size=n)},
        cell_masked=rng.random(n) < hidden_share,
        depth_um=depth,
    )


def _subset(features, keep):
    """The training features restricted by hand to the rows ``keep`` (a boolean over all cells), all of them hidden."""
    sub = features.select(np.flatnonzero(keep))
    sub.cell_masked = np.ones(int(keep.sum()), bool)
    return sub


# --------------------------------------------------------------------------- the draw


def test_the_draw_is_a_seeded_subset_of_at_most_the_cap_and_leaves_a_small_set_alone():
    rows = np.zeros(1000, bool)
    rows[::2] = True  # 500 candidate rows
    found = _fit_rows(rows, 120, seed=3)
    assert found.dtype == bool and found.sum() == 120 and not (found & ~rows).any()  # a subset of the candidates
    assert np.array_equal(found, _fit_rows(rows, 120, seed=3))  # deterministic
    assert not np.array_equal(found, _fit_rows(rows, 120, seed=4))
    assert _fit_rows(rows, 500, seed=3) is rows and _fit_rows(rows, 5000, seed=3) is rows  # nothing to cut
    assert _fit_rows(rows, None, seed=3) is rows
    with pytest.raises(ValueError, match="at least 2"):
        _fit_rows(rows, 1, seed=3)


# --------------------------------------------------------------------------- the cap


def test_a_capped_label_probe_equals_the_uncapped_probe_on_the_subset_taken_by_hand():
    train, test = _features(600, 0), _features(400, 1)
    cap, seed = 150, 5
    capped = probe_labels(train, test, ["label", "other"], depth_bins_um=[0, 100, 500], max_train_rows=cap, seed=seed)
    by_hand = _subset(train, _fit_rows(train.cell_masked, cap, seed))
    assert by_hand.n_cells() == cap
    expected = probe_labels(by_hand, test, ["label", "other"], depth_bins_um=[0, 100, 500], seed=seed)
    pd.testing.assert_frame_equal(capped, expected)
    full = probe_labels(train, test, ["label", "other"], depth_bins_um=[0, 100, 500], seed=seed)
    assert not np.allclose(capped["value"].to_numpy(float), full["value"].to_numpy(float), equal_nan=True)  # it did cut
    assert (capped["n"] == full["n"]).all()  # the test rows are all scored


def test_a_capped_covariate_probe_equals_the_uncapped_probe_on_the_subset_taken_by_hand():
    train, test = _features(600, 0), _features(400, 1)
    cap, seed = 150, 5
    capped = probe_covariates(train, test, ["c1", "c2"], depth_bins_um=[0, 100, 500], max_train_rows=cap, seed=seed)
    by_hand = _subset(train, _fit_rows(train.cell_masked, cap, seed))
    expected = probe_covariates(by_hand, test, ["c1", "c2"], depth_bins_um=[0, 100, 500])
    pd.testing.assert_frame_equal(capped, expected)
    assert not np.allclose(
        capped["value"], probe_covariates(train, test, ["c1", "c2"], depth_bins_um=[0, 100, 500])["value"]
    )


def test_a_cap_over_the_number_of_hidden_training_rows_changes_nothing():
    train, test = _features(300, 0), _features(300, 1)
    pd.testing.assert_frame_equal(
        probe_labels(train, test, ["label"], max_train_rows=10_000), probe_labels(train, test, ["label"])
    )
    pd.testing.assert_frame_equal(
        probe_covariates(train, test, ["c1"], max_train_rows=10_000), probe_covariates(train, test, ["c1"])
    )


def test_one_draw_serves_every_set_and_target(monkeypatch):
    """The fits all see the cap's rows, and the sets are paired: the draw is made once per call."""
    import interscale.evaluation.masked_probing as probing

    draws = []
    original = probing._fit_rows
    monkeypatch.setattr(probing, "_fit_rows", lambda *a, **k: draws.append(1) or original(*a, **k))
    fits = []
    from sklearn.linear_model import LogisticRegression

    fit = LogisticRegression.fit
    monkeypatch.setattr(
        LogisticRegression, "fit", lambda self, x, y, *a, **k: fits.append(len(y)) or fit(self, x, y, *a, **k)
    )
    train, test = _features(400, 0), _features(300, 1)
    probe_labels(train, test, ["label", "other"], max_train_rows=90, seed=1)
    n_sets = len(feature_sets(train))
    assert len(draws) == 1
    assert fits == [90] * (2 * n_sets)


def test_a_cap_changes_the_cost_not_the_finding():
    train, test = _features(1200, 0), _features(800, 1)
    for cap in (None, 300):
        labels = probe_labels(train, test, ["label"], max_train_rows=cap)
        accuracy = labels[labels["metric"] == "balanced_accuracy"].set_index("feature_set")["value"]
        assert accuracy["global"] > 0.75, cap  # the label is in the embedding
        assert accuracy["local"] < 0.6 and accuracy["position"] < 0.6, cap  # and not in what carries none


# --------------------------------------------------------------------------- the joint embeddings


def test_feature_sets_without_the_joint_embeddings_drop_exactly_the_sets_built_on_them():
    features = _features(10, 0)
    everything, split = feature_sets(features), feature_sets(features, joint_embeddings=False)
    dropped = set(everything) - set(split)
    assert dropped == {"local+global", "smoother+local+global", "smoother+position+local+global"}
    assert set(split) <= set(everything) and all(np.array_equal(split[k], everything[k]) for k in split)
    assert len(everything) == 12 and all("local+global" in name for name in dropped)
    only_global = ProbeBatchFeatures(embeddings={k: np.zeros((3, 2)) for k in ("global", "smoother", "position")})
    assert set(feature_sets(only_global, joint_embeddings=False)) == set(feature_sets(only_global))  # nothing to drop


def test_the_probes_without_the_joint_embeddings_equal_the_full_ones_on_the_other_sets():
    train, test = _features(500, 0), _features(300, 1)
    full = probe_labels(train, test, ["label"], depth_bins_um=[0, 100, 500])
    split = probe_labels(train, test, ["label"], depth_bins_um=[0, 100, 500], joint_embeddings=False)
    joint = full["feature_set"].str.contains("local+global", regex=False)
    assert joint.any() and not split["feature_set"].str.contains("local+global", regex=False).any()
    pd.testing.assert_frame_equal(full[~joint].reset_index(drop=True), split)
    full_cov = probe_covariates(train, test, ["c1", "c2"])
    split_cov = probe_covariates(train, test, ["c1", "c2"], joint_embeddings=False)
    joint_cov = full_cov["feature_set"].str.contains("local+global", regex=False)
    pd.testing.assert_frame_equal(full_cov[~joint_cov].reset_index(drop=True), split_cov)


def test_the_defaults_fit_everything_as_before():
    train, test = _features(300, 0), _features(300, 1)
    default = probe_labels(train, test, ["label"])
    explicit = probe_labels(train, test, ["label"], max_train_rows=None, joint_embeddings=True)
    pd.testing.assert_frame_equal(default, explicit)
    assert "smoother+position+local+global" in set(default["feature_set"])
