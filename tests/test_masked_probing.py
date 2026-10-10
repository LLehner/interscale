"""Post-hoc probes on masked passes (evaluation.masked_probing) and the attention helpers.

Synthetic arrays only: a representation that carries a target must score well, one that does
not must score ~0, a hidden value must never reach a probe, and the two prediction paths (one
fit for whole hidden cells, one per gene for hidden entries) must agree.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from interscale.evaluation.masked_probing import (
    draw_masks,
    feature_sets,
    probe_covariates,
    probe_expression,
    probe_labels,
)
from interscale.evaluation.online_probes import ProbeBatchFeatures
from interscale.tl.masking import MASK_VALUE

N, E, G = 1200, 6, 12


def _features(seed, *, kind="entries", rate=0.3, labels=False, covariates=False):
    """Cells whose first 8 genes are linear in `signal` (plus noise) and last 4 are pure noise.

    Representations: `signal` carries the structure, `noise` nothing, and `expression` is the masked
    input -- which, if a hidden value leaked into it, would predict even the noise genes."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(N, E))
    weights = np.random.default_rng(100).normal(size=(E, 8))
    y = np.concatenate([signal @ weights + 0.3 * rng.normal(size=(N, 8)), rng.normal(size=(N, 4))], axis=1) + 5.0
    if kind == "entries":
        hidden = rng.random((N, G)) < rate
    else:
        hidden = np.repeat((rng.random(N) < rate)[:, None], G, axis=1)
    feats = ProbeBatchFeatures(
        embeddings={
            "signal": signal,
            "noise": rng.normal(size=(N, E)),
            "expression": np.where(hidden, MASK_VALUE, y),
        },
        continuous={f"g{j}": y[:, j] for j in range(G)},
        cell_masked=hidden.any(1),
        gene_masked={f"g{j}": hidden[:, j] for j in range(G)},
    )
    if labels:
        feats.categorical["side"] = (signal[:, 0] > 0).astype(int)
        feats.categorical["random"] = rng.integers(0, 3, size=N)
    if covariates:
        feats.continuous["d1"] = signal @ np.array([1.0, 0, 0, 0, 0, 0]) + 10
        feats.continuous["d2"] = signal @ np.array([0, 2.0, 0, 0, 0, 0]) + 20
    return feats


GENES = [f"g{j}" for j in range(G)]
GROUPS = {"structured": GENES[:8], "background": GENES[8:]}


def _value(table, **where):
    sub = table
    for key, value in where.items():
        sub = sub[sub[key] == value]
    assert len(sub) == 1, where
    return float(sub["value"].iloc[0])


def test_draw_masks_hides_entries_or_whole_cells_from_its_own_generator():
    from torch_geometric.data import Data

    datas = [Data(x=torch.zeros(400, 10)), Data(x=torch.zeros(300, 10))]
    state = torch.get_rng_state()
    draw_masks(datas, "entries", 0.3, seed=1)
    assert torch.equal(state, torch.get_rng_state()), "drew from the global torch RNG"
    entries = [d.gene_mask.clone() for d in datas]
    assert abs(float(torch.cat(entries).float().mean()) - 0.3) < 0.03
    assert all(torch.equal(d.mask, d.gene_mask.any(1)) for d in datas)

    draw_masks(datas, "cells", 0.3, seed=1)
    for d in datas:
        assert (d.gene_mask.all(1) | ~d.gene_mask.any(1)).all(), "cells: a cell is hidden whole or not at all"
    draw_masks(datas, "entries", 0.3, seed=1)
    assert all(torch.equal(a, d.gene_mask) for a, d in zip(entries, datas, strict=True)), "same seed, same masks"
    with pytest.raises(ValueError, match="kind"):
        draw_masks(datas, "genes", 0.3, seed=1)


def test_the_gene_vector_probe_reads_structure_and_nothing_leaks():
    """Centred cosine: high from the representation that carries the structured genes, ~0 from
    noise, and ~0 on the background genes from every representation -- including the masked input,
    which would read them perfectly if a hidden value leaked into it."""
    summary, per_gene = probe_expression(_features(0), _features(1), GENES, gene_groups=GROUPS)

    assert _value(summary, feature_set="signal", scope="structured", metric="cosine_centred") > 0.9
    assert abs(_value(summary, feature_set="noise", scope="structured", metric="cosine_centred")) < 0.1
    for name in ("signal", "noise", "expression"):
        assert abs(_value(summary, feature_set=name, scope="background", metric="cosine_centred")) < 0.1, name
    assert per_gene.shape == (G, len(feature_sets(_features(0))) + 1)
    assert (per_gene.loc[GENES[8:], "signal"].abs() < 0.15).all()


def test_raw_cosine_has_the_mean_profile_as_its_floor():
    """Every gene has mean ~5 here, so predicting the training mean already scores >0.9 on the
    raw cosine; the centred cosine of that prediction is undefined, and of the signal well above 0."""
    summary, _ = probe_expression(_features(0), _features(1), GENES)
    floor = _value(summary, feature_set="mean", scope="all", metric="cosine_raw")
    assert floor > 0.9
    assert np.isnan(_value(summary, feature_set="mean", scope="all", metric="cosine_centred"))
    assert _value(summary, feature_set="signal", scope="all", metric="cosine_raw") > floor
    assert _value(summary, feature_set="signal", scope="all", metric="cosine_centred") > 0.5


def test_whole_hidden_cells_take_one_fit_and_agree_with_the_per_gene_path():
    from interscale.evaluation.masked_probing import _ridge_predictions

    train, test = _features(0, kind="cells"), _features(1, kind="cells")
    y = np.stack([train.continuous[g] for g in GENES], 1)
    m_tr = np.stack([train.gene_masked[g] for g in GENES], 1)
    m_te = np.stack([test.gene_masked[g] for g in GENES], 1)
    joint = _ridge_predictions(train.embeddings["signal"], y, m_tr, test.embeddings["signal"], m_te, 1.0)
    per_gene = np.column_stack(
        [
            _ridge_predictions(
                train.embeddings["signal"],
                y[:, [j]],
                m_tr[:, [j]] & m_tr[:, [0]],
                test.embeddings["signal"],
                m_te[:, [j]],
                1.0,
            )[:, 0]
            for j in range(G)
        ]
    )
    np.testing.assert_allclose(joint, per_gene, atol=1e-8)
    assert np.isnan(joint[~m_te]).all() and not np.isnan(joint[m_te]).any()


def test_labels_are_read_from_hidden_cells_only():
    train, test = _features(0, kind="cells", labels=True), _features(1, kind="cells", labels=True)
    table = probe_labels(train, test, ["side", "random"])

    assert _value(table, target="side", feature_set="signal", metric="balanced_accuracy") > 0.95
    assert abs(_value(table, target="side", feature_set="noise", metric="balanced_accuracy") - 0.5) < 0.1
    assert _value(table, target="side", feature_set="majority", metric="balanced_accuracy") == pytest.approx(0.5)
    assert abs(_value(table, target="random", feature_set="signal", metric="balanced_accuracy") - 1 / 3) < 0.1
    assert (table["n"] == int(test.cell_masked.sum())).all()


def test_covariate_vectors_are_scored_by_the_cosine_of_their_z_scores():
    train, test = _features(0, kind="cells", covariates=True), _features(1, kind="cells", covariates=True)
    table = probe_covariates(train, test, ["d1", "d2"])

    assert _value(table, feature_set="signal", metric="cosine_centred") > 0.99
    assert abs(_value(table, feature_set="noise", metric="cosine_centred")) < 0.1


# --------------------------------------------------------------------------- attention helpers


def test_subset_edges_renumbers_to_the_subset_and_drops_edges_leaving_it():
    from interscale.evaluation.downstream_regression import subset_edges

    edges = pd.DataFrame({"sender": [0, 2, 4, 5], "receiver": [2, 4, 5, 1], "range_class": ["a", "b", "c", "d"]})
    keep = np.array([False, False, True, False, True, True])  # cells 2, 4, 5 become 0, 1, 2

    out = subset_edges(edges, keep)

    assert out[["sender", "receiver"]].values.tolist() == [[0, 1], [1, 2]]
    assert out["range_class"].tolist() == ["b", "c"]


def test_attention_distance_profile_averages_per_bin():
    from interscale.evaluation.downstream_regression import attention_distance_profile

    pairs = pd.DataFrame({"dist": [50.0, 150.0, 250.0, 260.0, 900.0], "attn": [1.0, 3.0, 2.0, 4.0, 7.0]})
    out = attention_distance_profile(pairs, [0, 200, 500, 1000])

    assert out["mean"].tolist() == [2.0, 3.0, 7.0]
    assert out["count"].tolist() == [2, 2, 1]


def test_score_against_truth_takes_renumbered_edges():
    """Scoring a subset with subset_edges gives what scoring the full object does on those cells."""
    from anndata import AnnData

    from interscale.evaluation.downstream_regression import score_against_truth, subset_edges

    rng = np.random.default_rng(0)
    n = 60
    edges = pd.DataFrame({"sender": rng.integers(0, n, 400), "receiver": rng.integers(0, n, 400), "range_class": "x"})
    full = AnnData(np.zeros((n, 1)))
    full.uns["synthetic"] = {"interaction_edges": edges}
    keep = np.arange(n) >= 20
    i, j = rng.integers(20, n, 3000), rng.integers(20, n, 3000)
    attn = rng.random(3000) + np.isin((j * n + i), edges["sender"] * n + edges["receiver"]) * 0.5
    pairs_full = pd.DataFrame({"i": i, "j": j, "attn": attn})
    pairs_sub = pd.DataFrame({"i": i - 20, "j": j - 20, "attn": attn})

    a = score_against_truth(full, pairs_full)
    b = score_against_truth(AnnData(np.zeros((n - 20, 1))), pairs_sub, edges=subset_edges(edges, keep))
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[a["range_class"] == "x", "roc_auc"].iloc[0] > 0.6
