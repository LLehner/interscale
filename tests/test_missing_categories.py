"""Warning when a split contains no cells of some category of a configured obs column.

Both consequences of the narrowing are hard to trace back from where they surface: a
`prediction_obs` missing from a split stops training with "y_true and y_pred must have the same
shape", which names nothing useful, and an optional annotation missing from a split leaves the
encoding correct but the data empty for that category. The warning exists so neither has to be
diagnosed from the symptom.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from interscale.config import get_cfg_defaults
from interscale.tl.geome_utils import warn_missing_categories


@pytest.fixture
def cfg():
    return get_cfg_defaults()


def _adata(train_labels, val_labels, categories=("a", "b", "c", "d")):
    labels = list(train_labels) + list(val_labels)
    n = len(labels)
    rng = np.random.default_rng(0)
    adata = AnnData(X=np.abs(rng.normal(size=(n, 4))).astype(np.float32) + 0.1)
    adata.obs["cell_type"] = pd.Categorical(labels, categories=list(categories))
    adata.obs["split"] = pd.Categorical(["train"] * len(train_labels) + ["val"] * len(val_labels))
    return adata


def test_nothing_is_warned_when_every_split_has_every_category(cfg):
    cfg.dataset.prediction_obs = "cell_type"
    adata = _adata(["a", "b", "c", "d"], ["a", "b", "c", "d"])

    assert warn_missing_categories(adata, "split", cfg) == []


def test_nothing_is_checked_when_no_column_is_configured(cfg):
    """prediction_obs is None for a regression run and no annotation key is set by default."""
    adata = _adata(["a", "b"], ["a"])

    assert warn_missing_categories(adata, "split", cfg) == []


def test_a_missing_prediction_obs_category_warns_with_the_failure_it_causes(cfg):
    cfg.dataset.prediction_obs = "cell_type"
    adata = _adata(["a", "b", "c", "d"], ["a", "c", "d"])

    with pytest.warns(UserWarning) as record:
        messages = warn_missing_categories(adata, "split", cfg)

    assert len(messages) == 1
    message = str(record[0].message)
    assert "'val'" in message and "'cell_type'" in message and "['b']" in message
    # The point of the warning: name the failure it is about to cause.
    assert "y_true and y_pred must have the same shape" in message
    assert "n_output is 4" in message


def test_a_missing_annotation_category_warns_without_claiming_a_failure(cfg):
    """The encoding is repaired for these, so the warning is about the data, not a crash."""
    cfg.dataset.celltype_key = "cell_type"
    adata = _adata(["a", "b", "c", "d"], ["a", "b", "d"])

    with pytest.warns(UserWarning):
        messages = warn_missing_categories(adata, "split", cfg)

    assert len(messages) == 1
    assert "ordering is preserved" in messages[0]
    assert "y_true and y_pred" not in messages[0]


def test_every_split_and_every_column_is_reported(cfg):
    cfg.dataset.prediction_obs = "cell_type"
    cfg.dataset.celltype_key = "condition_like"
    adata = _adata(["a", "b", "c"], ["a", "c", "d"])
    adata.obs["condition_like"] = pd.Categorical(
        ["x"] * 3 + ["y"] * 3, categories=["x", "y", "z"]
    )

    with pytest.warns(UserWarning):
        messages = warn_missing_categories(adata, "split", cfg)

    # One message per (column, split), listing every category missing there -- not one per
    # category. cell_type: 'd' absent from train, 'b' from val. condition_like: 'y' and 'z' from
    # train, 'x' and 'z' from val. So four messages, two per split.
    assert len(messages) == 4
    assert sum("'train'" in m for m in messages) == 2
    assert sum("'val'" in m for m in messages) == 2
    assert any("['y', 'z']" in m for m in messages)


def test_the_column_is_named_once_per_split_not_once_per_category(cfg):
    cfg.dataset.prediction_obs = "cell_type"
    adata = _adata(["a", "b", "c", "d"], ["a"])

    with pytest.warns(UserWarning):
        messages = warn_missing_categories(adata, "split", cfg)

    assert len(messages) == 1
    assert "categories ['b', 'c', 'd']" in messages[0]
