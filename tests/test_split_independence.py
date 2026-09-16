"""Warning when one group's cells land in more than one split.

A probe fitted on train and scored on val measures what the embedding encodes -- unless cells of
one donor sit on both sides, in which case it can score by recognising the donor. The check
exists so that property is reported rather than assumed, and so it is reported for whatever column
carries non-independence in *this* dataset rather than for a hardcoded `donor`.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from interscale.tl.geome_utils import check_split_independence


def _adata(groups, splits):
    rng = np.random.default_rng(0)
    adata = AnnData(X=np.abs(rng.normal(size=(len(groups), 3))).astype(np.float32) + 0.1)
    adata.obs["donor"] = pd.Categorical(groups)
    adata.obs["split"] = pd.Categorical(splits)
    return adata


def test_nothing_is_checked_without_a_group_key():
    """The default. No claim either way -- not a false all-clear."""
    adata = _adata(["d1", "d1", "d2"], ["train", "val", "val"])

    assert check_split_independence(adata, "split", None) == []


def test_a_clean_split_is_silent():
    adata = _adata(["d1", "d1", "d2", "d2"], ["train", "train", "val", "val"])

    assert check_split_independence(adata, "split", "donor") == []


def test_a_straddling_group_is_reported_by_name():
    adata = _adata(["d1", "d1", "d2", "d2"], ["train", "val", "val", "val"])

    with pytest.warns(UserWarning) as record:
        messages = check_split_independence(adata, "split", "donor")

    assert len(messages) == 1
    assert "'d1'" in messages[0] and "train, val" in messages[0]
    assert "donor" in str(record[0].message)


def test_every_straddling_group_is_reported():
    adata = _adata(["d1", "d1", "d2", "d2"], ["train", "val", "train", "test"])

    with pytest.warns(UserWarning):
        messages = check_split_independence(adata, "split", "donor")

    assert len(messages) == 2


def test_a_missing_column_says_so_rather_than_passing_silently():
    """An unset-but-named key must not read as a clean bill of health."""
    adata = _adata(["d1", "d2"], ["train", "val"])

    with pytest.warns(UserWarning, match="not in adata.obs"):
        messages = check_split_independence(adata, "split", "patient")

    assert len(messages) == 1


def test_the_grouping_column_is_whatever_the_dataset_calls_it():
    """No `donor` anywhere in the API -- the next dataset may group by mouse, run or slide."""
    adata = _adata(["m1", "m1"], ["train", "val"])
    adata.obs["mouse_id"] = adata.obs["donor"]

    with pytest.warns(UserWarning):
        messages = check_split_independence(adata, "split", "mouse_id")

    assert "mouse_id" in messages[0]
