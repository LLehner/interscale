"""Load-time config checks must run for config files -- the path every real run takes.

`load_config` has two branches, defaults-only and file/overrides, and the checks used to be listed
per branch. `_validate_objective` only ever reached the defaults-only one, where `optim.loss` is
never `none`, so it guarded nothing.
"""

import pytest

from interscale.config import load_config


def _write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return path


def test_a_config_file_with_no_objective_is_rejected(tmp_path):
    """No reconstruction criterion and no auxiliary weight: the loss is a constant zero, so training
    runs and logs every metric while nothing learns."""
    path = _write(tmp_path, "optim:\n  loss: none\n")

    with pytest.raises(ValueError, match="constant zero"):
        load_config(path)


def test_an_auxiliary_weight_makes_a_reconstruction_free_config_valid(tmp_path):
    path = _write(tmp_path, "optim:\n  loss: none\n  aux_loss_weights:\n    vicreg: 1.0\n")

    assert load_config(path).optim.loss == "none"


def test_overrides_without_a_file_are_checked_too():
    """Overrides alone take the file branch, so they must not skip the check either."""
    with pytest.raises(ValueError, match="constant zero"):
        load_config(overrides=["optim.loss", "none"])
