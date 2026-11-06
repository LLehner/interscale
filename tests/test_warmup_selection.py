"""The best checkpoint and early stopping ignore the LR warm-up (``train._utils.PostWarmup*``).

Both classes override private Lightning hooks. A renamed hook would not fail: the override would simply never be called and
warm-up epochs would be selectable again, so these tests run a real ``Trainer``.
"""

import pytest

torch = pytest.importorskip("torch")

import lightning.pytorch as pl
from lightning.pytorch.demos.boring_classes import BoringModel

from interscale.train._utils import PostWarmupEarlyStopping, PostWarmupModelCheckpoint


class _Rising(BoringModel):
    """A monitored score that only gets worse, so the first epoch allowed is the best one."""

    def on_validation_epoch_end(self):
        self.log("score", float(self.current_epoch))


def _trainer(callbacks, tmp_path, epochs=6):
    return pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir=tmp_path,
        accelerator="cpu",
        logger=False,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


@pytest.mark.parametrize("start", [0, 3])
def test_the_best_checkpoint_is_the_first_epoch_after_the_warm_up(tmp_path, start):
    checkpoint = PostWarmupModelCheckpoint(
        start_epoch=start, dirpath=tmp_path, monitor="score", mode="min", save_top_k=1
    )
    _trainer([checkpoint], tmp_path).fit(_Rising())
    assert checkpoint.best_model_score.item() == start
    assert torch.load(checkpoint.best_model_path, weights_only=False)["epoch"] == start


def test_a_start_past_the_end_saves_nothing_so_the_caller_caps_it(tmp_path):
    checkpoint = PostWarmupModelCheckpoint(start_epoch=99, dirpath=tmp_path, monitor="score", mode="min", save_top_k=1)
    _trainer([checkpoint], tmp_path, epochs=2).fit(_Rising())
    assert checkpoint.best_model_path == ""


def test_patience_starts_after_the_warm_up(tmp_path):
    """Patience 1 on a score that only rises: tracking starts at epoch 3, the next epoch is one bad check, the one after stops."""
    stopper = PostWarmupEarlyStopping(start_epoch=3, monitor="score", mode="min", patience=1, min_delta=0.0)
    trainer = _trainer([stopper], tmp_path, epochs=12)
    trainer.fit(_Rising())
    assert stopper.best_score.item() == 3
    assert stopper.stopped_epoch == 4
