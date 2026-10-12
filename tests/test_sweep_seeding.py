"""A sweep trial builds its model from its own seed, whatever ran in the process before it.

wandb.agent runs every trial in one process. `main_sweep` used to build the model before anything
seeded (`model.train` seeds only once the model exists), so the first trial of a sweep started from
a fresh process's random state and every later one from the state the previous trial's training
left behind. In the first synth_spot round that made each sweep's first arm (`none`) a different
initialisation from all its other arms. Data, model and wandb are stubbed here: only the order of
seeding and model construction is under test.
"""

import types

import pytest

torch = pytest.importorskip("torch")

import interscale
import interscale.main_sweep as main_sweep_module
from interscale.config import get_cfg_defaults
from interscale.config.global_component_config import get_global_component_cfg


def _cfg(seed=7):
    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.wandb.use = True
    cfg.optim.seed = seed
    return cfg


@pytest.fixture
def trial(monkeypatch):
    """Runs one stubbed trial; returns the random draws its model made when it was built."""
    draws = []

    class _Model:
        @classmethod
        def _setup_anndata(cls, **kwargs):
            pass

        def __init__(self, adata, cfg):
            draws.append(torch.rand(4))

        def train(self, **kwargs):
            pass

    class _Run:
        def __init__(self, config):
            self.config, self.name, self.summary = config, "run", {}

    def run(cfg, sweep_values=None, sweep_params=None):
        values = dict(sweep_values or {})
        fake_wandb = types.SimpleNamespace(init=lambda **kwargs: _Run(values), config=values)
        monkeypatch.setattr(main_sweep_module, "wandb", fake_wandb)
        monkeypatch.setattr(main_sweep_module.sc, "read_h5ad", lambda path: object())
        monkeypatch.setattr(main_sweep_module, "remove_zero_expression_cells", lambda adata: adata)
        monkeypatch.setattr(main_sweep_module, "build_probe_obsm", lambda adata, cfg: False)
        monkeypatch.setattr(main_sweep_module, "prepare_geome_dataset", lambda adata, cfg: ([[], [], []], None))
        monkeypatch.setattr(main_sweep_module, "GraphAnnDataModule", lambda **kwargs: object())
        monkeypatch.setattr(interscale.model, "CombinedModel", _Model)
        main_sweep_module.main_sweep(lambda: cfg.clone(), "CombinedModel", "hyperparmeter", sweep_params=sweep_params)
        return draws[-1]

    return run


def test_every_trial_starts_from_its_seed_whatever_ran_before(trial):
    torch.manual_seed(123)
    first = trial(_cfg())
    torch.rand(10_000)  # what a previous trial's training draws
    second = trial(_cfg())

    assert torch.equal(first, second)
    assert torch.equal(first, torch.rand(4, generator=torch.Generator().manual_seed(7)))


def test_a_swept_seed_is_the_one_a_trial_starts_from(trial):
    swept = trial(_cfg(seed=7), {"optim.seed": 11}, sweep_params=["optim.seed"])

    assert torch.equal(swept, torch.rand(4, generator=torch.Generator().manual_seed(11)))
