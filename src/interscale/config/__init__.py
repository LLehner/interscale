from pathlib import Path

from yacs.config import CfgNode as CN

from .dataset_config import get_dataset_cfg
from .global_component_config import get_global_component_cfg
from .local_component_config import get_local_component_cfg
from .model_config import get_model_cfg
from .optim_config import get_optim_cfg
from .probe_config import get_probe_cfg
from .wandb_config import get_wandb_cfg


def get_cfg_defaults():
    """Loads the default settings from the .py files in the config folder."""
    cfg = CN()

    # Load configurations
    cfg = get_wandb_cfg(cfg)
    cfg = get_model_cfg(cfg)
    cfg = get_optim_cfg(cfg)
    cfg = get_dataset_cfg(cfg)
    cfg = get_probe_cfg(cfg)

    return cfg


def _normalise_cfg_paths(cfg_path):
    """Coerce the ``cfg_path`` argument into a list of existing ``Path`` objects.

    Accepts a single str/Path (the historical signature) or an iterable of them,
    so callers that layer several files can pass a list.
    """
    if cfg_path is None:
        return []
    if isinstance(cfg_path, str | Path):
        paths = [Path(cfg_path)]
    else:
        paths = [Path(p) for p in cfg_path]

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("config file(s) not found: " + ", ".join(missing))
    return paths


def _peek_component_names(cfg_paths):
    """Find the local/global component names declared across ``cfg_paths``.

    The component *parameter* schemas (``model.local_component.parameters.*``) only exist
    once ``get_local_component_cfg`` / ``get_global_component_cfg`` have added them, so the
    names have to be read before any file is merged. Files are scanned in merge order and
    the last file naming a component wins, matching what the merge itself would produce.

    Uses ``getattr(..., "name", None)`` rather than attribute access: a file may set
    ``model.global_component.parameters.max_seq_len`` without naming the component (a
    dataset-level file layered on top of a base file that does the naming), and plain
    attribute access raises ``AttributeError`` on the absent ``name`` key.
    """
    local_component_name = None
    global_component_name = None

    for path in cfg_paths:
        with path.open() as f:
            temp_cfg = CN.load_cfg(f)

        model = getattr(temp_cfg, "model", None)
        if model is None:
            continue

        local_component = getattr(model, "local_component", None)
        if local_component is not None and getattr(local_component, "name", None):
            local_component_name = local_component.name

        global_component = getattr(model, "global_component", None)
        if global_component is not None and getattr(global_component, "name", None):
            global_component_name = global_component.name

    return local_component_name, global_component_name


def _coerce_override_values(cfg, overrides):
    """Return ``overrides`` with ints promoted to float where the cfg default is a float.

    ``merge_from_list`` rejects an int for a float key outright (yacs only tolerates a type
    mismatch when one side is ``None``). YAML writes ``0`` as an int, so an override of
    ``dataset.mask_percentage: 0`` against the ``0.2`` default would raise even though the
    value is perfectly valid. Promote it instead of making callers write ``0.0``.

    ``overrides`` is the flat ``[key, value, key, value, ...]`` form ``merge_from_list`` takes.
    """
    coerced = list(overrides)

    for i in range(0, len(coerced) - 1, 2):
        key, value = coerced[i], coerced[i + 1]
        if not isinstance(value, int) or isinstance(value, bool):
            continue

        # Walk the dotted key to the current value; a missing key is left alone so that
        # merge_from_list raises its own (clearer) error about the unknown key.
        node = cfg
        try:
            *parents, leaf = str(key).split(".")
            for part in parents:
                node = node[part]
            current = node[leaf]
        except (KeyError, TypeError):
            continue

        if isinstance(current, float):
            coerced[i + 1] = float(value)

    return coerced


def _validate_masking(cfg):
    """Reject masking settings that would silently train on an uncorrupted input.

    A reconstruction task with ``mask_percentage`` at 0 has the identity map as its solution, and
    every metric looks excellent, so it fails loudly instead.

    Raises
    ------
    ValueError
        If ``mask_strategy`` is unknown, or ``mask_percentage`` is not a rate in (0, 1] for a
        regression task.
    """
    from interscale.tl.masking import MASK_STRATEGIES

    if cfg.dataset.mask_strategy not in MASK_STRATEGIES:
        raise ValueError(f"dataset.mask_strategy must be one of {MASK_STRATEGIES}, got {cfg.dataset.mask_strategy!r}.")

    if "regression" not in cfg.dataset.prediction_task:
        return

    if not 0 < cfg.dataset.mask_percentage <= 1:
        raise ValueError(
            f"dataset.mask_percentage must be in (0, 1] for a regression task, got "
            f"{cfg.dataset.mask_percentage}. With no masking a reconstruction target is its own input."
        )


def _validate_objective(cfg):
    """Reject a config that leaves the model with nothing to optimise.

    ``optim.loss: none`` switches the reconstruction criterion off so an auxiliary objective --
    VICReg, say -- can be the whole task. With no auxiliary weight set either, the total loss is
    a constant zero: training runs, every metric is logged, and not one parameter moves in a way
    that means anything. That is a slow failure to notice, so it fails at config load instead.

    Raises
    ------
    ValueError
        If no reconstruction criterion and no auxiliary term is enabled.
    """
    from interscale.train._trainingplans import NO_LOSS

    if cfg.optim.loss not in NO_LOSS:
        return
    weights = dict(cfg.optim.aux_loss_weights)
    if any(float(w) != 0.0 for w in weights.values()):
        return
    raise ValueError(
        f"optim.loss is {cfg.optim.loss!r} (no reconstruction criterion) and every "
        "optim.aux_loss_weights entry is 0, so the objective is a constant zero. Set a "
        "criterion, or give an auxiliary term a weight (e.g. aux_loss_weights.vicreg)."
    )


def _validate_optim(cfg):
    """Reject configs whose training-length settings stop a run inside the LR warm-up.

    ``CosineWarmupScheduler`` ramps the learning rate linearly over ``optim.lr_warmup`` epochs
    (interval ``"epoch"``), so a run that early-stops before the ramp finishes has never trained
    at ``optim.lr`` and reports a near-initialisation model -- typically a collapsed constant
    predictor, which is easy to misread as a class-imbalance problem.

    Raises
    ------
    ValueError
        If ``optim.min_epochs`` does not exceed ``optim.lr_warmup`` while the warm-up scheduler
        is in use.
    """
    if cfg.optim.lr_scheduler != "CosineWarmupScheduler":
        return
    if cfg.optim.min_epochs <= cfg.optim.lr_warmup:
        raise ValueError(
            f"optim.min_epochs ({cfg.optim.min_epochs}) must be greater than optim.lr_warmup "
            f"({cfg.optim.lr_warmup}): with CosineWarmupScheduler the LR is still ramping until "
            f"epoch {cfg.optim.lr_warmup}, so EarlyStopping (patience={cfg.optim.patience}) can "
            "end the run before the model has trained at optim.lr. Raise optim.min_epochs (2x "
            "lr_warmup is the convention here) or lower optim.lr_warmup."
        )


def _validate_probe(cfg):
    """Reject probe settings that would cost a forward pass and measure nothing.

    Every check here is a silent no-op that is otherwise only discoverable by reading an empty
    panel at the end of a sweep: a probe with no targets, a categorical target whose annotation
    was never attached to the graphs, or an embedding name no module produces.

    Raises
    ------
    ValueError
        If ``probe.use`` is set but the probe block cannot produce a single number.
    """
    from interscale.evaluation.online_probes import EMBEDDINGS

    if not cfg.probe.use:
        return

    if cfg.probe.every_n_epochs < 1:
        raise ValueError(f"probe.every_n_epochs must be >= 1, got {cfg.probe.every_n_epochs} (the probe never runs).")

    unknown = sorted(set(cfg.probe.embeddings) - set(EMBEDDINGS))
    if unknown:
        raise ValueError(f"probe.embeddings names no such representation: {unknown}. Known: {sorted(EMBEDDINGS)}.")
    if not cfg.probe.embeddings:
        raise ValueError(
            "probe.use is True but probe.embeddings is empty, so there is nothing to read a target out of."
        )

    if not (cfg.probe.classification_targets or cfg.probe.regression_genes or cfg.probe.regression_obs):
        raise ValueError(
            "probe.use is True but no probe target is named (classification_targets, "
            "regression_genes, regression_obs). The probe would pay for a full extra pass over "
            "train and val and log nothing."
        )

    # Numeric obs targets reach the graphs through an obsm matrix, because geome cannot attach a
    # numeric obs column. Without the key the graphs carry no probe_targets and the probe raises
    # mid-run instead of at config load.
    if cfg.probe.regression_obs and not cfg.dataset.probe_obsm_key:
        raise ValueError(
            f"probe.regression_obs names {list(cfg.probe.regression_obs)} but "
            "dataset.probe_obsm_key is unset, so those columns are never attached to the graphs."
        )

    # A categorical target is read off the PyG Data object, and the annotation only reaches it
    # when its `dataset.*_key` is set -- see `tl.geome_utils.OPTIONAL_FIELDS`. Caught here
    # rather than at the first probe epoch, which on a warm-up-limited run is 40 epochs in.
    from interscale.tl.geome_utils import OPTIONAL_FIELDS

    for target in cfg.probe.classification_targets:
        if target not in OPTIONAL_FIELDS:
            raise ValueError(
                f"probe.classification_targets names '{target}', which is not an attachable "
                f"annotation. Known: {sorted(k for k, (_, src) in OPTIONAL_FIELDS.items() if src == 'obs')}."
            )
        cfg_key, _source = OPTIONAL_FIELDS[target]
        if getattr(cfg.dataset, cfg_key, None) is None:
            raise ValueError(
                f"probe.classification_targets asks for '{target}' but dataset.{cfg_key} is unset, "
                f"so that annotation is never attached to the graphs and the probe has no labels."
            )


def load_config(cfg_path=None, overrides=None):
    """Loads and optionally overrides config values.

    Parameters
    ----------
    cfg_path : str or pathlib.Path or list, optional
        Path to the config file to load, or a list of paths merged left to right so
        later files override earlier ones. If None, only default values are used.
    overrides : list, optional
        Flat ``[key, value, key, value, ...]`` overrides applied after every file, in the
        form ``merge_from_list`` expects (e.g. ``["dataset.prediction_obs", "condition"]``).
        Keys are dotted paths and must already exist in the config, so a typo raises
        rather than silently doing nothing.

    Returns
    -------
    CN
        Configuration object with all settings loaded.
    """
    # First get all default configs including local component defaults
    cfg = get_cfg_defaults()

    cfg_paths = _normalise_cfg_paths(cfg_path)

    # Documented as defaults-only, but the code below dereferences cfg_path
    # unconditionally, so None used to raise AttributeError too.
    if not cfg_paths and not overrides:
        _validate_optim(cfg)
        _validate_masking(cfg)
        _validate_objective(cfg)
        _validate_probe(cfg)
        cfg.freeze()
        return cfg

    local_component_name, global_component_name = _peek_component_names(cfg_paths)
    if local_component_name:
        # Ensure local component configs are loaded before merging
        cfg = get_local_component_cfg(cfg, local_component_name)
    if global_component_name:
        # Ensure global component configs are loaded before merging
        cfg = get_global_component_cfg(cfg, global_component_name)

    for path in cfg_paths:
        cfg.merge_from_file(str(path))

    if overrides:
        cfg.merge_from_list(_coerce_override_values(cfg, overrides))

    _validate_optim(cfg)
    _validate_masking(cfg)
    _validate_probe(cfg)
    cfg.freeze()
    return cfg
