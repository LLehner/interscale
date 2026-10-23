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
    from interscale.tl.masking import MASK_STRATEGIES, hides_gene_set

    if cfg.dataset.mask_strategy not in MASK_STRATEGIES:
        raise ValueError(f"dataset.mask_strategy must be one of {MASK_STRATEGIES}, got {cfg.dataset.mask_strategy!r}.")

    gene_set = list(cfg.dataset.get("mask_genes", []) or [])
    if gene_set and cfg.dataset.mask_strategy != "gene":
        raise ValueError(
            f"dataset.mask_genes {gene_set} hides a gene set, which needs mask_strategy 'gene' "
            f"(got {cfg.dataset.mask_strategy!r})."
        )
    channels = float(cfg.dataset.get("mask_channel_fraction", 0.0))
    if channels:
        where = f"dataset.mask_channel_fraction {channels:g}"
        if not 0 < channels < 1:
            raise ValueError(
                f"dataset.mask_channel_fraction is the share of genes to hide, in (0, 1), or 0 for off; got {channels:g}."
            )
        if cfg.dataset.mask_strategy != "gene":
            raise ValueError(
                f"{where} hides a random gene set, which needs mask_strategy 'gene' (got {cfg.dataset.mask_strategy!r})."
            )
        if gene_set:
            raise ValueError(
                f"{where} and dataset.mask_genes {gene_set} both name the hidden gene set (a random one, a fixed one): "
                "give one."
            )
        if cfg.dataset.mask_percentage < 1:
            import warnings

            warnings.warn(
                f"{where} with dataset.mask_percentage {cfg.dataset.mask_percentage:g}: only that share of the cells "
                "hides the channels; the others show every gene, so a neighbour can give away a hidden one. "
                "mask_percentage 1.0 hides them in every cell.",
                UserWarning,
                stacklevel=2,
            )

    if "regression" not in cfg.dataset.prediction_task:
        return

    # Spatial blocks under "node", or selecting the cells of a gene set, are targets of their own,
    # so the Bernoulli cells may be off.
    blocks_alone = cfg.dataset.get("mask_block_fraction", 0.0) > 0 and (
        cfg.dataset.mask_strategy == "node" or hides_gene_set(cfg)
    )
    if blocks_alone and cfg.dataset.mask_percentage == 0:
        return
    if not 0 < cfg.dataset.mask_percentage <= 1:
        raise ValueError(
            f"dataset.mask_percentage must be in (0, 1] for a regression task, got "
            f"{cfg.dataset.mask_percentage}. With no masking a reconstruction target is its own input."
        )


def _validate_block_mask(cfg):
    """Spatial blocks (``dataset.mask_block_fraction``) need coordinates and a node-level reconstruction."""
    fraction = cfg.dataset.get("mask_block_fraction", 0.0)
    radius = cfg.dataset.get("mask_block_radius_um", 0.0)
    if fraction == 0:
        if radius != 0:
            raise ValueError(
                f"dataset.mask_block_radius_um is {radius} but dataset.mask_block_fraction is 0, so no disc "
                "would be drawn. Set the fraction too, or the radius back to 0."
            )
        return
    if not 0 < fraction < 1:
        raise ValueError(f"dataset.mask_block_fraction must be in (0, 1), got {fraction}.")
    if not radius > 0:
        raise ValueError(f"dataset.mask_block_fraction is {fraction}; it needs mask_block_radius_um > 0, got {radius}.")
    if "regression" not in cfg.dataset.prediction_task or cfg.dataset.prediction_level != "node":
        raise ValueError(
            "Spatial blocks hide cells to reconstruct; they need a regression task at prediction_level node."
        )
    if cfg.dataset.spatial_key is None:
        raise ValueError(
            "Spatial blocks are drawn over cell coordinates, but dataset.spatial_key is unset, so none are "
            "attached to the graphs. Set it (e.g. 'spatial')."
        )
    if not cfg.dataset.spatial_unit_um > 0:
        raise ValueError(f"dataset.spatial_unit_um must be > 0, got {cfg.dataset.spatial_unit_um}.")
    from interscale.tl.masking import hides_gene_set

    if cfg.dataset.mask_strategy == "gene" and cfg.dataset.mask_percentage == 0 and not hides_gene_set(cfg):
        raise ValueError(
            "Under mask_strategy gene every cell keeps at least one masked entry, so mask_percentage 0 would "
            "still mask one entry per cell. For the discs alone use mask_strategy node."
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

    if list(cfg.probe.embeddings) == ["pe"] and not _node_pe_names(cfg):
        raise ValueError(
            "probe.embeddings is ['pe'] but no node positional encoding is enabled "
            "(model.global_component.parameters.pe.node), so there is nothing to read a target out of."
        )

    if not (
        cfg.probe.classification_targets
        or cfg.probe.regression_genes
        or cfg.probe.regression_obs
        or cfg.probe.get("gene_vector", False)
        or cfg.probe.get("attention_diagnostics", False)
    ):
        raise ValueError(
            "probe.use is True but no probe target is named (classification_targets, "
            "regression_genes, regression_obs, gene_vector, attention_diagnostics). The probe would pay for a "
            "full extra pass over train and val and log nothing."
        )

    if cfg.probe.get("attention_diagnostics", False):
        if cfg.probe.attention_graphs < 0:
            raise ValueError(
                f"probe.attention_graphs must be >= 0 (0 reads every val graph), got {cfg.probe.attention_graphs}."
            )
        radii = list(cfg.probe.attention_radii_um)
        if any(not float(r) > 0 for r in radii):
            raise ValueError(f"probe.attention_radii_um must all be > 0 (µm), got {radii}.")
        if len(set(map(float, radii))) != len(radii):
            raise ValueError(f"probe.attention_radii_um names a radius twice: {radii}.")

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


def _node_pe_names(cfg) -> list[str]:
    """``pe.node`` of the global component, or ``[]`` when there is no global component or PE block."""
    params = cfg.model.global_component.get("parameters", None)
    pe = params.get("pe", None) if params is not None else None
    return list(pe.node) if pe is not None else []


def _validate_pe(cfg):
    """Reject positional-encoding settings that would fail mid-run.

    Each encoding checks its own sub-block (``PESpec.validate``); this adds what the registry
    cannot know -- whether the coordinates a coordinate encoding needs are attached at all. Without
    ``dataset.spatial_key`` the graphs carry no ``pos``, and the failure would otherwise come at
    the first training step, after the whole dataset was built.

    Raises
    ------
    ValueError
        If ``pe.node`` names an unknown or duplicate encoding, or an encoding's settings are invalid.
    """
    params = cfg.model.global_component.get("parameters", None)
    pe = params.get("pe", None) if params is not None else None
    kinds = {"node": "node encodings", "bias": "attention biases", "rotary": "rotary encodings"}
    if pe is None or not any(pe.get(kind, []) for kind in kinds):
        return

    from interscale.module.global_modules.positional_encodings import PE_REGISTRY

    names = []
    for kind in kinds:
        listed = list(pe.get(kind, []))
        unknown = sorted(set(listed) - set(PE_REGISTRY))
        if unknown:
            raise ValueError(f"pe.{kind} names unknown positional encodings {unknown}. Known: {sorted(PE_REGISTRY)}.")
        misplaced = sorted(name for name in listed if PE_REGISTRY[name].kind != kind)
        if misplaced:
            homes = sorted({PE_REGISTRY[name].kind for name in misplaced})
            raise ValueError(
                f"pe.{kind} lists {misplaced}, which belong under "
                + " / ".join(f"pe.{home} ({kinds[home]})" for home in homes)
                + "."
            )
        duplicates = sorted({name for name in listed if listed.count(name) > 1})
        if duplicates:
            raise ValueError(f"pe.{kind} lists {duplicates} more than once; each encoding is added once.")
        names += listed

    if len(pe.get("rotary", [])) > 1:
        raise ValueError(f"pe.rotary lists {list(pe.rotary)}; at most one rotary encoding turns the queries and keys.")

    for name in names:
        PE_REGISTRY[name].validate(pe)

    needs_pos = [name for name in names if PE_REGISTRY[name].requires == "pos"]
    if needs_pos and cfg.dataset.spatial_key is None:
        raise ValueError(
            f"pe enables {needs_pos}, which read cell coordinates, but dataset.spatial_key is "
            "unset, so no coordinates are attached to the graphs. Set it (e.g. 'spatial')."
        )
    if needs_pos and not cfg.dataset.spatial_unit_um > 0:
        raise ValueError(f"dataset.spatial_unit_um must be > 0, got {cfg.dataset.spatial_unit_um}.")

    if pe.get("rotary", []):
        # Pairs of a head's dims are rotated, one pair per axis and wavelength. The head width is
        # the attention's: attention.head_dim, or n_embed / n_heads when that is 0.
        n_embed, n_heads = cfg.model.n_embed, params.n_heads
        attention = params.get("attention", None)
        head_dim = int(attention.get("head_dim", 0)) if attention is not None else 0
        width = head_dim if head_dim > 0 else (None if n_embed % n_heads else n_embed // n_heads)
        if width is None or width % 4:
            raise ValueError(
                f"pe.rotary {list(pe.rotary)} rotates pairs of each head's dims, one per axis and "
                "wavelength, so the attention head width (attention.head_dim, or n_embed / n_heads) must "
                f"be a multiple of 4; got model.n_embed {n_embed}, n_heads {n_heads}, attention.head_dim "
                f"{head_dim}."
            )


def _validate_visible_keys_only(cfg, params, kind, spec):
    """``attention.visible_keys_only`` blocks the keys that are cells hidden whole: a dense mask, and cells hidden whole."""
    from interscale.module.global_modules.attention import ATTENTION_REGISTRY

    where = "attention.visible_keys_only"
    if spec.mask != "dense":
        carriers = sorted(k for k, s in ATTENTION_REGISTRY.items() if s.mask == "dense")
        raise ValueError(
            f"{where} edits the dense attention mask; attention.kind {kind!r} has none. Use one of {carriers}."
        )
    pooling = params.get("pooling", None)
    if pooling is not None and pooling.get("kind", "none") != "none":
        raise ValueError(f"{where}: a pooled patch already leaves its hidden cells out, and its keys are patches.")
    from interscale.tl.masking import hides_gene_set

    if hides_gene_set(cfg):
        raise ValueError(
            f"{where} blocks cells hidden whole, but a hidden gene set (dataset.mask_genes, or channel masking) hides none."
        )
    blocks = cfg.dataset.get("mask_block_fraction", 0.0) > 0
    if cfg.dataset.mask_strategy == "gene" and not blocks:
        raise ValueError(f"{where}: under mask_strategy gene only spatial blocks hide whole cells; use node or blocks.")
    if cfg.dataset.mask_percentage <= 0 and not blocks:
        raise ValueError(f"{where}: nothing is masked (mask_percentage 0, no blocks), so there is no hidden key.")


def _validate_attention(cfg):
    """Reject an attention backend that cannot carry what the rest of the config asks of it.

    Each backend declares what it can carry (``attention.AttentionSpec``): the attention-bias forms,
    RoPE, the mask form, and whether its training passes form the maps an attention-reading loss
    needs. A combination outside that would otherwise fail at the first forward pass -- or, worse,
    run without the encoding. The message names what would fit instead. Each backend checks its own
    sub-block (``AttentionSpec.validate``).
    """
    params = cfg.model.global_component.get("parameters", None)
    attention = params.get("attention", None) if params is not None else None
    if attention is None:
        return

    from interscale.module.global_modules.attention import ATTENTION_REGISTRY
    from interscale.module.global_modules.positional_encodings import PE_REGISTRY
    from interscale.train.losses import LOSSES_READING_ATTENTION

    kind = attention.kind
    if kind not in ATTENTION_REGISTRY:
        raise ValueError(f"attention.kind {kind!r} is not a registered backend. Known: {sorted(ATTENTION_REGISTRY)}.")
    if not attention.head_dim >= 0:
        raise ValueError(f"attention.head_dim must be >= 0 (0 means n_embed / n_heads), got {attention.head_dim}.")
    if attention.head_dim == 0 and cfg.model.n_embed % params.n_heads:
        raise ValueError(
            f"model.n_embed ({cfg.model.n_embed}) is not divisible by n_heads ({params.n_heads}); set "
            "attention.head_dim to choose the head width directly."
        )
    spec = ATTENTION_REGISTRY[kind]
    spec.validate(attention)

    pe = params.get("pe", None)
    for name in list(pe.get("bias", [])) if pe is not None else []:
        if name not in PE_REGISTRY or PE_REGISTRY[name].forms & spec.bias_forms:
            continue  # unknown names are _validate_pe's to report
        carriers = sorted(k for k, s in ATTENTION_REGISTRY.items() if PE_REGISTRY[name].forms & s.bias_forms)
        carried = sorted(n for n, s in PE_REGISTRY.items() if s.kind == "bias" and s.forms & spec.bias_forms)
        raise ValueError(
            f"attention.kind {kind!r} cannot carry the attention bias {name!r}: the bias comes in the forms "
            f"{sorted(PE_REGISTRY[name].forms)}, the backend takes {sorted(spec.bias_forms) or 'none'}. "
            f"Backends that carry it: {carriers}. Biases this backend carries: {carried or 'none'}."
            + (f" {spec.note}" if spec.note else "")
        )
    rotary = list(pe.get("rotary", [])) if pe is not None else []
    if rotary and not spec.rotary:
        carriers = sorted(k for k, s in ATTENTION_REGISTRY.items() if s.rotary)
        raise ValueError(
            f"attention.kind {kind!r} cannot apply pe.rotary {rotary}. Backends that can: {carriers}."
            + (f" {spec.note}" if spec.note else "")
        )
    if attention.get("visible_keys_only", False):
        _validate_visible_keys_only(cfg, params, kind, spec)
    loss = cfg.optim.get("loss", None)
    if loss in LOSSES_READING_ATTENTION and not spec.training_maps:
        carriers = sorted(k for k, s in ATTENTION_REGISTRY.items() if s.training_maps)
        raise ValueError(
            f"optim.loss {loss!r} reads the attention maps in training, but attention.kind {kind!r} does not "
            f"form them there. Backends that do: {carriers}."
        )


def _validate_pooling(cfg):
    """``pooling`` (patch tokens): a known kind with a size, and nothing that reads cells one by one."""
    from interscale.module.global_modules.pooling import POOL_FEATURES, POOLING_KINDS

    params = cfg.model.global_component.get("parameters", None)
    pooling = params.get("pooling", None) if params is not None else None
    if pooling is None or pooling.get("kind", "none") == "none":
        return
    where = "model.global_component.parameters.pooling"
    if pooling.kind not in POOLING_KINDS:
        raise ValueError(f"{where}.kind must be one of {POOLING_KINDS}, got {pooling.kind!r}.")
    if (pooling.size_um > 0) == (pooling.n_patches > 0):
        raise ValueError(
            f"{where}: give exactly one of size_um (the grid's side in um) and n_patches (patches per slide); "
            f"got size_um {pooling.size_um} and n_patches {pooling.n_patches}."
        )
    if min(pooling.size_um, pooling.n_patches) < 0 or not 0 <= pooling.offset < 1:
        raise ValueError(f"{where}: size_um and n_patches must not be negative, and offset must lie in [0, 1).")
    unknown = sorted(set(pooling.features) - set(POOL_FEATURES))
    if unknown:
        raise ValueError(f"{where}.features names {unknown}; choose from {list(POOL_FEATURES)}.")
    if cfg.dataset.get("spatial_key", None) is None:
        raise ValueError(f"{where} places a patch at its cells' centroid, so it needs dataset.spatial_key.")
    if params.get("long_range_attention", False):
        raise ValueError(
            f"{where} runs the transformer on patches, which have no neighbour graph to block: "
            "set long_range_attention: false."
        )
    pe = params.get("pe", None)
    if pe is not None:
        bad = sorted(set(pe.get("node", [])) & {"lap", "rw"}) + sorted(set(pe.get("bias", [])) & {"spectral", "expression"})
        if bad:
            raise ValueError(
                f"{where} cannot carry the encodings {bad}: they read the cell graph or each cell's expression, "
                "which a patch has neither of. Use naive, sinusoidal, distance, fourier or rope."
            )
    attention = params.get("attention", None)
    if attention is not None and attention.get("kind", "dense") == "exphormer":
        raise ValueError(f"{where} has no cell graph for Exphormer's attention pattern; use another attention.kind.")
    from interscale.train.losses import LOSSES_READING_ATTENTION

    if cfg.optim.get("loss", None) in LOSSES_READING_ATTENTION:
        raise ValueError(f"{where}: optim.loss {cfg.optim.loss!r} reads cell-by-cell attention maps, which patches do not give.")


def _validate_remask(cfg):
    """``remask`` (a learned token for wholly hidden cells before the transformer): a known kind, whole cells masked."""
    from interscale.module.base._base_global_module import REMASK_KINDS

    params = cfg.model.global_component.get("parameters", None)
    remask = params.get("remask", "none") if params is not None else "none"
    if remask not in REMASK_KINDS:
        raise ValueError(f"model.global_component.parameters.remask must be one of {REMASK_KINDS}, got {remask!r}.")
    if remask == "none":
        return
    where = f"model.global_component.parameters.remask {remask!r}"
    from interscale.tl.masking import hides_gene_set

    if hides_gene_set(cfg):
        raise ValueError(
            f"{where} replaces the tokens of cells hidden whole, but a hidden gene set (dataset.mask_genes, or "
            "channel masking) hides no cell whole: it would change nothing."
        )
    blocks = cfg.dataset.get("mask_block_fraction", 0.0) > 0
    if cfg.dataset.mask_strategy == "gene" and not blocks:
        raise ValueError(
            f"{where}: under mask_strategy gene only spatial blocks (mask_block_fraction) hide whole cells; "
            "random entries almost never do."
        )
    if cfg.dataset.mask_percentage <= 0 and not blocks:
        raise ValueError(f"{where}: nothing is masked (mask_percentage 0, no blocks), so no token would be replaced.")
    pooling = params.get("pooling", None)
    if pooling is not None and pooling.get("kind", "none") != "none":
        raise ValueError(
            f"{where}: a pooled patch token already leaves out its hidden cells; there is no cell token to replace."
        )


def _validate_context_only(cfg):
    """``context_only`` (the global embedding from the other cells' tokens and the position only): a known kind, a node-level prediction, torch's dense attention."""
    from interscale.module.base._base_global_module import CONTEXT_ONLY_KINDS
    from interscale.module.global_modules.attention import attention_head_dim, attention_kind

    params = cfg.model.global_component.get("parameters", None)
    context_only = params.get("context_only", "none") if params is not None else "none"
    if context_only not in CONTEXT_ONLY_KINDS:
        raise ValueError(
            f"model.global_component.parameters.context_only must be one of {CONTEXT_ONLY_KINDS}, got {context_only!r}."
        )
    if context_only == "none":
        return
    where = f"model.global_component.parameters.context_only {context_only!r}"
    if cfg.dataset.prediction_level != "node":
        raise ValueError(f"{where} is a per-cell notion (a cell's embedding without the cell); it needs prediction_level node.")
    attention = params.get("attention", None)
    kind = attention_kind(attention)
    default_width = (
        cfg.model.n_embed % params.n_heads == 0
        and attention_head_dim(attention, cfg.model.n_embed, params.n_heads) * params.n_heads == cfg.model.n_embed
    )
    if kind != "dense" or not default_width:
        raise ValueError(
            f"{where} reads the cells' tokens with queries of their own, which only torch's dense attention at the "
            f"default head width takes; attention.kind is {kind!r} at a head width of "
            f"{attention_head_dim(attention, cfg.model.n_embed, params.n_heads)} (n_embed / n_heads would be "
            f"{cfg.model.n_embed // params.n_heads}). The other backends are self-attention only."
        )


def _validate_regime_loss(cfg):
    """``optim.regime_loss`` (each head's loss on its regime's hidden cells): known values, cells hidden whole."""
    from interscale.tl.depth import REGIME_CELLS, REGIME_KINDS
    from interscale.train._trainingplans import NO_LOSS

    regime = cfg.optim.get("regime_loss", None)
    if regime is None or regime.get("kind", "none") == "none":
        return
    where = "optim.regime_loss"
    if regime.kind not in REGIME_KINDS:
        raise ValueError(f"{where}.kind must be one of {REGIME_KINDS}, got {regime.kind!r}.")
    if not regime.depth_um > 0:
        raise ValueError(f"{where}.depth_um must be > 0 (the local component's reach, in um), got {regime.depth_um}.")
    for key in ("local_cells", "global_cells"):
        if regime[key] not in REGIME_CELLS:
            raise ValueError(f"{where}.{key} must be one of {REGIME_CELLS}, got {regime[key]!r}.")
    if not cfg.model.decoder.get("dual_decoder", False):
        raise ValueError(f"{where} gives the two heads their own cells; it needs model.decoder.dual_decoder: true.")
    if "regression" not in cfg.dataset.prediction_task or cfg.dataset.prediction_level != "node":
        raise ValueError(f"{where} splits the hidden cells of a reconstruction: regression at prediction_level node.")
    if cfg.optim.loss in NO_LOSS or cfg.optim.loss == "SCE_EntropyATT_Loss":
        raise ValueError(f"{where} needs each head's own loss, which optim.loss {cfg.optim.loss!r} does not give.")
    from interscale.tl.masking import hides_gene_set

    if hides_gene_set(cfg):
        raise ValueError(
            f"{where} sorts cells hidden whole by their depth, but a hidden gene set (dataset.mask_genes, or "
            "channel masking) hides no cell whole: every cell would be near."
        )
    blocks = cfg.dataset.get("mask_block_fraction", 0.0) > 0
    if cfg.dataset.mask_strategy == "gene" and not blocks:
        raise ValueError(f"{where}: under mask_strategy gene only spatial blocks hide whole cells; use node or blocks.")
    if cfg.dataset.get("spatial_key", None) is None:
        raise ValueError(f"{where} measures depth from the cells' coordinates, so it needs dataset.spatial_key.")
    if not cfg.dataset.spatial_unit_um > 0:
        raise ValueError(f"dataset.spatial_unit_um must be > 0, got {cfg.dataset.spatial_unit_um}.")
    if "deep" in (regime.local_cells, regime.global_cells) and not (
        blocks and cfg.dataset.mask_block_radius_um > regime.depth_um
    ):
        drawn = f"blocks of radius {cfg.dataset.mask_block_radius_um:g} um" if blocks else "no spatial blocks"
        raise ValueError(
            f"{where} gives a head the deep cells (more than {regime.depth_um:g} um from a visible one), but "
            f"{drawn} leave none: that head would have nothing to learn from. Use blocks wider than depth_um "
            "(dataset.mask_block_radius_um), or lower depth_um."
        )


def _validate_own_cell_skip(cfg):
    """``model.decoder.own_cell_skip`` is a per-cell term: it needs a node-level prediction.

    ``own_cell_skip_input`` names what it reads and means nothing without the skip.
    """
    from interscale.module.base._base_global_module import OWN_CELL_SKIP_INPUTS

    read = cfg.model.decoder.get("own_cell_skip_input", "embedding")
    if read not in OWN_CELL_SKIP_INPUTS:
        raise ValueError(f"model.decoder.own_cell_skip_input must be one of {OWN_CELL_SKIP_INPUTS}, got {read!r}.")
    if not cfg.model.decoder.get("own_cell_skip", False):
        if read != "embedding":
            raise ValueError(
                f"model.decoder.own_cell_skip_input {read!r} sets what the own-cell skip reads, but "
                "model.decoder.own_cell_skip is off."
            )
        return
    if cfg.dataset.prediction_level != "node":
        raise ValueError(
            "model.decoder.own_cell_skip adds each cell's own token to its prediction; it needs prediction_level node."
        )


def _validate(cfg):
    """Run every load-time check.

    One list, called from both branches of :func:`load_config`. The checks used to be repeated
    per branch, and ``_validate_objective`` only ever reached the defaults-only one -- so it never
    ran for a config file, which is every real run.
    """
    _validate_optim(cfg)
    _validate_block_mask(cfg)
    _validate_masking(cfg)
    _validate_objective(cfg)
    _validate_probe(cfg)
    _validate_pe(cfg)
    _validate_attention(cfg)
    _validate_pooling(cfg)
    _validate_remask(cfg)
    _validate_context_only(cfg)
    _validate_regime_loss(cfg)
    _validate_own_cell_skip(cfg)
    _validate_sce(cfg)
    _validate_center_targets(cfg)
    _validate_global_residual(cfg)
    _validate_additive(cfg)
    _validate_gex_embedding(cfg)
    _validate_precomputed_embedding(cfg)
    _validate_schedule(cfg)


def _validate_schedule(cfg):
    """``optim.schedule``, ``optim.loss_weights`` and ``optim.frozen`` (``train._schedule``)."""
    from interscale.train._schedule import validate_schedule

    validate_schedule(cfg)


def _validate_precomputed_embedding(cfg):
    """``Precomputed`` tokens need an obsm key, a known mask fill, and whole-cell masking or a hidden gene set."""
    from interscale.module.base._base_global_module import PRECOMPUTED_MASK_FILLS

    params = cfg.model.global_component.get("parameters", None)
    if params is None:
        return
    fill = params.get("precomputed_mask_fill", "learned")
    if fill not in PRECOMPUTED_MASK_FILLS:
        raise ValueError(
            f"model.global_component.parameters.precomputed_mask_fill must be one of {PRECOMPUTED_MASK_FILLS}, "
            f"got {fill!r}."
        )
    if params.get("type_gex_embedding", None) != "Precomputed":
        return
    if cfg.model.global_component.get("latent_obsm_key", None) is None:
        raise ValueError(
            "type_gex_embedding Precomputed reads the embeddings from adata.obsm, but "
            "model.global_component.latent_obsm_key is unset."
        )
    if cfg.dataset.mask_strategy == "gene" and not list(cfg.dataset.get("mask_genes", []) or []):
        raise ValueError(
            "type_gex_embedding Precomputed cannot be combined with mask_strategy gene: a cell's embedding "
            "encodes all of its expression, so single hidden genes would still reach the model through it. "
            "Use mask_strategy node (or spatial blocks), or hide a gene set (dataset.mask_genes) with an "
            "embedding computed from the other genes (interscale.tl.record_embedding_genes). Channel masking "
            "(dataset.mask_channel_fraction) hides every gene at some step, so no embedding is free of them."
        )


#: The values ``GlobalModule`` builds a token input for (``scvi`` is named in old docstrings only).
GEX_EMBEDDINGS = (None, "PCA", "NMF", "Precomputed", "Linear")
#: Where the PCA basis is fitted (``pca_fit``): the first batch the module sees, or the training slides.
PCA_FITS = ("first_batch", "train")


def _validate_gex_embedding(cfg):
    """``type_gex_embedding`` must be one the global module can build."""
    params = cfg.model.global_component.get("parameters", None)
    if params is None:
        return
    value = params.get("type_gex_embedding", None)
    if value not in GEX_EMBEDDINGS:
        raise ValueError(
            f"model.global_component.parameters.type_gex_embedding must be one of {GEX_EMBEDDINGS}, got {value!r}."
        )
    fit = params.get("pca_fit", "first_batch")
    if fit not in PCA_FITS:
        raise ValueError(f"model.global_component.parameters.pca_fit must be one of {PCA_FITS}, got {fit!r}.")
    if fit == "train" and value != "PCA":
        raise ValueError(f"pca_fit 'train' fits the PCA tokens, but type_gex_embedding is {value!r}.")


def _validate_global_residual(cfg):
    """``model.decoder.global_residual`` needs a local prediction to be residual to, cell by cell."""
    if not cfg.model.decoder.get("global_residual", False):
        return
    if not cfg.model.decoder.dual_decoder:
        raise ValueError(
            "model.decoder.global_residual makes the global head predict the local head's error; it needs "
            "model.decoder.dual_decoder: true, where both heads exist."
        )
    if cfg.dataset.prediction_level != "node":
        raise ValueError(
            "model.decoder.global_residual pairs the two heads cell by cell; it needs prediction_level node."
        )


def _validate_additive(cfg):
    """``model.decoder.additive``: one prediction from the two heads' decoders, which needs both, cell by cell.

    The loss weights must be chosen: the default mean also trains the local part to predict the whole target alone, which
    is what an additive decoder is for not doing.
    """
    from interscale.train._schedule import resolve_loss_weights

    if not cfg.model.decoder.get("additive", False):
        return
    if not cfg.model.decoder.dual_decoder:
        raise ValueError(
            "model.decoder.additive adds the two heads' decoders into one prediction; it needs "
            "model.decoder.dual_decoder: true, where both heads exist."
        )
    if cfg.model.decoder.get("global_residual", False):
        raise ValueError(
            "model.decoder.additive and model.decoder.global_residual both make the global head a sum of the two "
            "decoders (with and without a gradient into the local one): choose one."
        )
    if cfg.dataset.prediction_level != "node":
        raise ValueError("model.decoder.additive pairs the two heads cell by cell; it needs prediction_level node.")
    default_weights = resolve_loss_weights(cfg.optim.get("loss_weights", None)) is None
    if cfg.optim.get("schedule", "joint") == "joint" and default_weights:
        raise ValueError(
            "model.decoder.additive with the default optim.loss_weights (0.5 / 0.5) also trains the local part to "
            "predict the whole target alone. Set optim.loss_weights: {local: 0.0, global: 1.0} for the loss on the sum "
            "only, or a small local weight (e.g. 0.1 / 0.9)."
        )


def _validate_center_targets(cfg):
    """Reject ``optim.center_targets`` where it would change nothing or make no sense.

    Subtracting the same per-gene constant from prediction and target leaves an element-wise loss
    (MSE, SmoothL1) unchanged, so only row-normalising criteria are allowed; and a class label has
    no gene mean.
    """
    if not cfg.optim.get("center_targets", False):
        return
    from interscale.tl.masking import _ROW_STRUCTURED_LOSSES

    if "regression" not in cfg.dataset.prediction_task:
        raise ValueError("optim.center_targets centres gene expression; it needs a regression task.")
    if cfg.optim.loss not in _ROW_STRUCTURED_LOSSES:
        raise ValueError(
            f"optim.center_targets only changes row-normalising losses {list(_ROW_STRUCTURED_LOSSES)}; "
            f"with optim.loss {cfg.optim.loss!r} the same per-gene shift of prediction and target changes nothing."
        )


def _validate_sce(cfg):
    """Reject a scaled-cosine-error exponent below 1, before any data is built.

    ``optim.sce_gamma`` is read by the regression metrics of every run, not only by ``SCELoss``,
    so a bad value would otherwise surface when the training plan is constructed.
    """
    if not cfg.optim.sce_gamma >= 1:
        raise ValueError(f"optim.sce_gamma must be >= 1 (GraphMAE's scaled cosine error), got {cfg.optim.sce_gamma}.")


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
        _validate(cfg)
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

    _validate(cfg)
    cfg.freeze()
    return cfg
