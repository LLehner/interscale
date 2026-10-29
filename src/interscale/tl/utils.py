import random
import re

import numpy as np
import torch


def set_full_reproducibility(seed: int = 42) -> None:
    """Set seeds across all libraries to ensure reproducibility.

    Parameters
    ----------
    seed : int (default: 42)
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def local_reach_hops(module) -> int:
    """How many hops of the spatial graph the local component has mixed into each embedding.

    Read off the module being trained, never off the config, so that a *loss* making a claim
    about "what the local component saw" uses the same number the long-range mask was built
    with. A combined module and a GlobalModel's transformer carry it as ``local_mask_hops`` (0
    when there is no local component); a local-only module reports its ``receptive_field_hops``.
    """
    global_module = getattr(module, "global_module", module)
    if hasattr(global_module, "local_mask_hops"):
        return int(global_module.local_mask_hops)
    return int(module.receptive_field_hops)


def check_and_update_cfg(
    cfg,
    prediction_task: str = None,
    prediction_level: str = None,
    prediction_obs: str = None,
    layer_key: str = None,
    sample_key: str = None,
):
    """Checks for jupyer notebook specifications and updates cfg if necessary."""
    cfg.set_new_allowed(True)
    cfg.defrost()
    if prediction_task != cfg.dataset.prediction_task:
        print(f"Update prediction task (from '{cfg.dataset.prediction_task}' to '{prediction_task}')")
        cfg.dataset.prediction_task = prediction_task
    if prediction_level != cfg.dataset.prediction_level:
        print(f"Update prediction level (from '{cfg.dataset.prediction_level}' to '{prediction_level}')")
        cfg.dataset.prediction_level = prediction_level
    if prediction_obs != cfg.dataset.prediction_obs:
        print(f"Update prediction obs (from '{cfg.dataset.prediction_obs}' to '{prediction_obs}')")
        cfg.dataset.prediction_obs = prediction_obs
    if layer_key != cfg.dataset.layer_key:
        print(f"Update layer key (from '{cfg.dataset.layer_key}' to '{layer_key}')")
        cfg.dataset.layer_key = layer_key
    if sample_key != cfg.dataset.sample_key[0]:
        print(f"Update sample key (from '{cfg.dataset.sample_key}' to '{sample_key}')")
        cfg.dataset.sample_key = sample_key
    cfg.freeze()
    return cfg


def _positional_encoding_tag(cfg) -> str:
    """``pe-<names>_`` when the global component enables positional encodings, else ``""``.

    Part of the checkpoint prefix because the prefix is also how a sweep keeps its trials apart:
    it carries the seed but nothing else that a PE ablation varies, so without the tag every arm
    of a PE sweep would overwrite the previous arm's checkpoint. Empty when off, so every name from
    before PEs is unchanged. A distance bias of the non-default `linear` kind is tagged
    `distance-linear`, RoPE of the non-default `mixed` kind `rope-mixed`, and a directional
    (non-isotropic) Fourier bias `fourier-aniso`, since each is a different model under the same
    encoding name.
    """
    params = cfg.model.global_component.get("parameters", None)
    pe = params.get("pe", None) if params is not None else None
    names = list(pe.node) + list(pe.get("bias", [])) + list(pe.get("rotary", [])) if pe is not None else []
    if "distance" in names and pe.distance.get("kind", "profile") == "scales":
        d = pe.distance
        window = f"{d.shape[0]}{float(d.scale_min_um):g}-{float(d.scale_max_um):g}"
        strength = f"-s{float(d.strength):g}" if float(d.strength) != 1.0 else ""
        names[names.index("distance")] = f"distance-scales-{window}{strength}{'-fz' if d.frozen else ''}"
    elif "distance" in names and pe.distance.get("kind", "profile") != "profile":
        names[names.index("distance")] = f"distance-{pe.distance.kind}"
    if "rope" in names and pe.rope.get("kind", "axial") != "axial":
        names[names.index("rope")] = f"rope-{pe.rope.kind}"
    if "fourier" in names and not pe.fourier.get("isotropic", True):
        names[names.index("fourier")] = "fourier-aniso"
    return f"pe-{'+'.join(names)}_" if names else ""


def _attention_tag(cfg) -> str:
    """``attn-<kind>[<settings>][-hd<head_dim>][-cqk]_`` when the global component uses a non-default attention, else ``""``.

    ``<settings>``: ``-a<alpha>`` (``l`` when learned) for entmax, ``-k<k>`` for top-k.

    Part of the checkpoint prefix for the same reason as :func:`_positional_encoding_tag`: an
    attention sweep's arms would otherwise overwrite each other's checkpoints. Empty for dense
    attention at the default head width, so every name from before backends is unchanged.
    """
    params = cfg.model.global_component.get("parameters", None)
    attention = params.get("attention", None) if params is not None else None
    if attention is None:
        return ""
    kind, head_dim = str(attention.get("kind", "dense")), int(attention.get("head_dim", 0))
    center = bool(attention.get("center_qk", False))
    visible = bool(attention.get("visible_keys_only", False))
    if kind == "dense" and head_dim == 0 and not center and not visible:
        return ""
    settings = ""
    if kind == "entmax":
        entmax = attention.get("entmax", {})
        learn = bool(entmax.get("learn_alpha", True))
        settings = f"-a{float(entmax.get('alpha', 1.5)):g}" + ("l" if learn else "")
    elif kind == "topk":
        settings = f"-k{int(attention.get('topk', {}).get('k', 32))}"
    return (
        f"attn-{kind}{settings}"
        + (f"-hd{head_dim}" if head_dim > 0 else "")
        + ("-cqk" if center else "")
        + ("-vk" if visible else "")
        + "_"
    )


def _pooling_tag(cfg) -> str:
    """``pool-grid<side>um`` / ``pool-grid-n<patches>`` (+ ``-o<offset>``, ``-<features>``) for pooled tokens, else ``""``.

    Pooled runs differ from per-cell runs in every output, so an arm without this tag would overwrite
    the checkpoint of the same model on cells (and of another patch size).
    """
    params = cfg.model.global_component.get("parameters", None)
    pooling = params.get("pooling", None) if params is not None else None
    if pooling is None or pooling.get("kind", "none") == "none":
        return ""
    size = f"n{int(pooling.n_patches)}" if pooling.n_patches > 0 else f"{float(pooling.size_um):g}um"
    offset = f"-o{float(pooling.offset):g}" if pooling.offset else ""
    features = "".join(f"-{f}" for f in pooling.features)
    return f"pool-{pooling.kind}{'-' if size.startswith('n') else ''}{size}{offset}{features}_"


def _hierarchy_tag(cfg) -> str:
    """``hier-...`` for the hierarchical attention (every setting away from its default), else ``""``.

    Builder and top level, the coarse levels (``l<counts>`` or ``c<cells per cluster>``), the prior mode, then only what differs
    from the defaults, so a sweep over any of them never overwrites a checkpoint.
    """
    params = cfg.model.global_component.get("parameters", None)
    hier = params.get("hierarchy", None) if params is not None else None
    if hier is None or hier.get("kind", "none") == "none":
        return ""
    top = hier.top_level
    tag = f"hier-{hier.builder}"
    if top.mode == "patches":
        tag += "-pt" + (f"{float(top.patch_size_um):g}um" if top.patch_size_um > 0 else "")
    if int(top.min_cells) != 1:
        tag += f"-mc{int(top.min_cells)}"
    if len(hier.levels):
        tag += "-l" + "x".join(str(int(c)) for c in hier.levels)
    elif len(hier.cells_per_cluster):
        tag += "-c" + "x".join(f"{float(c):g}" for c in hier.cells_per_cluster)
    tag += "-" + {"none": "pnone", "soft": "soft", "factorized": "fac", "factorized_percell": "fpc"}[hier.prior_mode]
    if not hier.size_correction:
        tag += "-nosc"
    if hier.direction_prior:
        tag += "-dir"
        if float(hier.mu_init) != 0.5 or hier.mu_learnable != "per_head":
            tag += f"-mu{float(hier.mu_init):g}{hier.mu_learnable[0]}"
    if float(hier.lambda_init) != 1.0 or hier.lambda_learnable != "per_head":
        tag += f"-lam{float(hier.lambda_init):g}{hier.lambda_learnable[0]}"
    if hier.prior_refresh == "once":
        tag += "-once"
    if not hier.share_level_params:
        tag += "-nshare"
    if hier.coarse_bias != "shared":
        tag += f"-cb{hier.coarse_bias}"
    if hier.readout != "none":
        tag += f"-ro{hier.readout[:2]}"
    if hier.pooling != "mean":
        tag += "-ap"
    if len(hier.pool_features):
        tag += "-f" + "".join(f[0] for f in hier.pool_features)
    if hier.shift:
        tag += "-sh"
    if int(hier.neighbors) != 8:
        tag += f"-nb{int(hier.neighbors)}"
    cand = hier.candidates
    if cand.mode == "routed":
        tag += "-rt" + "x".join(str(int(k)) for k in cand.topk) + f"e{int(cand.explore_n)}"
        tag += ("p" if cand.explore_strategy == "prior_sampled" else "") + f"c{int(cand.cap_per_query)}"
    return tag + "_"


def _decoder_tag(cfg) -> str:
    """``dec-<type>[-skip|-skipraw]_`` for a decoder other than the plain linear one or with the own-cell skip, else ``""``.

    The decoder type was never part of a checkpoint name, so an arm that varies it would overwrite the arm
    that does not; the default (``linear``, no skip) adds nothing and every earlier name is unchanged. ``-skipraw``:
    the skip reads the cell's genes (``model.decoder.own_cell_skip_input: raw``), ``-skip`` its token.
    """
    decoder = cfg.model.get("decoder", None)
    if decoder is None:
        return ""
    kind, skip = decoder.get("type", "linear"), bool(decoder.get("own_cell_skip", False))
    if kind in (None, "linear") and not skip:
        return ""
    raw = skip and decoder.get("own_cell_skip_input", "embedding") == "raw"
    return f"dec-{kind or 'linear'}{('-skipraw' if raw else '-skip') if skip else ''}_"


def _embedding_tag(cfg) -> str:
    """The token input of a GlobalModel, for its checkpoint name.

    ``emb-linear_`` for the learned linear input; ``emb-pre-<key>-learned_`` / ``emb-pre-<key>-m1_``
    for precomputed embeddings read from ``obsm[<key>]`` (letters and digits only, so ``X_pca`` is
    ``Xpca``) with a learned or a -1 mask fill, so two embeddings never share a checkpoint (added
    2026-10-17: checkpoints made with Precomputed before then carry no tag, and their hidden cells'
    embeddings were not masked). Only the global-only prefix carries it: a CombinedModel ignores
    ``type_gex_embedding``. PCA fitted on the training slides (``pca_fit: train``) is ``emb-pca-train_``;
    PCA on the first batch and NMF add nothing, so their names are unchanged.
    """
    params = cfg.model.global_component.get("parameters", None)
    embedding = params.get("type_gex_embedding", None) if params is not None else None
    if embedding == "Linear":
        return "emb-linear_"
    if embedding == "PCA" and params.get("pca_fit", "first_batch") == "train":
        return "emb-pca-train_"
    if embedding == "Precomputed":
        fill = params.get("precomputed_mask_fill", "learned")
        key = re.sub(r"[^0-9A-Za-z]+", "", str(cfg.model.global_component.get("latent_obsm_key", None) or ""))
        return f"emb-pre-{key + '-' if key else ''}{'learned' if fill == 'learned' else 'm1'}_"
    return ""


def _training_tag(cfg) -> str:
    """``var-<switches>_`` for training-side switches off their default, else ``""``.

    For the same reason as the encoding and attention tags: a sweep over one of these would
    otherwise write every arm to one checkpoint. ``ctr``: ``optim.center_targets``; ``res``:
    ``model.decoder.global_residual``; ``add``: ``model.decoder.additive``; ``blk<radius µm>f<fraction>``: spatial
    blocks; ``lrm``: the long-range attention mask (``long_range_attention``, off by default). ``lrm`` was added on
    2026-10-14: checkpoints trained with the mask on before then carry no tag. ``sch-<schedule>``,
    ``lw<local>-<global>`` and ``frz-<components>``: a staged schedule, loss weights other than
    the default 0.5 / 0.5, and frozen components (``train._schedule``). ``gs<cell rate>-<selectors>``:
    a hidden gene set (``dataset.mask_genes``). ``ch<fraction>c<cell rate>``: channel masking, a random share of the
    genes (``dataset.mask_channel_fraction``) in that share of the cells, ``-p<selectors>`` added when the channels come from
    a pool (``dataset.mask_channel_genes``). ``sets<k>of<n>c<cell rate>-<digest>``: a pool of ``n`` gene sets
    (``dataset.mask_gene_sets``), ``k`` drawn at a time, the digest a hash of the sets' selectors. ``rmk-<kind>``: re-masking before the transformer
    (``model.global_component.parameters.remask``). ``ctx-<kind>``: a context-only global embedding
    (``model.global_component.parameters.context_only``). ``rgm<depth um>-<local>-<global>``: each head's loss on
    the hidden cells of its regime (``optim.regime_loss``). ``aux-<name><weight>-...``: every auxiliary loss with a
    non-zero weight (``optim.aux_loss_weights``: the orthogonality and the contrastive terms), added 2026-10-23 --
    checkpoints trained with one before then carry no tag. A term's inner settings (``optim.orthogonality``,
    ``optim.contrastive``) are not in it.
    """
    # Here, not at module level: interscale.train imports this module.
    from interscale.train._schedule import resolve_loss_weights, schedule_tag

    parts = []
    if cfg.optim.get("center_targets", False):
        parts.append("ctr")
    if cfg.model.decoder.get("global_residual", False):
        parts.append("res")
    if cfg.model.decoder.get("additive", False):
        parts.append("add")
    params = cfg.model.global_component.get("parameters", None)
    if params is not None and params.get("long_range_attention", False):
        parts.append("lrm")
    if params is not None and params.get("remask", "none") != "none":
        parts.append(f"rmk-{params.remask}")
    if params is not None and params.get("context_only", "none") != "none":
        parts.append(f"ctx-{params.context_only}")
    regime = cfg.optim.get("regime_loss", None)
    if regime is not None and regime.get("kind", "none") != "none":
        parts.append(f"rgm{float(regime.depth_um):g}-{regime.local_cells}-{regime.global_cells}")
    schedule = schedule_tag(cfg)
    if schedule:
        parts.append(schedule)
    weights = resolve_loss_weights(cfg.optim.get("loss_weights", None))
    if weights is not None:
        parts.append(f"lw{weights['local']:g}-{weights['global']:g}")
    if list(cfg.optim.get("frozen", [])):
        parts.append("frz-" + "-".join(sorted(cfg.optim.frozen)))
    if cfg.dataset.get("mask_block_fraction", 0.0) > 0:
        parts.append(f"blk{cfg.dataset.mask_block_radius_um:g}f{cfg.dataset.mask_block_fraction:g}")
    gene_set = list(cfg.dataset.get("mask_genes", []) or [])
    if gene_set:
        # Letters, digits and dashes only: "role=response" -> "role-response".
        names = "-".join(re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-") for s in gene_set)
        parts.append(f"gs{float(cfg.dataset.mask_percentage):g}-{names}")
    channels = float(cfg.dataset.get("mask_channel_fraction", 0.0))
    if channels:
        tag = f"ch{channels:g}c{float(cfg.dataset.mask_percentage):g}"
        pool = list(cfg.dataset.get("mask_channel_genes", []) or [])
        if pool:
            tag += "-p" + "-".join(re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-") for s in pool)
        parts.append(tag)
    sets = [[str(g) for g in one] for one in (cfg.dataset.get("mask_gene_sets", []) or [])]
    if sets:
        import hashlib

        digest = hashlib.sha1(repr(sets).encode()).hexdigest()[:8]
        per_draw = int(cfg.dataset.get("mask_sets_per_draw", 1))
        parts.append(f"sets{per_draw}of{len(sets)}c{float(cfg.dataset.mask_percentage):g}-{digest}")
    weights = {name: float(w) for name, w in dict(cfg.optim.get("aux_loss_weights", {})).items() if float(w) != 0.0}
    if weights:
        parts.append("aux-" + "-".join(f"{name}{w:g}" for name, w in sorted(weights.items())))
    return f"var-{'+'.join(parts)}_" if parts else ""


def get_model_filename_prefix(cfg, local_component: bool, global_component: bool):
    """Generate the filename prefix for saving model files.

    Parameters
    ----------
    cfg : CN
        Configuration object containing dataset and model information

    Returns
    -------
    str
        Filename prefix in format: <dataset_name>_<prediction_task[:4]>_<prediction_level>_<local_component_name>_<global_component_name>_
    """
    file_name_prefix = (
        f"{cfg.dataset.name}_{cfg.dataset.prediction_task[:4]}_{cfg.dataset.prediction_level}_{cfg.optim.seed}_"
    )

    # Valid combinations: local only, global only, or both
    if local_component and not global_component:
        # LocalModel: only local component
        if cfg.model.local_component.name:
            file_name_prefix = file_name_prefix + f"{cfg.model.local_component.name}_"
        file_name_prefix = file_name_prefix + _decoder_tag(cfg) + _training_tag(cfg)
    elif global_component and not local_component:
        # GlobalModel: only global component
        if cfg.model.global_component.name:
            file_name_prefix = file_name_prefix + f"{cfg.model.global_component.name}_"
        file_name_prefix = (
            file_name_prefix
            + _embedding_tag(cfg)
            + _positional_encoding_tag(cfg)
            + _attention_tag(cfg)
            + _hierarchy_tag(cfg)
            + _pooling_tag(cfg)
            + _decoder_tag(cfg)
            + _training_tag(cfg)
        )
    elif local_component and global_component:
        # CombinedModel: both components
        if cfg.model.local_component.name:
            file_name_prefix = file_name_prefix + f"{cfg.model.local_component.name}_"
        if cfg.model.global_component.name:
            file_name_prefix = file_name_prefix + f"{cfg.model.global_component.name}_"
        file_name_prefix = (
            file_name_prefix
            + _positional_encoding_tag(cfg)
            + _attention_tag(cfg)
            + _hierarchy_tag(cfg)
            + _pooling_tag(cfg)
            + _decoder_tag(cfg)
            + _training_tag(cfg)
        )
        if cfg.model.decoder.dual_decoder:
            file_name_prefix = "dual_" + file_name_prefix
    else:
        # Invalid: neither local nor global component
        raise ValueError(
            f"Invalid combination of local and global components: {local_component} and {global_component}"
        )

    return file_name_prefix


def remap_state_dict_keys(state_dict):
    """
    Remap state dictionary keys to handle naming convention changes.

    This function handles the transition from interscale key names to graph_transformer_long_range_niches key names:
    - local_layers.* -> local_module.layers.*
    - global_* -> global_module.*

    Parameters
    ----------
    - state_dict: The state dictionary from the checkpoint

    Returns
    -------
    - new_state_dict: State dictionary with remapped keys
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle local module keys (InterScale: local_layers.* -> graph_transformer: local_module.layers.*)
        if key.startswith("local_layers."):
            new_key = key.replace("local_layers.", "local_module.layers.")

        # Handle global module keys (InterScale: global_* -> graph_transformer: global_module.*)
        elif key.startswith("global_"):
            new_key = key.replace("global_", "global_module.")

        new_state_dict[new_key] = value

    return new_state_dict


def detect_and_remap_state_dict_keys(state_dict):
    """
    Automatically detect the source of the state dict and apply appropriate remapping.

    This function detects whether the state dict is from interscale or graph_transformer_long_range_niches
    and applies the appropriate key remapping.

    Parameters
    ----------
    - state_dict: The state dictionary from the checkpoint

    Returns
    -------
    - new_state_dict: State dictionary with remapped keys
    - source_detected: String indicating the detected source ('InterScale' or 'graph_transformer')
    """
    # Check if this is an InterScale checkpoint (has local_layers or global_ keys)
    has_interscale_keys = any(key.startswith("local_layers.") or key.startswith("global_") for key in state_dict.keys())

    # Check if this is a graph_transformer checkpoint (has local_module or global_module keys)
    has_graph_transformer_keys = any(
        key.startswith("local_module.") or key.startswith("global_module.") for key in state_dict.keys()
    )

    if has_interscale_keys and not has_graph_transformer_keys:
        # This is an InterScale checkpoint, remap to graph_transformer format
        new_state_dict = remap_state_dict_keys(state_dict)
        source_detected = "InterScale"
        print("Detected InterScale checkpoint format. Remapping keys to graph_transformer format.")
    elif has_graph_transformer_keys and not has_interscale_keys:
        # This is already a graph_transformer checkpoint, no remapping needed
        new_state_dict = state_dict
        source_detected = "graph_transformer"
        print("Detected graph_transformer checkpoint format. No remapping needed.")
    else:
        # Mixed or unclear format, try remapping anyway
        new_state_dict = remap_state_dict_keys(state_dict)
        source_detected = "unknown"
        print("Unclear checkpoint format. Attempting remapping anyway.")

    return new_state_dict, source_detected
