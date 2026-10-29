from yacs.config import CfgNode as CN


def get_dataset_cfg(cfg):
    """
    prediction_task: str = [graph, node_classification, node_regression]
    prediction_obs: str = value to be predicted during training, must be in adata.obs
    subset_dict: {adata}
    sample_key: list of keys in adata.obs to split the data into PyG Data objects (e.i. sliding_window, FOV, sample etc)
    num_features: number of gene expressions (added in prepare_geome_function)
    num_features: number of classes in prediction_obs (added in prepare_geome_function)
    mask_strategy: granularity of the reconstruction corruption, "node" or "gene"
    mask_percentage: Bernoulli masking probability -- per cell under mask_strategy "node",
        per (cell, gene) entry under "gene"
    """
    cfg.dataset = CN()

    cfg.dataset.h5ad_data = ""
    cfg.dataset.name = ""
    cfg.dataset.description = ""
    cfg.dataset.prediction_task = "regression"
    cfg.dataset.prediction_obs = None
    cfg.dataset.prediction_level = "node"
    cfg.dataset.layer_key = None  # default: .X
    cfg.dataset.sample_key = []
    cfg.dataset.split_key = "split"

    cfg.dataset.batch_size = 32
    cfg.dataset.train_size = 0.7
    cfg.dataset.val_size = 0.2
    cfg.dataset.test_size = 0.1
    cfg.dataset.num_features = -1
    cfg.dataset.num_classes = -1

    # Reconstruction corruption. "node" blanks whole cells and scores all G genes of them;
    # "gene" blanks individual (cell, gene) entries in every cell and scores those entries only.
    # See interscale.tl.masking for why the two objectives behave so differently -- under "node"
    # the target cell contributes nothing about itself, so the population mean is already a
    # strong solution. "node" stays the default so every existing config is unchanged.
    cfg.dataset.mask_strategy = "node"
    # One rate, not one per strategy: mask_strategy already says what a unit is, so a second key
    # would only ever be the inert half of the pair -- and setting the wrong one silently gives a
    # run with no masking. Note the two strategies are not comparable at equal values: a per-entry
    # rate is a different quantity from a per-cell one (MAE/GraphMAE use 0.25-0.75 for features).
    cfg.dataset.mask_percentage = 0.2
    # Spatial blocks, on top of either strategy: whole cells in discs of this radius (µm, via
    # spatial_unit_um), until mask_block_fraction of each graph's cells lie in one; redrawn with the
    # rest of the mask. Both strategies leave a hidden entry's neighbours visible, so the local
    # component can fill it in; a disc wider than its reach cannot be (on a 100 µm lattice with a
    # 2-layer GCN on k-NN 8 the reach is ~283 µm, so a 500 µm disc keeps a blind core of ~13 spots).
    # Needs dataset.spatial_key, regression at node level; under "node" mask_percentage may then be
    # 0, leaving the discs as the only targets. Fraction 0 (the default) draws no discs.
    cfg.dataset.mask_block_radius_um = 0.0
    cfg.dataset.mask_block_fraction = 0.0
    # Programme masking (mask_strategy "gene" only): hide exactly these genes -- names or
    # "<var column>=<value>" selectors, e.g. ["role=response", "role=receiver"] -- in the selected
    # cells and nothing else. mask_percentage is then the share of cells (1.0: every cell), and
    # discs, if on, select cells for it too. Empty: random entries, as before.
    cfg.dataset.mask_genes = []
    # Channel masking (mask_strategy "gene" only): the label-free form of programme masking. A random share of
    # the genes -- round(fraction * G), the same set in every selected cell -- is hidden, and drawn again each time
    # the graphs' masks are redrawn (every epoch for the training graphs; validation and test graphs keep the set
    # drawn at setup). 0.3 hides 30% of the genes, 0.5 half. mask_percentage is then the share of cells (1.0: every
    # cell, and the sensible value: a lower one lets the other cells show the hidden genes, which the config warns
    # about), as for mask_genes, and discs select cells too. Not together with mask_genes; a Precomputed embedding
    # (made from all the genes) cannot be used. 0 (the default): off, and no random number is drawn.
    cfg.dataset.mask_channel_fraction = 0.0
    # The pool channel masking draws from: selectors as for mask_genes (e.g. ["is_spatial=True"], or a programme to hide a
    # random part of it); round(mask_channel_fraction * size of the pool) of its genes are hidden at each draw, every other
    # gene stays visible. Empty (the default): every gene. Needs mask_channel_fraction > 0 (.claude/masking_plan.md, M1).
    cfg.dataset.mask_channel_genes = []
    # A pool of whole gene sets (mask_strategy "gene" only; .claude/masking_plan.md, M1): a list of sets, each a list of
    # selectors as for mask_genes, e.g. [["MC7", "MC8", "MC9"], ["R1"]]. At every draw mask_sets_per_draw of them are chosen
    # at random -- the same for every selected cell of a graph -- and their union is hidden, drawn again as channels are
    # (every epoch for the training graphs, once at setup for validation and test). Between a fixed programme (mask_genes)
    # and random channels: the hidden unit is a whole set, which set rotates. Not with mask_genes or channels; cells are
    # selected by mask_percentage and discs as for them. Empty (the default): off, no random number drawn.
    cfg.dataset.mask_gene_sets = []
    cfg.dataset.mask_sets_per_draw = 1

    # Optional per-cell annotations attached to every PyG Data object when set. All default to
    # None, which attaches nothing and leaves the graphs byte-identical to before.
    #
    # These exist for the auxiliary/contrastive objectives (see `.claude/contrastive_plan.md`):
    # `slide_key` restricts a negative pool to one slide so batch effect does not become the
    # cheapest discriminator; `celltype_key` stratifies composition-matched negative sampling;
    # `spatial_key` supplies coordinates for spatial crops. `condition_key` is attached for
    # *evaluation and auditing only* -- it must never enter a training objective, or a probe on
    # condition stops meaning anything.
    #
    # An obs column arrives as a one-hot `[N, n_categories]` float, not as integer codes; use
    # `interscale.tl.label_codes` to read it. An obsm key arrives as its matrix, `[N, D]`.
    # obsm matrix holding NUMERIC probe targets (e.g. dist_to_center, hub_response). geome
    # cannot attach a numeric obs column -- it hands back a pandas Series and dies at torch.cat
    # -- so continuous covariates have to arrive as an obsm matrix. `probe.regression_obs` names
    # the obs columns and the training entrypoint stacks them into this key; the column ORDER is
    # `probe.regression_obs`, which is what lets the probe index them by name.
    cfg.dataset.probe_obsm_key = None

    cfg.dataset.slide_key = None
    # The unit of statistical independence -- usually donor, patient, or processing batch. Two
    # cells sharing a value here are not independent observations, so a split that puts one such
    # group on both sides lets any readout (the model's own val metrics, and every probe) score by
    # recognising the group rather than the biology. Setting it enables
    # `tl.check_split_independence`, which reports that rather than assuming it away.
    #
    # Deliberately not named `donor_key`: the column that carries non-independence differs by
    # dataset -- donor, patient, mouse, slide, run -- and hardcoding one of them into the API
    # would make the check unusable on the next dataset.
    cfg.dataset.group_key = None
    cfg.dataset.condition_key = None
    cfg.dataset.celltype_key = None
    cfg.dataset.spatial_key = None
    # Micrometres per unit of `obsm[spatial_key]`: 1.0 for coordinates already in µm, the pixel
    # size for pixel coordinates (0.138 for Resolve MC1). Positional encodings state their length
    # scales in µm, so this is what keeps those settings comparable across datasets.
    cfg.dataset.spatial_unit_um = 1.0

    # Any further obs columns to attach, by name. Each becomes a `Data` attribute of the same
    # name, one-hot encoded if categorical, and is covered by the same category-preservation and
    # missing-category warning as the named keys above.
    #
    # This exists so that adding a probe target, a stratifier or a grouping variable never
    # requires editing `OPTIONAL_FIELDS`. The named keys above are the ones the *code* reads by
    # name (negative sampling reads `slide`, the flow control reads `celltype`); everything else
    # a dataset happens to carry -- niche, region, tumour stage, timepoint -- belongs here.
    cfg.dataset.extra_obs_keys = []

    # Run trainer.test() after fitting. Costs a third full pass and triples the metric names in
    # the logger (every train_/val_ metric gains a test_ twin) for a number that should only be
    # looked at once, at the end of a project. Kept True so existing runs are unchanged.
    cfg.dataset.evaluate_test = True

    # Segmentation robustness parameters
    cfg.dataset.segmentation_robustness = None  # [node_fraction, overflow_fraction] or None
    # only needed for segmentation robustness experiments
    cfg.dataset.spatial_neigbors_kwargs = CN()
    cfg.dataset.spatial_neigbors_kwargs.radius = 50
    cfg.dataset.spatial_neigbors_kwargs.coord_type = "generic"
    cfg.dataset.spatial_neigbors_kwargs.library_key = ""
    cfg.dataset.spatial_neigbors_kwargs.n_neighs = 6

    return cfg
