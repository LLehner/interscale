from yacs.config import CfgNode as CN


def get_probe_cfg(cfg):
    """Online downstream probes evaluated during training.

    The reconstruction loss says how well the model copies expression back; it does not say
    *which component* learned the structure. These probes answer that during the run rather
    than after it: the same target is read out of the local embedding and out of the global
    one, so every logged number is a local-vs-global pair on one axis.

    Everything here is inert unless ``probe.use`` is True, so an existing config trains exactly
    as before. See :mod:`interscale.evaluation.online_probes`.
    """
    cfg.probe = CN()

    # Master switch. False means the callback is never constructed and no extra forward pass
    # is made, so the training loop is byte-identical to a run without this feature.
    cfg.probe.use = False

    # Probes run at the END of a validation epoch, on `epoch % every_n_epochs == 0`. Each run
    # costs one extra no-grad pass over the train split plus one over val, so on a long run
    # this is the knob that decides how much of the epoch budget goes to measurement.
    cfg.probe.every_n_epochs = 1

    # Which representations to read the targets out of. Names are the fields of
    # `ViewOutput`: "local" is the graph component's embedding, "global" the transformer's
    # per-cell token. A model without a component silently contributes no columns for it,
    # which is what makes the same probe block usable for Local/Global/CombinedModel.
    cfg.probe.embeddings = ["local", "global"]

    # Categorical targets, named by the OPTIONAL FIELD that carries them on the PyG Data
    # object -- so "celltype" here requires `dataset.celltype_key` to be set, and the probe
    # raises rather than scoring a constant if it is not. Scored with macro precision/recall.
    cfg.probe.classification_targets = []

    # Continuous targets, named by GENE (a column of `adata.var_names`). The target is the
    # uncorrupted expression in `dataset.layer_key` -- the probe reads `batch.x`, which
    # `apply_mask` leaves untouched because it corrupts a clone. Scored with MSE and R2.
    cfg.probe.regression_genes = []

    # Continuous targets that are NOT genes: numeric `adata.obs` columns such as
    # `dist_to_center` or `hub_response`. Like the categorical targets these never enter the
    # model's input, so they are scored on the clean pass. Requires `dataset.probe_obsm_key`,
    # which the training entrypoint fills from these names.
    cfg.probe.regression_obs = []

    # WHICH CELLS EACH TARGET IS SCORED ON. This is not one policy but two, because the two
    # kinds of target have opposite failure modes:
    #
    # * GENE targets are part of `batch.x`, i.e. the model's own input. Scoring them on a cell
    #   whose expression was handed to the encoder measures whether the embedding can be
    #   inverted, not whether the model learned anything -- measured on synth_data_0 at
    #   mask_percentage 0.3, the structure-free `noise_00` control probed at R2 0.33 from BOTH
    #   embeddings, and ~0.02 once restricted to masked cells. So gene targets are scored on the
    #   CORRUPTED pass, masked entries only, and `masked_cells_only` below governs that.
    #
    # * LABEL targets (`classification_targets`, `regression_obs`) are never model inputs, so
    #   there is nothing to leak and masking only deletes signal. They are scored on a CLEAN
    #   pass -- every cell, nothing blanked -- which is the ordinary linear-probing question
    #   "does this embedding encode cell identity". Masking them is why cell type scored ~0.2
    #   on synth_data_0: with the cell's own marker gene blanked, the label is only recoverable
    #   through the niche, whose Bayes-optimal macro precision is 0.19. Unmasked, a readout on
    #   raw expression reaches 0.74.
    #
    # Applies to GENE targets only. Leave it True; see above for what turning it off measures.
    cfg.probe.masked_cells_only = True

    # L2 penalty of the ridge readout. A probe is meant to measure what is linearly present,
    # not to be a good model, so this stays fixed rather than being tuned per target.
    cfg.probe.ridge_alpha = 1.0
    # lbfgs iterations for the logistic readout. 1000 converges on a 64-dim embedding; it is
    # exposed because a wider `model.n_embed` may need more.
    cfg.probe.max_iter = 1000

    # Cap on cells fed to the readout, per split, drawn with `probe.seed`. The fit is O(cells)
    # and 16 train slides of 1500 cells is already 24k rows; the cap keeps a probe epoch at a
    # few seconds on a dataset where the encoder pass is the expensive half. 0 means no cap.
    cfg.probe.max_cells = 20000
    # Seed for the subsample and for the readouts. Separate from `optim.seed` on purpose: the
    # probe must not consume draws from the training RNG stream, or enabling measurement
    # would change the model being measured.
    cfg.probe.seed = 0

    # Log a wandb custom chart per target with local and global as two lines on ONE plot.
    # The flat scalars below are logged either way -- they are what EarlyStopping,
    # ModelCheckpoint, a sweep's ranking metric and any CROSS-RUN comparison actually read;
    # the chart is a within-run convenience. Ignored when `wandb.use` is False.
    cfg.probe.wandb_charts = True
    # How often to re-render those charts, counted in probe rounds rather than epochs.
    #
    # Each refresh uploads one wandb Table per chart holding the FULL curve so far, so the cost
    # is quadratic in run length: refreshing every round over 200 epochs is ~1400 table files
    # per run and megabytes of duplicated history, which on a 5-trial sweep is mostly sync time.
    # The charts are for watching a run, not for reading a number off, so a coarse cadence
    # loses nothing. The final validation pass always publishes regardless of this, so the
    # end-of-run chart is complete whatever it is set to.
    cfg.probe.chart_every_n_epochs = 10

    return cfg
