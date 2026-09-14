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

    # Restrict every probe to the cells the model was actually asked to reconstruct.
    #
    # Leave this True. With it False the probe scores mostly cells whose own expression was
    # handed to the encoder as input, so a linear readout recovers the target by copying it
    # back out of the embedding -- measured on synth_data_0 at mask_percentage 0.3, the
    # structure-free `noise_00` control probed at R2 0.33 from BOTH embeddings after three
    # epochs, which is the identity map showing through, not a finding. Restricted to masked
    # cells the same control sits near 0, which is what a negative control is for.
    #
    # Under mask_strategy "gene" the restriction is per (cell, gene) for the regression
    # targets, and "cell had at least one gene masked" for the categorical ones.
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
    # ModelCheckpoint and a sweep's ranking metric can actually monitor; the chart is purely
    # for reading a run off the dashboard. Ignored when `wandb.use` is False.
    cfg.probe.wandb_charts = True

    return cfg
