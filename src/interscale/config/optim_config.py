from yacs.config import CfgNode as CN


def get_optim_cfg(cfg):
    """Defines model training optimization parameters:

    lr: float = Learning rate
    wd: float = Weight decay
    warm_up: int = Warm up epochs
    losss: str =
    seed: int
    """
    cfg.optim = CN()

    cfg.optim.accelerator = "auto"  # can also be "gpu" or "cpu"
    cfg.optim.lr = 0.001
    cfg.optim.lr_scheduler = "CosineWarmupScheduler"  # "ReduceLROnPlateau" or "CosineWarmupScheduler"
    cfg.optim.lr_warmup = 20
    cfg.optim.lr_max_epochs = 100
    cfg.optim.wd = 1e-4
    cfg.optim.loss = "GaussianNLL"  # classification: [CrossEntropy, WeightedCE], regression: [MSELoss, GaussianNLL, SmoothL1, BalancedPearsonCorrelationLoss, SCELoss]
    cfg.optim.seed = 40
    cfg.optim.cross_corr = "cell"  # Currently cell is the only one that really works
    cfg.optim.n_epochs = 100
    cfg.optim.early_stopping = True
    cfg.optim.patience = 5  # EarlyStopping patience in epochs
    cfg.optim.min_delta = 0.0  # EarlyStopping min_delta
    # Floor on training length. MUST stay above lr_warmup: CosineWarmupScheduler ramps the LR
    # linearly over lr_warmup epochs, so a run that stops inside the ramp has only ever seen a
    # fraction of cfg.optim.lr and is measured at (near) initialisation. The default is 2x
    # lr_warmup, matching chen22. `_validate_optim` enforces the invariant at config-load time
    # rather than leaving it to each dataset to remember.
    cfg.optim.min_epochs = 40
    # Metric driving EarlyStopping / ModelCheckpoint / the LR scheduler.
    # "auto" -> val_f1_macro for classification, val_loss for regression.
    cfg.optim.monitor = "auto"

    # Auxiliary loss terms added beside `optim.loss`, as name -> weight. A weight of 0.0 means the
    # term is never constructed, so every existing config keeps behaving exactly as before. Flat
    # float keys rather than a list of nested nodes, so a sweep can set
    # `optim.aux_loss_weights.<name>` through merge_from_list with no special handling. See
    # `interscale.train.aux_losses` and `.claude/contrastive_plan.md`.
    cfg.optim.aux_loss_weights = CN()
    # VICReg (variance-invariance-covariance). 0.0 leaves it unbuilt. See
    # `optim.contrastive.vicreg_*` for the coefficients within it, and `optim.loss: none` for
    # running it INSTEAD of a reconstruction criterion rather than beside one.
    cfg.optim.aux_loss_weights.vicreg = 0.0

    # Shared settings for the contrastive terms. Inert until one of them carries a weight.
    cfg.optim.contrastive = CN()
    # NT-Xent / InfoNCE temperature. Deliberately not 0.1: the softmax denominator already weights
    # negatives by similarity, and here the most similar cells are the ones most likely to share
    # the anchor's interaction program, so the weighting wants flattening rather than sharpening.
    cfg.optim.contrastive.temperature = 0.5
    # Which tokens may serve as negatives for an anchor. "within_slide" keeps batch effect from
    # becoming the cheapest way to tell two cells apart.
    cfg.optim.contrastive.negatives = "within_slide"
    # Hops of the anchor's own spatial neighbourhood removed from the negative pool; they are
    # near-certain false negatives. Match the local component's num_layers.
    cfg.optim.contrastive.exclude_khop = 2
    # Hidden/output widths of the heads the contrastive terms read through. Separate heads because
    # the loss families disagree about normalisation: InfoNCE wants L2-normalised outputs, VICReg
    # explicitly does not.
    cfg.optim.contrastive.projector_dims = [64]
    cfg.optim.contrastive.expander_dims = [256]
    # Which embedding the contrastive/VICReg terms read: "global" for the transformer's per-cell
    # tokens, "local" for the graph component's, "auto" for whichever the model has. "auto" is
    # what lets one config block serve LocalModel, GlobalModel and CombinedModel unchanged.
    cfg.optim.contrastive.embedding = "auto"

    # VICReg coefficients, named as in the paper: lambda weights invariance, mu variance, nu
    # covariance. These live INSIDE the term because they are part of VICReg's definition;
    # `optim.aux_loss_weights.vicreg` is the outer scale against the reconstruction loss.
    #
    # lambda > 0 is what makes the term need two views, so it is also the switch between "a
    # collapse regulariser beside reconstruction" and "a self-supervised objective in its own
    # right". Paper defaults are 25 / 25 / 1.
    cfg.optim.contrastive.vicreg_lambda = 25.0
    cfg.optim.contrastive.vicreg_mu = 25.0
    cfg.optim.contrastive.vicreg_nu = 1.0
    # Grouping for the variance hinge: "auto" uses the attached `slide` if there is one and the
    # graph index otherwise; "none" pools the whole batch. Leave it on "auto" -- a batch-level
    # hinge can be satisfied entirely by between-slide variance, i.e. by the batch effect, while
    # every cell inside a slide collapses to one point.
    cfg.optim.contrastive.vicreg_group = "auto"
    return cfg
