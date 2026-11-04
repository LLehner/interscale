from yacs.config import CfgNode as CN


def get_model_cfg(cfg):
    """Defines model training optimization parameters:

    model_type: str = select one from [gnn, transformer, gnn-transformer]
    """
    cfg.model = CN()
    cfg.model.n_embed = 16

    cfg.model.local_component = CN()
    cfg.model.local_component.name = None
    # cfg.model.local_component.load = None # not needed can be removed

    cfg.model.global_component = CN()
    cfg.model.global_component.name = None
    # cfg.model.global_component.load = None # not needed can be removed

    cfg.model.save = None
    cfg.model.loss = None
    cfg.model.decoder = CN()
    cfg.model.decoder.type = "linear"  # [linear, nonlinear, linear-lse]
    cfg.model.decoder.hidden_dims = [256, 128]
    cfg.model.decoder.dropout_decoder = 0.1
    cfg.model.decoder.dual_decoder = (
        False  # [True, False] If True, use dual decoder for combined module. Both local and global decoders are used.
    )
    # With the dual decoder: the global head predicts what the local head misses. Its prediction is
    # the local prediction (no gradient back into it) plus the global decoder's output, so the global
    # component is rewarded only for the part of each spot the local component cannot reconstruct --
    # under plain dual decoding copying the local head is already optimal for it. The reported
    # global prediction (metrics, get_model_output, gene ranks) is that sum.
    cfg.model.decoder.global_residual = False
    # With the dual decoder: one prediction from two parts (`.claude/decoder_plan.md`, D3.1), y = W_l h_local + W_g
    # h_global + b. The local half of the stacked output is the local part alone, the global half is the sum -- the
    # prediction the model reports, as under `global_residual`; neither part is asked to predict the target alone, so
    # set `optim.loss_weights` to {local: 0, global: 1} (or a small local weight). No gradient is cut (unlike
    # `global_residual`), and a part needs a decoder of its own: `dual_decoder: true`. Not with `global_residual`.
    cfg.model.decoder.additive = False
    # GlobalModel only: the cell's own visible genes reach the prediction around the transformer.
    # y = W_c h_global + W_o e_own + b, where e_own is the cell's token before the transformer (the
    # learned input layer's output, or the precomputed / PCA embedding) from the masked input. W_o starts
    # at zero, so the model starts as without it. Gives each cell's own-versus-context split by
    # construction (`.claude/decoder_plan.md`, D3.1 without a GCN). Pooled tokens: the skip stays per cell.
    cfg.model.decoder.own_cell_skip = False
    # What the skip reads: "embedding" (the cell's token, so W_o is limited to rank n_embed) or "raw" (its visible
    # genes themselves, a full-rank linear read: W_o is [n_output, n_input]; a hidden entry is -1 as in the model's input).
    cfg.model.decoder.own_cell_skip_input = "embedding"
    # LocalModel and GlobalModel: "visible_mean" adds a part `section` to the prediction, each gene's mean over the visible
    # entries of the cell's graph (a section) times a learned weight per gene starting at 1, so a section's own offset (donor,
    # condition, batch) need not be carried by the embedding (roadmap Tier 1; notes.md 2026-10-31: on CosMx 80-89% of the
    # centred cosine was the section's offset). "none": off. Reconstruction at node level only.
    # "label_mean" (GlobalModel): the part is the training mean of the cell's label instead (`label_offset_key`, an obs column
    # that reaches the graphs through dataset.extra_obs_keys; cell types, or clusters that imitate them), fixed, and the loss
    # reads prediction and target minus it: the decoder predicts only what the label does not explain.
    cfg.model.decoder.section_offset = "none"
    cfg.model.decoder.label_offset_key = ""

    return cfg
