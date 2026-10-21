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
    # GlobalModel only: the cell's own visible genes reach the prediction around the transformer.
    # y = W_c h_global + W_o e_own + b, where e_own is the cell's token before the transformer (the
    # learned input layer's output, or the precomputed / PCA embedding) from the masked input. W_o starts
    # at zero, so the model starts as without it. Gives each cell's own-versus-context split by
    # construction (`.claude/decoder_plan.md`, D3.1 without a GCN). Pooled tokens: the skip stays per cell.
    cfg.model.decoder.own_cell_skip = False

    return cfg
