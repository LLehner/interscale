from yacs.config import CfgNode as CN


def get_global_component_cfg(cfg, global_component_name):
    """
    Defines global component configuration.
    """
    cfg.model.global_component.parameters = CN()

    if global_component_name == "self-attn-transformer":
        cfg.model.global_component.parameters.n_heads = 4
        cfg.model.global_component.parameters.dim_feedforward = 256
        cfg.model.global_component.parameters.dropout_global = 0.1
        cfg.model.global_component.parameters.activation_func = "relu"
        cfg.model.global_component.parameters.num_layers = 2
        cfg.model.global_component.parameters.max_seq_len = (
            2000  # optionally adjust to maximum number of cells, ideally shouldnt be larger than 4000
        )
        # If True, blocks attention inside the local component's receptive field. There is no
        # radius to set: the mask always covers exactly what the local module mixed (its
        # `receptive_field_hops`), so the two components stay disjoint by construction.
        cfg.model.global_component.parameters.long_range_attention = False
        cfg.model.global_component.parameters.type_gex_embedding = None

        # Positional encodings, see `.claude/PE_plan.md`. An empty `node` list builds nothing and
        # is the model from before PEs existed. Node encodings are summed into each cell's token;
        # the list is swept through a sweep yaml's `arms:` block, never as a raw wandb list.
        pe = cfg.model.global_component.parameters.pe = CN()
        pe.node = []  # any of: naive, sinusoidal, lap, rw
        # Attention biases, added to the logit of every pair of tokens: any of: distance.
        pe.bias = []
        # Coordinate encodings read `data.pos` (needs dataset.spatial_key), in µm
        # (dataset.spatial_unit_um), with each graph's centroid subtracted: absolute slide offsets
        # are scanner artefacts. `rotate_train` rotates each graph by a random angle in training,
        # since tissue has no canonical orientation -- off by default, because some tissues have a
        # meaningful axis.
        pe.center_coords = True
        pe.rotate_train = False
        # An MLP of the coordinates in units of `length_scale` (µm).
        pe.naive = CN()
        pe.naive.hidden_dim = 32
        pe.naive.length_scale = 100.0
        # sin/cos at dim/4 geometric wavelengths per axis, from about a cell diameter to about a
        # graph's extent (µm). `tl.get_average_local_and_global_size` reports both for a dataset.
        pe.sinusoidal = CN()
        pe.sinusoidal.dim = 32
        pe.sinusoidal.min_wavelength = 10.0
        pe.sinusoidal.max_wavelength = 1000.0
        # The k lowest non-trivial eigenvectors of the symmetric normalised Laplacian of the same
        # neighbour graph the local component uses (`tl.laplacian_pe`), precomputed once per graph.
        # An eigenvector has no sign, so `sign_flip` flips each one per graph in training.
        pe.lap = CN()
        pe.lap.k = 8
        pe.lap.sign_flip = True
        # Return probabilities of a random walk after 1..steps steps on the same graph
        # (`tl.random_walk_pe`), precomputed once per graph. They encode local structure -- degree,
        # triangles, density -- not position. Check their spread across cells before reading a null
        # result: it vanishes only on a truly regular graph (a symmetrised kNN graph is not one).
        pe.rw = CN()
        pe.rw.steps = 16
        # A learned bias per head as a function of the distance between two cells. `kind`:
        # * `profile` -- any curve: piecewise linear between `num_kernels` knots over [0, max_dist]
        #   (µm) and flat beyond. Can single out a distance (a ring, a threshold). Set `max_dist` to
        #   about the largest distance that should still be told apart -- a graph's diagonal at most
        #   (~1400 for synth_data_0's 1000 x 1000 slides, ~7000 for synth_spot's 5 mm lattices).
        # * `linear` -- one slope per head on the distance in mm (ALiBi): attention only falls or
        #   rises with distance. `num_kernels` and `max_dist` are ignored.
        pe.distance = CN()
        pe.distance.kind = "profile"
        pe.distance.num_kernels = 16
        pe.distance.max_dist = 2000.0
        cfg.model.global_component.latent_obsm_key = None  # Use the obms key where precomputed embeddings are stored, only if type_gex_embedding is "Precomputed"
    return cfg
