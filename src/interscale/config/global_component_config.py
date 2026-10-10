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
        # Attention biases, added to the logit of every pair of tokens: any of: distance, spectral.
        pe.bias = []
        # The rotary encoding, which turns every query and key by its cell's position in each
        # attention layer: [] or [rope].
        pe.rotary = []
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
        #   (µm) and flat beyond. Can single out a distance (a ring, a threshold). `max_dist` 0
        #   derives it when the model is built: the largest slide diameter (farthest pair of cells
        #   within one `sample_key` group) over all slides -- ~1400 for synth_data_0's
        #   1000 x 1000 slides, ~7000 for synth_spot's 5 mm lattices. A positive value is used as is.
        #   The checkpoint stores the range the model was trained with.
        # * `linear` -- one slope per head on the distance in mm (ALiBi): attention only falls or
        #   rises with distance. `num_kernels` and `max_dist` are ignored.
        pe.distance = CN()
        pe.distance.kind = "profile"
        pe.distance.num_kernels = 16
        pe.distance.max_dist = 0.0
        # A learned filter of the neighbour graph's spectrum per head: the bias between two cells is
        # sum_i h_h(lambda_i) * N * v_i[j] * v_i[l] over the k lowest non-trivial eigenpairs of the
        # normalised Laplacian LapPE uses -- a learned diffusion kernel, so distances run along the
        # tissue (around holes) rather than straight across as for `distance`. h_h is piecewise
        # linear in log(lambda) between `num_knots` knots over [min_eigval, max_eigval], flat beyond,
        # and starts at zero. Sign- and basis-invariant (a tied eigenspace cut by k is dropped whole).
        # No coordinates needed. `k` sets the finest scale: on a slide of side L the shortest
        # wavelength is about L * sqrt(pi / k), 1.6 mm on synth_spot at k = 32. Eigenvalues measured:
        # ~1e-3 to 4e-2 on synth_spot (k = 32), ~4e-5 to 1.5e-3 on a 50k-cell kNN slide; the run logs
        # the range it got and warns when it leaves the knots.
        pe.spectral = CN()
        pe.spectral.k = 32
        pe.spectral.num_knots = 32
        pe.spectral.min_eigval = 1e-5
        pe.spectral.max_eigval = 2.0  # the largest eigenvalue a normalised Laplacian can have
        # Rotary position embedding in 2D (RoPE; Su et al. 2021, Heo et al. 2024): pairs of each
        # head's query and key dims are rotated by angles linear in the cell's coordinates, so the
        # logit of two cells depends on their offset, weighted by what the two cells express.
        # Needs n_embed / n_heads to be a multiple of 4. `kind`:
        # * `axial` -- fixed frequencies; half of each head's pairs along x, half along y.
        # * `mixed` -- learnable 2D frequencies, starting from that frame turned by 90/n_heads
        #   degrees more for each head, so the heads' axes cover the directions evenly.
        # Wavelengths (µm) are spaced geometrically over [min_wavelength, max_wavelength] and dealt
        # out across heads (each head its own scales; at the default n_embed 16 and 4 heads that is
        # one per axis per head). Below the distance between two cells a wave carries nothing for
        # them, so `min_wavelength` is about the shortest distance attended over (the long-range
        # mask's radius). `max_wavelength` 0 derives twice the largest slide diameter when the model
        # is built, so the longest wave never wraps around within a slide.
        pe.rope = CN()
        pe.rope.kind = "axial"
        pe.rope.min_wavelength = 50.0
        pe.rope.max_wavelength = 0.0
        cfg.model.global_component.latent_obsm_key = None  # Use the obms key where precomputed embeddings are stored, only if type_gex_embedding is "Precomputed"
    return cfg
