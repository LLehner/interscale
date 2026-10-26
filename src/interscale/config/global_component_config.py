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
        # The tokens of a GlobalModel (no local component; a CombinedModel ignores this): None for
        # none, PCA (fitted once and frozen), NMF, Precomputed (obsm[latent_obsm_key]) or Linear (a
        # learned layer on the masked expression, trained with the transformer; .claude/global_plan.md, A1).
        cfg.model.global_component.parameters.type_gex_embedding = None
        # With Precomputed: what a hidden cell's embedding is replaced by, since it was made from
        # the expression the model is scored on. learned: one trained vector for every hidden cell;
        # minus_one: every entry -1 (MASK_VALUE), as masked expression. Gene masking is refused
        # with Precomputed (an embedding cannot hide single genes).
        cfg.model.global_component.parameters.precomputed_mask_fill = "learned"
        # With PCA: which data the basis is fitted on. `first_batch` (the model from before this key):
        # the first batch the module sees -- in practice the first, masked validation slide, because
        # Lightning's sanity check runs before training -- which misses slide- and condition-level
        # structure of the training slides. `train`: the masked inputs of every training slide, once,
        # before training (deterministic solver). Prefer `Precomputed` with an embedding fitted on the
        # training slides outside InterScale where the data allow it.
        cfg.model.global_component.parameters.pca_fit = "first_batch"

        # Positional encodings, see `.claude/PE_plan.md`. An empty `node` list builds nothing and
        # is the model from before PEs existed. Node encodings are summed into each cell's token;
        # the list is swept through a sweep yaml's `arms:` block, never as a raw wandb list.
        pe = cfg.model.global_component.parameters.pe = CN()
        pe.node = []  # any of: naive, sinusoidal, lap, rw
        # Attention biases, added to the logit of every pair of tokens: any of: distance, spectral.
        pe.bias = []  # any of: distance, spectral, fourier
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
        # * `scales` -- a distance prior with a length scale per head (`.claude/roadmap.md`, T1): head h
        #   gets the window -strength * phi(d / l_h), `phi` u^2 / 2 for `shape` gaussian or u for
        #   exponential, the scales l_h spaced geometrically over [scale_min_um, scale_max_um] and dealt out
        #   across the heads, so some heads attend locally and some globally from step 0 and each head's
        #   scale is known. Not zero at step 0 (a prior). Scales and strength are learned unless `frozen`.
        #   With the long-range mask on, put `scale_min_um` about at the local reach (a head far below it
        #   attends to the nearest visible key). `num_kernels` and `max_dist` are ignored.
        pe.distance = CN()
        pe.distance.kind = "profile"
        pe.distance.num_kernels = 16
        pe.distance.max_dist = 0.0
        pe.distance.scale_min_um = 20.0
        pe.distance.scale_max_um = 2000.0
        pe.distance.shape = "gaussian"
        pe.distance.strength = 1.0
        pe.distance.frozen = False
        # A learned bias per head as a function of how alike two cells' expression is: each key's
        # rank among the query's keys by Euclidean distance of their expression (the kNN view of
        # scanpy.pp.neighbors on the expression), piecewise linear in log2(1 + rank) between
        # `num_kernels` knots up to rank `max_rank`, flat beyond. Read from the masked input over
        # the genes visible in both cells; a pair with a wholly hidden cell gets one learned value
        # per head, so no hidden value reaches it. Dense attention backends only.
        pe.expression = CN()
        pe.expression.num_kernels = 16
        pe.expression.max_rank = 4096
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
        # A learned function of the offset between two cells as a sum of plane waves: frequencies
        # on a polar grid of `num_directions` directions over 180 degrees x `num_scales`
        # wavelengths spaced geometrically over [min_wavelength, max_wavelength] (um); 0 for
        # max_wavelength derives twice the largest slide diameter. `isotropic` ties the
        # coefficients across directions (symmetric, close to a function of distance -- the
        # `distance` profile's counterpart); off, every frequency has its own cosine and sine
        # coefficient per head (a directional, asymmetric prior). Unlike `distance` it is low rank
        # (2 x directions x scales dims), so the Performer can carry it. Starts at zero.
        pe.fourier = CN()
        pe.fourier.num_directions = 8
        pe.fourier.num_scales = 16
        pe.fourier.min_wavelength = 200.0
        pe.fourier.max_wavelength = 0.0
        pe.fourier.isotropic = True
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
        # Pooled tokens (`.claude/roadmap.md`, T1b): the transformer runs on patches of cells instead of single
        # cells -- a patch token is the mean of its visible cells' tokens, placed at the patch's centroid --
        # and every cell's global embedding is its patch's. S drops by 10-1000x (whole sections fit dense
        # attention), a patch is a denoised unit, and its embedding cannot carry one cell's own token. The
        # loss and the evaluation stay per cell. `kind`: `none` (the model from before pooling) or `grid`
        # (squares by position alone). Give the grid either `size_um` (its side) or `n_patches` (about that
        # many occupied squares per slide, so slides of different extent get the same S). `offset` shifts
        # the grid by a fraction of its side. `features`: any of `count` (log number of visible cells) and
        # `std` (spread of their tokens), each learned and starting at zero. Needs dataset.spatial_key; no
        # long-range mask, no Laplacian/random-walk encodings, no spectral/expression bias.
        pool = cfg.model.global_component.parameters.pooling = CN()
        pool.kind = "none"
        pool.size_um = 0.0
        pool.n_patches = 0
        pool.offset = 0.0
        pool.features = []
        # Hierarchical coarse-to-fine attention (`.claude/hierarchical_attention_plan.md`). `kind`: `none` (the transformer as
        # before) or `coarse_to_fine`: every layer pools the cells into a cluster hierarchy of each slide (built from positions
        # alone, cached per slide), attends densely among the coarsest clusters and CLS, then level by level among finer ones,
        # steered by the attention of the level above (`prior_mode`), and fuses each cell's output with its ancestors'
        # (`readout`). Dense attention backend at the default head width; no RoPE, long-range mask, context query,
        # visible-keys-only mask or pooling; biases of distance or offset only (distance, fourier).
        hier = cfg.model.global_component.parameters.hierarchy = CN()
        hier.kind = "none"
        # The coarse levels, coarsest first: clusters per slide (`levels`), or cells per cluster (`cells_per_cluster`, which
        # keeps a cluster's size across slides of different size). Exactly one is set. `levels: [n]` with n = cells makes
        # singleton clusters (with prior soft, lambda 0, dense candidates: the dense transformer exactly).
        hier.levels = []
        hier.cells_per_cluster = []
        # How every finer level splits its parents: `kdtree` (balanced splits by cell count across the longer side) or
        # `random` (a balanced random partition, the negative control).
        hier.builder = "kdtree"
        # The top level: `tree` (the builder over all cells) or `patches` (a square grid of side `patch_size_um`, or about
        # levels[0] patches when it is 0; patches under `min_cells` cells merge into the nearest larger one).
        hier.top_level = CN()
        hier.top_level.mode = "tree"
        hier.top_level.patch_size_um = 0.0
        hier.top_level.min_cells = 1
        # A second hierarchy with boundaries elsewhere (kdtree on coordinates turned by 45 degrees, the grid moved by half a
        # patch); the layers alternate between the two (Swin's shifted windows).
        hier.shift = False
        # Spatial neighbour clusters per cluster (by centroid): always candidates under routing, besides the cluster itself.
        hier.neighbors = 8
        # Cluster tokens from their children: `mean` over the visible cells, or `attn_pool` (a learned query per level,
        # starting as the mean). `pool_features`: any of `count` (log visible cells) and `spread` (log spatial spread),
        # each a learned vector starting at zero.
        hier.pooling = "mean"
        hier.pool_features = []
        # How the level above steers a level: `none`, `soft` (+ lambda_h log(P + eps), P = A_parent / n_children),
        # `factorized` (parent weight x within-cluster softmax), `factorized_percell` (each query weighs the clusters itself).
        hier.prior_mode = "soft"
        hier.size_correction = True
        # + mu_h conf cos(x_j - x_i, the parent's coarse arrow): a direction prior.
        hier.direction_prior = False
        # lambda and mu: their start, and `per_head` / `global` learned, or `fixed`. One per level (and head).
        hier.lambda_init = 1.0
        hier.lambda_learnable = "per_head"
        hier.mu_init = 0.5
        hier.mu_learnable = "per_head"
        # `per_layer`: each layer's prior from its own coarse attention; `once`: from the first layer's.
        hier.prior_refresh = "per_layer"
        # One set of attention weights for every level (true), or a copy per coarse level learned apart.
        hier.share_level_params = True
        # The distance bias between cluster centroids at the coarse levels: `shared` (the cells' bias), `own` (a copy per
        # level, learned apart) or `none`.
        hier.coarse_bias = "shared"
        # A cell's output fused with its ancestors': `none`, `concat` (a learned linear map of [cell, ancestors], starting
        # at the cell alone), `transformer` (the cell attends over itself and its ancestors).
        hier.readout = "none"
        # Which keys each level >= 1 attends to: `dense` (every token of the level) or `routed` (the children of each
        # parent's top-k targets (`topk`, per level or one for all), of its spatial neighbours and of `explore_n`
        # exploration parents (`uniform` or `prior_sampled`, training only), at most `cap_per_query` keys per query).
        # `chunk_pairs` bounds one block of routed queries (heads x queries x keys scores).
        hier.candidates = CN()
        hier.candidates.mode = "dense"
        hier.candidates.topk = [8]
        hier.candidates.explore_n = 0
        hier.candidates.explore_strategy = "uniform"
        hier.candidates.cap_per_query = 512
        hier.candidates.chunk_pairs = 16777216
        # Re-masking before the transformer (GraphMAE's re-mask; `.claude/roadmap.md`, T2). Cells are masked
        # once, in the input, and the local component mixes a hidden cell's visible neighbours into its
        # token, so the transformer gets a hidden cell's neighbourhood in that cell's own token and the
        # global head can copy the local one. `hidden`: every cell hidden whole (cell masking, discs) hands
        # the transformer one learned token instead, so its global embedding comes from attention to other
        # cells alone (with the long-range mask on, from beyond the local reach). `none`: the model from
        # before. Models with a local component only; not with pooling (a hidden cell already weighs 0 in
        # its patch) nor a hidden gene set (no cell is hidden whole).
        cfg.model.global_component.parameters.remask = "none"
        # A context-only global embedding (`.claude/decoder_plan.md`, D1). Plain transformer: each cell is a token
        # that attends to the others, its own token riding along in its residual stream, so the global
        # embedding of a cell carries the cell (near visible cells the global head copies the local one,
        # CKA 0.93 at the rim). `query`: the transformer's queries come from one learned token plus the
        # cell's node encoding -- never from its own token -- and read the cells' input tokens through the
        # layers' weights (cross-attention from "this place" to the tissue, XLNet's query stream). A cell's
        # global embedding is then a function of the other cells' tokens and its position alone: with the
        # long-range mask on, of cells beyond the local component's reach only. Pair it with something that
        # supplies the cell itself: the dual model's local part (`model.decoder.additive`) or the global-only
        # model's own-cell skip (`model.decoder.own_cell_skip`). Torch's dense attention only (not the other
        # backends), node-level prediction. Attention rollout and relevance (which compose attention over
        # layers as token mixing) are not defined for it.
        cfg.model.global_component.parameters.context_only = "none"
        # The attention mechanism, see `.claude/attention_plan.md`. `kind`: any registered backend
        # (`module/global_modules/attention.py`): `dense` is softmax attention over every pair, the
        # model from before backends existed. `head_dim` sets the width of one attention head apart
        # from the model's width; 0 means n_embed / n_heads. Wider heads are cheap (queries, keys and
        # values are O(S * heads * head_dim)); more heads are not (every map is [heads, S, S]).
        att = cfg.model.global_component.parameters.attention = CN()
        att.kind = "dense"
        att.head_dim = 0
        # Queries and keys from each slide's mean-centred tokens, layer-normalised; values and the
        # residual stream keep the tokens as they are. Most of every spot's expression is the profile
        # all spots share (on synth_spot 84% of a spot's squared norm), so raw tokens are nearly
        # identical, their dot products barely tell keys apart, and attention stays uniform with no
        # gradient to leave it. Centring compares spots by what differs between them. Any backend.
        att.center_qk = False
        # Only visible cells are keys (`.claude/roadmap.md`, T1 / K5): a cell hidden whole (cell masking, discs) is
        # still a query, but no cell attends to it. Its token is the mask token (or, behind a GCN, its neighbours'
        # mean) and the same for every hidden cell, so under discs the hidden cells form a block that the
        # others can attend to for nothing, a sink. CLS stays open. Dense-mask backends only (dense, entmax,
        # topk); needs cells hidden whole, so not with a hidden gene set or pooling. No effect on a clean pass.
        att.visible_keys_only = False
        # `linear`: elu+1 linear attention, O(S) like the performer but with a fixed, deterministic
        # feature map instead of random features -- not an estimate of softmax. Each head's map has
        # rank <= head_dim, so it wants wide heads (head_dim >= 32; it warns below). It carries
        # node encodings and the long-range mask, but no attention bias and no RoPE (no exponential
        # for either to act through). No settings of its own.
        # `performer`: softmax attention estimated with `num_features` positive orthogonal random
        # features per layer (FAVOR+), O(S) in memory and time; each head's map has rank
        # num_features whatever its width. The long-range mask is kept exactly; attention biases
        # only in their low-rank form. `redraw_interval` > 0 draws new features every that many
        # training steps; 0 keeps the first draw (seeded from optim.seed).
        att.performer = CN()
        att.performer.num_features = 128
        att.performer.redraw_interval = 0
        # `exphormer`: exact softmax attention along a sparse attention graph (Shirzad et al. 2023):
        # the neighbour graph's edges (`graph_edges`), a random expander of `degree` (even; built
        # from degree / 2 random permutations, fixed per graph unless `resample` redraws it every
        # training step) and CLS as the virtual node linked to every cell -- minus every pair the
        # long-range mask blocks, so under the mask the graph edges drop out by themselves.
        # `edge_type`: `bias` learns a bias per edge type and head; `key` scales each key dimension
        # per type and head, as Exphormer's edge embeddings do. Carries every bias (as per-pair
        # values) and RoPE. O(pairs) instead of S x S.
        att.exphormer = CN()
        att.exphormer.degree = 8
        att.exphormer.graph_edges = True
        att.exphormer.resample = False
        att.exphormer.edge_type = "bias"
        # `entmax`: dense attention normalised by alpha-entmax instead of softmax -- sparse maps with
        # exact zeros, softmax at alpha -> 1 and sparsemax at 2 (needs the `entmax` package).
        # `learn_alpha` learns one alpha per head as 1 + sigmoid(a), starting at `alpha` (then in
        # [1.01, 2)); otherwise `alpha` is fixed. `n_iter` bisection steps. Carries dense biases,
        # RoPE and the long-range mask.
        att.entmax = CN()
        att.entmax.alpha = 1.5
        att.entmax.learn_alpha = True
        att.entmax.n_iter = 50
        # `topk`: dense attention over each query's `k` highest-scoring open keys only (content,
        # biases and mask included), softmax among them: a hard budget of k keys per query.
        att.topk = CN()
        att.topk.k = 32
        # `fused`: dense softmax attention, exactly -- the same masks (a cell never attends to itself without the
        # long-range mask) and biases -- computed by PyTorch's fused kernel (SDPA) a block of queries at a time, so
        # no S x S tensor is formed once a sequence takes more than one block: whole sections of ~10^5 cells.
        # `chunk_pairs` bounds a block: chunk_pairs // (graphs x tokens) queries, each against every key; a
        # block's additive mask and biases hold heads x chunk_pairs floats (2^24: ~0.3 GB at 4 heads, a few times
        # that while its biases are computed). Carries every bias with a pair form (distance, fourier, spectral),
        # RoPE and the long-range mask; maps only when read (explicit path, S x S).
        att.fused = CN()
        att.fused.chunk_pairs = 16777216
        cfg.model.global_component.latent_obsm_key = None  # Use the obms key where precomputed embeddings are stored, only if type_gex_embedding is "Precomputed"
    return cfg
