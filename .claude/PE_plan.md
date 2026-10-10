# Staged plan: positional encodings for the global module

Adds seven positional encodings (PEs) to the transformer, selectable from config and sweepable:
naive coordinates, 2D sinusoidal, Laplacian eigenvectors (LapPE), random-walk (RWPE), a relative
Euclidean-distance attention bias, 2D rotary embeddings (RoPE, Stage 7) and a spectral
(diffusion-kernel) attention bias (Stage 8). Sources: SpaFormer (Wen et al., arXiv 2302.03038) for
the spatial variants, GraphGPS (Rampášek et al., NeurIPS 2022) for the graph ones and for the
node-feature vs. attention-bias split, RoFormer (Su et al., arXiv 2104.09864) and RoPE for ViTs
(Heo et al., ECCV 2024) for RoPE, SignNet/BasisNet (Lim et al., ICLR 2023) for the spectral bias.
Read [`background.md`](background.md) first.

## Status

Last updated 2026-10-09. Same rules as [`contrastive_plan.md`](contrastive_plan.md): update this
table in the same commit as the work, and a stage is `done` only when something external verified
it. "Implemented, unverified" is a real state.

| stage | state | verified by | date |
|---|---|---|---|
| P — two pre-existing fixes (mask reach from the local module, objective validator) | **done** | 331 tests pass, incl. built-model checks (GCN/GIN 1–3 layers, single and dual decoder, SCVI → 0, `GlobalModel` → 0 + warning) and GIN's reach against its gradient-measured receptive field; equivalence harness IDENTICAL (4 cases / 12 epochs — none uses long-range masking); a real `CombinedModel` run with long-range on builds a 2-hop mask and trains finite (single and dual decoder) | 2026-10-03 |
| 0 — plumbing, no behaviour change (node half; see 'How Stages 0–2 differed') | **done** | 352 tests pass; equivalence harness IDENTICAL across 5 cases / 15 epochs incl. the long-range one; PE-on and PE-off at one seed share every other initial weight and the first forward (test, and mutation-checked: without the RNG fork it fails) | 2026-10-03 |
| 1 — naive PE | **implemented, unverified** — translation-invariant by centring (test, mutation-checked), unit conversion, rotation on its own generator; trains finite and moves off the PE-free run through `GlobalModel`, `CombinedModel`, dual decoder and long-range; `get_model_output` runs. **No real-data run yet** | 2026-10-03 |
| 2 — 2D sinusoidal | **implemented, unverified** — closed form and distance decay pinned; same end-to-end checks as Stage 1. **No real-data run yet** | 2026-10-03 |
| 3 — LapPE | **implemented, unverified** — equals PyG's `AddLaplacianEigenvectorPE` up to sign on connected graphs (both PyG solver paths); dense and sparse solvers agree exactly; one trivial eigenvector dropped per component, zero-padding, canonical signs, orthonormal columns; flips per graph, training only, off the global RNG (each mutation-checked); harness IDENTICAL with PEs off; trains finite through `GlobalModel`, `CombinedModel`, dual decoder and long-range; `get_model_output` runs; precompute 2.4 s for a 50k-cell 6-NN graph. **No real-data run yet** | 2026-10-03 |
| 4 — RWPE | **implemented, unverified** — equals the definition (full dense powers) at odd and even steps, and PyG's `AddRandomWalkPE`; isolated cells → zeros; BatchNorm keeps small late-step spreads instead of squashing them (`eps` 1e-8, see notes); each mutation-checked; harness IDENTICAL with PEs off; trains finite through `GlobalModel`, `CombinedModel`, dual decoder and long-range, alone and with lap + sinusoidal; `get_model_output` runs; precompute 9.4 s for a 50k-cell, 14-neighbour graph at 16 steps. **No real-data run yet** | 2026-10-04 |
| 5 — distance bias | **implemented, unverified** — `pad_like` reproduces `pad_batch`'s layout incl. subsampling and kept masked nodes; bias laid out like the mask (left pad, CLS last, graph-major), zero at CLS and padding, `-inf` exactly where the mask blocks; zero table changes nothing and raises no mask-type warning; finite with rows summing to 1 on the complete-graph NaN test at hops 1–3; translation and rotation invariant; each of five layout/merge mutations caught; harness IDENTICAL with PEs off (baseline from a worktree of `e0afa60`); trains finite through `GlobalModel`, `CombinedModel`, dual decoder and long-range, alone and with node PEs, and the table moves; `get_model_output` runs. **No real-data run yet** | 2026-10-06 |
| 6 — PE probe control + ablation sweep | **implemented, unverified** — the `pe` column holds each cell's own encoding when `pad_batch` subsamples (checked against the cell's coordinates, mutation-checked: an ungathered column fails), is the eval encoding and leaves the PE generator alone; absent without a node PE; on the same chart as local/global; default in `probe.embeddings`. Found and fixed on the way: the probe moved the global torch RNG (every `DataLoader` iterator seeds itself off it) and Python's (`pad_batch` subsampling), so probe-on trained a different model — now restored (test, mutation-checked). `max_dist: 0` derives the largest slide diameter; stored in the checkpoint; LapPE size warning. 431 tests pass; harness IDENTICAL with PEs off (baseline from a worktree of `1faf992`); a `CombinedModel` with sinusoidal + lap + distance and the probe on trains through `model.train` and logs `probe/*_pe`. **The sweep (cluster) is not run yet** | 2026-10-08 |
| 7 — RoPE (2D rotary) | **implemented, unverified** — logits depend on the cell-to-cell offset only: translating mm-scale coordinates leaves them unchanged and moving one cell changes only its row and column, both checked through the layer's attention itself (rotating only the queries fails it); the rotation follows each token through `pad_batch`'s subsampling, CLS last and padding unrotated, rows graph-major; all-zero coordinates reproduce the RoPE-free model exactly (output and attention maps); it reuses the node encodings' coordinates, so `rotate_train` turns both by one angle, and draws nothing from the global RNG; `forward` refuses a missing or foreign rotation and clears it from the layers even when one raises; finite, rows summing to 1, on the complete-graph NaN test at hops 1–3 with a distance bias on too; wavelengths geometric and dealt out across heads, `mixed` frames over 90°; derived `max_wavelength` and its buffer. 12 mutations, each caught. 465 tests pass; harness IDENTICAL with PEs off (baseline from a worktree of `7fc1299`); trains finite and moves off the PE-free run through `GlobalModel`, `CombinedModel`, dual decoder and long-range, as `axial`, `mixed` and `mixed` + naive + distance; `get_model_output` runs. Found on the way: `GlobalModel` built its module from the un-derived cfg, so `pe.distance.max_dist: 0` (the default) raised at construction — fixed. **No real-data run yet** | 2026-10-09 |
| 8 — spectral bias | **implemented, unverified** — `laplacian_spectrum` equals the dense eigendecomposition (compared as projectors, dense and shift-invert paths) and zero-pads; a tied eigenspace at the cut is dropped whole, so on a lattice the kernel no longer depends on how the cells are numbered (k = 1–11); the filter is piecewise linear in log λ and flat beyond its knots; the bias equals Σᵢ h(λᵢ)·vᵢvᵢᵀ and does not change under sign flips or a rotation within a tied pair (it does within an untied one); the √N scaling makes each kept mode add 1 to the mean diagonal at h = 1, at any N; laid out like the tokens with each graph's own eigenvalues, through `pad_batch`'s subsampling, CLS and padding zero; adds to the distance bias; a zero table changes nothing; finite, rows summing to 1, on the complete-graph NaN test at hops 1–3; PyG batches the eigenvalues per graph; the run reports the eigenvalue range against the knots and warns when the kept modes miss >10% of cells. 8 mutations, each caught. 486 tests pass; harness IDENTICAL with PEs off; trains finite through `GlobalModel`, `CombinedModel`, dual decoder and long-range, alone and with lap + distance + RoPE, and the table moves and the output with it; `get_model_output` runs. Measured on the data: synth_spot slides are connected, eigenvalues 1.0e-3–3.5e-2 at k = 32, all 32 kept; synth_data_0 at radius 30 is fragmented (see Open questions). **No real-data run yet** | 2026-10-09 |
| 9 — post-hoc evaluation (synth_spot) | **implemented, not run** — unit tests only: the gene-vector probe reads planted structure (centred cosine >0.9) and nothing from noise, the background genes stay at ~0 from every representation including the masked input (nothing hidden leaks), the raw cosine sits above its mean-profile floor, the one-fit and per-gene paths agree, labels and covariates are read from hidden spots only, `draw_masks` uses its own generator, the input baselines are the masked input and its neighbour mean in probe row order, and edges renumber to a subset and score the same there. 2 mutations, each caught. 498 tests pass. The script's gene-loadings step was checked on synthetic decoder weights: the planted dims are kept, the planted genes rank top, and each programme loads on the decoder that writes it. **Never run on a trained model** (no local model runs, by the user's rule); the evaluate mode is unexercised | 2026-10-10 |

## What the code already decides

- **There is no PE today.** Tokens are local embedding → `norm_input` → `pad_batch` → CLS appended
  last → encoder (`module/global_modules/transformer_encoder.py:62-160`). Space reaches the
  transformer only through the GCN and the long-range mask.
- **`pad_batch` subsamples with Python's `random`** when a graph exceeds `max_seq_len`
  (`tl/padding.py:23,28,110`). Every per-token quantity must therefore be added *before* the
  single `pad_batch` call, or be gathered with the `index_nodes` it returns, as the long-range
  mask does. A second `pad_batch` call for positions draws a different subset and misaligns
  positions and tokens without raising.
- **Same layout as the mask**: left-padded, CLS last, heads graph-major (`tl/masking.py:384,394`).
  An attention bias must match it exactly.
- **Float masks already work.** `MultiHeadAttentionWithEdits` adds a float `attn_mask` to the
  logits (`transformer_utils.py:345,362`), so a bias needs no change to the attention code.
- **The CombinedModel already is SpaFormer's best PE.** Their "Cond PE" is a GNN over the spatial
  graph producing each token's embedding — structurally our GCN → transformer. Expect gains
  mainly in `GlobalModel`, and in `CombinedModel` only from what a k-hop GCN cannot supply:
  absolute position and distance beyond its receptive field. Expect small effects either way —
  SpaFormer's best PE gained 0.004–0.028 Pearson over none (their Fig. 4), and in GraphGPS no PE
  beat LapPE on PascalVOC-SP. Hence ≥3 seeds, compared paired by seed.
- **Coordinates are opt-in and unit-less.** `data.pos` exists only when `dataset.spatial_key` is
  set (`OPTIONAL_FIELDS`), and `obsm["spatial"]` is µm for some datasets, pixels for others
  (`tl/_preprocessing.py`).

## Found while planning (Stage P)

Both pre-existing, each its own `Fix` commit, both **before** the Stage 0 baseline is captured,
because they change numbers.

1. **The long-range mask always blocks 1 hop.** The intent is that the mask covers exactly what
   the local model already mixed. But all three models build the transformer through
   `GlobalModule.from_config`, which reads `params.get("local_mask_hops", 1)`
   (`module/base/_base_global_module.py:374`). The config key is `long_range_mask_hops`, so the
   fallback 1 always wins. The derivation (`resolve_local_mask_hops`) is only called from
   `_register_global_component`, which nothing calls (`model/global_model.py:31`), and its tests
   check the function rather than the built module. Wrong in every run since the mask landed
   (`4190b83`, 2026-09-12). Checked: a 2-layer GCN resolves to 2; the built module has 1. Ring
   NCE uses the resolver, so its inner radius (2) and the mask (1) currently disagree.

   **Decided 2026-10-03: the config only switches the mask on or off.** `long_range_mask_hops` is
   removed. The reach comes from the local module that is actually built:
   - GCN / GIN: their number of message-passing layers.
   - SCVI, Precomputed: 0. They never look at neighbours, even though SCVI's config also carries
     a `num_layers` (its encoder depth).
   - No local module (`GlobalModel`): 0, so the mask reduces to the self-only mask, with a
     warning. Today's resolver would instead read a GCN's depth from a shared config for a model
     that has no GCN.

   Ring NCE reads the same number from the module. A yaml that still sets
   `long_range_mask_hops` fails to load with yacs's unknown-key error; delete the line.
2. **`_validate_objective` never runs for a config file** — only on the defaults-only branch of
   `load_config` (`config/__init__.py:288` vs. `307-309`). It guards runs with no reconstruction
   loss (`optim.loss: none`, added with VICReg in `b3fd9f8`) and every aux weight at 0, where the
   total loss is a constant zero and nothing learns. Fold the validators into one list, so
   `_validate_pe` cannot go missing from a branch the same way.

## Config

Lives in `get_global_component_cfg`, so `LocalModel` sweeps drop it automatically. Defaults
reproduce today's model. Each stage adds its own sub-block together with its registry entry, so
no key exists before something reads it.

```yaml
dataset:
  spatial_key: spatial       # required by naive, sinusoidal, distance
  spatial_unit_um: 1.0       # NEW: µm per obsm unit (0.138 for pixel-unit Resolve data)
model:
  global_component:
    parameters:
      pe:
        node: []             # any of [naive, sinusoidal, lap, rw]; summed into the token
        bias: []             # [distance]; added to the attention logits
        rotary: []           # [rope]; rotates queries and keys in every layer (Stage 7)
        center_coords: True  # subtract each graph's centroid
        rotate_train: False  # random rotation per graph, training only
        naive:      {hidden_dim: 32, length_scale: 100.0}                   # µm
        sinusoidal: {dim: 32, min_wavelength: 10.0, max_wavelength: 1000.0} # µm
        lap:        {k: 8, sign_flip: True}
        rw:         {steps: 16}
        distance:   {kind: profile, num_kernels: 16, max_dist: 2000.0}      # µm; set per dataset
        rope:       {kind: axial, min_wavelength: 50.0, max_wavelength: 0.0} # µm; 0 = 2 x slide diameter
```

`_validate_pe` rejects unknown or duplicate names (listing the registry), a coordinate PE without
`dataset.spatial_key`, `sinusoidal.dim % 4 != 0`, and non-positive sizes. Lists rather than one
string because node PEs compose (`[lap, rw]` is valid, as in GraphGPS).

## Design

**Registry** — `module/global_modules/positional_encodings.py`:
`PE_REGISTRY[name] = PESpec(kind="node" | "bias", requires=<Data attrs>, module=<cls>, precompute=<fn | None>)`.
A sixth PE (SignNet, Cond PE, shortest-path bias) is one entry plus one config sub-block.

**Three plug points**, all in `TransformerNodeEncoderHook`:

1. *Node PEs* each project to `n_embed` and are summed onto the flat `[N, E]` embedding right
   before `pad_batch`, so subsampling, node masking and padding carry them for free. Added, not
   concatenated: `n_embed` is the width the decoders and probes share. CLS gets none.
2. *Precompute* (LapPE, RWPE) — `tl/positional.py`, called at the end of `prepare_geome_dataset`
   on each `Data`: full graph, same `edge_index` the GCN uses (the loader batches whole graphs;
   only `pad_batch` subsamples). Attaches `lap_pe [N, k]` and `rw_pe [N, steps]`.
3. *Attention bias* — after `pad_batch`, positions go through a new deterministic
   `pad_like(x, batch, index_nodes, S)` (no RNG), become `[B·H, S+1, S+1]` with a zero CLS
   row/column and zero padding, and merge in `forward(..., attn_bias=None)` as
   `attn_bias.masked_fill(blocked, -inf)` — indexing, never arithmetic.
   `common_step_local_to_global` returns a named `GlobalInput` instead of a growing tuple
   (4 call sites; the `StepOutput` pattern).

**Also in Stage 0:**

- `from_config` passes the whole `pe` node through; it is the only build path (see P1).
- PE modules are built under `torch.random.fork_rng()`, and sign flips / rotations draw from a
  dedicated generator seeded from `optim.seed`. Enabling a PE then changes neither the init of
  any existing weight nor the batch order at a given seed, so seed-paired comparisons stay paired.
- The checkpoint prefix gains `pe-<names>_` when any PE is on (unchanged otherwise). The prefix
  carries the seed but not the PE, so without this the sweep arms overwrite each other's
  checkpoints. A checkpoint/config mismatch in `pe.*` weights already raises: `BaseModel.load`
  rejects any missing or unexpected key unless `allow_partial_load`.

### How Stages 0–2 differed from this plan

- **Stage 0 was built for node encodings only.** `pad_like`, `GlobalInput`, the `attn_bias`
  argument and the precompute hook have no consumer before Stages 3 and 5, so they arrive with
  those stages rather than as untested scaffolding. The precompute hook landed with Stage 3:
  `PESpec.precompute` plus `attach_positional_inputs`, called by `prepare_geome_dataset` and both
  models' `get_model_output`; LapPE's sign flip is the encoder's `augment` hook, which the
  container calls in training only (the contrastive plan deferred its projection
  heads the same way). The registry already carries `kind` and `requires`, which is what they will
  plug into.
- **Every node encoder's output layer starts at zero**, not only the distance bias's. With the RNG
  fork that makes "PE on" and "PE off" at one seed identical until the first optimiser step.
- **No load guard was needed** — see the last bullet above.
- **Each encoder validates its own sub-block** (`PESpec.validate`), so a new encoding brings its
  checks with it; `_validate_pe` adds only the cross-cutting one (coordinates need
  `dataset.spatial_key`).
- **Node PEs are added after the first `norm_input`** (`common_step_local_to_global`), so the
  second one, in `forward`, normalises token and PE together.
- Code: `module/global_modules/positional_encodings.py` (registry, `NaivePE`, `SinusoidalPE`,
  `NodePositionalEncoding`), built in `GlobalModule.from_config`; config in
  `get_global_component_cfg`; tests in `tests/test_positional_encodings.py`.

### How Stage 5 differed from this plan

- **No `GlobalInput`, no `attn_bias` argument.** Nothing but `forward` reads the mask
  `common_step_local_to_global` returns, so the bias is merged there —
  `bias.masked_fill(boolean_mask, -inf)`, indexing as the invariants require — and the returned
  mask is a float mask when a bias is on. The four call sites are unchanged. `forward` turns the
  key-padding mask into a float one for the encoder in that case (torch otherwise warns on every
  layer about mixed mask types) and still returns the boolean one.
- **Triangular kernels instead of Gaussian RBFs.** The same "K kernels → heads" form, but each
  distance touches two knots, so the bias is two lookups into a `[heads, K]` table instead of a
  `[B, S, S, K]` expansion. The table is directly the learned profile.
- **`max_dist` is derived from the data by default** (`0`, since 2026-10-08; was a fixed 2000 µm):
  the largest slide diameter — farthest pair of cells within one `sample_key` group, over every
  slide and split, in µm (`resolve_distance_range`, called in `BaseModel.__init__` before any
  module is built). It is a buffer, so a checkpoint keeps the range its table was trained on even
  when loaded onto other slides; checkpoints from before the buffer load with the re-derived value
  (`_OPTIONAL_STATE_PREFIXES`). A module built straight from a config with `0` raises. A positive
  value is used as is.
- **Two kinds, chosen by `pe.distance.kind`** (added 2026-10-07 on the user's request). `profile`
  learns any curve of distance and needs `max_dist`; `linear` is one slope per head on the
  distance in mm — attention can only fall or rise with distance, no range to set. Both start at
  zero. The checkpoint tag says `distance-linear` for the second, so a sweep can run both.
- **`b_h(d)` is readable, not logged.** `DistanceBias.profile()` returns it; logging it during
  training waits for a consumer (Stage 6's sweep).
- Code: `DistanceBias`, `AttentionBias` and `build_attention_bias` in
  `module/global_modules/positional_encodings.py`; `tl.pad_like` in `tl/padding.py`; the merge in
  `TransformerNodeEncoderHook.common_step_local_to_global`.

### How Stage 7 (RoPE) was built

Added 2026-10-09 on the user's request; no plan section preceded it.

- **A third kind, `rotary`.** RoPE is neither summed into the token nor added to the logits: it
  rotates each head's queries and keys inside every layer. Config: `pe.rotary: [rope]` (at most
  one) and the `pe.rope` sub-block. The rotation is computed per batch in
  `common_step_local_to_global` from coordinates laid out with `pad_like`, CLS appended last at the
  origin, and applied in `multi_head_attention_forward_with_gradients` right after the heads are
  split (`transformer_utils.apply_rotary`), so the attention maps the interpretability tools read
  include it.
- **`GlobalInput` now exists.** RoPE is the first consumer that cannot be merged into the mask, so
  `common_step_local_to_global` returns the named tuple foreseen above, with a fifth field
  `rotary`, and `forward(..., rotary=)` takes it; the four call sites unpack five values. `forward`
  refuses a missing rotation when RoPE is on (and any rotation when it is off), and one shaped for
  another batch. `nn.TransformerEncoder` passes its layers nothing but the masks, so the rotation
  reaches them as `layer.rotary`, set for the one call and cleared in a `finally`.
- **One `CoordinateFrame` per model.** Units, centring, `rotate_train` and the PE generator moved out
  of `NodePositionalEncoding` into a frame the node and rotary encodings share. RoPE reuses the
  coordinates the node encodings prepared in the same pass, so both see a graph turned by one angle.
  With RoPE off, the draw order is unchanged.
- **Narrow heads.** 2D RoPE needs `n_embed / n_heads` to be a multiple of 4 (checked at config
  load). At the defaults (16 / 4) a head has 4 dims, so one wavelength per axis. The
  `heads × head_dim/4` wavelengths are therefore spaced geometrically over the range and dealt out
  across heads (head h takes every heads-th, ALiBi-like): each head gets its own scale instead of
  all heads the same one.
- **CLS sits at the origin** (identity rotation), i.e. at the centroid with `center_coords`. That is
  the only place the origin matters; every cell-to-cell logit depends on the offset alone.
- **Not zero at step 0.** A rotation has no zero, so RoPE is the one encoding that changes the first
  forward. Initial weights (`axial` has no parameters; `mixed`'s are built under the RNG fork) and
  batch order are still shared with the PE-free run at a seed.
- **`max_wavelength: 0` is derived**, like `max_dist`: twice the largest slide diameter
  (`resolve_rope_range`, called in `BaseModel.__init__`), so the longest wave turns by at most half
  a cycle within a slide. The frequencies are buffers, so a checkpoint keeps the ones it was trained
  with. `min_wavelength` is a fixed 50 µm default: between two cells much farther apart than a
  wavelength, that wave turns by an effectively random angle and carries nothing, so it should be
  about the shortest distance attended over — the long-range mask's radius.
- **`mixed` learns `magnitude × direction`** with only the unit-scale `direction` trained, so an Adam
  step moves every frequency by about the same relative amount (they span three decades in rad/µm).
- **Angles in float64**, from the float32 coordinates; the rounding left is the coordinates' own
  (~1e-3 µm at 5 mm).
- **No `pe` probe column**: RoPE gives no per-cell vector, like the bias.
- Code: `RotaryPE`, `RotaryEncoding`, `CoordinateFrame`, `build_positional_encodings` and
  `resolve_rope_range` in `positional_encodings.py`; `apply_rotary` and the `rotary` argument in
  `transformer_utils.py`; `GlobalInput` and the `forward` plumbing in `transformer_encoder.py`;
  tests in the RoPE section of `tests/test_positional_encodings.py`.

### How Stage 8 (spectral bias) was built

Added 2026-10-09 on the user's request, after weighing SignNet/BasisNet: the transformer-native,
sign- and basis-invariant way to use the spectrum, at the cost of the distance bias.

- **The bias.** `b_h(j, l) = Σᵢ h_h(λᵢ) · N · vᵢ[j] · vᵢ[l]` over the k lowest non-trivial eigenpairs
  of LapPE's Laplacian. Each head learns `h_h`, piecewise linear in log λ between 32 knots over
  `[min_eigval, max_eigval]` = 1e-5–2 (2 is the largest eigenvalue `L_sym` has), flat beyond, zero
  at the start, one table for all layers. With `h = exp(−tλ)` it is the heat kernel truncated to k
  modes, the matrix diffusion distances are built from — the BasisNet-approximable object of Lim
  et al., Prop. 4, without the IGN.
- **A fixed log range, not a derived one.** Eigenvalues span ~4e-5 (a 50k-cell kNN slide) to ~5e-2
  (synth_data_0 at k = 32); 32 knots over 5.3 decades leave 6–10 intervals inside any one dataset's
  range, with no data pass when the model is built. The run logs the range it found
  (`report_spectral_range`, in `prepare_geome_dataset`) and warns when it leaves the knots.
- **Whole eigenspaces.** `tl.laplacian_spectrum` computes k + 1 eigenpairs and drops a tied group at
  the cut (relative tie 1e-6: lattice ties come out at ~1e-13, the smallest genuine gap there is
  6e-3). On the 50 × 50 lattice eigenspaces end at 2, 3, 5, 7, 8, …, 30, 32, 34, so k = 32 cuts none.
- **Summed over modes, scaled by √N.** Each mode adds `O(h)` whatever N and k. Averaging over the k
  modes was tried first: it shrank a filter that keeps only the slowest modes — the long-range case —
  by k, about 30x slower to matter.
- **Inputs.** Two `Data` attributes: `spectral_pe` `[N, k]` (eigenvectors times √N) and
  `spectral_eigval` `[1, k]` (PyG stacks them to `[B, k]`); `PESpec.precompute` may now return a
  dict, and both names are reserved fields. `AttentionBias` prepares each bias's input by what it
  `requires` (distances for `pos`, padded eigenvectors plus eigenvalues for `spectral_pe`); the knot
  lookup is shared with the distance profile.
- **Precompute cost**: 32 eigenpairs of a 50k-cell kNN slide take 2.7 s (64: 4.6 s), 2500-spot
  synth_spot slides 0.3 s. Runtime: one `[B·H, S, k] × [k, S]` product, the distance bias's memory.
- No coordinates needed; no `pe` probe column.
- Code: `SpectralBias`, `_spectral_inputs` and `report_spectral_range` in `positional_encodings.py`;
  `laplacian_spectrum` in `tl/positional.py`; tests in the spectral section of
  `tests/test_positional_encodings.py`.

### How Stage 9 (post-hoc evaluation on synth_spot) was built

Added 2026-10-10 on the user's request, to replace synth_data_0's probes and attention regression
in the run script; `MODE = "evaluate"` in `pe_test_2.py` reloads every arm's checkpoint
(`<prefix>model.pt` in `model.save`) and probes it, fit on the train slides, scored on the test
slides.

- **Every probe reads a masked pass.** `get_model_output` embeds each spot from its own unmasked
  expression, so predicting that expression from it inverts the encoder (the online probe's
  finding, R2 0.33 on a noise gene). `evaluation.masked_probing.draw_masks` hides, with a private
  seed, either entries at the training rate or whole spots, and `online_probes.collect_features`
  embeds that pass. `input_baselines=True` adds the spot's own masked input and the mean of its
  neighbours' masked input as feature sets, next to local, global, local+global and `pe`.
- **Regression by cosine, over the whole gene vector** (the user's choice, replacing MSE and R2):
  a ridge per gene, fit on the train spots where that gene was hidden, gives each test spot a
  predicted gene vector; it is compared with the true vector over the spot's hidden entries. Raw
  cosine as asked, beside the `mean` floor -- every spot's log1p vector has cosine 0.907 with the
  mean profile -- and the cosine of deviations from the training mean (0 for predicting the mean).
  Per programme (`var['program']`) the centred cosine is pooled over the programme's hidden
  entries; the background programmes are the negative control.
- **Two passes.** Entries hidden: the receptor is often visible while the response is hidden, so
  the sparse programmes are testable. Whole spots hidden: what the context alone says -- the gene
  vector, the context labels (region, R3/R4 sender state, M9 stripe, M10 cone; balanced logistic
  regression) and the vector of distances to every source on every slide (z-scored, cosine).
- **Attention** on the test slides from `get_model_output`: ROC AUC against the planted edges per
  range class (`subset_edges` renumbers them to the subset), mean attention per distance bin, and
  the attention regression with distance, region and counts as blocks. Short-range edges sit
  inside the long-range mask; read them accordingly.
- **Gene ranks** per arm (added 2026-10-10 at the user's request, as in the earlier scripts and
  `3_downstream_tasks.ipynb`): `calculate_gene_ranks` on the test slides' `get_model_output`, the
  rank plot and the mean ranks per programme. It scores against `adata.X`, which holds raw counts
  in the preprocessed file, so the training layer (`dataset.layer_key`, sparse) becomes `X` first. A
  higher rank is a better R2. The old pipeline's "lower = better" label was wrong. The predictions
  come from unmasked input; see the open question.
- **Gene loadings** per arm (added 2026-10-10, next to the ranks). These read the decoder weights,
  not the predictions. `gene_loadings` gives S_gk = W_gk·std(z_k)/std(x_g) for both linear
  decoders, using the test slides' embeddings. `calculate_dim_importance` (`mode="full"`, cutoff
  0.60) keeps dims, and `get_genes_dim` gives the top 20 genes per kept dim. The settings are
  copied from `plot_ranking.py`. Outputs: `gene_loadings_{local,global}.csv`, `dim_importance.csv`
  with the elbow plot, `top_genes_dim_{local,global}` heatmap and CSV, and
  `loadings_by_program.csv`. The last holds the mean ||S_g|| per programme for each decoder; with
  uncorrelated dims ||S_g||² is the share of the gene's variance that decoder reproduces. Both
  decoders reconstruct the whole vector, so these are two readouts, not a split. Skipped unless
  `dual_decoder` is on with a `linear` decoder. Checked on synthetic weights with planted dims and
  programmes; not yet run on a checkpoint.
- **When it runs:** in `MODE = "sweep"` right after each trial trained (`EVALUATE_AFTER_TRAINING`),
  while its W&B run is still open, so the headline numbers land in that run's summary under
  `eval/`, and the rank and elbow plots are logged along with the number of kept dims per decoder. A failed evaluation is printed and does not fail the
  trained run. `MODE = "evaluate"` redoes it from the checkpoints.
- Library code takes every name as an argument; the synth_spot names live in the script's config
  block (`CONTEXT_LABELS`, `COVARIATE_PREFIX`, `GENE_GROUP_KEY`, `ATTENTION_OBS`, `DISTANCE_BINS`).
  Output: everything of one run in `RESULTS_DIR/<arm>/` (one CSV per analysis, the gene-rank and
  loading plots), `RESULTS_DIR/all_arms.csv` combined from disk, and a printed headline per analysis.
- Code: `evaluation/masked_probing.py`; `input_baselines` in `online_probes.collect_features`;
  `subset_edges`, `attention_distance_profile` and the `edges` argument of `score_against_truth`
  in `evaluation/downstream_regression.py`; tests in `tests/test_masked_probing.py` and
  `tests/test_online_probes.py`.

## The encodings

| PE | kind | input | encoder | invariant to |
|---|---|---|---|---|
| naive | node | centred pos / `length_scale` | MLP 2 → hidden → `n_embed` | translation |
| sinusoidal | node | centred pos (µm) | sin/cos at `dim/4` geometric wavelengths per axis → Linear | translation |
| lap | node | `k` lowest non-trivial eigenvectors, sym. normalised Laplacian | Linear; random sign flip per vector per graph (training) | translation, rotation |
| rw | node | `diag((D⁻¹A)^t)`, t = 1..steps | BatchNorm (`eps` 1e-8) → Linear (GraphGPS) | translation, rotation |
| distance | bias | ‖pᵢ − pⱼ‖ in µm | `kind: profile`: K triangular kernels → a `[heads, K]` table, piecewise linear in d, flat beyond `max_dist`; `kind: linear`: one slope per head on d in mm (ALiBi). Zero-init, shared across layers | translation, rotation |
| spectral | bias | k lowest non-trivial eigenpairs of the sym. normalised Laplacian (precomputed) | per head a filter `h_h(λ)`, piecewise linear in log λ over 32 knots (1e-5–2); bias `Σᵢ h_h(λᵢ)·N·vᵢ[j]·vᵢ[l]`. Zero-init, shared across layers | translation, rotation; the eigenvectors' signs and bases |
| rope | rotary | centred pos (µm) | pair i of every head's q and k turned by ωᵢ·p, so qₘ·kₙ depends on pₙ − pₘ; wavelengths geometric over `[min_wavelength, max_wavelength]`, dealt out across heads; `kind: axial`: fixed, half the pairs along x, half along y; `kind: mixed`: learnable 2D frequencies from per-head frames turned by 90°/heads. Shared across layers | translation (cell-to-cell); not rotation |

- **naive / sinusoidal**: a fixed physical scale, not SpaFormer's per-FOV min-max — graphs here
  differ in size, so min-max would encode the same distance differently per graph. Sinusoidal
  wavelengths are in µm (cell diameter → window extent; `get_average_local_and_global_size`
  reports both), not DETR's temperature 10000, which assumes integer positions.
- **lap**: every connected component has its own zero eigenvalue, so exactly one trivial
  eigenvector is dropped per component (an exact count, not a tolerance — a long, thin graph's
  first real eigenvalue can be ~1e-6); isolated cells have eigenvalue 1 and are ordinary. Zero-pad
  when fewer than `k` remain; shift-invert `eigsh` for large graphs; canonical signs so rebuilt
  graphs match. **Scale:** columns are unit-norm over the whole graph, so entries shrink like
  `1/sqrt(N)` — a 50k-cell slide gets ~10x smaller values than a 500-cell window. Harmless while a
  run's graphs are of similar size; with mixed sizes (whole slides beside windows), scale by
  `sqrt(N)` (a one-flag change in `tl.laplacian_pe`). Not done: instead `prepare_geome_dataset`
  logs a warning when the largest graph of a run (all splits) is more than 4x the smallest, i.e.
  when entries differ by more than 2x.
- **rw**: isolated nodes get zeros; step 1 is always 0 without self loops (kept for parity with
  the standard definition). Computed from half-powers of the symmetric `D^-1/2 A D^-1/2`, which
  has the same diagonal powers as `D^-1 A`, so a 50k-cell slide never forms 16-hop fill-in; still
  ~9 s per 50k-cell graph at 16 steps, paid again in every sweep trial.
  *Measured, correcting the earlier worry:* a symmetrised kNN graph is **not** near-regular (6-NN
  gives mean degree 12, varying per cell), and return probabilities there spread across cells by
  as much as on radius graphs (std 0.007–0.011 against means 0.02–0.14). Still check the spread
  before reading a null result.
  *BatchNorm `eps`:* late-step spreads are ~5e-3 (variance ~1e-5) on a 14-neighbour graph and
  smaller on denser ones; BatchNorm's default `eps` 1e-5 would keep only 84% of the step-16 spread
  there and ~14% at a spread of ~4e-4. GraphGPS uses the default; this uses 1e-8.
- **distance**: zero-init, so switching it on changes nothing at step 0. With the long-range mask
  on, the bias inside the mask radius is never used — interpret `b_h(d)` only beyond it
  (`DistanceBias.profile()` returns the knots and each head's values). **`max_dist` is per
  dataset**: beyond it every distance looks the same, which is why it is derived by default
  (synth_data_0 ~1400, synth_spot ~7000). Memory, measured at B=4 graphs of S=2500 tokens and H=4 heads (fp32): the bias is 0.40 GB
  and autograd keeps 0.50 GB of it for the backward pass; a Gaussian `[B, S, S, K]` feature tensor
  at K=16 would be 1.60 GB on its own.
- **rope**: the content-dependent counterpart of the distance bias. A head can attend to one kind
  of cell at one offset, where the bias applies one profile to every pair. RoPE sees direction
  (MC12, MC13 on synth_spot), which LapPE, RWPE, the bias and the GCN cannot; `rotate_train` erases
  it. On a lattice, wavelengths below twice the spacing alias: on synth_spot's 100 µm lattice the
  default 50 µm wave turns by whole cycles between lattice points and is the identity along the
  axes, so set `min_wavelength` to ~200 µm there (the 2-hop mask radius). Defaults on synth_data_0
  (diameter ~1400 µm, 4 heads × 4 dims): wavelengths 50, ~190, ~740 and ~2800 µm, one per head.
- **spectral**: a learned diffusion kernel, so distance runs along the graph and around holes;
  synth_spot's programmes are Euclidean, which differs only near its holes. `k` sets the finest
  scale — the shortest wavelength is about `L·√(π/k)`: 1.6 mm at k = 32 on synth_spot, 1.1 mm at
  k = 64. On a fragmented graph it only reaches the pieces that host the lowest modes (synth_data_0,
  see Open questions).

## Stages

Every stage: tests pass, and it runs end to end in `GlobalModel`, `CombinedModel` and the
dual-decoder variant, plus the stage-specific check below. A stage only exercised on synthetic
tensors is `implemented, unverified`, not done.

- **P** — the two fixes. Verify, on the *built* model rather than a helper: for each model type
  and local component, the transformer's mask reach equals the local module's message-passing
  depth, and ring NCE's inner radius equals it too; a yaml with `long_range_mask_hops` raises; a
  file config with `optim.loss: none` and zero aux weights raises. Then capture the equivalence
  baseline.
- **0** — `pe.node`/`pe.bias` + `dataset.spatial_unit_um`, `_validate_pe`, registry, `pad_like`,
  `GlobalInput`, `attn_bias`, precompute hook, checkpoint tag, forked RNG. Verify:
  `scripts/equivalence_harness.py` IDENTICAL with PEs off — including `combined_longrange_mse_node`,
  the one case that runs the mask path the bias merge changes; `pad_like` reproduces `pad_batch`'s
  layout including the subsampling and kept-masked-node cases; a test-registered dummy node PE
  and a zero bias leave outputs unchanged.
- **1 naive** — verify: translating every coordinate leaves the outputs unchanged.
- **2 sinusoidal** — verify: matches the closed form; encoding similarity decays with distance.
- **3 lap** — verify: equals PyG `AddLaplacianEigenvectorPE` up to sign on a connected graph;
  a disconnected graph is handled; flips only in training, batch order identical with flips on.
- **4 rw** — verify: equals PyG `AddRandomWalkPE`; an isolated node is handled.
- **5 distance** — verify: bias layout equals the mask's (left pad, graph-major); CLS row/column
  zero; zero-init forward identical to no bias; finite on the existing dense-graph NaN test at
  hops 1–3. Log `b_h(d)` per head.
- **6** — a `pe` probe embedding (the summed node PE) as a control column, then the sweep below.
  *As built:* `probe.embeddings` includes `pe` by default; `online_probes.node_positional_encoding`
  returns the **learned** encoding as added to the tokens (eval mode, so no rotation or flip), not
  the raw coordinates/eigenvectors. Its output layers start at zero, so the column starts at
  chance and rises as the model learns to use position. Bias-only arms (distance) have no node PE
  and log no `pe` column — there is no per-cell quantity to probe.

## Sweeps

No sweep code changes needed:

- list-valued `pe.node` / `pe.bias` go through **arms**, so the lists never round-trip through
  wandb and every arm sets the same keys;
- scalar knobs (`pe.lap.k`, `pe.rw.steps`, `pe.sinusoidal.max_wavelength`, `pe.distance.max_dist`,
  `pe.rope.kind`, `pe.rope.min_wavelength`, `pe.spectral.k`) are plain dotted parameters, but only in a sweep whose
  arms all enable that PE — otherwise trials vary a knob nothing reads;
- `pe.rotary` is a list like the other two; an arm that does not set it keeps the default `[]`.

```yaml
# config_files/sweeps/pe_ablation.yaml   (untracked, like every yaml)
arms:
  none:       {model.global_component.parameters.pe.node: [],           model.global_component.parameters.pe.bias: []}
  naive:      {model.global_component.parameters.pe.node: [naive],      model.global_component.parameters.pe.bias: []}
  sinusoidal: {model.global_component.parameters.pe.node: [sinusoidal], model.global_component.parameters.pe.bias: []}
  lap:        {model.global_component.parameters.pe.node: [lap],        model.global_component.parameters.pe.bias: []}
  rw:         {model.global_component.parameters.pe.node: [rw],         model.global_component.parameters.pe.bias: []}
  distance:   {model.global_component.parameters.pe.node: [],           model.global_component.parameters.pe.bias: [distance]}
  spectral:   {model.global_component.parameters.pe.node: [],           model.global_component.parameters.pe.bias: [spectral]}
  rope:       {model.global_component.parameters.pe.node: [],           model.global_component.parameters.pe.bias: [], model.global_component.parameters.pe.rotary: [rope]}
sweep_config:
  method: grid
  parameters:
    arm:        {values: [none, naive, sinusoidal, lap, rw, distance, spectral, rope]}
    optim.seed: {values: [0, 1, 2]}
```

Run it once with `--model_type GlobalModel` and once with `CombinedModel`, since the expected
effect differs (see above). The base config must set `dataset.spatial_key`.

## Invariants

- Per-token values enter before `pad_batch` or are gathered with its `index_nodes`; never a
  second `pad_batch`.
- Bias: left-padded, CLS last, graph-major; CLS row/column and padding finite; merged by
  `masked_fill`.
- Rotation: laid out like the tokens (left-padded, CLS last, graph-major heads), CLS and padding
  unrotated; on the layers for one `forward` call only.
- Spectral bias: whole eigenspaces only (a tied group at the cut is dropped), so it depends on the
  eigenvectors through their projectors alone.
- PEs off ⇒ same parameters, same RNG draws, same checkpoint name as today.
- PE randomness only in training, and only from the PE generator.
- Lengths in µm via `dataset.spatial_unit_um`; no dataset constant in code.

## What PEs change about the measurements

- **Absolute PEs feed position to the probes.** A position-derived target (`dist_to_hub`, niche)
  can be read straight off a naive or sinusoidal PE, so a probe gain under one is no evidence of
  learned interaction until it beats the `pe` control column (Stage 6).
- **A distance bias adds a proximity prior to every attention map**, and so to net attention
  flow. Co-located non-interacting types (the `medulla` control) gain flow from proximity alone —
  rerun the flow control's null pairs before reading flow from a bias model.
- **The spectral bias adds a diffusion-geometry prior**, the distance bias's caveat along the
  graph instead of straight across: rerun the flow control's null pairs before reading flow.
- **RoPE shapes attention by offset**, weighted by what the two cells express. Like the bias it
  moves attention flow with geometry, so rerun the flow control's null pairs before reading flow
  from a RoPE model; it has no `pe` probe column.
- **LapPE/RWPE come from the GCN's own graph.** That does not break the long-range separation: the
  mask governs what the transformer attends to, not what its inputs encode.

## Benchmark dataset: synth_spot

Built 2026-10-05 for the PE comparison: a spot-based synthetic dataset whose geometry is exactly
what the encodings differ on — a regular lattice, its border and holes, and directions. Generator:
`src/interscale/evaluation/synth_spot.py` (`make_synth_spot`, or run it as a script; every rule
sits next to its code). Local copy: `/home/lehnerl/Arbeit/data/synth_spot.h5ad` (seed 0, 149 MB
gzip, ~19 s to regenerate). The script also writes per-gene maps of one slide next to the file,
`synth_spot_maps_<slide>_expected.png` and `..._counts.png` (`plot_slide_maps` draws any slide).
Extended 2026-10-06 with the sparse-response programme (item 18) and twice the background genes.

**Layout.** Conditions A and B, 20 slides each. A slide is a 50 x 50 square lattice, 100 µm centre
to centre, centred on the origin with no spot on an axis (±50 … ±2450 µm). 5 slides per condition
lose one border row or column (50 x 49), 10 get 1–5 interior holes (2x2, 3x3, 3x5). 99,090 spots
(2427–2500 per slide) x 253 genes. Split per slide, 12/4/4 per condition (60/20/20; was 14/3/3 until 2026-10-10); val and test each hold
both values of every slide-level factor (M4, M5, crop, holes).

**Levels.** `0` is a structural zero, `down` 1, `up` and `medium` 5, `high` 10. Structured genes
scatter by about a count (`noise_scale` 0.4 x sqrt(mean): `down` reads 0/1/2 at 16/68/16%).

| programme | genes | rule |
|---|---|---|
| condition | cg_1–3 | down/0/down in A, high/up/high in B |
| batch | bg_1–3 | one level per slide from {0, down, up, high}; the triple is unique per slide |
| short range | S1–S4 → R1–R4 | senders at random (S1, S2 5%; S3 only, S4 only 4%; both 2%), responses in the 4 nearest neighbours: R1 up (else 0), R2 0 (else high), R3/R4 high/0 for S3, 0/high for S4, medium/medium for both or neither |
| long range | M1–M3 → MC1–MC3 | one Gaussian cluster per slide (σ 700, 350, 1200 µm); MC mirrors M. M1 and MC1 never fall below `down`; M3/MC3 peak at `up` |
| | M4/MC4, M5/MC5 | as M2/M3 on every 2nd / 3rd slide of each condition |
| | M6/MC6 | 3–5 small clusters (σ 200) |
| | M7 → MC7–MC9, M8 → MC10–MC11 | sin² rings at 0–1, 1–2, 2–3 mm and 0–2, 2–4 mm around a σ-350 cluster |
| | M9 → MC12 | a 2–3 x 2–3 block; MC12 only in the block's rows, both ways along x: high for 300 µm, then decay length 800 µm |
| | M10 → MC13 | one spot; MC13 rises from `down` to high inside a 45° cone toward +y or −y |
| | M11 → MC14, MC15 | 3–6 single spots ≥ 1 mm apart; within 500 µm MC14 falls and MC15 rises exponentially |
| spatially variable | SP1–SP10 | 1–5 rotated Gaussians per gene, some elongated or skewed, high → 0 |
| region | CM1–CM3 | slide halved on the x-axis, the y-axis or a diagonal |
| background | HV1–10, NB1–40, ZINB1–40, POI1–40, H1–40 | BetaBinomial(100); NB (mean 1–10); ZINB (mean 1–50); Poisson(5) on 1–10 spots per slide; Poisson (mean 5–10) |
| sparse response | M12–M21 → MC16–MC25, receptors MR16–MR25 | compact sources (σ 200 µm; 1 cluster, or 1–3 for M13, M18, M21). Only receivers respond: 20–50 random spots per slide, 200–500 for MC23–MC25. A receiver expresses its receptor at `high` and a response set by its distance to the nearest cluster, in 1 mm bands: MC16, MC17 +5 per band from 0; MC18 +20 per band; MC19 100, −20 per band. Or by a threshold: MC20, MC23 5 beyond 2.5 mm; MC21, MC24 25 beyond 3 mm; MC22, MC25 10 beyond 1 mm |

**Decided with the user, 2026-10-05:** without an M4/M5 source, MC4/MC5 sit at their plateau (they
are functions of the local M level only); MC13's region is a 45° cone, not a diagonal ray; the
structured genes get the low noise of the spec's `down`, not Poisson; 20 NB genes. **Defaults
taken, not asked:** "nearest neighbours" is the 4-neighbourhood; S3 and S4 arriving from two
different neighbours cancel like a co-expressing spot; `medium` = `up`; crops and holes are applied
before simulating, so no response has a sender outside the data.

**Decided with the user, 2026-10-06 (item 18):** every sparse programme gets a receptor gene, MR16–MR25.
Without one, ~35 receivers among 2500 spots are indistinguishable from the rest, and the best
prediction of a masked response is ~0 for every model. The spec named both (ii) and (iii) MC17, so
the numbering shifts: (iii) is MC18, and each later response moves up by one; (viii), (ix) and (x)
copy MC20, MC21 and MC22. **Defaults taken, not asked:**

- A source "cluster" is a compact Gaussian. M is about 0.1 counts at 600 µm and reads 0 counts
  everywhere beyond 1 mm.
- Clusters sit within ±80% of the half-width, and one programme's clusters are at least 1 mm apart.
- Receivers are drawn uniformly over the slide, independently per programme.
- Distance and threshold are measured to the nearest cluster.
- Adding genes left every existing gene's counts byte-identical, since each gene has its own random
  stream. Their column positions did move, though, because NB21–40 and the other new background
  genes sit with their families, so address genes by name.

**Ground truth** is stored under `uns['synthetic']`, the same key synth_data_0 uses, so
`downstream_regression.score_against_truth` and `run_synthetic_pipeline.py`'s radius check work
unchanged. It holds the slide table, holes, every source position and the SP blobs. The noise-free
means are in `layers['expected']`. `obs` holds sender flags, exposures, `R34_state`, the distance
to every source (including `dist_M12`–`dist_M21`), the stripe and cone flags, `region` and
`is_receiver_MC16`–`MC25`. `uns['synthetic']['sparse_programmes']` holds item 18's table. Edge
classes, with counts:

- `short`: sender → its 4 neighbours, exactly 100 µm; 85,842.
- `mid`: M11, < 500 µm; 11,573.
- `directional`: M9 and M10; 26,874.
- `sparse` and `dense`: from the core of the nearest cluster (spots at half its peak or more) to
  every receiver, including those the distance holds at 0; 168,592 and 702,450.

The Gaussian programmes M1–M8 have no discrete sender, so no edges.

**Config:** `max_seq_len >= 2500`; `spatial_neigbors_kwargs.radius: 110` (anything in (100, 141)
gives the 4-neighbourhood the short-range rules use); `spatial_key: spatial`,
`spatial_unit_um: 1.0`; `sample_key: [slide]`; `layer_key: log1p_norm`.

**Run scripts (2026-10-10):** `notebooks_and_scripts/pe_test_2.{py,yaml}` + `pe_sweep_2.yaml`.
They preprocess as the legnini tutorial does (raw counts in `X`/`layers['raw']`,
`normalize_total(1e4)` + log1p into `log1p_norm`, written to `synth_spot_pp.h5ad`), set
`max_seq_len` to the largest slide, train 100 epochs under gene masking, and sweep ten arms. The
user chose a k-NN graph instead of the radius-110 one: k = 8 (`radius: null`, `n_neighs: 8`), after
k = 6 turned out to pick tied diagonals (below).

### What the lattice does to the encodings (measured on the data)

- **RWPE barely sees position.** On an intact slide the 16-step RWPE takes 40 distinct values over
  2500 spots, and the deep interior — 46% of spots — shares one. It encodes distance to the border
  and to holes (378 distinct values on a 4-hole slide), not where a spot is.
- **LapPE is degenerate on square slides.** On an intact 50 x 50 slide λ1 = λ2 to 1e-14 (the x and
  y modes; λ4 = λ5 too), so the first two columns are an arbitrary rotation within that plane, which
  sign flips do not cover. A crop or holes split the pair by ~4%, so intact and damaged slides get
  differently oriented encodings for the same layout.
- **Only coordinate PEs see direction.** MC12 runs along x and MC13 points along ±y. LapPE, RWPE,
  the distance bias and the isotropic 4-neighbour GCN are all blind to orientation, and
  `rotate_train` erases it. RoPE sees it as the direction between two cells.
- **Centring moves with a crop.** Losing one border line shifts a slide's centroid by 50 µm, so a
  centred coordinate PE sees intact and cropped slides 50 µm apart.
- **k-NN with k = 6 is directed and picks tied diagonals.** Every spot has 4 lattice neighbours at
  100 µm and 4 diagonals tied at 141 µm, so k = 6 keeps 2 of the 4 diagonals by tie-break. On
  slide A_01 (squidpy, measured 2026-10-10) the (−,−) diagonal is picked 1544 times and the
  (+,+) 890 times. The adjacency is not symmetric, and the symmetrised degree runs 6–10. **k = 8**
  (used since 2026-10-10) has no interior ties: every interior spot gets its 4 neighbours and 4
  diagonals, each diagonal direction equally often. Spots at a border or a hole reach out to
  200–283 µm (3% of edges). The symmetrised degree is 8 for 92% of spots and up to 13 at corners.
  The 2-layer GCN's reach, and so the long-range mask, is the 5 x 5 block (≤ 283 µm).

### What is local and what is not

Each cell is the fraction of a gene's `log1p_norm` variance explained by two predictors. The first
is the mean of its 4 neighbours for the same gene; the second is any function of the spot's own
source gene:

| genes | 4-neighbour mean | own source gene |
|---|---|---|
| R1–R4 | 0.00 | — (they follow the neighbours' S genes) |
| MC1–MC6 | 0.33–0.72 | 0.39–0.69 |
| MC7 | 0.98 | 0.82 |
| MC8–MC11 | 0.97 | 0.03–0.18 |
| MC12, MC14, MC15 | 0.80–0.88 | 0.00 |
| MC13 | 0.10 | 0.00 |
| SP, region markers | 0.75–0.95 | |
| condition, batch | 0.90–0.97 | |
| background | 0.00 | |

Every long-range field is smooth at 100 µm, so under node masking a spot's unmasked neighbours
already predict most of it. A PE can only win where they cannot: when the neighbourhood is masked
too, in the far rings (MC8–MC11, nothing at the spot itself), in the direction of MC12/MC13, and in
whether MC4/MC5 sit at their plateau. The short-range responses are the opposite case — invisible
to same-gene smoothing, fully determined by the neighbours' S genes.

**Item 18 is built to close that gap.** For each programme's receivers, `log1p_norm`, the table
gives three numbers. The first is how often another receiver lies within the 2-hop reach of a
2-layer GCN, and how much of the response that receiver's mean explains. The second is how much the
strongest source expression within that reach explains. The third is how much the true distance
to the nearest cluster explains:

| responses | another receiver ≤ 2 hops | R² from it | R² source trace ≤ 2 hops | R² distance |
|---|---|---|---|---|
| MC16–MC19 (steps) | 14–19% | 0.05–0.16 | 0.10–0.46 | 0.81–0.97 |
| MC20, MC21 (beyond 2.5 / 3 mm) | 15% | 0.09–0.14 | 0.07–0.08 | 0.97–0.99 |
| MC22 (beyond 1 mm) | 17% | 0.14 | 0.51 | 0.96 |
| MC23, MC24 (dense, 2.5 / 3 mm) | 82–83% | 0.73–0.75 | 0.03–0.06 | 0.96–0.98 |
| MC25 (dense, 1 mm) | 80% | 0.65 | 0.48 | 0.96 |

The sparse receivers are what the design intends. The neighbourhood gives the response away for
fewer than one in five, while the distance to a source 1–6 mm away explains almost all of it. The
dense programmes are the control, where neighbouring receivers leak it back. The source-trace column
is band 0 only: restricted to receivers beyond 1 mm it explains 0.03–0.05 for every programme. It
looks large because in log space "0 vs responding" carries most of the variance, and receivers
within ~600 µm of a cluster can see it. That makes the 1 mm threshold (MC22, MC25) the weakest
global test.

**Train it with gene masking.** Masking individual entries leaves the receptor visible while
the response is masked, so the model knows *that* a spot responds and needs the source for *how
much*. Node masking still trains, and a receiver is unmasked 70% of the time, but the response is
only scored in the epochs where the whole spot, receptor included, is masked. At those moments no
model can tell a receiver from the other 98.6% of spots, so the best prediction is
P(receiver) x level. Measured with a perfect-distance oracle, the share of the loss knowing the
source removes:

| | sparse MC16–MC22 | dense MC23–MC25 |
|---|---|---|
| node masking | 0.0–0.9% | 3–11% |
| gene masking, receptor visible | 66–100% | 69–100% |

The gene-masking floor of 66% (MC20) and 69% (MC23) comes from the oracle working in 1 mm bands
while those thresholds sit at 2.5 mm. It is not a property of the data.

## Open questions

- Resolved 2026-10-08: `max_dist` is derived per dataset (see the Stage 5 notes). The sinusoidal
  wavelengths stay fixed µm defaults.
- `rotate_train` stays off by default: some tissues have a meaningful axis (layered cortex). It
  only acts in training forward passes, on the coordinate node PEs (naive, sinusoidal) — one fresh
  angle per graph per forward pass; never in eval, `get_model_output` or the probes, and never on
  the distance bias (already rotation-invariant), LapPE or RWPE.
- LapPE `sqrt(N)` scaling: accepted as is (2026-10-08); a warning is logged when graph sizes in a
  run differ more than 4x.
- RoPE at the default width: 4 dims per head leave one wavelength per axis per head (4 scales over
  the model). `n_embed: 32`, or 2 heads, would give each head two; not decided.
- `pe.rope.min_wavelength` defaults to 50 µm, which suits cells; synth_spot wants ~200 µm (see the
  rope notes above).
- **synth_data_0's radius-30 graph is fragmented** (measured 2026-10-09 on six slides): mean degree
  4.3, at the percolation threshold, 75–93 components per slide, the largest holding 50–67% of the
  cells. Eigenvectors live on single components, so **31% of cells get an all-zero LapPE at k = 8**
  and 22% are in no spectral-bias mode at k = 32 — read the running sweep's `lap` arm with that in
  mind. At radius 40: 2% and 2%; at 50: 0%. Why: the generator draws cells as independent random
  points (an inhomogeneous Poisson process, `_sample_positions`), with no minimum spacing, so they
  clump and leave gaps (nearest neighbour 13 µm on average, under 5 µm for a tenth of cells); a
  radius graph of such points falls apart below a mean degree of ~4.5, the continuum-percolation
  threshold, and radius 30 gives 4.3. A **kNN-6 graph** (what the user intends to use) is connected
  on all 24 slides: every cell gets 6 edges whatever its local density. There LapPE and the spectral
  bias reach every cell, eigenvalues 1.3e-3 to 5.0e-2 at k = 32, and the 6th neighbour is 36 µm away
  on average. Map of one slide under both graphs: `/home/lehnerl/Arbeit/data/synth_data_0_graph_components.png`.
- `pe.spectral.k` defaults to 32 (shortest wavelength ~1.6 mm on synth_spot); the 1 mm bands of the
  sparse programmes may want 64.
- **Gene ranks from unmasked input (deferred by the user, 2026-10-10).** `calculate_gene_ranks`
  scores `get_model_output` predictions, which see every entry. A gene-masking model was only
  trained on hidden entries, so these ranks compare decoders on input they were never trained on.
  The user wants the variant that ranks genes on hidden entries only, later.
