# Staged plan: positional encodings for the global module

Adds five positional encodings (PEs) to the transformer, selectable from config and sweepable:
naive coordinates, 2D sinusoidal, Laplacian eigenvectors (LapPE), random-walk (RWPE), and a
relative Euclidean-distance attention bias. Sources: SpaFormer (Wen et al., arXiv 2302.03038) for
the spatial variants, GraphGPS (Rampášek et al., NeurIPS 2022) for the graph ones and for the
node-feature vs. attention-bias split. Read [`background.md`](background.md) first.

## Status

Last updated 2026-10-04. Same rules as [`contrastive_plan.md`](contrastive_plan.md): update this
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
| 6 — PE probe control + ablation sweep | not started | | |

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
        center_coords: True  # subtract each graph's centroid
        rotate_train: False  # random rotation per graph, training only
        naive:      {hidden_dim: 32, length_scale: 100.0}                   # µm
        sinusoidal: {dim: 32, min_wavelength: 10.0, max_wavelength: 1000.0} # µm
        lap:        {k: 8, sign_flip: True}
        rw:         {steps: 16}
        distance:   {num_kernels: 16, max_dist: 2000.0}                     # µm; set per dataset
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
- **`max_dist` defaults to 2000 µm, not 500**: 500 would make everything beyond half a millimetre
  one distance, which is where synth_spot's global test sits. It stays a per-dataset setting.
- **`b_h(d)` is readable, not logged.** `DistanceBias.profile()` returns it; logging it during
  training waits for a consumer (Stage 6's sweep).
- Code: `DistanceBias`, `AttentionBias` and `build_attention_bias` in
  `module/global_modules/positional_encodings.py`; `tl.pad_like` in `tl/padding.py`; the merge in
  `TransformerNodeEncoderHook.common_step_local_to_global`.

## The five encodings

| PE | kind | input | encoder | invariant to |
|---|---|---|---|---|
| naive | node | centred pos / `length_scale` | MLP 2 → hidden → `n_embed` | translation |
| sinusoidal | node | centred pos (µm) | sin/cos at `dim/4` geometric wavelengths per axis → Linear | translation |
| lap | node | `k` lowest non-trivial eigenvectors, sym. normalised Laplacian | Linear; random sign flip per vector per graph (training) | translation, rotation |
| rw | node | `diag((D⁻¹A)^t)`, t = 1..steps | BatchNorm (`eps` 1e-8) → Linear (GraphGPS) | translation, rotation |
| distance | bias | ‖pᵢ − pⱼ‖ in µm, clamped at `max_dist` | K triangular kernels → one value per head (a `[heads, K]` table, piecewise linear in d), zero-init, shared across layers | translation, rotation |

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
  `sqrt(N)` (a one-flag change in `tl.laplacian_pe`). Listed under open questions too.
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
  dataset**: beyond it every distance looks the same. The default 2000 µm covers synth_data_0
  (diagonal ~1400); synth_spot's sparse responses depend on distances of 1–6 mm, so it needs
  ~7000. Memory, measured at B=4 graphs of S=2500 tokens and H=4 heads (fp32): the bias is 0.40 GB
  and autograd keeps 0.50 GB of it for the backward pass; a Gaussian `[B, S, S, K]` feature tensor
  at K=16 would be 1.60 GB on its own.

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

## Sweeps

No sweep code changes needed:

- list-valued `pe.node` / `pe.bias` go through **arms**, so the lists never round-trip through
  wandb and every arm sets the same keys;
- scalar knobs (`pe.lap.k`, `pe.rw.steps`, `pe.sinusoidal.max_wavelength`, `pe.distance.max_dist`)
  are plain dotted parameters, but only in a sweep whose arms all enable that PE — otherwise
  trials vary a knob nothing reads.

```yaml
# config_files/sweeps/pe_ablation.yaml   (untracked, like every yaml)
arms:
  none:       {model.global_component.parameters.pe.node: [],           model.global_component.parameters.pe.bias: []}
  naive:      {model.global_component.parameters.pe.node: [naive],      model.global_component.parameters.pe.bias: []}
  sinusoidal: {model.global_component.parameters.pe.node: [sinusoidal], model.global_component.parameters.pe.bias: []}
  lap:        {model.global_component.parameters.pe.node: [lap],        model.global_component.parameters.pe.bias: []}
  rw:         {model.global_component.parameters.pe.node: [rw],         model.global_component.parameters.pe.bias: []}
  distance:   {model.global_component.parameters.pe.node: [],           model.global_component.parameters.pe.bias: [distance]}
sweep_config:
  method: grid
  parameters:
    arm:        {values: [none, naive, sinusoidal, lap, rw, distance]}
    optim.seed: {values: [0, 1, 2]}
```

Run it once with `--model_type GlobalModel` and once with `CombinedModel`, since the expected
effect differs (see above). The base config must set `dataset.spatial_key`.

## Invariants

- Per-token values enter before `pad_batch` or are gathered with its `index_nodes`; never a
  second `pad_batch`.
- Bias: left-padded, CLS last, graph-major; CLS row/column and padding finite; merged by
  `masked_fill`.
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
(2427–2500 per slide) x 253 genes. Split per slide, 14/3/3 per condition; val and test each hold
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
  `rotate_train` erases it.
- **Centring moves with a crop.** Losing one border line shifts a slide's centroid by 50 µm, so a
  centred coordinate PE sees intact and cropped slides 50 µm apart.

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

- Fixed µm defaults for the wavelengths and `max_dist`, or derived per dataset from the window
  extent? `max_dist` matters most: beyond it the distance bias cannot tell distances apart (see
  the distance note; synth_spot needs ~7000 µm against the 2000 default).
- `rotate_train` is off by default because some tissues have a meaningful axis (layered cortex).
- LapPE columns are unit-norm over the whole graph, so entries scale like `1/sqrt(N)`: a 50k-cell
  slide gets ~10x smaller values than a 500-cell window. Fine while graphs in one run are of
  similar size; with mixed sizes, scaling by `sqrt(N)` is the candidate fix (one config flag).
