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
| 5 — distance bias | not started | | |
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
        distance:   {num_kernels: 16, max_dist: 500.0}                      # µm
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

## The five encodings

| PE | kind | input | encoder | invariant to |
|---|---|---|---|---|
| naive | node | centred pos / `length_scale` | MLP 2 → hidden → `n_embed` | translation |
| sinusoidal | node | centred pos (µm) | sin/cos at `dim/4` geometric wavelengths per axis → Linear | translation |
| lap | node | `k` lowest non-trivial eigenvectors, sym. normalised Laplacian | Linear; random sign flip per vector per graph (training) | translation, rotation |
| rw | node | `diag((D⁻¹A)^t)`, t = 1..steps | BatchNorm (`eps` 1e-8) → Linear (GraphGPS) | translation, rotation |
| distance | bias | ‖pᵢ − pⱼ‖ in µm, clamped at `max_dist` | K Gaussian RBFs → Linear(K → heads), zero-init, shared across layers | translation, rotation |

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
  on, the bias inside the mask radius is never used — interpret `b_h(d)` only beyond it.

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

## Open questions

- Fixed µm defaults for the wavelengths and `max_dist`, or derived per dataset from the window
  extent?
- `rotate_train` is off by default because some tissues have a meaningful axis (layered cortex).
- LapPE columns are unit-norm over the whole graph, so entries scale like `1/sqrt(N)`: a 50k-cell
  slide gets ~10x smaller values than a 500-cell window. Fine while graphs in one run are of
  similar size; with mixed sizes, scaling by `sqrt(N)` is the candidate fix (one config flag).
