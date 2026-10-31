# Staged plan: positional encodings for the global module

Adds seven positional encodings (PEs) to the transformer, selectable from config and sweepable:
naive coordinates, 2D sinusoidal, Laplacian eigenvectors (LapPE), random-walk (RWPE), a relative
Euclidean-distance attention bias, 2D rotary embeddings (RoPE, Stage 7) and a spectral
(diffusion-kernel) attention bias (Stage 8). Sources: SpaFormer (Wen et al., arXiv 2302.03038) for
the spatial variants, GraphGPS (Rampášek et al., NeurIPS 2022) for the graph ones and for the
node-feature vs. attention-bias split, RoFormer (Su et al., arXiv 2104.09864) and RoPE for ViTs
(Heo et al., ECCV 2024) for RoPE, SignNet/BasisNet (Lim et al., ICLR 2023) for the spectral bias.
Read [`background.md`](background.md) first.

Trimmed 2026-10-31: test and mutation counts, the pre-build code audit and the build diaries were removed (git history
has them); the design decisions they recorded are kept under "Design". Where it stands: every encoding is built and used
in sweeps since round 1; RoPE and the Fourier bias (attention_plan.md stage 3) won on synth_spot, and the distance prior
per head (`pe.distance.kind: scales`, roadmap T1) is the encoding of the base model.

## Status

Last updated 2026-10-31. Same rules as [`contrastive_plan.md`](contrastive_plan.md): update this table in the same commit
as the work, and a stage is `done` only when something external verified it.

| stage | state | date |
|---|---|---|
| P — two pre-existing fixes (mask reach from the local module, objective validator) | **done** | 2026-10-03 |
| 0 — plumbing, no behaviour change | **done** (harness IDENTICAL; PE on and off share every other initial weight at a seed) | 2026-10-03 |
| 1 — naive PE | implemented; run in the PE sweeps (affects position-derived readouts only) | 2026-10-03 |
| 2 — 2D sinusoidal | implemented; run in the PE sweeps | 2026-10-03 |
| 3 — LapPE | implemented; run in the PE sweeps; degenerate on square slides (below) | 2026-10-03 |
| 4 — RWPE | implemented; run in the PE sweeps; barely sees position on a lattice (below) | 2026-10-04 |
| 5 — distance bias (`profile`, `linear`; `scales` added 2026-10-21) | implemented; `scales` is the base model's encoding | 2026-10-21 |
| 6 — PE probe control + ablation sweep | implemented; the `pe` probe column is on by default | 2026-10-08 |
| 7 — RoPE (2D rotary) | implemented; in use | 2026-10-09 |
| 8 — spectral bias | implemented; run in the PE sweeps, never learned beyond a faint preference | 2026-10-09 |
| 9 — post-hoc evaluation (synth_spot) | superseded by the masked-pass evaluation (global_plan.md Stage 0, `evaluation/masked_probing.py`) | 2026-10-17 |

## Stage P: two pre-existing fixes (2026-10-03)

1. **The long-range mask always blocked 1 hop** (a config fallback won over the derivation, in every run since `4190b83`).
   **Decided: the config only switches the mask on or off.** `long_range_mask_hops` is removed (a yaml that sets it fails to
   load); the reach comes from the local module that is actually built: GCN / GIN their number of message-passing layers;
   SCVI, Precomputed 0; no local module (`GlobalModel`) 0, with a warning. Ring NCE reads the same number
   (`tl.local_reach_hops`).
2. **`_validate_objective` never ran for a config file.** The validators are one list now, so none can go missing from a branch.

## Config

Lives in `get_global_component_cfg`, so `LocalModel` sweeps drop it automatically. Defaults reproduce the PE-free model.

```yaml
dataset:
  spatial_key: spatial       # required by naive, sinusoidal, distance, rope, fourier
  spatial_unit_um: 1.0       # µm per obsm unit (0.138 for pixel-unit Resolve data)
model:
  global_component:
    parameters:
      pe:
        node: []             # any of [naive, sinusoidal, lap, rw]; summed into the token
        bias: []             # any of [distance, spectral, fourier, expression]; added to the attention logits
        rotary: []           # [rope]; rotates queries and keys in every layer
        center_coords: True  # subtract each graph's centroid
        rotate_train: False  # random rotation per graph, training only
        naive:      {hidden_dim: 32, length_scale: 100.0}                   # µm
        sinusoidal: {dim: 32, min_wavelength: 10.0, max_wavelength: 1000.0} # µm
        lap:        {k: 8, sign_flip: True}
        rw:         {steps: 16}
        distance:   {kind: profile, num_kernels: 16, max_dist: 0.0}         # µm; 0 = the largest slide diameter
        rope:       {kind: axial, min_wavelength: 50.0, max_wavelength: 0.0} # µm; 0 = 2 x slide diameter
```

Each encoder validates its own sub-block (`PESpec.validate`); `_validate_pe` adds the cross-cutting checks (unknown or
duplicate names, coordinates need `dataset.spatial_key`). Lists, because node PEs compose (`[lap, rw]`, as in GraphGPS).

## Design

**Registry** — `module/global_modules/positional_encodings.py`: `PE_REGISTRY[name] = PESpec(kind="node" | "bias" | "rotary",
requires=<Data attrs>, module=<cls>, precompute=<fn | None>, forms=...)`. A new encoding is one entry plus one config sub-block.

- **Node PEs** each project to `n_embed` and are summed onto the token after the first `norm_input`, before `pad_batch`, so
  subsampling, masking and padding carry them; the second `norm_input` normalises token and PE together. Added, not
  concatenated: `n_embed` is the width the decoders and probes share. CLS gets none. **Every node encoder's output layer
  starts at zero**, so PE on and off are identical until the first optimiser step.
- **Precompute** (LapPE, RWPE, spectral): `PESpec.precompute` via `attach_positional_inputs`, called by `prepare_geome_dataset`
  and both models' `get_model_output`, on the full graph with the GCN's `edge_index`. LapPE's sign flip is the encoder's
  `augment` hook, training only.
- **Attention biases** are laid out with `pad_like(x, batch, index_nodes, S)` (deterministic, never a second `pad_batch`), as
  `[B·H, S+1, S+1]` with zero CLS row/column and padding, and merged into the mask in `common_step_local_to_global` by
  `bias.masked_fill(boolean_mask, -inf)` (indexing, never arithmetic); the returned mask is then a float mask.
- **Rotary** (RoPE): rotates each head's queries and keys inside every layer (`transformer_utils.apply_rotary`, right after the
  heads are split, so the maps the interpretability tools read include it). `common_step_local_to_global` returns a named
  `GlobalInput` with a `rotary` field; `forward(..., rotary=)` refuses a missing or foreign rotation and sets it on the layers as
  `layer.rotary` for one call only. One `CoordinateFrame` per model (units, centring, `rotate_train`, the PE generator) is shared by
  node and rotary encodings.
- PE modules are built under `torch.random.fork_rng()`; flips and rotations draw from a generator seeded from `optim.seed`, so
  enabling a PE changes neither existing initial weights nor the batch order at a seed.
- The checkpoint prefix gains `pe-<names>_` when any PE is on.

Decisions recorded while building:
- **Distance bias:** triangular kernels (two lookups into a `[heads, K]` table, the table is the learned profile) instead of
  Gaussian RBFs (a `[B, S, S, K]` expansion: 1.6 GB at S = 2500, K = 16). `max_dist: 0` derives the largest slide diameter
  (`resolve_distance_range`, a buffer, so a checkpoint keeps its range). `kind: linear` is one slope per head in mm (ALiBi).
  `DistanceBias.profile()` reads `b_h(d)`.
- **RoPE:** 2D RoPE needs a head width that is a multiple of 4; the `heads × head_dim/4` wavelengths are spaced geometrically
  and dealt out across heads (each head its own scale). CLS sits at the origin. **Not zero at step 0** (a rotation has no zero).
  `max_wavelength: 0` = twice the largest slide diameter; `min_wavelength` should be about the shortest distance attended over.
  `mixed` learns `magnitude × direction` with only the unit direction trained. Angles in float64. No `pe` probe column.
- **Spectral bias:** `b_h(j, l) = Σᵢ h_h(λᵢ) · N · vᵢ[j] · vᵢ[l]` over the k lowest non-trivial eigenpairs of the symmetric
  normalised Laplacian (`spectral_pe` `[N, k]`, `spectral_eigval` `[1, k]`); `h_h` piecewise linear in log λ over 32 knots on a
  fixed range 1e-5–2, zero at the start; with `h = exp(−tλ)` the truncated heat kernel (Lim et al., Prop. 4, without the IGN).
  Whole eigenspaces only (a tied group at the cut is dropped), so the bias depends on the eigenvectors through their projectors.
  Summed over modes and scaled by √N (averaging shrank the slowest modes by k). The run logs the eigenvalue range and warns when
  it leaves the knots. 32 eigenpairs of a 50k-cell slide: 2.7 s.

## Stage 6 — the PE probe control

`probe.embeddings` includes `pe` by default: `online_probes.node_positional_encoding` returns the learned node encoding as added
to the tokens (eval mode: no rotation or flip), gathered for each kept cell. Its output layers start at zero, so the column starts
at chance and rises as the model uses position. Bias-only and rotary arms have no `pe` column. **A global gain on a
position-derived target counts only where it beats `pe`.** Found while building it: the probe moved the global torch RNG and
Python's (`pad_batch` subsampling), so probe-on trained a different model; fixed, every generator is restored.

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

Every stage: tests pass, and it runs end to end in `GlobalModel`, `CombinedModel` and the dual-decoder variant. Stage-specific
checks: P on the *built* model (mask reach equals the local module's depth); 0 harness IDENTICAL with PEs off and `pad_like`
reproducing `pad_batch`'s layout; naive translation-invariant; sinusoidal the closed form; lap equal to PyG's
`AddLaplacianEigenvectorPE` up to sign; rw equal to PyG's `AddRandomWalkPE`; distance and spectral laid out like the mask, zero
at init, finite on the dense-graph NaN test at hops 1–3; RoPE logits depending on the offset only.

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
- **synth_data_0 (legacy) at radius 30 is fragmented** (mean degree 4.3, at the percolation threshold): 31% of cells get an
  all-zero LapPE at k = 8. A kNN-6 graph is connected. Use a kNN graph for any spectral encoding on point-process data.
- `pe.spectral.k` defaults to 32 (shortest wavelength ~1.6 mm on synth_spot); the 1 mm bands of the
  sparse programmes may want 64.
- **Gene ranks from unmasked input (deferred by the user, 2026-10-10).** `calculate_gene_ranks`
  scores `get_model_output` predictions, which see every entry. A gene-masking model was only
  trained on hidden entries, so these ranks compare decoders on input they were never trained on.
  The user wants the variant that ranks genes on hidden entries only, later.
