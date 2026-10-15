# Staged plan: why the global model struggles to learn, and three ways to find out

The global component (the transformer over every cell of a slide) is meant to learn what the local
component (the GCN over a cell's neighbourhood) cannot. Four rounds of sweeps on synth_spot and one
on legnini have not shown that it does, beyond what a position-aware average of nearby visible
cells already gives. This plan collects what we know, the problems and their suspected causes, and
stages three experiments that separate those causes:

1. **The global model alone** (no local component): can the transformer learn the task at all?
2. **Other local embeddings** under the same global model (scVI, PCA, frozen, shallow GCN): is the
   GCN's embedding the bottleneck?
3. **Separate training** (local first, then global, optionally alternating): is joint training of
   the dual structure the problem?

Builds on [`attention_plan.md`](attention_plan.md) (rounds 1–3b, the levers, the evaluation),
[`PE_plan.md`](PE_plan.md) (the encodings) and [`background.md`](background.md) (why the
local/global split and the dual decoder exist). Code paths are relative to `src/interscale/`. Same
rules as the other plans: update the status table in the same commit as the work, and a stage is
`done` only when something external verified it; "implemented, unverified" is a real state.
Nothing runs locally: verification is unit tests on toy modules and synthetic arrays, and the
user runs smoke, sweeps and evaluation on the cluster.

## Status

Last updated 2026-10-15.

| stage | state | verified by | date |
|---|---|---|---|
| 0a — head scores for single-head models (GlobalModel, LocalModel) | **implemented, unverified** — `HeadPredictions.heads` is a dict; a dual decoder gives `local` and `global`, a GlobalModel or single-decoder CombinedModel `global`, a LocalModel `local`, each the step's own output in the step's row order, with no hidden value reaching it; `score_heads` writes `global - local` columns only when both exist, under the same column names as before (the 3b script reads them unchanged). 1 mutation, caught | 2026-10-15 |
| 0b — reference predictors in the evaluation (slide mean, kernel smoother, nearest visible) | **implemented, unverified** — `masked_probing.reference_predictions`: `ref_slide_mean`, `ref_label_mean` (optional label key, e.g. an oracle region), `ref_kernel_<s>um`, `ref_nearest_<k>`, each the weighted mean of the *visible entries* of the same graph (per entry, so it works under cell, entry and block masking), never the cell itself; `collect_head_predictions(..., references=...)` adds them as heads gathered by the model's own row index and checks that both saw the same hidden entries. Exact on a constant per slide (slide mean) and on smooth fields (kernel, nearest beat the slide mean 5×), ~0 centred cosine on noise, graphs never mix, perturbing hidden values changes nothing, no kernel overflow at 5 mm, rows aligned on subsampled graphs. 8 mutations, each caught; 736 tests pass; harness IDENTICAL | 2026-10-15 |
| 0c — calibrated magnitude in the evaluation | not started, **deferred until the user decides** | | |
| A1 — learned linear input for the global-only model | **implemented, unverified** — `type_gex_embedding: Linear` builds `input_embedding` (an `nn.Linear`; the transformer's input LayerNorm normalises it), trained with the transformer; training maps the masked batch and inference (`GlobalModel.get_model_output`) the clean one, both through `GlobalModule.gex_embedding`; hidden cells' values never reach a prediction, visible ones do; the validator accepts only buildable inputs; `emb-linear` in the global-only checkpoint name only. 4 mutations, each caught; 735 tests pass; harness IDENTICAL (its GlobalModel/PCA and LocalModel cases included). **Not run on data** | 2026-10-15 |
| A2 — global-only sweep (input × encoding) and a local-only reference | not started | | |
| B1 — local-embedding variants under one global model (scVI joint, scVI frozen, frozen random GCN, 1-layer GCN) | not started | | |
| B2 — embedding sweep | not started | | |
| C1 — per-head loss weights and module freezing | not started | | |
| C2 — staged training driver (phases: local → global, fine-tune, alternating) | not started | | |
| C3 — staged-training sweep | not started | | |

## What we know (rounds 1–3b)

Detailed numbers are in `attention_plan.md`, "Why attention stays uniform, and what to change", and
in the round-results sections there. In short:

- **Rounds 1 and 2** (240 optimizer steps, raw SCE): every run of every backend and encoding
  converged to the same function at the gene-mean floor (val SCE 0.0834 against 0.0816).
  Attention stayed uniform even with the long-range mask off. W_q/W_k never left their
  initialisation.
- **Diagnosis** (2026-10-12):
  1. The tokens are nearly identical: 84% of each spot is the shared mean profile.
  2. Training was tiny.
  3. The task never needed the global model: gene masking leaves neighbours visible, raw SCE is
     ~85% satisfied by the mean profile, and the dual decoder lets global copy local.
- **legnini, round 2:** the same single function, at 0.141, while a plain average of visible
  neighbours within 55 µm reaches 0.122. The model underfits; the signal is there.
- **Round 3a** (4,800 steps, centred queries/keys, centred targets): the model now learns beyond
  the mean profile (0.0701 against the 0.0816 floor). The schedule did most of it, centred targets
  the rest. Centred targets made the tokens differ (cosine 0.95 → 0.70) and attention left uniform.
  The encodings still did not matter. Global and local probes improved together; `local+global`
  never beat `local`.
- **Round 3b**, valid rerun (500 µm discs over 30% of each slide as the only targets, residual
  global decoder, long-range mask; one seed):
  - The local head is blind beyond its reach: centred cosine 0.33–0.35 within 290 µm, 0.05 at
    290–500 µm, about 0 beyond.
  - The global head recovers 0.21 there without an encoding, which equals a slide-mean predictor.
    With RoPE or Fourier it reaches 0.29–0.31, which equals a Gaussian kernel of visible spots
    (0.30–0.33). **It never beats that smoother**: equal on slide- and region-level programmes,
    below on `long_range` (0.70 vs 0.77), `spatially_variable` (0.64 vs 0.79) and
    `sparse_response` (0.08 vs 0.30).
  - Fine context (the R3/R4 sender state) is at chance. Attention shows no preference for planted
    senders.
  - The residual decoder and the long-range mask did not help. Global's magnitude is off: R² −1.4
    to −3.4 on the high-variance programmes while the directions are right, because the centred
    SCE is scale-free.
- **One bug cost a round:** discs with `mask_percentage: 0` left the input unmasked
  (fixed in `b7cf317`). Every new masking or training mode needs a test through the real
  model and trainer that hidden values cannot reach the predictions.

## The problems, and what we believe causes them

| # | problem (observed) | suspected causes | experiment that tests it |
|---|---|---|---|
| P1 | Global never adds more than a smoother of visible cells; fine and long-range context is not learned | (a) the transformer cannot learn it from these tokens and this much data; (b) its tokens lack what it would need; (c) joint training steers it to the easiest solution | A (a), B (b), C (c) |
| P2 | The global component only sees GCN outputs | The GCN's embedding is shaped by the *local* loss: it keeps what reconstructs a cell from its neighbourhood and may drop what a distant cell would need. In the blind area every token is the same constant (all-masked input → same GCN output) | A removes the GCN; B swaps it |
| P3 | Global copies local, or its gain stays small | Dual decoder on one target: copying is optimal (round 3a). Joint gradients: the GCN also receives the global loss through the tokens, so local and global compete for the same parameters. The local loss is the easier one and is minimised first | C: train local first and freeze it, so global gets the residual problem with a fixed input |
| P4 | Global's magnitude is wrong | Centred SCE is scale-free | 0c (evaluation) now; a scale-aware loss later if 0c shows it is only scale |
| P5 | Attention never selects planted senders | The task does not reward it (a smoother suffices for the disc targets), or the transformer cannot find sparse pairs among ~2,500 keys with this little signal | A with relative encodings: any selectivity must come from attention when there is no GCN |
| P6 | Single seeds, small differences | Effects of ±0.02 between arms are unresolved | 3 seeds for the key arms of every sweep |

Hypotheses, stated so each stage can reject one:

- **H-A (the transformer itself):** the transformer alone, end to end on raw expression with
  relative encodings, cannot exceed the smoother either. Then the limit is the model or the data
  and objective, not the dual structure.
- **H-B (the embedding):** with an embedding not shaped by a neighbourhood loss (scVI, PCA,
  learned linear), global gets further than with the GCN's.
- **H-C (joint training):** with the local component trained first and frozen, global learns more
  (deep-field cosine above the smoother, or fine programmes recovered) than in joint training.

## Shared setup for every sweep

- **Testbed:** synth_spot, round 3b's disc task (node masking, rate 0, discs of 500 µm over 30%),
  with round 3a's settings: centred queries/keys, centred targets, batch size 1, 200 epochs.
  Dense attention, the long-range mask on (`lrm`) where a local component exists. Encodings `none`,
  `rope` and `fourier` (3b: position matters, and `none` is the slide-mean control). One
  gene-masking arm per sweep for continuity with 3a.
- **Evaluation:** the 3b script's, with the same seeded discs for every arm. Per head: centred
  cosine and raw SCE by depth, per programme, per gene, and the probes. Every run is compared with
  the reference predictors (stage 0b) on the same hidden cells; the bar is the kernel smoother,
  not zero. The question is the deep field, beyond 290 µm, where no visible cell is within the
  GCN's reach.
- **Seeds:** one seed per arm to screen; then 3 seeds for the arms a decision rests on.
- **Checkpoint names:** every new switch gets a tag (embedding type, schedule phase) before its
  sweep, or arms overwrite each other.

## Stage 0 — evaluation that works for every model

### 0a — head scores for single-head models

`masked_probing.collect_head_predictions` requires a dual decoder (`predict_local`). Generalise it
to return a dict of heads: `{"local", "global"}` for a dual decoder, `{"global"}` for a
`GlobalModel`, `{"local"}` for a `LocalModel`, from each module's `_common_step` output.
`score_heads` scores every head it is given; `global - local` columns only when both exist.

Tests:
- each model type returns its heads row-aligned with the depth (subsampled graphs, as now);
- hidden values do not reach the predictions of any model type;
- the dual-decoder output is unchanged (the existing tests).

### 0b — reference predictors

The comparisons that put 3b in context (slide mean 0.21, region mean 0.27, kernel 400 µm 0.33)
were computed by hand. Move them into the library (`evaluation.masked_probing.reference_predictions`):

- for each hidden cell, from the visible cells of its own graph, predict the slide mean, the mean
  of the k nearest visible cells, and Gaussian kernels at given bandwidths;
- score them through `score_heads` as extra "heads".

The script then reports, per depth bin and programme, each model head next to the best reference.
Dataset-agnostic: bandwidths and k are arguments; a region-mean oracle needs a label key, which
the script passes (none for legnini).

Tests: on planted fields (a constant per slide, a smooth gradient, white noise), the slide mean
wins on the constant, the kernel on the gradient, and nothing beats 0 on the noise; graphs never
mix; the reference never reads a hidden cell.

### 0c — calibrated magnitude (deferred)

Fit one scale per head on the training slides' hidden cells, apply it on the test slides, and
report R² and raw SCE before and after. Separates "right direction, wrong size" from "not
learned". Deferred: the user asked to find out why global struggles first. Build it if any sweep
below is read on raw SCE or R².

## Stage A — the global model alone

**Question:** can a transformer, given every cell's masked expression and its position, fill in
hidden discs better than a smoother of visible cells? This removes P2 and P3 entirely.

### What exists

`GlobalModel` wraps a `GlobalModule` without a local component (`local_mask_hops = 0`, so the
long-range mask has nothing to block and warns). Its tokens come from `create_gex_embedding`:

- `type_gex_embedding: PCA`: sklearn PCA fitted once on the **first masked training batch** and
  frozen; a fixed, unlearned projection that was fitted on −1 fills.
- `NMF`, which cannot take −1 inputs.
- `Precomputed`, which reads `obsm[latent_obsm_key]` attached to the graphs.

There is no learned input map, so the transformer cannot shape its own tokens. Note on PCA: it is
fitted on the first batch the module sees, and Lightning's sanity validation runs before training,
so in practice it is fitted on the first (masked) **validation** slide, not a training batch. Left
as is (changing it would change existing runs); `pca_rope` is read as "a fixed linear projection".

### A1 — a learned linear input

`type_gex_embedding: Linear`: an `nn.Linear(n_input, n_embed)` plus LayerNorm on the masked
expression, trained end to end with the transformer. This is the fair global-only model: the same
number of trainable input parameters as the GCN's input projection, but no neighbourhood mixing.

- Config value, validator (regression, node level), checkpoint tag (`emb-linear`), and the
  `get_model_output` path (`GlobalModel` calls `create_gex_embedding` directly; route it through
  the module so the learned map is used).
- Tests:
  - the map is trained (gradient reaches it);
  - hidden cells' values do not reach the predictions (all-masked rows map to one constant token);
  - inference uses the same map;
  - PCA, NMF and Precomputed are unchanged (harness IDENTICAL).

### A2 — the global-only sweep

| arm | input | encoding | notes |
|---|---|---|---|
| `lin_none` | Linear | none | control: the blind area should land at the slide mean |
| `lin_rope` | Linear | RoPE | main arm |
| `lin_fourier` | Linear | Fourier bias | main arm |
| `pca_rope` | PCA (frozen) | RoPE | does a learned input matter? |
| `lin_rope_gene` | Linear | RoPE | gene masking at 0.3, as 3a |

Plus two references trained with the same settings:

- **`LocalModel` (GCN only):** the local head without any global component. Is the dual model's
  local head the same?
- **The 3b joint `rope` arm**, already run.

Readouts:
- the global head by depth against the kernel smoother and against 3b's joint global head;
- attention: effective keys and span, and whether the deep-field queries attend to the nearest
  visible ring (attention mass by key depth);
- planted edges.

**Predictions.** If H-A holds, `lin_rope` lands at the smoother (~0.30 deep) like 3b's global.
If global-only beats 3b's global in the deep field, the dual structure was holding it back (go
to C). If it is far below the smoother, the transformer cannot even learn smoothing from these
tokens, and the earlier "global = smoother" came from the GCN's tokens.

Cost: 5 arms + 1 reference, ~the 3b sweep's runtime each.

## Stage B — other local embeddings under the same global model

**Question:** is the GCN's embedding what limits the global component? All arms keep the dual
decoder and joint training, so only the embedding changes.

### What exists

- `local_component.name: SCVI`: `SCVILocalModule`, the scvi-tools `Encoder` on each cell alone
  (the graph is ignored), returning a sampled `z`. A KL term is added through
  `compute_separate_losses` (`kl_loss`).
  - Two caveats to fix first: it embeds the **sample** `z` also in evaluation (the embedding is
    noisy at test time; use `q_m` when not training), and its input is our `log1p_norm` layer,
    not counts, with a normal latent and no likelihood decoder. It is an scVI-style encoder, not
    scVI.
- `Precomputed` local module: exists, but its `from_config` branch is commented out. Global-side
  `Precomputed` (from `obsm`) works.

### B1 — variants

1. **`SCVI` joint:** the encoder as above, with `q_m` at evaluation. Per-cell, so a hidden cell's
   embedding is one constant: the local head is blind at every depth, and the global head must do
   everything from context.
2. **scVI frozen, pretrained:** train scvi-tools' `SCVI` on the training slides' raw counts once,
   with the evaluation's masking regime so a hidden cell gets the embedding of an all-masked
   input. Write the latent mean to `obsm`, and feed it through `Precomputed`.
   - Enable the `Precomputed` local branch, so the dual decoder still has a local head.
   - Leakage guard: the latent of a **hidden** cell must not be computed from its true counts. In
     the precomputed path the masking happens after the embedding is read, so the hidden cells'
     latents must be recomputed from masked input every epoch, or replaced by one constant "masked"
     latent. Test this explicitly (the `b7cf317` lesson).
3. **GCN with frozen random weights:** a random 2-layer GCN as a fixed feature map. If global does
   as well on it, training the GCN adds nothing for global.
4. **GCN, 1 layer:** a reach of ~141 µm instead of ~283 µm. Global's share of the blind area grows,
   and the GCN smooths less.

Each gets a checkpoint tag and a validator entry (e.g. `Precomputed` needs `latent_obsm_key`).

### B2 — the embedding sweep

Arms: `gcn` (3b reference, rerun with the new evaluation), `scvi_joint`, `scvi_frozen`,
`gcn_random`, `gcn_1layer`, all with RoPE; `scvi_joint_none` as the no-position control.

Readouts: as A2. The key comparison is each arm's global head in the deep field against `gcn`'s
and against the smoother. Also compare local heads at depth ≤ 290 µm: GCN variants see
neighbours, scVI ones do not.

**Predictions.** If H-B holds, `scvi_*` or `gcn_random` give global a higher deep-field cosine than
`gcn`. If every embedding lands at the smoother, the embedding is not the limit.

## Stage C — separate training of local and global

**Question:** does training both components jointly hold the global component back? Joint
training lets the GCN absorb the global loss through the tokens, and the easy local solution wins
first. Separate training gives global a fixed input and the residual problem.

### C1 — per-head loss weights and freezing

- **`optim.loss_weights: {local: 1.0, global: 1.0}`.** The dual decoder's loss is now one masked
  loss over the stacked `[2N]` predictions, which equals the mean of the two heads' losses. Make
  it `w_local · L_local + w_global · L_global`, computed from `compute_separate_losses`, which
  already returns both. With the defaults it must equal the current loss (harness IDENTICAL).
- **`optim.freeze: []`**, a list of `local` / `global`: those submodules get
  `requires_grad = False` and are kept in eval mode (dropout off) for the whole fit. Lightning calls
  `module.train()` every epoch, so a callback re-applies eval to frozen parts at every train epoch
  start. The optimizer is built from trainable parameters only (it already filters on
  `requires_grad`).
- Tests:
  - frozen parameters are bit-identical after several steps, and dropout is off in them;
  - the default weights reproduce the current loss to the bit;
  - with `w_local = 0`, no gradient reaches the local decoder;
  - with the local component frozen, the global loss gives no gradient to the GCN.

### C2 — the staged training driver

`train.staged_train(model, datamodule, phases)`, a sequence of ordinary `model.train()` calls in
one process, each phase with its own epochs, freeze list, loss weights, learning rate and warm-up.
Weights carry over in memory, so no checkpoint naming problem between phases. Only the last phase
saves a checkpoint, under a schedule tag.

Schedules, in config (`optim.schedule`, a list of phases):

| schedule | phases |
|---|---|
| `joint` | one phase, nothing frozen (today) |
| `local_then_global` | (1) local only: global frozen, `w_global = 0`; (2) global only: local frozen, `w_local = 0` |
| `local_then_global_ft` | as above, then (3) joint fine-tuning at 0.1× learning rate |
| `alternating_k` | (1) local only, then k cycles of (global only, local only) with `w_global = 0` in the local phases, so the GCN never learns from the global loss |
| `alternating_k_coupled` | as `alternating_k`, but the local phases keep `w_global = 1`: the GCN may adapt its tokens to what global needs, while global is frozen |

- **Phase 1 can also be a plain `LocalModel`,** whose module has the same keys as the dual
  decoder's `local_module` (both come from `LocalModule.from_config`). It is copied in with
  `load_state_dict(strict=True)`, so the local-only reference of A2 doubles as phase 1. Test the
  key match.
- Each phase logs to the same W&B run with a `phase` prefix, and the efficiency monitor and probes
  keep working.
- Every phase gets the mask redraw callback (`redraws_masks`). Test that hidden values cannot
  reach the predictions in every phase.
- Epoch budget: the same total as the joint run (200), e.g. 100 + 100 for `local_then_global`, so
  the comparison is not about more steps.

### C3 — the staged-training sweep

Arms, all RoPE, discs: `joint`, `local_then_global`, `local_then_global_ft`, `alternating_4`,
`alternating_4_coupled`, plus `local_then_global` with the residual decoder off.

Readouts:
- as A2, per phase at its end, the deep field above all;
- the GCN's drift between phases (relative weight change), and the token cosine.

**Predictions.** If H-C holds, `local_then_global`'s global head beats `joint`'s in the deep field,
and the coupled alternation either helps (the GCN learns tokens that serve global) or collapses
back to the joint result.

## Order and decision points

1. **Stage 0a + 0b** (needed to read any of A–C) and **A1**. Then sweep **A2**; it is the cheapest
   test of H-A.
2. **C1 + C2**, built while A2 runs, then sweep **C3**. C needs no new embeddings.
3. **B1 + B2**: the scVI variants are the most work and carry a leakage risk, so they come last
   unless A2 shows global-only far above the joint global (then the embedding is suspect first).
4. After each sweep, 3 seeds for the arms a decision rests on.

| outcome | then |
|---|---|
| A: global-only ≈ smoother | The limit is the task or model, not the dual structure: change what global is asked (targets a smoother cannot get; see attention_plan.md for ideas) |
| A: global-only > joint global | The dual structure holds it back: C first |
| C: staged > joint | Make staged training the default for global experiments; revisit encodings and backends under it |
| B: another embedding > GCN | Investigate what the GCN's embedding drops; consider a skip of raw expression into the tokens |

## Invariants

- Hidden values never reach any prediction, for every model type, embedding and phase. A test
  through the real model and trainer, not just the datamodule (`b7cf317`).
- Every arm is scored on the same seeded discs, and every run reports the reference predictors
  next to its heads.
- Default configs train bit-identically (`scripts/equivalence_harness.py` IDENTICAL after every
  library change).
- Checkpoint names differ between arms: every new switch has a tag.
- Evaluation code stays dataset-agnostic: label keys, bandwidths and gene groups come from the
  caller.

## Open questions

- Precomputed scVI and masking: a constant "masked" latent for hidden cells is simplest, but
  differs from what the joint encoder sees. Decide at B1.
- Whether the deep-field metric should weight cells by depth (now: equal per cell within each bin).
- Whether staged training should restart the learning-rate schedule per phase (planned: yes, each
  phase its own warm-up and cosine).
- legnini after synth_spot: legnini has no planted truth, but its 55 µm neighbour average (0.122)
  is a hard reference the global-only model can be compared with.
