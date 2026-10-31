# Staged plan: why the global model struggles to learn, and three ways to find out

The global component (the transformer over every cell of a slide) is meant to learn what the local
component (the GCN over a cell's neighbourhood) cannot. Four rounds of sweeps on synth_spot and one
on legnini did not show that it does, beyond what a position-aware average of nearby visible
cells already gives. This plan collected what we knew, the problems and their suspected causes, and
staged three experiments that separate those causes:

1. **The global model alone** (no local component): can the transformer learn the task at all?
2. **Other local embeddings** under the same global model (scVI, PCA, frozen, shallow GCN): is the
   GCN's embedding the bottleneck?
3. **Separate training** (local first, then global, optionally alternating): is joint training of
   the dual structure the problem?

Builds on [`attention_plan.md`](attention_plan.md) (rounds 1–3b, the levers, the evaluation),
[`PE_plan.md`](PE_plan.md) (the encodings) and [`background.md`](background.md). Code paths are relative to
`src/interscale/`. Same rules as the other plans: update the status table in the same commit as the work; a stage is
`done` only when something external verified it. Nothing runs locally.

Trimmed 2026-10-31: test and mutation counts, the step-by-step specs of built stages and the sweep set-up notes were
removed (git history has them; CLAUDE.md has the invariants). **Where it led:** the disc task cannot show communication;
programme masking can (below), and the work continued in roadmap.md (T1, T2) and decoder_plan.md.

## Status

Last updated 2026-10-31.

| stage | state | date |
|---|---|---|
| 0a — head scores for single-head models | implemented, in use | 2026-10-15 |
| 0b — reference predictors (slide mean, label mean, kernel smoother, nearest visible) | implemented, in use | 2026-10-15 |
| 0c — calibrated magnitude | **deferred** (raw SCE is at the smoother's) | |
| 0d — masked-pass evaluation: probes by depth with a smoother set, attention by query, untrained twin | implemented, in use | 2026-10-17 |
| 0e — receptive field and representations | implemented, in use | 2026-10-17 |
| 0f — one results file per sweep (`evaluation.report`) | implemented, in use | 2026-10-17 |
| 0g — position baseline, gene groups by depth, blend test | implemented, read 2026-10-18 | 2026-10-18 |
| P — programme masking: responses predicted from their drivers | **run and read** (2026-10-19/20, 3 seeds): the first task on which a global model beats the bar | 2026-10-20 |
| F — patch flow rewrite, against the expression-permuted slide | implemented; read on legnini (attention_plan.md stage 9); superseded for whole sections by `evaluation/flow_maps.py` | 2026-10-18 |
| A1 — learned linear input for the global-only model | implemented; read in A2 (needs a position encoding) | 2026-10-18 |
| A2 — global-only sweep (input × encoding) and a local-only reference | **run and read 2026-10-18**: H-A holds (global-only at the smoother) | 2026-10-18 |
| B1 — local-embedding variants under one global model (scVI joint, frozen random GCN, 1-layer GCN) | not started; **parked** (B2 and C3 showed the embedding is not the limit) | |
| B2 — embedding sweep (precomputed PCA, scVI) | **run and read 2026-10-18**: H-B not supported | 2026-10-18 |
| C1 — per-head loss weights and module freezing | implemented, in use (`optim.loss_weights`, `optim.frozen`) | 2026-10-17 |
| C2 — staged training driver | implemented (`optim.schedule`) | 2026-10-17 |
| C3 — staged-training sweep | **run and read 2026-10-18** (7 of 9 arms): H-C not supported. `l2g_res` and the six missing arms not evaluated (C3's leftovers: drop proposed, roadmap decision 4) | 2026-10-18 |
| guard — hidden cells' precomputed embeddings (`precomputed_mask_fill`: learned / −1) | implemented | 2026-10-17 |

## What we knew before this plan (rounds 1–3b)

Details in attention_plan.md, "Why attention stays uniform, and what to change".
- **Rounds 1 and 2** (240 steps, raw SCE): every run converged to one function at the gene-mean floor; attention uniform.
- **Diagnosis:** near-identical tokens, too few steps, and a task that never needed the global model.
- **legnini, round 2:** the same single function (0.141) while a neighbour average within 55 µm reaches 0.122: underfitting.
- **Round 3a** (4,800 steps, centred queries/keys and targets): the model learns beyond the mean profile.
- **Round 3b**, valid rerun (500 µm discs, residual decoder, long-range mask): the local head is blind beyond its reach; the
  global head recovers what a slide mean (no encoding) or a Gaussian kernel of visible spots (RoPE, Fourier) gives, and never
  beats that smoother. The residual decoder and the long-range mask did not help.
- **One bug cost a round:** discs with `mask_percentage: 0` left the input unmasked (`b7cf317`). Every new masking or training
  mode needs a test through the real model and trainer that hidden values cannot reach the predictions.

## The problems, and what we believe causes them

| # | problem (observed) | suspected causes | experiment | outcome |
|---|---|---|---|---|
| P1 | Global never adds more than a smoother of visible cells | (a) the transformer cannot learn it; (b) its tokens lack it; (c) joint training steers it to the easiest solution | A (a), B (b), C (c) | none of the three: the disc task is solved by a smoother; programme masking passes the bar |
| P2 | The global component only sees GCN outputs | The GCN's embedding is shaped by the local loss; in the blind area every token is one constant | A removes the GCN; B swaps it | global-only models land where the joint global head does |
| P3 | Global copies local, or its gain stays small | Dual decoder on one target; joint gradients | C | staged schedules gain nothing; separation needs structure (decoder_plan.md) |
| P4 | Global's magnitude is wrong | Centred SCE is scale-free | 0c | raw SCE at the smoother's: deferred |
| P5 | Attention never selects planted senders | The task does not reward it, or ~2,500 keys need a large logit lead | A with relative encodings | solved later by programme masking + a distance prior per head (roadmap T1) |
| P6 | Single seeds, small differences | | 3 seeds for key arms | seed axis in every sweep (roadmap T0) |

Hypotheses: **H-A** the transformer alone cannot exceed the smoother either (then the limit is the task, not the dual
structure); **H-B** a token not shaped by a neighbourhood loss takes global further; **H-C** local trained first and frozen
lets global learn more.

## Shared setup of the disc sweeps

synth_spot, round 3b's disc task (node masking at rate 0, discs of 500 µm over 30%), round 3a's settings (centred queries/keys,
centred targets, batch size 1, 200 epochs), dense attention, the long-range mask where a local component exists; encodings
`none` (the slide-mean control), `rope`, `fourier`. Every arm scored on the same seeded discs, each head by depth, per programme
and per gene, next to the reference predictors: **the bar is the kernel smoother, not zero**, and the question is the deep field
beyond 290 µm. One seed to screen, 3 for decisions. Every new switch has a checkpoint tag.

## Stage 0 — evaluation that works for every model

All in `evaluation/masked_probing.py` and its neighbours; what each task answers and which old ones not to use is in
[linear_probing_docs.md](linear_probing_docs.md).
- **0a.** `collect_head_predictions` returns a dict of heads: `{"local", "global"}` for a dual decoder, `{"global"}` for a
  GlobalModel or single-decoder CombinedModel, `{"local"}` for a LocalModel; `global - local` columns only when both exist.
- **0b.** `reference_predictions`: `ref_slide_mean`, `ref_label_mean` (optional label key), `ref_kernel_<s>um`,
  `ref_nearest_<k>`, each from the *visible entries* of the same graph, never the cell itself; scored as extra heads.
- **0c (deferred).** One scale per head fitted on the training slides' hidden cells; R² and raw SCE before and after. Build it
  if a sweep is read on raw SCE or R².
- **0d, 0e.** Every analysis reads the model under the evaluation discs, by depth, against a model-free bar: probes with a
  `smoother` feature set, `attention_by_query` against uniform attention, planted pairs against distance-matched cells, the
  untrained twin; `receptive_field` (gradient of a hidden cell's prediction with respect to every visible cell's input) replaces
  the degenerate relevance; `representations` (CKA, participation ratio). Caveat: on the 100 µm lattice, k-NN 8 edges reach
  283 µm at slide edges, so the GCN is blind beyond 290 µm only in the interior.
- **0f.** `evaluation.report`: one `results.md` per sweep, a cross-arm summary on top, one section per arm and analysis.
- **0g.** `position_features` (offset from the slide centroid, its norm, one-hot patches) and the `smoother+position` feature
  sets; scores by gene group and depth; `blend_heads` (a head averaged with a reference).

## Stage A — the global model alone

**Question:** can a transformer, given every cell's masked expression and its position, fill in hidden discs better than a
smoother? This removes P2 and P3. **Purpose (the user, 2026-10-17):** simplify the model for finding bugs, remove every influence
of the local model, let the global model learn on its own. A fixed input serves all three.

**Tokens of a GlobalModel** (`type_gex_embedding`): `PCA` (sklearn, fitted once on the first batch the module sees -- the first
masked *validation* slide, because of Lightning's sanity check; `pca_fit: train` fits it on the training slides), `NMF` (cannot
take the −1 fill), `Precomputed` (from `obsm[latent_obsm_key]`), `Linear` (A1). **Design principle (the user, 2026-10-17):** other
embeddings (scVI, PCA, ...) are computed outside InterScale, stored in `obsm` and passed in through `Precomputed`; the pipeline's
only job with them is to hide the embeddings of hidden cells.

**A1 — a learned linear input** (`type_gex_embedding: Linear`): `input_embedding = nn.Linear(n_input, n_embed)` trained with the
transformer; training maps the masked batch, inference the clean one, both through `GlobalModule.gex_embedding`. It is the GCN's
skip branch (`input_proj`) without the message-passing branch, so GCN-only, joint and global-only arms get the same per-cell input
and their context in only one way. Without a position encoding every hidden spot gets the same token (`b − W·1`) and one shared
prediction: the slide-mean control. Checkpoint tag `emb-linear_` (global-only names only).

**Precomputed embeddings of hidden cells** (`model.global_component.parameters.precomputed_mask_fill`; CLAUDE.md, "Masking"):
`apply_mask` overwrites only `x`, so a hidden cell used to hand the transformer an embedding of its own true expression.
`GlobalModule.hide_embeddings` now replaces it in every masked pass by the fill: `learned` (one trained vector, from a private
generator) or `minus_one`. Gene masking is refused with `Precomputed` (except a recorded embedding under programme masking,
roadmap T0). Tag `emb-pre-<key>-learned_` / `-m1_`; Precomputed checkpoints from before 2026-10-17 leaked.

**Why any constant fill works (correction of 2026-10-17).** Every token first goes through the transformer's input LayerNorm,
which removes its mean and scale, so **any constant fill (−1, 0, 5) comes out as the LayerNorm's bias vector**, which no real
embedding becomes. `minus_one` makes the hidden token that bias (shared with every token's offset), `learned` gives it a direction
of its own. −inf is no option: it turns into NaN at that LayerNorm, and blocking hidden keys would still leave a hidden cell's own
token in its residual stream. A side effect: a per-cell offset or scale in a precomputed embedding (e.g. library size) never
reaches the transformer.

## Stage B — other local embeddings under the same global model (parked)

**Question:** is the GCN's embedding what limits the global component? B2 ran the precomputed PCA and scVI as GlobalModel inputs
(below); B1's variants were not built. What they would have been, if the question comes back:
- `SCVI` joint (`SCVILocalModule`: the scvi-tools `Encoder` on each cell alone; caveats: it embeds the sample `z` also in
  evaluation -- use `q_m` -- and it reads `log1p_norm`, not counts, so it is scVI-style, not scVI);
- a GCN with frozen random weights (C3's `frozen_local` covered it: global validation loss 0.571, the precomputed PCA's);
- a 1-layer GCN (reach ~141 µm).
The `Precomputed` local module exists but its `from_config` branch is commented out.

## Stage C — separate training of local and global

**Question:** does joint training hold the global component back? Invariants in CLAUDE.md, "Staged training".
- **C1** (`optim.loss_weights`, default 0.5 / 0.5 = the old stacked loss, same code path; `optim.frozen`): a head with weight 0
  sends no gradient; frozen components are out of the optimizer and in eval at every training step; tags `lw<l>-<g>`,
  `frz-<components>`. The key is `frozen`, not `freeze` (a yacs `CfgNode` has a `freeze()` method).
- **C2** (`optim.schedule`: `joint`, `local_then_global`, `local_then_global_ft`, `alternating`, `alternating_coupled`, `custom`):
  phases in one process, the weights carried over, each phase its own optimizer, learning rate, warm-up and cosine, one W&B run
  with a continuing step, only the last phase checkpoints. Alternating schedules end on a global phase, so the final global head
  is never trained on stale tokens. `CombinedModel.copy_local_from(local_model)` takes a trained LocalModel as phase 1 (not wired
  into the scripts). Tags `sch-l2g`, `sch-l2gft`, `sch-alt<k>`, `sch-altc<k>`, `sch-custom-<hash>`.
- **C3** arms (RoPE, discs): `joint`, `joint_res`, `l2g_res`, `l2g_ft_res`, `alt4_res`, `altc4_res`, `l2g`, `frozen_local`,
  `global_loss_only`.

## Results of the disc sweeps (2026-10-18; one seed per arm)

Files: `/home/lehnerl/Arbeit/results/global_only_sweeps/<sweep>/results.md` (first read) and
`/home/lehnerl/Arbeit/results/global_only_new/<sweep>/results.md` (second read, the 0g analyses on the same models). 6,251 hidden
test spots; the deepest bin (>500 µm) holds 135, so differences under ~0.02 there are noise. Centred cosine on hidden spots:

| model | all | 290–500 | >500 |
|---|---|---|---|
| best smoother (Gaussian, 400 µm) | 0.465 | 0.448 | 0.413 |
| slide mean | 0.260 | 0.256 | 0.261 |
| joint (`joint_rope`), global head | 0.459 | 0.438 | 0.418 |
| joint, local head | 0.333 | 0.060 | −0.020 |
| global only, precomputed PCA + RoPE | 0.442 | 0.433 | 0.413 |
| global only, scVI + Fourier | 0.414 | 0.404 | 0.378 |
| global only, learned linear + RoPE | 0.384 | 0.383 | 0.370 |
| global only, in-pipeline PCA + RoPE | 0.309 | 0.305 | 0.272 |
| best staged schedule (`altc4_res`) | 0.451 | 0.436 | 0.416 |

**Verdict.** Nothing passes the best smoother on the disc task. H-B and H-C are not supported: every variant lands at global
validation loss 0.545–0.57, a random frozen GCN included. H-A holds.

What else it showed:
- The global head fills the GCN's blind area (0.438 vs 0.060 at 290–500 µm; untrained twins ~0.005); global-only models are flat
  with depth while smoothers drop.
- Near visible spots the global head mostly copies the local one (CKA 0.93 at the rim, 0.13 at 290–500 µm); beyond the reach it
  relays inward (attention 12× uniform on nearby hidden tokens, influence on the nearest visible ring): a two-step smoother.
- Joint training barely changes the GCN (local head 0.333 vs GCN alone 0.327); no schedule beats joint.
- Inputs: precomputed PCA > scVI > learned linear > in-pipeline PCA (fitted on one masked validation slide: misses slide- and
  condition-level genes, halves throughput). The learned linear input fails without RoPE (0.15). RoPE beats Fourier here.
- The residual decoder hurts the deep field (0.389 vs 0.438). Without RoPE, visible spots attend almost only to the identical
  hidden tokens (a sink).
- The only gene group where a model beats the best reference: response genes beyond the GCN's reach (scVI + Fourier +0.035);
  a blend of PCA + RoPE with the 400 µm kernel beats both deep (0.429 vs 0.413). The disc task cannot test short-range receivers
  (a hidden receiver loses its sender with it) or sparse responses (the receptor is hidden too). Hence programme masking.
- "Tokens know more than the predictions use" was mostly position (`smoother+position` lifts source distances 0.435 → 0.524).
- **Bugs found and fixed:** `val_masked_scaled_cosine_error` averaged both heads (now also `_{local,global}`); short-range
  planted-pair ranks were inflated by a ±50 µm tolerance on the 100 µm lattice (now 10 µm; suits the lattice, not irregular data).
- **legnini** (attention_plan.md stage 9): every arm at the slide mean (~0.50 vs the 40 µm kernel's 0.594).

## Programme masking: what drives a response (built 2026-10-18)

**Why (the user, 2026-10-18).** On the disc task a hidden receiver loses its sender and its receptor with it, and a smoother of
the visible spots explains most of what can be filled in.

**Task.** Hide every response gene (`var` role receiver or response) in every spot and nothing else (`dataset.mask_genes`,
`mask_percentage` 1.0; CLAUDE.md, "Masking"). A response can then only be predicted from what drives it: an adjacent sender, the
distance to a source, a distant source with the spot's own receptor. A smoother of the responses has nothing to average.

**Bars, fitted on the training slides** (`masked_probing.FittedReferences`, ridge, penalty on held-out training graphs):
`ref_own` (the spot's visible genes), `ref_own+context` (+ Gaussian-kernel means of the visible genes at 150, 400, 1000, 2500 µm:
**the bar**), `ref_own+context+position`; and since 2026-10-21 `ref_oracle` (a ridge on the true drivers, roadmap T1). A model has
learned more than fixed kernels of its inputs only where it beats `ref_own+context`.

**Readouts.** Centred cosine per programme and per interaction; the sender's attribution rank in its receiver's predicted response
among cells as far away (`response_influence`, integrated gradients from the training-mean cell since 2026-10-20; read against the
**untrained twin**, not 0.5, and only where `response_check` < 0.1); every programme with planted receivers also scored within them
(`heads_by_group_cells`); wins / ties / losses per programme against the bar (`PROGRAMME_BAR`, ties within 0.05).

**legnini.** The same machinery with `var['role']` written by the script: the dorsoventral identity code (15 genes) and the SHH
readouts PTCH1, GLI1, HHIP (positive control); legnini_plan.md, `legnini_programme.py` (not run).

### Results (2026-10-19/20; three seeds for the main arms, 100 epochs)

File: `/home/lehnerl/Arbeit/results/response_masking/results.md`. Seed sd 0.003-0.017 overall, so differences above ~0.03 are real.

| arm | overall | short-range receivers | long-range responses | sparse responses |
|---|---|---|---|---|
| bar `ref_own+context` | 0.682 | 0.616 | 0.692 | 0.796 |
| `local_gcn` | 0.510 | 0.586 | 0.471 | 0.767 |
| `lin_none` | 0.405 | 0.015 | 0.421 | 0.758 |
| `lin_rope` | 0.709 | 0.009 | 0.806 | 0.774 |
| `lin_fourier` (3 seeds) | 0.764 ± 0.003 | 0.011 | 0.893 | 0.782 |
| `joint_rope`, global head (3) | 0.748 ± 0.011 | 0.638 | 0.785 | 0.777 |
| **`joint_fourier`, global head (3)** | **0.820 ± 0.005** | 0.644 | 0.884 | 0.780 |
| `pre_pca_rope` / `pre_pca_fourier` | 0.340 / 0.419 | 0.012 / 0.009 | 0.344 / 0.479 | 0.592 / 0.624 |
| `lin_fourier_pool500` (3) | 0.616 ± 0.017 | 0.168 | 0.788 | **0.069** |
| `joint_fourier_pool500`, global (local) head | 0.626 (0.498) | 0.249 (0.586) | 0.787 (0.457) | **0.082** (0.757) |

- **First robust positive.** `joint_fourier`'s global head is 0.138 above the bar over three seeds, carried by the rings (M7, M8:
  0.93 against 0.63 / 0.75). Fourier beats RoPE in every arm. Without position a global-only model reads only the cell itself
  (`lin_none` = `ref_own`).
- **Per programme it does not beat the bar overall:** equal weight over 25 programmes 0.628 against 0.685; 3 wins / 12 ties / 9
  losses.
- **Where it lost:** the mirror programmes (MC1, MC2, MC4, MC6, response copies the cell's own visible source: below `ref_own`;
  fixed by the raw own-cell skip, decoder_plan.md D3.0); M11 (0.54, bar 0.92; fixed by 300 epochs, roadmap K15); sparse threshold
  programmes (read within receivers); directional M9, M10 (0.08-0.18 everywhere; roadmap K19).
- **PCA of the visible genes is a poor token here** (0.34 / 0.42): its components keep variance (45% in noise genes) and drop the
  sparse receptor and sender signals.
- **Pooling at 500 µm** keeps the rings and long-range programmes, but a shared patch token cannot know which cell is the receiver:
  sparse responses collapse to 0.07 (the own-cell skip fixed it, roadmap T1b). Peak GPU memory 3.5-5.5× smaller at S = 2,500.
- **Time.** Training 2-3 min per run, evaluation 30-60 s; the 19-run sweep is minutes as an array.

What came next is in roadmap.md: the base model became the windowed global-only `lin_dist_skipraw` (T1), the decomposition the
additive decoder with the context query (T2).

## Invariants

- Hidden values never reach any prediction, for every model type, embedding and phase. A test through the real model and
  trainer, not just the datamodule (`b7cf317`).
- Every arm is scored on the same seeded masks, and every run reports the reference predictors next to its heads.
- Default configs train bit-identically (`scripts/equivalence_harness.py` IDENTICAL after every library change).
- Checkpoint names differ between arms: every new switch has a tag.
- Evaluation code stays dataset-agnostic: label keys, bandwidths and gene groups come from the caller.

## Open questions

- Whether the deep-field metric should weight cells by depth (now: equal per cell within each bin).
- legnini: no planted truth, but its 55 µm neighbour average (0.122) is a hard reference for any model there.
