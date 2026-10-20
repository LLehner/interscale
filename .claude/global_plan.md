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

Last updated 2026-10-18 (A2, B and C3 run and read twice; 0g built; programme masking P built).

| stage | state | verified by | date |
|---|---|---|---|
| 0a — head scores for single-head models (GlobalModel, LocalModel) | **implemented, unverified** — `HeadPredictions.heads` is a dict; a dual decoder gives `local` and `global`, a GlobalModel or single-decoder CombinedModel `global`, a LocalModel `local`, each the step's own output in the step's row order, with no hidden value reaching it; `score_heads` writes `global - local` columns only when both exist, under the same column names as before (the 3b script reads them unchanged). 1 mutation, caught | 2026-10-15 |
| 0b — reference predictors in the evaluation (slide mean, kernel smoother, nearest visible) | **implemented, unverified** — `masked_probing.reference_predictions`: `ref_slide_mean`, `ref_label_mean` (optional label key, e.g. an oracle region), `ref_kernel_<s>um`, `ref_nearest_<k>`, each the weighted mean of the *visible entries* of the same graph (per entry, so it works under cell, entry and block masking), never the cell itself; `collect_head_predictions(..., references=...)` adds them as heads gathered by the model's own row index and checks that both saw the same hidden entries. Exact on a constant per slide (slide mean) and on smooth fields (kernel, nearest beat the slide mean 5×), ~0 centred cosine on noise, graphs never mix, perturbing hidden values changes nothing, no kernel overflow at 5 mm, rows aligned on subsampled graphs. 8 mutations, each caught; 735 tests pass; harness IDENTICAL | 2026-10-15 |
| 0c — calibrated magnitude in the evaluation | not started, **deferred until the user decides** | | |
| 0d — masked-pass evaluation: probes by depth with a smoother set, attention by query, untrained twin | **implemented, unverified** — `collect_features` gathers each cell's depth and any extra feature set in probe row order (subsampled graphs included); `smoother_features` is the references side by side and never reads a hidden value; `probe_labels`/`probe_covariates` score by depth with their pooled rows unchanged, a single-class bin NaN; `score_heads(gene_metrics=("cosine",))`; `attention_by_query` reads the maps a masked step returns, gives exactly uniform attention 1 against every uniform reference and 0.5 on planted pairs, touches no generator; `attention_diagnostics(masked=...)` and `probe.attention_masked` (default on) for the callback. Script: relevance, gene ranks, loadings and R² dropped; untrained twin as `<arm>_untrained`. Checked on stub data with untrained toy models (all 7 arms, twins, headline, W&B summary, smoke readouts). Mutations: see 0e | 2026-10-17 |
| 0e — receptive field and representations | **implemented, unverified** — `receptive_field`: Hutchinson estimate within 10% of the exact squared Jacobian norm (correlation > 0.95), queries on subsampled graphs read their own row, a GCN reads exactly 0 beyond its hops while a transformer and a dual decoder's global head do not, planted pairs ranked on influence, no generator touched, parameters and `gex_embedding` restored; `representation_table`: CKA 1 under rotation and scale and low for unrelated sets, participation ratio counts directions, wide sets reduced. 15 mutations across 0d and 0e, each caught; full suite passes | 2026-10-17 |
| 0f — one results file per sweep | **implemented, unverified** — `evaluation.report`: `append_sections` appends an arm's compact tables as `## <arm> | <analysis>` sections (CSV, index columns counted in the fence) to `RESULTS_DIR/results.md`; `read_sections` reads them back, the later section winning; `write_report` rewrites the file with the cross-arm summary on top (counts and references once, gains over the smoother, untrained twins under their arms). `global_only.py` writes nothing else. 6 tests, 7 mutations caught; checked on stub data (7 arms + twins, 3,515 lines, summary in the first ~700) | 2026-10-17 |
| 0g — position baseline, gene groups by depth, blend test | **implemented, unverified** — `masked_probing.position_features` (offset from the slide centroid, its norm, one-hot patches per size; identical for a slide placed anywhere, no expression enters, cells beyond the grid clipped to its border), `combined_features`, `feature_sets` adds `smoother+position` and `smoother+position+<E>`; `score_heads` → `by_group_depth` (its `all` rows equal `by_group`); `blend_heads` (`blend_<head>`). Report: `heads_group_by_depth` sections, summary per group by depth (count, best reference, slide mean once; `write_report(groups=...)` limits the summary's group tables), label and distance summaries with the gain over `smoother+position`. Scripts: `GENE_GROUP_KEYS`, `SUMMARY_GENE_GROUPS`, `POSITION_PATCHES_UM`, `BLEND_REFERENCE` in `sweep_common.py`; all four sweeps set to `MODE = "sweep"` for the rerun. 9 tests, 8 mutations caught (a ninth showed a redundant sort, removed); 822 tests pass; checked end to end on a generated stub synth_spot (6 slides per condition, 20 × 20) with untrained GlobalModel, CombinedModel and LocalModel | 2026-10-18 |
| P — programme masking: responses predicted from their drivers | **implemented, unverified** — `dataset.mask_genes` (names or `var` selectors such as `role=response`) hides exactly that gene set in the selected cells (`mask_percentage` per cell, 1.0 every cell; discs select cells for it) under `mask_strategy gene`; validator, checkpoint tag `gs<rate>-<selectors>`, both entrypoints. `masked_probing.FittedReferences`: ridge references fitted on training graphs -- `ref_own` (the cell's visible genes), `ref_own+context` (+ kernel means of the visible genes around it), `ref_own+context+position` -- penalty chosen on held-out training graphs; `draw_masks(kind="gene_set")`; `receptive_field.response_influence`: the planted sender's rank in its receiver's predicted response genes (its programme's only) among cells as far, a cell counting through its visible entries. Script `slurm/response_masking.{py,yaml}` + sweep yaml (local_gcn, lin_none, lin_rope, lin_fourier, joint_rope), `EVAL_MASK = "gene_set"` in `sweep_common.py`. 12 tests + 1 report test; 10 mutations caught (one first survived and led to a test); 861 tests pass; harness IDENTICAL; smoke-trained and evaluated on a generated stub synth_spot. **Not run on data** | 2026-10-18 |
| F — patch flow rewrite: where a patch draws from, against the expression-permuted slide | **implemented, unverified** — `attention_flow.patch_flow` rewritten (attention_plan.md stage 9): gradient influence (own input left out by an unbiased half split) and attention rollout with the residual path, each against the same model on the slide with the visible cells' expression permuted; reach, own-patch share, KL and centre-of-mass shift against their permutation floors; flow fields, evenly spread source maps, label heatmaps. 11 tests (exact influence of a fixed map, its null at the floor; a content-gated map 6x above the floor; uniform attention its own null; self exclusion exact; rollout composition; visible entries only), 9 mutations caught (2 first survived and led to tests); 865 tests pass; both evaluation modes run on stubs. On in `legnini_attention.py` (100 µm, clusters) and `response_masking.py` (500 µm, region) | 2026-10-18 |
| A1 — learned linear input for the global-only model | **implemented, unverified** — `type_gex_embedding: Linear` builds `input_embedding` (an `nn.Linear`; the transformer's input LayerNorm normalises it), trained with the transformer; training maps the masked batch and inference (`GlobalModel.get_model_output`) the clean one, both through `GlobalModule.gex_embedding`; hidden cells' values never reach a prediction, visible ones do; the validator accepts only buildable inputs; `emb-linear` in the global-only checkpoint name only. 4 mutations, each caught; 735 tests pass; harness IDENTICAL (its GlobalModel/PCA and LocalModel cases included). **Not run on data** | 2026-10-15 |
| A2 — global-only sweep (input × encoding) and a local-only reference | **RUN 2026-10-18**, read: see "Results of the first sweeps". set up — `notebooks_and_scripts/slurm/global_only.{py,yaml}`, `global_only_sweep.yaml`; 7 arms validate with their model types, checkpoint names distinct; script checked on stub data (preflight, one W&B sweep per model type, stale-file guard, head tables and W&B summary for single- and dual-head arms). Evaluation switched to 0d/0e on 2026-10-17; the machinery moved to `slurm/sweep_common.py`, shared with the B and C3 sweeps | 2026-10-17 |
| B1 — local-embedding variants under one global model (scVI joint, scVI frozen, frozen random GCN, 1-layer GCN) | not started | | |
| B2 — embedding sweep | **RUN 2026-10-18** (PCA and scVI, separate sweeps `global_pca`, `global_scvi`), read: see "Results of the first sweeps". set up — two sweeps in `slurm/`, each computing its embedding itself whenever it reads the data (`PREPARE_ADATA`, also passed to `main_sweep(prepare_adata=...)`), 64 wide (= n_embed), fitted on the training slides: `global_pca.*` (PCA of log1p_norm into `obsm['X_pca']`) and `global_scvi.*` (scvi-tools SCVI on the raw counts, latent mean, into `obsm['X_scvi']`). Arms per sweep: learned fill + RoPE, -1 fill, no encoding, Fourier. Computed once per process and saved under `RESULTS_DIR/embeddings/`, so training, twins and a later evaluation job see the same embedding. Checked end to end on stub data | 2026-10-17 |
| C1 — per-head loss weights and module freezing | **implemented, unverified** — `optim.loss_weights` (default 0.5 / 0.5 = the old stacked loss, same code path, harness IDENTICAL) and `optim.frozen`; the weighted sum is what `<mode>_loss` reports; a head with weight 0 sends no gradient; frozen components are out of the optimizer, unchanged after steps, in eval at every training step, and get their own `requires_grad` back | 2026-10-17 |
| C2 — staged training driver (phases: local → global, fine-tune, alternating) | **implemented, unverified** — `optim.schedule` presets and `custom` phases; `train()` runs the phases with their epochs (exact split), learning rates, warm-ups and cosines; an end-to-end run on toy data trains each component only in its phases, carries the weights over, redraws masks every phase, checkpoints and saves once; one W&B run with a continuing step; `copy_local_from` takes a trained LocalModel as phase 1. 28 mutations across C1 and C2, each caught; 779 tests pass; harness IDENTICAL. **Not run on data** | 2026-10-17 |
| C3 — staged-training sweep | **RUN 2026-10-18**, 7 of 9 arms evaluated (`frozen_local` trained but not evaluated, `global_loss_only` did not run), read: see "Results of the first sweeps". set up — `slurm/staged_training.{py,yaml}`, `staged_training_sweep.yaml`: 9 arms of A2's `joint_rope` model: `joint`, `joint_res`, `l2g_res`, `l2g_ft_res`, `alt4_res`, `altc4_res`, `l2g`, `frozen_local` (GCN frozen at initialisation, global loss only), `global_loss_only`; untrained twins for the two baselines only. Readout per phase: `phase_summaries_` (drift per component, last validation losses) into the results file's `phases` section | 2026-10-17 |
| guard — hidden cells' precomputed embeddings (`precomputed_mask_fill`: learned / −1) | **implemented, unverified** — see Stage A, "Fixed inputs for debugging". 10 mutations caught; 748 tests pass; harness IDENTICAL. **Not run on data** | 2026-10-17 |

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

### 0d, 0e — masked-pass evaluation (built 2026-10-17)

Rounds 1–3b read attention and planted pairs on a clean pass, probed hidden spots pooled over
depth (mostly the rim), and leaned on a relevance rollout whose gradient target is degenerate.
Now every analysis reads the model under the evaluation discs, by depth, against a model-free
bar: the smoother (references as probe features), uniform attention, distance-matched spots, and
the same model untrained. The receptive field (gradient of a hidden spot's prediction with respect
to every visible spot's input) replaces relevance. What each task answers, and which old ones not
to use: [linear_probing_docs.md](linear_probing_docs.md). Every result of a sweep goes into one
file, `results.md` (0f): the summary across arms first, then one section per arm and analysis.

Caveat found on the way: on the 100 µm lattice, k-NN 8 edges reach 283 µm at slide edges, so the
GCN is blind beyond 290 µm only in the interior. The receptive field shows where it actually reaches.

## Stage A — the global model alone

**Question:** can a transformer, given every cell's masked expression and its position, fill in
hidden discs better than a smoother of visible cells? This removes P2 and P3 entirely.

**Purpose (the user's, 2026-10-17):**
1. Simplify the model for finding bugs.
2. Remove every influence of the local model.
3. Let the global model learn on its own.

None of these needs a learned input: a fixed input serves all three, and A2 keeps one (`pca_rope`).

### What exists

`GlobalModel` wraps a `GlobalModule` without a local component (`local_mask_hops = 0`, so the
long-range mask has nothing to block and warns). The transformer needs one vector per spot, its
**token**, as input. In a `CombinedModel` the token is the GCN's output. A `GlobalModel` has no
GCN, so `type_gex_embedding` says where its tokens come from. Before A1 there were three options:

- **`PCA`:** sklearn PCA, fitted once on the first batch the module sees and then frozen. That is
  the first (masked) **validation** slide, not a training batch, because Lightning's sanity check
  runs one validation step before training starts. Left as is, since changing it would change
  existing runs.
- **`NMF`:** cannot take the −1 fill of masked entries.
- **`Precomputed`:** reads an embedding from `obsm[latent_obsm_key]`.

None of these is trained. Being computed outside the transformer is not the issue: in the joint
model the GCN computes the tokens too, but the global loss flows back through them into the GCN's
weights, so they adapt to what the transformer needs. A fixed input never adapts. The transformer
can still recombine what a fixed input keeps, since its first operations are learned linear maps
(queries, keys, values), but it cannot recover what the input discarded. PCA squeezes 253 genes
into 64 directions once, chosen by variance before training. That would confound A2: a global-only
model doing badly could just mean PCA threw away the useful directions.

### A1 — a learned linear input

**What it is.** In one sentence: training, not variance, chooses which 64 directions of a spot's
expression reach the transformer. A convenience, not a requirement of Stage A (see its purpose
above); fixed inputs stay available for debugging (below). `type_gex_embedding: Linear` gives the `GlobalModule` one learned
layer,
`input_embedding = nn.Linear(n_input, n_embed)`, from a spot's expression vector to its token.
On synth_spot that is 253 genes → 64 dimensions, 16,256 parameters. It is trained together with
the transformer and the decoder, by the same reconstruction loss.

**What a spot's token is, step by step:**

1. **Masking.** The spot's expression is masked as in training: a hidden entry, or every entry
   of a hidden spot, is set to −1.
2. **The layer.** The token is `W · x + b`, a learned weighted sum of the spot's (masked) genes.
3. **The transformer's input path,** unchanged from the joint model: its input LayerNorm
   (`norm_input`), then the node positional encoding if any, then attention. A rotary encoding
   rotates queries and keys, and a bias is added to the attention logits.

**What that means for the disc task:**

- **Visible spot:** its token is a learned summary of its own expression, with no information
  from its neighbours.
- **Hidden spot:** its input is all −1, so every hidden spot gets the *same* token
  (`b − W · 1`). Only the positional encoding tells hidden spots apart. Without one (`lin_none`),
  all hidden spots of a slide receive identical attention output and one shared prediction: the
  slide-mean control.
- **Gene masking (`lin_rope_gene`):** a hidden entry enters as −1, which no real `log1p_norm`
  value takes (they are ≥ 0). A masked entry therefore pushes the token in a direction no
  measurement does, and the layer can learn to discount it.

**How it relates to the GCN.** The GCN computes
`final_norm(GCN_layers(x) + LayerNorm(input_proj(x)))`. Its skip branch, `input_proj`, is exactly
such a linear layer, with the same input and output sizes. The learned linear input is that
branch *without* the message-passing branch. This is what makes A2 a fair comparison:

| | GCN-only reference | joint model | global-only model |
|---|---|---|---|
| per-spot input | the same linear layer | the same linear layer | the same linear layer |
| neighbourhood mixing | 2 GCN layers | 2 GCN layers | none |
| attention over the slide | none | yes | yes |

Each arm gets its context in only one way.

**Why linear, not a deeper encoder.** The aim is to give the global side no capacity the local
side lacks at the input; the transformer's own feed-forward layers add the nonlinearity after
attention. If A2 shows the global-only model far below the smoother, a deeper input (an MLP) is
one follow-up, but not before we know.

**Why not keep PCA.** Its 64 directions are fixed by the variance of one masked validation slide,
not by what helps reconstruction. The transformer can recombine those 64, but not get back the
189 that were dropped. On the rescaled synth_spot about 45% of the deviation variance
sits in the pure-noise ZINB genes. On a disc-masked slide the contrast between hidden (all −1) and
visible rows is likely a leading component. So a frozen PCA probably spends much of its capacity
on noise and on "hidden or not". `pca_rope` stays in A2 to show what learning the input buys.

**Design principle (the user, 2026-10-17):** other embeddings are not computed inside InterScale.
The user computes them per cell outside it (scVI, PCA, ...), stores them in `adata.obsm`, and passes
them in through `Precomputed` (`latent_obsm_key`). The pipeline's only job with them is to hide the
embeddings of hidden cells (below). The in-pipeline inputs (PCA, NMF, Linear) remain, but they are
not the route for Stage B.

**Fixed inputs for debugging, and their caveats:**
- **`PCA` works.** It is computed from the *masked* input, so hidden values never reach it. It is
  fixed after the first slide it sees: a validation slide (sanity check), whose hidden rows are all
  −1. Fixed, but not fitted on what one would expect. Fitting it once on the training slides would
  be cleaner; not done, since it changes existing runs.
- **`Precomputed` is guarded (built 2026-10-17).** `apply_mask` overwrites only `x`, not the
  `embeddings` read from `obsm`, so a hidden cell used to hand the transformer an embedding of its
  own true expression -- a compressed copy of the answer (cf. round 3b's first run, where hidden
  discs were not hidden: SCE 0.037, noise genes at r = 0.997). Now:
  - **The fill.** `GlobalModule.hide_embeddings` replaces a hidden cell's whole embedding with
    the fill set by `model.global_component.parameters.precomputed_mask_fill`:
    - `learned` (default): one trained vector shared by every hidden cell, drawn from a private
      generator, so the rest of the model is initialised identically under both fills;
    - `minus_one`: every entry −1 (`MASK_VALUE`), as masked expression.
  - **Where it applies.** In every masked pass: training, validation, probes and head scores.
    Inference (`get_model_output`) and clean probe passes leave the embeddings as they are.
  - **Gene masking is refused,** by the config validator and by the module: an embedding cannot
    hide single genes.
  - **`latent_obsm_key` must be set.**
  - **Checkpoint tag:** `emb-pre-<key>-learned_` / `emb-pre-<key>-m1_`, with the `obsm` key in
    letters and digits (`X_pca` → `Xpca`), so two embeddings never share a checkpoint.
    Precomputed checkpoints from before 2026-10-17 carry no tag, and their hidden cells leaked.
  - **Tests,** through a GlobalModel built from a config:
    - a random change to the hidden cells' embeddings never changes a prediction, while a change
      to a visible cell's embedding does;
    - the learned fill trains, and the −1 fill adds no parameter;
    - a clean pass is untouched;
    - both fills initialise the rest of the model identically;
    - the config and the module refuse what they should.

    10 mutations, each caught; 748 tests pass; harness IDENTICAL.

  **Correction (2026-10-17) of what this plan said before.** I had recommended the learned vector
  because "zero is roughly an average cell" and "embeddings have no known range for an
  out-of-range value". That does not hold here. Every token first goes through the transformer's
  input LayerNorm, which removes a token's mean and scale. So **any constant fill (−1, 0, 5)
  comes out as the LayerNorm's bias vector**. No real embedding becomes that, since a real one is
  not constant across its dimensions, however close to zero it is. An average-looking real
  embedding is rescaled to unit variance and keeps its direction.

  So −1 is a valid fill, and the two settings differ in this way:
  - `minus_one`: the hidden token is the LayerNorm bias, which is shared with every token's
    offset;
  - `learned`: the hidden token is a direction of its own.

  Which works better is an empirical question, hence both. A side effect of the same LayerNorm:
  a per-cell offset or scale in a precomputed embedding (e.g. one that tracks library size)
  never reaches the transformer.

  Why not −inf: as a token value it turns into NaN at that LayerNorm (inf − inf). −inf only works
  on attention logits, and blocking hidden cells as keys would still leave a hidden cell's own
  token in its residual stream, straight to its own prediction. The token itself must be replaced.

**Implementation:**
- Both training and inference go through `GlobalModule.gex_embedding(batch)`. Training passes the
  masked batch, `GlobalModel.get_model_output` the clean one.
- PCA and NMF behave exactly as before; Precomputed now hides hidden cells' embeddings (above).
- The config validator accepts only inputs the module can build: None, PCA, NMF, Precomputed,
  Linear. The `scvi` value named in old docstrings is refused.
- Only the global-only checkpoint name carries `emb-linear_` (or `emb-pre-*_`). A `CombinedModel`
  ignores `type_gex_embedding`, so its name is unchanged, and PCA/NMF names are unchanged too.

**Verified by tests:**
- the layer is built and receives gradient;
- the hidden spots' values never reach a prediction, while a visible spot's value does;
- inference maps the clean input through the same layer;
- the validator and the checkpoint tag behave as described;
- PCA and LocalModel runs are unchanged (harness IDENTICAL).

Four mutations, each caught. Not run on data.

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

**Set up 2026-10-15** in `notebooks_and_scripts/slurm/` (the yaml paths inside point at
`/lustre/.../MA/yaml/`, results go to `/lustre/.../MA/15_10_results/global_only`):

- `global_only.yaml`: round 3b's disc task with round 3a's settings, dense attention, width 64,
  `type_gex_embedding: Linear`, the long-range mask off (a GlobalModel has nothing to block),
  `global_residual: false`.
- `global_only_sweep.yaml`: the five GlobalModel arms above plus `local_gcn` (LocalModel) and
  `joint_rope` (CombinedModel = round 3b's best joint arm, `rope_no_res`: plain dual decoder, mask
  on), with a `model_types` map. The sweep machinery skips keys under a component a model type does
  not have, so one base config serves all three model types.
- `global_only.py`: a copy of `dense_test_3b.py` that:
  - registers one W&B sweep per model type in the same project;
  - rebuilds `synth_spot_pp.h5ad` by itself when its recorded `signal_scale` differs from the
    source's;
  - scores every head next to the references (kernels at 150, 400 and 1000 µm, the 50 nearest
    spots, the oracle region mean);
  - runs the attention analyses only for models with a transformer, and gene ranks and loadings
    only for the joint model.

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
2. **scVI frozen, pretrained (the preferred route, by the design principle in Stage A):** as a
   **GlobalModel** with `type_gex_embedding: Precomputed`, not inside a CombinedModel: without the
   GCN there is no local component, and a second head decoding a hidden cell's own (filled)
   embedding could only predict a constant, which the reference predictors already cover. The text
   below predates the design principle; the user computes the embedding. train scvi-tools' `SCVI` on the training slides' raw counts once,
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

### C1 — per-head loss weights and freezing (built 2026-10-17)

- **`optim.loss_weights: {local: 0.5, global: 0.5}`.** A dual decoder's loss becomes
  `w_local · L_local + w_global · L_global`, from `compute_separate_losses`.
  - The default 0.5 / 0.5 is the mean of the two heads, i.e. the stacked loss of every earlier run,
    and it keeps that exact code path (harness IDENTICAL). The plan had said 1 / 1, which would
    have doubled the loss.
  - Any other weights replace `<mode>_loss` by the weighted sum, so checkpointing and early stopping
    follow the phase's objective.
  - A head with weight 0 sends no gradient. With `w_global = 0` the GCN learns from the local loss
    alone; with `w_local = 0` the local decoder gets nothing, but the GCN still learns through the
    tokens.
  - Refused on a single-head model, and for a loss that cannot give each head its own loss
    (`none`, `SCE_EntropyATT_Loss`).
- **`optim.frozen: []`** (`local` / `global`): those components get `requires_grad = False` for the
  fit (`train._schedule.freezing`, which restores each parameter's own previous flag afterwards),
  so the optimizer never sees them.
  - The training plan puts them back into eval mode at **every training step**, not just once per
    epoch: a probe pass can switch the model back to training mode between steps.
  - **Named `frozen`, not `freeze`,** because a yacs `CfgNode` has a `freeze()` method, and
    `cfg.optim.freeze` would return the method instead of the key.
  - Freezing every component of the model is refused.
- **Checkpoint tag:** `lw<local>-<global>` and `frz-<components>` in the training tag.

### C2 — the staged training driver (built 2026-10-17)

`train()` dispatches on `optim.schedule`. `joint` (the default) is one phase with the top-level
settings and runs exactly as before. Anything else runs a list of phases (`train._schedule.Phase`:
`share`, `frozen`, `loss_weights`, `lr_scale`) through `_train_phase`.

**Schedules:**

| schedule | phases |
|---|---|
| `joint` | one phase, `optim.frozen` and `optim.loss_weights` (today) |
| `local_then_global` | (1) local alone: global frozen, only the local loss; (2) global alone: local frozen, only the global loss; 50 / 50 |
| `local_then_global_ft` | as above on 40 / 40, then 20% joint fine-tuning at 0.1× learning rate |
| `alternating` | `optim.schedule_cycles` (default 4) cycles of (local alone, global alone), equal shares; the local phases use only the local loss, so the GCN never learns from the global one |
| `alternating_coupled` | as `alternating`, but the local phases keep the default mean of both losses: the GCN may adapt its tokens to what the (frozen) global component needs |
| `custom` | `optim.schedule_phases`, a list of `{share, frozen, loss_weights, lr_scale}` |

**Deviation:** the alternating schedules are cycles of (local, global), **ending on a global
phase**. The plan had (local, then cycles of (global, local)), which ends on a local phase: the
GCN would change while the global component is frozen, leaving the final global head trained on
stale tokens.

**Per phase:**
- `max_epochs` (= `optim.n_epochs` in a real run) is split by the shares with largest-remainder
  rounding, so a staged run gets exactly the joint budget. A smoke run with fewer epochs than
  phases gives every phase one epoch.
- Each phase gets its own optimizer, learning rate (`optim.lr × lr_scale`), warm-up
  (`optim.lr_warmup` scaled to the phase, at least 1) and cosine (over the phase).
- `min_epochs` is capped at the phase length. Early stopping, if on, applies within each phase.
- The mask redraw, efficiency monitor and online probes run in every phase.

**Run-level:**
- Only the first phase seeds and opens W&B; later phases continue the random streams. One W&B run
  serves all phases, and its `trainer/global_step` continues across them (`StepOffsetWandbLogger`,
  instead of the plan's per-phase prefixes, so every metric is one continuous curve).
  `schedule/phase` marks the boundaries.
- Only the last phase checkpoints, restores its best epoch, validates, tests and saves. Earlier
  phases hand their final weights on in memory, with Lightning's default checkpoint turned off.
- `self.history_` is the last phase's history, `self.phase_histories_` all of them.
- **Checkpoint tag:** `sch-l2g`, `sch-l2gft`, `sch-alt<k>`, `sch-altc<k>`, `sch-custom-<hash>`.
- Staged schedules need `model.decoder.dual_decoder: true`, and per-phase settings rule out the
  top-level `frozen` / `loss_weights` (the validator says so).

**Phase 1 from a trained LocalModel:** `CombinedModel.copy_local_from(local_model)` copies a
`LocalModel`'s module into the dual decoder's local part, strictly by name and shape, leaving its
auxiliary-loss heads behind. The local-only reference of A2 can therefore stand in for phase 1.
This is not wired into the scripts yet.

**Verified by tests** (`tests/test_staged_training.py`):
- the presets, the epoch split and the validator;
- the default loss is bit-identical to the stacked one, other weights give the weighted sum
  (also as `val_loss`), and a weight of 0 sends no gradient;
- a frozen component does not move, is out of the optimizer, stays in eval, and its flags come
  back; hidden values never reach a prediction with a component frozen;
- the driver hands each phase its epochs, learning rate, warm-up, cosine and `min_epochs`;
- an end-to-end `local_then_global` run on toy data:
  - each component moves only in its phase;
  - the weights carry over;
  - frozen parts are in eval at every step;
  - masks are redrawn in both phases;
  - the model is checkpointed and saved once, with no duplicate files;
- the W&B step offset; `copy_local_from`; the checkpoint tags.

28 mutations, each caught. 779 tests pass; harness IDENTICAL (its dual-decoder case included).

### C3 — the staged-training sweep

Arms, all RoPE, discs: `joint`, `local_then_global`, `local_then_global_ft`, `alternating_4`,
`alternating_4_coupled`, plus `local_then_global` with the residual decoder off.

Set up 2026-10-17 (`slurm/staged_training.*`) with the residual decoder on for the staged arms, and
three arms beyond the plan: `joint_res` (the baseline for them), `frozen_local` (the GCN frozen at
initialisation, only the global loss: B1's "frozen random GCN" as a dual model) and
`global_loss_only` (the GCN trained only through the global loss). The per-phase readout is the
drift and last validation losses of each phase, not a full evaluation at each phase's end.

Readouts:
- as A2, per phase at its end, the deep field above all;
- the GCN's drift between phases (relative weight change), and the token cosine.

**Predictions.** If H-C holds, `local_then_global`'s global head beats `joint`'s in the deep field,
and the coupled alternation either helps (the GCN learns tokens that serve global) or collapses
back to the joint result.

## Results of the first sweeps (run 2026-10-18; one seed per arm)

Files: `/home/lehnerl/Arbeit/results/global_only_sweeps/<sweep>/results.md` (`global_only`,
`global_pca`, `global_scvi`, `staged_training`). `global_only` and `staged_training` lack their
summary (the job ended early); rebuild with `write_report`. 6,251 hidden test spots; the deepest
bin (>500 µm) has only 135, so differences under ~0.02 there are noise. Metric: centred cosine on
hidden spots, overall / 290–500 µm / >500 µm.

| model | all | 290–500 | >500 |
|---|---|---|---|
| best smoother (Gaussian, 400 µm) | 0.465 | 0.448 | 0.413 |
| slide mean | 0.260 | 0.256 | 0.261 |
| joint (A2 `joint_rope` = C3 `joint`), global head | 0.459 | 0.438 | 0.418 |
| joint, local head | 0.333 | 0.060 | −0.020 |
| global only, precomputed PCA + RoPE (`pre_pca_rope`) | 0.442 | 0.433 | 0.413 |
| global only, scVI + Fourier (`scvi_fourier`) | 0.414 | 0.404 | 0.378 |
| global only, learned linear + RoPE (`lin_rope`) | 0.384 | 0.383 | 0.370 |
| global only, in-pipeline PCA + RoPE (`pca_rope`) | 0.309 | 0.305 | 0.272 |
| best staged schedule (`altc4_res`) | 0.451 | 0.436 | 0.416 |

**Verdict.** Nothing passes the best smoother on gene prediction. H-A (the GCN's embedding limits
global) and H-C (joint training holds global back) are not supported: every variant lands at
global validation loss 0.545–0.57, a random frozen GCN included. The remaining lever is what the
model is asked to predict, or combining near and far information.

**Positive.**
- The global head fills the GCN's blind area: at 290–500 µm 0.438 vs the local head's 0.060; at
  >500 µm it equals the best smoother. Untrained twins sit at ~0.005, so it is learned.
- The round-2 floor problem is gone (0.46 vs slide mean 0.26); raw SCE (0.074) is close to the
  smoother's (0.071), so the magnitude is fine now (0c stays deferred).
- A fixed PCA input is a strong global-only baseline (`pre_pca_rope`), beating the learned linear
  input and the in-pipeline PCA.
- Tokens carry more than the predictions use: scVI tokens beat the smoother on source distances
  (0.459 vs 0.435 overall; 0.532 vs 0.429 at >500 µm), and PCA + Fourier adds +0.066 over the
  smoother for region at >500 µm. So the decoder or objective may be the bottleneck, not the representation.
- Two strategies reach similar scores: `pre_pca_fourier` ignores everything within 2 mm of a deep
  hidden spot and reads 3–4 mm away, `pre_pca_rope` reads the nearest ring (0.5–1 mm); 0.413 vs 0.442.
- Training is bit-reproducible (A2 `joint_rope` and C3 `joint`, separate runs, identical losses).

**Global vs local.**
- Near visible spots the local head is slightly better (0.486 vs 0.475) and the global head
  mostly copies it (CKA 0.93 at the rim; 0.13 at 290–500 µm; 0.015 beyond).
- Beyond the GCN's reach the local head is blind: receptive field exactly 0 beyond 500 µm, probes
  collapse (source distances −0.14, region 0.53 vs 0.43 and 0.94 for global).
- The global head relays inward: its attention goes to nearby hidden tokens (12× uniform at
  0–300 µm); its influence sits on the nearest visible ring. A two-step smoother.
- Joint training barely changes the GCN (local head 0.333 vs GCN alone 0.327).
- Global-only models are flat with depth (PCA 0.451 → 0.413) while smoothers drop (0.481 → 0.413):
  relatively strongest deep in the holes. Global-only PCA matches the joint global head beyond
  290 µm; the GCN only adds detail near the rim (0.475 vs 0.451).

**Noteworthy: bugs and surprises.**
- Incomplete runs: C3 stopped during `frozen_local`'s evaluation (trained), `global_loss_only`
  never ran. Rerun `MODE = "evaluate"` with `EVAL_ARMS = ["frozen_local"]`, and run `global_loss_only`.
- The in-pipeline PCA (`type_gex_embedding: PCA`) is fitted on the first batch, one masked
  validation slide, so it misses slide- and condition-level genes (batch 0.02, condition 0.30 vs
  0.985 and 0.984 for the precomputed PCA) and halves throughput (CPU each step). Not a good input
  for multi-slide data; prefer Precomputed.
- The learned linear input without RoPE fails: `lin_none` 0.153, `lin_fourier` 0.146, below the
  slide mean; `lin_fourier`'s loss stays at `lin_none`'s. With PCA or scVI input Fourier works (0.41).
- **Metric bug (fixed 2026-10-18):** `val_masked_scaled_cosine_error` averages both heads of a dual model, so it
  reads ~0.57 for `frozen_local` (frozen local head ~1) and 0.553 in `l2g` phase 0. Training is
  unaffected; do not read the combined metric when a head is frozen; a dual decoder now also logs
  `<mode>_masked_scaled_cosine_error_{local,global}` (earlier runs lack them).
- **Evaluation bug (fixed 2026-10-18):** planted-pair ranks for short-range pairs are inflated (untrained GCN 0.86):
  the ±50 µm distance tolerance mixes 100 µm neighbours and 141 µm diagonals on the 100 µm lattice.
  Default tolerance now 10 µm (no longer lattice-neighbour vs diagonal); rerun `MODE = "evaluate"` to
  refresh the numbers, and note a 10 µm tolerance suits the 100 µm lattice, not irregular data. The other classes sit at ~0.5 in every model: no sender selection.
- The residual decoder hurts again in the deep field (`joint_res` 0.389 vs `joint` 0.438 at 290–500 µm).
- Without RoPE, visible spots attend almost only to hidden tokens (identical mask tokens act as a
  sink); the CLS gets ~0.
- `pre_pca_rope`'s untrained twin already reads region at 0.585 (others ~0.5): part of its probe
  gains is architectural.

**Neutral comparison.**
- Inputs, overall: precomputed PCA 0.442 > scVI 0.40–0.41 > learned linear 0.384 > in-pipeline PCA 0.309.
  RoPE beats Fourier for PCA (0.442 vs 0.413), Fourier slightly beats RoPE for scVI (0.414 vs
  0.396); no encoding gives about the slide mean (0.24–0.27). Learned vs −1 fill: 0.442 vs 0.427 (PCA), 0.396 vs 0.395 (scVI).
- Schedules: none beats joint (0.459); the others 0.424–0.452, with weaker local heads (0.27–0.30
  vs 0.333). Alternating settles: GCN drift per local phase 0.39 → 0.07, transformer 0.25 → 0.12.
  `frozen_local`'s global validation loss (0.571) equals the precomputed PCA arm with the −1 fill (0.570).
- Probes: region from hidden spots up to 0.96 (+0.01 over the smoother, +0.03 for PCA + Fourier);
  the short-range receptor state, M9 stripe and M10 cone stay at chance for all, smoother included.
- Previous rounds: the 3b conclusions hold on the rescaled data (local blind beyond 290 µm, global
  reaches but does not pass the smoother, encoding matters, residual decoder no help).

**Next** (superseded by the second read below; kept for the record): the 0g analyses, an objective
a smoother cannot satisfy, and a legnini attention sweep were proposed here; all three are built
(0g; programme masking P; attention_plan.md stage 9) and the second read below reads them.

## Results of the second read (2026-10-18; one seed per arm)

Files: `/home/lehnerl/Arbeit/results/global_only_new/<sweep>/results.md`. The synth_spot models are
the first read's, bit-identical (PCA arms within 0.006): every new number comes from the 0g
analyses. `global_only` and `staged_training` again ended during evaluation (job limit): `joint_rope`
and `l2g_res` are trained but not evaluated, six staged arms never ran.

**synth_spot.**
- First gene group where a model beats the best reference (the per-bin maximum over six): the
  response genes beyond the GCN's reach. scVI + Fourier +0.035 (290–500 µm) and +0.036 (beyond
  500 µm); the joint model's global head +0.015 and +0.033; long-range responses alone, scVI +
  Fourier 0.831 vs 0.801 at 290–500 µm. Sources, by contrast, sit far below the smoother (they are
  smooth Gaussians it reconstructs well). Worth 3 seeds before it is a result.
- Blend test: PCA + RoPE averaged with the 400 µm kernel beats both in the deep field (0.456 vs
  0.448 at 290–500 µm, 0.429 vs 0.413 beyond 500 µm): modest complementary information.
- The disc task cannot test the programmes built for global context: short-range receivers score
  ~0 for every model and reference (a hidden receiver loses its sender with it), sparse responses
  0.04–0.09 vs 0.19 (the receptor is hidden too). Hence programme masking (below).
- Position baseline: on source distances it lifts the smoother from 0.435 to 0.524; embeddings add
  at most ~0.02 beyond `smoother+position` net of their untrained twins. The first read's "tokens
  know more than the predictions use" was mostly position.
- Planted pairs still ~0.5 under the corrected 10 µm tolerance: no sender selection.
- Noise genes ~0 everywhere (the control works); `is_spatial=False` is not pure noise (it holds the
  slide-level condition and batch genes).

**legnini** (attention_plan.md stage 9): every arm at the slide mean (~0.50 vs 0.497; the 40 µm
kernel 0.594). Predictions are broad averages: hidden cells at the disc rim take 92% of their
gradient influence from more than 60 µm away. The attention mechanism changes nothing (0.495–0.511).

**What it changed.** The question moved from "beat the smoother on disc reconstruction" to
evaluations that can show communication at all: programme masking (built), the rewritten patch flow
(built), and separating local from global by structure and objective (`decoder_plan.md`, proposed).

**Next.**
- Run `slurm/response_masking.py` (programme masking, 5 arms); smoke first.
- Redo `MODE = "evaluate"` for `joint_rope` (global_only) and `l2g_res` (staged_training); train
  C3's six missing arms in a separate job with a longer limit.
- Re-evaluate the legnini sweep with the rewritten flow (`MODE = "evaluate"`).
- legnini programme masking: the hidden set was decided on 2026-10-20 (legnini_plan.md; the panel has the
  SHH readouts PTCH1, GLI1, HHIP; `var` carries no annotations, so the script writes `role` and `program`).
- `decoder_plan.md`: restructure the decoder and the dual model so that local and global separate.

## Results of programme masking, first read (run 2026-10-19; one seed per arm)

File: `/home/lehnerl/Arbeit/results/response_masking/results.md`. Task: every response gene hidden in
every spot (19,877 spots, 3 test slides); centred cosine over hidden entries. Bar: `ref_own+context`
(ridge on the cell's own visible genes and kernel means of the visible genes around it).

| arm | all | M7 rings | M8 rings | M3 | dense MC23 | short-range receivers |
|---|---|---|---|---|---|---|
| bar `ref_own+context` | 0.682 | 0.63 | 0.75 | 0.46 | 0.65 | 0.62 |
| `lin_fourier` | **0.775** | 0.95 | 0.96 | 0.80 | 0.81 | 0.02 |
| `joint_rope`, global head | 0.756 | 0.85 | 0.81 | 0.65 | 0.68 | 0.64 |
| `lin_rope` | 0.708 | 0.82 | 0.88 | 0.65 | 0.68 | 0.01 |
| `lin_none` | 0.407 | 0.44 | 0.45 | 0.52 | 0.66 | 0.01 |
| `local_gcn` | 0.516 | 0.44 | 0.51 | 0.66 | 0.64 | 0.61 |

- **First time a model passes the bar on a task built for global context.** The gain sits in the
  ring programmes (response a function of distance to a source), which fixed kernels of the inputs
  cannot express. Untrained twins are at ~0; `lin_none` equals `ref_own` (0.407 vs 0.405), so
  without position a global-only model reads only the cell itself.
- **The joint model combines the scales:** its global head reaches the short-range receivers
  (0.64-0.68, via the GCN's tokens) and the ring programmes (0.85/0.81); global-only models score ~0
  on short-range receivers.
- **Fourier beats RoPE** for global-only models here (0.775 vs 0.708), the reverse of the disc task.
- **Misses.** M11 > MC14/15 (150 um kernel solves it, bar 0.92): global-only 0.15, joint 0.57. The
  mirror programmes MC1, MC2, MC4, MC6 (response copies the cell's own visible source): all below
  `ref_own` (MC4: 0.36 vs 0.68). Direction programmes M9, M10: 0.1-0.3 for every arm and reference.
  Sparse responses only match the bar (~0.9): the visible receptor gives the receiver away.
- **The sender ranks (`response_edges`) are not interpretable yet.** `lin_fourier` ranks dense
  senders at 0.76 and sparse ones at 0.55; the GCN ranks R1-R4's senders at 0.40 (below blind) while
  predicting those responses at 0.61; global arms rank mid-range pairs at 0.05-0.24. Hypothesis, not
  checked: the measure is a gradient magnitude (sensitivity, not contribution), and the token
  LayerNorm divides gradients by the token's spread, so a high-expressing sender is less sensitive
  than its neighbours. Use gradient x input or integrated gradients (roadmap T4) before reading it.
- **Next:** 3 seeds of `lin_fourier`, `joint_rope`, `local_gcn`; a `joint_fourier` arm; the sender
  ranks with attributions; sparse responses scored within receivers only; per-head distance prior
  for M11; anisotropic Fourier bias for M9/M10; the additive decoder (decoder_plan D3.1) for the
  mirror programmes. Schedule and decision points: [roadmap.md](roadmap.md).

## Results of programme masking, second read (run 2026-10-19/20; three seeds, 100 epochs)

File: `/home/lehnerl/Arbeit/results/response_masking/results.md` (19 runs and 3 untrained twins, written
by `response_masking.py` as set up in roadmap.md T0). Same task and bar as the first read (`ref_own+context`
0.682). Seeds: sd 0.003-0.017 on the overall score, median per-programme sd 0.008 (largest 0.066), so
differences above ~0.03 are real. Centred cosine on the hidden entries; overall = over all hidden spots.

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
| `lin_rope_pool500` | 0.466 | 0.214 | 0.606 | 0.097 |
| `joint_fourier_pool500`, global (local) head | 0.626 (0.498) | 0.249 (0.586) | 0.787 (0.457) | **0.082** (0.757) |

**What it shows.**
- **First robust positive.** `joint_fourier`'s global head is 0.138 above the bar over three seeds and far
  above its own local head (0.506): the long-range responses 0.884 against 0.692 (bar) and 0.463 (GCN),
  and the rings (M7, M8) 0.93 / 0.93 against 0.63 / 0.75 (bar) and 0.44 / 0.49 (local). Fourier beats RoPE
  in every arm (joint 0.820 vs 0.748, global-only 0.764 vs 0.709). A blend with the bar no longer adds to
  the joint head (0.804 < 0.820): it holds what the references know.
- **But per programme the models do not beat the bar overall.** Equal weight over the 25 planted
  programmes: `joint_fourier` 0.628 against the bar's 0.685; it beats the bar by more than 0.05 on 3, ties
  on 12, loses on 9 (`lin_fourier`: 4 / 10 / 10). The overall score is carried by the rings, which hold
  most of the hidden entries.
- **Where it loses, and the likely reason.**
  - *Mirror programmes* (response copies the cell's own visible source: MC1, MC2, MC4, MC6): 0.54 / 0.41 /
    0.27 / 0.47 against `ref_own` 0.84 / 0.68 / 0.68 / 0.67, in every NN arm, the GCN alone included. A
    linear read of the cell's own genes beats the 64-dim embedding: the own-cell skip is the candidate fix
    (decoder_plan.md D3.1).
  - *M11 > MC14/15* (single source spots, response within 500 µm): `joint_fourier` 0.54, GCN 0.35, global-only
    0.16, bar 0.92 (a 150 µm kernel). The near-source scale is not learned.
  - *Sparse threshold programmes* (M16, M17, M20): at or below `ref_own` (0.64 / 0.44 / 0.57 against 0.67 /
    0.58 / 0.53). Pooled over all cells this is dominated by "is the cell a receiver", which the visible
    receptor gives away; whether the models know the distance to the source needs scoring within receivers.
  - *Directional* (M9, M10): 0.08-0.18 in every arm and reference. The Fourier bias used is isotropic.
- **PCA of the visible genes is a poor token here** (0.34 / 0.42, against 0.76 for the learned linear
  input): the 64 leading components keep variance (45% of it sits in noise genes) and drop the sparse
  receptor and sender signals. The disc task's best global-only input does not carry over.
- **Pooling at 500 µm (5 x 5 spots, ~100 tokens).** The ring and long-range programmes survive (0.79
  against 0.89 per cell), short-range receivers rise a little in the global head (0.17-0.25, a patch holds
  sender and receiver) but the sparse responses collapse to 0.07-0.08: a patch token is shared, so the
  global head cannot know which cell is the receiver (the joint arm's *local* head, which reads cells,
  has 0.757 on the same genes). The cell's own genes are what pooling removes, so the same own-cell skip
  is the fix here. Cost: training cells/s x1.5-1.6, peak GPU memory 5.5x (global-only) and 3.5x (joint)
  smaller, at S = 2,500; at 50-100k cells that is the difference between running and not.
- **Sender ranks are still unreadable** (0.35-0.57; the GCN, which predicts R1-R4 at 0.59 from its
  neighbours' senders, ranks them 0.41). The measure is a gradient magnitude; roadmap T4 replaces it.
- **Time.** Training 0.8-1.8 s per epoch (2-3 min per run); evaluation 30-60 s per run (head scores 5-7 s;
  sender influence 5 s GCN, 23 s global-only, 50 s joint, 9-21 s pooled; the references 61 s once). The
  whole 19-run sweep is minutes as an array. 100 epochs cost about 0.01 against 200 (`lin_fourier` 0.764
  against 0.775; `joint_rope` global 0.748 against 0.756).

**Provisional base model (DP1): `joint_fourier`**, pending the attribution instrument (T4) and the
own-cell skip.

## Programme masking: what drives a response (built 2026-10-18)

**Why (the user, 2026-10-18).** The disc task cannot test the programmes synth_spot was built for:
a hidden receiver loses its sender and its own receptor with it, so short-range receivers score ~0
for every model and reference, and sparse responses (0.04–0.09 vs 0.19) are capped by the hidden
receptor; meanwhile a smoother of the visible spots around a disc explains most of what can be
filled in. The second read of the sweeps (`/home/lehnerl/Arbeit/results/global_only_new`) showed
the same limits once more: the gains over the smoother are a few hundredths on response genes, and
on legnini every arm sat at the slide mean.

**Task.** Hide every response gene (`var` role receiver or response) in every spot and nothing
else; train and evaluate on that. A response can then only be predicted from what drives it: an
adjacent sender, the distance to a source, a distant source together with the spot's own receptor.
A smoother of the responses has nothing to average.

**Bars, fitted on the training slides** (`FittedReferences`): the spot's own visible genes (a
receptor marks a responder); plus Gaussian-kernel means of the visible genes around it at 150, 400,
1000 and 2500 µm (a fixed linear view of the drivers at several scales); plus position. A model has
learned more than fixed kernels of its inputs only where it beats `ref_own+context`. On the stub
these bars are already strong (receivers 0.61, long-range responses 0.68, sparse 0.70).

**Readouts.** Centred cosine per programme and per interaction (`M7>MC7/MC8/MC9`, ...), and for each
planted programme the sender's rank in its receiver's predicted response among spots as far away
(`response_edges`, 0.5 = blind to the sender) -- the test the disc task could never make.

**Arms** (`response_masking_sweep.yaml`): `local_gcn` (drivers within two hops), `lin_none`
(drivers anywhere, no position), `lin_rope`, `lin_fourier` (drivers anywhere, with position),
`joint_rope`. Predictions: short-range receivers are a local problem (GCN ≈ context reference);
long-range and sparse responses need distant sources -- only a global model with position can
beat the context reference there, and its senders should rank above 0.5.

**Legnini.** No annotations in `var` (`gene_ids`, `feature_types` only), but the panel has the SHH
readouts (PTCH1, GLI1, HHIP), the ligand (SHH) and the ventral/dorsal identity genes. The same
machinery runs with `mask_genes: [PTCH1, GLI1, HHIP]`. *Decided 2026-10-20:* the hidden set is the whole
dorsoventral identity code (15 genes, the main task) together with the three readouts (the positive control),
selected by `var['role']`; the genes, the reasons and the sources are in `legnini_plan.md`, and
`legnini_programme.py` runs it (5 arms, 7 runs; not run yet).

**Updated 2026-10-19 for the three-week roadmap** (`roadmap.md`, T0 and T1b): `response_masking` now runs
three seeds of `lin_fourier`, `joint_rope`, `joint_fourier` and `lin_fourier_pool500`, one of the rest
(19 runs); 100 epochs; no patch flow; twins for three architectures; and 11 arms -- `joint_fourier`,
`pre_pca_rope` and `pre_pca_fourier` (a PCA of the *visible* genes through `Precomputed`, now allowed under
programme masking when the embedding's genes are recorded), and three pooled arms (the transformer on 500 um
patches of 5 x 5 spots; 200 and 300 um held 4 and 9 spots and were dropped). Each run writes its own sections; `MODE = "list"` / `ARRAY_TASK` / `MODE = "report"` give one
job per run (see `sweep_common.py`).

**Updated 2026-10-20 for the evaluation fix** (roadmap T1 and T4; training is unchanged). The second read found
three things the sweep's evaluation could not say, so three things changed in how it reports:

1. *Sender ranks.* The squared gradient is a sensitivity, so a mean-aggregating GCN ranked its own senders
   0.41 while predicting their responses at 0.59. `response_influence` now defaults to integrated gradients
   from the training-mean cell on the summed response genes of a programme (`RESPONSE_MEASURE = "attribution"`,
   `RESPONSE_STEPS = 5`); a cell's weight is the size of its net contribution, the planted sender is still
   ranked among cells as far from the receiver, and `response_check` reports the share of the response's
   change the contributions fail to add up to (below 0.1 is fine). Read a rank against the **untrained twin**
   (random weights already rank dense and short-range senders at ~0.6 on the stub, because senders deviate
   most from the baseline), not against 0.5.
2. *Sparse programmes.* Every programme with planted receivers is also scored within them
   (`heads_by_group_cells`): the pooled score mixes "which cells respond" with "how strongly", and the sparse
   responses (1.4% receivers) are dominated by the first.
3. *Per-programme comparison.* The summary counts, with every programme weighing the same, the wins, ties and
   losses of each head against `ref_own+context` (`PROGRAMME_BAR`, ties within 0.05), for all cells and within
   receivers, and names the best and worst programme.

To read it: copy the fork's `main` and the updated `sweep_common.py` / `response_masking.py` to the cluster,
then `INTERSCALE_MODE=evaluate` for the 19 checkpoints (the saved references are reused; the sender step costs
about 2.5x the old gradient step) and `INTERSCALE_MODE=report`. The old sender-rank tables (gradient
magnitudes) are replaced by the newer sections, and are not comparable with them.

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
