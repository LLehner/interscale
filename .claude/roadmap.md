# Roadmap: 19 October – 8 November 2026

Three weeks to cover every goal of the project and to try a fix for every known problem of the
global model. The goals are the six questions and four tasks of the project description
(`/home/lehnerl/Arbeit/description.pdf`). The starting point is the draft manuscript
(`/home/lehnerl/Arbeit/InterScale_Manuscript.pdf`) and the plans next to this file. This file
schedules the work and links to the plans. Designs live in each track's plan file: an existing one,
or a new one written before the track's code, as for the other plans.

The rules are the same as in the other plans:
- Update the status table in the same commit as the work.
- A track is `done` only when a run has been read. "Implemented, unverified" is a real state.
- Nothing runs locally.

The terms used below:
- **build:** code with unit tests on toy modules, and the harness IDENTICAL.
- **run:** you, on the cluster.
- **read:** the results, written into the track's plan file.

## Status

Last updated 2026-10-19.

| track | question | state | verified by | date |
|---|---|---|---|---|
| T0 | infrastructure: seeds, per-arm evaluation, inputs, CosMx | **implemented, unverified** (2026-10-19; CosMx deferred by the user) — seed axis, run selection for job arrays, per-run section files, timing, cached references, `Precomputed` under programme masking, `pca_fit: train`; 8 of 9 mutations caught or removed, harness IDENTICAL; the sweep script checked on a stub synth_spot with untrained models. **Not run on data** | 2026-10-19 |
| T1 | does the global model learn communication at all | first read 2026-10-19: global arms with a position encoding beat the bar on ring programmes (one seed); sender ranks not interpretable; 3 seeds and fixes pending | `response_masking` run (5 arms + twins) | 2026-10-19 |
| T1b | coarse-grained tokens: pool cells into patches or niches before the global model | **grid pooling implemented, unverified** (2026-10-19) — `pooling.kind: grid` (`size_um` or `n_patches`), `count`/`std` features, 35 tests + an end-to-end run for GlobalModel and CombinedModel, mutation-checked, harness IDENTICAL; 5 pooled arms in `response_masking`. The `niche` assignment waits for the user's details; patch-level attention readouts not built. **Not run on data** | 2026-10-19 |
| T2 | disentangling local and global ([decoder_plan.md](decoder_plan.md)) | proposed | | |
| T3 | masking strategies | proposed | | |
| T4 | interpretation beyond raw attention; attention sinks | proposed | | |
| T5 | how cell-type driven attention is | proposed | | |
| T6 | complexity vs tissue coverage ([attention_plan.md](attention_plan.md) stages 6, 8) | proposed | | |
| T7 | disease-specific interactions | proposed | | |
| T8 | prior knowledge (NicheNet), downstream | proposed | | |
| T9 | literature review and write-up | proposed | | |
| T10 | comparison with other methods: classification (manuscript Fig. 2), communication methods, spatially variable genes, spatial domains | proposed (asked for 2026-10-19) | | |

## Where we stand

### The goals against what has been tested

| goal (description) | built | run and read | missing |
|---|---|---|---|
| **Disease-specific interactions** through conditioning | nothing. Condition is only a split stratifier, and in synth_spot a slide-level gene programme (cg_1–3) | – | a conditioned model; data with condition-specific interactions and known truth; a CosMx ND/T1D read |
| **How cell-type driven attention is** | label flow heatmaps against the expression-permuted slide (`attention_flow`) | once, on legnini Leiden clusters under discs. There is structure beyond distance, but no rule transfers between the two test organoids | a same-type metric against a distance-matched null; the CLS-by-type confound (Alpha/Beta); CosMx with annotated types |
| **Masking behaviour** | random cells (`node`), random entries (`gene`), discs (spatial blocks), programme masking (one gene set hidden in every cell) | `node`, `gene` and discs on synth_spot and legnini. Programme masking was never run | an AMICI-style same-type attention block; masking by cell type; random gene sets per step (channel masking); combinations; one train × evaluation matrix |
| **Complexity vs coverage** | Performer, linear, Exphormer, entmax and top-k attention; RoPE, Fourier, distance, spectral and expression biases | synth_spot (2.5k spots) and legnini (4.3k cells), both small enough for dense attention. No backend changes reconstruction | runs where dense attention cannot run (CosMx sections, 47k–100k cells); windows against whole slides; exact memory-efficient attention (SDPA); cross-attention; diagnostics without S × S maps; pooled (coarse-grained) tokens, which cut S by 50–1000× |
| **Disentangling local and global** | dual linear decoder; residual decoder; `linear-lse` decoder (never run); VICReg, context NCE and ring NCE (never evaluated on a task that can show communication) | the residual decoder, which hurts the deep field | decoder_plan.md D0–D4; LSE and an orthogonality term; scores against synth_spot's planted programmes |
| **Prior knowledge (NicheNet)** | nothing | – | priors, pair validation, ligand activity, hypotheses |
| **Comparison with other methods** (manuscript Fig. 2, Suppl. Figs. 3, 4, 6A; Table X) | the model can classify (graph, node) and the old scripts ran the manuscript's classification sweeps (`cosmx.py`, `track_cosmx.py`); nothing from the comparisons is rebuilt or rerun for the current model | – | classification against GCN, PCATransformer and baselines on whole slides and patient-grouped folds; AMICI and Steamboat against planted edges; GraphCompass; spatially variable genes and spatial domains against dedicated methods. Baselines and the Visium and IMC data are provided by you on the cluster |
| Task: literature review | the plans cite the efficient-attention and encoding papers | – | interpretability (IG, rollout, relevance, sinks); AMICI, Steamboat, VirTues |
| Task: interpretation beyond raw attention | gradient influence (`receptive_field`), attention rollout (`patch_flow`), untrained twin, permuted-slide null; relevance rollout (degenerate) | influence and rollout on synth_spot and legnini. No method finds planted senders, because no model uses them yet | integrated gradients; relevance with a valid target; one benchmark against planted ground truth, across backends |

### What the manuscript's results rest on

- **The cell- and tissue-level results** (Fig. 4B–E: CLS horizontal and vertical scores, net flow
  `M_net`, flow domains) read the relevance rollout. Its gradient target, the summed output after
  an affine LayerNorm, is degenerate ([linear_probing_docs.md](linear_probing_docs.md)). This
  includes the "Alpha and Beta dominate the CLS" observation that the description builds on.
- **The gene ranks** (Fig. 4G, Suppl. Fig. 5C) come from an unmasked pass, so they measure
  pass-through, not prediction from context.
- **The local and global gene programmes** (Figs. 3, 4F–H) come from two decoders that predict the
  same target. Near visible cells the global head copies the local one (CKA 0.93).
- **The enrichment background** is not stated (the comment on p. 8). It has to be the panel.
- **Not retested yet:** the classification results (Fig. 2), the parameter-space and radius
  robustness sweeps, the model-variant table (GCN, scVI, PCA, CellCharter as local component) and
  the comparisons with AMICI, Steamboat and GraphCompass. I had set these aside as not needed for
  the goals; they are back as T10, since the manuscript's claims need them (and the Visium and IMC
  datasets with them). Their numbers come from an older model state, so they are redone with the
  current one.

## Known problems of the global model, and the fix we try

| # | problem (observed) | what we believe causes it | fix we try | track, week |
|---|---|---|---|---|
| K1 | Global never beats the best smoother at disc reconstruction (synth_spot 0.459 vs 0.465). On legnini it sits at the slide mean (0.50 vs 0.594 for the 40 µm kernel) | A plain average of the visible cells solves the disc task. On legnini attention cannot become local: the biases start at 0 and learned ≤ 0.03 | Programme masking on synth_spot and legnini; a distance prior with a length scale per head; targets split by scale (D2.1) | T1, T2; W1 |
| K2 | Near visible cells the global head copies the local one (CKA 0.93 at the rim) | Both heads predict one target, and the cell's own token stays in the transformer's residual stream | A context-only global embedding (D1); an additive decoder (D3.1); each loss on the cells where its component has to work (D2.2); pooled tokens, whose embedding is shared by a patch and cannot carry one cell's own token (T1b) | T1b, T2; W1 |
| K3 | The residual decoder hurts the deep field (0.389 vs 0.438 at 290–500 µm) | In the blind area the local prediction is a constant that the residual has to undo | Retire it in favour of D3.1; keep it as one baseline arm | T2; W1 |
| K4 | No sender selection: planted pairs at ~0.5 for both attention and influence; attention is broad | No task so far needed a sender. With ~2,500 keys, a few relevant ones need a logit lead of ~6 | Tasks that need senders (programme masking); the distance prior; sparse normalisers and top-k retested there; a null key and registers | T1, T4; W1–W2 |
| K5 | Hidden tokens collapse (token cosine 0.78–0.91 under discs) and act as a sink: without RoPE, visible cells attend almost only to them | Every wholly hidden cell gets the same input | Programme masking (no cell is wholly hidden); an option to use only visible cells as keys; a null key | T1, T3, T4; W1–W2 |
| K6 | Inputs of the global-only model: the learned linear input fails without RoPE (0.15); the in-pipeline PCA is fitted on one masked validation slide (0.31); the best input, a precomputed PCA (0.44), is refused under gene masking | A linear map of a −1-filled row carries no position. The PCA is fitted on the wrong data. The leakage guard cannot know which genes an embedding saw | Allow `Precomputed` under programme masking when the embedding was computed without the hidden genes (recorded with the embedding and checked); fit the in-pipeline PCA on the training slides (opt-in) | T0; W1 |
| K7 | Single seeds; differences of ±0.02 are unresolved (e.g. scVI + Fourier's +0.035 on response genes) | – | A seed axis in every sweep; 3 seeds for every arm a decision rests on; on CosMx, its three splits as folds | T0; all |
| K8 | Evaluations die at the job limit (`global_only` and `staged_training`, twice) | Every arm is trained and evaluated in one job | One evaluation job per arm; evaluate `joint_rope` and `l2g_res`; drop C3's six missing arms (H-C is already rejected) unless you want them | T0; W1 |
| K9 | legnini: the flow maps are not readable, the SHH source (floor plate) is avoided, and no rule transfers between the two test organoids | No model beats the local kernel, so there is no learned flow to read; there are only two test organoids | Re-evaluate with the rewritten flow; SHH programme masking; read flows only from a model above the kernel; rotate the test organoids as folds | T1, T4; W1–W2 |
| K10 | The relevance rollout is degenerate, and so is everything built on `_attn_matrix` | It takes the gradient of a summed LayerNorm output | A valid target (one cell's predicted programme), checked in T4's benchmark; redo the manuscript's CLS and net-flow analyses with the methods that pass | T4, T9; W2–W3 |
| K11 | Gene ranks come from an unmasked pass; the enrichment background is not stated | – | Gene scores from the masked pass (`score_heads` per gene); the panel as background | T2, T9; W3 |
| K12 | Global's magnitudes are off under the centred SCE | The loss ignores scale | Nothing for now: raw SCE is at the smoother's (0.074 vs 0.071). 0c stays deferred unless a cosine reading needs it | – |
| K13 | Whole sections (47k–100k cells) cannot run with dense per-cell attention; windows cut interactions at their borders | S² memory and compute in the number of cells | Pool cells into patches or niches first (T1b): S drops to ~100–2,000 and dense, exact attention fits; sparse and linear attention (T6) as the alternative | T1b, T6; W1–W2 |

## Tracks

### T0 — infrastructure (W1)

- **Seeds (K7).** Add a seed axis to `sweep_common` sweeps:
  - arms are named `<arm>_s<seed>`;
  - the summary reports the mean and the spread over seeds;
  - untrained twins are made only for the arms that are read.
- **Evaluation per arm (K8).** Training and evaluation run as separate jobs. Each sweep gets one
  array job over `EVAL_ARMS`. The report already merges sections from several jobs.
- **Inputs (K6).**
  - `Precomputed` under gene-set masking: allowed when the embedding was computed from the visible
    genes only. The script stores the gene list next to the embedding, and the validator refuses a
    hidden set that overlaps it.
  - An opt-in flag fits the in-pipeline PCA on the training slides; existing runs stay unchanged.
- **CosMx in the current pipeline.** A `PREPROCESS` hook (log1p_norm), the graph, labels
  (`cell_type_coarse`), the condition, the three provided splits, whole sections as samples, and a
  smoke config. No sliding windows (decided 2026-10-19). `max_seq_len` must therefore cover the
  largest section (100,230 cells); below that, `pad_batch` drops cells at random. What the data
  looks like:
  - 386,727 cells × 979 genes;
  - six sections: three slides, each with one ND and one T1D section, 47k–100k cells each;
  - `split` uses slide 1 for training, slide 2 for validation and slide 3 for testing. Each split
    therefore holds one donor per condition, so within a fold condition and donor are confounded.
- **Housekeeping (K8).** Evaluate `joint_rope` (global_only) and `l2g_res` (staged_training). The
  round-3b rerun note in attention_plan.md is still uncommitted.

**Built 2026-10-19** (library in the fork, scripts in `notebooks_and_scripts/slurm/`, which is not under
version control; each library change has its own commit and a CHANGELOG line):
- *Seeds.* `SEEDS` and `SEED_ARMS` in `sweep_common.py` expand an arm into runs `<arm>_s<seed>` (each sets
  `optim.seed`, so checkpoints differ); the hidden cells and probes keep `EVAL_SEED`, so runs compare
  paired. The report shows each arm with two or more seeds once as the mean, with the standard deviation
  over seeds, every run's own value and the sender-rank spread in tables beside it
  (`evaluation.report.average_over_seeds`). Untrained twins: one per architecture (first seed only).
- *Per-run evaluation and job arrays.* `MODE = "list"` prepares the data, embeddings and fitted
  references once and prints the runs with their array index; `ARRAY_TASK = True` makes job `i` train
  and evaluate run `i` (`INTERSCALE_RUNS` / `INTERSCALE_MODE` override `RUNS` / `MODE` from the
  environment); `MODE = "report"` merges. Every run writes its own `sections/<run>.md`
  (`report.read_parts`, the newest file wins), so jobs never write one file together, and a single job
  rebuilds `results.md` after every run, so one that ends early leaves a readable file.
- *Time.* Each evaluation step is timed (a `timing` section and a summary table). The ridge references
  of the programme evaluation, whose penalty search takes minutes on 24 slides, are fitted once and
  saved (`FittedReferences.state`, `RESULTS_DIR/fits/`); the sender influence takes 8 receivers per
  programme (was 16), the receptive-field and planted-pair queries 8; `FLOW_ARMS` limits the flow;
  the synth_spot yamls train 100 epochs (legnini keeps 400, its slides are few); twins only for the
  arms that matter; `response_masking` has no flow.
- *Inputs.* `Precomputed` under programme masking: allowed when the embedding never saw a hidden gene,
  recorded with `tl.record_embedding_genes` and checked by `prepare_geome_dataset`; the
  `response_masking` script computes a PCA of the visible genes for the `pre_pca_*` arms. `pca_fit:
  train` fits the in-pipeline PCA on the masked training slides (default unchanged).

### T1 — does the global model learn communication at all (W1; K1, K4, K5, K9)

Before anything is interpreted, we need a model whose global part does what a fixed kernel of its
inputs cannot. Every later track builds on the model chosen here (DP1).

- **Run `response_masking`** (built, never run): its 5 arms, plus `pre_pca_rope` (a PCA of the
  visible genes, once T0 allows it), each with 3 seeds.
  - Read the programme scores against `ref_own+context`, and `response_edges` (the planted sender's
    rank), on long-range and sparse responses.
  - Short-range receivers are the GCN's positive control.
- **Build: an oracle bar.** A ridge on the true drivers, for synthetic data only: distances to the
  sources, and the receptor. The script names the obs keys. It separates "not learnable from these
  inputs" from "the model fails".
- **Build: a distance prior with a length scale per head.** Start the linear distance bias (it
  exists, in ALiBi form, and starts at 0) at fixed per-head length scales from ~20 µm to ~2 mm,
  optionally frozen.
  - Some heads look locally and some globally from step 0: Steamboat's multi-scale heads, without
    labels.
  - Each head's scale is known, which makes its maps readable.
  - Aimed at K1 (legnini cannot even reproduce the 40 µm kernel) and K4.
- **Build: visible keys only** (discs). Wholly hidden cells are queries but never keys (K5).
- **legnini.**
  - Re-evaluate the attention sweep with the rewritten flow.
  - Programme masking on the SHH readouts: `PTCH1`, `GLI1` and `HHIP` hidden in every cell, `SHH`
    visible. The gene set is your call (see the decisions below).
  - Arms: `local_gcn`, `joint_rope`, global-only PCA + RoPE, and the same with the distance prior.
  - Positive control: the readouts' predictions should draw on the floor plate (c13, SHH) more
    than distance alone explains.
- **First read of `response_masking`, 2026-10-19** (one seed per arm; details in
  [global_plan.md](global_plan.md)). Global models with a position encoding beat the bar
  (`ref_own+context`, 0.682) for the first time: `lin_fourier` 0.775, `joint_rope` global head
  0.756, `lin_rope` 0.708, against `local_gcn` 0.516 and the untrained twins at 0. The gain is in
  the ring programmes (M7, M8: 0.95–0.96 against 0.63–0.75). Open, and now part of this track:
  - **Sender ranks cannot be read yet.** They are gradient magnitudes, which do not follow the
    predictions (the GCN ranks the planted senders of R1–R4 below 0.5 while predicting them at
    0.61). Check the LayerNorm explanation, then switch to gradient × input or integrated gradients
    (T4, moved up to W1).
  - **M11 → MC14/15** (a 150 µm kernel solves it, bar 0.92): global-only 0.15, joint 0.57. Fix:
    per-head distance prior.
  - **Mirror programmes** (MC1, MC2, MC4, MC6, response copies the cell's own visible source):
    every model is below `ref_own`. Test D3.1 (additive decoder), which reads the cell's own genes
    linearly.
  - **Direction** (M9, M10): all arms at 0.1–0.3 including the bar; the Fourier bias is isotropic.
    Try the anisotropic Fourier bias.
  - **Sparse responses** are dominated by the visible receptor. Score them within receivers only.
  - **A `joint_fourier` arm** is missing; it is the likely base model.
- **Decides (DP1, Monday of W2).** The base model for T3–T8 is the arm that, over 3 seeds, scores
  highest above `ref_own+context` on long-range and sparse responses and ranks senders above 0.5.
  If no arm passes, compare the arms with the oracle bar:
  - the oracle passes: the model is the limit. Try more steps, more width, the distance prior,
    top-k;
  - the oracle fails too: the task is the limit.

### T1b — coarse-grained tokens (W1; K2, K5, K13)

**Asked for 2026-10-19:** instead of one token per cell, the global model works on pooled units,
patches or small clusters (niches). Reasons given: smaller domains may differ more from each other
than neighbouring cells do; the model does not have to check single cells, which are not that
relevant; and the number of observations drops drastically. It can be combined with the local
component: learn a local embedding per cell, cluster, then run the global model on the clusters.

More reasons, from what the sweeps showed:
- **Noise.** Counts per cell are sparse (legnini, CosMx). A patch mean is a denoised input, and
  the pooled target (below) a denoised target.
- **No copying (K2).** A patch token is shared by all its cells, so the global embedding cannot
  carry one cell's own token. That is the scale split of decoder_plan.md D2.1, by architecture.
- **Hidden-query design for free (K5).** A patch with nothing visible is a position-only token, as a
  hidden cell is under discs, and visible patches are its keys. No wholly hidden cells crowd the
  keys.
- **Cost, and CosMx without windows (K13).** Tokens per whole section if patches are squares of the
  given side (data only; the first row assumes `obsm['spatial']` is in µm, the second `CenterX/Y_global_px` at
  0.12 µm/px; I have not verified either, and the two disagree by 4×):

  | section sizes | cells | 100 µm | 200 µm |
  |---|---|---|---|
  | `obsm['spatial']` | 47k–100k | ~2,000 | ~530 |
  | `CenterX/Y_global_px` | 47k–100k | 350–580 | 120–190 |

  Either way a whole section becomes 100–2,000 tokens, the size of legnini's 4.3k cells, so dense
  attention (exact, with maps and attribution) runs without windows. synth_spot: 2,500 spots become
  625 / 278 / 100 tokens at 200 / 300 / 500 µm. legnini: ~4,300 cells become ~100–400 tokens at
  50–100 µm.
- **A tissue-level unit to interpret.** Patch-to-patch attention and influence, composition per
  patch for labels (T5), niche-level NicheNet pairs (T8).

**Design** (config switch `pooling`, default off, so every existing model is unchanged):
- **Token.** The mean of the cells' tokens over the *visible* cells of the patch, per gene when the
  input is expression (entry-wise denominators, so a gene hidden in some cells is averaged over the
  others). In a `CombinedModel` the cell token is the GCN's output, in a `GlobalModel` the input
  layer's or a precomputed embedding. Added: the patch centroid as position (RoPE and the biases
  read it), log cell count, optionally the spread (std) of the tokens. A patch with nothing visible
  gets the learned fill and its position.
- **Assignment** (`pooling.kind`), from safest to most flexible:
  1. `grid`: `floor(position / size)` per graph. Position only, so no hidden value can enter.
     `size` is a knob; test-time offset grids can be averaged.
  2. `spatial_clusters`: contiguous niches, for example Leiden on the spatial graph with edges
     weighted by similarity of the *visible* genes. Hidden cells are placed by position.
  3. `niche`: the proposal above. Fit k-means or a GMM on the local embeddings of the training
     slides with the local component frozen (staged training already trains it first and hands it
     on, `copy_local_from`). An embedding cluster is not contiguous, so a token is a
     (cluster, grid cell) pair, which keeps a location and a composition. A hidden cell is placed
     by the cluster of its nearest visible neighbours.
  4. Stretch: learned soft assignment to spatial anchors (attention_plan.md stage 7).
  Rule for 2–4: an assignment may read visible entries only.
- **Un-pooling.** A cell's prediction is its patch's global embedding through the global decoder, plus
  in a `CombinedModel` the cell's own local embedding through the local decoder, added (D3.1). The
  loss is still per hidden cell and the evaluation is unchanged: same references, same bars, same
  tables, so the numbers compare with the cell-level arms. Option: an auxiliary loss on the patch
  mean of the hidden entries (denoised target); weight is a knob.
- **Masks.** Cell, entry and programme masking as now. New, shared with T3: patch masking, hiding
  whole patches (the discs of the disc task, at patch scale).
- **Attention.** Tokens are tokens: every backend, bias and encoding works as is. The long-range
  mask is off (at patch scale it is moot). Maps are patch × patch; per-cell influence comes from
  broadcasting.
- **Tests, before any run.** Hidden values never reach a prediction through the pooled mean or the
  assignment (the `b7cf317` lesson); a patch of one visible cell equals the cell-level token; the
  default model is bit-identical (harness IDENTICAL); checkpoint tag `pool-<kind><size>`.
- **Where.** A pooling stage before `pad_batch` in `common_step_local_to_global` and an un-pooling
  after the transformer; positions come from `dataset.spatial_key`. I have not read those paths in
  detail yet, so the first step on Tuesday is the design note and the leakage tests.

**What it costs, and the checks for it.**
- Within-patch heterogeneity: a patch that mixes islet and acinar cells. Composition and spread
  features, and the patch size as the knob. Read the programme scores against patch size.
- Short-range structure (R1–R4, M11's 150 µm kernel) is lost below the patch size. That is the local
  component's job, so the combined arm is the one to read for it.
- Borders of the grid: shifted grids at test time, or overlapping patches.

**Built 2026-10-19** (`grid`; `.claude/roadmap.md` is the design, `CLAUDE.md` the invariants):
- `GlobalModule.step` replaces the input step + transformer calls of the three modules; pooled it pools
  the cell tokens of the (masked) batch, runs the transformer on a `PooledBatch`, and hands each cell its
  patch's output in the per-cell layout. The token is pooled *before* the input LayerNorm, so a patch of
  one visible cell is that cell's token (tested: with every cell its own patch the pooled model equals
  the per-cell model to 1e-5, and its state differs by one parameter, the fill).
- The grid starts at each slide's minimum corner (a slide's place in the scanner's coordinates does not
  matter); `n_patches` finds each slide's side by bisection, so slides of different extent get the same S.
- *Hidden cells.* A cell hidden whole weighs 0 in the mean; under programme masking no cell is hidden
  whole, so every cell contributes its visible genes. Tested through the real model under cell and
  gene-set masking: changing hidden values moves no prediction (also in a patch hidden entirely, which
  becomes a position-only token), changing a visible cell moves its patch and, through attention, the rest.
- *Refused, with the reason:* the long-range mask, `lap`/`rw` node encodings, spectral and expression
  biases, Exphormer, an attention loss (config validator); `evaluate()`, `attention_by_query`,
  `attention_diagnostics` and the rollout flow (they index attention by cell).
- *Not built:* `niche` and `spatial_clusters` assignments (the user will give details), a patch-level
  attention readout (patch-patch attention and flow), random grid offsets as augmentation.
- *Arms in `response_masking`:* `lin_fourier_pool200` / `_pool300` / `_pool500`, `lin_rope_pool300`,
  `joint_fourier_pool300` (no long-range mask: patches have no neighbour graph; the local head reads cells).

**Run, W1 (synth_spot, programme masking, discs as the second readout).**
- Arms: `lin_fourier` and `lin_rope` pooled on a grid of 200, 300 and 500 µm; `joint` pooled at the
  same sizes with the additive un-pooling; the cell-level arms from the first read as references.
  3 seeds of the main arms. `niche` pooling on W2 Monday–Tuesday, once the frozen local stage exists.
- Read: programme scores against `ref_own+context` and against the cell-level global arms,
  especially the ring programmes (0.95/0.96 today) and the short-range receivers; steps/s and
  peak memory; patch-level attention maps; sender ranks once T4's attributions exist.
- **Predictions.** Pooled matches or beats cell-level global on ring and long-range programmes
  (less noise, the same information), loses on short-range ones unless the local component is
  there, and the pooled attention is cheaper by about the square of the token ratio. If it matches
  cell-level at 20–50× fewer tokens, it is the route for whole CosMx sections (DP3).
- **legnini, W1 Thursday:** patches of 50 and 100 µm on the programme-masking task, the same arms.

### T2 — disentangling local and global (W1–W2; K2, K3, K11)

The design is in [decoder_plan.md](decoder_plan.md).

**The global-only model gets decoder arms too** (asked for 2026-10-19). A `GlobalModel` already
builds its decoder from `model.decoder.type` (`linear`, `nonlinear`, `linear-lse`); every sweep so
far used the linear one, and none varied it. With one head there is nothing to disentangle between
scales, but the decoder decides what the model can express and how it is read:
- `nonlinear`: the capacity check (loses the standardized loadings);
- `linear-lse`: DRVI-style, each gene driven by the dimensions that explain it best;
- **own-cell skip (new):** `ŷ = W_c h_global + W_o e_own + b`, where `e_own` is the cell's own
  visible genes through the learned input layer, added around the transformer. It is the additive
  decoder (D3.1) without a GCN. It tests the hypothesis that the global-only models miss the mirror
  programmes (MC1, MC2, MC4, MC6) because their read-out sees the own cell only through the
  transformer, and it gives each cell's own-versus-context split by construction. `e_own` is built
  from the masked input only, with the usual leakage test.
- With pooled tokens (T1b) the same decoders read the patch token, with the own-cell skip at cell
  level.

Order:

1. **D0, the yardsticks:**
   - redundancy: the ridge R² of `h_global` from `h_local`;
   - attribution shares by depth and by programme.
2. **D2.2:** each loss on its regime. Cheap.
3. **D1.1 and D3.1:** a context read-out and an additive linear decoder. Then D1.2, the context
   query, if two layers still leak the cell's own token.
4. **LSE, the description's DRVI-style decoder:**
   - the existing `linear-lse` on each head;
   - a joint LSE over both components' contributions, so each gene goes to the scale that explains
     it best. This is the competitive counterpart of D3.1.
   - Its loadings are each dimension's softmax share per gene, averaged over cells; the
     standardized loadings only hold for a linear decoder.
5. **Orthogonality** (the description's "orthogonality loss between embedding subspaces"). Three
   forms, from weakest to strongest:
   - D4.1, the cross-covariance between `z_l` and `z_g` over a batch: no linear redundancy across
     cells, on top of the winner;
   - the same on uncentred embeddings, `||Z_lᵀ Z_g||²` (the two column spaces orthogonal, not just
     uncorrelated), and on the decoder weights, `||W_lᵀ W_g||²` (each gene's loadings on the two
     scales point in different directions). Two knobs, one weight each;
   - D4.2: run ring NCE and context NCE once, under D0.

   A penalty alone can be met by the global embedding encoding noise or slide identity
   (contrastive_plan.md), so it is read only together with D0's redundancy and with the programme
   scores: it has to lower redundancy without lowering the global head's score on the long-range
   programmes.
6. **D2.1, the targets split by scale:** the alternative if 3–5 do not separate the components.

- **Run.** Sweep 1 in W1 (D2.2, D1.1 + D3.1, joint LSE, the residual decoder as baseline). Sweep 2
  in W2 (D4, D2.1, D1.2 if needed). Both on synth_spot under programme masking, with the disc
  evaluation as a second readout, 3 seeds.
- **Read:**
  - scale assignment against synth_spot's design: short-range should be local; long-range, sparse,
    condition and batch should be global. Report the share of programmes whose dominant component
    matches the design;
  - DRVI-style scores: each programme matched to its best latent dimension, and the gap to the
    second best;
  - no loss in whole-cell reconstruction.
- **In W3:**
  - gene programmes from the separated decoders, read from masked-pass gene scores, with enrichment
    against the panel (K11);
  - Moran's I AUC per dimension on legnini (the manuscript's Figs. 3C and 4I), on the separated
    model.

### T3 — masking strategies (W2)

The design goes into `masking_plan.md` before any code. Every training mask is an opt-in config
switch, with a checkpoint tag and a leakage test through the real model.

| mask | what is hidden | status |
|---|---|---|
| random cells | a fraction of the cells (the manuscript's default) | exists |
| random entries | a fraction of the (cell, gene) entries | exists |
| discs | every cell in discs wider than the local reach | exists |
| programme | a fixed gene set, in every cell | exists |
| random gene sets (channel masking) | a new random gene subset in every cell, at each step | new: the label-free form of programme masking |
| structured | gene sets inside discs; a mixture of masks drawn at each step | new: VirTues-inspired, masking along genes and space |
| by cell type | every cell of a sampled type within a region | new; needs labels |
| same-type attention block (AMICI-style) | no attention between cells with the same label | new: an attention mask; needs labels |
| same-cluster block | as above, with unsupervised clusters fitted on the training slides; or the expression-similarity bias started negative | new; label-free |

- **Leakage through labels.** A same-type block reveals a hidden cell's label: which keys it cannot
  see. Either block only visible queries, or score against a reference that knows the label
  (the label mean plus context).
- **Labels:**
  - CosMx: `cell_type_coarse`;
  - legnini: Leiden clusters, as in the legnini script;
  - synth_spot: a role label (sender, receiver, source, other).
- **Run.** A train × evaluation matrix on synth_spot: every mask × evaluation under programme
  masking, discs and random cells, with 3 seeds, on the DP1 base model. On legnini, a subset of the
  masks, with clusters as labels.
- **Read:**
  - communication: the programme score above `ref_own+context`, and the sender rank;
  - cell-type bias (T5's metric);
  - reconstruction against the smoother, as a no-regression check;
  - sink mass.
- **Decides (DP2):** the default training mask for CosMx.

### T4 — interpretation beyond raw attention (W2; K10)

This is the quantitative analysis of attention-based interpretation that the description asks for.
The design goes into `attribution_plan.md`.

- **One target for every method:** one cell's predicted programme (that programme's response
  genes, centred), as `response_influence` already uses. Never a summed LayerNorm output.
- **Methods:**
  - raw attention, per layer and head, and the head mean;
  - attention rollout (exists);
  - relevance propagation (Chefer et al.) with the valid target;
  - gradient influence (exists);
  - integrated gradients (new), with two baselines: the masked input and the permuted slide.
  - Also: how much of each head's log-attention distance explains (content vs position).
- **Ground truth:** synth_spot's planted edges (short, mid, directional, sparse, dense) under
  programme masking. The score is the matched rank against cells at the same distance, which
  already exists for attention and influence.
- **Sanity checks:**
  - the untrained twin (model randomisation);
  - the permuted slide (data randomisation);
  - agreement across seeds.
- **Models:** the DP1 base model and its backend variants (dense, Performer, linear, Exphormer,
  entmax, top-k), with and without the distance prior. Also a null key and registers
  (attention_plan.md stage 4, built here).
- **Read:** a table of method × backend × edge class. It shows where raw attention agrees with the
  attributions, and where sinks or approximations break that agreement.
- **Then, in W3 (T9):** redo the manuscript's CLS scores, net flow and flow domains with the
  methods that pass.

### T5 — how cell-type driven attention is (W2–W3)

- **The metric** (in attribution_plan.md): for each query, its share of attention or attribution on
  keys with its own label, over the expected share under two nulls:
  - the label composition of its open keys;
  - a distance-matched null, with labels shuffled within distance bins. Same-type cells cluster in
    space, so distance alone already puts mass on the same type.

  Reported per layer, head and receiver label, for attention and for T4's attributions.
- **The CLS.** A condition classifier on the CLS (the graph-level task exists), attributed to cells
  with integrated gradients. Per type, compare its share with the type's shift between conditions
  and its abundance. This tests whether "Alpha and Beta dominate" reflects variance rather than
  interaction. It has to be re-measured anyway, since the original came from the relevance rollout.
- **Data:**
  - CosMx, with annotated types;
  - legnini, with clusters;
  - synth_spot, with roles. Ground truth: receivers should draw on senders, i.e. on a different
    label.
- **Interventions:** T3's same-type block, same-cluster block and masking by type. The blocks
  lower same-type mass by construction. The question is what they do to cross-type recovery
  (synthetic senders) and to the NicheNet support of cross-type pairs (T8).

### T6 — complexity vs tissue coverage (W2–W3)

The description's goal is at least one sparse attention variant as a drop-in replacement for
sliding windows. **CosMx runs without windows** (decided 2026-10-19), on whole sections of
47k–100k cells. Dense attention per cell cannot run at that size. The options are pooling cells into
patches or niches first (T1b: ~100–2,000 tokens, dense and exact), linear or sparse attention on cells,
or exact attention that never forms the S × S map. The approximations below are validated against the
exact reference, and the pooled model is compared with them on cost and programme scores.

What each backend can carry at that size:

| backend | position | cost | caveat |
|---|---|---|---|
| linear | node encodings only (no RoPE, no bias) | O(S) | rank ≤ head width and no relative position: the weakest for spatial signals; a cheap control |
| Performer | RoPE; factor-form biases (Fourier, spectral) | O(S × features) | no distance-profile bias, so T1's per-head distance prior has to be given as a Fourier bias |
| Exphormer | RoPE; every bias, on its edges | O(edges) | a random expander gives each cell a few random partners per layer, so a given distant pair is almost never attended directly. Build the distance-stratified expander (attention_plan.md stage 5), which samples partners at every distance |
| SDPA, exact (new) | RoPE; factor-form biases | O(S) memory, O(S²) compute | the exact reference against which the approximations are validated at full size |

- **Build: SDPA, fused exact attention** (attention_plan.md stage 6). RoPE and the factor-form
  biases fold into q and k, so no S × S tensor is formed.
- **Build: remove what blocks whole sections today.** Stage 8 lists some of these; the rest are in
  the code.
  - `pad_batch` subsamples above `max_seq_len`: set it to the largest section.
  - The dense backends build a dense `[B·H, S+1, S+1]` mask even with the long-range mask off
    (`attn_mask_diagonal`). SDPA runs without a mask (key padding only, and none at batch size 1),
    with a leakage test.
  - The long-range mask densifies the k-hop closure (`_local_reach`). Keep it off at this scale: it
    changed nothing in round 3b or the first sweeps. The kernel backends and Exphormer then block
    only each cell's attention to itself, which is a list of S pairs.
  - `get_model_output` computes the relevance rollout and a dense `_attn_matrix`. Add an inference
    path without them.
  - Attention, influence and integrated gradients run on a sample of queries, without S × S maps.
- **Build: cross-attention to patch tokens** (stretch). Cells attend to a grid of patch tokens with
  positions (the mean of the visible tokens in each patch), at O(S × patches). These are
  attention_plan.md stage 7's inducing points with spatial anchors, and they give a tissue-level
  unit to interpret.
- **Run 1, synth_spot (with ground truth): what windows lose.**
  - Arms: fixed windows of 1.25 and 2.5 mm with dense attention; whole slides with dense, SDPA,
    Performer, linear and Exphormer; all under programme masking, 3 seeds.
  - Read the response scores for receivers whose source lies in another window against those in
    the same window, and the sender rank across window borders.
- **Run 2, CosMx, whole sections.**
  - Steps/s and peak memory per backend, at 47k–100k cells.
  - Each approximation against SDPA on the same sections: programme scores, and how far its
    influence maps agree with exact attention's. This is the validation of the drop-in.
  - Optional: one arm in the manuscript's window setup, as the reference for what whole sections
    add.
- **Decides (DP3):** the backend for the CosMx runs in T7 and T8.

### T7 — disease-specific interactions (W2–W3)

The design goes into `conditioning_plan.md`.

- **Data with known truth.** Extend the synth_spot generator with condition-specific programmes.
  New genes only; existing genes stay byte-identical. For example:
  - a sparse programme that responds only in condition B;
  - one whose sign flips between conditions;
  - a batch that correlates with condition in training but not in testing.

  Two versions:
  - condition visible only through the interaction: an unconditioned model has to average the two
    rules, which is the description's "averaged out";
  - condition also readable from marker genes, so the model can infer it from the slide.

  The generator's rule also gives the true counterfactual: B's rule applied to an A slide.
- **Conditioning modes** (opt-in):
  - a condition token, like CLS and never blocked;
  - FiLM on the tokens;
  - condition-specific low-rank adapters on `W_q` and `W_k`, so only the interaction pattern
    depends on the condition;
  - conditioning in the decoder only (scVI-style), as the baseline;
  - no conditioning.
- **Readout:**
  - per-condition programme scores;
  - a counterfactual swap (predict A slides with B's condition) against the generator's truth:
    does the predicted response switch on at the right cells, and does the influence move to the
    right senders?
  - Condition probes say nothing for a conditioned model, since the condition is an input. Read the
    swap instead.
- **CosMx.** The DP1 base model and the DP2 mask, on whole sections with the DP3 backend. The
  conditioning modes are the arms, on 3 folds.
  - Programme masking on an interferon and MHC class I set, all in the panel: `HLA-A`, `HLA-B`,
    `HLA-C`, `HLA-E`, `B2M`, `STAT1`, `MX1`, `IFIT1`, `OAS1`, `CXCL9`, `CXCL10`. Drivers such as
    `IFNG` and `TNF` stay visible.
  - Channel masking as the label-free default.
  - Swap ND ↔ T1D and read which types' predictions and influences change. Check them against
    known T1D features: islet HLA class I hyperexpression, and immune infiltration of the islets.
  - Each fold holds one donor per condition, so read across the three folds.

### T8 — prior knowledge, downstream (W2–W3)

The design goes into `priors_plan.md`. Python and dataset-agnostic: the ligand–receptor table and
the ligand–target matrix are inputs.

- **Load** NicheNet's v2 human priors: the ligand–receptor network and the ligand–target matrix
  (exported once from .rds). Restrict them to the panel, mapping panel names to HGNC symbols
  (legnini writes `NKX2.2` for NKX2-2).
- **(a) Validation.** For each sender → receiver label pair, compare the model's score with the
  prior support:
  - the model's score is T4's attribution beyond the distance-matched null;
  - the prior support is a ligand expressed in the senders and its receptor in the receivers.

  Report the rank correlation, and the enrichment of the model's top pairs, against nulls with
  shuffled labels and with distance only.
- **(b) Hypotheses.** NicheNet ligand activity (target potential correlated with a gene set, with
  the panel as background), for:
  - each receiver label's global programme (T2);
  - or the genes whose predictions draw most on distant senders.

  The result is a table: sender label · ligand · receptor · receiver label · targets · model
  evidence · prior evidence.
- **Positive control: legnini, SHH → `PTCH1`, `GLI1`, `HHIP`.** SHH should rank first for the
  readouts, and the model's influence should come from the floor plate. Then CosMx, per condition.

### T10 — comparison with other methods (W1 setup, W2–W3; the manuscript's Fig. 2 and Suppl. Figs. 3, 4, 6A)

**Asked for 2026-10-19.** Two parts: what the manuscript already compared (classification and the
other models), and two tasks that are not the project's goal but let InterScale be measured against
dedicated tools: spatially variable gene (SVG) detection and spatial domain detection.

**Protocol for every comparison** (written once, in `comparison_plan.md`, before the first run):
- Same preprocessing (`log1p_norm`), same splits, same metric tables for every method.
- A comparable tuning budget: the manuscript's random search for InterScale and the GCN, each
  baseline's recommended settings plus a small search.
- Splits by donor or patient (`dataset.group_key`), several of them, 3 seeds. The manuscript's own
  finding is that one patient per condition per split gives a high variance; leave-one-patient-out
  or repeated grouped folds replace the single split.
- Cost next to accuracy: parameters, steps/s, peak memory, and for whole slides without windows
  (CosMx, decided 2026-10-19) the same bookkeeping as T6.
- Baselines run in their own environments (`interscale_env` is pip-managed, nothing is installed
  into it). You provide the baselines and the Visium and IMC data on the cluster.

**A. Classification** (Fig. 2, Suppl. Fig. 3):
- *Tasks.* Graph-level condition: Visium AD vs control (Chen et al., GSE220442; 25k spots over 6
  slides, so whole slides of ~4k spots fit, no windows), IMC T1D stages (Damond et al.). Node-level
  cell type on CosMx, per type F1 and the radius robustness (0, 60, 110, 150 µm). Region
  classification on Visium layers. The parameter-space sweep (performance against parameters) is
  rerun for the base model.
- *Models.* GCN (local), PCATransformer (global), InterScale (joint), as in the manuscript, plus the
  DP1 base model, the pooled model (T1b) and the T7 conditioning modes where a condition is the label.
  Baselines the manuscript lacks: logistic regression and an MLP on expression, a neighbourhood
  composition classifier (cell type counts within the radius), embeddings from CellCharter and BANKSY
  with a linear probe.
- *Tissue-level comparison.* The CLS and the attribution-based cell scores (T5) against GraphCompass
  on the pancreas, as in Suppl. Fig. 6A.
- *Read.* Accuracy and macro-F1 over patient-grouped folds with the spread, per-class F1, the
  radius curves. The manuscript's claim to test is that global context helps, especially for rare
  types and region labels, and reduces the sensitivity to the radius.

**B. Communication methods:**
- *Methods.* AMICI (needs labels, one attention layer), Steamboat (no labels; ego, local and global
  heads), and the context-expression models NCEM and MISTy; ligand–receptor tools (LIANA+ or
  CellChat) on CosMx and legnini.
- *Ground truth.* synth_spot's planted edges. Every method yields a sender–receiver score; the
  readout is the matched rank (0.5 blind) and the AUROC against distance-matched cells, the one
  used for InterScale in T4. InterScale enters with the attribution that passes T4 and with raw
  attention, so the table also shows what the manuscript's raw-attention comparison would have said.
- *Real data.* The manuscript's attention-correlation comparison (Suppl. Fig. 4) on CosMx and
  legnini, now next to agreement with NicheNet support (T8) and the SHH control.

**C. Spatial domain detection** (not a goal, but a standard test of the embeddings):
- *Embeddings.* InterScale's local, global and concatenated embeddings and the pooled niche tokens
  of T1b, clustered with Leiden or k-means at the known number of domains. The manuscript's own
  local-component variants (GCN, scVI, PCA, CellCharter, Table X) are the controls.
- *Truth.* synth_spot's `region` and the slide-level structure, Visium layers 1–6 and white matter
  (Chen), legnini's SHH rings (the manuscript's Otsu rings) and Leiden clusters, CosMx islet vs
  exocrine areas.
- *Baselines.* BANKSY, CellCharter, STAGATE or GraphST, SpaGCN, and PCA or scVI with Leiden.
- *Metrics.* ARI and NMI against the annotation, and without one a spatial coherence score (CHAOS,
  PAS). The model is not trained for this, so a result within the field is enough, and I will say so
  where it is not better. The domain scores also pick T1b's `niche` assignment.

**D. Spatially variable genes** (not a goal):
- *InterScale scores.* (i) The context predictability of a gene: the centred cosine of its hidden
  entries under the masked pass, above a slide-mean predictor, per head; (ii) the share of the
  gene's decoder variance that sits on global rather than local dimensions, which is the
  manuscript's local/global gene rank, now from the masked pass; (iii) the Moran's I AUC over
  neighbourhood sizes (Fig. 3C, 4I) as the shared length-scale yardstick.
- *Baselines.* Moran's I (squidpy, installed), SpatialDE, SPARK-X, nnSVG.
- *Truth.* synth_spot: the `is_spatial` flag and the programme scale of each gene (short, long,
  sparse, region), noise genes as negatives. Real data: legnini's SHH-patterned genes and the
  known islet markers on CosMx. A marker of a cell type is spatially variable through composition,
  so the context score is also read within a cell type when labels exist.
- *Metrics.* AUPRC and AUROC against the planted flag, the top-k overlap and rank correlation
  between methods, false positives among noise genes, runtime.

**Setup, W1 Friday.** A `comparison` folder with a standard input (AnnData, split column, label
keys) and one results table per task in `results.md` style; the baseline wrappers
for the first set (GCN and PCATransformer already exist as model types; BANKSY, CellCharter,
SpatialDE, SPARK-X, AMICI, Steamboat, GraphCompass); the Visium and IMC data on the cluster.

### T9 — literature review and write-up (throughout)

**Reading, grouped by question:**
- **Attention-based communication:** InterScale, AMICI, Steamboat.
- **Context methods:** NCEM, NicheCompass, MISTy, Kasumi, CellCharter, BANKSY.
- **Spatial transformers:** SpaFormer, VirTues.
- **Interpreting attention:**
  - attention as explanation (Jain & Wallace; Wiegreffe & Pinter);
  - rollout and attention flow (Abnar & Zuidema);
  - relevance propagation (Chefer et al.);
  - integrated gradients (Sundararajan et al.);
  - sanity checks (Adebayo et al.);
  - attention sinks and registers (Xiao et al.; Darcet et al.).
- **Efficient attention:** Performer, linear attention, Exphormer, BigBird and Longformer,
  FlashAttention, Long Range Arena.
- **Disentangling and conditioning:** DRVI; conditional VAEs (scVI); FiLM.
- **Priors:** NicheNet v2; LIANA+.

**Deliverables:**
- an annotated reading list (W1–W2) and a review draft (W3);
- each track's results read into its plan file, in the same commit;
- a list of the manuscript's claims with what holds up: K10, K11, and T2's programmes.

## Schedule

### W1 (Mon 19 – Sun 25 Oct): a task that can show communication, a model that cannot copy

| day | build | run | read, decide |
|---|---|---|---|
| Mon | T0: seeds, per-arm evaluation | `response_masking` (3 seeds); legnini attention re-evaluation; `joint_rope`, `l2g_res` evaluation | – |
| Tue | T0: `Precomputed` under programme masking; T1: oracle bar, distance prior; T4: gradient × input and integrated gradients for the sender ranks (moved up from W2); **T1b: design note and leakage tests** | `joint_fourier` added to response masking, 3 seeds of the main arms | you: the legnini gene set, the CosMx choices, T1b's patch sizes |
| Wed | **T1b: grid pooling and un-pooling in `GlobalModel` and `CombinedModel`**; T2: D0 | `pre_pca_rope` added to response masking | – (response masking first read done 2026-10-19) |
| Thu | **T1b: decoder arms for the global-only model (own-cell skip, nonlinear, LSE)**; T2: D2.2 | pooled smoke; legnini programme masking (cell-level and pooled) | legnini re-evaluation |
| Fri | T0: the CosMx pipeline; masking_plan.md, attribution_plan.md; **T10: comparison harness and baseline environments** | **T1b sweep (grid sizes × global-only and joint, 3 seeds)**; decoder arms; T10: Visium and IMC data and baselines on the cluster (yours) | – |

### W2 (Mon 26 Oct – Sun 1 Nov): masks, whole slides, conditioning, attribution

| days | build | run | read, decide |
|---|---|---|---|
| Mon | T3: the new masks; T2: D1.1 + D3.1, joint LSE (moved from W1); T1b: `niche` pooling on a frozen local stage | – | T1b sweep, legnini programme masking → **DP1** |
| Tue | T4: integrated gradients, relevance target, null key and registers | T3 matrix (synth_spot); **T10 A: classification, CosMx cell type and Visium condition (base model, GCN, PCATransformer, baselines)** | – |
| Wed | T6: SDPA, the whole-section blockers, query-sampled diagnostics, the distance-stratified expander, windows on synth_spot; **T10 C, D: domain and SVG scoring code** | T4 benchmark on W1's checkpoints; T3 subset (legnini) | – |
| Thu | T7: generator, conditioning modes, swap evaluation; conditioning_plan.md | T6 run 1 (synth_spot); CosMx whole sections per backend (run 2, part 1); **T10 C, D on synth_spot with baselines** | T3 matrix → **DP2**; T10 A |
| Fri | T8: priors, validation, ligand activity; T5 metric; T6 patch cross-attention (stretch) | T7 sweep (synth_spot); T2 sweep 2 | T4 table; T6 run 1 → **DP3** |

### W3 (Mon 2 – Sun 8 Nov): the questions on real tissue, and the write-up

| days | build | run | read, decide |
|---|---|---|---|
| Mon | fixes from W2's reads; T5: CLS classifier with integrated gradients | T8 positive control (legnini, evaluation only); **T10 B: AMICI and Steamboat on synth_spot's planted edges** | T7 synth_spot, T2 sweep 2 |
| Tue–Wed | T2: gene programmes from the masked pass | CosMx main sweep on whole sections with DP3's backend, 3 folds: base model, conditioning modes, same-type block; **T10: IMC classification, domains and SVGs on Visium, legnini, CosMx** | T5 on synth_spot and legnini; T10 C, D on synth_spot |
| Thu–Fri | – | – | CosMx: T7 swap, T5, T8 validation and hypotheses; T10 tables |
| Sat–Sun | – | – | T9: write-up, manuscript claims, review draft |

### Decision points

| | when | decides | on what |
|---|---|---|---|
| DP1 | Mon of W2 | the base model | programme score above `ref_own+context` on long-range and sparse responses, sender rank above 0.5; 3 seeds; pooled (T1b) and per-cell arms compete |
| DP2 | Thu of W2 | the default mask for CosMx | communication readouts without loss of reconstruction |
| DP3 | Fri of W2 | pooled dense attention or per-cell sparse/linear attention for whole CosMx sections, and which backend | programme scores of T1b's pooled arms against per-cell arms at equal cost; agreement with exact attention (SDPA); cross-window responses on synth_spot |
| DP4 | Mon of W3 | the conditioning mode to report | the swap against the generator's truth |

## If we fall behind

Keep at least this much for each goal:

| goal | minimum |
|---|---|
| disease | the synth_spot condition test (token, q/k adapters, none) and one CosMx swap read |
| cell type | the metric on existing checkpoints (legnini clusters, CosMx types); the AMICI-style block on synth_spot and CosMx |
| masking | the synth_spot matrix with random cells, discs, programme masking, channel masking and the AMICI-style block |
| coverage | whole CosMx sections with Performer or Exphormer (checked against SDPA if it is built); synth_spot windows against whole slides (dense, Performer) |
| disentangling | D2.2, D1.1 + D3.1, LSE and the cross-covariance term, on synth_spot |
| priors | the legnini SHH control and the CosMx validation table |
| interpretation | integrated gradients, influence and raw attention on planted edges |
| comparison | classification on CosMx (cell type) and Visium (condition) against GCN and PCATransformer; SVGs on synth_spot against Moran's I and SPARK-X; domains on synth_spot and Visium against BANKSY and CellCharter; AMICI and Steamboat on the planted edges |
| literature | the annotated reading list |

Drop first:
- patch cross-attention, and stage 7's free inducing points;
- FlexAttention;
- D3.2 (gated mixture) and D3.3 (FiLM decoder);
- contrastive two views (contrastive_plan.md stage 3);
- C3's six missing arms;
- calibrated magnitude (0c);
- the linear backend at whole-section scale.

## Decisions for you

1. **legnini programme set.** Proposed: `PTCH1`, `GLI1`, `HHIP` (direct SHH targets), with `SHH`
   visible. A second arm: the ventral identity genes `NKX2.2`, `NKX6.1`, `OLIG2`. Optionally
   `MSX2` as a BMP4 readout, the dorsal control.
2. **CosMx setup.**
   - Label: `cell_type_coarse` (9 types) or `CellTypes_max` (14).
   - Graph radius: the manuscript used 60, 110 and 150.
   - Folds: the three provided splits, or two training slides per fold.
3. **CosMx programme set.** The interferon and MHC class I set above, or channel masking only.
4. **C3's six missing arms.** Drop them (proposed), or run them.
5. **The tissue-level CLS analysis** (condition classifier with integrated gradients). Is it in
   scope? It is the only way to retest the manuscript's Fig. 4C/D.
6. **NicheNet route.** The v2 priors, exported once from R to CSV (proposed), or LIANA+'s
   resources in Python.
7. **D1.** The read-out that is exact for one layer first (proposed), or the context query
   directly.
8. **T1b.** Patch sizes (proposed: synth_spot 200/300/500 µm; legnini 50/100 µm; CosMx 100/200 µm),
   `grid` first and `niche` second (proposed), and whether W1's other items may slip to W2 to make
   room (D1.1 + D3.1 and joint LSE already moved).
9. **CosMx coordinates.** Is `obsm['spatial']` in µm? The two coordinate sources give token counts 4×
   apart, which matters for the patch sizes.
10. **T10 methods and data.** Proposed baselines: classification GCN, PCATransformer, logistic regression, MLP,
    composition classifier, CellCharter/BANKSY probes, GraphCompass; communication AMICI, Steamboat, NCEM, MISTy,
    LIANA+; domains BANKSY, CellCharter, STAGATE or GraphST, SpaGCN; SVGs Moran's I, SpatialDE, SPARK-X, nnSVG.
    Is a foundation-model baseline (scGPT, Nicheformer) wanted? I left it out.

## Rules for every track

- **Defaults stay the same.** Every option is a config switch whose default leaves the model
  unchanged (harness IDENTICAL), and every switch has a checkpoint tag.
- **No leakage.** Hidden values never reach a prediction. Every new mask, model type, conditioning
  mode and phase gets a test through the real model and trainer (the `b7cf317` lesson). Masks built
  from labels also guard the label.
- **Seeds.** Decisions rest on 3 seeds or folds, with arms compared paired by seed.
- **What to read, in order.** Communication first: programme scores above `ref_own+context` and
  sender ranks, against the untrained twin and the permuted slide. Reconstruction against the
  smoother only as a no-regression check. Use centred cosine and SCE, not R².
- **Dataset-agnostic evaluation.** Labels, gene sets, conditions and priors come from the caller.
- **Outputs.** Each sweep writes one `results.md`. Each read goes into its track's plan file, with
  synthetic spatial maps (predicted vs true) next to the numbers.
