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

Last updated 2026-10-22.

| track | question | state | verified by | date |
|---|---|---|---|---|
| T0 | infrastructure: seeds, per-arm evaluation, inputs, CosMx | **implemented, unverified** (2026-10-19; CosMx deferred by the user) — seed axis, run selection for job arrays, per-run section files, timing, cached references, `Precomputed` under programme masking, `pca_fit: train`; every mutation caught (two survivors led to a stronger test and a simplification), harness IDENTICAL; the sweep script checked on a stub synth_spot with untrained models. **Not run on data** | 2026-10-19 |
| T1 | does the global model learn communication at all | second read 2026-10-20 (3 seeds, 19 runs): `joint_fourier` global head 0.820 ± 0.005 against the bar 0.682 and its local head 0.506, rings +0.3; per programme 3 wins / 12 ties / 9 losses against the bar (mirror programmes, M11, directional); pooled 500 µm loses the sparse responses; legnini programme masking built 2026-10-20 (`legnini_programme`, decided gene set in legnini_plan.md; not run); sender ranks unreadable (gradient magnitude) → replaced 2026-10-20 by an integrated-gradients attribution, sparse programmes also scored within their receivers, per-programme wins/ties/losses in the report (**built, not yet run: the 19 checkpoints await `MODE = "evaluate"`**). Details in global_plan.md. **Oracle bar built 2026-10-21** (`ref_oracle`, below): a ridge on the true drivers reaches 0.983 over the programmes on synth_spot, against 0.685 for `ref_own+context`, checked without any model. **Model-side options built 2026-10-21** (implemented, unverified on data): a distance prior with a length scale per head (`pe.distance.kind: scales`) and visible keys only (`attention.visible_keys_only`); 31 tests, 20 mutations caught, harness IDENTICAL. No arms yet: they are devised once the pending runs are read | `response_masking` run (11 arms, 19 runs, 3 twins) | 2026-10-20 |
| T1b | coarse-grained tokens: pool cells into patches or niches before the global model | **grid pooling implemented, unverified** (2026-10-19) — `pooling.kind: grid` (`size_um` or `n_patches`), `count`/`std` features, 39 tests, an end-to-end run for GlobalModel and CombinedModel, every mutation caught (one survivor led to a stronger test), harness IDENTICAL; 5 pooled arms in `response_masking`. The `niche` assignment waits for the user's details; patch-level attention readouts not built. **Not run on data**. **2026-10-21: the decoder arms for the global-only model built** (own-cell skip `model.decoder.own_cell_skip`, `nonlinear` and `linear-lse` as sweep arms, the skip also on pooled tokens), implemented, unverified; sweep `global_decoders` (10 runs) **run and read 2026-10-22** (notes.md): pooled tokens with the skip match the per-cell model's programme mean (0.582 against 0.577, one seed) at about 1/25 of the tokens and 0.14 against 0.77 GiB; without the skip pooling loses the sparse programmes (0.00-0.03); the nonlinear and LSE decoders are worse. **2026-10-22: the full-rank skip `own_cell_skip_input: raw` built**, arms `lin_fourier_pool500_skipraw` and `lin_fourier_skipraw` added to the sweep, **not run** | 2026-10-22 |
| T2 | disentangling local and global ([decoder_plan.md](decoder_plan.md)) | proposed; **re-masking before the transformer implemented, unverified** (2026-10-21, task 3 below) -- `remask: hidden`: a wholly hidden cell enters the transformer as one learned token instead of its GCN token; 16 tests, 8 mutations caught, harness IDENTICAL; sweep `remask_test` (8 runs) **run and read 2026-10-22** (notes.md): the mechanism works (with the long-range mask a rim cell's global head draws 94% of its influence from beyond 300 µm, was 7%; CKA local~global at the rim 0.89 → 0.78), at a cost of 0.02-0.05 at every depth with the mask on and none without it (where the copying moves into attention); no head beats the smoother. **D2.2, each head's loss on the hidden cells of its regime, implemented, unverified** (2026-10-21) -- `optim.regime_loss`, near / deep by the distance to the nearest visible cell; 55 tests with the own-cell skip (34 regime loss, 21 skip and decoders), 32 mutations all caught (two only after two more tests), harness IDENTICAL; sweep `regime_loss` (9 runs) **run and read 2026-10-22**: the heads specialise as assigned (the local head near the rim 0.443 → 0.483, the global head loses the rim 0.428 → 0.385 and holds deep) but no head beats the smoother in its own regime, and discs of 1000 µm make the global head worse everywhere (0.40 → 0.34). **D0, the yardsticks, implemented, unverified on data** (2026-10-21): ridge redundancy of the two embeddings per depth group, attribution shares of a head's parts (`StepOutput.parts`), 15 tests. **D3.1, the additive decoder of the dual model, implemented, unverified** (2026-10-22): `model.decoder.additive`, `y = W_l h_local + W_g h_global + b` with the loss on the sum, the parts for `attribution_shares`; 22 tests, 41 mutations caught (with the raw skip and the probe cost), harness IDENTICAL; sweep `additive_decoder` (10 runs) written, **not run**. **Cheaper disc evaluation** (2026-10-22): `max_train_rows` and `joint_embeddings` for the label and distance probes, `PROBE_*` in `sweep_common.py`; the new numbers are not comparable to the first disc sweeps'. **D1.2, the context-only global embedding, implemented, unverified** (2026-10-22): `model.global_component.parameters.context_only: query` -- the transformer's queries from one learned token plus the node encoding, reading the cells' input tokens, so a cell's global embedding does not hold the cell (bit for bit; 1 / S with `center_qk`); 36 tests, 38 mutations, harness IDENTICAL; sweep `context_query` (13 runs + 2 twins) written, **not run**. Moran's I split (task 1) and embedding dimensions vs genes (task 2) proposed | tests, harness | 2026-10-22 |
| T3 | masking strategies | **channel masking implemented, unverified** (2026-10-22): `dataset.mask_channel_fraction` (30% first, 50% the same code), 31 tests, 36 mutations, harness IDENTICAL; sweep `channel_masking` (8 runs, evaluated on the programme set via `EVAL_MASK_GENES`) written, **not run**. The other new masks (structured, by cell type, same-type blocks) proposed | tests, harness | 2026-10-22 |
| T4 | interpretation beyond raw attention; attention sinks | **sender attribution built, unverified on data** (2026-10-20) — `response_influence(measure="attribution")`: integrated gradients from the training-mean cell on a programme's summed response genes, completeness check, hidden entries fixed, one random state along the path; 16 + 6 tests, 31 mutations caught. Not built: the other methods of the table below, the null key and registers, the permuted-slide baseline. **Not run on data**. **2026-10-21 (task 4 below): attention checked against torch and the readouts compared; run and read 2026-10-21 on the 13 + 3 `response_masking` checkpoints (notes.md): the attention is computed correctly (zero difference to the definition and to torch), no readout finds the planted senders (ranks 0.47-0.55, twins 0.48-0.52), gradient x input is the only one above its twin (+0.04-0.05), RoPE attention ranks the mid-range sender far below chance (0.09-0.35; pair counts missing, now kept)**: the vendored attention equals torch's stock `nn.MultiheadAttention` and encoder layer (tests, every mask form); relevance fixed (heads averaged per graph, tensors on the output's device, a target can be given); `evaluation.attention_check` (`check_attention` on the real inputs, `attention_readouts`: 7 readouts against planted pairs); 40 tests; of 15 mutations 14 caught (2 after two more tests), the survivor a redundant gradient clear; sweep `attention_check` (evaluation only, on the `response_masking` checkpoints), **not run**. Label-free flow (task 5) proposed | 2026-10-21 |
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

## Added 2026-10-21: five high-priority tasks

Asked for on 2026-10-21, sorted into the tracks below (details there):

| # | task | track | state | when |
|---|---|---|---|---|
| 1 | **Moran's I split:** how a gene's Moran's I changes as the neighbourhood grows; a fast decay marks local variability (a likely local interaction), a slow one global variability; group the genes so, and check whether the local or the global component picks each group up | T2 (a label-free yardstick for the scale assignment, real data included), feeds T10 D | proposed | build W1 Thu, run with the remask and decoder sweeps |
| 2 | **Embedding dimensions and genes:** the most important dimensions of each learned embedding, the genes correlating with each, and whether a gene's expression and the dimension's value agree across cells and in space, especially for genes of interest | T2 (read-out of the separated components), T9 (the manuscript's gene programmes) | proposed | build W1 Fri |
| 3 | **Re-masking before the transformer:** the dual model masks cells once, before the GCN; a hidden cell then hands the transformer its GCN token, built from its visible neighbours. Re-mask it before the transformer too | T2 (K2) | **built 2026-10-21**, sweep `remask_test` not run | run W1 Thu, read W1 Fri |
| 4 | **Is attention computed correctly:** the module that takes the gradient into account (relevance), torch's attention against the custom one, averaging against gradient-based readouts | T4 (K10) | **built 2026-10-21**, sweep `attention_check` not run | run W1 Thu, read W2 Mon |
| 5 | **Flow without annotation:** vector fields of sender→receiver dynamics in space that need no cell-type labels | T4 (K9), feeds T5, T8 | proposed | design W2 Mon after task 4's read, build W2 Tue |

Tasks 1 and 2 answer the same question from two sides as T2's other readouts: which genes each component carries,
read without synth_spot's planted design, so they also work on legnini and CosMx.

## Known problems of the global model, and the fix we try

| # | problem (observed) | what we believe causes it | fix we try | track, week |
|---|---|---|---|---|
| K1 | Global never beats the best smoother at disc reconstruction (synth_spot 0.459 vs 0.465). On legnini it sits at the slide mean (0.50 vs 0.594 for the 40 µm kernel) | A plain average of the visible cells solves the disc task. On legnini attention cannot become local: the biases start at 0 and learned ≤ 0.03 | Programme masking on synth_spot and legnini; a distance prior with a length scale per head; targets split by scale (D2.1) | T1, T2; W1 |
| K2 | Near visible cells the global head copies the local one (CKA 0.93 at the rim) | Both heads predict one target, and the cell's own token stays in the transformer's residual stream; a hidden cell's token is its GCN token, made of its visible neighbours | Re-masking before the transformer (task 3, built 2026-10-21): a hidden cell's token is one learned token, so its own residual stream holds nothing of its neighbourhood; a context-only global embedding (D1.2, **built 2026-10-22**: every cell, hidden or not, queries from a position); an additive decoder (D3.1, **built 2026-10-22**); each loss on the cells where its component has to work (D2.2); pooled tokens, whose embedding is shared by a patch and cannot carry one cell's own token (T1b) | T1b, T2; W1 |
| K3 | The residual decoder hurts the deep field (0.389 vs 0.438 at 290–500 µm) | In the blind area the local prediction is a constant that the residual has to undo | Retire it in favour of D3.1; keep it as one baseline arm | T2; W1 |
| K4 | No sender selection: planted pairs at ~0.5 for both attention and influence; attention is broad | No task so far needed a sender. With ~2,500 keys, a few relevant ones need a logit lead of ~6 | Tasks that need senders (programme masking); the distance prior (**built 2026-10-21**, `pe.distance.kind: scales`); sparse normalisers and top-k retested there; a null key and registers | T1, T4; W1–W2 |
| K5 | Hidden tokens collapse (token cosine 0.78–0.91 under discs) and act as a sink: without RoPE, visible cells attend almost only to them | Every wholly hidden cell gets the same input | Programme masking (no cell is wholly hidden); an option to use only visible cells as keys (**built 2026-10-21**, `attention.visible_keys_only`); a null key | T1, T3, T4; W1–W2 |
| K6 | Inputs of the global-only model: the learned linear input fails without RoPE (0.15); the in-pipeline PCA is fitted on one masked validation slide (0.31); the best input, a precomputed PCA (0.44), is refused under gene masking | A linear map of a −1-filled row carries no position. The PCA is fitted on the wrong data. The leakage guard cannot know which genes an embedding saw | Allow `Precomputed` under programme masking when the embedding was computed without the hidden genes (recorded with the embedding and checked); fit the in-pipeline PCA on the training slides (opt-in) | T0; W1 |
| K7 | Single seeds; differences of ±0.02 are unresolved (e.g. scVI + Fourier's +0.035 on response genes) | – | A seed axis in every sweep; 3 seeds for every arm a decision rests on; on CosMx, its three splits as folds | T0; all |
| K8 | Evaluations die at the job limit (`global_only` and `staged_training`, twice) | Every arm is trained and evaluated in one job | One evaluation job per arm; evaluate `joint_rope` and `l2g_res`; drop C3's six missing arms (H-C is already rejected) unless you want them | T0; W1 |
| K9 | legnini: the flow maps are not readable, the SHH source (floor plate) is avoided, and no rule transfers between the two test organoids | No model beats the local kernel, so there is no learned flow to read; there are only two test organoids; the label heatmaps need an annotation, and patch fields average senders and receivers of every programme together | Re-evaluate with the rewritten flow; SHH programme masking; read flows only from a model above the kernel; rotate the test organoids as folds; a label-free flow per programme from a validated attribution (task 5) | T1, T4; W1–W2 |
| K10 | The relevance rollout is degenerate, and so is everything built on `_attn_matrix` | It takes the gradient of a summed LayerNorm output. Found 2026-10-21: it also averaged the heads over every graph of a batch together, and built its identity and mask on the CPU (fixed; `evaluate()` runs one graph, so no number made so far changes) | A valid target (one cell's predicted programme; `generate_relevance(..., target=)` since 2026-10-21), checked in T4's benchmark (`attention_check`); redo the manuscript's CLS and net-flow analyses with the methods that pass | T4, T9; W2–W3 |
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
  **Built 2026-10-21 (implemented; checked without a model, not yet read for any trained arm).**
  - *What it is.* `ref_oracle`, one more fitted reference (`masked_probing.FittedReferences(drivers=...)`): the same
    ridge, penalty search and training slides as `ref_own+context`, on the spot's own visible genes plus the features of
    `masked_probing.oracle_features`: each numeric obs key (a distance to a source) as 24 Gaussian bumps at quantiles of its
    values, with an indicator where it is missing (M4, M5, the M9 stripe exist on some slides only); a flag as it is;
    another categorical one-hot; and a key's features also multiplied by a boolean gate (the sparse responses exist only at
    receivers, the M10 response only in its cone). The sweep script names the keys (`ORACLE_KEYS`, `ORACLE_GATES`,
    `ORACLE_BASIS` in `sweep_common.py`); `response_masking.py` and `global_decoders.py` name synth_spot's 28.
  - *Where it appears.* In the head-score tables as `ref_oracle`, beside the other references but never as "the best
    reference" (the bar the wins / ties / losses count against); the summary adds a second wins table against it.
  - *What it gives* (fresh synth_spot, the sweep's pipeline, 24 training slides, test slides, no model; fit 2 min): mean
    centred cosine over the 24 programmes **0.983**, against 0.685 for `ref_own+context` (the sweeps' bar, 0.682) and 0.580 for
    `ref_own`. At least 0.94 on every programme but the cone M10→MC13 (**0.83**; its response is a plateau and a decay inside a
    cone, which 24 bumps resolve only roughly). Where the bar is far below it: M9 (0.27 → 1.00), M10 (0.16 → 0.83), the
    short-range S1/S2/S3S4→R (0.65, 0.66, 0.55 → 1.00, 1.00, 1.00), M3 (0.46 → 0.98), M5 (0.58 → 0.98), M7/M8 rings (0.63, 0.76
    → 1.00). So the task is solvable from the planted drivers, and a programme a model misses is the model's limit, not the
    task's.
  - *Caveats.* It is a bar built on the truth, not a model: it uses `is_receiver_*` and `R34_state` as they are, where a model
    must infer them. Its penalty sits at the grid's lowest value (0.1), so the noiseless drivers may want less
    regularisation than `FITTED_ALPHAS` allows. Changing the keys refits the references (the cache key includes them).
- **Build: a distance prior with a length scale per head.** Start the linear distance bias (it
  exists, in ALiBi form, and starts at 0) at fixed per-head length scales from ~20 µm to ~2 mm,
  optionally frozen.
  - Some heads look locally and some globally from step 0: Steamboat's multi-scale heads, without
    labels.
  - Each head's scale is known, which makes its maps readable.
  - Aimed at K1 (legnini cannot even reproduce the 40 µm kernel) and K4.
  - **Built 2026-10-21 (implemented, unverified on data):** `pe.bias: [distance]` with `pe.distance.kind: scales`
    (`ScaledDistanceBias`). Head `h` gets the window `-strength * phi(d / l_h)`, `phi(u) = u²/2` (`shape: gaussian`, a
    Gaussian attention window of standard deviation `l_h`) or `u` (`exponential`), the scales `l_h` spaced geometrically over
    `[scale_min_um, scale_max_um]` (default 20 µm to 2 mm: for 4 heads 20, 93, 431, 2000 µm; one head takes the geometric
    mean). Unlike every other bias it is not zero at step 0: it is a prior. `log l_h` and `strength` are learned unless
    `frozen: true` (then buffers). Dense and pair form (dense, entmax, top-k, Exphormer); no low-rank form, so not the
    Performer. With the long-range mask on, set `scale_min_um` about at the local reach (~290 µm on synth_spot): a head far
    below it sees only keys it penalises by hundreds of logits and attends to the nearest. Checkpoint tag
    `distance-scales-<g|e><min>-<max>[-s<strength>][-fz]`.
- **Build: visible keys only** (discs). Wholly hidden cells are queries but never keys (K5).
  **Built 2026-10-21 (implemented, unverified on data):** `attention.visible_keys_only: true`. The dense attention mask gets
  every key that is a cell hidden whole (`GlobalModule.wholly_hidden`) blocked for every query (laid out like the tokens with
  `pad_like`, so left padding and subsampling stay aligned); CLS stays open, so no row is empty, and a hidden cell still queries.
  Not on a clean pass. Needs a dense-mask backend and cells hidden whole, so it is refused with pattern backends
  (Performer, linear, Exphormer), pooling, a hidden gene set, gene masking without discs, or nothing masked. Tag `-vk`
  in the attention part of the checkpoint name. Natural pair with `remask: hidden` (the hidden tokens are then one identical
  token that nobody attends to).
  - *Tested:* 31 tests (the scales and windows; the real attention with the content logits zeroed matches each head's window
    and wider scales look further; hidden keys get exactly 0 weight in every layer and head on a batch of two padded graphs;
    a dual-decoder model trains through `train()` with both on); 20 mutations, all caught (three only after stronger tests);
    harness IDENTICAL.
- **legnini.**
  - Re-evaluate the attention sweep with the rewritten flow.
  - Programme masking, **decided 2026-10-20**: the whole dorsoventral identity code (15 genes) is the main
    task and the SHH readouts `PTCH1`, `GLI1`, `HHIP` are the positive control, hidden together in every
    cell; `SHH`, `FOXA1`, `SMO`, `GLI2`, `GLI3` and `BMP4` stay visible. The genes, why each was chosen,
    and where the paper and the manuscript mention them are in [legnini_plan.md](legnini_plan.md).
    Built as `legnini_programme` (7 runs: `local_gcn`, `lin_fourier`, `joint_fourier` at 3 seeds, and the
    pooled 100 µm `lin_fourier_pool100` and `joint_fourier_pool100`); not yet run. The earlier idea of
    PCA + RoPE arms with the distance prior is dropped.
  - Positive control: the readouts' predictions should draw on the floor plate (c13, SHH) more
    than distance alone explains. **Weak in this data** (legnini_plan.md): in the SHH test organoid PTCH1
    and GLI1 correlate slightly positively with the distance to SHH+ cells, while FOXA2 (-0.41), NKX2.2
    (-0.19), PAX6 (+0.33) and IRX3 (+0.27) are ordered around the source as the paper describes.
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
  attention (exact, with maps and attribution) runs without windows. synth_spot (100 µm lattice): 2,500 spots
  become 625 / 278 / 100 tokens at 200 / 300 / 500 µm, holding 4 / 9 / 25 spots each; 500 µm it is
  (decided 2026-10-19: the smaller patches held too few spots). legnini: ~4,300 cells become ~100–400 tokens at
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
- *Arms in `response_masking`:* `lin_fourier_pool500`, `lin_rope_pool500`, `joint_fourier_pool500`
  (500 µm = 5 x 5 spots, about 100 tokens per slide; the joint arm has no long-range mask: patches have no
  neighbour graph; its local head reads cells). 200 and 300 µm were dropped by the user's decision.

**Run, W1 (synth_spot, programme masking, discs as the second readout).**
- Arms: `lin_fourier` and `lin_rope` pooled on a 500 µm grid, and the joint model pooled the same way
  (the local head reads cells, the global head the patches; the additive un-pooling is the sum of the
  two heads' decoders of decoder_plan.md D3.1, not built); the cell-level arms from the first read as
  references. 3 seeds of `lin_fourier_pool500`. `niche` pooling on W2 Monday–Tuesday, once the frozen local stage exists.
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

**Built 2026-10-21 (implemented, unverified).**
- *Own-cell skip:* `model.decoder.own_cell_skip: true` (a `GlobalModel` only; a combined or local model refuses it).
  `ŷ = decoder(h_global) + W_o e_own`, `e_own` the cell's token before the transformer from the masked input
  (the learned input layer's output, or the precomputed / PCA embedding), gathered into token order like the
  decoder's rows; `W_o` starts at zero and draws no random number, so the model starts as without it and every
  other weight is initialised as before. `GlobalModule.predict` takes `own_embedding` and `node_idx` and refuses
  to run without them rather than predict as if there were no skip; `get_model_output` decodes with it too.
  Tested: it adds exactly `W_o e_own` (also pooled, where it is the only per-cell part), a hidden entry never
  reaches a prediction through it while a visible one does, and a model without tokens of its own refuses it.
- *Decoder types:* a checkpoint name now carries a decoder other than the plain linear one or the skip
  (`dec-nonlinear_`, `dec-linear-lse_`, `dec-linear-skip_`): the decoder type was never in the name, so arms that
  vary it would have overwritten each other. Every earlier name is unchanged. Each type trains finite through a
  global-only model (tested, with and without the skip).
- *Sweep `global_decoders`* (slurm folder; `lin_fourier` of `response_masking` read out six ways): `lin_fourier`,
  `lin_fourier_skip` (3 seeds each), `lin_fourier_nonlinear`, `lin_fourier_lse`, `lin_fourier_pool500`,
  `lin_fourier_pool500_skip` (one seed each); an untrained twin of `lin_fourier` only (the skip starts at zero, so
  its twin would be the same model). **Read:** per programme against `ref_own` and the bar `ref_own+context`, first the
  mirror programmes (MC1, MC2, MC4, MC6: 0.36 against `ref_own` 0.68 for MC4) and the sparse ones; the ring and
  long-range programmes (M7, M8: 0.95) must not drop; the planted senders' rank against the twin.

**Built 2026-10-22 (implemented, unverified): D3.1, the full-rank skip, cheaper probes.** Asked for after the reads of 2026-10-22.
- *D3.1, `model.decoder.additive`* (dual decoder, node level, not with `global_residual`): `y = W_l h_local + W_g h_global + b`. The
  stacked output keeps its halves -- the **local half is the local decoder alone, the global half the sum** (the reported prediction,
  as under `global_residual`), no gradient cut. `optim.loss_weights` must be chosen (the validator refuses 0.5 / 0.5 under the
  `joint` schedule, which would also train the local part to predict the target alone): `{local: 0, global: 1}` is the loss on
  the sum only, `{0.1, 0.9}` a small per-head term; a staged schedule sets its own per phase. `StepOutput.parts["global"]` hands
  over the two outputs (the sum's own tensors) for `attribution_shares`; the evaluation adds the head `global_part` (the global decoder alone), which
  the influence readouts differentiate too (a test runs `response_influence` through an additive model).
  Checkpoint tag `add`. It separates nothing by itself (the global tokens are built from the local ones): the shares say how much
  each part carries, and D1 is what removes the global part's ability to repeat the local one.
  *Tested:* 22 tests (the sum and the halves, no state added and the same initial weights, the gradient reaching both decoders and
  the GCN against the residual head's cut, the parts, no hidden entry in either half, inference, `collect_head_predictions` and the
  shares adding to 1, pooled tokens sharing the global part within a patch, the config and name, refusals, a real `train()` run and
  one under `local_then_global`).
  *Sweep `additive_decoder`* (slurm folder, 10 runs; programme masking, `joint_fourier`'s model): `joint_fourier` and `add_fourier`
  (loss on the sum) at 3 seeds; `add_fourier_l01`, `add_fourier_l2g` (staged), and the pooled `add_fourier_pool500` /
  `joint_fourier_pool500` at one. **Read:** the attribution shares per programme (short-range on the local part, long-range on the
  global part?), the sum's programme scores against `ref_own+context`, the oracle and the plain model's global head (first the
  short-range programmes and M11, then that the rings do not drop), `local` and `global_part` alone, the senders' ranks.
- *D1.2, `model.global_component.parameters.context_only: query`* (default `none`): the transformer's queries are `norm_input(context_token +
  node encoding)` for every cell (CLS as the CLS token), never the cell's own token; in each layer they read the cells' **input** tokens
  through that layer's weights (`read_context`: `q + attention(norm1(q), norm1(memory))`, `q + feed_forward(norm2(q))`), and the output is the
  encoder's final norm. The memory is the same at every layer and nothing is written into it, so a cell's global embedding is a function of the
  other cells' tokens and its position; with the long-range mask, of cells beyond the GCN's reach. Checkpoint tag `ctx-query`.
  *Tested:* 36 tests: a cell's embedding does not move, bit for bit, with its own expression (1-3 layers; node encoding, bias, RoPE; a pooled
  patch; behind a GCN with the long-range mask, where without the mask it does), the plain transformer's does; the read-out equals its definition
  written out; the queries follow the kept cells of a subsampled slide; padding is never read; blocked pairs get weight 0; gradients reach the token,
  the layers and CLS; refusals (other backends, graph level, unknown kind, forward without queries, rollout / relevance readouts); a global-only
  model with the skip and a dual additive model train and infer; a toy where the other cells determine the target is learned (0.5-0.7) and a copy of
  the cell's own gene, which the plain transformer learns (0.99), is not (0.0). 38 mutations caught. *Exception, documented:* `attention.center_qk`
  centres on the mean over all cells' tokens, the cell's own included at weight 1 / S (falls with S, tested). *Sweep `context_query`* (slurm folder, 13
  runs + 2 twins; programme masking): global-only `ctx_fourier`, `ctx_fourier_skip` (3 seeds), `ctx_fourier_skipraw`; dual `add_fourier` (3 seeds),
  `add_ctx_fourier` (3 seeds, the main bet), `add_ctx_fourier_nocenter`, `ctx_joint_fourier`. **Read:** `ctx_fourier` against `lin_fourier` (the
  long-range and ring programmes hold, the mirror, short-range and sparse ones fall?); the skip arms against `lin_fourier_skip`; the attribution
  shares of `add_ctx_fourier` against `add_fourier` per programme; the planted senders' rank. *Caveat:* two linear parts add, they do not gate, so
  the sparse programmes (the product of the spot's receptor and a distance) are out of reach of the pair; D3.2 / D3.3 are for that.
- *`own_cell_skip_input: raw`* (default `embedding`): the own-cell skip reads the cell's visible genes (hidden entries at the fill, as in
  every input) instead of its token, so `W_o` is `[n_output, n_input]` with full rank, where the token's has at most `n_embed`.
  The first read had the skip closing about half the gap to `ref_own` on the mirror programmes: if `lin_fourier_skipraw` (3 seeds, in
  `global_decoders`) reaches `ref_own`, the limit was the skip's width, otherwise its training. Checkpoint tag `dec-<type>-skipraw`.
- *Cheaper probes (K8's cost, "results must come fast").* The label and distance probes were about 25 of the 30 minutes of an arm's
  disc evaluation. `probe_labels` / `probe_covariates` take `max_train_rows` (each probe fitted on one seeded random subset of the
  hidden training rows, the same for every set and label; every test row is scored) and `joint_embeddings=False` (without
  `local+global` and the sets built on it, 3 of 12); `sweep_common.py` sets them through `PROBE_MAX_TRAIN_ROWS`, `PROBE_MAX_ITER`,
  `PROBE_JOINT_EMBEDDINGS`, and `PROBE_FITS = False` skips the probes altogether (e.g. to evaluate old checkpoints for the
  redundancy and attribution tables only). 9 tests (a capped probe equals the uncapped one on the subset taken by hand, one draw per call,
  the test rows never cut), 10 mutations caught. `regime_loss.py` and `remask_test.py` use 6,000 rows, 300 iterations and no joint sets;
  the numbers move a little against the first disc sweeps (notes.md, 2026-10-22: on synthetic arrays 3.9 x faster at 6,000 rows; the
  real cost could not be reproduced, so the speed-up on the cluster is unmeasured).

Order:

1. **D0, the yardsticks:** **built 2026-10-21 (implemented, unverified on data).**
   - redundancy: the ridge R² of `h_global` from `h_local`;
   - attribution shares by depth and by programme.
   - *Redundancy* (`representations.redundancy_table`, in the disc evaluation next to the CKA): per query group (visible,
     hidden, each depth bin) a ridge fitted on the group's training cells maps the source to the target (z-scored; a source
     wider than 64 columns is cut to its leading components of the training cells; the penalty by leave-one-out), scored by
     R² on the group's test cells. Pairs: `global <- local`, `local <- global`, and as controls `global <- smoother` and
     `global <- pe` (what the visible expression's average, or the position alone, already gives). `global <- local` near 1 on a
     group says its global tokens hold nothing the local ones do not. Each group is fitted on its own cells, so the question
     is whether a *hidden* cell's global token is a function of its local token, not whether a map learnt on visible cells
     carries over.
   - *Attribution shares* (`masked_probing.attribution_shares`, in both evaluations): a head whose prediction is a sum of named
     parts is taken apart. Parts come from the modules (`StepOutput.parts`): a global-only model's transformer read-out
     (`context`) and its own-cell skip (`own_cell`); a residual dual decoder's `local` (detached) and `global` part; later
     D3.1's two parts. Over a group of genes and a set of cells (all, then each depth bin), the parts are centred per gene
     over the hidden entries; `cov_share` is each part's share of the whole's variance (sums to 1, negative for a part
     against the others), `norm_ratio` its norm over the whole's. A head with one part has none (a plain dual decoder's heads).
     Needs `global_residual` or the skip to show anything today; it is the readout D3.1 and the regime sweeps will use.
   - *Tested:* 15 tests (constructed embeddings where the answer is known; the parts through the real modules and
     `collect_head_predictions`; the report); 25 mutations over the oracle and D0, all caught (four only after stronger tests).
2. **D2.2:** each loss on its regime. Cheap. **Built 2026-10-21 (implemented, unverified):**
   - *Regimes.* A cell hidden whole is *near* when the nearest visible cell of its slide lies within
     `optim.regime_loss.depth_um` (default 290 µm, the 2-hop GCN's reach on the 100 µm lattice; you do not want it
     lower), *deep* beyond (`tl/depth.py`; `hidden_depth` moved there from the evaluation module, now chunked, and is
     still importable from `online_probes`). `local_cells` and `global_cells` name the regime of each head's loss:
     `near`, `deep` or `all`; the default is near / deep, `all` / `all` is the loss as before.
   - *Mechanism.* The module builds a per-head loss mask (`StepOutput.loss_mask`, the entry mask cut to each head's
     cells) and the training plan hands that to `compute_separate_losses`; the metrics keep every hidden entry, so
     the logged `*_masked_*` numbers mean what they always did. Each head's loss is the mean over its own cells and
     the heads are weighed by `optim.loss_weights` (0.5 / 0.5), not by how many cells each regime has (one mean over
     the stacked rows would weigh them 40 : 9 on a typical disc). A head with no cell in its regime in a batch gets a
     zero. `train_regime_cells_{local,global}` log the cells each loss covered.
   - *Refused:* a hidden gene set (no cell is hidden whole), gene masking without discs, a head given the deep cells
     when the discs are not wider than `depth_um`, no dual decoder or no coordinates; a model without two heads
     refuses it in `train()`. Checkpoint tag `rgm<depth>-<local>-<global>`.
   - *Geometry, measured* (50 x 50 lattice, 20 draws of discs over 30% of the slide): the share of hidden spots
     deeper than 290 µm is 7% for 300 µm discs, **34% for 500 µm**, 50% for 750 µm and **64% for 1000 µm**; at 500 µm the
     depth bins 0-150 / 150-290 / 290-500 / 500+ hold 38 / 28 / 32 / 2% of the hidden spots.
   - *Sweep `regime_loss`* (slurm folder; `joint` is `staged_training`'s): `joint`, `regime` (3 seeds each),
     `regime_remask` (the regime loss and re-masking), and `joint_d1000`, `regime_d1000` (discs of 1000 µm: the wider
     discs you thought of; the threshold stays 290 µm and the evaluation's discs stay 500 µm). **Read:** CKA
     local~global by depth bin (should fall at the rim), each head's score by depth (the local head should hold on the
     near bins and may fall on the deep ones, the global head the reverse; a head that holds on both did not
     specialise), the better head per bin against the baseline (no regression).
   - *Not built:* a gate that combines the two heads by depth (D3), fractional weights per regime.
3. **D1 and D3.1:** a context-only global embedding and an additive linear decoder (**both built 2026-10-22**, below). D1.1, the read-out
   of the output minus the input token, was **not built**: the feed-forward block can re-encode the token and the attention query is the token,
   so it does not remove the cell; D1.2, the context query, was built directly.
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

**Re-masking before the transformer (task 3, built 2026-10-21).** Cells are masked once, in the input
(`_common_step_masking`), and the GCN mixes a hidden cell's visible neighbours into its token, which the
transformer then carries in that cell's own residual stream: the global head of a hidden cell near a disc's rim
can copy the local one. `model.global_component.parameters.remask: hidden` (GraphMAE's re-mask) replaces the
token of every cell hidden whole by one learned token in `GlobalModule.step`, before the input LayerNorm and the
position, so its global embedding is what attention gathers from other cells -- with the long-range mask on,
from beyond the GCN's reach only. Visible cells keep their tokens; the local head is unchanged.
- *Where it applies:* models with a local component (a `GlobalModel` tokenises the masked input itself and is
  refused), under cell masking or discs. Refused with a hidden gene set (no cell is hidden whole: programme
  masking has nothing to re-mask; its analogue is D1's context query for every cell), with random entries only,
  and with pooling (a patch already leaves its hidden cells out). Checkpoint tag `rmk-hidden`.
- *Verified by:* the transformer's gradient with respect to a hidden cell's own token is exactly 0 (and nonzero
  without re-masking); with the long-range mask and one layer, no token within the GCN's two hops reaches a hidden
  cell's global embedding while every token beyond does; no hidden value reaches a prediction; the token trains
  through `train()`; off, the model gains no state and every other weight is initialised as before. 16 tests, 8
  mutations caught, harness IDENTICAL.
- *Sweep `remask_test`* (slurm folder; discs as in `staged_training`, its `joint` as the baseline): `joint` and
  `joint_remask` (long-range mask on, 3 seeds each), `joint_nolrm` and `joint_nolrm_remask` (mask off, one seed);
  twins of `joint` and `joint_remask`. **Read:** CKA local~global at the rim (should fall), the global head by
  depth (with the mask on, a copying head drops at the rim and holds deep; the local head should not move),
  visible lift and the receptive field of hidden spots.
- *Caveat:* all hidden cells now share one key and value too, the sink of K5 (programme masking avoids it;
  T1's visible-keys-only option would remove it).

**Moran's I split (task 1).** Per gene, Moran's I over growing neighbourhoods (k-nearest neighbours at k = 4 ...
256, or radii in µm -- a config list, per slide, averaged over the training slides), on the expression itself
(no model). Its decay is the gene's length scale: ℓ_g, the neighbourhood size at which I falls to half its
smallest-neighbourhood value, and the AUC over log size (the manuscript's Figs. 3C, 4I). Fast decay = local
variability (a likely local interaction), slow or none = global (tissue-scale) variability. Genes are grouped by
ℓ_g (thresholds or quantiles from the config, so dataset-agnostic), then each group is scored by component on the
masked pass: per gene the local and the global head's centred cosine on its hidden entries (`score_heads`'
per-gene table), and the share of its decoder variance on each embedding. Expected: local-scale genes go to the
local head, global-scale genes to the global head.
- *Sanity check first, on synth_spot:* the short-range receivers must come out local, the long-range responses
  and region genes global, the noise genes with I ≈ 0 at every size.
- *Two cautions.* For a smooth field I falls with the neighbourhood anyway; the class is the decay *scale*, not
  its sign. A rising I means short-range anti-correlation (neighbours that exclude each other, e.g. alternating
  senders and receivers), not more global variation, and is reported apart. A cell-type marker is spatially
  variable through composition, so with labels the profile is also computed within each type (as T10 D).
- *Where:* a function in `evaluation` (squidpy's `spatial_autocorr` per graph, or a sparse weights matrix per
  neighbourhood), a `gene_scales` table and a per-group head table in the report; synth_spot first, then legnini
  (pixel coordinates, `PIXEL_TO_UM`).

**Embedding dimensions and genes (task 2).** For each learned embedding (local, global; also the pooled patch
token), the most important dimensions -- `calculate_dim_importance` (standardised decoder loadings, on the
`get_model_output` adata) and, model-agnostic, the variance each dimension explains of the head's prediction on
the masked pass -- then, per top dimension:
- *genes:* the genes most correlated with the dimension's value across cells (Spearman), from the masked pass's
  embeddings and the true expression, so a hidden gene's correlation is what the embedding knows about it;
- *expression space:* that correlation per gene of interest, against the untrained twin and a PCA of the visible
  genes (a dimension that tracks a gene as well untrained only copies the input);
- *coordinate space:* whether the dimension and the gene vary together in space -- Lee's L (bivariate Moran's I)
  between the dimension's map and the gene's map at the gene's length scale from task 1, and the correlation of
  the two kernel-smoothed maps; spatial maps of the dimension next to the gene's (synthetic data: predicted and
  true side by side).
- *Genes of interest* from the config (synth_spot: each programme's sender, receptor, response; legnini: SHH,
  FOXA2, NKX2.2, NKX6.1, OLIG2, PAX6, IRX3, PTCH1, GLI1).
- Read on the DP1 base model and on the remask and decoder arms: a separated model should put the local-scale
  genes on local dimensions and the global-scale ones on global dimensions.

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
| random gene sets (channel masking) | the same random gene subset in every selected cell, drawn again each epoch | **built 2026-10-22**: `dataset.mask_channel_fraction` (below); the label-free form of programme masking |
| structured | gene sets inside discs; a mixture of masks drawn at each step | new: VirTues-inspired, masking along genes and space |
| by cell type | every cell of a sampled type within a region | new; needs labels |
| same-type attention block (AMICI-style) | no attention between cells with the same label | new: an attention mask; needs labels |
| same-cluster block | as above, with unsupervised clusters fitted on the training slides; or the expression-similarity bias started negative | new; label-free |

**Built 2026-10-22 (implemented, unverified): channel masking, `dataset.mask_channel_fraction`.** The label-free form of programme masking.
- *What it is.* A random share of the genes -- `round(fraction * G)`, the same set in every selected cell of a graph, so no neighbour shows a hidden
  gene (`tl.masking.sample_channel_columns`, `sample_channel_mask`) -- is hidden, and drawn again whenever the graph's mask is redrawn: every epoch for
  each training graph (`NodeMaskResampleCallback`), once at setup for validation and test (so the validation loss is of a fixed set). It is programme
  masking with a fresh gene set: cells are selected as for `mask_genes` (`mask_percentage`, 1.0 = every cell; discs select cells too), the loss is on
  the hidden entries, the fill is -1. The fraction is the one number (0.3 first; 0.5 is another arm, nothing else changes); 0 (default) is off and draws
  nothing. Exact count, never none and never all, every gene equally likely.
- *What it asks.* Not together with `mask_genes`; gene masking only; re-masking, the regime loss and visible-keys-only attention refuse it (no cell is
  hidden whole: `tl.masking.hides_gene_set`); a `Precomputed` embedding is refused (every gene is hidden at some step); a `mask_percentage` below 1 warns
  (the other cells show every gene). Checkpoint tag `ch<fraction>c<cell rate>`.
- *Tested:* 31 tests: the draw (exact count, never none or all, every gene equally likely over 3,000 draws, reproducible), one set per graph shared by
  its selected cells, the cell rate and the discs, the datamodule redrawing the training sets (and only those) at every epoch, the config, the name, the
  refusals, no hidden gene reaching a prediction through the real model, and a real `train()` run whose training masks change from epoch to epoch.
  36 mutations caught (one only after a stronger test).
- *Sweep `channel_masking`* (slurm folder, 8 runs + 1 twin; programme-evaluated): `lin_fourier_ch30` and `lin_fourier_ch50` (3 seeds each), `joint_fourier_ch30`
  and `joint_fourier_ch50` (1 seed). The sweep machinery got `EVAL_MASK_GENES`: the gene set the gene-set evaluation hides, because a channel arm has no
  set of its own; here the programme (`role=receiver`, `role=response`), so the numbers sit beside the programme-trained `lin_fourier` and `joint_fourier`.
  **Read:** the programme scores of a channel arm against the programme-trained arm of the same model (long-range and ring programmes first, then the
  mirror and sparse ones), 30% against 50% paired by seed, the price of not naming the programme, and the senders' rank against the untrained twin.
  *Caveat:* a random set rarely hides a whole programme together, so a hidden response can often be predicted from its visible co-responders in the same
  cell, which needs no communication; the larger fraction hides more of them at once, and `ref_own` is the bar for it.

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

**Built 2026-10-20 (the part programme masking needed).** The squared-gradient sender rank was a
sensitivity, not a use: a mean-aggregating GCN weighs its neighbours alike, so it ranked its own senders
0.41 while predicting their responses at 0.59. `receptive_field.response_influence` now defaults to
integrated gradients (`measure="attribution"`): the target is one receiver's summed predicted response
genes, the baseline is the training-mean cell (a model without a GCN moves from its graph's mean token),
a hidden entry does not move, and `steps` Gauss-Legendre points integrate the path (5 in the sweeps; a
ReLU network lands 1-3% off completeness, reported as `response_check`). A cell's weight is the size of its
net contribution; the planted sender is ranked against cells at the same distance as before. On toy heads a
uniform aggregator ranks its sender 0.5 by gradient and 1.0 by attribution; a cell the head is very
sensitive to but that sits at the baseline ranks 1.0 by gradient and 0.0 by attribution. See `tests/test_attribution.py`.

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

**Is attention computed correctly (task 4, built 2026-10-21).** The default attention is a vendored copy of
torch's (`transformer_utils.multi_head_attention_forward_with_gradients`), edited so the map stays in the autograd
graph and comes back per head; the relevance (`tl/self_attn_relevance.py`) hooks those maps for their gradients.
- *Checked by tests:* the vendored attention equals torch's stock `nn.MultiheadAttention` with the same weights
  (outputs, per-head maps, input gradients; boolean and float masks, key padding, both together, train and eval;
  separate query/key and value tokens as under `center_qk`), and the custom layer torch's stock encoder layer; the
  hooked gradient is autograd's gradient with respect to the map the layer used. So the attention itself is
  computed correctly.
- *Found and fixed:* the relevance averaged the heads over all `B × heads` maps of a batch together, mixing graphs
  (latent: `evaluate()` runs one graph); it built its identity and masks on the CPU (it failed for a model on a GPU);
  a target can now be given (`generate_relevance(output, target=...)`), the summed output stays the default.
- *Found, left as they are (documented):* `average_attn_weights` is ignored (maps always per head, which every
  reader expects); in training the returned map is the one after dropout (the attention-entropy loss reads it);
  `evaluate()`'s unused `raw_attn[0]` is head 0, not a head mean; the input LayerNorm is applied twice (once
  before the node encoding is added, once on token + encoding).
- *Built:* `evaluation.attention_check`. `check_attention` recomputes every layer's attention of a masked pass from
  the inputs the layer received, by the definition (map and output), by torch's stock attention (without RoPE)
  and by torch's stock layer (without RoPE or centring). `attention_readouts` compares seven ways of reading it
  for sampled queries -- head-mean attention of the last layer and of all layers, rollout with the residual path,
  Chefer's rule 5 on the last layer and full relevance from the query's own target (½‖ŷ − μ‖² over its scored
  entries), the same relevance from the summed output, and gradient × input at the tokens -- by the planted
  sender's matched rank, their rank agreement and span. Masked passes may now hook their maps
  (`hooking_attention`). 40 tests; of 15 mutations 14 are caught (two only after two more tests), the survivor removes a redundant gradient clear; harness IDENTICAL.
- *Sweep `attention_check`* (slurm folder): evaluation only, on the 7 per-cell transformer arms of
  `response_masking` (all seeds), twins of `lin_fourier`, `lin_rope` and `joint_fourier`; no standard tables
  (`STANDARD_EVALUATION = False`, a new `sweep_common` setting next to the `EXTRA_EVALUATION` hook). **Read:** the
  check should be at rounding (~1e-6) everywhere; then which readout ranks planted senders above its untrained
  twin. That decides which attribution task 5's flow is built on.

**Flow without annotation (task 5).** Today's flow (`attention_flow.patch_flow`) gives patch-to-patch fields and
label heatmaps; the heatmaps need an annotation and the fields mix every programme. Proposed:
- *Per-cell sender vectors.* For a receiver `i` and a target (one programme's predicted response under programme
  masking, or the whole hidden prediction), the attribution `a_ij` of every cell `j` from the readout task 4
  validates (integrated gradients as `response_influence` does, or whichever ranks planted senders best), minus
  its expectation under the expression-permuted slide (the same geometry, no content), and with the distance
  kernel taken out (each `a_ij` over the mean of cells as far away). The vector `v_i = Σ_j a⁺_ij (x_j − x_i) / Σ_j
  a⁺_ij` points from the receiver to where its prediction draws from; its length is the reach.
- *Fields, no labels.* `v_i` smoothed on a grid gives the field (arrows), per programme. The divergence of the
  sender→receiver graph (a cell's outgoing minus incoming attribution) marks sources and sinks. Senders that feed
  many receivers are called *source hubs*. All of it reads positions and attributions only.
- *Summaries without annotation:* clusters of the cells by their vectors and source profiles (communication
  niches), the sources' expression against the receivers' (which visible genes the sources share: a ligand
  candidate list for T8), and with labels only optionally (T5).
- *Checks:* on synth_spot the field must point from receivers to their planted sources (the angle between `v_i`
  and the true direction to the sender, against the permuted slide and the untrained twin); on legnini the SHH
  readouts' field should point to the floor plate (K9).
- The design goes into attribution_plan.md with T4; built after task 4's read decides the attribution.

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
| Tue | T0: `Precomputed` under programme masking; T1: oracle bar (**built 2026-10-21**), distance prior; T4: gradient × input and integrated gradients for the sender ranks (moved up from W2); **T1b: design note and leakage tests** | `joint_fourier` added to response masking, 3 seeds of the main arms | you: the legnini gene set, the CosMx choices, T1b's patch sizes |
| Wed | **T1b: grid pooling and un-pooling in `GlobalModel` and `CombinedModel`**; T2: D0; **2026-10-21: re-masking (task 3) and the attention check (task 4) built** | `pre_pca_rope` added to response masking | – (response masking first read done 2026-10-19) |
| Thu | **T1b: decoder arms for the global-only model (own-cell skip, nonlinear, LSE)**; T2: D2.2; **the Moran's I split (task 1)** | pooled smoke; legnini programme masking (cell-level and pooled); **`remask_test` (8 runs), `attention_check` (evaluation only), `regime_loss` (9 runs) and `global_decoders` (10 runs)**; **built 2026-10-21: D2.2 and the decoder arms** | legnini re-evaluation |
| Fri | T0: the CosMx pipeline; masking_plan.md, attribution_plan.md; **T10: comparison harness and baseline environments**; **embedding dimensions vs genes (task 2)** | **T1b sweep (grid sizes × global-only and joint, 3 seeds)**; decoder arms; T10: Visium and IMC data and baselines on the cluster (yours) | **`remask_test`** |

### W2 (Mon 26 Oct – Sun 1 Nov): masks, whole slides, conditioning, attribution

| days | build | run | read, decide |
|---|---|---|---|
| Mon | T3: the new masks; T2: D1.1 + D3.1, joint LSE (moved from W1); T1b: `niche` pooling on a frozen local stage; **flow without annotation: design (task 5)** | – | T1b sweep, legnini programme masking → **DP1**; **`attention_check` → the attribution for task 5** |
| Tue | T4: integrated gradients, relevance target (built 2026-10-21), null key and registers; **flow without annotation (task 5)** | T3 matrix (synth_spot); **T10 A: classification, CosMx cell type and Visium condition (base model, GCN, PCATransformer, baselines)**; Moran's I split and dimension readouts on the DP1 model | – |
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

1. **legnini programme set.** *Decided 2026-10-20:* the whole identity code as the main task, the SHH
   readouts as a positive control ([legnini_plan.md](legnini_plan.md)). Open within it: FOXA2 (hidden, since
   the paper finds it spreading non-cell-autonomously around SHH+ cells) and CHAT (no source in the paper
   or the manuscript) are my choices; both are one line in the script's gene table.
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
7. **D1.** ~~The read-out that is exact for one layer first (proposed), or the context query directly.~~ Decided 2026-10-22 (by me, reasons in
   decoder_plan.md): the context query directly; the read-out is not exact at one layer.
8. **T1b.** Patch sizes (synth_spot: 500 µm, decided 2026-10-19; legnini 50/100 µm proposed -- 10 and 38
   cells per patch at the median; CosMx 100/200 µm proposed),
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
