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
| T0 | infrastructure: seeds, per-arm evaluation, inputs, CosMx | proposed | | |
| T1 | does the global model learn communication at all | proposed | | |
| T2 | disentangling local and global ([decoder_plan.md](decoder_plan.md)) | proposed | | |
| T3 | masking strategies | proposed | | |
| T4 | interpretation beyond raw attention; attention sinks | proposed | | |
| T5 | how cell-type driven attention is | proposed | | |
| T6 | complexity vs tissue coverage ([attention_plan.md](attention_plan.md) stages 6, 8) | proposed | | |
| T7 | disease-specific interactions | proposed | | |
| T8 | prior knowledge (NicheNet), downstream | proposed | | |
| T9 | literature review and write-up | proposed | | |

## Where we stand

### The goals against what has been tested

| goal (description) | built | run and read | missing |
|---|---|---|---|
| **Disease-specific interactions** through conditioning | nothing. Condition is only a split stratifier, and in synth_spot a slide-level gene programme (cg_1–3) | – | a conditioned model; data with condition-specific interactions and known truth; a CosMx ND/T1D read |
| **How cell-type driven attention is** | label flow heatmaps against the expression-permuted slide (`attention_flow`) | once, on legnini Leiden clusters under discs. There is structure beyond distance, but no rule transfers between the two test organoids | a same-type metric against a distance-matched null; the CLS-by-type confound (Alpha/Beta); CosMx with annotated types |
| **Masking behaviour** | random cells (`node`), random entries (`gene`), discs (spatial blocks), programme masking (one gene set hidden in every cell) | `node`, `gene` and discs on synth_spot and legnini. Programme masking was never run | an AMICI-style same-type attention block; masking by cell type; random gene sets per step (channel masking); combinations; one train × evaluation matrix |
| **Complexity vs coverage** | Performer, linear, Exphormer, entmax and top-k attention; RoPE, Fourier, distance, spectral and expression biases | synth_spot (2.5k spots) and legnini (4.3k cells), both small enough for dense attention. No backend changes reconstruction | runs where dense attention cannot run (CosMx sections, 47k–100k cells); windows against whole slides; exact memory-efficient attention (SDPA); cross-attention; diagnostics without S × S maps |
| **Disentangling local and global** | dual linear decoder; residual decoder; `linear-lse` decoder (never run); VICReg, context NCE and ring NCE (never evaluated on a task that can show communication) | the residual decoder, which hurts the deep field | decoder_plan.md D0–D4; LSE and an orthogonality term; scores against synth_spot's planted programmes |
| **Prior knowledge (NicheNet)** | nothing | – | priors, pair validation, ligand activity, hypotheses |
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
- **Not retested, and not needed for the goals:** the classification results (Fig. 2) and the
  IMC and Visium datasets.

## Known problems of the global model, and the fix we try

| # | problem (observed) | what we believe causes it | fix we try | track, week |
|---|---|---|---|---|
| K1 | Global never beats the best smoother at disc reconstruction (synth_spot 0.459 vs 0.465). On legnini it sits at the slide mean (0.50 vs 0.594 for the 40 µm kernel) | A plain average of the visible cells solves the disc task. On legnini attention cannot become local: the biases start at 0 and learned ≤ 0.03 | Programme masking on synth_spot and legnini; a distance prior with a length scale per head; targets split by scale (D2.1) | T1, T2; W1 |
| K2 | Near visible cells the global head copies the local one (CKA 0.93 at the rim) | Both heads predict one target, and the cell's own token stays in the transformer's residual stream | A context-only global embedding (D1); an additive decoder (D3.1); each loss on the cells where its component has to work (D2.2) | T2; W1 |
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
  (`cell_type_coarse`), the condition, the three provided splits, sections or windows as samples,
  and a smoke config. What the data looks like:
  - 386,727 cells × 979 genes;
  - six sections: three slides, each with one ND and one T1D section, 47k–100k cells each;
  - `split` uses slide 1 for training, slide 2 for validation and slide 3 for testing. Each split
    therefore holds one donor per condition, so within a fold condition and donor are confounded.
- **Housekeeping (K8).** Evaluate `joint_rope` (global_only) and `l2g_res` (staged_training). The
  round-3b rerun note in attention_plan.md is still uncommitted.

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
- **Decides (DP1, Monday of W2).** The base model for T3–T8 is the arm that, over 3 seeds, scores
  highest above `ref_own+context` on long-range and sparse responses and ranks senders above 0.5.
  If no arm passes, compare the arms with the oracle bar:
  - the oracle passes: the model is the limit. Try more steps, more width, the distance prior,
    top-k;
  - the oracle fails too: the task is the limit.

### T2 — disentangling local and global (W1–W2; K2, K3, K11)

The design is in [decoder_plan.md](decoder_plan.md). Order:

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
5. **Orthogonality:**
   - D4.1, the cross-covariance between `z_l` and `z_g`, on top of the winner;
   - D4.2: run ring NCE and context NCE once, under D0.
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
sliding windows.

- **Build: SDPA, fused exact attention** (attention_plan.md stage 6).
  - It is exact and needs O(S) memory.
  - RoPE and the factor-form biases (Fourier, spectral) fold into q and k, so no S × S tensor is
    formed.
  - The long-range mask stays off at this scale: it changed nothing in round 3b or the first
    sweeps.

  Together with Performer, linear and Exphormer, that gives four ways to cover a whole section.
- **Build: cross-attention to patch tokens** (stretch). Cells attend to a grid of patch tokens with
  positions (the mean of the visible tokens in each patch), at O(S × patches). These are
  attention_plan.md stage 7's inducing points with spatial anchors, and they give a tissue-level
  unit to interpret.
- **Build: diagnostics for large S.** Attention, influence and integrated gradients on a sample of
  queries, without S × S maps. Take the densified k-hop closure out of the path (one of stage 8's
  blockers).
- **Build: stochastic windows.** Windows at random positions each epoch for training, and the
  whole section at inference (the manuscript's "adaptive or stochastic window sampling").
- **Run 1, synth_spot (with ground truth).**
  - Arms: fixed windows of 1.25 and 2.5 mm with dense attention; whole slides with dense, SDPA,
    Performer, linear and Exphormer; all under programme masking, 3 seeds.
  - Read the response scores for receivers whose source lies in another window against those in
    the same window, and the sender rank across window borders.
- **Run 2, CosMx.**
  - Memory, steps/s and the largest S per backend.
  - Whole sections against windows: programme scores by distance to the nearest window border.
  - Maps over a whole section against window maps stitched together.
- **Decides (DP3):** windows or whole sections, and the backend, for the CosMx runs in T7 and T8.

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
- **CosMx.** The DP1 base model, the DP2 mask and the DP3 coverage, with the conditioning modes as
  arms, on 3 folds.
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
| Tue | T0: `Precomputed` under programme masking; T1: oracle bar, distance prior, visible keys | – | you: the legnini gene set, the CosMx choices |
| Wed | T2: D0, D2.2 | `pre_pca_rope` added to response masking | response masking (first read) |
| Thu | T2: D1.1 + D3.1, joint LSE | legnini programme masking | legnini re-evaluation |
| Fri | T2: D4.1; T0: the CosMx pipeline; masking_plan.md, attribution_plan.md | T2 sweep 1 (3 seeds) | – |

### W2 (Mon 26 Oct – Sun 1 Nov): masks, whole slides, conditioning, attribution

| days | build | run | read, decide |
|---|---|---|---|
| Mon | T3: the new masks | – | T2 sweep 1, legnini programme masking → **DP1** |
| Tue | T4: integrated gradients, relevance target, null key and registers | T3 matrix (synth_spot) | – |
| Wed | T6: SDPA, query-sampled diagnostics, windows on synth_spot, stochastic windows | T4 benchmark on W1's checkpoints; T3 subset (legnini) | – |
| Thu | T7: generator, conditioning modes, swap evaluation; conditioning_plan.md | T6 run 1 (synth_spot); CosMx feasibility (run 2, part 1) | T3 matrix → **DP2** |
| Fri | T8: priors, validation, ligand activity; T5 metric; T6 patch cross-attention (stretch) | T7 sweep (synth_spot); T2 sweep 2 | T4 table; T6 run 1 → **DP3** |

### W3 (Mon 2 – Sun 8 Nov): the questions on real tissue, and the write-up

| days | build | run | read, decide |
|---|---|---|---|
| Mon | fixes from W2's reads; T5: CLS classifier with integrated gradients | T8 positive control (legnini, evaluation only) | T7 synth_spot, T2 sweep 2 |
| Tue–Wed | T2: gene programmes from the masked pass | CosMx main sweep, 3 folds: base model, conditioning modes, same-type block, at DP3's coverage | T5 on synth_spot and legnini |
| Thu–Fri | – | – | CosMx: T7 swap, T5, T8 validation and hypotheses |
| Sat–Sun | – | – | T9: write-up, manuscript claims, review draft |

### Decision points

| | when | decides | on what |
|---|---|---|---|
| DP1 | Mon of W2 | the base model | programme score above `ref_own+context` on long-range and sparse responses, sender rank above 0.5; 3 seeds |
| DP2 | Thu of W2 | the default mask for CosMx | communication readouts without loss of reconstruction |
| DP3 | Fri of W2 | windows or whole sections, and the backend | cross-window responses on synth_spot; memory and throughput on CosMx |
| DP4 | Mon of W3 | the conditioning mode to report | the swap against the generator's truth |

## If we fall behind

Keep at least this much for each goal:

| goal | minimum |
|---|---|
| disease | the synth_spot condition test (token, q/k adapters, none) and one CosMx swap read |
| cell type | the metric on existing checkpoints (legnini clusters, CosMx types); the AMICI-style block on synth_spot and CosMx |
| masking | the synth_spot matrix with random cells, discs, programme masking, channel masking and the AMICI-style block |
| coverage | synth_spot windows against whole slides (dense, Performer); the CosMx memory and throughput table |
| disentangling | D2.2, D1.1 + D3.1, LSE and the cross-covariance term, on synth_spot |
| priors | the legnini SHH control and the CosMx validation table |
| interpretation | integrated gradients, influence and raw attention on planted edges |
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
