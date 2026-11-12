# Thesis outline: the global (transformer) component of InterScale

First version 2026-11-11 (roadmap task 9). 60 pages without front matter and references. Each section lists what it
says, the plan file or `notes.md` entries it draws on, the figures that exist or have to be made, and its state.

States: **have** (run and read), **partial** (read, but a run or a seed is missing), **slot** (planned for 11-24 Nov,
the section can be written without it and gains a paragraph or a panel if it lands). A slot that does not land becomes
a sentence in 5.4.

Working title: *When does a transformer learn cell-cell communication from spatial transcriptomics? The global
component of InterScale.*

The thesis story in one paragraph (the spine every chapter serves): the original global component converged to one
function at the gene-mean floor with uniform attention, and its interpretation rested on a degenerate relevance target.
What changed that was the task, not the architecture: only a task that hides a response the cell's own genes and its
neighbours cannot supply (programme masking) made the transformer learn senders. On top of that task, an isotropic
distance bias with a length scale per head, a raw own-cell skip and enough training steps give a global-only model that
reads short-, mid- and long-range planted programmes on synthetic data, and a decomposable dual model separates local
and global parts at no cost. Integrated gradients and head-mean attention then find the short- and mid-range senders;
nothing finds dense, sparse or directional ones. On CosMx pancreas, whole sections run, but the model sits at the k-NN
smoother within section x cell type, the per-cell flow is reproducible only as a seed ensemble, and the strongest
"flows" are segmentation spillover. The noise ceiling decides whether that is the data or the model.

## Page budget

| chapter | pages |
|---|---|
| 1 Introduction | 15 |
| 2 Methods | 17 |
| 3 Data and experimental setup | 5 |
| 4 Results | 18 |
| 5 Discussion | 5 |
| **total** | **60** |

Against the brainstormed layout (69 pages): Introduction 18 -> 15 (order since 2026-11-11, the user's: spatial omics,
communication biology with long range in focus, CCC methods, then a detour through representation learning and
transformers on graphs, ending with InterScale; representation learning and GNNs cut to the parts the thesis uses), Materials 9 -> 5 (synthetic data kept, baselines and implementation folded into one section, details
to the appendix), Results 16 -> 18 (they carry the thesis; on 2026-11-11 the user moved 2 pages from results to the introduction's representation learning and transformers sections), Methods 18 -> 17.

Appendix (not counted): config keys per arm, the full ablation tables, per-programme tables, the synth_spot gene table,
cluster cost and timing, the tests and harness checks, the list of every negative arm.

---

## 1 Introduction (15)

### 1.1 Spatial omics (1.5)

- From dissociated single-cell RNA-seq to measurements in place: what dissociation loses (neighbours, position) and what
  spatial methods give back.
- Platforms: spot-based sequencing (Visium, Slide-seq; a spot holds several cells), imaging-based single-cell (CosMx,
  Xenium, MERFISH, Molecular Cartography; panels of 100-1000 genes, subcellular resolution), spatial proteomics (IMC,
  CODEX) in one sentence.
- What the data look like: counts per segmented cell, FOVs, sections, donors; the panel limit (most ligand-receptor pairs
  are not measured, which is why LR-free methods matter).
- Technical structure that looks like biology, named here and used in 4.7: segmentation spillover, FOV and section
  offsets, handling programmes (stress, ribosomal).
- Sources: cosmx-coordinates memory, legnini_plan.md, notes.md 2026-11-02 (FOV / type residuals), 2026-11-07 (spillover).

### 1.2 Cell-cell communication, and long-range interactions in particular (2.5)

- Modes: juxtacrine (contact, Notch-Delta), autocrine, paracrine, endocrine; mechanical and ECM-mediated signalling in one
  sentence.
- What sets a signal's range: diffusion against decay (length scale sqrt(D / k)), receptor uptake and ECM binding,
  transport (vesicles, flow), relay chains; ranges from < 10 um to mm.
- Long range in tissue: morphogen gradients read as thresholds and rings (SHH in the neural tube, the design of the legnini
  organoids; Wnt, BMP), zonation, chemokine gradients that guide immune infiltration, islet-acinar crosstalk and insulitis
  in T1D (the CosMx context).
- Why long range is hard to see in one section: a smooth gradient looks like a region or composition effect; a sparse
  source reaches few receivers; direction does not carry between sections; responses mix with cell identity.
- These cases became synth_spot's programme classes (short range, rings, directional, dense, sparse, rare): the link to 3.1.
- Sources: background.md (motivation), PE_plan.md "Benchmark dataset: synth_spot", legnini_plan.md.

### 1.3 Methods for cell-cell communication (2.5)

- Prior-based: CellPhoneDB, CellChat, LIANA+ consensus, squidpy `ligrec`, NicheNet (ligand activity), COMMOT (optimal
  transport).
- Niche and spatial regression: NCEM (within-type residual framing, used in 2.1), MISTy (views by scale), NicheCompass,
  CellCharter and BANKSY (neighbourhood embeddings).
- Attention and deep models: GITIII, Steamboat (multi-scale heads), AMICI, CellAgentChat; what each attends over and what
  it reads out.
- Spatially variable genes and length scales: Moran's I, SpatialDE / nnSVG (the model-free yardstick of 2.7).
- Open problems, one table: no ground truth, one scale per method, prior dependence, composition confound, readouts that
  are not validated against known senders; which of them concern long range most.
- Sources: comparison_plan.md, gene_ranking_literature.md, cellagentchat memory, notes.md 2026-11-10 (Steamboat, GITIII).

### 1.4 Representation learning: a detour (2.5)

- What a representation is for here: an embedding per cell that carries its own state, its neighbourhood or its tissue
  context. Dimensionality reduction (PCA, autoencoders, scVI) in a few sentences; neighbourhood embeddings (CellCharter,
  BANKSY) as the bridge from 1.3.
- Self-supervision by masking: BERT, MAE, GraphMAE; what a masked-prediction task can and cannot force a model to learn
  (the lead-in to 2.2).
- Learning on spatial graphs: message passing, GCN and GIN, receptive field = number of layers; over-smoothing and
  over-squashing (why InterScale fixes the GCN at 2 layers, and why a mean-aggregating GCN cannot select one sender:
  result 4.3).
- Geometric deep learning as the frame: a model's layers should respect the symmetries of its domain (the blueprint of
  Bronstein et al.: grids, groups, graphs, geodesics). For cells in a section: permutations of the cells (invariance and
  equivariance), rigid motions of the section (rotation, translation, reflection: E(2)), and what breaks them (absolute
  coordinates, fixed directions).
- Invariant and equivariant designs on point clouds and graphs: distances and relative positions (SchNet, EGNN) against
  absolute coordinates; why a direction learned on one section does not carry to another (K19, the directional programmes
  of synth_spot), and why every bias that worked in 4.3 is isotropic.

### 1.5 Transformers on graphs (4)

- Self-attention as message passing on the complete graph; multi-head; residual stream and LayerNorm (needed for 2.7's
  relevance argument).
- Graph transformers: GraphTrans (InterScale's template), GraphGPS, Exphormer.
- Transformers against GCNs, side by side (a table): a GCN weighs given edges by fixed, degree-normalised weights
  (isotropic, local); GAT learns the weights on given edges; a transformer learns weights on all pairs and sees structure
  only through encodings and biases. Inductive bias against flexibility; cost O(E) against O(N^2).
- Expressivity and its limits: message passing is bounded by the 1-WL test (GIN reaches it); structural and positional
  encodings and global attention go beyond it. Global attention removes over-squashing (every pair has a direct path) but
  has its own over-smoothing: uniform attention is a mean, which is the gene-mean floor of 4.1.
- Bringing locality back: distance biases and windows as a learned, soft spatial inductive bias (the per-head window of
  2.4); attention as a generalised kernel smoother, which is why the smoother bars of 2.8 lie in the model's function class.
- Positional information: absolute (LapPE, RWPE, coordinates), relative (RoPE, ALiBi-style distance biases, Fourier
  features); why a set of cells has no order and what the encodings make of a lattice.
- Efficient attention: kernel (Performer, linear), sparse (Exphormer, top-k, entmax), exact memory-efficient
  (FlashAttention / SDPA), hierarchical (H-Transformer, coarse-to-fine routing).
- Attention as explanation: Jain & Wallace, rollout, Chefer relevance, integrated gradients, attention sinks and registers.
- Sources: attention_plan.md, PE_plan.md, hierarchical_attention_plan.md section 10 (literature H1-H9).

### 1.6 InterScale and its open problems (1.5)

- The model: GCN (local) -> LayerNorm -> sliding windows -> CLS + transformer with the long-range mask M = 1 - A
  (global) -> two linear decoders; masked reconstruction; relevance-based net flow, CLS scores, standardised gene
  loadings (Drummer, Jiménez et al. 2026).
- What its results rest on (roadmap "What the manuscript's results rest on"): the relevance target, gene ranks from an
  unmasked pass, two decoders predicting one target, the sliding-window ceiling.
- Stated here as open questions; shown in 4.1.

### 1.7 Aims and research questions (0.5)

From the project description, narrowed to the global component:

- **RQ1** Does the global component learn cell-cell communication at all, and what makes it?
- **RQ2** Which architecture choices (position, attention structure, decoder, training length) let it read interactions
  at which range?
- **RQ3** Can local and global contributions be separated within one prediction?
- **RQ4** Which readout recovers who influences whom, checked against known senders?
- **RQ5** What does it find on real tissue (SHH organoids, T1D pancreas), and how does that compare with other methods?

Contributions list (one paragraph each, half a page): the evaluation protocol with bars and twins; programme masking;
the windowed global-only model; the decomposable dual model; the readout benchmark; whole-section attention; the
CosMx negative result with its controls.

---

## 2 Methods (17)

### 2.1 Problem formulation (1.5)

- Cells, genes, coordinates, sections; a mask over cells or entries; the model predicts hidden values from visible ones.
- "Learns communication" defined operationally: the gain of a prediction from context over (a) the cell's own visible
  genes and (b) a local smoother, on hidden values, read per programme or per gene.
- Targets: raw, centred (gene mean out), section offset, type residual (`section_offset: label_mean` on section x cell
  type; NCEM's framing). Why the target decides what is left for context to explain.
- Sources: global_plan.md 0a-0g, notes.md 2026-10-14 (round 3a), 2026-11-04 (type-residual plan), long-range plan v2.

### 2.2 Tasks: what the model is asked to fill in (2.5)

The chapter's central idea: each task is solved by something, and only some tasks need a sender.

- Cell (node) masking, entry (gene) masking, discs (spatial blocks wider than the GCN's reach; the blind area).
- Programme masking: one response set hidden in every cell, so neither the own cell nor a visible co-responder can
  supply it.
- Channel masking (random gene sets per step), Moran's I gene sets (a local and a global set without labels),
  randomised vs pre-selected sets, cell-type residual target on whole cells.
- Leakage rules: re-masking before the transformer (`remask: hidden`), visible keys only, precomputed embeddings fitted
  without the hidden genes, PCA fitted on training slides.
- Sources: masking_plan.md M0-M3, global_plan.md (programme masking P), roadmap T3, K14, K17; notes.md 2026-10-19/20,
  2026-10-22 (remask), 2026-10-23 (channel), 2026-10-29/30 (Moran sets, M1).
- Figure (to make): one schematic, five tasks side by side, what each hides and what can fill it.

### 2.3 The global model (3)

- Two forms: the dual `CombinedModel` (GCN + transformer) and the global-only `GlobalModel` (no GCN).
- Tokens: learned linear, PCA, scVI, `Precomputed` (CellCharter), pooled grid tokens (`pooling.kind: grid`).
- The raw (full-rank) own-cell skip: why a 64-wide token is a rank limit for the cell's own genes.
- The context query (`context_only: query`): a cell's global embedding that does not contain the cell.
- Decoders: dual linear, residual (retired), LSE, nonlinear, additive (`y = W_l h_l + W_g h_g + b`, loss on the sum).
- CLS token, the long-range mask and its invariants (CLS, boolean, left padding), centred queries and keys.
- Size: width, heads, layers, epochs as knobs (32-64 for small panels; 2 layers; 4-8 heads).
- Sources: decoder_plan.md D0-D4, global_plan.md, long-range-mask memory, attention_plan.md stage 0.
- Figure (to make): architecture diagram with the two forms and the decoder variants marked.

### 2.4 Position and attention biases (2.5)

- Absolute PEs (LapPE, RWPE, coordinates) and what the 100 um lattice does to them (tied eigenvalues, k = 8).
- RoPE; isotropic and anisotropic Fourier biases; spectral bias; expression bias.
- The distance prior with a length scale per head (`pe.distance.kind: scales`): a learned window per head; per-head
  roles (a bias for a set of heads).
- Sources: PE_plan.md (encodings, "What the lattice does"), attention_plan.md stages 3, 9, 10; notes.md 2026-10-10.
- Figure (exists, in PE_plan): PE similarity vs distance on the lattice. To make: window per head as a curve.

### 2.5 Attention structure and whole sections (2.5)

- Dense; kernel (Performer, linear); Exphormer; sparse normalisers (entmax, top-k).
- Exact fused attention (SDPA in query blocks, no S x S tensor; equal to dense to 1e-5 in outputs, maps, gradients).
- Hierarchical coarse-to-fine attention: kd-tree clusters per section, cluster tokens (mean / attention pooling), prior
  modes (none / soft / factorised), routed candidate sets; lambda-0 equals dense.
- Training on crops, inference on whole sections (no sliding windows); cost table (memory, s / epoch, CosMx ~1 h).
- Sources: attention_plan.md stages 2, 5, 6, 9; hierarchical_attention_plan.md; notes.md 2026-10-25 (fused),
  2026-10-26/27 (hier), 2026-10-30 (cost), 2026-10-31 Tier 3 (crops).
- Table (to make from plans): mechanism x complexity x exact? x readout available.

### 2.6 Losses and auxiliary terms (1)

- Scaled cosine error (centred and raw), Gaussian NLL; why the centred SCE ignores magnitude (K12).
- Regime loss (each head on the hidden cells of its regime, near / deep).
- Auxiliary terms in one registry: orthogonality (CKA, CCA, gene correlation, loadings), VICReg and context / ring NCE
  (built, gated by the yardstick; short paragraph), sparse-key penalty.
- Rare-event weighting: [slot, only if run].
- Sources: contrastive_plan.md, decoder_plan.md D2.2, D4; notes.md 2026-10-22, 2026-10-23 (orthogonality),
  2026-11-09 (sparse keys).

### 2.7 Interpretation readouts (2.5)

- Why the original relevance fails: the gradient of a summed LayerNorm output is an arbitrary direction (two final-norm
  scales on one model: maps correlate 0.14); the attention implementation checked against torch.
- Raw attention per head and head mean; rollout; integrated gradients from the training-mean cell with a completeness
  check; in-silico deletion (conditional draw, permutation; regions all / context / far).
- Net flow: cell level `A_ij - A_ji`, from hidden queries (`fold_flow`: every cell hidden once), aggregated by cell type
  and condition; flow maps without labels (sources, sinks, in- and outflow).
- Stability: seeds, mask draws, split draws; ensembles and Spearman-Brown.
- Gene side: decoder dimension readout (top genes per dimension, dimension importance, spatial variability), gene
  priority, programmes; Moran's I across neighbourhood size as a model-free length scale; attribution shares of a
  decomposable prediction.
- Sources: linear_probing_docs.md, interaction_readout_plan.md, ranking_plan.md, yardsticks_plan.md; GCN-gradient-probe
  memory; notes.md 2026-10-21 (attention check), 2026-10-31 (flow maps), 2026-11-09 (ensembles).

### 2.8 Evaluation protocol (1.5)

- Metrics: centred cosine, within section and within section x cell type; SCE; why not R².
- Bars: gene mean, `ref_own` (ridge on own visible genes), `ref_own+context` (plus kernel means), k-NN and kernel
  smoothers (8 / 32 / 128 neighbours), the oracle ridge on the true drivers, the untrained twin.
- Per-programme wins / ties / losses against a bar; seeds and the decision rule (3 seeds for anything a claim rests on;
  differences only beyond the seed spread).
- Splits: synth_spot 12 / 4 / 4 slides per condition; CosMx donor folds (one slide = two donors trained) vs FOV-shuffled
  splits (every donor in every split; not about new donors).
- Sources: roadmap T0, D0 yardsticks; notes.md 2026-10-21 (oracle); cosmx memories (folds, section offset).

---

## 3 Data and experimental setup (5)

### 3.1 synth_spot (2)

- 2 conditions x 20 slides, 50 x 50 lattice at 100 um, crops and holes; programme classes with planted senders:
  short-range receivers (R1-R4), mid range (M11), long-range rings (M7 / M8), directional (M9 stripe, M10 cone), dense,
  sparse responses with receptor markers (MC / M12-M21), frequent vs rare twins (MC23 / MC24 vs MC20 / MC21), noise
  genes, region markers.
- Ground truth kept in `uns['synthetic']`; the oracle bar 0.983; Moran's I recovers the planted scale at rank 0.59.
- synth_data_0 in one sentence (legacy, the probe battery of 2026-09-18).
- Sources: PE_plan.md "Benchmark dataset: synth_spot", synth_spot memory, notes.md 2026-10-21 (oracle, disc geometry).
- Figure (exists, beside the data file): per-gene maps; pick one per class.

### 3.2 Real tissue (2)

- legnini SHH organoids (Molecular Cartography): optogenetic SHH source, pixel size 0.138 um, radius-200 graph, two test
  organoids; the identity code (15 genes) and SHH readouts as the hidden set; the positive control.
- CosMx pancreas, ND vs T1D: 386,727 cells, 979 genes (NegPrb dropped), 3 slides x (one ND + one T1D donor), global
  coordinates (`CenterX/Y_global_px` x 0.18 um; `obsm['spatial']` is FOV-local), cell types (`cell_type_coarse`), the
  spillover and handling gene groups.
- vannan lung: [slot, only if a run lands; else one sentence in 5.4].
- Sources: legnini_plan.md, legnini memory, cosmx-coordinates memory, notes.md 2026-11-02, 2026-11-07.

### 3.3 Baselines and implementation (1)

- Steamboat (tutorial settings), GITIII (20 epochs), CellAgentChat [slot], LIANA+ / `ligrec` (prepared, not run);
  the reference smoothers and ridges as model-free baselines.
- Training: epochs (100 / 300), crops, seeds, GPUs, cost per epoch; tests and the harness (one sentence; details in the
  appendix).
- Sources: comparison_plan.md, notes.md 2026-11-10, cluster memories.

---

## 4 Results (18)

### 4.1 Why the original global component learned nothing (2)

- Rounds 1-2 (34 runs on synth_spot, the same on legnini): every attention backend and PE converged to one function at
  the gene-mean floor; attention uniform, biases unlearned.
- Round 3a: centred targets and more steps broke the floor, the first change that gave architecture something to work
  with.
- Discs: the global head never beats a smoother (0.459 vs 0.465 on synth_spot; legnini at the slide mean); near
  visible cells it copies the local head (CKA 0.93 at the rim).
- The interpretation layer: attention is computed correctly, but the relevance target is degenerate; the gene ranks of
  an unmasked pass measure pass-through; the probe battery (synth_data_0) could not show the global component is needed.
- Sources: notes.md 2026-09-18, 2026-10-12 (rounds 1-2), 2026-10-13, 2026-10-14 (3a), 2026-10-18, 2026-10-21; memories
  round1/round2/uniform-attention.
- Figures (exist in results / to collect): round-2 scores vs gene-mean floor; attention entropy; CKA by depth;
  relevance maps under two LayerNorm scales (to make, small).
- Shortened (2026-11-11): the round-by-round numbers and the synth_data_0 probe battery go to the appendix; one figure.

### 4.2 A task that needs senders: programme masking (2)

- Bars: oracle 0.983, `ref_own+context` 0.682-0.685, `ref_own` 0.405.
- Without position the transformer equals `ref_own`; with the isotropic Fourier bias the global head reaches 0.820
  (3 seeds), the gain being the rings; per programme 3 wins / 12 ties / 9 losses.
- Label-free tasks fail for a reason each: discs (smoother), cell masking (the GCN reads every smooth gene, K14),
  channel masking (0.245, does not transfer); randomised vs pre-selected and Moran's I sets (M1).
- legnini under programme masking: every trained arm at its bar for the first time (0.77-0.80 vs 0.75).
- Sources: notes.md 2026-10-19/20, 2026-10-21 (oracle), 2026-10-23 (channel), 2026-10-24 (legnini), 2026-10-30 (M1).
- Figure (to make): per-programme bars, model vs oracle vs `ref_own+context`.
- Shortened (2026-11-11): channel masking and the randomised-set arms in two sentences, their tables to the appendix.

### 4.3 What the architecture adds on top (3)

One master ablation table, then the five findings:

- Position: Fourier beats RoPE in every arm; no position = own-cell ridge.
- Distance window per head + raw own-cell skip: 0.909 ± 0.003 over 3 seeds, programme mean 0.729, short-range receivers
  0.95-0.97 (GCN 0.56-0.68: a narrow head selects a sender, a mean-aggregating GCN cannot); windows barely move from
  their initial scales.
- Training length: 300 epochs move more than any arm (M11 0.29 -> 0.96, programme mean 0.730 -> 0.800, no loss against
  the bar); capacity sweep (heads 8, layers 4, epochs 900 help).
- Attention structure: backends do not change reconstruction; entmax / top-k no gain; hierarchical `hier_none` within
  0.01 of dense, every prior costs 0.05-0.18; pooled tokens with the skip match per-cell at 1/25 of the tokens.
- Inputs: PCA and scVI tokens drop drivers (0.61 / 0.57); `Precomputed` CellCharter on CosMx [slot].
- What fails: directional programmes 0.22-0.25 (oracle 1.00 / 0.83; every working bias is isotropic; anisotropic bias
  [slot]); rare interactions (MC20 / MC21 stay at 0.67 / 0.58 while their frequent twins reach 0.94 / 0.90).
- Receiver gate / low-rank band map (long-range plan v2 step 2): [slot].
- Sources: notes.md 2026-10-12 to 10-25, 2026-10-26/27, 2026-11-05 (capacity); attention_plan.md, PE_plan.md,
  hierarchical_attention_plan.md.
- Shortened (2026-11-11): the capacity sweep, pooled tokens and the token inputs in the master table only; the text
  covers position, the window, training length and the failures.

### 4.4 Separating local and global within one prediction (2)

- Additive decoder: 0.820 -> 0.843; without the context query the parts repeat and cancel.
- Context query + additive decoder: 0.842 at no cost, the context part carries the rings and nothing else.
- Re-masking: a rim cell's global head draws 94 % of its influence from beyond 300 um (was 7 %), at a cost of 0.02-0.05.
- Regime loss: heads specialise as assigned, neither beats the smoother in its regime.
- Moran's I split: with the distance prior the global share follows the gene's scale (Spearman 0.30-0.52 over seeds);
  under cell masking the local part takes nearly every gene (K14).
- Sources: decoder_plan.md, notes.md 2026-10-22 to 10-24, 2026-11-02 (V1-V3).
- Figures (exist in `results/local_global_split`): scale vs global share per arm.
- Shortened (2026-11-11): the regime loss and re-masking in one paragraph, their tables to the appendix.

### 4.5 Which readout finds the senders (2.5)

- Benchmark against planted pairs, every readout against its untrained twin: raw attention, rollout and relevance at
  chance (0.47-0.55); integrated gradients rank short-range and M11 senders at 0.96-0.98 in the dual models; head-mean
  attention of the windowed model finds short and mid-range senders; nothing finds dense, sparse or directional ones.
- Gene rankings: conditional deletion hides smooth senders, permutation flags collinear genes; seed stability;
  remove and retrain.
- Sources: notes.md 2026-10-21 (attention check), 2026-10-23, 2026-10-31 (readouts compared), 2026-11-03/04 (ranking);
  influence-readouts memory.
- Figure (to make): readout x programme class heatmap of sender rank, twin as the reference row.

### 4.6 Real tissue I: SHH organoids (1.5)

- Programme masking reaches the bar; no arm separates from another; the flow maps avoid the floor plate and no rule
  transfers between the two test organoids.
- Constrained variant (senders pass ligands only, receivers predict SHH target genes): [slot; the user's next step after
  the noise ceiling].
- Sources: legnini_plan.md, notes.md 2026-10-13, 2026-10-18, 2026-10-24.

### 4.7 Real tissue II: CosMx pancreas (3.5)

- Feasibility: whole sections with fused / hierarchical attention (~1 h, 6.5 GiB; 3-8 s / epoch).
- What the score is made of: 80-89 % of per-cell centred cosine is the section's own offset; read within section and
  within section x type.
- The model against the smoothers: within section x type the type-residual model holds at the bar (0.087 ± 0.002 vs
  knn128 0.083, twin ~0); within section below knn128.
- Noise ceiling by count splitting and band-pass fields (run 2026-11-11 / 12; ceiling_plan.md): the within-type residual is
  ~19 % reproducible at full depth (per-cell ceiling ~0.44 vs models ~0.087); neighbours explain 4-7 % of the reproducible part
  (median gene 1-3 %), all within ~100 um; the far-band gain (+0.0025) is a FOV offset for the stress genes (HSPA1A, FOS) and
  islet hormones otherwise (INS, GCG; by-type follow-up `far_by_type.py` [slot]); the reproducible rest is cell-intrinsic (SST,
  IGF2, GCG, MALAT1). Read: the models tie the smoother because there is nothing beyond ~100 um to learn linearly, not because
  they underfit. (The first draft's rule "0.1 vs 0.3" compared the smoothers with a split-half reliability; corrected 2026-11-11.)
  Figure to make: within-type scores of models, smoothers and band fits against the ceiling, plus per-gene share vs reliability.
- Flow: single-run per-cell maps are not seed-stable (0.14-0.46); seed ensembles behave as Spearman-Brown predicts
  (~0.9 at 10 seeds), but the 1-layer map is largely geometry and the two depths flip cell-type signs. Type-level roles
  that hold over three models: acinar a source, ductal and fibroblasts sinks, read as spillover rather than signalling.
- Conditions: Immune T1D - ND +0.065 (3 / 3 slides, beyond the twin) is the only difference that repeats over models;
  three donor pairs.
- Downstream programmes: the spatial-state programme (ANXA4, COL18A1, FGFR3) recurs over folds and matches the local
  Moran's I set; nothing repeats as a flow over folds; the known-pair test.
- Sparse-key penalty arm: [slot, ~1 h].
- Sources: notes.md 2026-10-30 to 11-10; cosmx memories; downstream-interpretation memory.
- Figures (exist): flow ensemble maps and curve over ensemble size; cell-type net flow bars; condition per slide. To
  make: scores within type vs smoothers vs ceiling in one panel.

### 4.8 Comparison with other methods (1.5)

- Within-type shuffle (expression shuffled within section x cell type, 0 vs 100 %): Steamboat's pair ranking hardly moves
  (it is composition); what changes is strength (local explained variance halves). GITIII 0-10 % only so far.
- InterScale on the same shuffle: [slot, notebook written, InterScale env must avoid B300 nodes].
- CellAgentChat on global coordinates: [slot].
- Sources: notes.md 2026-11-10 (two entries), comparison_plan.md, cellagentchat memory.

---

## 5 Discussion (5)

### 5.1 Summary: the task, not the architecture (1.5)

- RQ1-RQ5 answered in turn; the ranking of what moved numbers (the "Major improvements" index of notes.md, reversed).

### 5.2 What a global transformer can read, and what attention says (1)

- Isotropic distance is learnable, direction and rarity are not; attention is an explanation only as an ensemble and
  only where IG agrees; what a reconstruction score does not say about what a model reads.

### 5.3 Limitations (1)

- synth_spot is a regular lattice with exact rules (K20); one positive-control tissue with two test organoids; CosMx
  has three donor pairs; training length unsettled on CosMx; segmentation spillover; seeds; the panel.

### 5.4 Future directions and conclusion (1.5)

- Programmes planted into CosMx geometry (semi-synthetic, K20); conditioning on disease (T7); prior knowledge as a
  constraint (ligand -> target) rather than a filter (T8); receiver gate and low-rank band maps; foundation-model and
  niche embeddings as tokens; ensembles by default; whatever slots did not land.

---

## Slots for 11-24 November, in priority order

| # | experiment | state | feeds | cost |
|---|---|---|---|---|
| S1 | noise ceiling + band-pass fields (`slurm/new/fable/fields.sbatch`) | run and read 2026-11-11 / 12 (ceiling_plan.md) | 4.7, 3.1 | CPU job |
| S2 | within-type shuffle, InterScale + baselines notebooks | written | 4.8 | 4 h + 12 h |
| S3 | sparse-key arm of cosmx_typeres + its twin | built | 4.7 | ~1 h |
| S4 | legnini constrained SHH variant | after S1 | 4.6 | to design |
| S5 | CellAgentChat on global coordinates | script written, env pending | 4.8 | job |
| S6 | receiver gate / low-rank band map (plan v2 step 2-3) | dropped on CosMx 2026-11-12 (no far signal); synth_spot / legnini only | 4.3, 4.7 | build + run |
| S7 | `Precomputed` CellCharter / scVI tokens on CosMx | `Precomputed` built | 4.3 | one job |
| S8 | anisotropic Fourier bias on the windowed model | built, never run | 4.3 | one job |
| S9 | INS / GCG far signal by receiver type and distance (`fable/far_by_type.py`, `STEPS="bytype"`) | built 2026-11-12, toy-checked | 4.7 | CPU job |
| S10 | the same audit (ceiling + band fields) on legnini: is there a long-range target for the global component? SHH vs Ctrl organoids as the negative control | built 2026-11-12 (`fields.sbatch` step `legnini`), stub-checked | 4.6, 5.1 | CPU job (run 3) |
| S11 | composition control + spillover index on CosMx: is the near term composition, spillover, or neighbours' expression? | built 2026-11-12 (`fields.sbatch` run 3, steps `cosmx`, `spill`) | 4.7 | CPU job (run 3) |
| S12 | conditional reading by the global model: own genes, type mean and FOV offset given, per-gene share against ridges from the same inputs (`cosmx_conditional.sbatch`, fork branch `conditional-context`) | built 2026-11-12, not run | 4.7, 5.2 | ~5 h GPU, after run 3 |

Writing from the start in parallel: chapters 1-3 and 4.1-4.5 depend on no slot.
