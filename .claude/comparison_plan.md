# Comparison workflow: interactions, spatially variable genes, ligand–receptor priors and classification (CosMx first)

Proposed 2026-10-25, at the user's request: on CosMx a standalone run says nothing; every training run should feed an
immediate comparison with (1) other cell–cell communication (CCC) methods, (2) spatially variable gene (SVG) detection,
(3) ligand–receptor (LR) interaction databases, and (4) a downstream classification (cell type), with plots and the time
and memory of every step, over several model settings. It turns roadmap T10 (A, B, D), T8 (a) and T5's CosMx part into
one workflow. Datasets are inputs, never names in the code (memory: evaluation stays dataset-agnostic): CosMx is the
first, synth_spot the calibration where the truth is known, legnini the SHH control.

## Status

| stage | state | verified by | date |
|---|---|---|---|
| C0 — shared inputs: the CosMx input file, folds, LR resource, positive controls, the output tables | **implemented, not run on the cluster** (2026-10-25): `make_inputs.py` (coordinates from the global pixels, checked: no FOV overlap, nearest neighbours 7-8.7 µm), `make_resource.py` (LIANA+ consensus: 792 of 4,620 pairs have every subunit on the panel, 265 ligands), `tables.py` (the four schemas, checked writes, a timer); positive controls with the user | run on a 2-FOV-per-section copy (63,543 cells) locally | 2026-10-25 |
| C1 — the settings, trained once per fold (InterScale arms) | proposed | | |
| C2 — InterScale's readouts into the shared tables | proposed | | |
| C3 — baselines into the same tables | **4 of 7 implemented** (2026-10-25): LIANA+ rank aggregate, `ligrec`, Moran's I (also within types), logistic regression (penalty chosen on the validation slide); AMICI, Steamboat, NicheCompass not yet. **2026-10-29: the runner scripts were removed by the user** (below); to be written again | each run on the 2-FOV copy and its tables read back | 2026-10-29 |
| C4 — the comparison: metrics, figures, report | proposed | | |
| C5 — calibration on synth_spot (planted truth) and legnini (SHH) | proposed | | |
| C6 — condition-specific interactions (T7) and NicheNet hypotheses (T8 b) | proposed | | |

**Where the code lives:** `notebooks_and_scripts/slurm/comparison/` (with the other cluster scripts, not under version
control): `tables.py` (the contract; pandas and numpy only, so every environment imports it), `make_inputs.py`,
`make_resource.py`, `run_<method>.py`, `run_baseline.sbatch` (one step in a given conda environment). Library code for
InterScale's readouts (C2) goes into the fork with tests.

**Removed by the user on 2026-10-29** (they did not like the code). Only `make_inputs.py` remains, as a copy in
`notebooks_and_scripts/slurm/hierarchical/scripts/`. `tables.py`, `make_resource.py`, the four runners and
`run_baseline.sbatch` are to be written again from this plan's design (C0 has the schemas). Agree how with the user before
writing, and put them in the fork (`scripts/comparison/`), under version control.

**Second dataset: lung fibrosis (added 2026-10-29, lung_plan.md).** The same workflow on the lung fibrosis data, with niches. Its
data is not ready yet. What it adds here: `niche` as a label column next to the cell types in every table (interactions by niche
pair, SVG within niches, niche classification); its own positive controls (lung_plan.md L4); IPF vs control as the condition
split. The C1 arms there are the hierarchical SOTA (`hier_none`) with the niche variants instead of the dual arms, because the
context query does not run on whole sections (below, "Whole sections").

## Decisions (the user, 2026-10-25)

1. **Mask:** cell masking with discs.
2. **Labels:** coarse types (9) for interactions; coarse and fine (14) for classification.
3. **Graph:** k nearest neighbours for the GCN; its reach is then counted in hops (the manuscript's radius curve becomes a
   hops curve). *Found when checking the units (2026-10-25):* `obsm['spatial']` is **not** µm and not a section's layout --
   it holds each FOV's local pixel coordinates (slope 1 to the global pixels within an FOV), so the 11-15 FOVs of a section
   lie on top of each other in one 5,440 x 3,620 px frame. A graph or a distance on it joins cells of different FOVs. The
   section's layout is `CenterX_global_px` / `CenterY_global_px`: no FOVs overlap, sections span 8-13 mm. The pixel size is
   taken as **0.18 µm** (an FOV of 5,472 x 3,648 px is CosMx's 0.985 x 0.657 mm field; cells then lie 7-8.5 µm from their
   nearest neighbour and are 12 µm wide) -- a setting, to be confirmed from the paper's methods. The manuscript's windows
   were cut within FOVs, so they were unaffected.
4. **Baselines (the user sets up their environments):** LIANA+ and squidpy's `ligrec` for LR interactions; AMICI,
   Steamboat and NicheCompass for learned interactions; Moran's I for SVGs; logistic regression for cell types.
5. **LR resource:** LIANA+'s consensus now; CellChatDB and NicheNet v2 later.
6. **Positive controls:** the user checks the list; not blocking.
7. **Arms:** no PCA-transformer, no pooled global-only arm (the user is designing another pooling).
8. **Supervised cell-type models** as well as probes on frozen embeddings.

## What the comparison has to show, and what it cannot

The claims, from the description and the manuscript:

1. Global context finds interactions the local methods miss, at longer range, and they are supported by LR databases and
   by what is known of the T1D pancreas.
2. The local and global parts carry genes of different spatial scale (SVG scale, the manuscript's Fig. 3C/4I).
3. Global context helps cell-type classification for some types (immune, endothelial, mast in the manuscript) and lowers
   the sensitivity to the graph radius.
4. Whole sections run without windows, at a stated cost.

CosMx has **no ground truth** for interactions. So every interaction result is stated as agreement between methods,
support by LR priors against nulls, and recovery of a short list of positive controls -- never as accuracy. The same
workflow runs on synth_spot, where the planted pairs are the truth (C5): that is what says whether each method's score
means what the CosMx tables read into it. Two confounds are read against nulls throughout: co-location (cells of two types
that sit together score as interacting by distance alone; distance-matched nulls) and composition (a marker of a type is
"spatially variable" because the type is; SVG scores also within a cell type).

## C0 — shared inputs (computed once, cached, read by every method)

- **Input file** (`inputs.h5ad`): `layers['raw']` counts and `layers['log1p_norm']`; `obsm['spatial']` the section's
  coordinates in µm (CosMx: the global pixels times the pixel size; the FOV-local pixels kept as `obsm['spatial_fov_px']`);
  `obs`: section (= donor), slide, condition (ND / T1D), cell type coarse (9) and fine (14), the fold columns. One
  preprocessing for every method; checks that no two FOVs of a section overlap and that nearest neighbours lie 3-30 µm apart.
- **Folds.** CosMx's three slides each hold one ND and one T1D section; `split_0`, `split_1`, `split_2` rotate them (each
  slide tested once, each test fold one donor per condition; roadmap K7: the folds are the replicates, not seeds).
- **LR resource** (`lr_resource.csv`: ligand, receptor -- complexes as `A_B` --, source database, category): LIANA+'s
  `consensus` resource (OmniPath: CellPhoneDB, CellChatDB, ICELLNET, ...) as the primary, restricted to the panel; per pair
  whether ligand and receptor (every subunit) are on the panel. CellChatDB and NicheNet v2 (ligand–receptor network and
  ligand–target matrix, exported once from R) as the alternates and for C6. The panel coverage is the first number reported:
  what no method can see on 979 genes.
- **Positive controls** (`controls.csv`; drafted from the literature, validated by the user against the panel): interactions
  any method should find, each with the expected direction and condition. Candidates: paracrine within islets (GCG → GCGR /
  GLP1R on beta cells, INS → INSR, SST → SSTR2/SSTR5 from delta cells), endothelial–islet (VEGFA → KDR / FLT1), immune
  infiltration in T1D (CD8 T cells near beta cells, `CXCL10` → `CXCR3`, MHC class I up in islets: `HLA-A/B/C`, `B2M`), mast
  cells entering islets in T1D (manuscript, Fig. 4D). Each is a (sender type, ligand, receptor, receiver type, condition).
- **Output tables** (parquet, one schema per task; every method writes the same):
  - `ccc_scores`: method, setting, fold, section, condition, sender_type, receiver_type, ligand, receptor (null for a
    type-level method), score, p_value (nullable), distance_um (the mean sender–receiver distance behind the score, nullable).
  - `svg_scores`: method, setting, fold, section, gene, score, p_value, rank, scale_um (nullable), within_type (null or the type).
  - `label_predictions`: method, setting, embedding, fold, cell, true, predicted, probability.
  - `costs`: method, setting, fold, step (train / evaluate / one readout), n_cells, seconds, peak_gib, device, epochs.

## C1 — the settings (trained once per fold, read many ways)

Every arm has the same mask, folds, epochs (100; memory: results must come fast) and one job per (arm, fold).

| arm | model | why |
|---|---|---|
| `gcn` | GCN alone | the manuscript's local baseline; the neighbourhood-only floor every global claim is read against |
| `dual_ctx` | GCN + transformer, additive decoder, context query, distance prior (`add_ctx_dist`) | the best dual model on synth_spot; its parts split genes by scale |
| `dual_ctx_orth` | `dual_ctx` + an orthogonality term between the local and global embeddings (`orth_cka`, 0.1; dropout 0 on its path, which the term requires) | the description's disentanglement question; first run of D4 on data |
| `global_window` | GlobalModel, learned linear token, raw own-cell skip, distance window (`lin_dist_skipraw`), fused attention on whole sections | the best model on synth_spot; no GCN, no windows |
| `gcn_cls`, `dual_cls`, `global_cls` | the GCN, the dual model and the global-only model trained to predict the cell type (node classification), once per label level | decision 8: the supervised comparison of Fig. 2D, next to the probes on the self-supervised arms |

The self-supervised arms are 4 x 3 folds = 12 jobs; the supervised ones 3 x 2 label levels x 3 folds = 18.
*Proposed settings for CosMx (not yet decided):* k = 8 neighbours (the 8th lies at ~15-19 µm), the GCN at 1, 2 and 3 hops
for the hops curve (2 by default); discs of 60 µm radius over 10% of each section plus 20% of cells at random; windows over
10-1000 µm.

- **The mask:** cell masking with discs wider than the GCN's reach (K14: under plain cell masking the global part carries
  almost nothing, since the neighbours show every smooth gene). Every gene is a target, which the SVG and gene-programme
  readouts need. Programme masking on the interferon / MHC-I set (roadmap T7) is a second task, not a replacement.
- **CosMx's scales:** the window range from the cell spacing (~15 µm) to ~1 mm (the windows hardly move from their start,
  notes 2026-10-25, so the range is a choice); the GCN's radius from the manuscript (60 / 110 / 150 µm) -- the radius curve
  of claim 3 is the GCN and dual arms at the three radii.
- **Whole sections:** fused attention (stage 6); the first training job also gives the speed and memory at 47k-100k cells.
  If a step is too slow, the fallback is the pooled arm, then the manuscript's windows.

## C2 — InterScale's readouts (per arm and fold, into the shared tables)

- **Cell-type classification (downstream).** A logistic-regression probe on the frozen embeddings, trained on the training
  slides, scored on the test slide: local, global (or context), both, and the parts' predictions; per-type F1, macro-F1,
  confusion. The supervised end-to-end classifier of the manuscript (Fig. 2D) is one more arm only if Fig. 2 must be redone.
- **Interactions by cell type.** For each receiver type B, the integrated-gradients attribution of B's predicted expression
  (summed over sampled receivers of B, so one attribution per receiver type and spatial tile, not per cell -- IG is linear in
  its target) to every input cell, summed over the senders of each type A: `score(A -> B)`, against the same attribution on
  the expression-permuted section and against a distance-matched null. Kept with its distance profile (attribution mass by
  distance bin), which is where "longer range than the local methods" is read. The label flow of `patch_flow` (gradient
  influence against the permuted null) as the second, existing readout; raw attention by type only for the manuscript's
  comparison (fused attention gives rows for sampled queries; to build).
- **Interactions by ligand.** The same attribution resolved by input gene: summed over the senders of type A, gene L's
  attribution to B's prediction is `score(A, L -> B)`, comparable to an LR method's ligand-level score; joined with the
  resource where B expresses a receptor of L.
- **SVG scores.** (i) Context predictability: a gene's centred cosine on hidden cells above a slide-mean and a 15-40 µm
  kernel predictor; (ii) the global part's share of the gene (decomposable arms); (iii) Moran's I against the neighbourhood
  size and its length scale (`evaluation.gene_scales`), also within each cell type.
- **Gene programmes.** Local and global genes per condition (T2, Fig. 4G/H), enrichment against the panel as background.
- **Costs.** `optim.log_efficiency` (epoch time, steps/s, peak memory) and every readout's time (the sweep's timing section).

## C3 — baselines (each in its own environment, the same inputs, folds and tables)

`interscale_env` stays untouched (memory: pip-managed). One runner per method, `comparison/runners/run_<method>.{py,R}`,
reading `inputs.h5ad` and writing the tables, timed by the same wrapper (wall clock, peak RSS or GPU memory).

Decided (decision 4); the rest of the first draft (CellChat, COMMOT, MISTy, SPARK-X, Hotspot, nnSVG, SpatialDE, CellCharter,
BANKSY) waits.

| task | methods | notes |
|---|---|---|
| CCC, LR-level | LIANA+ rank aggregate (non-spatial consensus of CellPhoneDB, CellChat, NATMI, ...) and its spatial bivariate scores; `squidpy.gr.ligrec` | per section; the LR resource is LIANA+'s consensus, cut to the panel |
| CCC, learned | AMICI (labels, attention), Steamboat (multi-scale heads), NicheCompass (LR-prior GNN) | the methods closest to InterScale's claim |
| SVG | Moran's I (squidpy) | per section, on the global coordinates |
| classification | logistic regression on expression | the same folds as C2 |

## C4 — the comparison (one script, one report, figures)

The report is `results.md`-style (sections per question, a summary on top), with these readouts and figures:

- **Interactions.**
  - Sender × receiver heatmaps per method, ND, T1D and their difference; methods side by side.
  - Agreement: Spearman of type-pair scores between every two methods (a method × method matrix), top-k overlap.
  - LR support: of each method's top-k (type pair, ligand), the share whose receptor is expressed by the receiver type, against
    label-shuffled and distance-matched nulls (an enrichment with its interval); for InterScale's ligand scores the AUPRC of
    resource ligands against non-ligand genes per receiver type.
  - Range: the distance distribution behind each method's top interactions -- claim 1 is that InterScale's supported pairs
    reach further than the local methods'.
  - Positive controls: each control's rank per method and condition (a dot plot), and maps of one control on an ND and a T1D
    section (senders coloured by their attribution to the receivers; memory: show spatial maps).
  - Stability: each score's spread over the three folds (one donor pair per condition each).
- **SVG.** Rank agreement between methods; overlap of top genes (and what each finds alone, with maps); the global share
  against Moran's length scale (claim 2); within-type SVG; runtime and memory against the number of cells.
- **Classification.** Macro-F1 and per-type F1 per arm and embedding over the folds; the radius curves (claim 3);
  confusion matrices; maps of errors on a test section.
- **Costs.** Training time per epoch and peak memory per arm against cells per section; every method's wall clock per task;
  a cost-against-quality plot per task.

## C5 — calibration with known truth

- **synth_spot:** the planted pairs as a custom LR resource (S1 → R1, M7 → MC7, ...: each source gene a "ligand", each
  response a "receptor"), so LIANA+, COMMOT, ligrec and MISTy run as on CosMx; every method's sender–receiver scores against
  the planted edges (matched rank and AUROC against distance-matched cells, as T4); SVG methods against `is_spatial` and the
  planted scale (AUPRC). This says, before CosMx is read, which readout recovers what.
- **legnini:** the SHH control of T8: SHH → PTCH1 / GLI1 / HHIP from the floor plate.

## What exists and what has to be built

- **Exists:** the sweep infrastructure (arms, folds as seeds, job arrays, timing, per-run sections, reports), efficiency
  logging, label probes on embeddings, the label flow (`patch_flow`), gene scales, attribution shares of decomposable
  models, the orthogonality terms, pooling, fused attention, the oracle and reference machinery.
- **To build:**
  - CosMx in the pipeline (roadmap T0): the `PREPROCESS` hook, labels, condition, folds, whole sections as samples.
  - IG attribution by receiver type with sampled receivers and summed targets, resolved by sender type and by input gene,
    with the distance-matched and permuted nulls, at section scale.
  - Attention rows for sampled queries under fused attention; an inference path without dense maps (T6).
  - The table writers in the sweep's evaluation, the runners per baseline, the comparison script and its figures.

## Whole sections, no sliding windows (asked 2026-10-25; to settle in the next chat)

The user wants the next CosMx run on whole slides, not sliding windows (roadmap T6 had decided it already).

- **The sample is a section, not a physical slide.** Each slide holds two separate pieces of tissue, one ND and one T1D donor,
  5.6-7.9 mm apart (slide 1: 108,711 cells, 2: 126,421, 3: 151,595). One sequence per slide would put two donors in one
  attention sequence (CLS, centring and the widest heads would mix them) and break every per-donor readout. So the unit is
  `section` (`slide_condition`): 47k-100k cells. Assumed, **to confirm with the user**.
- **What runs at that size today:** the global-only windowed model (fused attention; speed unmeasured) and the GCN models.
  **Not:** the dual arms with the context query (`dual_ctx`, `dual_ctx_orth`): `context_only: query` needs queries different
  from keys, and the transformer refuses every backend but torch's dense attention for it (`transformer_encoder.py`, ~l.153).
  **Not yet:** the evaluation -- `masked_probing.reference_predictions` forms a dense float64 `cdist` per section (~80 GB at
  100k cells), and relevance / attention diagnostics form full maps.
- **Open choice for the dual arms:** (a) extend the fused attention to the context query (same blocks; queries from the query
  stream, keys and values from the cells' tokens; recommended: without it the global part mostly copies the local one), or
  (b) run the dual arms without the context query, under the long-range mask the fused attention already supports (quicker;
  weaker local-vs-global readouts).
- **To build either way (C2):** the smoother references on neighbour lists, IG attributions on sampled receivers, an inference
  path without dense maps (`get_model_output`), readouts on sampled queries.
- **Far fewer optimiser steps:** one step = one section, 2 training sections per fold -> 2 steps per epoch (100 epochs = 200
  steps; synth_spot runs had 2,400). More epochs will be needed, and seconds per step at 100k cells are unknown: run
  `scripts/fused_attention_benchmark.py --cells 2500 10000 30000 50000 100000 --offset-um 50000` on a cluster GPU first
  (also reports fused-vs-dense agreement and peak memory).

## Decisions still open

1. **Pixel size:** 0.18 µm (proposed from the FOV size) -- to confirm from the paper.
2. **CosMx settings:** k = 8, the hops curve at 1-3, the disc and random-cell rates, the window range (proposed under C1).
3. **Positive controls:** the user's check of the draft list.
4. **Whole sections:** the unit is the section (above); the dual arms' route, (a) or (b); epochs after the benchmark.

## Order

1. C0 and the tier-1 baselines first: they need no new InterScale code and run while it is built.
2. C1 with the fold machinery; its first job measures whole-section cost.
3. C2's readouts (the new attribution by type and gene first), then C4's script on whatever tables exist.
4. C5 on synth_spot as soon as C2 runs, before the CosMx tables are interpreted.
5. C6 last.

## Risks

- Whole-section cost is unmeasured: fused attention's speed at 10^5 cells, and IG on sections (summed targets per receiver
  type and tile keep it to tens of attributions per section). Fallbacks: pooled patches, then windows.
- 979 genes cover only part of any LR resource; panel coverage is reported first, and controls are chosen from what is on it.
- Without truth, agreement can be shared bias (methods that all read co-location agree); the nulls and C5 are the guard.
