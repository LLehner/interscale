# Noise ceiling and band fields on CosMx (plan, status, results)

One place for the ceiling work: the goal, what the job computes, how to run it, what the runs found and what decides the next step.
Started 2026-11-10 (the user's rule: "noise ceiling first", roadmap.md "Reminder set 2026-11-10"). This is steps 0-1 of the
long-range plan `/home/lehnerl/Arbeit/long-range-interaction-plan-v2.md` (steps 2-3, the low-rank map and the comparison with the
attention model, stay there). Dated results entries are also in notes.md (2026-11-11, two entries).

## Goal

Every CosMx model so far ties the k-NN smoothers: within section x cell type the best model (type-residual target, 3 seeds) reaches
a centred cosine of ~0.087, the 128-nearest-neighbour mean 0.083 (notes.md 2026-11-10, cosmx_typeres). **Is that all the data
allow?** Two questions, one CPU job, no model:

1. **Noise ceiling (step 0).** How much of a hidden cell's within-type residual (its expression minus its section x type mean) is
   reproducible at all? Count splitting: each cell's raw counts are thinned binomially into two independent halves A and B
   (Poisson noise; Neufeld et al. 2024). How much of that reproducible part does the neighbourhood explain?
2. **Band fields (step 1).** Does any part of the neighbourhood beyond the smoothers' reach (rank > 128) carry linear signal about
   the residual? A ridge reads the mean visible expression in rank shells of neighbours, one band at a time and cumulatively.
   synth_spot (planted senders and length scales) is the positive control that must work first.

**How the answer decides the next step.**

- Ceiling close to the smoothers -> count noise is the limit; no architecture can do better on this target. Report the CosMx
  negative result with the ceiling.
- Ceiling far above the smoothers -> signal is left. Whether it is *neighbourhood* signal (worth a better spatial model) or
  cell-intrinsic state (subtype, quality) is read per gene from the explained share: the genes the neighbourhood explains, against
  the reproducible genes it does not.
- Far bands add nothing above noise on CosMx (and the FOV control does not explain them) -> skip steps 2-3 on CosMx, run them on
  synth_spot and legnini SHH only, move to the constrained ligand-to-response variant (roadmap). Far bands add something that is
  not the FOV -> steps 2-3 on CosMx.

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Noise ceiling by count splitting (`noise_ceiling.py`, step `ceiling`) | built 2026-11-10; run, results in `MA/11_11_results/fields/`; read 2026-11-11 |
| 0b | Depth model, full-depth ceiling, explained shares, `--reread` (step `reread`) | built 2026-11-11; validated on the first run's CSVs; full run 2026-11-12 (`12_11_results/fields/`), read 2026-11-12 |
| 1 | Band fields on the type residual, CosMx (`band_fields.py --dataset cosmx`, step `cosmx`) | built 2026-11-10; run, read 2026-11-11 |
| 1b | FOV control, alpha grid to 1e8 with an edge flag | built 2026-11-11, toy-checked; run 2026-11-12, read 2026-11-12 |
| 1c | synth_spot positive control (step `synth`) | built 2026-11-10; run, read 2026-11-11 |
| run 2 | Full job into `MA/12_11_results/fields/` (all four steps) | run and read 2026-11-12 |
| idea | Within-type residual by a finer cell-type label | not built (needs a fine label in `cosmx_inputs.h5ad`) |
| 2b | INS / GCG far signal: far gain by receiver type + distance profiles with mirror and background controls (`far_by_type.py`, step `bytype`) | built 2026-11-12, toy-checked (planted contact spillover found, no false far effect); not run |
| decision | Steps 2-3 of the long-range plan on CosMx | **skip** (2026-11-12, see run 2 read); synth_spot / legnini SHH only |
| 1d | Composition control: neighbours' cell-type shares per band + FOV (`comp<k>+fov`), the expression bands fitted on its residual (`cum<k>\|comp<k>+fov`, stagewise, out of fold) | built 2026-11-12, toy-checked (a planted type-independent field: 0.153 vs comp+FOV 0.094); run 3 pending |
| 2c | Spillover index for every gene + the near-term genes annotated (`spillover_index.py`, step `spill`) | built 2026-11-12, toy-checked (planted 15 um spillover: +0.99 SD, flagged alone; field genes "flat"); run 3 pending |
| L | The same audit on legnini (`--dataset legnini` in `noise_ceiling.py` / `band_fields.py`, step `legnini`): ceiling + bands + per programme x condition (SHH vs Ctrl organoids as negative control) | built 2026-11-12, stub-checked (planted SHH readouts: 0.17 in SHH, ~0.02 in Ctrl organoids); run 3 pending |
| run 3 | `fields.sbatch` default steps `ceiling cosmx reread spill legnini` into `MA/13_11_results/fields/` | written 2026-11-12, not run |
| 4 | Conditional reading by the global model: gene masking with the cell's other genes visible, type-residual target, FOV offset as a known input (`model.decoder.group_offset`, fork branch `conditional-context`), soft 150 um window; per-gene shares against ridges from the same inputs (`cosmx_conditional.*`, `slurm/new/`) | built 2026-11-12 (8 unit tests, configs validated); not run; after run 3 |
| idea 3 | Per gene, the typeres models (3 seeds) against the band ridge on the same 6,051 cells: genes where the model's share beats `cum3+fov` beyond the seed spread | noted 2026-11-12, for later (needs the per-gene within-type tables of `10_11_results/cosmx_typeres`) |

## Files and how to run

Local: `notebooks_and_scripts/slurm/new/fable/` (the user's folder for this job; overrides the flat-folder rule for it).

- `fields_common.py`: data as the sweeps see it (zero cells and NegPrb out, FOV-shuffled split 0 via a verbatim copy of
  `cosmx_stability.fov_split`), one graph per section and split, the hidden cells (discs of 60 um over 10 % of each graph,
  `interscale.tl.masking.sample_block_mask`, seed 0 train / 1 test: the same 6,051 test cells as the model reads), rank-shell
  neighbour means, the per-cell centred cosine of `masked_probing`, a block ridge fitted from accumulated moments (alpha chosen by
  sections held out in turn). `RIDGE_ALPHAS` 1 ... 1e8.
- `noise_ceiling.py`: step 0 and 0b; `--reread DIR [--fields fields_per_gene.csv]` recomputes the reliability tables from a
  finished run without the data.
- `band_fields.py`: step 1 (`--dataset cosmx` or `synth_spot`; `--fov-key`, default `slide_fov`, '' to leave the control out).
- `far_by_type.py` (2026-11-12): the follow-up to run 2. Part A refits cum3 / cum5 / cum3+fov / cum5+fov on the same hidden test
  cells (alphas from run 2's `fields_alpha.csv`), scores them per receiver cell type with FOV-block bootstrap intervals for the
  target genes, and lists the far bands' largest loadings per target gene. Part B (every cell, no fit) profiles a gene's within-type
  residual against the distance to the nearest source cell (pairs INS:Beta, GCG:Alpha; mirror controls PRSS2:Acinar, CPB1:Acinar),
  with the cells' negative-probe and total counts as a background control (B3), and the partial correlation with the source count
  in the far annulus 150-450 um among receivers beyond 50 um of any source cell (B2). Reading rule in its report: spillover = a
  contact profile the mirror controls share; background = raised over 50-450 um in every type with the negative probes;
  biology = confined to some receiver types, unmirrored, no background rise. Results in `$RESULTS/by_type/`.
- `spillover_index.py` (2026-11-12): per gene its source type (highest linear mean), and in every other type's cells its within-type
  residual (SD units of the receiver's type) by the distance to the nearest source cell: `z_contact` (< 15 um), `z_ref` (25-50 um),
  `z_mid` (50-150 um), `z_far`; `spill_index` = contact - ref with SE; flagged spillover at specificity >= 2, index >= 0.1 SD and
  >= 3 SE. With the run's `ceiling_reliability_genes.csv` and `fields_per_gene.csv`: the near-term genes (share beyond the FOV >
  0.05) annotated, `near_genes_annotated.csv`.
- `band_fields.py` composition control (2026-11-12): `comp<k>+fov` (cell-type shares of the visible neighbours in bands 1..k + the
  FOV mean) and `cum<k>|comp<k>+fov` (the expression bands fitted on its out-of-fold residual). Stagewise because one ridge would
  shrink the 27 composition columns with the expression bands' alpha ~1e6. Per gene `comp_gain`, `expr_beyond_comp`,
  `far_expr_beyond_comp`.
- legnini (2026-11-12): `fields_common.load_legnini` (raw counts from `layers['raw']`, coordinates x 0.138 um, `var['program']` from
  legnini_plan.md's genes, split train = the usual training organoids, test = the usual val + test organoids: 7 evaluation
  organoids, discs 60 um over 20 %); `noise_ceiling.py --dataset legnini` (scores within the organoid) and `band_fields.py --dataset
  legnini` (per programme x condition table). No FOVs, no cell types: no FOV or composition control there.
- `fields.sbatch`: one CPU job (96 GB, 8 cores, 4 h limit, time not measured yet). Steps: `ceiling cosmx synth reread spill
  legnini bytype`; default (run 3, 2026-11-12) `ceiling cosmx reread spill legnini`, results in `$FIELDS_RESULTS` (default
  `MA/13_11_results/fields`; runs 1 and 2 stay in `11_11_results` / `12_11_results`). Run 2 used `ceiling cosmx synth reread` into
  `12_11_results/fields`.

On the cluster: the Python files (`fields_common.py`, `noise_ceiling.py`, `band_fields.py`, `far_by_type.py`,
`spillover_index.py`) in `MA/scripts/` (flat), `fields.sbatch` in `MA/slurm/`; inputs `MA/cosmx_inputs.h5ad`, `MA/synth_spot.h5ad`,
`MA/legnini23.h5ad`; submit from MA:
`cd $MA && sbatch slurm/fields.sbatch` (`.out` / `.err` land in MA). One step: `STEPS="ceiling" sbatch slurm/fields.sbatch`.
Inputs: `MA/cosmx_inputs.h5ad` (layers raw + log1p_norm; obs section / slide_fov / cell_type_coarse) and `MA/synth_spot.h5ad`.
Env `interscale` (only `interscale.tl.masking` is imported; no model code).

Outputs (each report opens with a "Goal" section):

- `ceiling.md` + `ceiling_cells.csv`, `ceiling_per_gene.csv` (full rows carry `truth_var` from run 2 on);
- `ceiling_reliability.md` + `ceiling_reliability_genes.csv` (depth model, ceilings, shares; with the band fits after `reread`);
- `fields.md` + `fields_cells.csv`, `fields_per_gene.csv`, `fields_bands.csv`, `fields_alpha.csv` (`at_edge` column);
- `synth_spot/fields.md` + `synth_recovery.csv`, `synth_programmes.csv`, `synth_far_gain_by_programme.csv`.

Checks to read first in the log: 6,051 hidden test cells; `knn<k>_full` within type 0.049 / 0.069 / 0.083 (the model reads' bars);
no `alpha ... at the edge of the grid` warnings (or note which fits).

## Design

### Step 0: count splitting

Each cell's raw counts are split binomially, `p` into A (0.5 and 0.3 in the job), each half normalised like the model's input
(total 1e4, log1p). On the 6,051 hidden test cells:

- `own_half`: the cell's A half against its B half (both directions averaged; `own_half_AB`, `own_half_BA` kept);
- `knn<k>_half`: the mean A half of the k nearest visible cells against the cell's B half (k = 8, 32, 128);
- `knn<k>_full`: the k-NN means on the file's log1p_norm against the full truth (the check against the model reads).

Scores: per-cell centred cosine within section and within section x cell type; per gene the cosine pooled within type (a pooled
cosine of within-type-centred columns = the Pearson correlation over the hidden cells).

### Step 0b: what own_half means (corrected 2026-11-11)

`own_half` is a split-half **reliability** rho = S / (S + N), not a ceiling. The best possible predictor (the noise-free profile)
correlates sqrt(rho) with one noisy copy. Proof in the first run: for HSPA1A the 128-NN mean of other cells (0.567) beats the cell's
own other half (0.409). The first draft's read rule ("~0.1 within type -> data are the limit; ~0.3 -> models underfit") compared
the smoothers with rho directly and was wrong.

Depth model: noise variance shrinks as 1 / depth, so the reliability at depth fraction f is `r(f) = f / (f + c)`; the split-half
value is `sqrt(r(p) r(1 - p))`. Fitted per gene (and for the per-cell score) at the first thinning value:

- `r_full` = r(1) (Spearman-Brown for p = 0.5); `ceiling_full` = sqrt(r_full): what a perfect predictor of the full truth reaches;
- `share_<head>` = cos^2 / r_full (full-depth heads; half heads divide by r(1 - p)): the share of a gene's reproducible within-type
  variance the head explains (its R^2 against the noise-free residual);
- the model check: the split-half value it predicts at the other thinning against the observed one;
- summaries over the reproducible genes (r_full >= 0.1): median share, pooled share (sum cos^2 / sum r: genes in standardised
  units), variance-weighted share (log-expression units; needs `truth_var`, run 2 on).

The ceiling counts everything reproducible in a cell, its intrinsic state included; neighbours need not determine it. A low share
says "not predictable from the neighbourhood by this head", not "the model underfits". Per gene the read is sound; the per-cell
ceiling is approximate (a per-cell cosine pools genes of very different reliability).

### Step 1: band fields

Target: the hidden cell's residual `y - m(section x type)` (the training mean of its group; on synth_spot the slide's visible
mean). Features: the mean visible expression in rank shells of visible neighbours (CosMx ranks 1-8, 9-32, 33-128, 129-512,
513-2048; synth_spot 4 / 24 / 120 / 500 / 2000), standardised, centred per section. Fits, each `base + ridge(bands)`:

- `base` (label mean alone, 0 within type), `b<k>` (one band), `cum<k>` (bands 1..k; `cum3` reads what knn128 reads, `cum5` the
  full-rank kernel field), `far` (ranks > 128 only);
- CosMx FOV control (2026-11-11): `fov` = the mean visible expression of the cell's own field of view as one more block, alone and
  added to `cum3` and `cum5`. If `cum3+fov` reaches `cum5`, the far bands read a FOV offset (handling, imaging), not a field over
  distance. Per gene `fov_gain` = cum3+fov - cum3 and `far_gain_after_fov` = cum5+fov - cum3+fov;
- per gene: within-type cosine per fit, `gain<k>`, `far_gain` = cum5 - cum3, `best_band`.

Decisions taken while building (2026-11-10):

- **Rank shells, not um annuli.** One k-NN query serves every band; the first three shells are the k-NN bars; a band spans the same
  number of cells at any density. A 1.5 mm ball in dense tissue holds ~10k cells (10^8-10^9 pairs over 50k queries). Each band's
  radius in um is reported per section.
- **No "own" feature on CosMx.** The reads hide whole cells, so the base is the section x type training mean.
- **synth_spot recovery is two scores**, not best band vs planted scale (weak for any linear reader: a smooth peak is read best in
  the nearest band whatever its range): (a) sender identity, the largest source loading of a source-genes-only ridge against the
  planted sender; (b) far gain by programme (positive for long-range / sparse responses, ~0 for noise).

## Results

### Run 1 (job of 2026-11-10/11, `MA/11_11_results/fields/`, local copy `/home/lehnerl/Arbeit/results/11_11_results/fields/`)

Checks pass: 6,051 hidden cells; `knn<k>_full` within type 0.049 / 0.069 / 0.083.

Noise ceiling (per-cell centred cosine):

| thinning | head | within section | within type |
|---|---|---|---|
| full | knn8 / knn32 / knn128 | 0.090 / 0.122 / 0.143 | 0.049 / 0.069 / 0.083 |
| 0.5 | own_half | 0.139 | 0.106 |
| 0.5 | knn8 / knn32 / knn128 (half) | 0.087 / 0.118 / 0.140 | 0.046 / 0.064 / 0.079 |
| 0.3 | own_half | 0.131 | 0.098 |
| 0.3 | knn128 (half) | 0.134 | 0.075 |

Reliability read (`--reread`, 2026-11-11, `ceiling_reliability.md`):

- **The depth model holds.** Per gene, fitted at 0.5, it predicts the 0.3 split with Pearson 0.923, median observed / predicted
  0.995 (874 genes with r_split > 0.02); per cell within type 0.097 predicted vs 0.098 observed.
- **Ceiling within type at full depth ~0.44 per cell** (r_full 0.19); within section ~0.50 (r_full 0.25). knn128 reaches 0.19 of the
  within-type ceiling (0.29 within section).
- **Per gene** (585 of 960 genes with r_full >= 0.1; r_full quantiles 10 / 50 / 90 %: 0.046 / 0.115 / 0.227):

  | head | median share | pooled share |
  |---|---|---|
  | knn8 / knn32 / knn128 (full) | 0.003 / 0.005 / 0.012 | 0.021 / 0.031 / 0.039 |
  | band ridge cum3 / cum5 / far | 0.026 / 0.027 / 0.022 | 0.040 / 0.042 / 0.038 |

- **What the neighbourhood explains is stress / quality:** HSPA1A (share 0.39-0.50), HSPA1B, HSPB1, FOS, JUN, HSP90AA1/AB1, BAG3,
  ribosomal RPL34 / 21 / 22 / 32, TNFRSF12A, MIF, MT1X.
- **The most reproducible genes are cell-intrinsic:** SST (r_full 0.70, share < 0.01), IGF2, MALAT1, GCG, CHGA, TPSB2 / TPSAB1,
  ANXA4: subtype / state identity inside the coarse types that no neighbourhood predicts.
- Approximate parts: half vs full shares agree per gene (Spearman 0.85, median ratio 0.87), but knn128 half / full (0.079 / 0.083)
  is closer than the per-cell model predicts (0.062): neighbour-predictable signal sits in high-reliability genes. Read the genes.

Band fields, CosMx (within type): base 0; b1-b5 0.074 / 0.076 / 0.076 / 0.080 / 0.068; cum1-cum5 0.074 / 0.081 / 0.083 / 0.085 /
0.085; far 0.080; knn128 0.083. **Far gain over cum3 +0.002**; the genes that gain are heat-shock / immediate-early (HSPA1A +0.056,
FOS, HSPA1B, BAG3, HSP90AA1, HSPB1) and islet hormones (INS, GCG). Band radii (median outer radius per section): rank 8 ~34-41 um,
32 ~52-64 um, 128 ~85-106 um, 512 ~145-193 um, 2048 ~297-470 um. Ridge alphas mostly at the top of the grid then (1e5-1e6, 1e6
the largest tried): the grid now goes to 1e8.

Band fields, synth_spot (within slide): nearest band (rank 4, 283 um) best, 0.399 vs knn4 0.303; more bands lower it (cum5 0.346).
Source-only far gain by programme: long_range +0.101 (median 0.064), sparse_response +0.019, noise programmes ~0, region_marker
-0.067; it grows with the planted scale (2-4 mm responses +0.22 to +0.37, <= 1.2 mm ~0 except MC14/15). Sender identity 5 / 25
(20 %): long-range 5 / 15 (planted sender median rank 3), sparse 0 / 10 (chance). Sparse responses are not linearly recoverable.

**Read of run 1.** The data are not the limit: about a fifth of the within-type residual is reproducible at full depth, and the
smoothers and every model explain only ~4 % of that. But the reproducible part the neighbourhood misses looks cell-intrinsic
(subtypes, hormones, MALAT1), and what it does explain looks technical (stress, ribosomal). No linear long-range signal beyond rank
128 on CosMx; the positive control works for long-range, not for sparse responses.

### Run 2 (2026-11-12, `MA/12_11_results/fields/`, local copy `/home/lehnerl/Arbeit/results/12_11_results/fields/`)

Added: variance-weighted shares (`truth_var`), the FOV control, alphas to 1e8 with edge flags, the reliability read with the band
fits in one go. To read: does `cum3+fov` reach `cum5` (far gain = FOV offset)? Do the shares change with unclipped alphas? Are
the far-gain genes still heat-shock after the FOV block?

**Read 2026-11-12.**

- **Reproducible.** Every number run 1 also computed is identical (same split, discs, thinning seed): ceiling, k-NN bars, band
  fits, synth_spot. Checks pass (6,051 cells, bars 0.049 / 0.069 / 0.083).
- **The alpha grid was not the limit.** With 1e7 and 1e8 added, every fit still chooses 1e5 or 1e6 and the held-out score falls
  above it (cum5: 0.102 at 1e6, 0.095 at 1e7, 0.091 at 1e8); no fit at the edge, on CosMx or synth_spot. Run 1's fits stand.
- **Variance-weighted share** (log-expression units, reproducible genes): knn8 / 32 / 128 0.037 / 0.055 / 0.067; band ridge cum3 /
  cum5 0.060 / 0.062; far 0.057. In expression units the neighbourhood explains ~6-7 % of the reproducible within-type variance
  (standardised 4 %, median gene 1-3 %): higher because the genes it explains (stress) are highly expressed. The plain 128-NN mean is
  as good as the ridge (0.067 vs 0.062 weighted; within type per cell 0.083 vs 0.085).
- **FOV control, per cell (within type):** fov alone 0.048 (= knn8); cum3 0.0828, cum3+fov 0.0837, cum5 0.0853, cum5+fov 0.0852.
  The far gain (cum5 - cum3) is +0.0025; the FOV takes +0.0010 of it, +0.0015 remains (cum5+fov - cum3+fov). Both are in the
  third decimal, no spread measured (one draw, no bootstrap).
- **FOV control, per gene: the stress far gain is the FOV.** HSPA1A far gain 0.056 -> 0.004 after FOV (fov alone 0.504 vs cum3
  0.418), FOS 0.036 -> 0.002, BAG3 0.024 -> -0.004, HSP90AA1 0.011 -> -0.008, HSPB1 0.012 -> -0.006. Of 22 genes with far gain >
  0.01, 6 keep it after the FOV.
- **What survives the FOV is islet-related:** INS (far gain 0.022, after FOV 0.025, FOV gain 0), GCG (0.014 -> 0.011), plus small
  ones (TIMP1, RGS2, INSR, CUZD1, 0.008-0.011). The INS level 150-450 um away (ranks 129-2048) predicts a cell's within-type INS
  residual beyond its 128 nearest cells: how much islet tissue lies around it. Could be biology (peri-islet acinar / ductal
  cells, the insulo-acinar axis) or technique (INS / GCG transcripts spilling into neighbouring segmentations, ambient RNA near
  islets). Not separable from these tables: needs the receiver's cell type.

**Split of the neighbour term (2026-11-12, from the run 2 tables; notes.md same date):** pooled over the reproducible genes, FOV
mean 2.7 %, near neighbourhood given the FOV 1.4 %, far given the FOV 0.1 %. 30 genes have > 5 % of their reproducible variance
in the near neighbourhood beyond the FOV (RPL34, RPL21, JAK1, MIF, GSTP1, TNFRSF12A, TUBB, EZR ... ; acinar markers PRSS2 / REG1A at
contact scale = spillover-like, the rest regional). Open: composition vs expression (a type-count block per band), not built.

**Decision (2026-11-12).** Far bands add +0.0015-0.0025 within type, the stress part of it is the FOV, and the rest sits in one
or two islet hormones. By the rule above: **steps 2-3 of the long-range plan are skipped on CosMx**; the CosMx result for the thesis
is "within-type residual ~19 % reproducible at full depth (ceiling ~0.44 per cell); neighbours explain ~4-7 % of that, all within
~100 um (rank 128) except an islet-hormone signal; the reproducible rest is cell-intrinsic (subtype markers SST, GCG, IGF2,
MALAT1)". Optional follow-up: INS / GCG far gain split by receiver cell type (cheap, model-free).

## Open ideas

- **Finer cell types.** SST inside the coarse types suggests that subtype identity is the largest reproducible part of the
  within-type residual. A residual by fine label (if `cosmx_inputs.h5ad` has one) would show how much reproducible signal is left
  after subtypes, and whether the neighbourhood explains more of that remainder.
- **Gene classes for the thesis.** `ceiling_reliability_genes.csv` already sorts genes into reproducible-and-spatial,
  reproducible-and-intrinsic, and noise: a model-free reference for "SVGs among the most influential genes" (roadmap success
  criterion).
- Steps 2-3 (low-rank per-band map, comparison with the attention model): see the long-range plan, after the decision point.
- **Idea 3 (the user, 2026-11-12: for later).** The typeres runs (3 seeds, `10_11_results/cosmx_typeres`) were scored on the same
  6,051 cells. Per gene, their within-type cosine against the band ridge's (`fields_per_gene.csv`, `cum3+fov` or the composition
  fits of run 3), both as shares of `r_full`: the genes where the nonlinear model beats the linear neighbourhood beyond its seed
  spread. No training; needs the typeres sections' per-gene within-type table (check that the sweep wrote one).

## Item 4: conditional reading by the global model (2026-11-12, built, not run)

What a transformer can add over a band ridge on CosMx is conditional reading (weights that depend on the receiver's state and on
which sender is present), not range. The sweep `cosmx_conditional` gives the model what a ridge already has, as known inputs, and
scores it per gene against ridges built from the same inputs.

- **Model option** (fork, branch `conditional-context`): `model.decoder.group_offset: visible_mean` with `group_offset_key` (an obs
  column on the graphs): a part `group` = the FOV's visible mean of the residual from the type means (under `section_offset:
  label_mean`), times a weight per gene starting at 1; no hidden entry enters it; not with `section_offset: visible_mean`;
  checkpoint tag `-grp-<key>`. `tl.masking.group_visible_mean`. Tests `tests/test_group_offset.py` (8; two mutations caught).
- **Evaluation**: `draw_masks(kind="gene_set+blocks")` hides a gene set only in disc cells (rate 0: the cells of a disc
  evaluation, random number for random number: the 6,051 cells); `FittedReferences(group_attr=, composition_attr=)` adds the
  `group` and `composition` parts and the references `ref_own+group`, `ref_own+context+group`, `ref_own+composition+group`,
  `ref_own+context+composition+group`. `sweep_common.py`: `EVAL_GENE_SET_DISCS`, `FITTED_GROUP_KEY`, `FITTED_COMPOSITION_KEY`.
- **Arms** (300 epochs, FOV-shuffled split 0, typeres' model with `visible_keys_only` off): half the genes hidden (channels) in 20 %
  of the cells + 60 um discs over 10 %; `cond_fov_w150` (seeds 42, 43; FOV offset; windows 15-150 um frozen: soft, ~e^-4.5 at
  450 um), `cond_w150` (no FOV offset), `cond_fov_w1000` (learned 15-1000 um windows), twin of `cond_fov_w150`. Evaluated on
  cosmx_masking's random half (`rand_eval`) hidden in the 6,051 disc cells.
- **Read** (`cosmx_conditional_read.py`): per gene the share of the reproducible within-type variance (cos^2 / r_full from run 3's
  ceiling) of the model and of each ridge; `gain` = model minus the best ridge; conditional genes = gain > 0.01 in both seeds and
  <= 0 in the twin, annotated with the near-term share and the spillover profile; the FOV and range arms against the lead arm.
- **Expectation, stated before the run:** cosmx_masking's rand50 tied the own-cell ridge (0.166 vs 0.167 within type); the
  model-free near term beyond the FOV is 1.4 % pooled. A positive result is a handful of genes with gains of a few % in both
  seeds; a null is "no conditional reading beyond what ridges get from the same inputs", which closes the CosMx question for
  the thesis.

## What it means for InterScale (2026-11-12)

**The data, then the task, then the architecture.** On CosMx pancreas, a cell's expression decomposes into: its cell type and
section (most of it), a FOV-sized offset, a small neighbourhood term within ~100 um (4-7 % of the reproducible within-type
variance), and a large cell-intrinsic part no neighbourhood predicts (subtype markers: SST, GCG, IGF2; MALAT1). Beyond ~100 um,
linearly, nothing is left but the FOV and possibly an islet-hormone signal (under test, `far_by_type.py`). Every model tying the
128-NN smoother is therefore the correct outcome on this dataset and target, not underfitting. Consequences:

1. **A global component has no long-range target on CosMx.** Whatever a transformer reads beyond the smoother's reach here is
   FOV-sized offsets (handling, imaging or tissue region; not separable from these data) and composition. An attention-based
   readout will display both as long-range "flows". The within-type shuffle already showed that type-level flow maps (Steamboat;
   InterScale's at the floor) are composition; the FOV control shows that the stress genes' spatial signal is FOV-level.
2. **The preprint's CosMx programmes should be re-read with these controls.** It reports a local programme with INS and oxidative
   stress and a global immune / stromal programme, on full expression. On full expression both are expected from composition
   (which cell types surround a cell) and offsets; within type, the stress genes' "neighbourhood" signal is mostly a FOV offset
   (HSPA1A: FOV mean alone 0.504 vs the 128-NN ridge 0.418), and INS is under test. Not a refutation (the targets differ), but
   the claims need the within-type target, the FOV covariate and the smoother before they read as communication.
3. **Linear null, not a proof of absence.** The bands are linear in neighbour means. Sparse or rare-sender effects (synth_spot's
   sparse responses: not linearly recoverable) and thresholded or combinatorial ones could escape, but every nonlinear model tried
   on CosMx also ties the smoother, so there is no positive evidence for them either. State it as "none detectable with tools that
   recover planted long-range signals on synth_spot".
4. **What a long-range model needs is a dataset where the audit finds something.** The audit (ceiling + band fields + FOV control)
   is model-free, CPU, about an hour, and should run before any model: on CosMx it would have predicted the whole sequence of
   "ties the smoother" results. The candidate with known long-range biology is legnini (SHH morphogen; the preprint's positive
   control): if the far bands gain on the SHH readouts there, the global component has a real target and the constrained SHH
   variant (slot S4) is the thesis' positive real-tissue result; if not, the global component is shown only on synth_spot.
5. **For InterScale as a package:** within-type (or within-niche) targets, section / FOV offsets as nuisance terms, the k-NN /
   band-ridge bars and the noise ceiling belong in its standard evaluation; credit a model only for its explained share of the
   reproducible variance beyond the smoother. Being model-free, the audit could go upstream without touching the original model
   (the "PRs use the original model only" rule is about model code).
6. **For the thesis:** 4.7 becomes a negative result with its mechanism (ceiling, shares, FOV, composition); 5.1 "the task, not the
   architecture" extends to "the data first"; S6 (low-rank band maps on CosMx) is dropped; legnini audit next.
