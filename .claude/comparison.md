# Prototype vs best global-only model: gene rankings (2026-11-23)

Status: **built 2026-11-23, not run** (see "Round 2" at the end; the status table there supersedes the one below). Not to be confused with `comparison_plan.md` (2026-10-25: CosMx runs
against other CCC / SVG tools).

## Question

Does the redesigned global-only model rank interacting genes better than the original InterScale prototype, when both are trained on the
same task and read by the same interventions? The answer is a thesis result (what the redesign gained), not a tuning step.

## The user's decisions (2026-11-23)

- **Prototype** = the original CombinedModel (GCN local component + self-attention transformer, as in the upstream / older checkout) with
  **only the masking changed**, so that every gene is a target sometimes and the model can be ranked (see "Why the masking must change").
- **Datasets**: synth_spot and the normal CosMx data (not the spiked semi file).
- Approach as proposed in the session: identical readouts and bars for both models; the region control added to the readout first.

## Why the masking must change

The prototype reads its own cell (the GCN mixes the cell's own row into its token, the transformer attends over every token). A gene can
therefore be ranked only where it is hidden, and a gene that is never hidden is never a target and gets no gradient in the decoder: with
the original masking (random cells / genes) or J1's programme masking, a ranking over all genes would rediscover the mask (the reason J5
retrained synth_spot, ccc_model_plan.md E7). Channel masking hides 20 % of the genes **in every cell** (`dataset.mask_channel_fraction:
0.2`, redrawn every epoch), so a hidden gene can only be predicted from other genes, of the cell itself and of other cells, and every
gene is a target sometimes. Both models are read in the same channel folds (`gene_ranking.gene_folds`: ~20 % of the genes hidden in
every cell, 2 partitions), which is the only pass on which the prototype's ranking means anything.

## Arms

Everything trained in the fork's code (the channel masks, the folds and the readouts exist only there); the prototype's architecture is
the upstream one expressed in fork config keys. Check before the run: build the prototype from the fork and from the older checkout with
the same config and compare `state_dict` shapes (recipe in the memory "PRs use the original model only"); the masking keys are the only
difference allowed.

| Dataset | Arm | Model | What | Seeds |
| --- | --- | --- | --- | --- |
| synth_spot | `proto_ch` | CombinedModel | GCN (upstream default depth) + transformer, no positional input, shared decoder, long-range mask as upstream; channel masks 0.2; 300 epochs | 42, 43 + twin s42 |
| synth_spot | `ir_ch` | GlobalModel | J5's arm (J1's ir2_rate: context query, two read layers, read-out attention, null key, content rate, additive), channel masks 0.2 | 42, 43 + twin s42 |
| synth_spot | `ctx_ch` | GlobalModel | J5's reference read-out (context query, dense attention, feed-forward on the output) | 42, 43 + twin s42 |
| CosMx | `proto_ch` | CombinedModel | as on synth; J4's split, preprocessing and target | 42, 43 + twin s42 |
| CosMx | `ir_c_ktype_ch` | GlobalModel | J4's `ir_c_ktype` (keys from type: the seed-stable arm, C1 0.61) with channel masks 0.2, read in the same folds | 42, 43 + twin s42 |
| CosMx | `ir_c_ktype` | GlobalModel | J4's checkpoints, read as in J6 (clean pass, every cell a target): a reference for what the channel masks cost | 42 (existing) |

All arms are retrained in one job per dataset (not J5's checkpoints) so that one code version trains everything; 300 epochs (CosMx cap).
`ctx_ch` stays because J5 showed it as good as `ir_ch` on responders.

**Expected risk.** On 2026-11-11 the original model (no positional input, mask off, 20 epochs) never left the gene-mean floor on CosMx.
If `proto_ch` stays at the floor here too, its rankings equal its twin's and the comparison says so: that is the result, not a failure of
the job. The fit is read first (centred cosine on hidden entries per arm against the references), before any ranking. An optional arm
`proto_ch_pe` (the prototype + J1's distance window) would say whether a floor comes from the missing positional input; **not in the
first run unless the user asks** (the decision was "only the masking changed").

## Readouts (identical for every arm)

E7's interventions (`evaluation/gene_ranking.py`), per receiver group (synth: every spot; CosMx: cell type), read in the channel folds:

1. **Responders**: other cells' visible states swapped within label; a target's drop.
2. **Drivers / pairs**: one sender gene permuted within label among the other cells; the drop of every other target.
3. **Sender types** (CosMx): identity minus identity_null.

Extended before the run (C0 below) with the three controls of the anchored SVG task's R2 (`evaluation/anchored.py`), so both models are
judged beyond the confounds the 2026-11-23 reads found:

- **Beyond region**: swaps and permutations drawn within label x region (synth_spot: `obs['region']`; CosMx: a niche label computed on the
  training cells from neighbourhood composition, `tl.niches` / imitation labels; the key comes from config, as everything dataset-specific).
  A region-level effect survives the permutation and is not scored. Both versions (plain, beyond region) are reported.
- **Sender groups**: genes whose local means co-vary across neighbourhoods (|r| >= 0.6 of k-NN means over training cells, average
  linkage, label-free) permuted together, and with one member kept (sufficiency, as in anchored.rank_tables): a source and its responses
  are ranked as a group, then split by sufficiency.
- **Null and stability**: each score as an expression-matched z against the arm's twin and within-label permutation nulls
  (`ranking.rank_by_effect`), and seed agreement (RBO); a gene counts as found when it is in the top k in both seeds.

Bars (model-free, same cells): within-type Moran's I, within-type variance, the neighbour ridge, each arm's twin (as E7).

## Outputs

Per dataset `MA/<dd>_11_results/compare_<dataset>/`:
- `ranking/<arm>_{responders,drivers,pairs,groups}_<plain|region>.csv`: the final rankings (seed means, twin, z, found).
- Plots: per arm the top-25 senders and responders (local / global where the reach is defined: synth by distance bins of the pair
  effect); **prototype vs global-only**: rank-rank scatter per ranking (truth roles coloured on synth), precision-recall curves against
  synth_spot's truth with the bars, CosMx overlap (RBO between the two models' rankings, per type) and flagged share (stress, ribosomal,
  spillover) of each model's top 50.
- `compare.md`: fit table, rules, the rankings' heads, timing.

## Decision rules (proposed; to be fixed by the user before the run)

synth_spot (truth: `roles_truth`; hard negatives = spatial / marker / batch / condition genes):
- **P1 pairs**: AUPRC of the global-only arm minus the prototype's >= 0.05, and each above its own twin.
- **P2 responders beyond region**: AUROC against the hard negatives >= 0.70 for the global-only arm (E7 plain: 0.32), and above the prototype's.
- **P3 drivers (groups)**: the planted source's group in the top 3 groups of its responses for at least half of the programmes; reported
  per arm.
CosMx (no truth):
- **Q1 stability**: RBO over seeds >= 0.5 per arm (reported for both; the prototype's against its twin's).
- **Q2 artefacts**: flagged share of the top 50 responders per type, beyond region; lower is better, reported.
- **Q3 known pairs**: LR support AUC (E7's `lr_support_by_type`), reported (null in E7 for every arm).
- **Q4 agreement**: RBO between the prototype's and the global-only ranking per type: high = the redesign finds the same genes.

## Jobs and cost (estimates; measured times in the job logs)

One sbatch per dataset (the read-out jobs may be arrays: user 2026-11-21), each run under 12 h; flat in `slurm/November/`:
`compare_rank.py` (`COMPARE_JOB=synth|cosmx`, built on `gene_rank.py` / `gene_rank_common.py`), `make_compare_yaml.py`, `compare_job.sbatch`,
`compare_submit.sh`, `compare_read.py`.
- synth: 3 arms x 2 seeds + 3 twins; training ~15-30 min per run (J5), readouts in folds 15-25 min per model (J5) x 2 (plain + region) +
  groups -> one array task per run, ~1.5 h each.
- CosMx: training ~3 s / epoch for the global model (~15 min at 300 epochs); the prototype with the GCN on whole sections unmeasured;
  E7 readouts took 2.3 h per model on the clean pass, in folds and with the region version perhaps 2-3x -> one array task per run, measure on
  a smoke run first; split senders across tasks if a task would pass 12 h.

## Status

| Stage | What | Status |
| --- | --- | --- |
| C0 | `gene_ranking`: label x region swaps / permutations (config key), sender groups + sufficiency, z / found columns; tests | proposed |
| C1 | Prototype arms in fork keys + state_dict check against the older checkout; yaml makers for synth and CosMx | proposed |
| C2 | Job files (`compare_*`), reader, plots; toy checks of the hook on real modules at toy size | proposed |
| C3 | Runs: synth, CosMx (smoke first) | proposed |
| C4 | Read against the rules; notes.md entry | proposed |

## Open points

1. The rules P1-P3 and Q1-Q4: confirm or change before C3.
2. CosMx region key: a niche label from neighbourhood composition (annotation-derived, conservative) or the FOV (purely technical)?
3. `proto_ch_pe` as an optional second prototype arm: yes / no.

## How the separation could be fixed (the levers behind the readout controls; session 2026-11-23)

The rankings fail where interaction is confounded with region, co-expression and noise. Readout controls (above) remove the confound from
the scores; model changes remove it from what the model learns. Both, in order of expected effect:

- **Region / position.** Readout: permute within region (above). Model: a decomposed prediction, `y = position part + interaction part`,
  the position part a low-frequency function of the cell's coordinates fitted per slide on visible cells (low Fourier / spline modes, or the
  existing group offset over regions), trained jointly so that the interaction part (sum over senders, IR) only gets what position cannot
  explain; rank from the interaction part only. The resolution of the position part sets the scale split: a field coarser than it can still
  be read as interaction, so it is a knob (the local / global question itself), chosen below the longest planted programme on synth.
- **Co-expression.** Readout: group knockouts and conditional replacement (a sender gene redrawn from its conditional given the other genes,
  `deletion.sender_deletion`, so only its unique information is removed) and sufficiency within groups; report groups, not single genes,
  when a group is collinear. Model: sender bottlenecks that force a choice (keys from type, a sparse / group-lasso sender projection) only
  help where a mechanistic assumption holds (responses sit in receivers, sources are focal); observational data alone cannot tell a source
  from a co-varying response.
- **Noise.** Rankings by z against twins and permutation nulls, stability selection over seeds (top k in both), ensembles (flow ensembles
  2026-11-09: ~0.9 agreement at 10 seeds), genes ranked only where their split-half reliability allows (noise ceiling), artefact flags as
  negative controls (stress, ribosomal, spillover = neighbouring types' markers).


## Round 2 (2026-11-23): the user's answers, the fixes, the build

### The user's answers

- CosMx region key: **a niche label** (open point 2). The optional prototype arm `proto_ch_pe`: **yes** (open point 3). Rules P1-P3 / Q1-Q4: not
  answered; reported as proposed, not applied as pass / fail.
- Also asked: implement fixes 1-3 (the levers above), a script that compares them **step-wise** (fix 1, then 1 + 2, then 1 + 2 + 3), commit to main,
  and find why the CosMx E7 run lost `typeres` (the .err files are gone; check the code for anything missing or buggy).

### Why typeres was lost (found, fixed)

`typeres` (the reference: hierarchical attention + raw own-cell skip + label_mean) reads its own cell. The J6 hook read every CosMx run with
`TARGETS = "all"` (the clean pass, every cell a target), and `interpretation.scored_entries` refuses a model that is not blind to its own cell (it would
score copying). Both `typeres` tasks (the run and its twin) raised in the hook and sweep_common printed "evaluation failed" and went on. The toy check
before J6 used a typeres-like module without the own-cell skip, so it passed. Fix: `TARGETS = "auto"` (gene_rank_common): the clean pass for models
blind to their own cell, channel folds for models trained with channel masks, else `cells`: a random 20 % of the cells hidden whole (one draw,
`gene_ranking.hide_cells`), as typeres was trained. Re-read: `RANK_RUNS=typeres bash slurm/rank_submit.sh cosmx` (joins the same results folder).

### The fixes as built

| Fix | Readout (every arm) | Model (step-wise arms) |
| --- | --- | --- |
| 1 region | `strata`: `gene_flows` / `context_flows` permute within label x stratum (synth `obs['region']`, CosMx `niche_c`); level R1 | `ir_ch_f1`: `section_offset: label_mean` on the region (synth) / section x type x niche (CosMx); the label's training mean is a fixed part, so the model learns what lies beyond it |
| 2 co-expression | `co_varying_groups` (|r| of k-NN local means, average linkage at 0.6, training cells) -> `group_sets` (whole group, group minus one member) passed as `gene_flows(sender_sets=)` (one shared permutation per set, member targets left out); `group_table` (sufficiency), `group_pair_scores` (a member credited with group drop x sufficiency); level R2 | `ir_ch_f12`: + `optim.aux_loss_weights.sender_sparsity` 0.1 (L1 / L2 of the expression embedding's per-gene column norms over sqrt(G): scale-free, because the tokens are layer-normalised and a plain group lasso would only shrink them) |
| 3 noise | `robust_z` (each drop against the other senders of the same target: median / MAD, MAD floored), averaged over seeds, pair stability (share of seeds with the pair in the top k); level R3 | `ir_ch_f12` at 4 seeds (the ensemble) |

The niche for CosMx: `tl._preprocessing.neighbourhood_niches` (composition of the 30 nearest cells of the section, 6 k-means niches fitted on the
**training** cells; J4's `niche` column was fitted on a sample of all cells, so the comparison uses its own, `niche_c`).

### Step-wise design (compare_read.py)

| Step | Arm | Readout level | What changes |
| --- | --- | --- | --- |
| S0 | `ir_ch` | R0 | nothing (E7's read) |
| S1 | `ir_ch_f1` | R1 | fix 1: label part + read beyond region |
| S2 | `ir_ch_f12` | R2 | fix 1 + 2: + sender sparsity + group read |
| S3 | `ir_ch_f12` (4 seeds) | R3 | fix 1 + 2 + 3: + robust z, ensemble, stability |

Beside them: the readout alone (`ir_ch` at R1, R2, R3) and the model alone (`ir_ch_f1`, `ir_ch_f12` at R0), so each fix's two halves are told apart;
the prototype comparison (`proto_ch`, `proto_ch_pe`, `ir_ch` at R0 and R3, against their twins). Every arm is read at every level.

### Decisions taken in the build (say so when reading)

- **The training recipe is shared** (SCE on centred targets, width 64 synth / J4's on CosMx, 300 epochs, J4's split and crops): the prototype differs
  in the model only (CombinedModel, GCN + transformer, dual decoder, no positional input, no centred queries, no own-cell skip, no label part).
- **CosMx prototype attention = `fused`** (exact dense attention; dense S x S cannot hold a whole section), trained on J4's crops.
- **CosMx senders: the 200 most variable within type** on the training cells (`SENDERS_TOP`), one fold partition (5 folds), to keep a task under
  12 h; synth reads every gene in 2 partitions.
- **synth region = the generator's annotation** (an oracle stratum; CosMx's niche is estimated). The label part on synth's region is therefore an
  upper bound of what fix 1 can do there.
- Sender-sparsity weight 0.1 (one value, not swept).

### Files

Fork (main): `evaluation/interpretation.py` (`strata`, `sender_sets`), `evaluation/gene_ranking.py` (`hide_cells`, `co_varying_groups`, `group_sets`,
`split_flows`, `group_table`, `group_pair_scores`, `robust_z`, `strata` in `context_flows`), `train/aux_losses.py` (`SenderSparsity`),
`config/optim_config.py`, `tl/utils.py` (tag `ssp`), `tl/_preprocessing.py` (`neighbourhood_niches`); tests `test_gene_ranking.py` (+5),
`test_sender_sparsity.py` (3), `test_niches_composition.py` (1). Cluster (`slurm/November/`, not under git): `make_compare_yaml.py`, `compare_rank.py`,
`compare_read.py`, `compare_job.sbatch`, `compare_submit.sh`; `gene_rank_common.py` (TARGETS auto / cells, the fix reads, truth json, flags on the
region table), `gene_rank.py` (TARGETS auto for CosMx, `RANK_RUNS`).

### Checks run (no model on real data)

- Library tests: strata constant = no strata draw for draw, singleton strata move nothing; a one-gene set = the single sender; a set's cells keep their
  own combination of the set's genes (a mutant with one permutation per column caught); group table and credited pairs by hand; robust z; hidden
  cells; co-varying groups. Sender sparsity: bounds, scale invariance, concentration lowers it, a gradient step concentrates, refusal without Linear.
- Every arm of both jobs through `main_sweep.apply_sweep_config` with the local copies of the base yamls (validators, checkpoint names <= 231
  characters); `sweep_common.configure()` for both jobs (12 runs each).
- The hook end to end on toy graphs with a planted-dependence module (synth path with truth and regions; CosMx path with types, niches, flags,
  SENDERS_TOP), then compare_read.py on its outputs (tables, csvs, plots looked at).
- Not checked locally: building the real models (no model is built here); the first cluster step is the smoke run (`SMOKE=1`).

### Run order

1. `cd $MA && SMOKE=1 bash slurm/compare_submit.sh synth` (5 tasks: each arm's first seed, 2 epochs, small readouts, -> `23_11_results/smoke/`); the
   same for cosmx. 2. `bash slurm/compare_submit.sh synth` and `cosmx` (12 tasks each + report). 3. `RANK_RUNS=typeres bash slurm/rank_submit.sh cosmx`.

### Expectations (before any run)

- `proto_ch` near its twin on every ranking (11-11: the original model at the gene-mean floor); `proto_ch_pe` better on pairs if the floor is the missing
  position.
- Fix 1 on synth: responders' AUROC against the hard negatives up from 0.32 (region markers no longer read as responders), at some cost on the
  longest programmes (MC11, 4 mm, absorbed by the region where it is region-sized).
- Fix 2: drivers above the twin (E7: 0.37 vs 0.30) once collinear partners share credit; the source most sufficient in its group uncertain.
- Fix 3: little change in AUPRC, fewer unstable pairs; on CosMx a higher seed RBO.

### Status (round 2)

| Stage | What | Status |
| --- | --- | --- |
| C0 | readout controls (strata, sender sets, groups, robust z, hidden cells) | built, tested |
| C0b | model fixes (label part on region / niche: existing; sender sparsity: new; niche labels: new) | built, tested |
| C1 | arms (make_compare_yaml.py), validated | built |
| C2 | compare_rank.py, compare_read.py, sbatch / submit; toy end-to-end | built |
| C3 | runs: smoke, synth, cosmx; typeres re-read | not run |
| C4 | read, notes.md | open |

## Round 3 (2026-11-23): CosMx read on tiles, not whole sections (the user's decision)

The user asked why the CosMx prototype uses the fused attention ("100x longer per iteration"; whole slides fit on the GPUs). The measured numbers
(notes.md, the entry before 2026-10-31): exact dense attention on a whole 100k-cell section takes 127 s per training step and 41.7 s per evaluation
pass at 5.6 GiB -- memory is not the limit, the quadratic compute is; plain dense attention cannot form the 10^10-entry map at all, and what runs fast
on whole sections is the routed hierarchy (1.2-2.1 s per step) or the GCN. On the training crops (1 mm) the fused attention computes the same as dense
(plus a recompute in the backward pass; not measured at crop size), so training was never the problem. **The readout was:** every CosMx arm read on
whole test sections means ~42 s per pass for the prototype x ~2,000 passes per model (2 reads x 5 folds x 200 senders + groups) = ~23 h, past the
12 h limit -- my cost estimate had used the global-only arms' ~8 s per pass from J6. Options offered: (1) read every arm on 1 mm tiles of the test
sections, (2) the routed hierarchy for the prototype (no longer the prototype), (3) whole sections with ~50 senders. **The user chose (1).**

Built: `gene_ranking.tile_graphs(graphs, size, min_cells=)` (square tiles over each graph's bounding box by position alone, every cell in exactly one
tile, tiles under `min_cells` merged into the nearest kept one, every per-cell attribute and `obs_names` sliced, edges kept within a tile, `tile_of`
= the source graph; tested) and `EVAL_TILE_UM` / `EVAL_TILE_MIN_CELLS` in gene_rank_common (test graphs only; training graphs for the bars, groups
and senders stay whole). compare_rank.py, CosMx: `EVAL_TILE_UM` 1000 (the training crops' side, make_cosmx_tiers_yaml.CROP), `EVAL_TILE_MIN_CELLS`
200. synth is unchanged (whole slides of 2,500 spots, one attention block). The prototype keeps the fused attention for training (exact dense
attention without the S x S map: crops hold up to ~20k cells, and a dense map with its backward at that size would not fit a 20 GB GPU slice).

What it changes in the reading: every score is centred within (tile, receiver group) instead of (section, receiver group), so variation between
1 mm tiles of a section (FOV-level offsets, the section's large-scale structure) is not scored; no model sees context beyond ~1 mm at readout, as in
training. The neighbour ridge and Moran bars on the test cells are computed per tile as well. Toy check: the hook on 2 toy sections read as 18 tiles,
then compare_read.py, end to end.

## C3 read (2026-11-24; results MA/23_11_results/compare_{synth,cosmx}; jobs 43855175 synth, 43855321 cosmx, both 12 tasks + report, no errors)

### synth_spot (complete)

Fit (centred cosine on hidden entries): ir_ch 0.16-0.17, proto_ch 0.131, proto_ch_pe 0.136, twins ~0. The prototype is **not** at the floor here
(the expectation from 11-11 was wrong for synth with channel masks).

Rules (proposed, not fixed by the user; applied here as a reading aid):
- **P1 pairs:** ir_ch R0 0.156 vs proto_ch 0.073 (+0.083, both above their twins 0.010 / 0.018): pass. Against proto_ch_pe 0.147 the gap is 0.009:
  the gain over the prototype is the positional input, not the redesign. The model-free bar (0.196) is above every model at R0; only
  `ir_ch R2` (readout fix 2 alone) reaches it (0.195).
- **P2 responders beyond region (AUROC_hard >= 0.70):** fail everywhere (best 0.476, model-only ir_ch_f1 R0; bar 0.555; the untrained ir_ch twin
  0.519). Region markers / spatial genes still rank with the responders.
- **P3 source group in the top 3 for half the programmes:** proto_ch R3 0.556 (pass), proto_ch_pe 0.481, ir_ch R2/R3 0.444, ir_ch_f12 0.259
  (twins 0.15-0.22).

Step-wise (S0 -> S3): fix 1 helps on every ranking (pairs 0.156 -> 0.174, drivers 0.376 -> 0.404, responders AUPRC 0.338 -> 0.404, AUROC_hard
0.305 -> 0.399); its readout half alone gives the same responder gain (ir_ch R1 0.453 AUROC_hard). Fix 2's model half (sender sparsity 0.1)
hurts: pairs 0.174 -> 0.132, drivers 0.404 -> 0.313, source group top 3 0.444 -> 0.259; its readout half alone helps pairs (ir_ch R2 0.195).
Fix 3 changes nothing measurable (planted pairs stable in 13 % of cases for ir_ch / proto_ch_pe, 7 % proto_ch). Best combination on these
numbers: ir_ch (no sparsity) read at R1/R2, or ir_ch_f1 read at R2 (not run as such).

### CosMx (incomplete: most receiver types unread)

**Bug: small receiver types are dropped by the tiles.** `readable[r] &= variance[t] >= ...` is an AND over graphs (gene_ranking.context_flows
line ~158, interpretation.gene_flows line ~787), and `_type_variance` returns zeros for a type with <= 1 cell in a graph. Read on 19 tiles
(6 sections), any type absent from one tile loses every gene: **Alpha and Beta 0 targets read (alphabetical "top responders" in every arm),
Mast 2, Endocrine 22-23, Immune 29-33, Fibroblasts 40-53, Endothelial 55-76, vs Acinar 120 / 882 and Ductal 170 / 504** (driver reads /
responder genes). J6 (whole sections) was hit mildly (Mast 63, Endocrine 394 of 960). Every CosMx number below is therefore an Acinar / Ductal
number. Fix (not applied): pool the within-type variance over graphs (sums, squares, counts per type) and threshold once, or skip graphs where
the type has fewer than k cells; then re-read (readouts only, the checkpoints exist: ~2.1 h per ir arm, ~0.65 h per prototype).

What can be read now (Acinar / Ductal):
- Fit: proto_ch 0.137, proto_ch_pe 0.138 vs ir_ch 0.085-0.090. Not like for like: under channel masks the prototype sees the receiver's own ~80 %
  visible genes; ir_ch on CosMx has a type-only query and no own-cell skip. Hypothesis: the prototype's lead is within-cell gene-gene prediction.
- Q1 stability: responders RBO 0.82 (ir_ch) vs 0.73 (proto), drivers 0.70 vs 0.61: pass for all. The fixes lower it (0.82 -> 0.72).
- Q2 flagged share of the top 50: 0.11-0.13, the same as the twins (0.10-0.15): uninformative.
- Q3 known LR pairs: AUC 0.46-0.54 for every arm and twin (in rank_lr.csv; compare_read does not report it).
- Q4 agreement ir_ch vs prototype: RBO 0.46 (twins 0.32-0.35); proto_ch_pe vs proto_ch 0.83.
- Co-varying sender groups: only 3 among the 200 senders (endocrine hormones, acinar enzymes, MHC-I): type-marker sets, so fix 2 hardly acts here.
- Reader gaps: no LR table, no sender-type table, no PR plot (no truth, expected), steps.png and the driver rank-rank plot only.

Cost: ir arms ~7,600 s per readout, prototype ~2,300 s, proto_pe ~7,300 s.

## Fixes and re-run (2026-11-24; the user: "implement all the fixes, update the scripts, commit to main")

- **Library**: `interpretation._WithinVariance`, used by `gene_flows` and `gene_ranking.context_flows`: one within-type variance per gene, each graph's sum of
  squares about its own type mean pooled over graphs, thresholded once (a graph with < 2 cells of a type adds nothing). Test
  `test_a_type_missing_from_one_graph_keeps_its_targets` (a type with one cell in the second graph keeps its responders and pairs; the pooled variance by
  hand); the old AND, restored in either function, fails it (mutation checked). Whole sections change little: J6 lost Mast / Endocrine genes the same way.
- **Jobs**: `compare_job.sbatch STEP=evaluate` + `EVALUATE=1 compare_submit.sh` (readouts only, on the checkpoints in COMPARE_OUT); `compare_read.py`
  writes a **coverage** table (targets read per receiver group and run: the check that would have caught Alpha / Beta), the **LR** table (Q3), the
  **sender-type** table (proto, ir_ch, fixes, twins) and `ranking/compare_responders_rank.png` beside the driver rank-rank plot; checked on the downloaded
  tables of both jobs. `gene_rank_common.EVAL_MAX_TILES` (seeded subset of tiles); `gene_rank.py` vannan: 1 mm tiles, 40 drawn, 200 senders;
  `gene_rank_read.py` accepts `vannan`; `rank_job.sbatch` 11:55.
- Re-runs: `EVALUATE=1 bash slurm/compare_submit.sh cosmx` (12 tasks, ~2.1 h per ir arm, ~0.7 h per prototype; same folder, sections replaced),
  `RANK_RUNS=typeres bash slurm/rank_submit.sh cosmx` (J6's typeres, never read), `bash slurm/rank_submit.sh vannan`. synth needs no re-read (one receiver
  group per slide; unaffected).

## What helped, and how to expand it (2026-11-24; the user asked after the C3 read)

What the reads of 11-22 to 11-24 say helped, with the evidence:

1. **Reading beyond the region (fix 1)**, in three independent places: E7 permutations within label x region (readout alone: responders against hard
   negatives 0.305 -> 0.453), the model's label part on the region (0.476, the best value), the anchored task's region donors (CM3 as a responder
   0.62 -> 0.04; T2'' 0.80 -> 0.91). Same lesson each time: the region carries most of what passed for a response; once it is held fixed, the planted
   programmes stay and the markers go.
2. **Integrated gradients** as the effect measure: on the trained typeres the only method close to complete (gap 0.13; grad x input, DeepLIFT,
   conservative ~1); with the within-type permuted section as control it separates content from geometry (|contribution| by distance unchanged, signed
   push at contact 3.4 vs 0.35). On synth (2026-10-31) IG found mid-range senders 100 % where attention found 73 %. Caveat: one CosMx model so far; the
   read-out arms and the synth A1 rule are still to come.
3. **The distance prior / positional input**: the prototype with the window (proto_ch_pe) reaches ir_ch on pairs (0.147 vs 0.156). Not a fix, but
   the largest model-side gain in the comparison.
4. **Matched nulls**: the untrained twin and the within-type permuted section are what make every table readable (flagged share and LR AUC are
   uninformative exactly because twins score the same).

Did not help: sender sparsity (fix 2, model half: pairs 0.174 -> 0.132), the robust z / ensemble (fix 3), the cheap gradient methods for maps, LR
databases on CosMx (chance in every arm).

Proposed expansions (nothing built; in order of what they would tell):

- **E1. IG beyond the region.** The two things that helped, combined: IG with a reference of the sender's type **within its region / niche** (type x niche
  mean; sampled same type x niche donors as the check). The attribution then measures only what lies beyond the region, as fix 1 does for the
  permutations. Read on synth against truth (planted top-5 by class; pairs via programme targets) and on CosMx beside the plain type-mean IG.
- **E2. A scale ladder for fix 1.** Strata at several scales (niches from the k = 10, 30, 100, 300 nearest cells; or donors matched by neighbourhood
  composition at those scales): a responder's drop as a function of the stratum scale is the scale beyond which its signal no longer survives. On synth
  the planted length scale is the truth (Spearman, as T5''); on CosMx it gives each gene a local / global label from the model's own readout --
  the local-vs-global separation the thesis asks for, without a disc layout. The anchored task's rings are the same idea on the target side.
- **E3. Gene-resolved IG.** The sender's contribution split over its genes (`gene_resolved`) for the deviation targets: which sender genes carry the
  contact-distance push. The same gene in the receiver and its neighbours = spillover / smoothness (the E7 `self` term); a different gene = a candidate
  interaction. This answers the Acinar-spillover question of the typeres read and gives IG drivers / pairs to set beside the permutation rankings
  (agreement as a confidence measure).
- **E4. IG for the comparison.** proto_ch / proto_ch_pe / ir_ch on synth with IG (distance profile, planted top-5): does the redesign read the same senders
  at the same distances as the prototype + window, or only rank them better?
- **E5. Fix 1 on the model side at the readout's scale.** On CosMx the label part over section x type x niche (ir_ch_f1) did not change fit or stability;
  with E2's ladder pick the niche scale per dataset instead of one k-means niche.

Order: E1 and E3 are evaluation-only (existing checkpoints), E2 needs niche labels at several scales (cheap, model-free) and one readout per scale,
E4 needs the A1 synth job to run first, E5 retrains.
