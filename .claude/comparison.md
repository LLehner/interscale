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

## Expansions 1.x and 2.x (2026-11-24; the user: "implement and commit all of this, take notes, I need plots for all of it")

The user's list after the C3 read: 1.1 IG beyond the region, 1.2 a scale ladder for fix 1, 1.3 IG split by sender gene, 1.4 IG for the prototype
vs the global-only model, 1.5 the niche scale for the model's label part, 1.6 twin-subtracted maps; 2.1 cell vs gene masking for stability and a
combined loss, 2.2 1-16 heads for ir_ch and proto_ch, 2.3 a loss rewarding sender fields that correlate with receiver genes. 1.x in one run per
dataset, 2.x one run each, sequential on one GPU where possible (the user can use 14 GPUs at once), plots for everything.

### Library (fork main; tested)

| Piece | Where | What |
| --- | --- | --- |
| IG in expression space | `attribution.attribute(space="expression", gene_senders=)`, `reference_expression` | the path in the visible input entries: any model kind (the prototype's GCN included); `extra["gene_total"]` [Q, G] (each sender gene summed over cells), `extra["gene_near"]` [Q, K, G] (per near sender and gene); references read visible entries only; a `type|niche` label gives the type mean within the niche (1.1). Equals the token path for a linear embedding (tested to 1e-3) |
| attention rows | `evaluation/attention_rows.py` | each receiver's head-mean attention over the cells for read-out (recorded weights), dense and prototype models (`recording_attention`); `row_summary` (mass, entropy vs uniform, max share, effective keys, mean distance), `row_agreement` (Spearman over the union of top cells, JS divergence) |
| scale ladder | `gene_ranking.interaction_scales` | per target the scale where the drop within niches reaches half the plain drop (log interpolation; below / within / above the ladder) |
| twin-subtracted maps | `attribution.subtract_fields` | trained minus twin `out - in`, each scaled to its 99th percentile first (the trained map is orders larger) |
| combined loss | `optim.dual_mask` (weight, strategy node / channel, rate, channel_fraction); `TrainingPlan._dual_mask_loss` | a second pass per training step with the other kind of mask (private generator; the module's strategy switched for that pass only), same loss and centring, added to `train_loss` only (`val_loss` stays the primary mask's: comparable across arms); GlobalModel; tag `dm<n|c><w>` |
| field correlation | `optim.aux_loss_weights.field_corr`, `optim.field_corr` (temperature, center_key); `GlobalModule.sender_fields`, `StepOutput.fields`; `evaluation/field_correlation.py` | sender field `f_s(i) = sum_j a_ij x_js` (last layer's head-mean map, visible genes); per graph the Pearson r of every visible gene's field with every hidden gene (channel masks: disjoint sets, no self pairs), centred per graph or per `center_key` label; loss = minus the mean over targets of `sum_s softmax(r^2 / T) r^2`; only the attention carries gradient. Dense attention + channel masking (validated). Readout `field_pairs` (folds; the model's fields and a Gaussian-kernel bar), `pair_table`; tag `fc<w>` |
| caption | `flow_maps.plot_source_sink(what=, scale_label=)` | (earlier today) |

Tests: expression IG = token IG; the prototype read through its GCN, hidden entries never move; attention rows = the attribution `attention` method
(another code path), a context query never weighs its own cell; summaries / agreement / interaction scales / subtraction by hand; the dual mask's
second batch (other kind, shared channels per graph, the global generator untouched), a channel model trained with a cell-mask pass (logged,
tagged, strategy restored), validators; the field term by hand (r^2 = 1 for a linear sender, gradient only into sender fields), a dense
context-query model trained with it (the term's gradient reaches the last layer's attention), the readout's unread pairs and its kernel bar on a
planted neighbour dependence. Mutants caught: the strategy switch removed (dual mask), senders and targets swapped (field term). The bilinear part
now shares `_attend_values` with the fields (its tests pass unchanged).

### Jobs (`slurm/November/`, flat, not under git): `extra_job.py`, `extra_common.py`, `extra_read.py`, `expand_scale.py`, `make_extra_yaml.py`, `extra_job.sbatch`, `extra_submit.sh`

| Job | GPUs | Steps | Plots (RESULTS/plots/) |
| --- | --- | --- | --- |
| `expand_synth` | 1, in turn | evaluate the comparison's 7 synth runs + 2 twins: ladder (folds, all_state swaps within niches k10/k30/k100/k300 + region), IG to every planted receiver's response genes (planted genes hidden in every cell; graph / region / niche_k30 references), IG pairs from the gene totals, the rows' top cells, disc maps (6 discs of 300 µm, 3 slides); `expand_scale.py` -> k*; train + read `ir_ch_f1n` (label part on niche_k*, 2 seeds) | ladder_curves (ratio vs radius, lines coloured by planted length), ladder_scales (scale vs planted length, Spearman), ladder_scale_hist, ig_planted (top-5 by class x reference x model), ig_distance, ig_pairs_pr (IG vs permutation pairs), ig_agreement (models' IG rows), maps/<run>_graph<k>_{trained,twin,minus_twin}, niche_vs_region |
| `expand_cosmx` | 1, in turn | the same on 4 CosMx runs + 2 twins, on 6 drawn 1 mm tiles: ladder (composition niches), IG to the deviation over a hidden fold with type vs type|niche_k30 references (sender-type shares, distance), the top 3 deviating genes one by one with the 16 nearest senders split by gene (the fold hidden in the receivers only, so neighbours show the target's gene), disc maps (4 x 30 µm per tile); k*; `ir_ch_f1n` (section x type x niche_k*), 1 seed, the comparison's readout | ladder_*, ig_types (type, type|niche, difference), ig_distance, spillover (self share by distance bin and receiver type), maps, niche_vs_region |
| `maskstab` | 1, in turn | 9 runs (cell / chan / comb x seeds 42-44); per run the light rankings in two read modes (channel folds; 20 % hidden cells), the model reading each with that strategy; attention rows of planted + 100 random receivers per slide | stability (seed agreement: per-gene fit, responders, drivers, pairs, attention rows), tables of AUPRC / AUROC_hard |
| `heads` | array of 14 | ir_ch_h{1,2,3,4,8,12,16}, proto_ch_h{...} (head width 64 // heads); light rankings (folds), attention statistics, IG planted top-5 | heads (every metric against the head count, two lines) |
| `fieldcorr` | 1, in turn | ctx_ch with field_corr 0 / 0.1 / 1 / 1 centred within region, seeds 42-43; light rankings, attention, the field pairs (model) and the kernel bar (300 µm) | fieldcorr_bars, fieldcorr_pr (field pairs vs planted, kernel bar), fieldcorr_matrix (|r| drivers x responders, planted boxed) |

Light rankings = E7's responders / drivers / pairs with the senders limited to the planted drivers, the hard negatives and 40 other genes, one fold
partition (cost); metrics against synth's truth.

Rule for 1.5 (fixed before the run, `expand_scale.py`): k* = the ladder level whose radius is closest (log) to the median interaction scale over
the trained ir_ch runs' targets with a plain drop >= 0.01; on synth the truth-based best level (AUROC responders vs hard negatives per stratified read)
is printed beside it, not used.

Checks run (no model on real data): every arm through `apply_sweep_config` (names <= 224 characters); run lists and `configure()` through the
real `__main__` for all 7 jobs with stand-in yamls written by the yaml makers; every hook end to end on toy GlobalModels / a toy prototype with a
fake synth truth (planted edges, roles, niches), then the reader on their outputs (all sections; plots drawn, toy maps below the 100-cell
minimum by design); the ladder preparation on a 1,200-cell toy for both datasets; `expand_scale.py` on toy ladders. Found and fixed on the way:
the toy cell-masked models read a hidden gene set as whole hidden cells (the IG readouts now switch to gene masks explicitly); with the fold hidden
in every cell the target's own gene could never contribute (1.3 now hides it in the receivers only); `extra_job.py` overwrote its JOB with
compare_rank's; empty-table guards in the reader.

### Expectations (before any run)

- 1.1: the region / niche reference lowers the share of region-wide senders; on synth planted top-5 for short / mid unchanged or higher, the
  dense / sparse classes lower (they live at region scale). On CosMx the Acinar lead shrinks if it is niche composition.
- 1.2: on synth the interaction scale rises with the planted length (Spearman > 0.5 for ir_ch; lower for proto_ch); many long programmes "above".
- 1.3: synth IG pairs beat the permutation pairs at the Gaussian sources; CosMx contact push carries a large own-gene share for acinar genes in
  non-acinar receivers (spillover) if the Acinar signal is segmentation.
- 1.4: proto_ch_pe's IG rows close to ir_ch's (both read by distance), proto_ch's not.
- 2.1: channel arms more stable on gene rankings in folds, cell arm more stable on attention; comb in between or above both.
- 2.2: fewer heads -> lower entropy per head mean? Not necessarily: the head mean averages out; expected flat fit, the attention statistics
  telling. The prototype without positions likely near-uniform at any count.
- 2.3: field pairs above the kernel bar for weight 1; region centring removes the region markers' fields.

## CosMx re-read (2026-11-25; `EVALUATE=1 compare_submit.sh cosmx`, results MA/23_11_results/compare_cosmx, synced 2026-11-25)

The readability fix (pooled within-type variance) worked: every run reads all 960 targets in all 9 receiver groups. **The cluster's compare.md is
stale for ir_ch_f12_s44**: it shows the old dropout (Alpha / Beta 0, Mast 2) although the synced arrays of s44 are complete (identical coverage to
s43) -- the report ran before s44's re-read landed (likely a GPU-check resubmit outliving the report's afterany). Rebuilt locally from the synced
tables: `compare_cosmx/compare_rebuilt_2026-11-25.md` (numbers below). Rerun the report on the cluster before quoting the folder's own compare.md.

| | responders seed RBO | drivers seed RBO | flagged top 50 |
| --- | --- | --- | --- |
| S0 ir_ch R0 | 0.637 | 0.647 | 0.207 |
| S1 ir_ch_f1 R1 | 0.612 | 0.641 | 0.209 |
| S2 ir_ch_f12 R2 | 0.522 (stale report 0.433) | 0.650 | 0.207 |
| S3 ir_ch_f12 R3 | 0.522 | 0.614 | 0.207 |
| readout only ir_ch R3 | 0.591 | 0.752 | 0.207 |
| proto_ch R0 / R3 | 0.465 / 0.489 | 0.573 / 0.482 | 0.189 |
| proto_ch_pe R0 / R3 | 0.465 / 0.475 | 0.507 / 0.468 | 0.218 / 0.227 |

- **Fit** (centred cosine within the receiver group, all types now): proto_ch 0.111, proto_ch_pe 0.111, ir_ch 0.065, ir_ch_f1 0.063, ir_ch_f12 0.061;
  twins ~0. The prototype still fits CosMx better (the first read's 0.137 vs 0.09 were Acinar / Ductal only).
- **Stability:** the global-only model's responder rankings are the most seed-stable (0.64); fix 1 changes nothing on CosMx, fix 2 (sender sparsity)
  costs 0.09; the readout's R3 (ensemble + robust z) raises driver stability (0.65 -> 0.75), not responders. The prototype is less stable (0.47-0.49).
- **The two model families find different responders:** RBO with the prototype's ranking 0.14-0.18 for every ir_ch variant (twins 0.03-0.05;
  proto_ch_pe vs proto_ch 0.66). Fitting better (prototype) and ranking stably (global-only) do not point to the same genes.
- **LR support null for every arm** (AUC 0.48-0.51; the untrained ir_ch 0.545, p 0.18, highest).
- **Sender -> receiver effects** are within-type first (Mast -> Mast 0.025 on ir_ch_f12, Acinar -> Acinar 0.013-0.016, Ductal -> Ductal 0.006-0.009),
  then Acinar -> Endothelial (0.010-0.012); the prototype's effects are 5-10x smaller; twins ~0.
- Cost: ~7,500 s per ir arm (plain + region), ~1,500-1,850 s per prototype on 1 mm tiles.

## Local / global split of the gene rankings (2026-11-25; built, not run)

The user asked whether the orthogonality losses make sense for the global-only model (mostly not: one embedding; see notes.md) and for a split
local / global ranking with the global-only model, "similar as interscale does with the dual-model", in four variants, each a plot to judge, plus
Moran's I (local on the neighbourhood graph, every cell a centre, vs across the whole slide) as the model-free comparison and colour.

**Library (fork):**
- `attribution.decomposition_by_distance(module, batch, receivers, radii)`: the interaction read-out's exact context part of every gene at once,
  split by the senders' distance from the receiver (`within` [Q, K, G], `total` [Q, G]); equals the one-hot-per-gene `decomposition` (tested to 1e-4,
  per radius by hand; mutants with the wrong origin or radius caught; leaving the receiver in is equivalent: the read-out never attends to it).
- `gene_ranking.distance_flows(module, graphs, labels, genes, radii)` / `merge_distance_flows`: per receiver and radius, every other cell farther than
  r swaps its visible entries within its label, only that receiver's row is read (one perturbed copy per receiver and radius, `batch_size` copies per
  pass, identical to one at a time, tested); `near(k) = drop(0) - drop(k)`, `far(k) = drop(k)`. Toy: the 150 µm responder is all near (far exactly 0),
  the 400 µm partner all far. **Noise:** near is the difference of two independent permutations; where both remove the whole signal it scatters
  (toy, 30 receivers: +-0.4 on a drop of 1.0), so read it over many receivers.

**The variants** (cluster `split_common.py`, `split_read.py`; jobs `split_synth`, `split_cosmx` in `extra_job.py`, evaluation only, on the
comparison's checkpoints; synth: ir_ch s42/s43, proto_ch s42/s43, proto_ch_pe s42 + twins; CosMx: ir_ch s42/s43, proto_ch s42, proto_ch_pe s42 + twins,
8 tiles of 1 mm; one partition of gene folds at 0.2, every gene hidden in every cell once):

| | what | models | plot |
| --- | --- | --- | --- |
| V1 decomposition | spread over receivers (centred within graph x label) of the context part from within / beyond r | read-out (ir_ch) | `v1_decomposition.png` |
| V2 permutation | responder drop carried within / beyond r (`distance_flows`); for the prototype also each head's responder drop (GCN vs transformer: interscale's own split, the reference) | all | `v2_permutation.png`, `v2_dual_heads.png` |
| V3 radius profile | V1 and V2 at r = 0, the local reach and RADII_UM (synth 100-1500 µm, CosMx 15-250 µm): share of the effect within r against r, r50 per gene | all | `v3_profiles.png`, `v3_r50.png` |
| V4 ladder | `extra_common.ladder` (swaps within niches of k = 10 ... 300 nearest cells; synth + region): the scale where the drop is half gone | all | `v4_ladder.png` |
| Moran | `gene_scales.scale_sets` on the whole test slides: I_local (the dataset's own graph, 1-2 hops, every cell a centre) vs I_global (inverse distance across the slide); CosMx also on the (section, type) residual. locality = (I_local - I_global) / I_local for genes with I_local >= 0.05 (else not spatially variable, grey) | -- | `moran.png`, the colour of every plot |
| rankings | the global-only model's near / far rankings against the prototype's (by distance, and by its two heads) | ir_ch, proto_ch | `rankrank.png` |

r (the local reach) is Moran's 2-hop neighbourhood in µm (median), so "local" means the same for the model readouts and for Moran. `split.md`: per
run and variant the median near share, its Spearman with Moran locality (> 0: the model calls Moran-local genes local), r50, on synth the median near
share per planted class; the prototype heads' table; rank Spearman; the top 15 near / far responders per run.

**Expectations, fixed before the run:**
- synth: short / mid responders near (share > 0.5 at the reach), dense / sparse / directional far; the twins near by geometry only; V1 and V2
  agree on ir_ch (same model, two readouts); the prototype's GCN head carries the short responders, its transformer head the rest.
- CosMx: most of every gene's effect within the reach (A4: 47-56 % of |IG| within 60 µm, ~90 % within 500 µm); far rankings thin and FOV-driven
  (ceiling audit); Moran locality mostly positive on the raw expression (type markers) and closer to 0 on the residual.
- Moran vs model: Spearman > 0 but small on CosMx (Moran is spatial variability, not interaction; the user knows and wants the axis anyway).

**Checks run:** library tests (39 in `test_gene_ranking.py` + `test_attribution_methods.py`), the jobs' real `__main__` with `sweep_common.main`
stubbed (modes, runs, twins, results folder), the hook end to end on the toy read-out model and the toy prototype (every table written), every plot
drawn from the toy output in both modes (CosMx: Moran colour; synth: planted classes) and looked at. Not run on real data.

**Run:** after syncing the fork's main: `bash slurm/extra_submit.sh split_synth eval` and `... split_cosmx eval` (one GPU each; estimates ~1-2 h synth,
~2-4 h CosMx; every step prints `[time]`). Results MA/25_11_results/split_<data>/ (split.md, plots/). Files (flat, slurm/November): split_common.py,
split_read.py, extra_job.py, extra_job.sbatch, extra_submit.sh, plus what extra_job imports.
