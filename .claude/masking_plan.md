# Masking: what the model is asked to fill in, and how we score it

Written 2026-10-29. This is the design file roadmap T3 asked for ("the design goes into `masking_plan.md` before any code").
It holds four of the points the user set for the days up to Tuesday 2026-11-03 (roadmap, "Focus until Tuesday 3 November"):

- point 2: masked vs unmasked prediction. The user's choice (2026-10-29): one trained model, scored on the cells it had to
  fill in against the cells whose expression it was given (M0);
- point 3: randomized vs pre-selected gene sets under programme masking (M1);
- point 4, first half: gene sets picked by Moran's I at the end of the GCN's reach, as a label-free programme (M2). The second
  half, using the sets to push the embeddings apart, is decoder_plan.md D2.4;
- point 1, one of the four niche roles: the niche as a masking unit (M3; the lung data, lung_plan.md).

The rules are the other plans': a stage is `done` only when a run has been read; "implemented, unverified" is a real state;
every training mask is an opt-in config switch with a checkpoint tag and a leakage test through the real model; nothing runs
locally. Results go into this file, the verdicts into notes.md (append-only).

## Status

| stage | question | state | verified by | date |
|---|---|---|---|---|
| M0 — hidden vs visible scores | how much of a reconstruction score is the cell's own input | **implemented, unverified on data** (2026-10-29, branch `masking-sets`): `masked_probing.visibility_scores`, `visible_view`, `HeadPredictions.near_hidden_um`, the `cells+blocks` evaluation mask, the slide mean leave-one-out at visible cells; evaluation-only sweep `slurm/new/visibility.py` (synth_spot and legnini cells task, CosMx fold 0), not run | 11 tests, 7 mutations caught (one after a further test), harness IDENTICAL; the hook on toy models | 2026-10-29 |
| M1 — randomized vs pre-selected gene sets | does a fixed programme matter, or does any rotating set teach communication | **implemented, unverified on data** (2026-10-29, branch `masking-sets`): `dataset.mask_channel_genes` (the pool channels come from), `dataset.mask_gene_sets` + `mask_sets_per_draw` (whole sets drawn from a pool); sweep `slurm/new/masking_sets.py` (7 arms, 13 runs, 300 epochs, own-task scores against `ref_own`), not run | 22 tests, 14 mutations caught (one after a further test), harness IDENTICAL; list mode on a 8-slide stub (every arm validates, names distinct, references fit) | 2026-10-29 |
| M2 — Moran's I gene sets | a local and a global gene set picked without labels, at the GCN's reach | **implemented; checked on synth_spot without a model** (2026-10-29, notes.md): `gene_scales.scale_sets` (local Moran's I on the graph's own hops, global with inverse-distance weights across the slide, the **relative** fall as the rule), `write_scale_sets`, `plot_scale_sets`; tool `slurm/new/scale_sets.py`; the sets feed M1's `moran_*` arms | 18 tests, 14 mutations caught (one after a further test); synth_spot training slides against the planted scales | 2026-10-29 |
| M3 — niches as a masking unit | can a hidden niche be filled in from the other niches | **implemented, unverified on data** (2026-10-30, branch `niches`): `dataset.mask_block_kind: label`, `mask_block_label_key`, `mask_block_max_cells`; `tl.masking.sample_unit_mask`; `masked_probing.draw_masks(label=, max_cells=)` (below); no run, the bar `ref_label+context` not built | tests/test_niches.py (whole patches, the disc draw unchanged draw for draw, redrawn for train and fixed for val, the evaluation draw, config, names), harness IDENTICAL | 2026-10-30 |

## What exists (do not rebuild)

| mask | key | state |
|---|---|---|
| random cells | `mask_strategy: node`, `mask_percentage` | exists; the loss reads hidden cells only (`_process_batch_for_metrics`: a visible cell's row is all-False in `entry_mask`) |
| random entries | `mask_strategy: gene` | exists |
| discs | `mask_block_radius_um`, `mask_block_fraction` | exists |
| programme (pre-selected, fixed) | `mask_genes` (names or `var` selectors such as `role=response`) | exists; the only task under which the global model learned communication |
| channel masking (randomized) | `mask_channel_fraction` | exists; read 2026-10-23 at 100 epochs: 0.245 on the programme against 0.764 for the programme-trained model, below `ref_own`; training loss 0.76-0.83 |

Facts the stages rest on (notes.md):
- Training length is a confound. 300 epochs moved the windowed model more than any arm did (programme mean 0.730 → 0.800). Every
  negative verdict at 100 epochs, channel masking included, is to be rerun at 300 before it is believed.
- The SOTA model on synth_spot is the dense windowed global-only model (`lin_dist_skipraw`; hierarchical Stage A `full`: 0.947 ±
  0.002 at 300 epochs). `hier_none` is within 0.01 (0.939) and is the one that runs on whole sections.
- A random gene set rarely hides a whole programme, so a hidden response can be predicted from its visible co-responders in the
  same cell. No communication is needed for that; `ref_own` is the bar for it.

## M0 — hidden vs visible scores (point 2)

**Question.** A model trained with cell masking is asked about two kinds of cells on a masked pass: hidden ones (all genes at
`MASK_VALUE`) and visible ones (their expression is the input). The loss only ever read the hidden ones. How good is the
prediction on each? The manuscript's evaluations read clean passes, where every cell is visible. So the visible-cell score is
the kind of number the manuscript reported, and the gap to the hidden-cell score says how much of it is copying.

**What is scored.** One masked pass per test slide, corrupted as in training (same strategy, rate and discs, seeded), so hidden
and visible cells come from the same pass and the same geometry. Per head (the parts too, for decomposable models):
- hidden cells: centred cosine and SCE per cell, by disc depth (exists: `score_heads`);
- visible cells: the same metrics on their rows, by distance to the nearest hidden cell (0-1 hop, 2 hops, beyond);
- per gene: centred cosine over hidden cells and over visible cells, plus the scatter of one against the other.

**Bars.** Different for the two groups, and the reading depends on them:
- hidden cells: the slide mean, `ref_nearest_<k>` and the Gaussian-kernel smoothers (exist);
- visible cells: the identity (the cell's own input, cosine 1 by construction: the ceiling a copying model reaches) and the same
  smoothers **leaving the cell out** (its context alone: the score of a model that ignores its own input). Check when building
  whether `reference_predictions` already leaves the query cell out for a visible query. For a hidden cell the question never
  arose, because its entries are not visible.

**How to read it.**
- Visible score near 1: the output map is close to the identity where the input is given, although no loss asked for that. A
  clean-pass evaluation then measures copying.
- Visible score near the leave-one-out smoother: the model ignores its own input and fills every cell in from context. Its
  clean-pass numbers would then be fill-in numbers.
- Visible score below the smoother: the outputs at visible cells sit where no loss ever put them. That is expected for paths
  that only ever saw `MASK_VALUE` at a hidden cell: the global-only raw own-cell skip `W_o` was trained on a constant input
  under cell masking. Clean-pass numbers of such a model are meaningless.
- The gap per gene: genes with a high visible and a low hidden score are the ones a clean pass flatters most.

**Scope.** Cell masking (random cells and discs) only. Under programme masking no cell is visible as a whole, and the visible
genes' decoder rows get no gradient, so their outputs are noise by construction. List that in the report instead of scoring it.

**Models, evaluation only (no training).** Existing cell-masking checkpoints: the `cells` task of `local_global_split`
(synth_spot and legnini, the dual arms), CosMx fold 0 (`gcn`, `hier_*`) and the CosMx `long` run once read; on the lung data
every arm of lung_plan.md. Add the untrained twin of one arm: its visible-cell score is the floor of an untrained map.

**Build.**
- `masked_probing`: `score_heads(..., rows="hidden" | "visible" | "both")` (the rows from `HeadPredictions.cell_masked`; default
  `hidden`, today's numbers) and a table `visibility_scores` (head x group x depth band). Tests: on a toy module whose output is
  its input, the visible score is 1 and the hidden score is the fill's; on one that outputs a fixed smoother, visible =
  leave-one-out smoother; no RNG drawn; the default output unchanged (harness IDENTICAL).
- `sweep_common`: an `EXTRA_EVALUATION` entry; the per-gene table as CSV.
- Cost: one pass per test slide plus the references; seconds on synth_spot. On CosMx and the lung data, the references need
  neighbour lists, not the dense `cdist` (the CosMx critical path needs that anyway).

**Plots.** Per arm: hidden vs visible cosine per gene (scatter, the diagonal and the identity line marked), and bars of the two
groups with their bars. On synth_spot, one test slide with true / hidden-cell / visible-cell prediction maps for three genes
(memory: show spatial maps).

**Built 2026-10-29 (implemented, unverified on data).**
- *Library* (fork, branch `masking-sets`): `masked_probing.visible_view` (the same pass with every entry of the visible cells as
  the scored set), `visibility_scores` (tables `groups`, `gap`, `per_gene`, `hidden_by_depth`, `visible_by_distance`), the field
  `HeadPredictions.near_hidden_um` (each visible cell's distance to the nearest hidden cell, filled by `collect_head_predictions`),
  and the evaluation mask `draw_masks(kind="cells+blocks", rate, radius, block_rate)`: cell masking with discs as a config trains it.
- *Found while building:* `reference_predictions`' slide mean included the cell itself (weight 1 / n). At a hidden cell that never
  mattered (its entries are not in the sum); at a visible cell it leaks the answer (on a 64-cell toy slide 0.17 against 0.05 left
  out). It now leaves the cell's own visible entries out; every hidden-cell number is unchanged (tested).
- *Sweep* `slurm/new/visibility.py` + `visibility_common.py` + `visibility.sbatch`: evaluation only, one job, four sources in turn
  (`VIS_SOURCE`): the local_global_split cells task on synth_spot and legnini (6 dual arms, 12 runs + a twin each, 30% of the cells)
  and CosMx fold 0 (fast part: gcn, hier_soft / fac / fpc; long part if it ran), each loaded with its own sweep's config and arms so
  the checkpoint names match; missing checkpoints are skipped. CosMx references: k nearest visible cells by a k-d tree, the cell
  left out, in blocks of queries. Figures per run: `scores.png`, `per_gene.png`, `maps.png` (the genes a clean pass flatters most).
- *For future runs (2026-10-29, the user's steer):* the evaluation is also a setting of the shared sweep machinery, `VISIBILITY = {}` in
  `sweep_common.py` (`visibility_tables`): any sweep that trains with cell masking gets the visible-cell tables and figures beside its
  normal evaluation, for every run and twin, with no extra job; a run trained with gene masking is skipped with a message. The CosMx
  and lung sweeps set it. `visibility.py` stays for re-reading old checkpoints, but they are weak sources (100-epoch dual models,
  undertrained CosMx), so read its numbers as a trial of the instrument, not as a result.
- *Not covered:* models trained with programme masking (no visible cell) and the global-only raw-skip arms of the cells task (the
  cells task never trained them).

## M1 — randomized vs pre-selected gene sets (point 3)

**Question.** Programme masking (a fixed, pre-selected gene set hidden in every cell) is the only task that taught the global
model communication. Channel masking (a random set redrawn every epoch) did not transfer. Is that because the set must be
fixed, because a random set over all genes is mostly noise genes (170 of synth_spot's 253), because co-responders leak, or
because 100 epochs was too short? The arms separate these explanations.

**Arms** (the SOTA model, `lin_dist_skipraw`, dense, synth_spot, programme evaluation, 300 epochs):

| arm | what is hidden each epoch | separates | new code | seeds |
|---|---|---|---|---|
| `fixed` | the planted programme (`role=receiver`, `role=response`) | the ceiling | none: hierarchical Stage A's `full` (3 seeds, 300 epochs) is this arm, reuse it | – |
| `ch30_all` | 30% of all genes, random | 100 epochs vs 300 (the old verdict) | none | 3 |
| `ch30_svg` | 30% of the spatially variable genes only (`is_spatial`; on real data the M2 union) | noise genes diluting the task | `dataset.mask_channel_genes`: the pool the channels are drawn from (names or selectors; default all) | 3 |
| `pool_k1` | one whole programme drawn from a pre-selected pool (synth_spot: the 24 planted programmes), per graph and epoch | a rotating set of *whole* programmes against a fixed set | `dataset.mask_gene_sets` (a list of sets, each a selector list) and `mask_sets_per_draw` (k) | 3 |
| `pool_k4` | four whole programmes drawn per graph and epoch | how many at once | as above | 1 |
| `sub50` | a random half of the fixed programme's genes | partial programmes: do co-responders let the model shortcut | `mask_channel_genes` = the programme, `mask_channel_fraction` 0.5 | 1 |
| `moran_global` / `moran_local` | the M2 sets, fixed | a pre-selected set chosen without labels | none beyond M2 (the sets become `var` columns) | 3 / 1 |

About 15 runs plus twins of `ch30_svg` and `pool_k1`: ~15 min each at 300 epochs on synth_spot, one job per run.

**Evaluation of every arm.**
- The programme evaluation (`EVAL_MASK_GENES` = the planted programme): whole-cell centred cosine, per-programme wins / ties /
  losses against the bar, short range / rings / M11 separately. This is the transfer test.
- Each arm's own task on test-time draws of its own mask, against `ref_own` (the diagnosis the 2026-10-23 read asked for: does
  channel masking fit its own task at all).
- Sender rank by integrated gradients against the untrained twin (`response_influence(measure="attribution")`).

**Decides.** Whether a label-free set (`ch30_svg`, `moran_*`) comes close enough to `fixed` to be the real-data task. If `pool_k1`
matches `fixed` but `ch30_*` does not, the unit has to be a whole co-regulated set (the module-masking idea of the week-1 review,
item 3), and M2's sets or co-expression modules are how to get such units without labels.

**Build.** `tl/masking.py`: the pool for channels and the drawn whole sets, both through the existing `sample_gene_set_mask` path
(the redraw is `NodeMaskResampleCallback`'s, as for channels). Tests as for channel masking: exact counts, every set equally
likely, one draw per graph shared by its cells, val/test fixed, refusals (`mask_genes` together with a pool, `Precomputed`), no
hidden gene reaching a prediction through the real model, a real `train()` whose masks change by epoch. Checkpoint tags
`chp<fraction>-<pool>` and `sets<k>-<n>`. Defaults off: no RNG drawn.

**Built 2026-10-29 (implemented, unverified on data).**
- *Library:* `dataset.mask_channel_genes` (selectors; `round(fraction * pool)` of the pool hidden at each draw, every other gene
  visible; without a pool the draw is the old one, random number for random number) and `dataset.mask_gene_sets` (a list of sets of
  selectors) with `dataset.mask_sets_per_draw` (`tl.masking.sample_set_columns`, `sample_sets_mask`; the union of `k` distinct sets
  per graph and draw). Both through `gene_set_mask_kwargs` into the datamodule, redrawn by `NodeMaskResampleCallback` as channels are;
  `hides_gene_set` is true for both. Validator: a pool needs a fraction, sets exclude `mask_genes` and channels, each set non-empty,
  `k` in range, a flat list of strings refused (it would be split into characters), gene masking, a warning below 100% of the cells.
  Tags `ch<f>c<rate>-p<selectors>` and `sets<k>of<n>c<rate>-<sha1 of the sets>`.
- *Arms as built* (`slurm/new/masking_sets_sweep.yaml`, every one the dense windowed model `full` at 300 epochs): `fixed`, `ch30_all`,
  `ch30_svg`, `sub50`, `pool_k1` (the 24 planted programmes' response sets; their union is `fixed`), `moran_global`, `moran_local`
  (20 genes each). `pool_k4` dropped to keep the arms few. Seeds 42 and 43 (`fixed` 42 only), 13 runs, the twin of `fixed`.
- *Own task* (`masking_sets.own_task`): 3 test-time draws of an arm's own mask (1 for a fixed set), every head and `ref_own` (a
  `RidgeCV` from the visible to the hidden genes on the training slides, per draw) on the hidden entries.
- *Data:* the sweep's own preprocessed file (`synth_spot_sets_pp.h5ad`), into whose var `PREPARE_DATA` writes the M2 sets once (locked,
  atomic, skipped when the recorded settings match); the sbatch prepares under `flock` before each task's run.

## M2 — Moran's I gene sets: a local and a global programme without labels (point 4)

**Idea (the user's, 2026-10-29).** Split each gene's Moran's I curve at the end of the GCN's reach (e.g. 2 hops). A gene with high
autocorrelation inside the reach varies locally; a gene still autocorrelated at the largest range varies on a tissue scale. Take
the top n of each as a local and a global gene set. They are programmes chosen in an unbiased way (no labels, no biology), usable
on any dataset, the lung data and CosMx included.

**The selection, per dataset (model-free, training slides only).**
1. *Curves.* `evaluation.gene_scales` as built (shells, sampled centres, a permutation null, pooled over the training slides),
   k from 4 to `k_max` (256 by default; 512 on whole sections).
2. *The threshold* `k*` is the GCN's reach in neighbours, measured on the dataset's own graph: the median number of distinct cells
   within `hops` hops (2 by default) over the training cells, not a guess. Its µm value (the median distance of the `k*`-th
   neighbour) goes next to it. Synth_spot's 2-hop reach is ~290 µm (`depth_um`); CosMx's k = 8 graph gives a few tens of cells.
3. *Two numbers per gene:*
   - `I_local` = the maximum of the shell curve over k ≤ `k*` (the user's "max Moran's I before");
   - `I_global` = the shell value at `k_max` (the user's "value at max range"; the mean over the last two shells as a less noisy
     variant). Both read against the permutation null (Bonferroni over the shells, as built); a gene with neither significant goes
     into no set.
4. *The sets:* the top n by `I_local` and the top n by `I_global` (n a setting: 10, 20, 50). A smooth gene ranks high in both lists.
   The default `overlap: global` puts it in the global set only (it stays autocorrelated past the reach, so a model that reads
   only the neighbourhood cannot be what predicts it there). `drop` and `keep` are the alternatives. A second ranking,
   `contrast`, takes the local set by `I_local - I_global` (variable nearby, flat far away), which is closer to "a likely local
   interaction" in the roadmap's task 1.
5. *Composition.* Where cell types exist, the same curves on the expression minus its type mean (exists as "within types" for
   the comparison's Moran's I). A marker is spatially variable because its type is. The within-type sets are the stricter
   programmes; report how much the two versions overlap.
6. *Output:* `var` columns `scale_I_local`, `scale_I_global`, `scale_set` ∈ {local, global, none} (and `_within` variants). The
   existing programme masking then takes `mask_genes: ["scale_set=global"]` through its selectors, with no new masking code.

**Checks before any training.**
- On synth_spot (planted truth): precision and recall of the local set against the short-range receivers (R1-R4, S-class
  responses) and of the global set against the long-range responses and region genes, for each n, overlap rule and ranking. This
  says whether the rule picks what it should, before it is used on data without truth. The 2026-10-23 model-free read is the
  starting point: rings read at their width, senders showed no scale, sparse responses were barely autocorrelated (expect the
  sparse programmes to be missed).
- On legnini, CosMx and the lung data: the two sets with their curves (Plot A style), maps of the top genes of each set (memory:
  show spatial maps), and the overlap with the within-type sets.

**Uses.**
- As programmes (M1's `moran_*` arms; then the lung data and CosMx): hide the global set in every cell and read the prediction
  against `ref_own` and `ref_own+context`; the same for the local set. The global set should need context beyond the reach. If
  the cell's own visible genes predict it (`ref_own` high), the set is a co-expression module, not a communication readout: say
  so per gene.
- As per-head targets to push the two embeddings apart: decoder_plan.md, D2.4.

**Circularity, kept in view.** Once the sets are training targets or per-head losses, the Moran's I evaluation of the same genes
(task 1's Plot A, the Spearman of scale vs global share) no longer tests anything. Read such models on held-out genes (D2.4) or
on quantities the selection did not use.

**Build.** `evaluation.gene_scales.select_scale_sets(curves, reach_k, n, overlap, ranking)` and `reach_in_neighbours(graph, hops)`;
a small script that writes the `var` columns into a dataset's h5ad from its training slides. Tests: on constructed fields (a field
smooth at 2 cells, one smooth at 50, white noise) the sets are what they must be; the threshold equals a hand count on a lattice;
the test slides are never read.

**Built 2026-10-29, with two changes to the design above (the user's, 2026-10-29: "for the global Moran's I simply calculate it
across the slide, then rank the genes by the absolute difference of local_I - global I"), both checked on synth_spot first.**
- *The global statistic.* "Across the slide" read literally -- every other cell of the slide a neighbour, weighted alike -- is
  degenerate: Moran's I of a z-scored gene is then exactly `-1 / (n - 1)` for every gene. Built instead: inverse-distance weights over
  every other cell of the slide, `1 / max(d, d0)` with `d0` the slide's median nearest-neighbour distance, row-standardised
  (`gene_scales.global_moran`). In two dimensions every distance band then carries about the same weight, so it is the correlogram
  averaged over the slide's own range: no bandwidth, nothing tuned to a dataset. Sampled centres (2,000 per slide) against every cell,
  a ratio estimator as `moran_by_rank`. Alternatives tried (shell or cumulative at k = 512, Gaussian kernels at a quarter or an eighth of
  the slide, at the geometric mean of reach and slide) rank the planted scales about as well under the relative rule (0.69-0.83).
- *The rule.* The absolute difference fails: a weakly patterned gene has a small difference whatever its scale. On synth_spot's 24
  training slides the four short-range receivers (planted 100 µm, local Moran's I ~0.2) made up 4 of the 10 "global" genes, and the
  Spearman of the ranking with the planted scale swung between +0.12 and -0.55 with the global statistic and the prefilter. The
  relative fall `(I_local - I_global) / I_local` divides the size out: -0.75 to -0.83 in every combination. `rule: relative` is the
  default; `rule: absolute` is kept.
- *The local statistic* is Moran's I with the graph's own hop neighbourhoods (`hop_neighbourhoods`: the k-NN graph made symmetric,
  as a GCN reads it), every cell a centre, the largest over 1 .. `hops` (`local_form: max`; `reach` uses all hops at once). On
  synth_spot's k-NN 8 graph 2 hops are 24 cells / 283 µm, measured. Candidates: above a permutation null by 3 SDs Bonferroni over the
  genes, and `I_local >= 0.05`. With fewer than `2 n` candidates each set takes half. The overlap rule of the design is gone: one
  ranking, two ends.
- *synth_spot (24 training slides, n = 10):* global = the three region markers, SP1, the sources M1 / M3 / M5, the responses MC3 /
  MC5 / MC23 (median planted scale 1,200 µm); local = R1-R4, SP8-SP10, MC6, MC14, M9 (180 µm). The sparse responses MC16-MC22 are no
  candidates (local Moran's I 0.00-0.03), as in the 2026-10-23 read. 10 s on 60k spots.
- *Tool* `slurm/new/scale_sets.py`: any dataset through its training config (data, layer, graph, split, GCN layers), `--residual-of`
  for the within-type sets, `--write` for the var columns, tables and figures (`*_scatter.png`, `*_maps.png`); `prepare()` for a sweep.

## M3 — niches as a masking unit (point 1; the lung data)

**Question.** Hide every cell of one niche region and ask the model to fill it in from the surrounding niches. Which other niches
does the prediction draw from? This is between-niche communication, read with a mask shaped by the tissue instead of a disc.

**The mask.** Per training graph and epoch: one or more connected components of a niche label (components on the spatial graph,
so two patches of the same niche are separate units) are drawn until `mask_block_fraction` of the cells is covered; those cells
are hidden whole (cell masking). Plus random cells, as with discs. Config: `dataset.mask_block_kind: disc | label` (default `disc`,
today's behaviour) with `mask_block_label_key`. The draw is the discs' draw with components in place of discs.

**Leakage.** The shape of the hole tells the model which cells belong to one niche, and the label was computed from every cell's
expression. Bars: `ref_label+context` (the niche's training mean plus the context smoothers), next to the usual ones. The read is the
gain over that bar, not the raw score.

**Read.** Hidden-cell scores by niche and by depth into the hole; the integrated-gradients attribution of a hidden niche's
prediction summed by the sender's niche (a niche x niche table, against the expression-permuted slide). Disc-masked arms of the
same model are the comparison.

**Build.** `tl/masking.py` (`label_blocks`, sharing the discs' code), the validator (needs `mask_block_label_key` in `obs`), tests
as for discs (coverage, components never split, no RNG drawn when off, no hidden value reaching a prediction).

**Built 2026-10-30 (branch `niches`).** `dataset.mask_block_kind: label` with `mask_block_label_key` (the column must be in
`dataset.extra_obs_keys`) and `mask_block_fraction`; no radius (refused). The units are the patches of `tl/niches.py` (lung_plan.md
L2), drawn by the discs' own loop (`_draw_blocks`, shared; the disc draw is unchanged random number for random number):
`sample_unit_mask`. `mask_block_max_cells` > 0 cuts a patch into kdtree pieces of at most that many cells -- added because one large
niche (one patch covering half a slide) would otherwise be hidden whole by one draw. Everything after the draw is the discs' path
(cell masking, a gene set's cells, the regime loss: `deep` is accepted with niche blocks, since their depth is the data's). The
evaluation draws them too: `masked_probing.draw_masks(kind="blocks" | "cells+blocks", label=<attribute>, max_cells=)`; the sweep's
`sweep_common.py` still draws discs, so an M3 arm is read on discs until it passes `label`. Checkpoint tag
`blklab-<key>[-mx<max>]f<fraction>`. Not built: the bar `ref_label+context`, the read by niche and by depth into the hole.

## Order (until Tuesday 2026-11-03)

1. M0: evaluation only, on existing checkpoints. Build Friday, run as soon as it is built; it needs no new training.
2. M2's selection and its synth_spot check: model-free, Friday. It feeds M1's `moran_*` arms and lung_plan.md.
3. M1: build the pool keys Friday, run over the weekend (300 epochs), read Monday.
4. M3: build once the lung data and its niche key are known; run with lung_plan.md's arms.

## Open questions

- M2: n (10 / 20 / 50) and the overlap rule. Proposed: report all three n with `overlap: global`, and pick by the synth_spot check.
- M1: the pool on real data (no planted programmes): M2's sets, co-expression modules from the training slides, or both.
- M3: whether the niche labels the lung data comes with are allowed as a mask unit, or should be recomputed from training slides
  only (lung_plan.md, the leakage section).
