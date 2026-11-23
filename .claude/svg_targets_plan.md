# Anchored SVG targets: diffsvg picks the programmes InterScale must explain (plan, 2026-11-22)

## Goal

The user (2026-11-22): combine InterScale with diffsvg by letting diffsvg define the reconstruction targets. Define several
anchors (patches or discs; they need not cover all cells), stack the slides, test every gene for a non-flat expression curve
along the distance to each anchor (diffsvg's test without a condition comparison: the SVGs with respect to that anchor), cluster
the curves, keep the clusters with non-flat shapes (the significant genes and the genes grouped with them), and train InterScale
to predict these genes from the other genes, possibly step-wise in rings of increasing radius around the anchor (e.g. three
rings), so the model learns which genes predict the spatially variable ones. The genes most useful for that are ranked as sender
genes, with sender-receiver pairs, split by local and global. The anchors were meant to be "the same across slides" (a rough
match in position is enough). The selection and combination of several anchors was left open ("you will come up with a
solution"); the question was how this can detect cell-cell communication (CCC) and how to implement it.

This file: what the combination can and cannot show about CCC, a design that answers the open parts, a staged build with
verification, decision rules to fix before the runs, and the open decisions. Nothing is built or run.

## Status

| Stage | What | Where | Status |
| --- | --- | --- | --- |
| D0 | Flatness test on condition-free fits (`parallel_curves_test` on `beta`) | diffsvg, branch | proposed |
| D1 | Slide (replicate) model for condition-free fits | diffsvg, branch | proposed |
| D2 | Anchors, µm distances, fits per anchor x receiver group x condition, ACAT, specificity, clusters, roles, rings -> sidecar files | bridge script in diffsvg's env (decision 1: or a diffsvg API) | proposed |
| B1 | Sidecar -> `var` / `obs` columns, masking selectors | fork `tl/svg_targets.py` | proposed |
| B2 | Anchored readouts: model and knockout curves, attributable share, nested knockouts, anchor senders, ring groups | fork `evaluation/anchored.py` + small options in `interpretation` / `gene_ranking` | proposed |
| R1 | synth_spot: four anchor strategies (model-free table), set-masked models, rankings, rules T1-T5 | one cluster job | proposed |
| R2 | legnini: SHH anchor per condition, organoid replicates, curated programme as reference, rule L1 | one cluster job | proposed |
| R3 | vannan (niche anchors) once its model exists; CosMx (islets) | one job each | proposed |
| X | Optional: ring curriculum, condition contrasts, LR / NicheNet annotation | | proposed |

**MVP (smallest end-to-end test):** D0 + D1 + D2 with one anchor rule per dataset, then the existing programme masking
(`mask_genes: ["svg_role=target"]`) and the existing E7 rankings. No new InterScale library code; B2 comes second.

**Round 2 (2026-11-22, end of file) revises this design after the user's points:** anchors are probes (patches of arbitrary
cells allowed, placed and fitted per slide), ranking by the Wald statistic, refined modules, masking units = within-cell leak
groups, rotation, three equal rings. Where a section above disagrees with round 2, round 2 holds; the table above is superseded by
the round-2 status table.

## What it can and cannot show

*(Round 2: the user's framing holds: diffsvg detects response-like correlation sets and is not meant as causal evidence. The
ladder below is kept as a way to say how strong a finding is; L3 applies only to anchors that are a sender hypothesis.)*

**An anchored curve is spatial variability, not communication.** diffsvg says: the expression of g changes with the distance to
anchor a. Four things produce that:

1. composition: the cells near a are of other types (an islet's neighbours are not the average acinar cell);
2. a shared axis or environment: a and the gene follow the same tissue structure (depth, vessels, a region) without a acting;
3. technical gradients: spillover from a's cells into neighbouring segmentations, FOV / TMA / edge effects, handling stress;
4. a signal from cells at a (or co-located with a) changing the receivers' state: the CCC case.

With arbitrary patches as anchors, (2) and (3) dominate: a set of patch curves is a radial-basis SV screen (like SPARK's kernel
set). That is useful as a negative control, not as a CCC filter, and it matches E7's rule (2026-11-22) that genes which only vary
in space are negatives.

**Evidence ladder.** A CCC call should pass L1-L3; L4 gives the mechanism hypotheses.

- **L1 (diffsvg, model-free).** Non-flat within a receiver type (composition out), consistent across slides (random curve per
  slide), specific to the anchor (stronger than around pseudo-anchors). Where a condition perturbs the sender (legnini's
  optogenetic SHH), the condition curve test adds a perturbation argument no snapshot model can give.
- **L2 (InterScale, prediction by intervention).** The hidden target is explained by the other cells' states beyond the receiver's
  own genes, its type and its position: E7 responder score (every other cell's state swapped within type, receivers keep their
  rows), above the twin and the bars. Prediction accuracy alone says nothing: an SV target can be predicted from position or from
  visible genes that vary in space the same way.
- **L3 (InterScale, anchor intervention).** Replacing the anchor cells flattens the model's distance curve (the anchor-attributable
  share), replacing as many random cells does not, and the model's curve reproduces diffsvg's.
- **L4 (sender genes, pairs, range).** Which genes of the anchor cells (and of other cells) carry it, near vs far, with
  ligand-receptor support as annotation, not as a filter.

**What each tool contributes.** diffsvg is the sensitive, calibrated part: which genes vary with which structure, consistently
across replicates, with which shape. InterScale is the specific part: which of those the model explains from other cells, by
which cells and genes, at which distance. Neither alone is a CCC test. The two strongest links: (a) diffsvg's clusters are the
right masking unit (hiding a whole co-shaped cluster removes the leak through visible partner genes in the same cell); (b) the
observed curve against the model's knockout curve is a direct check that the gradient is attributed to the anchor cells.

**Scope.** Anchors find source-organised communication: morphogen sources, structures (islets, TLS, fibrotic foci, floor plate).
They miss responses to dispersed senders (synth_spot's short-range S1-S4 -> R1-R4, contact signalling between scattered cells):
those need the type-pair readouts (E7 with channel masks). Interventions say what the model uses, not what the tissue does:
"reading where the anchor is" and "a signal from the anchor" cannot be told apart in a snapshot without a perturbation or prior
knowledge.

## Facts checked in the code (2026-11-22)

- **synth_spot places every source at a random position per slide** (`evaluation/synth_spot.py`, morphogen block: `cluster()`
  draws `_uniform_point(_rng(seed, slide, name), grid.half, frac)`; sparse sources by `_spread_points` per slide). A patch at fixed
  coordinates is near M7's source on one slide and far from it on the next: stacked over slides its curve averages unrelated
  geometry, and the slide random curve absorbs what is left. The same holds on every real dataset (sections are not registered).
  **Anchors must be a rule evaluated on each slide (content), not coordinates.** *(Round 2: superseded. The fact argues against
  stacking fixed patches across slides, not against patches: patches are placed and fitted per slide.)*
- **diffsvg** (`HiWi/diffsvg`, main c83b17e): the anchor is a [0, 1] column of `obsm["design_matrix"]` (`squidpy.tl.var_by_distance`:
  distance to the closest anchor cell of the same slide, scaled per slide); `bin_obs` refuses `sample` without `condition`;
  `fit_profiles` refuses slides ("does not model slides; bin without `sample`") and computes no test, but stores
  `varm[f"{key}_beta"]` and `uns[f"{key}_beta_cov"]`, which is all a flatness test needs (`parallel_curves_test(beta,
  shape_metric(K, degree), tau, cov)`, the idea of 2026-11-12); `_exact.slide_design` / `slide_posterior` take any fixed design, so
  the profile model with slides is the fixed design `[1, B]` instead of `[1, B, c, B*c]`; `cluster_curves(by="profile")` reads one
  key.
- **Environments:** diffsvg needs `anndata>=0.13`; interscale_env has anndata 0.12 and pandas 2 (pymc_env has pandas 3: its pickles
  do not load in interscale_env). The bridge runs in diffsvg's env and hands InterScale plain files (CSV / JSON / npz), no pickle.
  The cluster's diffsvg clone still needs the reset to the new main (memory, 2026-11-12).
- **InterScale already has:** `dataset.mask_genes` / `mask_gene_sets` with `var` selectors; E7 `gene_ranking.context_flows`
  (`all_state`), `interpretation.gene_flows` (`far_radius`: only cells beyond a radius change), `hide_genes(graphs, columns)`
  (folds can be any column sets), `spillover_flags` (fold 3); `ranking.rank_by_effect` (expression-matched robust z);
  `attribution.attribute(method="occlusion", reference=graph_mean | type_mean | sampled)` and `distance_profile`; the IR's exact
  decomposition. **Missing:** receiver groups separate from the permutation labels (one `labels` array serves both in
  `context_flows` / `gene_flows`), a sender-cell restriction for `gene_flows`, prediction curves by distance to an anchor, anchor
  knockouts. Gene-strategy models hide no cell whole, so removal by the model's own mask (`visible_keys_only`) is unavailable to them:
  knockouts replace rows.

## Design

### Anchors

*(Round 2: patches of arbitrary cells are anchors in their own right, placed and fitted per slide; pseudo-anchor specificity is
an annotation for rule anchors only; the exclusion of anchor-defining genes applies only to anchors defined by genes.)*

An anchor is a rule that marks a set of cells on every slide. Allowed rules:

1. **annotation:** levels of an obs column (cell type, niche, region): islets on CosMx (Alpha, Beta, Endocrine in
   `cell_type_coarse`), CNiche levels on vannan, the dentate gyrus on Xenium;
2. **marker:** the top q of a gene or gene-set score within each slide (legnini: top 1 % SHH, as in diffsvg's tutorial); q per
   slide so slides compare;
3. **niche (label-free):** clusters of neighbourhood composition (or smoothed expression), fitted on the training slides, assigned
   on all;
4. **coordinates** (grid patches, discs): only for slides in a common frame; everywhere else as pseudo-anchors.

**Pseudo-anchors** (the generic-SV control): per slide P = 20 random discs centred on random tissue cells, each holding as many
cells as the real anchor holds on that slide. Pooled exact fits suffice for them (they give a specificity score, not a p-value).

**Duplicates:** two anchors whose distance fields correlate above 0.9 (Spearman over cells, within slides) are merged; the table
keeps both names.

**Anchors never enter the model.** They define the targets (D2) and the groups and interventions of the evaluation (B2). The model
stays annotation-free, apart from the receiver type the IR's query may read, as now.

**Leakage of the anchor definition:** the genes that define the anchor (the marker gene, a niche's top-loading genes) and every
marker of the anchor cells (anchor-cell mean >= 3x the receivers', E7's spillover fold) are never targets: they are non-flat by
construction (graded identity, spillover). They stay visible as candidate senders.

### Distances and receivers

- **Distance in µm** (signalling ranges are physical), not per-slide [0, 1]: `d / R_max` with one `R_max` for all slides, the
  largest distance at least 75 % of the slides reach; cells beyond are NaN (`bin_obs` leaves them out). Per-slide scaling (the
  poster's choice) makes a ring mean different µm on different slides.
- **Anchor cells are left out** of the curves (they are the source, d = 0).
- **Receiver groups:** each cell type other than the anchor's (annotation anchors), or all non-anchor cells (no types: legnini,
  synth_spot). Composition along the distance then makes no curve.
- **Expression:** the layer the model trains on (log1p_norm). Under `section_offset: label_mean` the within-type curve has the same
  shape (the label mean is constant within type and section).
- **Conditions:** one fit per condition, both feeding the combination. The condition curve test (`fit_model(test="parallel")`,
  slides as replicates) is a separate flag where a condition acts on the sender.

### Tests

*(Round 2: ranking by the Wald statistic; patches fitted per slide; the slide model optional, for anchors that correspond across
slides.)*

Per (anchor, receiver group, condition): `fit_profiles` with a random level and curve per slide (D1); per gene the flatness test
(D0): `shape = sqrt(beta' Q beta)` (level-free, |slope| for a line), H0 shape <= tau, Wald statistic on the K - 1 shape
directions, F(K - 1, S - 1) reference with S replicates (chi-square without). tau follows diffsvg's defaults (0 with slides).
Replicate = the independent unit: organoid on legnini (8, not the 17 sections), patient on vannan, donor (= section) on CosMx (6,
three per condition: weak), slide on synth_spot.

**Why the slide model is required:** pooled slides inflated the condition test's null (shape_test_plan.md: within-SHH split null
94 % rejected pooled vs 4 % with slide effects, tau 0). "Stack all slides" therefore means one fit with slide effects, the curve
tested against the variation between slides; a pooled fit calls almost every gene non-flat.

### Combining anchors

- Per gene and receiver group: the **Cauchy combination (ACAT)** of its p-values over anchors and conditions (valid under dependence;
  SPARK combines its kernels the same way), BH over genes. Bonferroni of the minimum is the conservative check.
- The table keeps every (gene, anchor, group, condition) row: shape, statistic, p, the curve, and the anchor where the gene is most
  significant.
- **Specificity** per (gene, anchor): `(shape_a - median_pseudo) / MAD_pseudo` and the share of pseudo-anchors with a larger shape;
  anchor-specific when the shape beats 95 % of the pseudo-anchors. A filter and a column, not a p-value (no empirical nulls in
  diffsvg's tests, 2026-11-12).

### Clusters and roles

*(Round 2: refined and replaced by "Refined modules" and "Masking units: leak groups"; the hidden unit is no longer the whole
cluster.)*

- **Features per gene:** the centred spline coefficients of every anchor (and group) concatenated, unscaled (amplitude counts, flat
  blocks are ~0: diffsvg's `scale=False` default for the same reason), k-means with `n_init` 100 (`cluster_curves`, extended to
  several keys); k by silhouette over 4-12, or the user's choice.
- **Selection:** a cluster is a target cluster when its centre's shape exceeds tau for some anchor and at least max(2, 20 %) of its
  genes are BH-significant and anchor-specific.
- **Roles** (`var['svg_role']`): `target` (genes of selected clusters; `svg_significant` marks the individually significant ones),
  `anchor_marker` (anchor-defining genes, anchor-cell markers), `flagged` (handling / stress / FOV genes from
  `biology_resource.json`, spillover flags), `sender` (everything else, the flat clusters included). Only `target` is ever hidden.
- **Why the whole cluster is hidden, its non-significant members included:** a hidden gene whose co-shaped partner stays visible in
  the same cell is predicted from that partner, no other cell involved (the channel-masking caveat, 2026-10-23). The user's "genes
  grouped with significant SVGs" is what closes that leak.

### Rings

*(Round 2: the user decided three rings of equal width at a physical distance set by the slide size; near / far for senders stays
the local reach.)*

- **Edges** per anchor in µm, tied to the model's local reach `r_loc` (2-hop graph reach, or the narrowest attention window): ring 1
  `[0, r_loc)` (the anchor lies within the local component's reach), ring 2 `[r_loc, 3 r_loc)`, ring 3 `[3 r_loc, R_max]` (only
  long-range paths reach the anchor). Beside them, the response range read off the curves: where a selected cluster's centre has
  made 50 % / 90 % of its change.
- **Uses:** receiver groups for every score; the knockout curves by ring; the local / global question by construction (a response
  that still changes in ring 3 and is attributed to the anchor there is long range).
- **"Step-wise", three readings:**
  1. evaluation by ring (scores, rankings, attributions per ring): no training change; first;
  2. **nested knockouts** for ring-3 receivers: replace the anchor (rings 1-2 kept) vs replace rings 1-2 (anchor kept) vs both, each
     against replacing as many random cells. Anchor replacement matters -> direct long-range reading; only the rings matter -> the
     model reads the gradient from nearby cells (a relay, or a smooth field read locally: fields run 4 showed that with genes visible a
     long-range field is read from the near neighbours);
  3. a **ring curriculum** in training (loss on ring 1 first, then rings 1-2, then all): changes the optimisation, not what can be
     learnt; a lever against uniform attention (ring-1 targets need the nearby anchor cells). One optional arm (stage X), read by its
     ring-3 attributable share against the same arm without it; needs a loss mask by ring label (`optim.regime_loss` needs cells
     hidden whole).

### Training

- **Arms:** `mask_gene_sets` with the selected clusters as sets, one per draw, in every cell (rotation: every cluster is a target in
  some steps and a candidate sender in others, so a relay gene can be both); `mask_genes: ["svg_role=target"]` (all targets at once:
  the programme-masking path, which trains well, 0.76-0.82 on synth_spot) as the MVP and the control.
- **Models:** the IR (J1's `ir2_rate` recipe) and the context-query reference (`ctx`, as J5's `ctx_ch`), 2 seeds, untrained twins;
  300 epochs on synth_spot (J5), 240 on legnini (legnini_programme).
- The model never sees anchors, rings or clusters.

### Readouts (per receiver group x ring)

1. **Responders within the targets:** E7 `context_flows` (`all_state`). On synth_spot every target is SV by construction, so the
   ranking asks exactly: which SV genes does the model explain from other cells (planted responders), which not (hard negatives)?
2. **Sender genes:** `gene_flows` drivers; near / far from `far_radius = r_loc` (far = only cells beyond r_loc change; near =
   full - far); expression-matched robust z (`rank_by_effect`). With types, a gene driver score measures within-type variation of the
   sender gene (a type marker permuted within its type changes little); the type-level effect is readout 6 and the knockout.
3. **Anchor senders:** `gene_flows` with the permutation restricted to anchor cells (new `sender_mask`): which gene of the anchor
   cells matters. Report co-marker groups (anchor-cell genes correlated above 0.8) rather than single genes (legnini: SHH with
   FOXA1).
4. **Pairs:** sender gene -> target cluster (mean drop over the cluster's genes), near / far, per ring; ligand-receptor support (liana
   resource) as a column.
5. **Curves:** observed (diffsvg), model (prediction averaged per distance bin), knockout (anchor cells replaced by sampled far-ring
   cells of the same type; `attribution` occlusion, `sampled` reference). Numbers: `reproduced = corr(c_mod, c_obs)` with
   `||c_mod|| / ||c_obs||`; **attributable share** `1 - ||c_ko||^2 / ||c_mod||^2` (centred curves); the same for the pseudo-knockout
   and the twin.
6. **Sender types** (annotated data): `identity` - `identity_null` per receiver group and ring (exists).
7. **IR models:** the exact decomposition gives the anchor's share of each receiver's context part and the share beyond r_loc
   (`distance_profile`).

### Bars and controls

Untrained twin per arm; pseudo-anchors (curves and knockouts); hard negatives (SV-only genes, flagged genes); the band-field cross
reader (`slurm/new/fable/`, passes the synth control) as the model-free sender / range bar (MISTy's design with rank shells); E7's
Moran / variance / neighbour-ridge bars; seeds (RBO >= 0.5, as C1); targets selected on the training slides, every readout on the
test slides.

## Datasets and what to expect

- **synth_spot (truth).** Four anchor strategies, model-free first: *known* (top 1 % of each planted source gene per slide, the
  analyst's analogue of "SHH+ cells"), *found* (from expression alone: genes whose per-slide top 1 % forms a few connected pieces
  on most training slides), *grid* (16 fixed patches, the user's version: expected to fail by the fact above), *pseudo*. Truth:
  responders of the morphogen programmes (MC1-MC15) for T1; the sparse responses (MC16-MC25) are diluted in an all-spot receiver group
  (about 1 % of spots respond), reported separately; senders M1-M21; pairs; planted scales; hard negatives = roles spatial / marker /
  batch / condition (SP1-10, CM1-3, bg, cg); R1-R4 (dispersed senders) out of scope, reported. Mirror programmes (MC1/2/4/6) are a
  function of the spot's own source gene, so their responder score is small by design. Directional responses (stripe, cone) are
  invisible to a radial test.
- **legnini (no truth, known biology).** Anchor SHH top 1 % per condition, organoid replicates; the curated identity + readout set
  (18 genes, legnini_plan.md) as the reference for the targets; SHH (with FOXA1) the expected anchor sender; diffsvg's FDR hits
  (HHIP, FOXA2, GLI1, PAX7, NEUROG2; condition test, sections) as a cross-check. Caveat: PTCH1 / GLI1 correlate weakly with distance
  in the test organoid (+0.13 / +0.14).
- **vannan.** The best real candidate for anchors: annotated niches comparable across 45 samples, 35 patients as replicates. CNiche
  anchors (fibrotic, TLS, airway, IFN). Needs a trained vannan model (job submitted 2026-11-22).
- **CosMx.** Islet anchor, receiver types acinar / ductal / immune / endothelial / stroma, ND vs T1D; 6 donors (df 5), and the audit
  found little neighbour signal within type (ceiling_plan.md): expect weak results. Biology to look for: peri-islet acinar cells
  (the islet "halo"); INS / GCG near islets are partly spillover (fields run 3).

## Build stages (with their verification)

- **D0 (diffsvg)** `tl.flatness_test(adata, profiles_key="dsvg_profiles", tau=None)`: reads the stored `beta` and covariance, writes
  `var` columns `{key}_flat_shape`, `_flat_stat`, `_flat_pval`, `_flat_pval_adj`; F reference when the fit had slides (`uns`
  records S). Verified: flat toy curves give uniform p (KS), a straight line of slope s gives shape s, adding a constant changes
  nothing, a power curve.
- **D1 (diffsvg)** slides for condition-free fits: `bin_obs(sample=, condition=None)` allowed, `_exact_profile_slide_results` =
  `slide_posterior` on `[1, B]`. Verified: the exact posterior equals a brute-force GLS on a tiny problem; a toy null with zero-mean
  slide curves (pooled rejects often, the slide model <= 1.5 alpha at alpha 0.05 over 500 genes); a distance-shuffled null within
  slides on legnini (cluster).
- **D2 (bridge)** `svg_targets.py` in diffsvg's env: anchors, distances, fits, ACAT, specificity, clusters, roles, rings; sidecar
  `targets.csv`, `anchors.json`, `curves.npz`, `rings.csv`, `meta.json` (split, slide list, settings, versions). Verified on a toy
  synth_spot (`make_synth_spot(n_slides_per_condition=6, n_side=20)`: a few hundred spots locally, < 1 min) with one planted
  source; the full data on the cluster only.
- **B1 (fork)** `tl/svg_targets.py`: sidecar -> `var['svg_role']`, `var['svg_cluster']`, `var['svg_significant']`,
  `obs['svg_ring_<anchor>']`, `obs['svg_anchor_<anchor>']`; refuses a sidecar fitted on anything but the training slides, or naming
  unknown genes. Verified: a sidecar listing legnini's curated 18 genes gives the same `gene_mask` as `legnini_programme`.
- **B2 (fork)** `evaluation/anchored.py` (rings, curves, knockouts, shares, nested knockouts), `gene_flows(sender_mask=)`,
  `receiver_labels` separate from the permutation labels. Verified on a toy grid: an anchor-driven target (share ~1), a position-only
  gradient (~0), a pseudo-anchor (~0); mutants (receivers' own rows moved, anchor cells kept, curves uncentred).

## Decision rules (proposed; to fix before R1)

- **T1** (synth, model-free): known and found anchors: recall of MC1-MC15 among the targets >= 0.6; hard negatives <= 40 % of the
  targets after the specificity filter. Grid reported; if grid >= known, the alignment argument is wrong.
- **T2** responders within the targets: AUROC planted vs SV-only >= 0.80 and >= twin + 0.15.
- **T3** senders: the planted source first among anchor-cell genes for at least half of the programmes; E7 G2 on drivers.
- **T4** attributable share: median over planted programmes >= 0.5 and >= twin + 0.2; pseudo-knockout <= 0.1; SV-only targets <= 0.2.
- **T5** range: far share vs planted scale, Spearman >= 0.5; long-range programmes in ring 3: direct > relay.
- **L1** (legnini): at least half of the curated genes among the targets; SHH or its co-marker group first among the anchor senders
  for the readout / ventral cluster; RBO over seeds >= 0.5.

## Jobs, cost, files

One job per experiment, two env steps (diffsvg's env on the CPU cores, then interscale on the GPU), flat in the month's slurm
folder (`slurm/November/` for now).

- **R1 synth:** diffsvg step ~1 h on 12 CPUs (estimate: ~20 slide fits for known / found anchors at 2-5 min each, from the
  measured 67-118 s for 88-347 genes; pooled fits for grid and pseudo); training 2 arms x 2 seeds + twins (~30 min each, J5's
  estimate); rankings and anchored readouts ~30 min: ~4 h in all.
- **R2 legnini:** diffsvg ~20 min (SHH anchor x 2 conditions), training ~2 h, readouts ~30 min.
- **Checks before handing over** (cluster-jobs-must-finish): both envs import the expected versions (diffsvg with D0 / D1), `load_arms`,
  `_validate`, name length, the GPU check, the sidecar's `meta.json` against the run's split.

## Risks and limits

- Selection finds mostly region or technical gradients (CosMx FOV, vannan TMA / site). Within-type fits, specificity, flags and the
  responder ranking address it; expect many targets to be negatives.
- Hidden-cluster models may fit poorly (channel masking hardly fitted its task, 2026-10-23); all targets at once (programme masking)
  is the fallback and trains well.
- Sender genes of one cell population are collinear (report groups); see readout 2 for what a gene driver score measures.
- The additive IR cannot represent a threshold after aggregation (J1: M19-M21): such programmes look like non-responders in the IR;
  read the ctx arm.
- Observational: an attributable share says the model uses the anchor cells, not that they signal.
- Time: the thesis is being written; the MVP first.

## Open decisions (user)

1. Multi-anchor code in diffsvg (a public API, upstream PR) or only in the bridge script. D0 / D1 belong in diffsvg either way (the
   flatness test was wanted as a core test, 2026-11-12). Proposed: D2 as a bridge script first, generic parts (ACAT, multi-key
   clustering, µm distances) into diffsvg once they prove useful.
2. Anchor rules per dataset (the defaults above).
3. Ring edges: multiples of the local reach (default), curve-derived, or fixed µm.
4. Main masking arm: cluster rotation (default) or all targets at once.
5. legnini replicate unit: organoid (default) or section.
6. Scope now: the MVP only, or the full plan.

## Related work (positioning; novelty not checked)

- C-SIDE (Cable et al., Nat Methods 2022): cell-type-specific DE along spatial covariates such as the distance to a landmark, with
  replicates. L1 within receiver type is this with spline shapes.
- Niche-DE (Mason et al., Genome Biol 2024): a type's genes against the abundance of niche types at several kernel bandwidths, then
  niche-LR. The closest published analogue of anchors + rings + senders, without a learned model.
- MISTy (Tanevski et al., Genome Biol 2022): each marker predicted from intrinsic, juxta and para views at several radii, with view
  gains and predictor importances: the model-free form of "predict SV targets from other genes in rings, rank the senders by scale"
  (gene_ranking_literature.md, P1).
- NCEM (Fischer et al., Nat Biotechnol 2023): expression from neighbourhood types, linear vs nonlinear.
- NicheNet (Browaeys et al., Nat Methods 2020): ranks ligands by how well their prior regulatory potential predicts a response gene
  set. diffsvg's target clusters are such sets (a knowledge-based bar, stage X).
- SPARK (Sun et al., Nat Methods 2020): Cauchy combination over spatial kernels, the precedent for combining anchors.
- GASTON (Chitra et al.): a learned isodepth instead of a hand-picked anchor; an alternative anchor source, not needed now.

The combination here (replicate-aware curve tests choose the hidden programmes of a masked whole-slide model; the model's
interventions are checked against the observed curves) is not known to me as published. Check before calling it new.

## Round 2 (2026-11-22): the user's points, their decisions, the refined design

### The user's points

1. diffsvg is not meant as a causal explanation of CCC. It helps detect correlations that resemble, e.g., long-range responses to
   a morphogen: a set of SVGs can be a set of response genes to some source. Its limit: it needs an anchor, so the sender must sit
   in or around the anchor.
2. Anchors need not be defined by a gene (SHH+ cells can be one): morphological regions, clusters or coordinates; patches / discs
   are arbitrary cells. Then there is no anchor gene: the model looks for the genes that predict the SVGs well.
3. Starting with source-organised communication is fine: "we need to start with something. So far our models barely work at all."
4. The test: a flatness test (the curve against a flat line) or the parallel test for the single spline without conditions; my
   preference asked, or both. It must be calibrated like the parallel test (no p-value inflation).
5. p-values matter little: rank by the Wald statistic and pick the clusters that contain SVGs (genes with non-flat curves, a high
   Wald statistic), treating their significant and non-significant members together as SVGs.
6. Refine the cluster rule ("cluster the genes on their curves for all anchors side by side; keep a cluster if its centre curve is
   non-flat and it holds enough significant, anchor-specific genes").

**Decisions:** the code that combines anchors stays out of the diffsvg API for now; rings at a fixed physical distance set by the
slide size, at most three (close, mid, far) of equal width; masking: the user leans to rotation and asked for my view; the full plan
as fast as possible once these points are settled.

### What changes

- **Framing.** diffsvg detects response-like correlation sets (the targets); the model ranks the genes that predict them (the
  senders). The ladder above stays as a way to say how strong a finding is, not as a requirement. The anchor knockout (L3) applies
  only to anchors that are themselves a sender hypothesis (SHH+ cells); optional.
- **Anchors are probes.** Every kind is allowed, patches of arbitrary cells included. The synth_spot fact (sources drawn per slide)
  argues against *stacking* fixed patches across slides, not against patches: patches are placed on each slide and fitted per slide,
  and their statistics are aggregated over slides. That also answers point 1's limit: a grid of patches on every slide puts any
  source near some anchor on every slide, wherever it sits. Stacking with a slide model stays possible for anchors that mean the same
  on every slide (SHH+, an annotated region, a cluster), not needed for the first runs.
- **Leakage exclusion** only for anchors defined by genes (the defining genes); none for patches. **Pseudo-anchor specificity**
  becomes an annotation for rule anchors; selection uses consistency across slides and module stability instead.
- **Direction.** A source that is itself an SVG will also be "predicted" from its responses; a snapshot cannot tell direction, so
  pairs come out bidirectional. The ranking finds the coupled pair; priors (ligand-receptor) or a perturbation give the direction.

### The test (point 4)

The two options are one test. The parallel test applied to a single curve's coefficients is the flatness test against the best
flat line at any height (`shape = sqrt(beta' Q beta)`, `Q 1 = 0`, so the level drops out; |slope| for a straight line). A flat line
at a fixed height (the intercept, the far end) lets the level in and picks up offsets: not proposed. The 1-df slope test sees only
monotone trends (no arches or rings): not as the main test. **Build once:** `tl.flatness_test` (D0).

**Calibration.** With replicates (a slide model: D1, or `fit_model` with conditions and slides) the flatness test uses the parallel
test's reference with the variation between replicates as the denominator, so it should be calibrated the same way (to verify with a
null, as the within-SHH split null did for the condition test). **Within one slide**, no test that treats cells as independent is
calibrated against spatially correlated expression: a smooth field unrelated to the anchor has a flat expected radial mean, but its
realised mean wanders more than independent cells allow, most near the anchor where the annuli are small. For patches fitted per slide
the p-value is therefore descriptive and the Wald statistic a ranking score, which is what point 5 asks. D0's checks include Gaussian
random fields on toy coordinates (locally) and on real coordinates (cluster) to measure how inflated per-slide p-values are. If
calibrated per-slide p-values are wanted later: a spatial block-jackknife covariance of the coefficients with an F reference (not
planned).

### Ranking by Wald (point 5)

Agreed, with two adjustments:

- **The Wald statistic grows with the number of cells** (a tiny shape is measured precisely on a 60k-cell section): compare genes
  within a slide (percentile of a gene's best-anchor Wald among the genes of that slide), aggregate the percentiles over slides
  (median), and require a minimum shape `tau_min` in expression units (0.1, the condition test's default, tuned on synth_spot). Sums
  or means of raw Wald statistics over slides only where slides have similar cell counts (vannan's range from 6k to 125k does not).
- **Consistency across slides replaces significance:** a gene counts only if its best anchor's shape exceeds `tau_min` on at least half
  the slides. An SVG of one slide is not a programme.

### Refined modules (point 6)

Per gene g and block b = (slide, anchor[, receiver group, condition]): posterior mean `beta_gb`, covariance, Wald `W_gb`, shape `s_gb`.
Per-slide patch fits with fewer splines (K = 6: coarse shapes, less noise; a knob).

1. **Shape coordinates.** `z_gb = L' beta_gb` with `Q = L L'` (the K - 1 shape directions): `|z_gb| = s_gb`, and Euclidean distances
   between z's are shape differences between curves: the test's own geometry, no ad-hoc scaling, the level gone.
2. **Gene scores.** `rank_g` = median over slides of the within-slide percentile of the best-anchor Wald; `size_g` = median over slides
   of the best-anchor shape; `cons_g` = share of slides with best-anchor shape >= tau_min. **Core SVGs:** rank_g >= 0.9 and cons_g >= 0.5.
   **Candidates:** rank_g >= 0.7 and cons_g >= 0.5. Thresholds tuned on synth_spot against structured vs noise genes.
3. **Pattern modules among the candidates.** Feature = the z's of all blocks concatenated, scaled to unit length per gene (the pattern,
   not the amplitude: among reliable SVGs the amplitude would split one programme into strong and weak members; diffsvg's unscaled
   default stays right for its single-anchor clustering, where noise genes must stay in the flat cluster). Average-linkage clustering on
   1 - correlation, cut at correlation 0.6: no k to choose. Mirror modules (centres correlated <= -0.7) are recorded as pairs.
4. **Stability.** Resample the slides (bootstrap, 50 times), rebuild features and modules, count how often each gene pair shares a
   module; a module is stable when its mean within-module co-clustering is >= 0.6; unstable genes stay candidates without a module.
5. **Weak members** (the user's "non-significant SVGs grouped with significant ones"): a non-candidate gene joins the module whose
   centre its pattern correlates with best, if that correlation is >= 0.6 and its shape exceeds tau_min / 2 on a quarter of the slides.
   Flat noise genes fail the size floor and stay out.
6. **Selection.** A module is a target module when its centre is non-flat (centre shape >= tau_min at its home anchor on >= half the
   slides), it holds >= max(2, 20 %) core SVGs, and it is stable. All its members (core, candidate, weak) are targets; `svg_core` marks
   the core.
7. **Home anchor.** Per module and slide, the anchor with the largest centre shape; rings and receiver groups are relative to it.

### Masking units: leak groups, not the modules themselves

*(Round 3: superseded by the flipped task. The SVGs are hidden only in the receivers (the rings) while the disc shows every gene,
so no masking units are needed.)*

Hiding a whole module closes the leak through a co-shaped partner in the same cell, but it also hides genes that share the shape
**without sharing cells**: a ligand made by sender cells and its target genes in nearby receiver cells have similar distance curves
(both peak at the source), so they fall into one module. Hiding them together hides the sender the model is meant to find, and with
patches there is no anchor gene to exempt. What leaks is co-expression *within the cell*:

- `r_gh` = correlation of g and h over cells, each slide centred (and each cell type, where annotated), averaged over the training
  slides (Fisher z);
- among the targets, genes with |r| >= r_leak (0.3; a knob) are linked; **masking units** = connected components (a component larger
  than a quarter of the targets is cut by average linkage at a higher threshold);
- a unit can split a module (sender-side and receiver-side genes) and join two modules (mirror genes in the same cells);
- reported per unit: its own-cell predictability from the genes visible while it is hidden (ridge on training cells: the `ref_own`
  bar), i.e. how much the receiver's own genes still give away.

Limits: where sender and receiver genes share cells (autocrine signals; floor plate SHH and FOXA2 may be such a case), no masking rule
separates them. On spot data (synth_spot) a spot mixes cells; check T1' measures what the rule does there.

### Rotation vs all at once (question 3)

*(Round 3: superseded. With the flipped task all SVGs are hidden at once, but only in the ring cells; the disc keeps every source
visible, which was the reason for rotation.)*

**Rotation, one unit per draw** (`mask_gene_sets` = the units, `mask_sets_per_draw: 1`), as the main arm:

1. sources are SVGs themselves: hidden all at once, a source gene could never be ranked as a sender (with patches no anchor gene is
   exempt);
2. every unit is a target in some steps and visible in others, so pairs between SVG units (relays) can be read;
3. each step hides little, so the context the model reads is nearly complete.

Cost: each unit is a target in 1 / U of the draws, and channel masking fitted poorly on 2026-10-23 (random channels, partners
visible, 170 noise genes). **All at once** (`mask_genes: ["svg_role=target"]`) is the control arm in the same job: the fallback if
rotation does not fit, and its sender ranking (non-SVG genes only) is the comparison. E7 reads rotation models in folds = the units
(`hide_genes`).

### Rings (decision 2)

- One `R_max` per dataset from the slide size: half the median slide extent (the larger side of the bounding box) over the training
  slides, capped where long distances cross unimaged tissue (CosMx's sparse FOVs: cap to choose, e.g. 1 mm); the same µm on every
  slide (my reading of "fixed physical distance"; per-slide widths were the poster's choice and make a ring mean different µm on
  different slides: to confirm).
- Three rings of equal width: close [0, R/3), mid [R/3, 2R/3), far [2R/3, R]; cells beyond R are not ringed.
- Rings are receiver groups, relative to the module's home anchor. Near / far for **senders** stays the model's local reach (the
  thesis' local / global: what the local component can see), via `gene_flows(far_radius=)`.
- synth_spot (50 x 50 spots at 100 µm, ~4.9 mm across): R ~2.45 mm, rings ~0.8 mm. legnini: to measure on the training slides (the
  largest farthest pair is 1,286 µm; the per-section maximal distance to the SHH anchor 299-768 µm), so rings of roughly 100-200 µm.

### Patches (default, to confirm)

Per slide a 3 x 3 grid of disc centres over the tissue's bounding box, each snapped to the nearest cell (dropped where no tissue lies
within a grid step), disc radius R_max / 10; distance = to the nearest disc cell, disc cells left out of the curve. With 3 x 3 on
synth_spot the centres are ~1.6 mm apart, so every source lies within ~1.1 mm of some patch. R1's model-free table compares these
per-slide patches with the stacked patches of the first proposal (the same coordinates on every slide, slide model), the known and the
found rule anchors.

### Status (round 2)

| Stage | What | Where | Status |
| --- | --- | --- | --- |
| D0 | `tl.flatness_test` (the parallel test on the single curve); checks incl. random fields | diffsvg branch | proposed |
| D1 | Slide model for condition-free fits | diffsvg | optional, later (calibrated p-values for anchors that correspond across slides) |
| D2 | Bridge: anchors per slide (patch grid + rule anchors), per-slide fits, Wald / shape table, gene scores, modules, stability, weak members, selection, leak groups, rings, sidecar | bridge script, diffsvg env | proposed |
| B1 | Sidecar -> `var` / `obs`; units as `mask_gene_sets` | fork `tl/svg_targets.py` | proposed |
| B2 | Receiver groups (ring x type) separate from the permutation labels; E7 per unit fold; drivers / pairs near / far; per-ring tables; anchor knockout for rule anchors (optional) | fork | proposed |
| R1 | synth_spot: anchor strategies (model-free table), rotation + all-at-once arms, rankings | cluster | proposed |
| R2 | legnini | cluster | proposed |
| R3 | vannan (after its model), CosMx | cluster | proposed |

### Decision rules, revised (proposed; to fix before R1)

- **T1' (synth, model-free):** structured vs noise genes by rank_g, AUROC >= 0.9; recall of the morphogen responses MC1-MC15 among the
  targets >= 0.7 and of their sources M1-M11 among the SVGs >= 0.7; planted source -> response pairs that share a masking unit <= 10 %
  (the sender stays visible while its response is the target).
- **T2** as before (responders within the targets vs SV-only targets); sources are left out of T2's negatives (see Direction).
- **T3' senders:** the planted source among the top 3 drivers of its response's unit for at least half of the programmes; E7 G2 / G3.
- **T4** only for the known (rule) anchors, optional. **T5** as before.
- **L1' (legnini):** at least half of the curated genes among the targets; SHH or FOXA1 among the top drivers of the unit holding the
  readouts or the ventral genes; RBO over seeds >= 0.5.

### Still open (asked 2026-11-22)

1. Ring width the same µm on every slide (default) or per slide?
2. Masking units by within-cell co-expression (leak groups) instead of the modules: agreed?
3. Patches: the 3 x 3 grid per slide (default), random cells, or another number?
4. D1 (slide model) left out of the first runs?

## Round 3 (2026-11-22): the flipped task

### The user's points

- The flatness test as the parallel test on the single curve (compared with the best flat line at any height): agreed.
- No simulations for p-value calibration: the histogram shows inflation (significant genes dominating the rest); following the
  parallel test it should not happen.
- "No need to hide clusters. We mask the SVGs within the rings and try to predict them from the discs (the discs should be large
  enough for this). Basically we flip the tasks as we look for senders, not receivers." Asked whether this is reasonable.
- Answers: ring width the same µm on every slide (yes, the simple default); round 2's question 2 (leak groups) unclear; patches a
  3 x 3 grid per slide to start (4 x 4 considered), and 9 fits per slide must run in parallel; round 2's question 4 (D1) unclear.

### Verdict and why it supersedes round 2's masking

Reasonable, and cleaner than rotation. Round 2's masking units, the rotation and the hidden-source problem all came from hiding SVGs
in *every* cell. Hiding them only in the receivers (the rings), with the disc showing every gene, keeps every source visible and
hides every same-cell partner of a target (all SVGs are hidden in the receivers together). The rings become the sender-receiver
distance itself (all senders sit in the disc), so close vs far ring is the local / global split without an attribution step, and
diffsvg's curves around the same discs are the observed profiles the model has to reproduce. The modules stay for choosing the SVG
set ("clusters that contain SVGs", weak members included) and for reading results.

### Design

- **One disc per graph and draw** (radius `r_disc`, default one ring width = R_max / 3; centre a random tissue cell, redrawn every
  epoch for training graphs; the 3 x 3 grid discs for evaluation): the **senders**, every gene visible.
- **Rings** close / mid / far of equal width from the disc's edge out to R_max (the same µm on every slide): the **receivers**; the
  SVG targets hidden in their cells (gene strategy, `mask_genes: ["svg_role=target"]` restricted to the ring cells of the drawn disc).
  Cells beyond R_max are neither keys nor targets.
- **Keys = the disc's cells only** (my addition, so that "predict them from the discs" holds; otherwise the receivers' neighbours
  explain the SVGs locally and the disc's share is a readout instead of the task). A per-cell key vector set by the draw, through
  the hidden-keys path the fused / readout attention already has (`StructuredMask.hidden_keys`); the IR's null key lets a receiver
  ignore a disc without a source. Keys from one disc also make attention cheap on whole CosMx sections.
- **Receivers read their own visible (non-SVG) genes** as query input (`context_query.inputs` with the visible genes), so
  receptor-gated responses (synth_spot's MR-gated sparse programmes) are learnable.
- **Model:** GlobalModel only (a GCN would mix the ring cells): the IR (additive read-out, null key, distance prior with windows up to
  R_max + 2 r_disc) and the ctx reference; untrained twins.
- **Same-gene continuation.** The disc shows the target genes too, so a ring target can be extrapolated from its own level in the disc
  (autocorrelation; a regional gene looks like this too: high in the disc, high in the close ring). Every score is therefore read
  *beyond the target's own level in the disc*: self-pairs are left out, and perturbations of other genes keep the target gene in the
  disc. If the self-share (the drop when only the target's own gene is replaced in the disc) dominates the fit, an optional arm with
  disc channel dropout (a random half of the SVGs hidden in the disc cells per draw) makes the model learn the other genes too.
- **Signal:** only discs that hold a source carry a response (a disc of one ring width covers ~9 % of a synth_spot slide); larger discs
  or more epochs if the fit is weak.

### Evaluation (test slides, the 3 x 3 grid discs, one draw each)

1. **Senders:** per disc gene s, its values in the disc cells replaced by values from cells of the same type *outside* the disc (a
   permutation within the disc would keep the disc's mean level); the drop of each target per ring, the target gene itself kept in the
   disc; self-pairs out; expression-matched z (`rank_by_effect`). Needs `gene_flows(sender_mask=disc, pool=outside)`.
2. **Disc responders (beyond self):** per SVG, the drop when every disc gene except the SVG itself is replaced the same way.
3. **Pairs:** sender gene -> target SVG and -> module, per ring; near / far = close vs far ring.
4. **Sender types** (annotated data): identity swaps restricted to the disc.
5. **Curves:** the model's ring means by distance against diffsvg's observed curve around the same disc; the bar is the average curve
   over discs (position only): beating it means reading the disc's content.
6. **Bars:** untrained twin; the average curve; the **disc regression**: ridge from each disc's mean expression to its SVGs' ring means
   across all training discs, beyond self (the target's own disc mean as a covariate): the model-free version of "predict the SVGs from
   the discs" and the first result, before any training.
7. legnini, extra draws: discs centred on the SHH anchor (the expected positive case).

### p-values: the histogram, no simulations

Per slide the histogram of the flatness p-values. Around patches many genes are genuinely non-flat, so a histogram heavy at small p
over *all* genes is not by itself inflation; the genes without spatial structure must give a flat histogram, and the data already
hold them: synth_spot's ~160 background genes (40 each NB / ZINB / POI / H), CosMx's 19 negative probes (NegPrb; kept for this check).

### diffsvg fits in parallel

Fits per (slide, disc) are independent: joblib (loky) over fits, 12 workers, BLAS / OpenMP threads 1 per worker (oversubscription was
~10x slower), one slide per worker in memory. The exact path is vectorised over genes, so a pooled fit takes seconds (< 1 s for 88
genes on legnini, ~2 s for 347 on Xenium). synth_spot: 24 training slides x 9 discs = 216 fits, ~1-2 min on 12 workers; CosMx: 2
training sections x 9 per fold; vannan: ~27 training samples x 9 per fold. A 4 x 4 grid later costs 16 / 9 of that.

### D1 explained, and dropped for now

D1 = a slide model for the condition-free fit: one curve per anchor fitted over several slides at once, with a random level and curve
per slide, so that the p-value compares the common curve with the variation between slides (as the condition test does with
`sample=`). It is needed only when one anchor is stacked over slides (an anchor that means the same on every slide, with calibrated
p-values). Discs are fitted per slide, so the first runs do not need it: dropped from the build, a later option.

### Status (round 3; supersedes the round-2 table)

| Stage | What | Where | Status |
| --- | --- | --- | --- |
| D0 | `tl.flatness_test` (the parallel test on the single curve) + per-slide p-value histograms | diffsvg branch | proposed |
| D1 | Slide model for condition-free fits | diffsvg | dropped for now (later option) |
| D2 | Bridge: 3 x 3 grid discs per slide, parallel per-(slide, disc) fits, Wald / shape table, gene scores, modules -> SVG target set, R_max / ring width / r_disc, disc regression bar, sidecar | bridge script, diffsvg env | proposed |
| B1 | Sidecar -> `var` / `obs`; the flipped mask (one disc per draw, SVGs hidden in its ring cells); keys restricted to the disc | fork (`tl/svg_targets.py`, `tl/masking.py`, the attention's key vector) | proposed |
| B2 | Evaluation on the grid discs: senders (replacement from outside, self kept), disc responders beyond self, pairs by ring, sender types, curves vs diffsvg | fork `evaluation/anchored.py`, `gene_flows(sender_mask=, pool=)`, ring receiver groups | proposed |
| R1 | synth_spot: model-free table (SVG detection, histograms, disc regression), IR + ctx flipped arms (+ twins), rankings | cluster | proposed |
| R2 | legnini (+ SHH-centred discs) | cluster | proposed |
| R3 | vannan (after its model), CosMx | cluster | proposed |

### Decision rules, revised (proposed; to fix before R1)

- **T1'' (synth, model-free):** SVG detection by rank_g, AUROC structured vs noise >= 0.9; recall of MC1-MC15 among the targets >= 0.7;
  the background genes' flatness p-values flat (histogram; KS against uniform reported).
- **T2'' disc responders (beyond self):** AUROC planted responses vs SV-only targets >= 0.80 and >= twin + 0.15.
- **T3'' senders:** on grid discs that hold a planted source, the source gene among the top 3 senders (self out) of its responses for at
  least half of the programmes, and ranked above where the disc regression puts it.
- **T5'' range:** the far-ring share of a source's effect grows with the planted scale (Spearman >= 0.5).
- **L1'' (legnini):** at least half of the curated genes among the targets; SHH or FOXA1 among the top senders of the readouts / ventral
  genes on discs that hold SHH+ cells; RBO over seeds >= 0.5.

### Still open (asked 2026-11-22)

1. Keys from the disc only, receivers reading their own non-SVG genes: agreed?
2. Disc radius = one ring width (R_max / 3) by default?
3. Training discs at a random position every epoch (default) or only at the 9 grid positions? Evaluation uses the grid either way.

## Round 4 (2026-11-22): the user's last answers, and the build

### The user's answers

- Agreed with the round-3 design, the costs and dropping D1 for now, but fine with adding the slide variant to the flatness test; asked
  for my view. Mine: worth adding (it reuses the condition test's exact slide machinery) for anchors that mean the same on every slide
  (SHH+ cells, islets, niches): one calibrated p-value per gene and anchor. For the 3 x 3 grid the targets still come from per-slide fits:
  grid position 5 is not the same place on two slides, so a stacked curve averages unrelated geometry and favours patterns that happen to
  line up across slides. Built (D1); the stacked-grid comparison arm of R1's model-free table is not written yet.
- Keys from the disc only: yes ("then the discs must be reasonably large"; a way to speed up computation if it works).
- Disc radius = one ring width: yes; if smaller discs are needed, narrower rings and more of them: a second run with arms of different ring
  settings (each needs its own bridge pass: the SVG selection depends on the disc geometry).
- Training discs: **the fixed grid positions first** (the SVGs are computed with respect to those discs; recomputing per iteration is too
  costly and random discs make interpretation hard); random discs (with pre-computed SVGs) later. Built that way: every draw picks one of the
  graph's 9 grid discs. The target set is one gene set (the selected modules' genes), hidden in the rings of whichever disc is drawn; the
  per-disc curves are what the readouts compare with. (If per-disc target sets were meant, that is a change of the draw: ask.)

### Built (2026-11-22; uncommitted, on branches)

diffsvg, branch `feature/flatness-test` (from main c83b17e):
- `tl.flatness_test(adata, profiles_key, tau=None)`: `parallel_curves_test` on the stored `beta` and its covariance; `var` columns
  `{key}_flat_{shape, stat, df, df_denominator, pval, pval_adj}`; F(K - 1, S - 1) after a fit with slides, chi-square without; tau default
  0 with slides, 0.1 without (as `fit_model`).
- D1: `bin_obs(sample=..., condition=None)` allowed; `fit_profiles` fits a random level and curve per slide (exact only; `slide_posterior`
  on `[1, B]`) and stores `n_samples` with the covariance; NUTS with slides refused.
- Tests `tests/test_flatness.py` (6: 60 flat genes not inflated, slope and arch found, shape of a line = its slope, level drops out, tau
  protects a tiny shape, zero-mean slide curves: pooled rejects > 50 % vs slide model <= 20 %, a shared curve survives the slide model);
  `test_samples.py` updated (the refusal of profiles with slides is now a fit). Full suite 73 passed, ruff clean.

Bridge `notebooks_and_scripts/slurm/November/svg_targets.py` (diffsvg env): presets synth, legnini (CosMx / vannan not yet); cells without
expression dropped (as InterScale); 3 x 3 grid discs, R_max = half the median training-slide extent, rings and r_disc = R_max / 3; fits per
slide in parallel (joblib, BLAS threads 1); gene scores, modules (shape coordinates, unit length, average linkage at 0.6, 50 slide
bootstraps, weak members, mirror pairs), selection (`min_core` 1: on synth_spot each gene's shape is its own, so a module is often one gene),
disc regression, histograms, report, sidecar. Toy check (generator at 16 x 16 spots, 12 slides, locally, 36 s): 28 targets = 15 sources,
7 responses, 6 spatial genes, no noise / marker genes; MC1-MC15 recall 0.47, M1-M11 among SVGs 0.91, AUROC structured vs noise 0.78 (tiny
slides); the null genes' p-values are **conservative** (piled at 1, KS 0.14: the prior shrinks curves fitted on ~250 spots; to read again on
the full slides); the disc regression loses to self-only at 72 toy discs for 253 genes (overfit; context only).

Fork, branch `svg-targets` (from main 93f3c0a):
- `tl.masking.anchor_grid_discs` (the bridge's rule, pinned against a numpy copy), `anchor_ring_masks`, `sample_anchor_rings`.
- `dataset.anchor_rings` (`grid` 0 = off, `r_disc_um`, `r_max_um`): validator (gene strategy, `mask_genes`, rate 1.0, no discs, spatial key,
  no hierarchy / pooling: they do not read `key_mask`), checkpoint tag `ar<grid>-<r_disc>-<r_max>`, `block_mask_kwargs` converts to the units
  of `pos`; `GraphAnnDataModule(anchor_rings=)` draws one grid disc per graph and draw: `gene_mask` = the gene set in the ring cells,
  `mask` = ring cells, `key_mask` = disc cells, `anchor_disc`.
- Encoder: a batch carrying `key_mask` lets only those cells be keys (training and evaluation; the `visible_keys_only` path, dense and fused /
  read-out); CLS stays open.
- `evaluation/anchored.py`: `disc_views`, `read_discs` (base correlation per target and ring; senders replaced in the disc by donors from
  outside, target kept; responders beyond self; distance-binned curves), `curve_agreement`.
- Tests `tests/test_anchor_rings.py` (5: geometry = the bridge's, ring boundaries = `np.digitize`, the draw, keys dense + fused, validator /
  tag / units; a mutant that drops `key_mask` caught by both key tests) and `tests/test_anchored_readout.py` (4: views, only disc rows change,
  the target's column kept in responder passes, constant sender = no effect, no self pairs, curve agreement). Related suites 140 passed.

Cluster files (flat in `slurm/November/`): `svg_targets.py`, `make_svg_synth_yaml.py` (arms ir_ar = J1's ir2_rate, ctx_ar = J5's ctx_ch, both
with the flipped task; disc radius / R_max / the distance window from the bridge's meta.json), `svg_synth.py` (PREPARE_ADATA: var['svg_role']
from targets.csv; EXTRA_EVALUATION: the anchored readouts on every grid disc of the test slides, curve agreement against diffsvg and the
average curve, T2''/T3''/T5'' metrics; standard evaluation off), `svg_read.py` (T1''-T5'' across runs and twins), `svg_job.sbatch` (STEPS
bridge yaml train read; SVG_ENV / IR_ENV). Checked locally: the yaml from the toy meta, every run through `load_sweep` / `arm_cfg` /
`_validate` / `check_masking` with the real `__main__` path (`runpy`, `sweep_common.main` stubbed; found an upper-case helper name that
`configure` refuses, fixed); checkpoint names <= 195 characters. **Not checked:** the hook on real prepared graphs with a model (a local build
and forward pass was refused, rightly: no model runs locally) -> first cluster step: a 2-epoch smoke run (`MODE = "smoke"`).

### Status (round 4; supersedes the round-3 table)

| Stage | What | Status |
| --- | --- | --- |
| D0 | `tl.flatness_test` | built, tested (diffsvg branch, uncommitted) |
| D1 | Slide model for condition-free fits | built, tested (same branch) |
| D2 | Bridge `svg_targets.py` | built, toy-checked; legnini preset written, CosMx / vannan presets not; stacked-grid comparison arm not written |
| B1 | `dataset.anchor_rings`, key restriction | built, tested (fork branch `svg-targets`, uncommitted) |
| B2 | `evaluation/anchored.py` | built, tested on toy modules |
| R1 | synth_spot job | written, arms validated; hook unchecked on real graphs (smoke first); not run |
| R2 / R3 | legnini / vannan / CosMx | not written |

## Commits and expectations for R1 (written 2026-11-22, before any run)

### Where the code is

- diffsvg `main` (local, 2 ahead of origin/main c83b17e; the user pushes): a9f586e "Add slide binning without a condition" (`pp/utils.py`,
  `tests/test_samples.py`), e207005 "Add flatness test and slide effects for condition-free profiles" (`tl/model.py`, `tl/__init__.py`,
  `tests/test_flatness.py`). Branch `feature/flatness-test` = the same commits. Note: a9f586e alone fails one test (its updated test calls
  `flatness_test`, added in e207005); the pair passes (73).
- Fork `main` (fast-forwarded from branch `svg-targets`): the code commit (`tl/masking.py`, `geome_dataloader.py`, `config/*`, `tl/utils.py`,
  `transformer_encoder.py`, `evaluation/anchored.py`, two test files) and a docs commit (CLAUDE.md masking table, this plan). Full suite
  1876 passed before the commit.
- Cluster files (not under git), flat in `notebooks_and_scripts/slurm/November/`: `svg_targets.py`, `make_svg_synth_yaml.py`, `svg_synth.py`,
  `svg_read.py`, `svg_job.sbatch`. Results: bridge `MA/22_11_results/svg_targets/synth/` (report.md, meta.json, targets.csv, modules.csv,
  discs.csv, fits.csv.gz, curves.npz, disc_regression.csv, histograms.png); training `MA/22_11_results/svg_synth/` (results.md,
  `anchored/<run>_anch_*.csv`, `svg_read.md`).

### Run order

1. Smoke: `STEPS="bridge yaml" sbatch svg_job.sbatch`, then `svg_synth.py` with `MODE = "smoke"` (2 epochs; the hook on real graphs has never
   run: a local build was refused). 2. Full: `STEPS="train read" sbatch svg_job.sbatch` (MODE back to "sweep").

### What to check first (sanity, before any rule)

- Bridge log: "R_max ... um" should read ~2450 um, ring width / disc radius ~817 um (synth_spot slides ~4.9 mm across); 9 discs per slide
  (fewer on cropped slides); fits time minutes, not hours.
- `discs.csv`: every slide has discs; `n_disc_cells` ~200 spots, `n_ring_cells` most of the slide.
- `histograms.png`, right panel (background genes): flat or piled at 1 (conservative) is fine, a spike at 0 is inflation. On the toy it was
  piled at 1 (KS 0.14); with full-size slides expect closer to flat.
- The training log: hidden entries per draw = targets x ring cells; the loss falls.

### Expectations (my predictions, to compare tomorrow)

| Item | Expected | Why / what would surprise |
| --- | --- | --- |
| T1'' AUROC structured vs noise | 0.85-0.95 | toy 0.78 on 16 x 16 slides; full slides give much better fits. Receivers R1-R4, markers CM and batch / condition genes have no radial shape around arbitrary discs, so they lower it |
| Targets | ~30-60 genes: most sources M1-M21, morphogen responses, SP spatial genes; no noise genes | toy: 15 sources, 7 responses, 6 spatial. Noise genes among the targets = a bug or inflation |
| MC1-MC15 recall | 0.5-0.8 (T1'' needs 0.7: may fail) | toy 0.47. Directional MC12 / MC13 (stripe, cone) and the narrow rings may be missed by a radial test on 9 discs |
| Null histogram | flat to conservative | the parallel test's construction; inflation would point at binning / weights |
| Disc regression (bar) | gain beyond self <= 0 for most targets | 216 training discs for 253 genes; self extrapolation is strong |
| Model fit (base corr, close ring) | clearly > 0 for responses near sources, near 0 for far rings | only discs that hold a source carry a response (~10-20 % of draws per source) |
| T2'' responders vs SV-only | uncertain, 0.6-0.8; the risk is spatial genes (SP) scoring as responders | a regional field is extrapolated from the disc as a response is; "beyond self" removes the same-gene part only |
| T3'' source in top 3 | likely for big Gaussian sources (M2, M7, M8), unlikely for sparse ones | sparse sources are single spots, rarely inside a disc; the IR misses thresholded programmes (J1) |
| T5'' far share vs scale | weak (Spearman 0.2-0.5) | three wide rings (~0.8 mm) are coarse against planted scales |
| ir_ar vs ctx_ar | ir_ar >= ctx_ar on senders (exact additive read-out); ctx_ar may fit better | as J1 |
| Twins | T2'' / T3'' near chance, but planted sources can rank high at random weights (downstream 2026-11-08: twin 5/15) | read every number against the twin |
| Cost | bridge < 30 min; training ~30 min per run (keys = one disc: cheaper than J1); readouts 253 senders x 9 discs x 8 test slides ~20k passes, perhaps 10-30 min per run | if readouts are slow: restrict ANCHORED["SENDERS"] |

### What a result would mean

- T1'' passes, T2'' / T3'' fail: diffsvg finds the right genes, the flipped task does not learn senders -> look at base corr per ring and at
  whether discs with sources are learned at all (curves vs the average curve).
- T3'' passes for Gaussian sources only: the method works where its premise holds (a sender inside an anchor) -> larger discs / more rings
  (the planned second run) for the rest.
- Spatial genes high in T2'': regional extrapolation; a "beyond position" control (the average-curve bar per target) is the next step.
- Everything fails including fit: check the draw (key_mask, ring cells) in the smoke log before anything else.

### diffsvg rebased (2026-11-22, later)

origin/main had moved 32 commits (docs restructure, scverse-misc logging, gene-set test, two-stage test, anisotropic slide scales /
effective df), so local main had diverged. The two commits were rebased onto origin/main 639678a: now **ffa5df1** "Add slide binning without
a condition" and **1d31e27** "Add flatness test and slide effects for condition-free profiles" (replacing a9f586e / e207005; the branch
`feature/flatness-test` points at the same). Conflicts: the `sample` docstring (upstream's "biological replicate" wording kept, the
condition-free sentence appended) and `tl/__init__.__all__` (both `flatness_test` and upstream's `gene_set_test`). The profile slide path
calls `slide_posterior` without the new optional `slope_prior` (the isotropic slide model). Full suite on the rebased main: 84 passed,
1 skipped; ruff clean. **Upstream now needs `scverse-misc[datasets]>=0.1.8`** (and pytest >= 9.1 for its tests): pymc_env was upgraded locally
(0.1.3 -> 0.1.9); the cluster's diffsvg env needs the same before `svg_job.sbatch` (its bridge preflight imports diffsvg).

## R1 read (2026-11-23, intermediate; job 43841310, results 23_11_results/)

Job ran to the end (4 arms x 300 epochs, read written; no error in .err). ir_ar 5.7 s/epoch, ctx_ar 2.7 s/epoch.

**Not a failed run:** the `run` row in svg_read.md is the 2-epoch smoke run (no W&B -> `run_name()` = "run", one graph) left in
`anchored/`; `svg_read.py` globs every `*_anch_metrics.csv`. Delete `run_anch_*` or filter twin/arm names.

**Bug (curve agreement only):** `svg_synth._slide_names` matches graphs by (cell count, rounded position sum); every full-size synth_spot
slide is the same lattice, so 4 of the 8 test graphs (A_03, A_13, B_10, B_20) are all named B_20 and 3 of them were compared with B_20's
diffsvg curve. `names` only labels curve rows: T2/T3/T5 unaffected; curve numbers diluted. Fix: carry the slide name on the graph (obs
slide of its cells) instead of matching by geometry.

Bridge (T1''): AUROC structured vs noise 0.869 (bar 0.9), MC1-MC15 recall 0.47 (bar 0.7) -> FAIL on detection; 23 targets (7 responses,
7 SP, 6 sources, 3 CM markers), below the predicted 30-60. Null histogram close to flat (slight excess in the first bin, KS 0.028 at
n 36k): calibrated, not inflated. Disc regression gain beyond self -0.04 / -0.05 / -0.11 by ring (as predicted: overfits, no bar).

Models (seed mean; twin in brackets): T2'' ir 0.857 (0.70) PASS, ctx 0.821 (0.67) borderline FAIL (+0.150 margin); T3'' both 0.857
= 6/7 (twins 0.43 / 0.57) PASS; T5'' ir 0.92 (-0.08) PASS, ctx 0.54 (0.75) not above twin; curve corr model 0.55-0.57 vs average curve
0.17, beats it on 67 % (twins 42-44 %); base corr ring 0/1/2 ~0.67/0.49/0.40 (ir s42), twin ~0.01.

Reading:
- Senders: M3, M7, M8 rank 1-2 with drops 0.1-0.25 (twin top drops ~0.02-0.05); MC1 / M1 (700 um, shorter than the 817 um disc radius)
  missed in every run (rank 5-31). Twins put sources in the top 3 for 3-4 / 7 (amplitude), so read drop size, not rank.
- The co-responses are the competitors: for MC7 the top senders are MC8 / MC9 / M7 together, for MC10 it is M8 then MC11. Inside a disc
  the source and its other responses are collinear; one-at-a-time replacement cannot separate them (as the 11-04 ranking runs found).
  "Top 3" is met partly because the competitors are the same programme.
- Hard negatives: SP genes mostly ~0 beyond self (SP1/2/4/6) but SP3/5/7 0.14-0.23 in the far ring; CM3 0.91 (CM1/2 ~0.2) — CM are the
  region markers (slide halves), so the disc predicts the ring's region: regional extrapolation, the predicted T2'' risk. The AUROC is
  carried by the planted responses being strong, not by the negatives being null.
- ir_ar is the better reader (T5 clean, slightly higher T2); ctx_ar fits a bit better (val SCE 0.207 vs 0.255) — as J1.

Next (proposed): fix the two script issues and re-read from checkpoints (MODE evaluate); a conditional / group knockout (replace the
whole co-response set except the source) to split source from co-responses; a "beyond region" control (region label or disc mean of CM
as covariate) for T2''; smaller discs / more rings for MC1-like short programmes (the planned second run).

## R2 design (2026-11-23; the user's requests after the R1 read; built, not run)

The user asked: fix the two bugs, replace a source's whole response group at once, add a "beyond region" control as a covariate, five
rings (disc size kept), a full rerun (not from R1's checkpoints), and the model's own gene rankings as the final output (csv + plots),
with sender-receiver pairs and a local / global split. Also asked: global-only? (yes: both arms are GlobalModel, keys = the disc only)
and why the histogram shows thousands (counts are (gene, disc) fits: 167 null genes x 216 training discs = 36,098).

Built:
- Bridge `svg_targets.py`: `n_rings` 5 (490 um rings), `disc_rings` 3 (r_disc = R_max / 3 = 817 um, as R1, so the target selection is
  unchanged), `sender_groups()`: |Pearson| of the training discs' mean expression, average linkage at 0.6 -> `sender_groups.csv` and
  `meta["sender_groups"]` (multi-gene groups); histogram axes say "fits (one per gene and disc)".
- `evaluation/anchored.py`: `read_discs(donor_labels={scheme: labels per graph | None}, groups=, batch_size=)`; every intervention per
  scheme (scheme 0 draws as before, tested); group sets G - {t} and G - {t, m} per target t / kept member m; `rank_tables(tables,
  local_rings, scheme)` -> rank_pairs / rank_senders / rank_responders / rank_groups (sufficiency = 1 - drop(m kept) / drop(group)).
  Tests: batched = one at a time, scheme 0 = the single-scheme read, label donors keep their label, group with a constant member ->
  the group equals the member's single drop and keeping it gives sufficiency 1, target's own gene never replaced (12 passed).
- `svg_synth.py`: graph names from the cells' slide (`hier_common.graph_names`; refuses duplicates), region donors from `obs['region']`
  (`ANCHORED["REGION_KEY"]`), groups from the bridge, rings from meta, local = rings within 1000 um of the edge (rings 0-1), metrics per
  scheme (`_region` suffix) + T3 group metrics (`T3_source_group_top3`, `T3_source_most_sufficient`), T5 on the global share,
  `<run>_rank_*_<scheme>.csv`; smoke runs write to `anchored_smoke/`; readouts timed. Results in `MA/23_11_results/` (SVG_OUT; changed from 24_11 at the user's request).
- `svg_read.py`: drops a smoke row, rules per scheme, `ranking/`: per arm and scheme the seed-averaged raw tables ranked by
  `rank_tables`, twin beside it (`twin_total`, `excess`), seed Spearman; plots `<arm>_senders_<scheme>.png` (local / global bars, twin
  tick), `<arm>_pairs_<scheme>.png` (top-15 senders x targets, local and global panels), `<arm>_local_global_<scheme>.png` (scatter,
  synth roles coloured), `<arm>_responders.png` (beyond self vs beyond self and region).
- `svg_job.sbatch`: SVG_OUT, time 11:55 (readouts ~2.6x R1's passes, batched 8; worst case ~10.5 h).
Checked locally: toy-model hook end to end (fake AnnData, curves, meta), svg_read on toy outputs (csv + png looked at), yaml from a
fake meta. Not checked locally: the bridge on data (local run refused, as the rule says): it runs first in the job and stops it on error.

Expectations (before the run): targets unchanged (23); groups {M7, MC7, MC8, MC9}, {M8, MC10, MC11}, {CM1, CM2, CM3} likely (perhaps M3
with MC3, M1 with MC1); with region donors CM3 / CM1 / CM2 beyond-self near 0 and T2'' up; group top 3 near 1 for the Gaussian sources;
the source most sufficient within its group uncertain (the model may read a co-response as well as the source); the 5-ring split
moves short programmes (MC1) into rings 0-1, so MC1 may now be read.
