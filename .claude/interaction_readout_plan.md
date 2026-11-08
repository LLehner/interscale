# Sender-receiver readouts: integrated gradients downstream, uncluttered plots, flow without cell types

**A brief for a separate session** (the user, 2026-10-29: points 7, 8 and 9 of "Focus until Tuesday 3 November", roadmap.md).
Nothing here is designed yet. The user asked that this work **not be too biased towards what we did before**. The section "What
exists" is context, so that facts are not re-derived; it is not a template to extend. This file is also the roadmap's planned
`attribution_plan.md` (T4) for the downstream part: the session writes its design here, with a status table, before any code.

## The three questions

7. **Integrated gradients in the sender-receiver analysis, downstream, with plots.** IG is the readout that found planted senders
   (below). It is used inside sweep evaluations today, not as an analysis a user runs on a trained model to get "who signals to whom,
   how far, through which genes", with figures.
8. **A visualisation that is less cluttered than the per-head plots**, and shows sender-receiver relationships meaningfully. Today's
   field figures have one row per head and class (arrows, arrows minus the head's own window, top-5 per head, top-5 by contribution).
   The user finds them too busy to read.
9. **How is attention flow calculated without cell types? THIS IS IMPORTANT** (the user's emphasis). The description's goal is
   interactions without annotation. Every summary that reads well today (sender type x receiver type heatmaps, the manuscript's
   net flow by cluster) needs labels, and the label-free ones (patch fields, head-mean arrows) do not read.

## Constraints any answer must meet

- **Annotation-free first.** Labels may summarise a result afterwards (an optional overlay), never define it.
- **Whole sections.** It must work on 10^5 cells with fused or routed attention, where no `S x S` map exists. Attention rows for
  sampled queries can be built (fused_attention.txt in `/home/lehnerl/Arbeit/notes/`); full maps and full rollouts cannot.
- **Validated where the truth is known** before it is read on real data: synth_spot's planted pairs by class (short, mid, dense,
  sparse, directional), against the untrained twin (model randomisation) and the expression-permuted slide (data randomisation).
- **Dataset-agnostic** (memory): datasets enter through config.
- **Fast enough** (memory: results must come fast): IG costs a backward pass per receiver and integration point (~0.05 s on the
  GPU); all ~3,000 receivers of a synth_spot slide would take ~40 min per run. Sampling, summed targets (IG is linear in its
  target) or cheaper surrogates are part of the design.
- **Feeds** comparison_plan.md (C2: `ccc_scores` and its distance column; C4: the figures) and lung_plan.md L4 (niche-pair tables).

## What exists (context, not a template)

- **Attribution:** `receptive_field.response_influence(measure="attribution")`: IG from the training-mean cell on one receiver's summed
  predicted response genes, Gauss-Legendre points, a completeness check (`response_check`), the sender ranked against cells as far
  away. `receptive_field.receiver_dependence`: IG per receiver, retried with more points while the completeness gap is over 0.1.
- **Flow:** `evaluation.attention_flow.patch_flow`: a grid of square patches; per source patch, the share of its cells' predictions
  drawn from each target patch, by `influence` (squared Jacobian, Hutchinson probes, the whole model) or `rollout` (head-mean
  attention with the residual, composed over layers, the transformer only), each against permutations of the visible expression.
  The patch fields need no labels; only the label-pair table (`plot_label_flow`) does.
- **Fields:** `evaluation.hierarchy_readouts.fields` (per head: cell mass, top-5 targets, `window_vec_heads` = the head's own trained
  window over the same keys) and `notebooks_and_scripts/slurm/hierarchical/scripts/fields_by_class.py` (the per-head figures of
  question 8).
- **Attention flow maps** (`evaluation/flow_maps.py`, built 2026-10-31 for the CosMx run at the user's request; context, not a template): per slide the net
  attention `A_ij - A_ji` of the last layer (head mean, cell keys, mutual pairs), deposited on a grid and drawn as streams, one tile per tissue block, no
  labels, whole sections (a sparse matrix from the routed rows). Checked on synth_spot's Stage B exports against the untrained twin (the later-reads bullet
  below, notes.md 2026-10-31). The user scrapped the quiver plots on CosMx (empty slide, tiny arrows) and asked for these.
- **A proposal never built** (roadmap.md T4, "Flow without annotation (task 5)"): per-cell sender vectors from attributions minus the
  permuted-slide expectation and the distance kernel; divergence for sources and sinks; clusters of cells by their vectors. Read it,
  but do not treat it as decided.

What earlier reads found (notes.md):
- Raw attention, rollout and relevance do not find planted senders (matched ranks 0.47-0.55, twins 0.48-0.52; 2026-10-21). The
  attention itself is computed correctly.
- IG does: the dual models rank short-range senders 0.98 and M11's 0.96 (twins 0.48-0.55); global-only models at 0.50-0.59 at 100
  epochs (2026-10-23); `hier_none` mid 1.00, short 0.97 (2026-10-26). It is trustworthy where the completeness gap stays under 0.1,
  and converges slowly where the response barely moves.
- Head-mean arrows are slide geometry: wide heads pull every arrow to the slide centre, and the untrained twin already scores 31°.
- **Later reads qualify the first bullet (2026-10-31, Stage B exports of synth_spot, 300 epochs, windowed hierarchical models, 22 slides):** the rank-based
  readouts at chance were early near-uniform models. Trained, the attention's top 5 (head mean) holds a planted sender for 97% of the short-range receivers
  (untrained twin 69%: the distance window) and 73% of the mid-range ones (twin 7%; the best head 80%), and the sign of the net flow `A_rs - A_sr` points from
  sender to receiver for 61% / 94% of the pairs (twin 49% / 42%). IG: 99% / 100%, dense 9% (3x chance); no readout finds the sparse or directional senders
  (<= chance; the directional programmes are learned by no model). The choice is not "attention or IG" but what each is for: attention is cheap, whole-section,
  every receiver; IG is the stronger and gene-resolved readout, at a backward pass per receiver and integration point.
- The directional programmes (stripe, cone) are learned by no model (~0.22-0.25), so a field cannot be validated on direction yet.
- Gradient magnitude is a sensitivity, not a use (a mean-aggregating GCN ranked its own senders 0.41).

## Deliverables of the session

1. An explanation of how a flow can be computed without cell types, and a method: the objects (per cell, per pair, per gene), the
   null each is read against, and its cost at 10^5 cells.
2. The downstream IG analysis as a function on a trained model (not a sweep hook), with its outputs: sender-receiver strength,
   distance profile, genes (which visible input genes carry a receiver's attribution), and the tables comparison_plan.md expects.
3. Two or three figure designs that replace the per-head rows, drawn on synth_spot against the planted truth, the twin and the
   permuted slide.
4. A staged plan in this file (status table, tests, the synth_spot check), then the code.

## A starting prompt for the session

> Read `.claude/interaction_readout_plan.md` and the files it names. Do not extend the per-head field plots. Propose, from first
> principles and the literature, how to read sender-receiver relationships without cell types from a trained InterScale model,
> using integrated gradients where they are valid, and how to show them in a few uncluttered figures. Check every proposal on
> synth_spot's planted pairs against the untrained twin and the permuted slide. Write the design into the plan file before code.

## Design 2026-11-06: from a trained model to biology (the user: "a good way to interpret downstream results ... flows of information, not purely attention, correlated with actual biology, same with the embedding dimensions; I need useful output")

| stage | state | verified by | date |
|---|---|---|---|
| `evaluation.interpretation` (the readouts), `evaluation.interpretation_report` (report per run, summary over runs, from the tables alone) | **implemented**; committed 2026-11-08 | 19 tests, 14 mutations caught (4 after stronger tests), the hook dry-run on a toy graph with the real resource | 2026-11-06 |
| `slurm/new/downstream_read.py`, `downstream.sbatch`, `biology_resource.json` | **written, not run** | settings accepted by the real `sweep_common`; `bash -n` | 2026-11-06 |
| The check on synth_spot's planted senders (`DOWNSTREAM=synth`) | **run and read**: context use 0.51 (twin -0.002); planted sender first for 12 of 15 targets, the twin 5 of 15 (planted senders have large effects at random weights): read against the twin | `07_11_results/downstream_synth` | 2026-11-08 |
| CosMx run (`downstream.sbatch`, step cosmx; 04_11 checkpoints) | **run and read**: identity, spillover, handling only; ΔR² scoring and the known-pair null found faulty (notes.md) | `results/downstream_cosmx` | 2026-11-07 |
| Scoring by centred cosine within the receiver type (flows, targets, gene flows; fixed section-wide type centres, sums pooled over draws and sections); known pairs ranked within each receiver type (`lr_support_by_type`, matched on detection there); the label-mean twin gets the training means (`sweep_common.give_label_means`); the readers load the shipped copies first; results to `07_11_results` | **run and read** on CosMx and synth (`07_11_results`, notes.md 2026-11-08); committed 2026-11-08 | 25 tests, 10 of 10 mutations caught (4 after stronger tests); the new pair test 6.5% / 1.0% at p < 0.05 / 0.01 over 200 null seeds, the old one AUC ~0.23 against a null of 0.50 on type-specific receptors; a miscalibrated model (x5, shifted) gives identical flows; the twin fix on a real label-mean GlobalModel | 2026-11-07 |
| Known-pair p over shared genes (`lr_cluster_p`: bootstrap over ligand and receptor genes; the twin's sub-null AUC was the per-pair null's too narrow spread); the planted check against the twin (`twin_of`, `planted_against_twin`); conditions with sections as replicates (`condition_by_section`); gene flows over runs; `DOWNSTREAM=stability` (the task 4 checkpoints) | **implemented**; the 07_11 summaries rebuilt from their tables (scratch copy); stability not run; committed 2026-11-08 | 30 tests; shared-gene nulls: per-pair p < 0.05 in 42% of 12 seeds, the clustered p in 0% | 2026-11-08 |
| Gene influence against spatial pattern (report section 7: `gene_flows(far_radius=)`, `gene_influence`, `moran_genes`, `gene_spatial`, `influence_read`; `DOWNSTREAM_READOUTS=influence`, steps `synth_influence`, `stability_influence`); summary figures `flows_over_runs.png`, `condition_over_runs.png` | **implemented, unverified on data**; committed 2026-11-08 | 34 tests (near sender exactly unchanged in the far region, `moran_genes` = `moran` to 1e-12, a toy with known local / global senders); figures drawn from the 07_11 tables | 2026-11-08 |

**Why not integrated gradients first.** IG costs a backward pass per receiver and integration point. An intervention needs one forward pass per perturbation, works on whole sections, is directional by construction and states what the prediction uses. `sender_deletion` with `context` + permute already found 10 of 10 planted senders on synth_spot (notes.md, 2026-11-03/04). IG stays the gene-resolved check for single receivers (questions 7 and 9 above stay open for the annotation-free part).

**The objects and their nulls.**

| question | object | null |
|---|---|---|
| does the prediction use other cells | (since 2026-11-07: the drop in centred cosine within the receiver type; was ΔR²) on the receivers' hidden genes when all other cells swap profiles within type (`all_state`) or among all (`all_arrangement`) | untrained twin; spread of the draws |
| which type affects which | `identity` (sender cells take another type's profile) and `state` (they swap among themselves), per receiver type, in the `context` (only the receivers kept) and `far` (their neighbourhood kept too) | `identity_null`: the same change on the nearest cells of other types, for co-location; twin |
| through which genes | one gene permuted within type in the context; per receiver type and target | known pairs against the same ligand's other targets of similar expression (`lr_support`); twin; planted pairs on synth_spot (`edge_recovery`) |
| what the attention's sources and sinks are | smoothed `flow_out - flow_in` against type, against each gene within section and type, against neighbourhood composition; top sources by type | labels shuffled within 100 µm squares; twin |
| what the programmes are | varimax of the prediction's first 12 directions: genes, gene sets (the panel as background: K11), type (eta²), Moran's I of what the type leaves at 8 and 128 neighbours, neighbourhood R², condition | twin; a class by stated thresholds |
| disease | per flow and programme, later minus earlier condition, per run: one donor pair each, so read over folds by sign | three pairs keep one sign by chance a quarter of the time |

**Design points.**
- `state` is the flow that does not suffer from the permutation notebook's label mixing (notes.md 2026-11-06): the type at each position stays.
- Receivers are chosen per type, rare types first, with a share kept back so a type also has senders; fresh per draw.
- Labels come from config. On data without annotation they are clusters (`imitation_labels`), so this is a summary by label; the annotation-free question above stays open.
- Every table fills whatever the model is (a random stub in the dry run gets pathways and top pairs as well), so the summary puts the twin beside each run.
- The report is rebuilt from `runs/<run>/tables/`, so wording and figures can change without a GPU.

**Not in this stage:** sender gene restricted to one sender type (a flow's own ligands), NicheNet targets instead of receptors (T8), LIANA+ comparison (T10).

