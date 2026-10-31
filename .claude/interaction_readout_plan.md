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
