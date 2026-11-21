# Attribution beyond attention: who influences whom, faithfully (plan, 2026-11-21)

## Goal

Read which cell influences which (and sources / sinks in the tissue) with attributions whose meaning is clear, not only with
attention weights. Three items were never done (listed 2026-11-21, the user: "worth doing now"):

1. **IG on real data**: integrated gradients on sampled hidden receivers of CosMx (and vannan), not only synth_spot.
2. **An attribution-based source / sink map**: per-cell out- and in-strength from attributions (IG on disc targets), drawn like
   the attention source / sink views of `flow_maps`.
3. **Influence against distance** for the trained model, its untrained twin and the expression-permuted section.

Plus what is worth trying from DeepLIFT (Shrikumar et al. 2017) and Ancona et al. 2018 ("Towards better understanding of
gradient-based attribution methods"), alone or with IG.

Constraints from memory: evaluation stays dataset-agnostic (targets, controls and keys from config); never run models locally;
results must come fast (time per step reported); never call what a model "reads" from reconstruction scores alone; the user
wants this for newer models on the more relevant data (the flow pipeline waits for that too, 2026-11-21).

## Status

| Stage | What | Status | Date |
|---|---|---|---|
| A0 | One attribution interface: methods x references, signed, cell- and gene-resolved, completeness gap and time per receiver | proposed | 2026-11-21 |
| A1 | synth_spot validation: planted senders, completeness, method agreement, Sensitivity-n, cost; decides whether DeepLIFT may replace IG for maps | proposed | 2026-11-21 |
| A2 | Item 1: sampled receivers on CosMx / vannan, against the twin and the within-type permuted section | proposed | 2026-11-21 |
| A3 | Item 2: attribution source / sink maps from disc targets | proposed | 2026-11-21 |
| A4 | Item 3: influence against distance (trained, twin, permuted; attention next to it) | proposed | 2026-11-21 |
| A5 | Optional: positive / negative split (RevealCancel idea), multi-reference DeepLIFT (DeepSHAP), conservative transformer propagation | proposed | 2026-11-21 |
| A7 | **Later (user, 2026-11-21):** the attribution methods on data without cell-type annotation (legnini the candidate): what replaces the type-mean reference and the deviation target there | to note, not planned in detail | 2026-11-21 |
| A6 | **Must pursue (user, 2026-11-21):** is the learned neighbourhood function linear? Method agreement, Sensitivity-n and pairwise sender interactions on real data, with synth_spot's thresholded programme as the positive control | proposed | 2026-11-21 |
| M | Retrain one model per dataset with the best settings so far (CosMx, synth_spot, vannan; legnini skipped, user 2026-11-21) | recipes proposed below, to confirm | 2026-11-21 |

## What exists (2026-11-21)

- `evaluation/receptive_field.py`: `_program_attribution` (IG, Gauss-Legendre points, 5 steps, retried at 20 when the completeness
  gap exceeds `gap_limit`), `receiver_dependence` (IG per sampled receiver, `dep_top_cells` = the 32 cells with the largest
  |contribution|, `own` share, completeness `gap`), `response_influence`. **Reference:** the graph's mean cell for every cell at once
  (or one fixed row); for a global-only model the path runs in token space from the graph's mean token. Contributions are signed and
  summed over a cell's genes; the target is a sum of predicted genes of the receiver. Cost: per step one forward and one backward per
  receiver (`retain_graph`).
- Results with it: the planted sender ranks in every synth programme-masking sweep (0.5-0.6 for no-window models, read vs twin);
  2026-10-31 on the windowed Stage B models: IG top 5 holds the planted sender for 99 % (short) / 100 % (mid, M11) of receivers vs
  head-mean attention 97 % / 73 %, dense 9 % vs 1 %, sparse and directional at chance for both. Completeness gap 0.45-0.62 for the
  distance arms' global part (notes.md, 2026-10-2x), against 0.13-0.17 elsewhere.
- Attention side: `flow_maps` (last-layer head-mean raw attention, net flow, source / sink views, whole CosMx sections), `fold_flow`
  (flow from hidden queries), ensembles (`cosmx_ensemble`: reproducible at ~0.9 over 10 seeds, but 1- and 2-layer models flip
  cell-type signs), `flow_domains` (`gradrel` = grad x attention of the masked loss; built, never run).
- Perturbation side: gene-level sender deletion / permute / remove-and-retrain (`ranking`, 2026-11-03), `interpretation.gene_flows`
  (far cells changed, near cells kept). The model's own masking is a removal that needs no retraining.
- Architecture facts that matter: the FFN nonlinearity is ReLU (`transformer_encoder_layer.py`, `F.relu`); the decoder MLP has
  LayerNorm + ReLU; the PE MLP has a GELU; attention is softmax(QK^T + bias) V (multiplicative); the hierarchical routed variants use
  top-k (discontinuous in the inputs); CosMx sections use the fused backend (no explicit maps). Captum is not installed.

## What the two papers say that applies here

1. **Global versus local attributions (Ancona 3.2).** Gradients and saliency describe infinitesimal sensitivity; IG, DeepLIFT,
   occlusion describe the marginal effect of an input against a reference. "Which cell influences which" is the second question.
   Attention weights are neither: they are routing, not an effect on the output. This is the thesis argument for moving the flow
   readouts off raw attention, and it predicts the 10-31 result (attention finds what the distance window makes near, IG finds the
   mid-range sender the window does not favour).

2. **The reference is the real decision (DeepLIFT 3.3; Ancona App. B).** Our IG asks "how does this cell's deviation from the
   section's mean cell change the receiver", for every cell at once. Options:
   - R1 the graph mean cell (current; continuity);
   - **R2 the sender's own type mean** (in the same section): attribution of what a sender expresses *beyond its type*. This is the
     within-type question of the type-residual target, the ceiling audit and the within-type shuffle benchmark, and it removes the
     composition term (which types are around) from the flow by construction;
   - **R4 several references drawn from the data** (DeepLIFT's shuffled references, App. J; Expected Gradients / DeepSHAP average
     the attribution over a reference distribution, my addition): same-type cells from other sections or slides; average over k
     references, report the spread;
   - R3 the model's own absence: the sender hidden (mask fill). Only for occlusion (exact), because a straight path from an
     expression vector to the mask fill passes through inputs the model never saw (Ancona App. B: a baseline must not be a valid
     input that means something else, like a zero-count cell in log1p).
   Proposal: R2 as the default for real data, R4 (k ~ 5-10) as its check, R1 kept for synth continuity.

3. **Completeness as a gate.** IG and DeepLIFT satisfy summation-to-delta; we already report the gap. The 0.45-0.62 gaps on the
   distance arms are a warning that the path integrand is not smooth there (sharp distance biases; top-k routing in hierarchical
   models is discontinuous, DeepLIFT Fig. 2's thresholding problem). Rule: a receiver whose gap stays > 0.1 after the retry is
   dropped from maps and counted; report the dropped share per arm.

4. **DeepLIFT-Rescale as a one-pass IG (Ancona Prop. 2 and point 4).** DeepLIFT-Rescale = (x - x_ref) times a modified gradient in
   which every elementwise nonlinearity's derivative is replaced by its secant (f(z) - f(z_ref)) / (z - z_ref) (the derivative where
   z = z_ref). In PyTorch: one forward on the reference to record z_ref at each ReLU / GELU, then a custom autograd function in place
   of `F.relu` / `nn.ReLU` / `nn.GELU`; linear layers, softmax and LayerNorm keep their gradients (as the public DeepLIFT does for
   multi-input operations). Cost: 2 forwards + 1 backward per receiver instead of 5-20 of each: the factor that makes dense maps
   affordable. **Caveat that decides it (Ancona 3.1, the LSTM case):** with multiplicative interactions DeepLIFT loses completeness and
   stops approximating IG. A transformer is multiplicative (q.k, attention x value, LayerNorm's division). So it is a candidate to
   *measure*, not to assume: A1 compares it with 20-step IG on synth_spot. (Gradient x input = epsilon-LRP for an all-ReLU network,
   Ancona Prop. 3; with LayerNorm and softmax in between it is not.)

5. **Sensitivity-n: an evaluation that needs no ground truth (Ancona 4.1).** For a receiver, hide random subsets of n sender cells,
   measure the change of the receiver's target, and correlate it with the sum of the attributions of those cells, for n = 1 ... ~100.
   Our models were trained with masking, so "hide" is the model's own removal: App. B's problem (no way to remove a feature without
   retraining) is solved here by construction. This is the faithfulness test for CosMx, where no planted truth exists, and it ranks
   readouts (attention, gradient x input, IG, DeepLIFT, occlusion) on the same footing. Occlusion-1 is Sensitivity-1 by construction;
   IG / DeepLIFT should win at larger n (joint effects); Ancona Fig. 3-4.

6. **Agreement between methods measures nonlinearity (Ancona Prop. 4).** All of these attributions coincide when the model is linear
   in the task. Our models tie ridges and k-NN smoothers on CosMx; if gradient x input, IG, DeepLIFT and occlusion agree there, the
   model uses its inputs almost linearly and the cheapest readout suffices, and that is itself a finding (the ceiling audit's "the
   neighbourhood term is linear" from the model side). Disagreement localises where the nonlinearity is used.

7. **Occlusion misses redundancy (DeepLIFT Fig. 1, saturation; Ancona Fig. 3).** Hiding one sender whose neighbours carry the same
   smooth signal changes little: the 11-03 sender deletion had the same failure for smooth senders ("the other genes rebuild them").
   So occlusion understates smooth fields and IG / DeepLIFT are needed for them; occlusion is still the exact single-cell effect.
   Cost note: hiding sender j changes every receiver at once, so one forward per sender gives a whole column of the cell x cell
   effect matrix (N forwards per section: 2,501 on a synth slide, 5-10 k on a CosMx section; measure the time per forward first).

8. **Positive and negative contributions (DeepLIFT 3.4-3.5.3).** Keep signs: the current top-32 by |contribution| mixes senders that
   raise and lower a receiver's gene. Maps per sign ("activating" vs "repressing" senders) are a cheap win. RevealCancel finds AND-type
   cooperativity that IG and Rescale miss (TAL1-GATA1, 8-13 % misses, Fig. 5b); our sparse programmes are AND-type (ligand in the
   sender x receptor in the receiver within a distance) and no readout found them. But no model learned them either (scores at
   ref_own), and RevealCancel needs custom rules per layer, not a gradient override. Only if a model learns a sparse programme.

9. **Transformer-specific conservative propagation (my addition, not in the two papers):** LRP for transformers (Ali et al. 2022,
   "XAI for Transformers: better explanations through conservative propagation"; AttnLRP, Achtibat et al. 2024) treats the attention
   weights and the LayerNorm scale as constants in the backward pass, which restores conservation through the attention block.
   Implemented as gradient x input with the softmax output and the LN denominator detached. It needs explicit maps (dense / vendored
   attention, not fused). Cheap (one backward), sits between raw attention and IG; A1 can include it at little cost.

## Design

### A0: one interface

`evaluation/attribution.py` (new; `receptive_field` keeps working and later calls it):

    attribute(module, batch, receivers, target, method, reference, *, steps, gene_resolved, model_heads) -> Attribution
    Attribution: contribution [Q, N] (or [Q, N, G]), change [Q], gap [Q], seconds, method, reference

- methods: `attention` (last layer, head mean; a routing weight, not an effect: kept for comparison), `grad_input`, `ig`
  (the existing Gauss-Legendre integrator, refactored), `deeplift` (Rescale by secant gradient override at ReLU / GELU; reference
  forward cached), `conservative` (gradient x input with softmax output and LN scale detached; dense attention only), `occlusion`
  (hide one sender per forward; columns of the effect matrix; `senders` subset to bound cost).
- references: `graph_mean` (R1), `type_mean` (R2; `label_key` from config), `sampled` (R4; k same-type cells from other graphs,
  seeded, averaged; spread reported), `mask` (R3; occlusion only).
- targets: a receiver's predicted genes (a set, summed), a named gene, or the **deviation direction**: the receiver's predicted
  profile minus its type mean, normalised, as fixed weights over genes ("what moved this cell off its type"; linear, so one backward
  per step). Disc targets for A3 = the sum over a disc's hidden receivers (IG is linear in the target).
- Every call reports seconds per receiver; the hidden receivers' own visible entries do not move (as now).

### A1: synth_spot validation (decides the method for maps)

Checkpoints: windowed global-only `lin_dist_m11_skipraw` (3 seeds + twin; may need retraining: the flow pipeline did not find them
on 2026-11-21) and one no-window arm. Programme masks. Per method x reference:

- planted sender in the top 5 of its receivers, by class (short, mid, dense, sparse, directional), as 2026-10-31; trained vs twin;
- completeness gap distribution (IG at 5 / 20 / 50 steps, DeepLIFT, conservative);
- agreement with 50-step IG per receiver (Spearman over senders; top-5 overlap);
- Sensitivity-n curves (n = 1, 2, 5, 10, 20, 50, 100; 50 random subsets per receiver; hide = mask);
- seconds per receiver.

**Decision rule, fixed before the run:** DeepLIFT (or `conservative`) may replace IG for dense maps if its median gap < 0.1, its
median Spearman with 50-step IG > 0.9, and its planted top-5 share within 0.05 of IG's for short and mid. Otherwise maps use IG on
sampled receivers and disc targets only.

### A2: sampled receivers on real data (item 1)

CosMx (then vannan; dataset-agnostic): ~500-1,000 hidden receivers per test section (stratified by type), deviation-direction
targets plus named genes (CosMx INS / GCG / HSPA1A, vannan IFIT1). Reference R2, check R4. Controls: the
untrained twin, the within-type permuted section (expression shuffled among same-type cells, positions kept: the shuffle benchmark's
null), Sensitivity-n. Outputs: per receiver the signed contributions over senders (sparse, top 64 + the rest's sum), own share, gap.
Readouts: receivers' dependence on others vs own cell (the "context is tiny" question from the model side), sender types per
receiver type (signed, vs twin and permuted), agreement between methods (Prop. 4 diagnostic). Needs gradients through the fused
backend (training does it); `conservative` not available there.

### A3: attribution source / sink map (item 2)

Disc targets: for each hidden disc, the target is the sum of its receivers' deviation-direction targets; one IG (or DeepLIFT) per
disc gives every sender's contribution to that disc. Over the fold passes (every cell hidden once), out-strength of cell j = sum over
discs of its contribution (signed and absolute), in-strength = the discs' change attributed to cells outside them. Map: cells
coloured by smoothed out - in (diverging), positive and negative senders separately, same layout as `flow_maps` (one slide per
figure, tissue only); against the twin and the permuted section. Gene-resolved option: the sender's contribution split over its
genes gives "which genes in the senders drive" per region. Caveat to state in the read: hidden cells are receivers by construction,
so in-strength follows the mask layout; compare across mask draws.

### A4: influence against distance (item 3)

From A2 (and A3) contributions: |contribution| and signed contribution per sender-receiver distance bin, per cell in the bin and
cumulative share, for trained, twin and permuted; the attention mass by distance on the same receivers next to it; on synth_spot the
planted range per class as reference lines. One figure per dataset, the bins from the data's spacing (CosMx ~10 um cells, synth 200 um
spots).

### A5: optional

Positive / negative split everywhere (cheap; part of A0); multi-reference DeepLIFT (DeepSHAP-like, R4 x DeepLIFT); RevealCancel only
if a model learns a sparse programme; Sensitivity-n for the gene-level readouts (`ranking`).

## Cost (to measure in A1 before any real-data run)

IG: steps x (1 forward + Q backward) per graph; DeepLIFT / conservative / gradient x input: ~2 forwards + Q backward; occlusion:
one forward per sender (whole column). For a CosMx section with ~1,000 receivers: IG at 5 steps = 5,000 backward passes through the
section, DeepLIFT ~1,000. Measure seconds per backward on one section before writing the jobs (cluster jobs must finish: < 12 h,
few arms).

## Decisions (user, 2026-11-21)

- **References:** R2 (the sender's type mean) is the default, R4 (sampled same-type cells) the check. synth_spot: R1 for continuity with the 10-31 read, plus R2 if a type label is used.
- **Models:** retrain one model per dataset with the best settings found so far (stage M below), not old checkpoints.
- **Order of datasets:** CosMx, synth_spot, vannan. **legnini skipped** (user, 2026-11-21): it has no cell types in our pipeline
  (every score there is within the organoid), so the type-mean reference and the deviation-from-type target do not apply; it moves
  to A7 (data without cell-type annotation, later). IG itself is already validated on synth_spot (2026-10-31), so CosMx can
  start with IG + Sensitivity-n; DeepLIFT / `conservative` enter the CosMx maps only after A1's rule passes on synth_spot (A1 can run
  in parallel; it does not hold up the IG read).
- **Targets:** each receiver's predicted deviation from its type mean, plus named genes: CosMx INS, vannan IFIT1; synth_spot one medium and one hard planted programme (below).

### synth_spot targets (medium and hard)

- **Medium: M11 -> MC14 / MC15** (`mid`, single sources, response within 500 um). Learned at 300 epochs (0.955, sender rank 1.00;
  at 100 epochs only ~0.25). The readouts disagree here: head-mean attention top 5 holds the sender for 73 % of receivers, IG for
  100 % (2026-10-31). Tests whether a method finds a sender the distance window does not make near.
- **Hard: M19 -> MC23** (sparse_response, `_SparseProgramme` (viii): 1 sender per site, 200-500 receivers per slide, response only
  within a 2,500 um threshold: an AND of the sender's ligand and the distance cut, i.e. DeepLIFT Fig. 2's thresholding). Learned
  at 300 epochs (the frequent pair rose 0.65 / 0.55 -> 0.94 / 0.90, distance_window memory; check which numbers are MC23 when
  reading). IG found the dense class at 9 % (3x chance) on the 100-epoch Stage B models only.
- **Its control: M16 -> MC20**, the identical rule with 20-50 receivers per slide, which no model learned (flat at 300 epochs). An
  attribution method must find M19 for MC23 and must *not* find M16 for MC20 above the twin: what the model did not learn cannot be
  attributed. Same pair for A6: the learned thresholded rule is where the methods should disagree.

### Stage M: what "best settings" means per dataset (my proposal from the reads; confirm before the job)

- **CosMx:** the `typeres` arm of `cosmx_typeres` (hier_tier, whole-cell masking 20 % + 60 um discs, FOV-shuffled split 0, target =
  expression minus the section x cell_type_coarse mean, 2 layers, 300 epochs): the only arm at the bar within section x type (0.087
  +- 0.002 vs knn128 0.083, 3 seeds). Its target is the deviation this plan attributes, and the 2-layer map is learned (twin overlap
  0.01) where the 1-layer map is geometry. Option: 8 heads (the capacity read found h8 the cheap gain, but on synth_spot genes, not
  CosMx). 3 seeds + twin (per-cell flow is not seed-stable: read sender types and maps over seeds).
- **legnini:** skipped for now (A7).
- **synth_spot:** `lin_dist_skipraw` (GlobalModel, `pe.distance.kind: scales`, raw own-cell skip; 0.909 over 3 seeds) at **300 epochs**
  (solves M11 and the frequent sparse pair), optionally 8 heads; programme masks; 3 seeds + twin. Dense attention: every method runs.
- **vannan:** never trained. `lung_hier.yaml` (lung_plan.md L0: 40-NN, discs 200 um, patient folds) with the CosMx typeres target by
  sample x final_CT, 300 epochs; fold 0 to match the fields audit.

**2026-11-21, later: Stage M waits for the model decision in `ccc_model_plan.md`** (the user asked to rethink the model choices from
the CCC papers before building this plan). If its synth_spot gate (J1, rules R1-R2) passes, the interaction read-out replaces the
recipes above (CosMx: type residual with all-cell targets; synth_spot: IR with the receiver's visible genes in the query); A0 gains
the exact `decomposition` method, and A1 runs on J1's checkpoints. If it fails, the recipes above stand.

## A7: attribution without cell-type annotation (later; user, 2026-11-21)

The user: "in the end it would be good to have these attribution methods tested on data without cell type annotation too". The plan
leans on types twice: the reference R2 (the sender's type mean) and the receivers' target (predicted deviation from the type mean).
On data without types both need a replacement, and which one is the open question of this stage:

- references: the graph's mean cell (R1, as the existing IG), a local mean (the sender's k-NN neighbourhood in expression space, so the
  reference is "a cell like this one" without a label), unsupervised clusters (Leiden on the visible genes, fitted on training graphs)
  as stand-in types, or sampled cells from the whole graph (R4 without the type restriction);
- targets: deviation from the graph mean or from the cluster mean;
- what to compare: on CosMx / vannan, the type-based run against the same run with clusters or k-NN references instead of the labels:
  do sender rankings, maps and the A6 linearity read survive losing the annotation?
- the candidate dataset is legnini (SHH organoids; the SHH readouts GLI1 / PTCH1 / HHIP in SHH vs Ctrl organoids as named targets,
  the organoid's mean cell as the reference to start from).

Not designed further; it starts once A2-A4 have run on CosMx.

## A6: is the learned neighbourhood function linear? (must pursue, user 2026-11-21)

**Why it matters.** Three questions about CosMx have been answered at two levels so far: the score level (models tie the linear k-NN
smoother) and the data level (the ceiling audit: the neighbourhood's linear signal is small and within ~100 um, and no linear reader
finds more). Neither says what the trained transformer *computes*: the same score can come from a nonlinear function. Ancona
Prop. 4 gives the model-level test: attributions of different kinds coincide, and Sensitivity-n holds for every n, only where the
model is (locally) linear in its inputs. If on CosMx the methods agree and Sensitivity-n stays high up to large n, while on synth_spot
they part exactly on the learned thresholded programme (MC23), the claim is: *the transformer can learn nonlinear cell-cell rules
(shown where they exist), and on CosMx it uses its neighbourhood as an approximately additive, linear function*: a third,
independent line for "the learnable cell-cell signal in this tissue is linear and short-range", from inside the model.

**Novelty: not established.** Prop. 4 is known; using method agreement / Sensitivity-n as a diagnostic of what spatial cell-cell
models learn is not something I know of, but no literature check has been done. Related prior work to check first: NCEM (Fischer et
al., theislab), which compared linear and nonlinear node-centric expression models on spatial data at the score level (to verify
what they found before citing it), and the attribution / LRP-for-transformers literature. Do the check before calling it novel.

**Measures.**
- Agreement between methods per receiver (Spearman over senders, top-k overlap): gradient x input, IG (R2 and R4), DeepLIFT,
  occlusion; on the context part only.
- Sensitivity-n curves (hide n random senders with the model's mask; n = 1 ... 100): a linear model keeps the correlation at all n.
- **Pairwise interaction of senders** (the quantity in Ancona's proof): I_jk = Delta_jk - Delta_j - Delta_k, the change when hiding
  j and k together minus the two single changes, for near sender pairs of a receiver; its size relative to Delta_j + Delta_k, as a
  distribution over receivers. Direct and method-free: zero for an additive function.
- All against the untrained twin and on synth_spot (MC23 learned, MC20 not, M11 mid) as positive and negative controls.

**Traps, to state with any result.**
- Trivial agreement: if the context contributes almost nothing (CosMx "context is tiny"), all methods agree because all are ~0.
  Report the context share, restrict to receivers whose context part is beyond the twin, and compare with synth_spot where the
  context part is large.
- Small deviations look linear: with the type-mean reference the deltas are within-type residuals, and any smooth function is linear
  to first order over a small range. So "linear over the variation the data has" is the claim, not "linear everywhere"; the
  sampled references (R4) and whole-cell hiding (large perturbations) probe beyond the first-order range.
- One model per dataset: repeat over the 3 seeds; agreement must hold for each.

**Where it goes:** thesis 4.7 (CosMx) and 5.2 (what the transformer can read), next to the ceiling audit; slot S13 in
thesis_outline.md.
