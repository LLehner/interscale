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
| A0 | One attribution interface: methods x references, signed, cell- and gene-resolved, completeness gap and time per receiver | **built** (`evaluation/attribution.py`: decomposition, attention, grad_input, ig, deeplift, conservative, occlusion x R1 / R2 / R4; removal by reference or by the model's own mask; tests) | 2026-11-22 |
| A1 | synth_spot validation: planted senders, completeness, method agreement, Sensitivity-n, cost; decides whether DeepLIFT may replace IG for maps | job written and checked (`attr_submit.sh synth`, J1's ref + ir2_rate), **not run** | 2026-11-22 |
| A2 | Item 1: sampled receivers on CosMx / vannan, against the twin and the within-type permuted section | CosMx and vannan jobs written and checked (`attr_submit.sh cosmx` on J4; `attr_submit.sh vannan` on `ir_submit.sh vannan`'s new checkpoints), **not run** | 2026-11-22 |
| A3 | Item 2: attribution source / sink maps from disc targets | in the CosMx job (disc targets, `plot_source_sink` layout), **not run** | 2026-11-22 |
| A4 | Item 3: influence against distance (trained, twin, permuted; attention next to it) | in both jobs (`attr_distance`), **not run** | 2026-11-22 |
| A5 | Optional: positive / negative split (RevealCancel idea), multi-reference DeepLIFT (DeepSHAP), conservative transformer propagation | signs kept everywhere; `deeplift` with `reference="sampled"` is the DeepSHAP-like average; `conservative` built; RevealCancel not built (only if a model learns an AND-type programme) | 2026-11-22 |
| A7 | **Later (user, 2026-11-21):** the attribution methods on data without cell-type annotation (legnini the candidate): what replaces the type-mean reference and the deviation target there | to note, not planned in detail | 2026-11-21 |
| A6 | **Must pursue (user, 2026-11-21):** is the learned neighbourhood function linear? Method agreement, Sensitivity-n and pairwise sender interactions on real data, with synth_spot's thresholded programme as the positive control | measures built (`sensitivity_n`, `pairwise_interactions`, `agreement`) with small and large perturbations (type mean, sampled cells, the model's mask), in all jobs (MC23 / MC20 / M11 named on synth); run 2026-11-24/25 (read: synth control weak, typeres ~4x twin); **S13 built 2026-11-27**: planted-AND control (`attr_submit.sh and`, trains + reads), CosMx read-out twins + sections 4-6 (`attr_submit.sh cosmx_s13`), NCEM check done (section "S13 build 2026-11-27"); weekend build 2026-11-28 (frozen attention, AND-capable arms, paired rule v2; jobs and_eval / and_mlp / and_strong not run); run + read 2026-11-27 (notes.md): AND control NOT passed, no model learned the AND (index 0.43-0.45 vs 1) and the additive R6 reads as large as R5 (0.17 / 0.17); instrument unvalidated; cosmx_s13 read 2026-11-27 (read-out arms additive by construction, 0.006-0.010 over 3 seeds, singles not tiny; typeres 0.078 = 4.3x twin, not separable from the floor) | 2026-11-27 |
| M | Retrain one model per dataset with the best settings so far (CosMx, synth_spot, vannan; legnini skipped, user 2026-11-21) | recipes proposed below, to confirm | 2026-11-21 |
| check | Status check after the CCC runs (J1 / J3 / J4 read; section "Status check 2026-11-22" at the end): nothing of A0-A6 built; Stage M answered by J1 / J4 for synth_spot and CosMx with checkpoints that exist (no retraining), vannan untrained; A6 has its first architecture-level evidence | **done**; building waits for the user | 2026-11-22 |

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

## Status check 2026-11-22 (after the CCC runs; the user asked whether the files are ready or something still needs building)

### What exists

- **IG** (`receptive_field._program_attribution`, `receiver_dependence`, `response_influence`): Gauss-Legendre points, retry on a large completeness gap,
  reference = the graph's mean cell (R1) or a fixed row; summed over a target gene set. Validated on synth_spot (2026-10-31). It is what A0 refactors.
- **The exact decomposition** of the interaction read-out (`interaction_readout.influence`, `ReadoutPass`): per receiver every sender's contribution
  through the linear decoder, top-k senders, null mass, gene-resolved. This is A0's `decomposition` method, already built (ccc_model_plan.md, S0); on J1 it
  found the planted short / mid senders for 1.00 / 0.99-1.00 of receivers (head-mean attention 0.97 / 0.73, IG 0.99 / 1.00), without a gradient.
- **Interventions** (`interpretation.type_flows`, `gene_flows`, and E7's `gene_ranking.context_flows`): type and gene removals with location-matched nulls,
  receivers' own rows kept, pooled over sections. The within-type permuted section of A2's control is `context_flows`' `all_state` kind.
- **Gene rankings** (E7, 2026-11-22): the gene-resolved part of A3 ("which genes in the senders drive") by permutation, with truth and bars; jobs written.
- **Checkpoints** on the cluster: J1 (synth_spot, 300 epochs: `lin_dist_skipraw` as `ref` x 2 seeds, `ir2` / `ir2_rate` / `ir2_kd` / `ir1` x 2), J4 (CosMx:
  typeres, ir_c, ir_a, ir_b, ir_c_ktype, ir_c_knone x 3 seeds, the single-seed options), J3 (spiked CosMx). Untrained twins are built at evaluation from the
  config and seed (`EVAL_UNTRAINED`), so no twin checkpoint is needed.

### Not built (everything else of the plan)

| Stage | Missing | Size |
| --- | --- | --- |
| A0 | `evaluation/attribution.py`: one interface (methods `attention`, `grad_input`, `ig` (refactored from receptive_field), `deeplift` (secant override at ReLU / GELU), `conservative` (softmax and LN scale detached; dense only), `occlusion` (one forward per sender: a column of the effect matrix), `decomposition` (wrap `interaction_readout.influence`)); references R1 / R2 (type mean) / R4 (sampled same-type cells) / R3 (mask, occlusion only); targets incl. the deviation direction; seconds and gap per receiver; tests on a module of known dependence | ~1.5 days |
| A1 | synth_spot validation job on J1's checkpoints: planted top-5 by class per method x reference, gap distributions, agreement with 50-step IG, Sensitivity-n (hide = the model's mask), seconds; the DeepLIFT decision rule | ~0.5 day + one job |
| A2 | CosMx sampled receivers (500-1,000 per test section, stratified by type), R2 + R4, deviation targets + INS; twin, within-type permuted section, Sensitivity-n; gradients through the fused backend (training does it; `conservative` not available there) | ~0.5 day + one job |
| A3 | disc-target attributions over the fold passes, out / in strength per cell, maps in `flow_maps`' layout (one slide per figure, tissue only), twin and permuted section | ~0.5 day |
| A4 | influence against distance from A2 / A3 (trained, twin, permuted; attention mass beside it) | ~0.25 day |
| A6 | pairwise sender interaction I_jk (hide j, k, both; the model's own removal), method agreement, Sensitivity-n on CosMx with synth_spot's controls (MC23 learned, MC20 not, M11) | ~0.5 day (on A0) |
| vannan | `vannan25_pp.h5ad` exists on the cluster (ceiling run 4 used it: 34,134 test cells, 8 samples; corrected 2026-11-22); no trained model | one training job (Stage M recipe) |

### What the CCC runs settle

- **Stage M is answered for two datasets without retraining** (the plan's outcome rules for J1 and J4): synth_spot keeps `lin_dist_skipraw` for scores
  (J1 `ref`, 300 epochs, 2 seeds; the plan asked for 3) and adds `ir2_rate` as the per-sender readout model (exact decomposition, R2 passed); CosMx moves
  from typeres to the read-out in mode (c), `ir_c_ktype` the candidate (3 seeds: within section x type 0.141 clean / 0.098 masked, raw SCE 0.436, the most
  seed-stable maps). vannan has nothing. **To confirm by the user** (the plan's rule: confirm before the job). If the E7 rankings (J5 / J6) disagree with
  this choice -- e.g. ctx_ch ranks interacting genes better than ir_ch on synth_spot -- that should weigh in before A2.
- **A6, first evidence at the level of the architecture** (not yet the attribution test): on CosMx a model whose output is a sum over senders with routing
  by distance and sender type only (`ir_c_ktype`, `ir_c_knone`) scores at least as well as one that reads sender expression in its keys (0.141 / 0.138 vs
  0.137) and as typeres (masked 0.098 vs 0.092): nothing on CosMx shows that a nonlinear use of the neighbourhood helps. On synth_spot the same additive model
  loses exactly the thresholded programmes (M19-M21; within receivers -0.13 / -0.44 / 0.63 vs the reference's 0.54 / 0.57 / 0.90): the positive control
  that the nonlinearity matters where the data has it. A6's attribution measures (method agreement, Sensitivity-n, I_jk) are still needed to say it from
  inside one trained model; the synth controls in the plan stay (MC23 learned by the reference, not by the IR).
- **A0's reference question is partly moot for the read-out**: its decomposition needs no reference; IG / DeepLIFT still need R2 / R4 for the
  non-decomposable models (typeres, the reference).

### Recommended order (nothing started)

1. The user confirms Stage M as above (or waits for J5 / J6's rankings).
2. A0 with `decomposition`, `ig`, `grad_input`, `occlusion` first (DeepLIFT and `conservative` after), tests.
3. A1 on J1's checkpoints and A2 on J4's in one job each (A1 does not block A2's IG read, as decided 2026-11-21); A4 from A2's outputs; A6 on A0.
4. A3 maps; vannan preprocessing and training when the user returns to it.

## Build 2026-11-22 (the user: "while we wait for the evaluation run can you start implementing stage A0-A6?")

### Library: `evaluation/attribution.py`

- `attribute(module, batch, receivers, weights, method, reference=...)`: one call for every method; the path runs in **token space** (what the model reads;
  for the linear token embedding the same as expression space). Targets are fixed gene weights per receiver: `deviation_targets` (predicted deviation from
  the type mean, unit length), `gene_targets` (named genes), or a programme's genes. **Fixed rows:** the receivers' own rows and cells hidden whole never
  move, so a receiver's prediction is attributed to other cells only, for models that read their own cell too (the J1 reference).
- Methods: `decomposition` (the read-out's exact share; the intercept is the decoder bias plus each read layer's output-projection bias through the
  decoder, which the first version missed: the toy test caught a gap of 0.3-0.6, now < 1e-4), `attention` (head- and layer-mean read-out weight),
  `grad_input`, `ig` (Gauss-Legendre, retry on gap), `deeplift` (Rescale: the layers' stored activation functions and the `nn.ReLU` / `nn.GELU` modules
  swapped for a secant-gradient function during the pass and restored after; softmax and LayerNorm keep their gradients, so completeness is measured, not
  assumed), `occlusion` (one forward per sender gives a column for all receivers; restricted to each receiver's nearby cells and planted senders).
- References: `graph_mean` (R1), `type_mean` (R2), `sampled` (R4: a random other cell of the same label, `draws` times averaged, spread kept).
- Measures: `planted_top_k`, `agreement` (Spearman over the union of the two methods' top cells, top-k overlap), `sensitivity_n` and
  `pairwise_interactions` (many receivers per forward: receivers chosen far apart by `separated_receivers`, each perturbing only its own nearby cells;
  cross-talk between distant receivers is the stated approximation), `distance_profile`, `strengths`, `disc_strengths` (A3: by linearity in the target a
  disc's attribution is the sum of its receivers' rows), `map_fields` (the fields `flow_maps.plot_source_sink` draws).
- Tests (`tests/test_attribution_methods.py`, real GlobalModels at toy size): the decomposition sums to the context part; IG's gap shrinks with steps; module mode,
  parameter flags, the graph and the global generators left as found; hidden cells never move; occlusion = the change when one sender takes its reference,
  and with one receiver Sensitivity-1 against occlusion is exactly 1; the secant gradient is the secant and the activations are restored; references and
  measures by hand.

### Jobs (`slurm/November/`, flat; evaluation only, no training; results `MA/22_11_results/attr_{synth,cosmx}/attr_read.md`)

| Job | Checkpoints | What |
| --- | --- | --- |
| `synth` (A1, A4, A6 controls) | J1: `ref` (lin_dist_skipraw), `ir2_rate` x 2 seeds + twins | programme masks as trained; 6 planted receivers per programme and graph, target = its response genes; decomposition, attention, grad_input, IG 5 / 20 / 50 (the 50-step IG is the reference), DeepLIFT, occlusion; R1; planted top-5 by class, gaps, agreement, seconds, distance; A6 on receivers 600 um apart, grouped MC23 (threshold, learned by the reference), MC20 (unlearned), M11 (mid), classes |
| `cosmx` (A2-A4, A6, A3) | J4: `ir_c_ktype` x 3, `typeres_s42`, `ir_c_s42` + twins of typeres / ir_c | clean pass; up to 40 receivers per type 150 um apart; deviation targets (INS for Beta cells); R2 for every method, R4 for IG; the within-type permuted section (non-receivers' rows swapped within type); sender type x receiver type, distance, A6; A3 disc maps (30 discs of 30 um per section, IG, receivers with gap > 0.1 dropped) |

Files: `attr_job.py` (`ATTR_JOB=synth|cosmx`, every setting from ir_synth.py / ir_cosmx.py so the data and checkpoints match), `attr_common.py` (the hook),
`attr_read.py` (the A1 rule; the A6 tables beside the twins; uses gene_rank_read.py's helpers), `attr_job.sbatch`, `attr_submit.sh`. Copy them with
`gene_rank_read.py` to `MA/scripts/` (sbatch and submit to `MA/slurm/`), install the fork's main, then `bash slurm/attr_submit.sh synth` (and `cosmx`).

### Checks run (no model on real data)

The hook end to end on real GlobalModels at toy size on the CPU: a full-encoder reference and a read-out under programme masks with planted edges (synth
path), a mode-c read-out with type keys and a typeres-like fused model on the clean pass (CosMx path), both readers on the tables; planted senders map to
the right nodes. The A3 map fields drawn by `plot_source_sink` on a 3,000-cell synthetic section (the toy sections are below its 100-cell tissue block,
so the toy maps are skipped with a message, by design). Found and fixed: the decomposition's missing output-projection bias; type tables keyed by
`Attribution.method` (all IG variants are "ig") instead of their names.

### Not built / open

- `conservative` (softmax and LN scale detached) and RevealCancel (A5). Correction (2026-11-22, the user): `conservative` needs explicit attention weights, not dense attention; the read-out and fused backends have that path (`need_weights`), and the CosMx test graphs (~10k cells) fit an explicit map (~0.4 GB per head and layer), so it can cover every J4 arm. Only whole 100k-cell sections would be heavy (~40 GB per head); fused attention's measured cost there was compute (127 s per training step), not memory (5.6 GiB).
- vannan: training (A2's second dataset); the preprocessed file exists.
- A7 (data without labels): R1 and the k-NN / cluster references are possible with `labels` set to clusters; not wired into a job.
- Cost is estimated, not measured: CosMx ~5,000 backward passes per section per model (IG 5 steps, R4, permuted, discs), at an assumed 0.1 s each about
  1 h per model and twin; the job reports seconds per receiver and per run.
- Before calling A6 novel: the NCEM check (linear vs nonlinear node-centric models) remains to do.

### Build, second part (2026-11-22; the user: dense attention is not the limit, vannan's preprocessed file exists)

- **`conservative`** (`_conservative_rules`): during the pass `torch.softmax`, `F.softmax`, `Tensor.softmax` (and `log_softmax`) return detached outputs,
  the SDPA kernel runs in explicit form with detached weights, `F.layer_norm` divides by a detached standard deviation; forward values unchanged (tested on
  a read-out and a dense model), every function restored. Covers dense, fused / read-out and hierarchical (Tensor.softmax) models.
- **Removal by the model's own mask** (`removal="mask"` in `attribute(..., "occlusion")`, `sensitivity_n`, `pairwise_interactions`): the cells hidden whole
  through `mask` (under `visible_keys_only` no key); node-masked models only (refused otherwise). Tested: the column equals the prediction with that cell
  hidden; Sensitivity-1 against mask occlusion is 1.
- **A6 with large perturbations**: the jobs run every A6 measure for `(type_mean, reference)`, `(sampled, reference)` and `(type_mean, mask)` (synth: graph
  mean and sampled; no mask, the J1 models are gene-masked); tables carry `variant`.
- **vannan**: `make_ir_vannan_yaml.py` + `ir_vannan.py` (training only; `bash slurm/ir_submit.sh vannan`): J4's `typeres` and `ir_c_ktype` with fine types
  (the type residual by sample x final_CT), fold `split_0`, graphs per sample, 2 crops per sample and epoch, 3 seeds; controls among the genes dropped.
  `attr_submit.sh vannan` (ir_c_ktype x 3, typeres_s42; IFIT1 as named target, 15 receivers per fine type) and `rank_submit.sh vannan` read them.
- **Renamed** `attr.py` -> `attr_job.py`: a script named `attr` in the scripts folder shadows the `attrs` package that `rich` (scvi-tools) imports as
  `attr`, so every job started from there would have crashed at import. Found by the configure check.
- Checks: vannan arms through `_validate` / masking / names (<= 217 characters); `configure()` and run lists for `attr_job.py` / `gene_rank.py` with vannan;
  the attribution hook end to end again (conservative everywhere, three variants on node-masked models, two on gene-masked ones); 90 tests pass.

## First run read (2026-11-24; results MA/22_11_results/attr_{synth,cosmx,vannan}, logs MA/results/logs)

### What ran and what did not

| Job | Arms | Outcome | Cause |
| --- | --- | --- | --- |
| cosmx (43841210, 5 tasks) | typeres_s42 + twin | done (12,448 s + 7,498 s in one task) | |
| cosmx | ir_c_s42, ir_c_ktype_s42/43/44 | crashed in the first section | `conservative` x activation checkpointing (below) |
| synth | J1 ref, ir2_rate | no tables, no log in the folder | unknown: need the .out/.err |
| vannan (43840145, 43841271; 4 tasks each) | typeres_s42, ir_c_ktype x3 | all cancelled at the 11 h limit; one A3 map (typeres graph0) | whole samples (1.63 M cells in 45 samples) x 15 receivers per fine type x 7 methods + A6: far past 11 h at CosMx's measured rate |

**The crash (every read-out arm).** `ReadoutAttention` / fused attention run in query blocks and wrap each block in `torch.utils.checkpoint`
when `torch.is_grad_enabled() and len(starts) > 1` (attention.py:803 / :990). `_one_reference` applies `_conservative_rules()` around the forward
only and calls `_grad_rows` (the backward) after leaving the context, so the blocks are recomputed with the real SDPA kernel while the forward had
saved the explicit-form tensors: `CheckpointError: ... saved [4, 22, 9388], recomputed [4, 9388, 16]`. The toy tests have one block (no
checkpoint), so they never reached this path. typeres survived because the hierarchy checkpoints only in training. grad_input and IG are unaffected
(no patch); DeepLIFT patches activations only, and the checkpointed blocks hold only SDPA / softmax, so it is unaffected too. Fix (not applied): take
the gradients inside the patch context (`with rules: value = reader(leaf); contribution = _grad_rows(...)`, the same for `_secant_activations`), or
switch recompute off while attributing; add a test with `block_rows` forced small.

**vannan.** No progress lines inside the hook, so the time per sample is unknown; the only output is one A3 map. To fit 11 h: tiles (as the
comparison does on CosMx) or `MAX_GRAPHS`, coarse types or fewer receivers per fine type, IG + occlusion only for a first pass, one arm per task;
measure on one sample first. The report step also needs vannan in `gene_rank_read.py`'s choices (it rejects `vannan`; attr_read accepts it).

**Reader.** (Corrected the same day: the pairwise table is complete; my `grep -v` that hid the Sensitivity-n rows hid it too, since its first
column is the variant.) The A3 figure's caption is the attention flow plot's ("net attention flow, last layer, head mean"); it should name IG
disc targets.

**Receivers.** `separated_receivers` at 150 um gives ~42 receivers per section (2-11 per type and section), not up to 40 per type: the
type-level tables rest on ~250 receivers per arm.

### typeres on CosMx (the only arm read; one seed; 6 test sections)

- **Completeness (A1 / A6).** IG at 5 steps: median gap 0.13 (19 % of receivers > 0.1) trained, 0.007 twin. grad_input / DeepLIFT /
  conservative: median gaps 0.9-1.5 trained (0.15-0.9 twin). The trained model is far from linear on the path from the type mean to the actual
  neighbours; the twin is nearly linear. Agreement with IG5 (Spearman over senders): occlusion 0.65, the gradient methods 0.45-0.51, IG with
  sampled references 0.53 (twin 0.45-0.82). No cheap method can stand in for IG on this model.
- **Sensitivity-n.** With type-mean replacement IG predicts the effect of removing 1-16 senders well (median r 0.73-0.95; occlusion 0.89-0.99,
  trivially at n = 1). With removal by the model's own mask the trained model drops (IG 0.19-0.71; twin 0.75-0.95): hiding a cell is not the same
  intervention as giving it its type mean, for the trained model only.
- **Pairwise sender interactions.** Median |I| / (|D_j| + |D_k|) 0.08 trained vs 0.02 twin; 24-29 % of pairs above 0.2 (twin 11-17 %); highest for
  Beta and Fibroblast receivers (0.11). Same under all three variants. Small at the median, not zero: mostly additive near senders with a tail of
  interacting pairs.
- **A2 sender types.** Signed share of each receiver's |contribution|, trained minus within-type permuted: Acinar senders lead for almost every
  receiver (Acinar 0.23, Endothelial 0.15, Ductal 0.14, Fibroblasts 0.14, Mast 0.13, Immune 0.11); the permuted section's shares are ~0 (they
  cancel). The real Acinar neighbours push receivers along their predicted deviation; consistent with the acinar-transcript spillover the E7 read
  suspected (acinar secretory genes as top responders in non-acinar types). Not tested here.
- **A4 distance.** |contribution| by distance is the same for the real and the permuted section (56 % within 120 um, 85 % within 500 um; the twin
  84 % within 120 um: training spreads the reach). The signed contribution is not: mean 3.37 in the first bin (< 15 um) vs 0.35 permuted, 0.77 vs
  0.06 at 15-30 um, ~0 beyond 120 um. The content-dependent push comes from contact distance (also where segmentation spillover lives).
- **A3 maps.** Both trained and twin maps are dominated by the disc layout (hidden discs = sinks), as the plan warned; not readable without a
  twin-subtracted map.
- **Cost.** IG5 ~1.9 s per receiver; DeepLIFT / grad_input ~0.15 s, conservative 0.07 s, occlusion ~1.1 s.

### Next (proposed)

Fix the checkpoint path and rerun CosMx for the four read-out arms (IG / decomposition are what A6 needs on the IR); find the synth log; size
vannan on one sample and tile it; fix the map caption; a twin-subtracted A3 map; the spillover question on the
first-bin push (does it survive removing the receiver type's marker genes from the senders?).

### Fixes built (2026-11-24; the user: "implement all the fixes, update the scripts for the remaining runs, commit to main")

- **Library** (`evaluation/attribution.py`): `conservative` and `deeplift` take their gradients inside their patch contexts, so a checkpointed block is
  recomputed with the forward's functions. Test `test_gradients_through_attention_computed_in_checkpointed_blocks` (read-out model, `chunk_pairs` 64:
  many blocks, `checkpoint` counted) gives every method the one-block result; the old placement of the backward reproduces the cluster's
  `CheckpointError` (mutation checked).
- **Maps** (`flow_maps.plot_source_sink(what=, scale_label=)`; caption wrapped to the width left of the colour bar, header grows with it): the A3 maps
  say "IG attributions to disc targets (type-mean reference): out = a cell's contributions to discs it is not in, in = what its disc drew from outside".
  The attention flow maps keep their caption.
- **Jobs** (`slurm/November/`, not under git): `attr_job.py` `ATTR_RUNS` (a subset of arms or runs; the twins follow), `TIME_BUDGET_H` 4.5 per model
  (no new graph after it; the tables hold the graphs done), vannan on 1 mm tiles (its training crops), 12 drawn across the test samples (`MAX_GRAPHS`, seeded),
  `EVAL_TILE_MIN_CELLS` 500; `attr_common.py` per-graph progress lines (cells, receivers, seconds per method) and an `attr_graphs` table; `attr_job.sbatch`
  11:55. Checked: run lists (`--write-runs`, with and without `ATTR_RUNS`), `configure()` through the real `__main__` (runpy, `sweep_common.main` stubbed),
  the CosMx / vannan hook end to end on the toy read-out model with tiles, `MAX_GRAPHS` and a budget, a map drawn with the new caption.
- **Not a bug**: the A6 pairwise table (see the corrected "Reader" paragraph above). **attr_synth** has no log: not diagnosable; resubmit. Its read-out
  arm runs in one block on synth_spot (2,501 tokens < `chunk_pairs`), so the crash above cannot explain it.

Re-runs: `ATTR_RUNS=ir_c_ktype,ir_c_s42 bash slurm/attr_submit.sh cosmx` (4 tasks; joins 22_11_results/attr_cosmx), `bash slurm/attr_submit.sh synth`
(send the .out / .err if it fails again), `bash slurm/attr_submit.sh vannan` (4 tasks).

## Re-run read (2026-11-25; results MA/22_11_results/attr_{synth,cosmx,vannan}, synced 2026-11-25; logs MA/results/logs)

### What ran

| Job | Runs | Coverage | Seconds |
| --- | --- | --- | --- |
| synth (43894205, 4 tasks + report) | J1 `ref` s42/s43 + twin, `ir2_rate` s42/s43 + twin | 18 receivers per run, all classes | ref ~1,270 s, ir2_rate ~6,400 s per run |
| cosmx (re-run, ATTR_RUNS) | ir_c_s42, ir_c_ktype s42/43/44 (+ typeres_s42 + twin from the first run) | **3 of 6 test sections** per read-out arm (4.5 h budget); ~49 receivers per section | 16,200-17,900 s per arm |
| vannan (tiled) | ir_c_ktype s42/43/44 + twin, typeres_s42 + twin | 12 tiles of 1 mm each; ~1.5 receivers per fine type and tile | ktype ~20,000 s, typeres ~6,000 s |

Missing: the **untrained twins of ir_c / ir_c_ktype on CosMx** (no tables; ATTR_RUNS should have taken them along). Without them the CosMx
read-out arms have no trained-vs-twin contrast in A2 / A6.

### A1 on synth_spot (the method decision)

- **Planted sender in the top 5:** both trained models find short (0.96-1.0) and mid (1.0) senders with every method except raw attention (ir2_rate mid
  0.76); the twins find short senders too (0.56-0.68; attention 1.0: geometry) but not mid (0.02-0.06). dense: only J1 `ref` (0.22-0.29 vs twin
  ~0.01); directional and sparse at twin level for both (<= 0.10).
- **Completeness:** IG is complete on both (ir2_rate median gap 0.023 at 5 steps, 0.003 at 50; ref 0.11 / 0.02). grad_input / DeepLIFT / conservative
  0.8-1.5. decomposition is exact by construction (gap 0) but agrees poorly with IG (Spearman 0.34).
- **Rule (fixed before the run): fails for every cheap method on the trained models** (gap > 0.1, Spearman with IG50 0.66-0.82 < 0.9). Only the
  untrained ir2_rate twin's grad_input passes. Maps stay on IG for sampled receivers and disc targets, as planned.

### A6 (linearity) -- the synth controls are weak

- **Pairwise sender interactions are small everywhere on synth** (median |I| / (|D_j| + |D_k|) 0.02-0.07), including the learned thresholded programme
  MC23 on `ref` (0.059 with the graph-mean reference, 0.032 sampled), where the target clearly moves (median |change| 1.49). MC20 (unlearned) is
  trivially small (|change| 0.004-0.007). ir2_rate hardly moves MC23 (|change| 0.059): it did not learn the threshold.
- Sensitivity-n with the graph-mean reference stays high for the trained models (IG50 r 0.70-0.96 at n = 1 and 16; ref MC20 0.28-0.43 is trivial);
  with sampled references (large perturbations) it collapses for the trained models (0.01-0.47, short senders at n = 16 the exception: 0.91-0.95)
  while the twins hold 0.50-0.72. Large perturbations show non-additive responses; the first-order (graph-mean) read does not.
- **Consequence for the CosMx claim:** the measure separates the learned nonlinear rule only weakly on synth at the median (MC23 0.059 vs M11 0.024
  on ref, 2.5x), and typeres on CosMx (0.078-0.092) sits *above* that learned threshold programme. So the 11-21 hypothesis ("CosMx uses its
  neighbourhood as an approximately additive function") is not supported for typeres; for the read-out arms additivity is built in. A positive control
  with a large planted interaction is needed before the median is read either way.

### CosMx (ir_c, ir_c_ktype on 3 sections; typeres as before)

- **Completeness:** IG5 complete on the read-out arms (ir_c 0.022, ir_c_ktype 0.081; ig5_sampled 0.02-0.03); cheap methods 0.6-2.1. No cheap method
  can stand in for IG (agreement 0.54-0.72; occlusion 0.81-0.85).
- **Pairwise interactions (A6), median over receiver types and sections:** ir_c 0.007-0.010, ir_c_ktype 0.006-0.010 under all three variants (type
  mean / reference, type mean / mask, sampled) vs typeres 0.078-0.092 (twin 0.018-0.020). The interaction read-out models are additive in their senders;
  part of that is the architecture (additive read-out; on vannan its twin is as additive), so it says nothing about the tissue. typeres, the dense
  hierarchical transformer, is ~4x its twin on both datasets.
- **A2 sender types (trained minus within-type permuted):** Acinar senders lead again for every model kind (ir_c_ktype: Acinar -> Acinar 0.34,
  -> Endothelial 0.31, -> Ductal 0.22, -> Immune 0.19). New: **Beta senders -> Alpha receivers +0.24 (ir_c_ktype) / +0.21 (ir_c)**, and Acinar ->
  Alpha negative (-0.16 / -0.09): the islet neighbourhood pushes Alpha cells, the exocrine one pulls them. Islet co-location or INS / GCG spillover
  between neighbouring endocrine cells: not separable here (Item: test with the receiver's own type markers removed from the senders).
- **A4 distance:** |contribution| by distance is the same in the real and the permuted section for every arm (ir_c 47 % within 60 um vs 45 %
  permuted, 91 % within 500 um): the reach is geometry; content changes the sign, not the reach (as typeres on 11-24).

### vannan (12 tiles)

- IG complete on ir_c_ktype (0.006) and its twin; typeres 0.33 (cheap methods 1.8-3.3). Pairwise: ir_c_ktype 0.003-0.009 (twin 0.000-0.009), typeres
  0.083-0.095 (twin 0.020-0.034): the same split as CosMx.
- **A2 needs the twin:** the untrained ir_c_ktype reproduces most top sender -> receiver pairs (Secretory <- Basal 0.26 twin vs 0.47 trained; RASC <-
  Basal 0.29 vs 0.36; Interstitial macrophages <- Alveolar FBs 0.25 vs 0.37; Alveolar FBs <- Alveolar FBs 0.25 vs 0.38). What training adds is the
  difference (e.g. Adventitial FBs <- Alveolar FBs 0.41, absent from the twin's top), not the table.
- A4: trained vs permuted differ by <= 0.05 per bin; ir_c_ktype puts 52 % of |IG| within 30 um (twin 41 %).
- ~1.5 receivers per fine type and tile: the per-type tables rest on ~18 receivers per type.

### Next (proposed)

Rerun the CosMx read-out twins (ATTR_RUNS=ir_c_s42_untrained,ir_c_ktype_s42_untrained) and sections 4-6; a synth positive control where
pairwise interaction must be large (a planted AND of two senders) before A6 is used as a linearity argument; A2 tables as trained minus twin.

## S13 status and the all-cell IG map (2026-11-26; the user: "note these things down", pursue from 2026-11-27)

### S13 / A6 in one paragraph (as explained to the user)

S13 asks, from inside the trained model, whether CosMx's neighbourhood is used as an additive (linear) function: the third line next to the
score level (models tie the linear k-NN smoother) and the data level (ceiling audit: the linear neighbourhood signal is small and within ~100 µm).
Measures: pairwise sender interaction I_jk (hide j, k, both: 0 for an additive function), agreement between attribution methods (Ancona Prop. 4),
Sensitivity-n. State after the 2026-11-25 re-run: the read-out arms (ir_c, ir_c_ktype) are additive (0.007-0.010) by construction, so nothing about
the tissue; typeres (dense hierarchical transformer) is not (0.078-0.092, ~4x its twin, CosMx and vannan); the synth positive control is weak (the
learned threshold programme MC23 0.059, below typeres). Hypothesis not supported, instrument not validated.
**To close it:** (1) a planted AND of two senders on synth_spot (an interaction that must be large) as the positive control; (2) the CosMx read-out
twins (ATTR_RUNS=ir_c_s42_untrained,ir_c_ktype_s42_untrained) and sections 4-6; (3) the NCEM novelty check. **Fallback if time runs out:** one
paragraph in thesis 5.2 (read-out additive by construction, the free transformer ~4x its twin, the measure unvalidated -> future work).

### Why the A3 / 1.6 IG maps show the discs, and the all-cell map

The CosMx maps are already on a clean pass (nothing hidden). The discs dominate because only the cells of ~30 discs of 30 µm per section (4 per tile
in 1.6) are receivers: `disc_strengths` divides a disc's whole received total over its ~10-30 cells ("in" concentrated) while "out" is spread over
thousands of senders, and the plot marks disc cells as hidden (`visible[members] = 0`). The layout of the receivers, not the model; the twin shares
it, so subtracting the twin does not remove it (A3's stated caveat).

**Every cell a receiver, for models blind to their own cell** (`interpretation.blind_to_own`: ir_c, ir_c_ktype, any context-query model without the
own-cell skip; ir_ch if only its context part is the target, its raw skip being a separate additive part). IG is linear in the target and the path
depends on the inputs only, so:
- out(j) for every cell = ONE IG with the summed target sum_i w_i f_i (the deviation targets of all cells); a blind model's self-term c_jj is 0.
- in(i) for every cell = completeness: w_i f_i(x) - w_i f_i(x_ref) (two forward passes, all rows at the type-mean reference), minus the gap.
- Cost per section ~ IG steps x one forward + backward of the whole section (fused attention with checkpointing, as in training): seconds to minutes,
  vs hours for ~50 receivers per section now.
- Code change: `attribute()` keeps the receivers' own rows fixed; with every cell a receiver nothing would move -> a path where every row moves to
  the reference (equivalent for a blind model). Test: the summed IG equals the sum of per-receiver IGs on a toy blind model; the completeness in(i)
  equals the row sums.
- Run: evaluation only on existing checkpoints: CosMx ir_c / ir_c_ktype + twins, synth ir_ch (context part) + twin; maps in `plot_source_sink`'s layout,
  twin-subtracted.

**Models that read their own cell** (typeres, the prototype, ir_ch's full prediction): a clean pass measures copying (they were never trained to
predict visible cells). There the original A3 design: random cell folds matching the training mask (not discs), one summed IG per fold, every cell a
receiver once; the mask layout averages out over folds. Cost: one IG per fold.

Estimate: ~0.5 day build + tests, one short eval-only job. Worth it only if IG maps go into thesis 4.7; otherwise caption the disc layout.


## S13 build 2026-11-27 (the user: "proceed to implement everything needed for point 2, S13 / A6 linearity")

### NCEM novelty check (Fischer, Schaar & Theis, Nat. Biotechnol. 41, 332-336, 2023; read from the PDF, main text + Methods + Ext. Data Fig. 8)

- **What they did.** Node-centric expression models: predict a cell's expression vector from its own type label and its niche. The
  linear NCEM's design matrix is the receiver type, the binary presence of each sender type within a radius (an indicator, not a count) and
  their outer product (receiver type x sender type interaction terms), plus batch covariates; coefficients by OLS, a Wald test per
  coefficient, "type coupling" = the number of significant genes per sender-receiver pair. Nonlinear NCEMs: a GNN encoder (an indicator
  kernel over the niche, "IND", or a one-layer GCN) and a nonlinear decoder; also a CVAE and a ligand-receptor graph kernel (NL-NCEM LR).
- **The linearity result** (Ext. Data Fig. 8b, MERFISH brain, chip cytometry colon, MIBI-TOF cancer): held-out R^2 by niche radius; "the
  nonlinear models did not outperform linear models in gene expression prediction, which suggests that the spatial dependencies in the
  given datasets are well described by linear models". Score level only. Their single model-internal readout is gradient saliency of the
  nonlinear model aggregated per sender type x receiver type (Ext. Data Fig. 8c), with no linearity or interaction test.
- **What it means for S13.** The question (is the learnable cell-cell dependency linear?) is NCEM's, and must be cited as such: our
  score-level line (models tie the linear k-NN smoother, the ceiling audit) repeats their kind of argument on CosMx. What I have not found
  in NCEM: a test of what a *trained* model computes (pairwise sender interactions, method agreement by Ancona Prop. 4, Sensitivity-n),
  and a positive control showing the instrument sees an interaction where one is planted. Score-level ties cannot tell "the tissue is
  additive" from "the nonlinear model underfits" (their own MIBI-TOF nonlinear models sit ~0.10 R^2 below the linear one, Fig. 8b), which
  is exactly what a model-side test with a planted AND separates. Wording for the thesis: "NCEM found at the score level that nonlinear
  models do not beat linear ones; we test inside the trained model, with a positive control." Never "novel" without the LRP-for-
  transformers / attribution literature check, which is still open (Chefer et al., Ali et al. 2022 conservative propagation; interaction
  attributions: Integrated Hessians, Shapley-Taylor / Faith-Shap; Tsang et al. NID / Archipelago); those test feature interactions in
  general, not sender-sender interactions of a spatial cell model, as far as I know.
- Also from NCEM, useful for 4.7 / 5.2: their linear model uses sender-type *presence* indicators, so its "linear" already saturates in the
  number of senders of a type (an OR within type); the pairwise test here (one sender of each of two types) is outside what an indicator
  model can express only for pairs of different types (their interaction terms are receiver x sender, never sender x sender).

### (1) The planted-AND positive control

- **Data** (`evaluation/synth_spot.py`, `make_synth_spot(and_programme=True)`, CLI `--and-programme`; opt-in, default unchanged): senders S5,
  S6 (8 % of spots each, disjoint), responses R5 = **AND** (base mean 2.5 with neither or one sender type among the 4 neighbours, 10 with both)
  and R6 = the **matched additive control** (2.5 -> 5 -> 10: doubling counts is a constant step in the log-normalised space the models
  predict; a sum in counts, 0 -> 5 -> 10, would be strongly non-additive after log1p at a zero baseline: relative interaction ~2, more than the
  AND's 0.5). Same receivers, same senders, same endpoints: only the rule differs. Roles `sender` / `receiver`, so programme masking hides
  R5 / R6 in every spot. Appended last with their own keyed stream: every J1 gene is count for count the default dataset's (tested; checked on
  the full design locally). Truth: obs `is_S5`, `is_S6`, `n_S5_nbrs`, `n_S6_nbrs`, `R56_state`; edges `S5&S6>R5`, `S5+S6>R6`. Full design:
  ~131 receivers per slide with exactly one S5 and one S6 neighbour, 1,060 on the 8 test slides; X by state (log-normalised) R5 2.77 / 2.77 /
  4.08, R6 2.77 / 3.43 / 4.08.
- **Library** (`attribution.pairwise_interactions(given_pairs=)`): `[Q, P, 2]` named pairs instead of random near pairs (tested by hand
  against the definition; a mutant reading k as j is caught).
- **Job** (`attr_job.py` `ATTR_JOB=and`, `make_and_synth_yaml.py`, `attr_submit.sh and`): a CPU prep job makes `MA/synth_spot_and.h5ad`,
  checks it (`--check-and-data`: levels, test receivers, J1 genes identical to `MA/synth_spot.h5ad`) and preprocesses it (MODE list); then an
  array trains J1's `ref` (lin_dist_skipraw) and `ir2_rate` (additive read-out: the architecture control that cannot express an AND)
  x seeds 42, 43, 300 epochs, programme masking, twins at s42, checkpoints in `27_11_results/attr_and` (J1's never touched); evaluation =
  the standard programme evaluation (R5 / R6 by interaction group; `ref_oracle` with `R56_state`) + the attribution hook as `synth` + the AND
  block (`attr_common._and_control`): receivers with exactly one S5 and one S6 neighbour, 400 um apart, <= 40 per graph; per response:
  IG20 / grad x input / DeepLIFT (agreement, planted pair in the top 2 / 5), pairwise interaction of the **planted pair** and of random near
  pairs, Sensitivity-n, under graph_mean and sampled references; `and_learned` = prediction and truth by exposure state over every cell.
  Estimated per task: ref ~2-3 h, ir2_rate ~6 h with its twin (J1 timings + attr_synth's 1.75 h per ir2_rate model).
- **Rule, fixed before the run** (`attr_read.py`): the instrument is validated if a model that learned the AND (AND index (both - one) /
  (both - none) within 0.2 of the truth: ~1 for R5, ~0.5 for R6) shows a planted-pair median |I| / (|D_j| + |D_k|) >= 0.25 on R5 and <= 0.1 on
  R6 (graph-mean reference), its twin <= 0.1 on both. Expected for a perfect AND: 0.5 (D_j = D_k = D_jk). If random near pairs stay near 0 on
  R5 while the planted pair passes, the CosMx protocol (random near pairs) cannot see a sparse AND: then the CosMx read is "no dense
  interaction among near senders", not "additive".

### (2) CosMx read-out twins and sections 4-6 (`ATTR_JOB=cosmx_s13`, `attr_submit.sh cosmx_s13`)

- The 11-24 re-run stopped the read-out arms after 3 of 6 sections (4.5 h budget) and left no twin tables for them: `ir_c_ktype` was not in
  `UNTRAINED_ARMS`; `ir_c` was, but `ir_c_s42_untrained` has no tables or section (why is in the task's log, not synced here: the 11:55 limit
  after a 4.6 h trained read is a guess). The plan's `ATTR_RUNS=..._untrained` would not have selected them (the array lists trained runs;
  a twin is evaluated inside its run's task). typeres and its twin already have all 6 sections.
- New settings in `attr_common`: `GRAPH_START` / `TWIN_GRAPH_START` (absolute graph index kept), `TWIN_ONLY` (no tables written for the
  trained model), `A6_ONLY` (skips occlusion, the permuted section, named targets, the A3 discs: what A6 does not read, ~half a graph's time);
  env overrides `ATTR_GRAPH_START`, `ATTR_TWIN_GRAPH_START`, `ATTR_TWIN_ONLY`, `ATTR_A6_ONLY`.
- The job: ir_c_ktype x 3 seeds and ir_c_s42 on sections 4-6 (trained), twins of ir_c_ktype_s42 and ir_c_s42 on all 6, A6 only, 5 h per model,
  own folder `22_11_results/attr_cosmx_s13`; the report reads it with `attr_cosmx` (`attr_read.py cosmx --also`), and adds an A6 table per run
  (each seed must agree).
- Read it for: twin vs trained interaction of the read-out arms (architecture: both ~0 expected), the 3 seeds' agreement, and agreement /
  Sensitivity-n with all 6 sections. typeres stays the model the CosMx claim rests on (its 6 sections + twin exist).

### Checks run (no model run locally)

- Tests: `tests/test_synth_spot_and.py` (3; mutants: R5 additive, S5 / S6 overlapping, both caught), `test_attribution_methods.py` +
  `test_attribution.py` (21, incl. the given-pairs definition test).
- A toy smoke (14 x 14 lattice, untrained 8-dim GlobalModel on CPU, programme mask as the evaluation draws it): `_and_control`, `_synth` and
  `attr_read.py and` end to end (tables and report written). `sweep_common.configure` accepts both new jobs; `--write-runs` gives
  ref_s42, ref_s43, ir2_rate_s42, ir2_rate_s43.
- The local `data/synth_spot.h5ad` is from 2026-10-10 (before `signal_scale`): stale, not the cluster's file; the prep job compares the
  AND file with the cluster's `MA/synth_spot.h5ad`.

### To run (the user submits)

Sync the fork (synth_spot.py, attribution.py) and copy to `MA/scripts/`: attr_job.py, attr_common.py, attr_read.py, make_and_synth_yaml.py;
to `MA/slurm/`: attr_submit.sh, attr_job.sbatch. Then `bash slurm/attr_submit.sh and` and `bash slurm/attr_submit.sh cosmx_s13`.


## The all-cell IG map, built 2026-11-27 (the user: "move on to the all-cell IG map")

- **Library** (`evaluation/attribution.py`): `all_cell_map(module, batch, weights [N, G], reference=, part=, steps=, ...)`: one IG of the summed target
  `S = sum_i w_i . f_i` on a path where every movable row moves (no receiver row fixed). `out[j]` = what cell j sends to all others (signed);
  `in[i]` = `w_i . (f_i(x) - f_i(x_ref))` from the two end passes (= what i drew, by completeness); `gap`, retry at more steps. Guards: `token_blind`
  (context query; an own-cell skip only on the raw input, which is constant along a token path; no visible-mean / group offset) and an empirical one
  (the gradient of `check_cells` cells' own targets on their own rows; raises above 1e-6 of the row terms). `part_prediction(module, batch, part=)`
  gives the profile the targets are made from (`part="context"`: a model with a raw skip). `_PartReader` reads a head or one named part.
- **Tests** (`test_attribution_methods.py`, 3 new): on the read-out (`ir`) and the dense context query (`stream`) toy models, `out` equals the
  per-receiver IGs (one receiver at a time) summed over receivers (rtol 1e-3, float32 contributions), `in` equals each receiver's change, out and
  in add up; a plain encoder is refused by the static check and, with that check patched away, by the empirical one. Mutant (no quadrature
  weights) caught.
- **Jobs** (`attr_job.py`, `attr_common._all_cell`, `ALL_CELL_MAP`): `allmap_cosmx` = J4's ir_c_ktype x 3 + ir_c_s42, twins of both at s42, the clean
  pass with every cell scored, targets = each cell's deviation from its type mean (`deviation_targets`), references type mean (maps) and sampled
  cells x 3 (Spearman of out / in / net as the check); tables per graph (gap, steps, self share, seconds), by cell type (mean out, in, net, share of
  |out|). `allmap_synth` = J1's ir2_rate x 2 + twin **under its programme mask** (in distribution: it was trained with the responses hidden
  everywhere), `part="context"` (the raw skip excluded), the deviation over the hidden response genes only, references graph mean and sampled;
  AUROC of out for the planted senders and of in for the planted receivers, by range class, trained vs twin. I chose J1's ir2_rate over the
  plan's ir_ch: ir_ch is channel-trained and the clean pass would be off its training distribution. Maps per graph (npz + png) for the run and
  its twin, and **trained minus twin** (`subtract_fields`, each scaled to its 99th percentile) in the twin's pass (the trained map is written first
  in the same task). `plot_source_sink` cannot draw synth_spot's lattice (1.6's "no tissue block"): a plain per-cell scatter is drawn instead
  (`_scatter_map`, symmetric scale, nothing smoothed or selected). `attr_read.py allmap` writes the tables.
- **Smoke** (toy lattice, untrained read-out model with the raw skip, programme mask; CPU): both jobs' paths end to end, report and maps written;
  the clean path's `show_everything` refuses a gene-masked toy model (expected: the CosMx read-out models are node-masked, as in `_cosmx`).
- **Read rules (fixed before the run):** a map is read only where the gap <= 0.05 and the self share ~0; a type's or planted class's value counts
  only beyond the twin's; the sampled reference must agree in rank (Spearman of net >= 0.5) before a map is described. Cost: one forward + backward
  per IG step on a whole section (as a training step), x (5-20 steps) x (1 + 3 sampled draws): minutes per section; 5 h budget per model.
- **Run:** `bash slurm/attr_submit.sh allmap_cosmx`, `bash slurm/attr_submit.sh allmap_synth`; results `MA/27_11_results/allmap_{cosmx,synth}/`.

### legnini variant (2026-11-27; the user: "add a legnini variant of the IG script; if it requires cell types use spatial patches (8x8)")

- `ATTR_JOB=allmap_legnini` (`attr_submit.sh allmap_legnini`; results `MA/27_11_results/allmap_legnini`): split_legnini's **ir_ch** x 2 seeds + twin
  at s42 (checkpoints `MA/25_11_results/compare_legnini`; the submit writes `compare_legnini_sweep.yaml` again with make_compare_yaml). proto_ch /
  proto_ch_pe read their own cell (GCN): refused by `token_blind`, not readable this way.
- **Cell types replaced by spatial patches** (`PATCH_GRID: 8`, `attr_common._patch_labels`): each section's bounding box cut 8 x 8, positions only
  (~65 cells per patch at ~4,300 cells). The patch is the label of the reference (patch mean token; sampled: a cell of the same patch) and of
  the target (deviation from the patch's mean prediction, over the hidden entries). A first concrete piece of A7 (attribution without annotation):
  a label-free reference that keeps "a cell like this one" local.
- **Read as trained** (`MAP_FOLDS: 0.2`): ir_ch was trained with 20 % of the genes hidden in every cell, so 5 gene folds (`gene_ranking.gene_folds`,
  `hide_genes`), each hidden in every cell; every gene a target once; the five maps summed into one (IG is linear in the target; each fold's own
  completeness gap is reported). **Named map** (`MAP_NAMED`): the SHH readouts PTCH1, GLI1, HHIP hidden alone in every cell and mapped on their
  own (legnini_plan.md's positive control): which cells drive the SHH response; 3 hidden genes vs 18 in training, a milder input than trained.
- Report (`attr_read.py allmap`): per reading (`all`, `named_SHH_readouts`) and reference the gap, self share, seconds; references' agreement;
  per patch the spread (min / median / max) of the per-cell means within each section, then the mean over sections (patch names repeat across
  the two test sections, control and SHH, which must not be pooled). Maps `graph<k>.png`, `graph<k>_named_SHH_readouts.png` and both minus twin.
- Smoke (toy read-out model, 4 x 4 patches, folds 0.2, one named set and one with no gene in the data): tables, maps and report written; the
  missing set skipped with a message; `configure` and the run list (ir_ch_s42, ir_ch_s43) checked.
- Read: the two test sections are slide4_A2-3 (control) and slide4_B2-2 (SHH): the SHH readouts' map should differ between them (senders near
  the induced region in the SHH section); a map counts only beyond the twin's and where type_mean and sampled agree (Spearman of net >= 0.5).


## Weekend build 2026-11-28 (weekend_plan.md A-C; the user: "implement A, B and C now")

Library (fork): `AttentionTape` (record / replay every softmax and SDPA kernel's weights, in call order, shape-checked, every recorded call must be
used; patches `torch.softmax` / `log_softmax`, the `Tensor` methods, which `F.softmax` and the vendored attention's `softmax` dispatch to, and both
SDPA names), `pairwise_interactions(freeze_attention=)` (reference removal only) returning the drops `d_j`, `d_k`, `d_jk`; `subtract_fields(normalise=
"common")`; `make_synth_spot(and_rate=, and_high=)` with `and_levels(high)` (R6 the geometric middle; defaults = the 11-27 data value for value).
`_conservative_rules` now uses the same patcher (forward and every existing test unchanged).
Tests: tests/test_attention_tape.py (5: replay exact, changed step refused, mutant without the Tensor methods records nothing, every softmax name
reached once, frozen read-out additive to float precision, closed-form bilinear head), test_synth_spot_and.py (+1), test_attribution_methods.py (+1).

Cluster (slurm/November, copy to MA/scripts: attr_common.py, attr_job.py, attr_read.py, allmap_checks.py, make_and_synth_yaml.py, legnini_truth.py;
MA/slurm: attr_submit.sh, attr_job.sbatch, compare_job.sbatch; sync the fork):
- `bash slurm/attr_submit.sh and_eval`: the 27_11 `and` checkpoints, AND block only, free + frozen pairs, per-receiver table -> 28_11_results/attr_and_eval
  (minutes per model; ir2_rate's AND block was ~10 min of its 40).
- `bash slurm/attr_submit.sh and_mlp`: ref_mlp, ir2_rate_mlp (decoder nonlinear [64, 64], no dropout) x 42 / 43 + twins on synth_spot_and.
- `bash slurm/attr_submit.sh and_strong`: prep makes synth_spot_and_strong.h5ad (`--and-rate 0.15 --and-high 40`); ref, ref_mlp x 2 + twins.
- `bash slurm/attr_submit.sh allmap_synth_refs`: allmap_synth with 10 sampled draws, both references kept per cell (B7).
- Every allmap report now also runs allmap_checks.py into `<results>/checks`. Locally: `python allmap_checks.py --maps <allcell_maps> --out <dir>
  [--labels cosmx_pancreas.h5ad]`; `python legnini_shh_region.py --maps ... --data legnini23_pp.h5ad --out ...`; `python legnini_truth.py ...`.

Rule v2 for S13 (fixed before the runs, attr_read.py docstring): per run, the paired R5 - R6 relative interaction (graph mean, planted, attention
free) has median >= 0.1 and bootstrap lo95 > 0 for a run that learned both rules (AND index within 0.2 of the truth on R5 and R6), every seed;
twin |median| < 0.05. The frozen reading is reported beside it and decides nothing.

Read locally 2026-11-28 (notes.md): A3 R5 - R6 = 0.000 +- 0.005 on the 11-27 tables; maps: synth and CosMx seed-stable, legnini not; CosMx
Alpha / Beta / Immune / Endothelial draw beyond a within-100-um label null in all 3 seeds; the legnini SHH map's sender block is not at the source.


## 2026-11-28, later: frozen A6 on CosMx and the pair-draw fix

- `ATTR_JOB=cosmx_frozen` (`attr_submit.sh cosmx_frozen`; 28_11_results/attr_cosmx_frozen): `PAIRS_ONLY` + `A6_FROZEN` on every CosMx section for typeres_s42
  and ir_c_ktype x 3 (+ twins); per-receiver pairs (`pairwise_receivers`) and attr_read.py's paired free - frozen table. The tape holds typeres' routed
  candidates (tested). Read: free minus frozen = the softmax's (and the routing's) re-weighting; frozen = token-wise non-linearity. If typeres' frozen
  interaction falls to its twin's level, its ~4x-twin reading was attention re-weighting, not a non-additive use of the neighbourhood.
- `A6_FROZEN` also on the synth path (variant `.../frozen`).
- `pairwise_interactions` random draws: no cell shared between receivers of one pass, no receiver as a sender (a false interaction otherwise; notes.md
  2026-11-28); given pairs that overlap are refused.
