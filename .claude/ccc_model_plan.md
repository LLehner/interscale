# Model choices from the CCC papers and SAFFRON (plan, 2026-11-21)

## Goal

Before the attribution plan is built (its Stage M retrains "one model per dataset with the best settings"), decide which
model the global component should be. Read the cell-cell communication (CCC) papers the user collected
(`/home/lehnerl/Arbeit/to_read/`), keep what works there and in our own runs, drop what did not help, and say how each kept
idea enters the global transformer. SAFFRON (an evaluation of spatial foundation models) is read for what it offers
CCC training and evaluation.

Constraints from memory: evaluation stays dataset-agnostic; never run models locally; results must come fast (single
sbatch jobs < 12 h, few arms); judge by centred cosine within section x type, never claim what a model reads from
reconstruction scores alone; papers are inspiration, every new piece is a switchable option; build the extension point
first, gate with decision rules fixed before the run.

## Status

| Stage | What | Status | Date |
|---|---|---|---|
| R | Read the 11 papers, verdict per idea (this file) | **done** | 2026-11-21 |
| S0 | Build the interaction read-out (IR): receiver-aware query, null key, additive read-out, exact influence export; options: content-dependent decay, entropy + L1 with ramp, key dropout, same-type keys, key inputs, all-cell targets | **built**, on main since 2026-11-21 (c9a43aa, 90ae83d; section "Build") | 2026-11-21 |
| E1 | SAFFRON-style probes: sparse autoencoder on embeddings against known spatial variables, nulls, OMP, naive baselines | **built** (`evaluation/sae_probe.py`), runs inside J1 / J3 / J4 | 2026-11-21 |
| S1 | synth_spot gate: IR arms against the current best (`lin_dist_skipraw`, 300 epochs), readouts E1 + E3 | **run, read**: R2 pass, R1 / R3 / R4 fail -> IR = readout model beside the score model; entropy arms not evaluated (synth-only failure) | 2026-11-22 |
| E2 | Semi-synthetic CosMx: planted interactions on real coordinates, types and cells (AMICI's generator idea), incl. long range | **run, read**: generator works model-free; model readout inconclusive (binary receiver truth near-degenerate, n = 48); GITIII not run | 2026-11-22 |
| S4 | CosMx + vannan with the IR (type residual, all-cell targets), key-input and same-type ablations, readouts E4 / E5 | CosMx **run, read**: J4 rule passes (0.137 clean / 0.096 masked vs knn128 0.109 / 0.089), keys need no expression; vannan not built | 2026-11-22 |
| S5 | Optional, gated by E1: Matryoshka / TopK context code, nuisance subspaces (scDIVA), metagene heads (Steamboat), JEPA auxiliary (TERRA) | to note | 2026-11-21 |
| E7 | Gene rankings of interaction (the user, 2026-11-22: judge a model by which genes it finds interacting, not by reconstruction): responders, drivers, pairs by intervention; truth on synth_spot (roles) and the spiked file; bars (Moran, variance, neighbour ridge, twin); rules G1-G3, S1-S3, C1 | **built** (`evaluation/gene_ranking.py`, `gene_flows(targets=)`; tests, mutants; section "E7") | 2026-11-22 |
| J5 / J6 | Ranking jobs: J5 synth_spot (ir_ch, ctx_ch trained with channel masks, 2 seeds + twins), J6 CosMx (J4 checkpoints, evaluation only), J6b spiked file (J3 checkpoints) | written and checked (`rank_submit.sh synth\|cosmx\|semi`), **not run** | 2026-11-22 |
| D2 | Second round: the user's decisions, the two reviews read in full, SAFFRON's gradient / niche / nesting analyses, the semi-synthetic generator made concrete, the training modes explained (section "Second round" at the end) | **done** | 2026-11-21 |

## What was read

| Paper | What it is | Read |
|---|---|---|
| AMICI (Hong, Desai, ..., Azizi; bioRxiv 2025) | one cross-attention layer: receiver **type** as query, neighbour expression as keys / values, monotone distance decay with a learned per-pair rate, an "empty" neighbour, entropy + L1 penalties, prediction = type mean + residual | full |
| GITIII (Xiao, Zhang, Zhao, Wang; Nat Mach Intell 2025) | single-layer graph transformer (GRIT-based) on 50-cell neighbourhoods, predicts the cell **state** (expression minus type mean); influence tensor `[cell, neighbour, gene]` | main text + methods |
| Steamboat (Liang, Tang, Wang, Ma; bioRxiv 2025) | nonnegative rank-1 "metagene" heads (scalar query / key, gene-space value), three scales per head: ego, local (k-NN, k = 8), global (whole sample), no softmax, 10 % masking | full |
| CellNEST (Zohora et al.; Nat Methods 2025) | graph attention network with Deep Graph Infomax on ligand-receptor-gated edges (300 um), unnormalised attention as CCC strength, 5-seed rank-product ensembles, relay networks | main text + methods |
| TERRA (Birk, Sanian, ..., Lotfollahi; bioRxiv 2026) | spatial foundation model, 112 M cells: gene tokens of a cell + its k <= 10 neighbours, JEPA (latent prediction, EMA target), batch metatoken, in-silico knockout with controls | main text + methods (skimmed the biology) |
| scDIVA (Rapolu et al.; bioRxiv 2026) | semi-supervised DIVA for scRNA-seq: separate latent subspaces for type, batch and residual with auxiliary classifiers; Milo DA for query-only states | main text + model |
| SAFFRON (Handa, ..., Chitra; bioRxiv 2026) | evaluates SFM embeddings: Matryoshka BatchTopK SAE, max Spearman of a feature with a known spatial variable within domains, permutation null, matched dictionary size, OMP minimal feature sets, PCA / gene-SAE baselines | full (attached) |
| EV range (Colombo et al.; J Extracell Vesicles 2025) | how far extracellular vesicles travel in tissue | abstract, results on range |
| CellChat protocol (Jin, Plikus, Nie; Nat Protoc 2025) | mass action + Hill, label permutation, LR classes, centrality, NMF patterns, condition comparison | skimmed |
| Cesaro et al. (Brief Bioinform 2025), Armingol et al. (Nat Rev Genet 2024) | reviews of CCC tools, resources, validation | skimmed for evaluation and spatial methods |

## What the papers agree on

1. **Predict a cell's within-type residual from the other cells.** AMICI (type mean + predicted residual, receiver masked),
   GITIII (target = expression minus type mean), NCEM's metric (within-type variance explained, used by GITIII); Steamboat
   masks 10 % of cells.
2. **One interaction layer whose output is a sum over senders.** AMICI (single layer, "can be directly interpreted as pairwise
   interactions without compounded effects"), GITIII ("does not apply any nonlinear transformation after aggregation";
   1-4 layers give only marginal gains, Fig. 6d), Steamboat (one attention step). The interpretation is read from the model's
   own decomposition, not from gradients.
3. **Distance enters as a learned, mostly decreasing decay.** AMICI (`b0 - b1 d`, `b1 > 0` learned per pair; recovers planted
   10 / 20 um scales over 10 replicates, where fixed-radius NCEM / NicheDE fail), GITIII (distance scaler and MLP distance
   features; removing them costs), EV paper (80 % of vesicle signal within 40 um of the donor; denser tissue keeps it closer).
4. **Interpret by intervention, with controls.** AMICI ablates a sender type and tests the change per gene (Wald on the
   correlation with the observed expression); Steamboat knocks out a gene in one type and "transplants" cells to another region;
   TERRA knocks out genes against random-gene and housekeeping-gene knockouts; CellNEST stops using softmax-normalised attention
   because it is not comparable across a tissue.
5. **Composition and segmentation are the main confounds.** AMICI drops stromal cells and flags genes higher in distant senders
   than in interacting receivers; GITIII subtracts the expected composition share from its network strength; GITIII and AMICI
   hide same-type neighbours (type mean / exclusion) so the model cannot copy the nearest same-type cell.
6. **Naive baselines are hard to beat.** SAFFRON: most SFMs match PCA / gene-SAE on gradients, none beats them on the local
   plaque microenvironment; TERRA cites the same for single-cell FMs.
7. **No ground truth; planted data and known spatial variables stand in.** AMICI and CellNEST plant interactions; SAFFRON scores
   against known axes; Armingol: validating by co-localisation "may obscure biological processes involving long-distance
   interactions".

All four learned CCC methods are short range by construction (k = 8 to 50 neighbours, or 300 um). None learns long-range
attention over whole sections; Steamboat's "global" is the sample mean of a metagene.

## Per paper: what works, what did not help, what it means for us

### AMICI

- **Works (their evidence):** on a PBMC-based semi-synthetic benchmark (10 replicates) it recovers the planted length scales,
  matches GITIII on downstream genes (AUPRC) and beats GITIII (p = 0.028) and CGCom (p = 0.0003) on which neighbours are
  senders and which cells are receivers; they attribute the gain to the attention penalties. Same interactions in two Xenium
  replicates trained separately.
- **Design details worth copying:** query = receiver type embedding, keys / values = MLP of the neighbours' log expression; 50
  nearest neighbours **of other types** ("to prevent the model from bypassing neighbor influences by copying the gene expression
  of the nearest cell in the same annotation group"); an "empty" neighbour token with zero value lets a head attend to nothing;
  entropy penalty on the attention, ramped from 0 to its final value over epochs, plus L1 on the values; prediction = empirical
  type mean (optionally learned: fixed means "can result in false-positive interactions") + linear map of the heads; length scale
  per sender type, receiver type and head read from counterfactual attention at a given distance (threshold 0.1); hyperparameters
  chosen by held-out MSE ("the model that minimized the test MSE correctly recovered the ground truth interactions").
- **Limits:** needs labels; one layer; neighbourhood of 50 cells; stromal cells removed for mis-segmentation.
- **For us:** our context query is receiver-agnostic (a learned token plus position, `global_component_config.py`), so a hidden
  receiver's attention cannot depend on its type. That is a plausible reason the 1-layer CosMx map was geometry and only the
  second layer (which reads what the first read) learned a map (typeres, twin overlap 0.01): a hypothesis, tested in S4.

### GITIII

- **Works:** highest within-type variance explained by CCI against COMMOT, HoloNet, NCEM, GAT and a graph transformer on four
  datasets; on CosMx NSCLC that number is **0.0136** (1.4 %): the field's best learned CCC model explains little within type,
  in line with our CosMx ceiling (neighbours explain 4-7 % of the reproducible within-type variance, all within ~100 um).
- **Ablations (NSCLC):** removing cell-type input changes nothing (0.01361 vs 0.01366); removing their GRIT changes costs most
  (0.0097); extra layers give marginal gains; the score plateaus at 50 neighbours.
- **Readouts worth copying:** the influence tensor (per receiver, per sender, per gene contribution, exact because nothing
  nonlinear follows the aggregation); CCI networks by Pearson correlation between a sender type's aggregated influence and the
  receivers' observed state, BH-corrected, Stouffer over slides; a composition-adjusted network strength (subtract `n_A / (k - 1)`).
- **Did not help / weak:** their cell-type-level validation is the correlation of network strength with the Wasserstein distance
  between types, i.e. it rewards proximity (Armingol's caveat); spillover is acknowledged, not tested; 80/20 random cell split.
- **For us:** in our within-type shuffle GITIII responded most (edge Spearman 0.71, magnitude 0.49), but its fit was weak
  (median per-gene Pearson of the state 0.08).

### Steamboat

- **Works:** heads are readable as programmes ("in an environment defined by k, if the centre expresses q, it also expresses v");
  ego / local / global weights per head; global heads separate patient groups (CRC prognosis); replacing its attention graph by
  the plain k-NN graph lowers domain segmentation (their ablation); in-silico transplant: transplanted cells shift towards the
  native cells of the new region (log-fold changes agree).
- **Did not help / limits:** its type x type matrices agree with CellChat (rho 0.76 on one sample) but in our within-type shuffle
  the pair ranking held (0.87-0.95): it is composition; what the shuffle changed was strength (local explained variance halved).
  No distance within a scale (k = 8 neighbours or the whole sample).
- **For us:** the ego / local / global split is what our own-cell skip, context part and section offset already give; the
  metagene head is a readable form of the read-out (S5 option); type x type summaries only with a composition control.

### CellNEST

- **Works:** unnormalised attention scores, because softmax-normalised ones are not comparable across the tissue (their
  Supp. Fig. 14a); five seeds combined by rank product; tanh more stable than ReLU in the scoring.
- **Limits:** needs a ligand-receptor database and treats only "active" ligand-receptor co-expression within 300 um as candidate
  edges; synthetic benchmarks with Gaussian expression and planted LR co-expression; relay networks need many LR pairs on the panel.
- **For us:** comparable-across-tissue strength = the null key's complement (interaction mass); seed ensembles we already found
  necessary (~0.9 at 10 seeds). LR gating stays out of the core (panels cover few pairs; GITIII found 0-2 database pairs on the
  MERFISH panels); LR priors stay for the downstream NicheNet check (T8).

### TERRA

- **Works (authors' ablations):** predicting in latent space (JEPA, EMA target encoder, L1 on latents) rather than counts "is
  well matched to spatial transcriptomics data, which are sparse, noisy"; rank + value tokens beat either alone; a batch metatoken
  (zeroed at inference) integrates assays, removing it collapses mixing; knockouts read against random-gene and housekeeping
  knockouts; during training the number of neighbours is drawn uniformly (0-10) per batch.
- **Limits:** neighbourhood k <= 10 (no long range), niche definitions need expert reading, no interaction readout as such.
- **For us:** key dropout (TERRA's neighbour sampling) makes "hide some senders" a training-time event, which is what the
  attribution plan's occlusion and Sensitivity-n need; random-gene / housekeeping controls for every knockout readout; JEPA only
  as an optional auxiliary (our CosMx target is mostly noise, but a latent target does not create signal).

### scDIVA

- **Works:** separate latent subspaces for type, batch and residual, each with its own encoder and (for type, batch) a classifier,
  priors conditioned on the labels, no adversary; batch-invariant type space by construction; best macro-F1 in 5 of 6 atlases
  (macro-F1 exposes rare classes that accuracy hides); KL warm-up over 50 epochs against posterior collapse.
- **For us:** a context embedding split into a nuisance part (classified to section / FOV / donor), a composition part and an
  interaction part would put the ceiling audit's controls inside the model (S5 option). Milo-style DA on a disentangled context
  embedding is a readout for condition differences (T7). Macro metrics per programme (rare interactions MC20 / MC21).

### SAFFRON

- **Works:** a Matryoshka BatchTopK SAE turns dense embeddings into sparse features (near-zero dead latents; TopK and BatchTopK
  equal; L1 / KL sparsity unstable); score = max |Spearman| of any feature with a known spatial variable, within domains,
  averaged; robustness by within-domain permutation nulls (95th percentile ~0.10-0.14, rising with dictionary size M) and by
  matched M; OMP finds how few features rebuild the variable (two Novae features: R^2 > 0.8 for the tumour axis).
- **Findings:** only Novae (domain prototypes built in) and SpatialFusion beat PCA / gene-SAE on gradients; on the local plaque
  microenvironment (DAM score, microglia density within 100 um, distance to plaque) no SFM beats them; a Novae feature with a
  higher |rho| did not match the variable when drawn.
- **For us:** this is the embedding-level evidence the user asked for (what an embedding holds, not what it reconstructs), with
  ready-made rules against chance (null, matched M, baselines, maps next to numbers). Training-side ideas (Matryoshka nesting,
  TopK codes) are S5 options.

### EV range, CellChat, the reviews

- EV: most vesicle signalling is short (80 % within 40 um in the tumour microenvironment) and shorter where cells are dense:
  supports short windows by default and the existing density-scaled distances (`pe.distance.units: local`); thesis 1.2.
- CellChat: LR classes with different ranges (contact, secreted, ECM); out- / in-degree for sources and sinks (our source / sink
  maps are the spatial analogue); NMF patterns of senders and receivers; its own limits: within-group heterogeneity ignored,
  few benchmarks.
- Reviews: no gold standard, combine validation kinds, simulations rarely realistic, robustness to annotation errors and noise
  should be tested, only 24 % of tools report compute; co-localisation benchmarks bias against long range.

## Our own evidence on the same questions (keep / drop for the new default)

**Keep (worked):** programme masking (the task that needs senders); the per-head distance prior `pe.distance.kind: scales`
with the raw own-cell skip (0.909 +- 0.003, 3 seeds); 300 epochs (M11 0.29 -> 0.96, programme mean 0.730 -> 0.800); 8 heads,
4 layers or 900 epochs as small gains (0.949 -> 0.96); the additive decoder with the context query (0.842, the context part
carries the rings only); the type-residual target on CosMx (0.087 +- 0.002 within section x type vs knn128 0.083); whole sections
with fused / hierarchical attention; IG over attention for senders (mid range 100 % vs 73 %); seed ensembles for maps; the bars
(smoothers, ridges, oracle 0.983, untrained twin) and scoring within section x type.

**Drop from new defaults and sweeps (code stays so thesis tables reproduce):** the long-range mask (round 3b: no gain); entmax /
top-k (no gain, 2.1x time); RoPE (Fourier wins in every arm); hierarchical priors (cost 0.05-0.18); PCA / scVI tokens (drop
drivers: 0.61 / 0.57); channel masking (0.245, no transfer); the residual decoder and the regime loss; relevance rollout and CLS
scores (degenerate target); the bilinear add-on part (0.000-0.001 of the variance); per-cell net flow from one seed as evidence
(seed agreement 0.14 on typeres; the acinar "source" reads as spillover).

## Verdicts

| # | Idea | Source | Verdict | Where in InterScale |
|---|---|---|---|---|
| 1 | Type-residual target | AMICI, GITIII, NCEM; ours | **default on annotated data** (exists); add learned-means option | `section_offset: label_mean`; new `label_mean_learned` |
| 2 | Receiver-aware query (type, own visible genes) | AMICI, GITIII | **adopt** | `context_query.inputs` (new) |
| 3 | Read-out additive in senders (exact influence tensor) | GITIII, AMICI, Steamboat | **adopt**: values additive, routing may be deep | `readout: additive` (new) |
| 4 | Null key ("empty neighbour") | AMICI; attention_plan Stage 4 | **adopt** | `attention.null_key` (planned, not built) |
| 5 | Monotone decay with content-dependent rate | AMICI | **arm** (rate = head scale x sender factor x receiver-type factor) | `pe.distance.content_rate` (new) |
| 6 | Entropy penalty with ramp + L1 on values | AMICI | **arm, only with the null key**; judged by sender / receiver AUPRC, not score | `optim.aux_loss_weights.attn_entropy`, `value_l1` (new; old `SCE_EntropyATT_Loss` has no null, no ramp, all rows) |
| 7 | Same-type keys blocked or replaced by type mean | AMICI, GITIII | **arm on CosMx / vannan** (block first) | `attention.same_type_keys` (new) |
| 8 | Every cell a receiver, no masking | AMICI, GITIII | **arm on annotated data** (5x the targets of 20 % cell masking) | `dataset.all_cell_targets` (new) |
| 9 | Neighbour / key dropout | TERRA | **arm** (makes removals in-distribution) | `attention.key_dropout` (new) |
| 10 | Keys from type only vs expression | GITIII ablation; project goal 2 | **ablation on CosMx**: is sender choice within-type or composition | `context_query.key_inputs` (new) |
| 11 | Ego / local / global with scale weights | Steamboat | **have** (own skip, context part, section offset); add per-head scale shares | readout |
| 12 | Metagene heads (nonnegative rank-1 q / k, gene-space value) | Steamboat | **S5 option**, as the only context path | `head_form: metagene` |
| 13 | Interaction mass instead of softmax weights | CellNEST | **adopt** via null key: `1 - null mass`, influence magnitude | readout |
| 14 | Seed ensembles, rank product | CellNEST; ours | **adopt** for every map | readout |
| 15 | Group ablation + per-gene test, composition-adjusted strength | AMICI, GITIII | **adopt** (model's own masking as removal) | E4 |
| 16 | Segmentation test (receiver near sender vs distant sender) | AMICI | **adopt** as a filter next to the spillover index | E4 |
| 17 | Transplant and knockout with random / housekeeping controls | Steamboat, TERRA | **adopt**; on synth against the generator | E4 |
| 18 | Semi-synthetic planted interactions on real cells | AMICI | **adopt** on CosMx coordinates (fills K20) | E2 |
| 19 | SAE + known spatial variables + nulls + OMP + baselines | SAFFRON | **adopt** | E1 |
| 20 | Matryoshka / TopK on our own context embedding | SAFFRON | **S5 option**, only if E1 shows entangled features | S5 |
| 21 | Nuisance / composition / interaction subspaces with classifiers | scDIVA | **S5 option** (CosMx, vannan) | S5 |
| 22 | JEPA latent prediction, EMA target | TERRA | **defer** (S5): cannot create signal the ceiling says is absent | S5 |
| 23 | LR-gated edges, relay networks, mass action | CellNEST, CellChat | **drop from the core**; LR only downstream (T8) | – |
| 24 | Type x type matrices from attention sums | Steamboat | **drop** unless composition-adjusted (our shuffle) | – |
| 25 | Proximity (Wasserstein) as validation | GITIII | **drop** as evidence (biased against long range) | – |
| 26 | Fixed k / radius neighbourhoods | NCEM, NicheDE, TERRA, GITIII | **keep whole-section keys with learned decay**; a k-NN candidate set only as a cheap arm | – |
| 27 | Density-dependent range | EV paper | **keep option** `pe.distance.units: local` (exists); one CosMx arm | – |

## Design: the interaction read-out (IR)

A mode of the global component built from the context query (`context_only: query`), which already reads the cells' **input**
tokens (the memory, never updated, a cell never reads its own token) through each layer's weights. What changes:

1. **Sender memory (as now).** Each visible cell's input token (its visible genes through the token projection; optional type
   embedding added). Hidden cells are no keys (`visible_keys_only`). The memory is per cell and never mixed, so a contribution
   belongs to one sender.
2. **Receiver query (new).** `context_query.inputs`: any of `position` (today's node encoding), `type` (an embedding of the
   receiver's label, AMICI / GITIII), `own` (a projection of the receiver's **visible** genes: under programme masking the
   receptor markers, so a receptor-positive cell can choose ligand senders). Default `[position]` = today. Only visible entries
   ever reach the query; under whole-cell hiding `own` is the mask fill and `type` is the only receiver information.
3. **Attention per head:** `logit_hij = q_hi . k_hj / sqrt(d) - r_h * phi(d_ij)` with today's scales prior (`r_h`, shape
   gaussian / exponential), plus a **null key** (zero key, zero value, learned logit `c_h`). Option `content_rate`:
   `r_hij = r_h * exp(w_h . k_j) * exp(u_h[t_i])` (sender factor from its token, receiver-type factor; both start at 0, so step 0
   is the scales prior exactly). Separable in i and j, so the fused attention can build it a block of rows at a time.
4. **Read layers:** L in {1, 2}. Each layer reads the memory; the stream's state (query side) is updated with the layer's
   feed-forward block **for routing only**. With `readout: additive` the output is the **sum of the layers' attention outputs**
   (after each layer's output projection, before any feed-forward or norm), so the prediction's context part is
   `sum_l sum_h sum_j a^l_hij W^l_o V^l_h LN(m_j)`: exactly a sum over senders, whatever the routing depth.
5. **Decoder parts** (all exist but the first): `context` = linear map of the additive read-out; `own_cell` = raw own-cell skip
   (synth; off under all-cell targets); `label` = type mean (`label_mean`, fixed or learned); `section` / `group` offsets on CosMx.
   `StepOutput.parts` carries each, so `attribution_shares` reads them.
6. **Influence export:** for sampled receivers, each sender's contribution per gene (`[receiver, top-k senders, gene]` plus the
   rest's sum), the null mass per head, and the effective length scale per head (and per sender type x receiver type with
   `content_rate`). Exact for `readout: additive` with a linear decoder; the decomposition is not an intervention effect (softmax
   renormalises), so E4 and the attribution plan's occlusion stay the check.

**Training modes.**
- synth_spot: programme masking as now (every cell a target, the programme hidden in all cells), query `[type?, own]`, own skip on.
- Annotated real data: `dataset.all_cell_targets: true`: no masking, every cell is a receiver, target = expression minus its
  type mean, query `[type]`, no own skip (the validator refuses `own`, the own skip and the plain encoder with it). The current
  arm (20 % whole cells + discs) stays as the control.

**Regularisers (arms).** `attn_entropy`: the entropy of each target row (null included), mean over heads and layers, weight
ramped 0 -> beta over `ramp_epochs` (AMICI); refused without the null key (without a place for "no interaction" an entropy penalty
forces a false sender). `value_l1`: L1 on the value projections. `key_dropout`: a share `U(0, p_max)` of the visible keys of each
graph hidden per training step (one key vector per graph: the fused path's `hidden_keys`), from a private generator; never at
evaluation.

**Options for the CosMx questions.** `same_type_keys: block` (a pair mask by label equality; null and CLS never blocked; a
receiver with no open key reads the null). `key_inputs: expression | type | none` (keys from the expression tokens, from the type
embedding only, or constant; values stay the expression): if `type` ties `expression`, sender choice is type and distance, i.e.
composition-level CCC; this answers the project's "how cell-type driven is attention" directly.

**Invariants (tests, as for the context query).**
- No hidden value of the receiver reaches its query, the memory or its output (bit for bit, 1-2 layers, every option).
- `readout: additive` + linear decoder: the parts and the per-sender contributions sum to the prediction (rounding).
- Null key: rows sum to 1 with the null; `c_h -> -inf` gives the model without it; null value is exactly zero.
- `content_rate` at initialisation gives the scales-prior model's predictions exactly.
- Off, every new option holds no state and draws no random number (harness IDENTICAL).
- `key_dropout` and `all_cell_targets` never act at evaluation; `all_cell_targets` refuses any path that reads the cell itself.

**Config sketch (names to settle when building).**

    model.global_component.parameters:
      context_only: query
      context_query: {inputs: [type, own], type_key: cell_type_coarse, key_inputs: expression}
      readout: additive            # default "stream" = today (feed-forward and final norm on the output path)
      attention: {null_key: true, same_type_keys: open, key_dropout: 0.0}
      pe.distance: {kind: scales, content_rate: false}
    optim.aux_loss_weights: {attn_entropy: 0.0, value_l1: 0.0}   # optim.attn_entropy.ramp_epochs
    dataset.all_cell_targets: false
    model.decoder.label_mean_learned: false

**Backends.** Dense (synth_spot, 2,501 tokens) and fused (whole CosMx sections; the fused path already reads with separate query /
key / value inputs and builds biases per block). The hierarchy refuses the context query today and stays out. Cost per epoch should
match today's context-query models (no S x S kept); the entropy penalty on fused rows has to be computed inside the block, as the
sparse-key penalty is in the hierarchy.

**Build estimate.** Core (receiver query, null key dense + fused, additive read-out, export, validators, tests) ~1.5 days;
options (content rate, entropy + L1 with ramp, key dropout, same-type block, key inputs, all-cell targets) ~1 day.

## Evaluation additions

### E1: SAFFRON-style probes of the embeddings

- **Embeddings:** the context part (read-out before the decoder), the own-cell part, the full token; baselines PCA of the visible
  expression, PCA of the k-NN-smoothed expression (the smoother as an embedding: our models tie it), a gene-SAE on the expression,
  the untrained twin; other methods' outputs where available (Steamboat embedding, GITIII influence features).
- **SAE:** Matryoshka BatchTopK, M = 4 D (64 -> 256), K / M ~ 6 %, prefixes M/4, M/2, M with equal weights, 100 epochs, Adam 1e-3,
  batch 512, dead-feature resampling every 10 epochs, input noise annealed 0.02 -> 0 (SAFFRON App. B). Fitted on training graphs,
  applied to test graphs; the same M for every embedding (the chance ceiling grows with M).
- **Known spatial variables tau (from config, dataset-agnostic kinds):** `distance_to_label`, `density_of_label(radius)`,
  `layer_field` (a noise-free layer), `obs` (any numeric column), `coordinate_projection`. Proposed per dataset:
  - synth_spot (from `uns['synthetic']` and the `expected` layer): each programme's noise-free response field; driver-side
    variables (distance to the nearest planted sender, sender density within the programme's range, distance to the ring source,
    the stripe / cone coordinate); nuisance: slide, spot density, distance to the tissue edge or a hole.
  - CosMx: distance to the nearest endocrine cell (islet proximity), immune density within 100 um, immune cells within 50 um of an
    endocrine cell (insulitis, T1D), distance to the nearest ductal cell, local density; nuisance: FOV index, section.
  - legnini: distance to the SHH source / floor plate (or a GASTON isodepth, optional); vannan: distance to airway epithelium,
    fibroblast-focus density, IFN-niche score from genes held out of the model.
- **Scores:** max |Spearman| over features within domains (synth: slide; CosMx: section x type; legnini: organoid), averaged;
  1,000 within-domain permutations of tau (95th percentile); OMP curve (cross-validated R^2 against 1-15 features, and all features
  by ridge); per prefix (does M/4 hold type and section, the later features the interaction variables?); the best feature's map
  drawn next to the number (SAFFRON's Fig. 4b lesson; [[show-spatial-maps]]); each feature's top genes through the linear decoder.
- **Why it matters here:** for the classes no model reconstructs (directional M9 / M10, rare MC20 / MC21) it says whether the
  embedding holds the variable and the decoder misses it, or the embedding lacks it; on CosMx it asks whether the context part
  holds islet proximity or insulitis better than the smoother's embedding does.
- Build ~1 day (`evaluation/sae_probe.py`; embeddings from `dimensions.collect_dimensions`, every cell hidden once).

### E2: semi-synthetic CosMx (planted interactions on real geometry and real cells)

AMICI's generator on our data: keep the real coordinates and types of CosMx test / validation sections; for a receiver type R,
split its training cells into a default and an "interacting" state (median split along the first within-type residual PC,
cells drawn from other sections); for each planted pair (S -> R, length scale l), an R cell with an S cell within l gets a profile
from the interacting state, else from the default. Pairs at l = 20, 50, 200 and 500 um, one receiver type without a pair (null),
two effect sizes. Ground truth: interacting receivers, senders within l, the state's genes, l. It answers "would the model have
seen a 200-500 um interaction on CosMx if one were there" (power), next to the ceiling audit's synth control, and fills K20
(lattice only). Models: IR, the current typeres model, the references, GITIII and Steamboat (benchmark env), AMICI if installed.
Build ~0.5-1 day (`evaluation/semi_synthetic.py`, pairs from config).

### E3: AMICI's three tasks and length scales (synth_spot, E2)

Per programme class: receiver AUPRC (interaction mass `1 - null mass` or influence magnitude), sender AUPRC (contribution of each
neighbour), gene AUPRC (genes the planted senders move), each against the untrained twin; learned length scale vs planted range;
the same scores for head-mean attention and IG (attribution plan A1) on the same checkpoints.

### E4: interventions and controls (CosMx, vannan; synth against the generator)

- Group ablation: hide all senders of type S (the model's own key removal), the change per receiver gene; Pearson / Wald against
  the observed residual, BH, Stouffer over sections (AMICI, GITIII); interaction strength = number of significant genes, against
  a distance-matched random ablation of as many cells (composition control).
- Segmentation filter: a gene flagged if not higher in receivers within 20 um of S than in S cells >= 100 um from any R (AMICI),
  next to our spillover index.
- Transplant: put a receiver's visible state into another place of the same section, compare the predicted change with the native
  difference (Steamboat); knockout of a gene in a type against random-gene and housekeeping knockouts (TERRA). On synth_spot the
  generator can recompute the expected responses after the same change: an interventional ground truth (needs the response rules
  callable on modified senders in `evaluation/synth_spot.py`; ~0.5 day).

### E5: maps

Interaction mass and influence magnitude per cell (comparable across the tissue, CellNEST's point), rank product over seeds
(CellNEST) next to our mean-over-seeds ensembles; per-cell seed agreement reported.

### E6 (optional): conditions

Milo-style differential abundance of T1D vs ND on the context embedding, donors as replicates (scDIVA's use of Milo); weak with
three donor pairs, a T7 readout.

## Stages, jobs and decision rules

**Order (recommended):** build E1 first (independent of the model, gives thesis material) and run it on the existing synth and
CosMx checkpoints; build S0 meanwhile; then J1.

**J1 (S1, synth_spot, one job < 8 h, 300 epochs, 2 seeds + twin for ref and IR-2):** `ref` = `lin_dist_skipraw`; `IR-1`
(1 read layer, query `[own]`, null key, additive); `IR-2` (2 layers); `IR-2 + content_rate`; `IR-2 + attn_entropy + value_l1`;
`IR-2 + key_dropout 0.3`. Readouts: the usual tables, E3, E1, time per epoch.

Rules, fixed before the run:
- **R1 score:** the best IR arm within 0.02 of `ref` on the programme-mean centred cosine (mean of 2 seeds), no class mean worse
  by > 0.05.
- **R2 senders from the model itself:** the influence decomposition's top 5 holds the planted sender for >= 0.95 (short) and
  >= 0.90 (mid, M11) of receivers (head-mean attention of `ref`: 0.97 / 0.73; IG 0.99 / 1.00); the twin at chance.
- **R3 receivers:** interaction mass separates receivers from non-receivers (AUROC >= 0.8) for short, mid and sparse classes.
- **R4 length scales** (with `content_rate`): within x1.5 of the planted range for short and mid.
- **R5 model selection:** the arm with the lowest validation loss is within 0.02 of the best R2 / R3 arm; otherwise validation loss
  cannot pick models on real data, and that is reported.
- **Outcome:** R1 + R2 pass -> the IR is Stage M's model for synth_spot and goes to S4; R2 passes but R1 fails -> the IR is kept as
  the readout model beside the score model; both fail -> Stage M keeps the current recipes and only E1-E5 carry over.

**J2 (E1 on CosMx, evaluation only, < 2 h):** typeres seeds + twin, rand50 / cell_l2 if present, baselines.

**J3 (E2, < 8 h):** generation on the cluster, IR vs the current typeres model vs references (GITIII / Steamboat in the benchmark
env as a separate step). Rule: a pair counts as detected when receiver AUPRC is >= 2x its base rate and above the k-NN reference;
report the shortest l and smallest effect detected, and false positives on the null receiver type against the twin.

**J4 (S4, CosMx then vannan, 3 seeds + twin):** IR typeres with `all_cell_targets` (main), the same with the current masking
(control), `key_inputs: type`, `same_type_keys: block`, optionally `units: local`. Rules: within section x type >= knn128 (0.083) and
>= current typeres - 0.005; per-cell interaction-mass seed agreement >= 0.5 (current flow 0.14); `key_inputs: type` within 0.002 of
`expression` is reported as "sender choice by type and distance only". Then E4 / E5 and the attribution plan's A2-A6 on these
models.

**S5 (optional):** only if E1 shows the context part holds the variables but spread over many features: a TopK context code or a
Matryoshka loss on the context embedding; nuisance subspaces for FOV / section on CosMx; metagene heads as the only context path.

## Relation to the attribution plan

- A0 gains a method `decomposition` (the exact per-sender influence of an additive read-out), cheapest of all; A1 compares it with
  IG, DeepLIFT and occlusion on J1's checkpoints (same job outputs).
- Stage M waits for J1's outcome; if the IR passes, its recipes replace the ones in the attribution plan.
- A6 (is the learned neighbourhood function linear?) gets a sharper form: with values additive by construction, the remaining
  nonlinearity is routing; `key_inputs: type` vs `expression` on CosMx tests whether routing uses sender content at all.
- A7 (no labels): `context_query.inputs: [own]` and clusters for `type` (GITIII: type input made no score difference).

## Not carried forward (with the reason)

- LR-gated edges and relay networks (CellNEST), mass-action LR scores (CellChat): panel coverage, prior dependence; LR stays a
  downstream check.
- Type x type matrices from summed attention without a composition control (Steamboat; our shuffle).
- Proximity to other types as validation of interactions (GITIII's Wasserstein test).
- Entropy penalties without a null key; sparse normalisers (entmax, top-k: no gain here, 2x cost).
- Fixed radius / k as the model's reach (NCEM, NicheDE, TERRA, GITIII): the reach is learned per head and reported instead.
- JEPA as the main objective: a gene-space prediction is what the readouts decompose.
- From our own runs: long-range mask, RoPE, hierarchical priors, PCA / scVI tokens, channel masking, residual decoder, regime loss,
  relevance rollout and CLS scores, the bilinear add-on part, single-seed per-cell flow as evidence.

## Open decisions (user)

1. Receiver query on CosMx / vannan: the type label in the query (AMICI / GITIII)? My recommendation: yes, `[type]` under all-cell
   targets, `[type, own]` where genes are visible.
2. All-cell targets as the main CosMx objective (every cell a receiver, no masking), with today's masking as the control?
3. Same-type keys: run `block` on CosMx (it removes homotypic interactions by design, as AMICI does)?
4. E2 design: real profiles on CosMx coordinates (recommended), synth_spot's rules on CosMx coordinates (geometry only), or both?
5. E1's spatial variables per dataset: the lists above are my proposal; the biology choice is yours (CosMx: islet proximity,
   immune density, insulitis; vannan: airway, fibrotic foci).
6. Order: E1 first on existing checkpoints (recommended), or the IR build first? Which synth (`lin_dist_skipraw`, 300 epochs) and
   CosMx (typeres s42-s44 + twin) checkpoints are still on the cluster?
7. Install AMICI (github.com/azizilab/amici) in the benchmark env for E2?

## References

- Hong J, Desai K, et al. AMICI: Attention Mechanism Interpretation of Cell-cell Interactions. bioRxiv 2025.09.22.677860.
- Xiao X, Zhang L, Zhao H, Wang Z. Inferring spatial single-cell-level interactions through interpreting cell state and niche
  correlations learned by self-supervised graph transformer (GITIII). Nat Mach Intell 2025, doi 10.1038/s42256-025-01161-0.
- Liang S, Tang J, Wang G, Ma J. Steamboat: attention-based multiscale delineation of cellular interactions in tissues. bioRxiv
  2025.04.06.647437.
- Zohora FT, Paliwal D, et al. CellNEST reveals cell-cell relay networks using attention mechanisms on spatial transcriptomics. Nat
  Methods 2025, doi 10.1038/s41592-025-02721-3.
- Birk S, Sanian MV, et al. Multi-scale modeling of human tissues from spatial transcriptomics with TERRA. bioRxiv
  2026.07.29.741565.
- Rapolu V, et al. scDIVA: semi-supervised integration and fine-grained annotation of tumor-immune single-cell atlases. bioRxiv
  2026.08.07.743520.
- Handa D, et al. Evaluating the ability of spatial transcriptomics foundation models to learn multi-scale spatial variation
  (SAFFRON). bioRxiv 2026.08.01.742217.
- Colombo F, Nimkar K, et al. Exploring the spatial limits of extracellular vesicles-mediated intercellular communication. J
  Extracell Vesicles 2025.
- Jin S, Plikus MV, Nie Q. CellChat for systematic analysis of cell-cell communication from single-cell transcriptomics. Nat Protoc
  2025, doi 10.1038/s41596-024-01045-4.
- Cesaro G, Nagai JS, et al. Advances and challenges in cell-cell communication inference. Brief Bioinform 2025, 26(3), bbaf280.
- Armingol E, Baghdassarian HM, Lewis NE. The diversification of methods for studying cell-cell interactions and communication.
  Nat Rev Genet 2024, doi 10.1038/s41576-023-00685-8.

## Second round (2026-11-21): decisions, the reviews, SAFFRON's analyses, the generator

### The user's decisions

| # | Question | Answer | Consequence |
|---|---|---|---|
| 1 | Type label in the receiver's query (CosMx, vannan) | yes | `context_query.inputs: [type]` (plus `own` where genes are visible) |
| 2 | All-cell targets | asked what the alternative is | three modes below; recommendation: mode (c) as the main arm, (a) as the control |
| 3 | Same-type block arm | yes | `same_type_keys: block` in J4 |
| 4 | E2 design | asked how | the generator below (count spike-in on real data; AMICI's swap as the second variant) |
| 5 | E1 spatial variables | my choice | lists below, position-only axes first |
| 6 | Existing checkpoints | avoid them; build the probes | E1 runs on J1 / J4 models only (no J2 on old checkpoints) |
| 7 | AMICI | hard to install | no AMICI package run; an "AMICI-like" IR arm (1 read layer, query `[type]`, 50 nearest other-type cells as keys, monotone decay, null key, entropy ramp) stands in, named as such; GITIII and Steamboat (benchmark env) stay the external baselines |

### Training modes on annotated data (question 2)

Masking today does two things at once: it picks the **targets** (the hidden cells) and removes **senders** (a hidden cell is no
key). The context query's read already keeps a cell's own token out of its prediction (the memory is never updated, so no path
leads back), which lets the two be separated.

- **(a) Masking (today):** 20 % of cells hidden whole + 60 um discs. Hidden cells are targets and no keys; visible cells are keys
  and no targets. ~20 % of cells give a loss per step; the discs make blind areas, so some receivers must read beyond their
  nearest neighbours.
- **(b) All-cell targets, nothing hidden (AMICI / GITIII):** every cell is a target and a key; its own token is never read by its
  own query; the type mean is the fixed part, the context part explains the residual. 5x the targets per step. Risk: the nearest
  neighbour is always visible, so nothing pushes the model to read far; and the receiver's own transcripts leaked into a
  neighbour's segmentation are readable (as they already are in (a)).
- **(c) All-cell targets, key-side masking (recommended main arm):** every cell is a target; each step a random share of cells and
  discs is removed **as keys only** (the `hidden_keys` vector of the fused path; the key-dropout machinery), so receivers inside a
  disc read only beyond it. Keeps 100 % of the targets and the long-range pressure of (a). Validator: only with a read that is
  blind to its own cell (context query, no own skip, no `own` query input under it).

J4 runs (c) as main and (a) as the control; (b) is (c) with the key-side share at 0, one cheap arm.

### The two reviews, read in full (the user asked whether the benchmarking content was used)

On the first pass I had only skimmed them. Read in full, they summarise four benchmark studies (ESICCC / Luo et al., Liu et al.,
Zhang et al., Xie et al.) whose methods sit in Cesaro's supplement, which is not in the folder; the main text gives rankings and
the lessons. What is useful for us:

1. **Rankings disagree between benchmarks** (CellChat top in two, mid-to-low in a third; SingleCellSignalR second to last in one),
   because metrics, versions, parameters and LR resources differ; the best tools of one study are often absent from the others.
   For us: report several metrics side by side (score, sender / receiver / gene AUPRC, length scales, stability), never one
   ranking; include the external methods others found strong (GITIII, Steamboat) on the same data.
2. **Robustness is its own axis** (Zhang et al.: batch effects, dropout, **errors in the cell-cluster annotation**; rankings differ
   again from accuracy). Now that the type enters the query and the target, J4 gets two label-robustness arms: coarse vs fine
   labels (`cell_type_coarse` vs `CellTypes_max` on CosMx) and 10 % of training labels shuffled within section. AMICI says the same
   from the method side: the annotation's resolution decides which interactions can be seen.
3. **Low overlap between tools** (LIANA) and data-driven methods that change with the seed (Armingol: "similarity metrics to assess
   and stabilize results from separate runs"): our seed ensembles and per-cell / per-type seed agreement are the right reply; report
   agreement for every map and table.
4. **Validation by classifying samples by phenotype** (Armingol, Tensor-cell2cell and LIANA+): communication features that separate
   conditions. For us: does the per-section influence summary (sender type x receiver type x gene) separate T1D from ND sections
   better than composition alone? Weak with 6 sections; a T7 readout, not a gate.
5. **Multi-sample factorisation** (Tensor-cell2cell: tensor factorisation of CCC scores over samples x LR pairs x senders x
   receivers; MOFAcell / DIALOGUE multicellular programmes): the IR's influence tensor aggregated per section (section x sender
   type x receiver type x gene) is exactly such a tensor; its factors are "communication programmes" that can be tested against
   condition. Cheap once the export exists (E6, extended).
6. **Curated ground truth lists:** CITEdb, Cellinker, a tumour-immune CCC database and an **idiopathic pulmonary fibrosis CCC
   database** (Armingol ref. 194): a positive-control list for vannan (lung fibrosis), next to NicheNet for T8.
7. **Simulators:** scMultiSim simulates spatial CCC with ground truth; an alternative to our generators if a reviewer asks for an
   external simulator.
8. **Co-localisation benchmarks bias against long range** (Armingol; already in the first round) and **compute is rarely
   reported** (24 % of tools): every job reports time per epoch and per readout (already our rule).

Nothing in the reviews changes the model design; they change the evaluation (items 1-6).

### SAFFRON: the ideas, not the package (user, 2026-11-21)

The user does not want the package reimplemented; the interpretation idea, the early-vs-late finding, and the 1-D gradient and niche
analyses are what to take. A small Matryoshka BatchTopK SAE (~100 lines of torch in `evaluation/sae_probe.py`) is enough.

**What SAFFRON does and finds, precisely.**
- *Interpretation:* SAE features of an embedding are scored against a known spatial variable (max |Spearman| within each spatial
  domain, averaged); the best feature is **drawn** next to the variable (Fig. 2b/c, Fig. 4b: a feature with a higher |rho| did not
  match the plaque distance when drawn); OMP finds the fewest features that rebuild the variable (two Novae features, R^2 = 0.81);
  genes are then plotted **against the reference axis and against the axis rebuilt from the features** (Fig. 3e/f: S100A6 and PTPRO
  along tumour edge -> core, SCD and FCGBP across the boundary): the embedding's axis is read through genes.
- *Early vs late features:* the Matryoshka loss rebuilds the embedding from the first M/4, M/2 and M features, so early features
  must carry the dominant variation and later ones finer variation. In the paper this is the design intent rather than a measured
  result; the concrete case is Fig. 3d: the two features that rebuild the tumour axis split it **by domain** (feature 210 the
  stroma -> tumour-edge axis in the stroma, feature 45 the edge -> core axis in the tumour), one from the first prefix (45 < 64) and
  one from the last block (210 > 128). The appendix ablation: Matryoshka SAEs have near-zero dead features where vanilla ones do not.
- *1-D gradients:* six datasets with a 1-D axis (tumour-stroma, kidney radial, cortical depth, crypt-villus, crypt-surface), the
  axis from GASTON's isodepth or a given pseudospace, scored **within domains**. Key control (Fig. S2): the same GI sections scored
  against a **position-only** axis (MorphoGAM's signed distance to a curve through the mucosa) drop sharply for every model and for
  PCA (Novae TFS 0.87 -> ~0.46, PCA 0.86 -> ~0.44): an axis derived from expression rewards any expression embedding.
- *Niches / local microenvironments:* plaque metrics at cell level (DAM score = mean of 26 signature genes; microglia within 100 um;
  distance to the nearest plaque), scored per replicate; no SFM beats PCA / gene-SAE.

**E1, revised (replaces the first version's variable lists; the scoring rules there stand).**
1. *Embeddings:* the IR's context part (and own-cell part) from J1 / J4, their untrained twins, PCA of the visible expression, PCA of
   the k-NN-smoothed expression (the smoother as an embedding), a gene-SAE on the expression. No old checkpoints.
2. *1-D axes, position-only first* (the MorphoGAM lesson), expression-derived second:
   - synth_spot: radial distance to each ring source (M7 / M8), the stripe coordinate (M9), the cone's angular coordinate (M10),
     distance to the nearest M11 source; each **within its region** (the region markers define domains).
   - CosMx: **signed distance to the islet boundary** (islets = connected groups of endocrine cells on the section's graph; inside
     negative), distance to the nearest duct cell; within section x niche (niches = k-means of 30-NN type composition, fitted on
     training sections).
   - vannan: distance to the nearest airway epithelial cell, distance to the nearest CTHRC1-high fibroblast group (fibrotic focus),
     within the given niches (`niche`, `niche_c`).
   - legnini: distance to the floor plate (the FOXA2 / SHH cluster: label-derived, said as such), within organoid; a GASTON isodepth
     only as the expression-derived second axis.
3. *Local microenvironments* (SAFFRON's plaque analogue): CosMx immune cells within 50 um of an endocrine cell (insulitis, T1D),
   immune density within 100 um; vannan IFN-niche score from genes the model never sees as inputs, density of lymphoid aggregates;
   synth_spot short-range sender density within 150 um, sparse source within 2,500 um.
4. *Nuisance controls:* FOV index, section, local density, distance to the tissue edge: a feature tracking these is technical or
   geometric, and the probe says so.
5. *Readouts:* per variable the best feature's |rho| against its permutation null and the baselines; its map drawn next to the
   variable (one slide, tissue only); the OMP curve; **genes along the axis**: the decoder's genes for the selected features,
   plotted against the reference axis and the axis rebuilt from the features (SAFFRON Fig. 3e/f; on legnini the SHH targets along
   the floor-plate distance, on CosMx INS / GCG-proximal genes along the islet boundary).
6. *Early vs late, made into a test of our local / global question:* for every SAE feature, its prefix block (first M/4, next M/4,
   last M/2), the length scale of its activation map (`gene_scales.ell_um` applied to the feature as if it were a gene) and the kind
   of variable it tracks best (1-D axis / domain vs local microenvironment vs nuisance). Prediction if the nesting orders by scale:
   early features are wide (domains, gradients), late ones narrow (microenvironments). Spearman of feature index vs `ell_um`, per
   embedding, against the twin. If it holds, the nesting gives a label-free local / global split of one embedding, and S5's
   Matryoshka loss on the model's own context embedding becomes a real candidate (ordered context dimensions instead of two parts).
7. Cost: the SAE is trained once per embedding (minutes); scored on test graphs; runs inside J1 / J4 after training.

### E2 made concrete: planted interactions on real CosMx data (question 4)

**Main variant: count spike-in on real cells (exact control of range and size).**
1. Start from a CosMx section's real counts (`layers['counts']`), coordinates (global um) and `cell_type_coarse`; nothing moves.
2. Planted pairs from config (example): Immune -> Ductal, 20 um step (contact); Endothelial -> Acinar, exponential decay 50 um;
   Endocrine -> Acinar, exponential decay 200 um; Fibroblasts -> Ductal, a field (sum over sender cells within 500 um, saturating);
   one receiver type with no pair (null). Two effect sizes (fold change 1.5 and 2 at the kernel's maximum).
3. Response genes per pair: 6-10 panel genes expressed in R, with little spatial structure within R in the real data (low
   within-type Moran's I), **not** markers of S (so segmentation spillover from S cannot imitate the response), different genes per
   pair; a matched set of untouched control genes.
4. Spike-in: for each R cell i and response gene g, `x_ig += Poisson(beta * f(d_i) * mu_Rg * s_i / s_R)` with `f` the pair's kernel
   of the distance to the nearest S cell (or the field), `mu_Rg` R's mean count, `s_i` the cell's total counts: count noise like the
   real data's, the cell's own state, FOV offset and spillover untouched.
5. Preprocess exactly as the CosMx inputs (`make_inputs.py`: totals, log1p), keep the split; store the truth in
   `uns['synthetic']`'s format (pairs with senders, receivers, weights, programme, range class) so every existing planted-pair
   readout reads it unchanged.
6. **Model-free check first:** the ceiling plan's fields reader (stage 1e) must find the planted far gain on the response genes at
   200 / 500 um and nothing on the control genes: it validates the generator and gives the model-free power on the same file.
7. Then J3: IR (mode c), the current typeres recipe, the references, GITIII and Steamboat (benchmark env). Readouts: receiver /
   sender / gene AUPRC, learned length scale vs planted, the shortest range and smallest effect detected, false positives on the
   null receiver type and the control genes against the twin.

**Second variant: AMICI's swap.** Split R's training cells into a default and an "interacting" state (median split along the first
within-type residual PC) and replace an R cell's whole profile by a cell of the matching state from another section. A more
realistic state change (many genes, real covariance) but less control: the swapped profile brings its origin's FOV offset and
spillover, and the effect size is whatever the split gives. Built only if the spike-in reads cleanly and a "whole-state" test is
wanted.

**Why it matters:** it answers "would the model have seen a 200-500 um interaction on CosMx if one were there?" with CosMx's own
geometry, composition, segmentation and noise, which synth_spot (lattice, K20) cannot.

### Changes to the jobs

- J1 unchanged, plus the "AMICI-like" arm; E1 (revised) runs inside it on its own models.
- J2 (E1 on old checkpoints) **dropped** (decision 6).
- J3 = E2's spike-in generator + its model-free check + the model runs.
- J4 adds: mode (c) main / (a) control / (b) one arm; `same_type_keys: block`; label robustness (coarse vs fine; 10 % labels shuffled
  within section); after training E1, E4, E5 and the per-section influence tensor for E6.
- E6 extended: per-section influence tensor -> tensor factorisation (communication programmes) and the T1D vs ND separation test
  against composition-only features (T7; 6 sections, reported as weak).

### Still open

- None blocking S0 or E1. The positive-control list for vannan (the IPF CCC database) has to be fetched when vannan is read.

## Build (2026-11-21): what exists, the jobs, the checks before submitting

The user (2026-11-21): implement what these runs need, test everything discussed, use several GPUs through arrays, make sure the
results are proper and can be evaluated; the attribution plan follows.

### Library (fork, main since 2026-11-21: c9a43aa read-out, 90ae83d probes and spike-in)

- `attention.kind: readout` (`ReadoutAttention`, a `FusedAttention`): null key (`readout.null_key`), content-dependent decay rate
  (`content_rate`: the key's factor and the receiver type's factor on the distance bias, log factors clamped to +-8, scaled before the
  mask so a blocked pair never meets an infinite logit), same-type block (`same_type_keys: block`), key dropout (`key_dropout`), the
  row entropy for the penalty, recording for the exact decomposition. `context_readout: additive` (the reads summed: the output is a
  sum over senders, exact per-sender influence through the linear decoder), `context_query.inputs` (`type`, `own`, `position`),
  `context_query.key_inputs` (`expression`, `type`, `none`), `dataset.score_all_cells` (modes b / c), aux losses `attn_entropy`
  (ramped) and `value_l1`. Validator `_validate_interaction_readout`. Checkpoint tags `ro-additive`, `q-<inputs>`, `k-<keys>`,
  `attn-readout-n-r-st-kd<p>-vk`, `all`; the two new aux losses tag as `ent` / `vl1` (a full name pushed an arm to 258 characters).
- `evaluation/interaction_readout.py` (exact influence, planted scores, cell maps, head windows, type contributions by
  decomposition and ablation with a shuffled-label control, AMICI's segmentation flags), `evaluation/sae_probe.py` (E1),
  `evaluation/semi_synthetic.py` (E2). Tests: `test_interaction_readout.py`, `test_interaction_readout_training.py` (the three modes
  train), `test_sae_probe.py`, `test_semi_synthetic.py`.

### Cluster jobs (`notebooks_and_scripts/slurm/November/`, flat)

`ir_common.py` (the evaluation hook, tables as CSV under `RESULTS_DIR/ir_tables/<run>/`, maps under `ir_maps/`), `ir_synth.py` (J1),
`ir_cosmx.py` (J4, and J3 with `IR_DATA=semi`), `make_ir_synth_yaml.py`, `make_ir_cosmx_yaml.py`, `make_spiked_inputs.py`,
`semi_gitiii.py`, `ir_split.py`, `ir_read.py` (the decision rules), `ir_job.sbatch`, `ir_submit.sh`; with `hier_common.py`,
`make_cosmx_tiers_yaml.py`, `sweep_common.py`. One array task per run (`INTERSCALE_ARRAY=1`, sweep_common's `ARRAY_TASK`), a GPU check
that resubmits its index excluding a B300 node, a report job after the array (`afterany`). Results `MA/21_11_results/ir_{synth,cosmx,semi}`.

| Job | Runs | Arms | Per task |
| --- | --- | --- | --- |
| J1 `synth` | 14 | ref, ir1, ir2, ir2_rate, ir2_pen, ir2_kd, amici_like x seeds 42, 43; twins of ref, ir2 | 1 GPU, 11 h limit |
| J4 `cosmx` | 26 | typeres, ir_c, ir_a, ir_b, ir_c_ktype, ir_c_knone x 3 seeds; ir_c_same, ir_c_rate, ir_c_pen, ir_c_kd, ir_c_local, ir_c_fine, ir_c_noisy, amici_like x 1; twins of typeres, ir_c | 1 GPU, 11 h |
| J3 `semi` | 7 + spike + GITIII | typeres, ir_c x 2 seeds; ir_a, ir_c_rate, amici_like; GITIII 20 epochs (benchmark env) | spike CPU 2 h, GITIII 8 h |

CosMx arms: FOVs shuffled within sections (`split_fov0`), NegPrb dropped, 300 epochs, 1 mm crops, 8 per section and epoch in every
arm (the read-out arms first had 4, copied from the tiers job's fused arm; changed so a gap between typeres and the IR is not a
difference in training steps).

### Checks run before handing over (no model on real data; toy-sized modules on the CPU as in the tests)

- Yamls written, `load_arms` + `_validate` + masking check per arm, unique checkpoint names, longest 244 of 250, run order equal to
  the `*_runs.txt` the array indexes (a scratch copy of the check: every arm as `sweep_common.preflight` sees it, without data).
- `ir_common`'s two hooks end to end on toy modules (read-out arm, reference arm, both twins), `ir_read.py` on their tables,
  `make_spiked_inputs.py` on a toy CosMx-shaped file. Full test suite.
- Bugs these found, all fixed: (1) `ir_cosmx.py` imported `FOV_KEY` / `SPLIT_SEED` at module level, which `sweep_common.configure`
  refuses (every CosMx task would have stopped at start); (2) the E1 domain labels added numpy string arrays (every run's
  evaluation would have failed after training); (3) in the baseline run the baselines' E1 tables replaced the model's; (4) modes b /
  c lost hier_common's standard tables (its references score hidden cells; the hook now scores hidden cells there, which does not
  change a prediction); (5) `ir_read.py` needed `tabulate`; (6) a tie with the untrained twin counted as a detection; (7) the spike
  could pick NegPrb probes as response genes (the runs drop them); (8) the `field` kernel saturated at 1 for every receiver (20
  senders in reach), a constant the within-type target removes: now `1 - exp(-ln 2 S / median S)` over every sender within 3 l.

### Deviations from the plan, with the reason

- **J3's rule on AUROC, not AUPRC.** `planted_scores` draws as many negatives as receivers (base rate ~0.5), so "AUPRC >= 2x the base
  rate" cannot be met. The call: receiver AUROC >= 0.6 and strictly above the k-NN reference (`ir_planted_refs`: the receivers' own and
  their 32 neighbours' within-type residual on the programme's genes, every receiver-type cell) and the untrained twin; a `null:` pair
  that passes is a false positive.
- **Negatives within the receiver type** on the spiked file (`planted_scores(negatives=)`): the query holds the type, so receivers
  against every other cell would score the type. GITIII's readout compares within the type too.
- Mode (b) trains without hidden cells and without `visible_keys_only` (the validator refuses it with nothing hidden); its disc-masked
  scores read hidden cells as keys it never saw hidden. Read (b) on the clean pass.
- hier_common's attention readouts (sender lift, pair AUPRC) skip a context-query model by design; the read-out arms' planted senders
  come from the exact decomposition (`ir_planted`), the reference's from IG in the standard programme evaluation.

### Not built yet

E4's transplant and gene knockouts against random / housekeeping genes, the synth_spot interventional truth, Stouffer over sections
(the type tests pool the test cells), Steamboat on the spiked file (no per-programme readout), E6 (tensor factorisation, Milo), the
vannan arm of S4 (no loader / preprocessed file yet).

### Submit

Copy the November files above to `MA/scripts/` (the sbatch and `ir_submit.sh` to `MA/slurm/`), install the fork's main (pull, editable) in the
`interscale` env, then `cd $MA && bash slurm/ir_submit.sh synth` (and `cosmx`, `semi`). `MAX_PARALLEL` caps concurrent tasks (default
14). A task resubmitted after a B300 node finishes after the report: run `STEP=report` again (minutes).

## Results (2026-11-22): J1, J4, J3 read

Results: `MA/21_11_results/ir_{synth,cosmx,semi}` (`ir_read.md` = the rules, `results.md` = the standard tables, `ir_tables/<run>/`).
The short version is in notes.md (2026-11-22). Centred cosine throughout.

### J1, synth_spot (2 seeds, 300 epochs)

| arm | programme mean | mid (M11) | dense recv | top-5 short / mid | val loss | s / epoch |
|---|---|---|---|---|---|---|
| ref (`lin_dist_skipraw`) | **0.805** | 0.894 | 0.671 | (attention 0.97 / 0.73; IG 0.99 / 1.00) | 0.052 | 2.3 |
| ir1 | 0.727 | 0.551 | 0.019 | 1.00 / 0.60 | 0.110 | 3.3 |
| ir2 | 0.757 | 0.719 | 0.019 | 1.00 / 0.99 | 0.073 | 5.5 |
| ir2_rate | 0.766 | **0.927** | 0.019 | 1.00 / 1.00 | 0.063 | 5.7 |
| ir2_kd | 0.761 | 0.751 | 0.019 | 1.00 / 0.99 | 0.074 | 5.5 |
| ir2_pen, amici_like | not evaluated | | | | 0.101 / 0.117 | 6.0 / 3.6 |
| ir2 untrained | 0.135 | | | 0.60 / 0.07 | | |

- **R1 fails** (0.766 vs 0.805; worst class: dense, -0.13). The loss sits in M19-M21: the 200-500-receiver sparse-response programmes
  whose receivers respond only **beyond** 1-3 mm of the nearest source. Within receivers every IR arm gives the same -0.129 / -0.443 /
  0.628 (the raw own-cell skip's solution), the reference 0.54 / 0.57 / 0.90. A threshold needs a nonlinearity after the aggregation,
  which `context_readout: additive` removes by design (the reference keeps its feed-forward block on the output path). This is the price
  of the exact decomposition, measured.
- **R2 passes** (ir2, ir2_rate, ir2_kd): the decomposition matches IG for short and mid senders with no gradient. Gene AP 0.97-0.98.
- **R3 fails** on the sparse class only (no arm, the reference included, solves sparse receivers: within-receiver 0.466 for all). The
  interaction mass does not separate receivers (0.55-0.66): null mass ~0.2 everywhere, the null key is a constant sink without the
  entropy penalty, and the entropy arms were not evaluated.
- **R4 fails**, partly by its estimator: `range_um` is the influence-weighted mean distance over all keys (long tail; twin 1,183 um for
  short). Read the radius holding 50 % of the influence instead.
- **R5** by hand: lowest IR val loss = ir2_rate = best R2 arm (the reader's val_loss column was empty).
- Outcome by the rule fixed before the run: **R2 passes, R1 fails -> the IR is the readout model beside the score model**; on synth
  Stage M keeps `lin_dist_skipraw` for scores and adds ir2_rate for the per-sender readout.

### J4, CosMx (3 seeds for typeres, ir_c, ir_a, ir_b, ir_c_ktype, ir_c_knone; one seed for the rest)

Within section x coarse type (`ir_read.md`): clean = every cell a target, every other cell a key; masked = hidden disc cells.

| arm | clean | masked | raw SCE (masked) | note |
|---|---|---|---|---|
| knn128 / knn32 / knn8 | 0.109 / 0.096 / 0.078 | 0.089 / 0.074 / 0.052 | 0.439 (nearest_128) | the bar |
| typeres | – | 0.092 | 0.870 | |
| ir_c (mode c) | 0.137 | 0.096 | 0.451 | J4 rule passes |
| ir_a (mode a, masking) | 0.135 | 0.097 | 0.491 | |
| ir_b (mode b, nothing hidden) | 0.132 | (0.121) | 0.465 | masked not comparable (reads hidden cells as keys) |
| ir_c_ktype | **0.141** | 0.098 | **0.436** | maps most stable (sent 0.83, reach 0.79) |
| ir_c_knone | 0.138 | 0.097 | 0.440 | routing by distance alone |
| ir_c_same | 0.103 | 0.079 | 0.491 | same-type keys carry a quarter of the gain |
| ir_c_rate / kd / local / noisy | 0.138 / 0.138 / 0.137 / 0.138 | 0.096 / 0.094 / 0.099 / 0.096 | | no effect; label noise tolerated |
| ir_c_pen | 0.127 | 0.068 | 0.432 | entropy pushes towards the type mean |
| ir_c_fine | (0.150) | (0.125) | | not comparable: fine-type mean scored within coarse type |
| amici_like | 0.091 | 0.056 | 0.478 | below the smoother |

- **Keys:** type only >= none >= expression (0.1406 / 0.1384 / 0.1369, seed SD <= 0.001). Sender expression in the keys adds
  nothing; the reader's "no" comes from a rule that tests a tie. For A6 / project goal 2: routing needs distance and, by 0.002, sender
  type; sender expression enters through the values only.
- **Maps** are seed-stable (ir_c received 0.86, mass 0.73, sent 0.65; typeres' flow 0.14), but stability is not interaction: the twin's
  maps are not yet compared.
- **Type tables** are heat-shock / immediate-early genes and hormone / enzyme spillover, unstable across arms: no CCC reading.
- **E1:** the trained context embedding ties knn-PCA and is at or below its untrained twin on composition variables; the early-vs-late
  test is negative for every embedding, the twins and PCA included (a property of the SAE).
- **Bar missing:** the +0.028 clean gain is over a raw smoother. Before it is called interaction: a fitted ridge on neighbourhood means per
  type and an FOV offset reference (fov_depth is the context embedding's best-tracked variable).
- Outcome by the rule: **ir_c passes**; Stage M's CosMx model becomes the IR in mode c, with `key_inputs: type` as the candidate default
  (best score, SCE and stability; one key input fewer).

### J3, spiked CosMx (2 seeds for ir_c / typeres)

- Generator: model-free Spearman of the receivers' within-type response-gene residual with the kernel 0.05-0.25 (planted), 0.04 / -0.04
  (null pairs): it works, the signal is small.
- Calls (`receiver_auroc >= max(0.6, k-NN reference, twin)`): ir_c 3 of 6 callable pairs (Endothelial>Acinar@exp50 both effects,
  Beta>Acinar@exp200 x2), ir_c_rate 2, ir_a and amici_like 0. **Not evidence:** the receiver truth (any receiver-type cell with a sender
  in reach) covers 94 % / 81 % / 100 % of the receiver type for exp50 / exp200 / field, so the negatives are cells far from any sender
  (geometry; the twin's interaction mass reaches 0.86-0.87); 48 + 48 cells per programme (SE ~0.06). The clean test, contact
  Immune>Ductal@20 um (25 % receivers, own-residual AUROC 0.67), is missed by every arm. Gene AP shows some learned response (Beta>Acinar
  x2: 0.17 vs twin 0.011). GITIII did not run past preprocessing.
- E1 on the kernels cannot tell planted from null pairs (null 0.30-0.54 vs planted 0.29-0.61).

### Next (proposed, nothing started)

1. **Entropy arms on synth:** read the `.err` of the four array tasks (ir2_pen, amici_like x 2 seeds), fix the synth-only evaluation
   failure, evaluate the existing checkpoints (STEP=report afterwards). Only this arm tests whether the null key can mean "no interaction".
2. **Readout fixes (evaluation only, existing checkpoints):** range as the 50 %-influence radius; J3 with graded truth (Spearman of each
   receiver-type cell's influence on the response genes with its kernel, every cell, not 48) and a response-gene within-type score
   against knn; `ir_read.py` val_loss and the key rule (`type >= expression - 0.002`).
3. **CosMx bars:** a fitted ridge on k-NN neighbourhood means per type (`FittedReferences(k_nearest=...)`) and an FOV group offset
   reference on the same clean / masked passes; the twin's maps next to the seed agreement.
4. **Thresholds without losing the split:** a receiver-level (or gene-wise) gate on the additive context, `y_ctx,i = g_i * sum_j c_ij`
   with `g_i` from the stream state: still exactly a sum over senders per receiver. One synth arm against ref on M19-M21; switchable, off
   by default.
5. The attribution plan's Stage M can start with the outcomes above (synth: ref scores + ir2_rate readout; CosMx: ir_c_ktype), or wait for
   items 1-3. The user decides.

## E7: gene rankings of interaction (2026-11-22)

The user (2026-11-22), after the J1 / J3 / J4 read: "I feel like you are still too focused on predicting gene expression. The model should find genes that
interact, i.e. show some kind of variability and respond to others. We need a ranking to judge how well the model worked." Then: "Write and build it."
They agreed that genes that only vary in space (synth_spot's spatially variable and region genes, spillover on CosMx) count as negatives, not hits.

### Design: three rankings, by intervention

Built on `interpretation.py`'s interventions: receivers keep their own rows, only other cells change, the score is a target's centred cosine within
the receiver group (cell type, or every cell without types), pooled over graphs, so a level or scale error of the model cancels. Per receiver group:

1. **Responders** (`gene_ranking.context_flows`, kind `all_state`): every other cell's visible row swapped with another cell's of the same label; a target's
   drop is what the model took from the arrangement of the other cells' states, beyond composition. A gene scores only if it varies within the group
   *and* the model explains that from other cells. `responder_cross` takes off the part the **same gene in other cells** carries (the drop when that gene
   alone is permuted): smoothness, regional states, spillover. With the target hidden in every cell (channel folds) that pair is not read.
2. **Drivers** (`interpretation.gene_flows`): one sender gene permuted within label among the other cells; the driver score is its largest drop over
   the other targets (`top`; the planted drivers move 1-3 targets each), the mean (`influence`) beside it.
3. **Pairs** sender gene -> target gene: the same drops, `sender != target`.
4. **Sender types** (labelled data): `identity` (a type's cells take random other-type profiles) minus `identity_null` (the same on as many nearby cells
   of other types).

**Bars** (model-free, same cells, `BASELINE_RUN` only): within-type Moran's I (spatially variable genes), within-type variance (variable genes), the
**neighbour ridge** (each target on the k nearest cells' within-type residuals and label shares, the target's own neighbour feature left out exactly by the
one-column downdate; a weak bar on purpose: the neighbour means do not tell the sender's type), and each arm's untrained twin.

**Targets.** A model trained with channel masks is read in **folds** (`gene_folds`: ~20 % of the genes hidden in every cell, 2 partitions, so almost
every pair is read in some fold; `hide_genes` sets `gene_mask` and `mask` as the dataloader does). A model that never reads its own cell is read on the
**clean pass** (`targets="all"`, every cell a target; `scored_entries` refuses it for a model that reads its own cell, `_scoring_every_cell` switches
`score_all_cells` on for mode-a models such as typeres).

**Why synth_spot needs new training.** J1 hid only the 29 response genes, so no other gene was ever a target: a ranking over all genes would rediscover
the mask. J5 trains with channel masks (20 % of the genes in every spot, redrawn every epoch): every gene is a target sometimes and can then be predicted
only from other genes. The comparison arm is the reference recipe as a context query (`ctx_ch`: feed-forward on the output path, blind to its own spot):
`lin_dist_skipraw` itself reads its own token, so its context part is not a statement about other cells.

### Truth

- synth_spot (`roles_truth`): responders = roles `receiver` / `response` (R1-R4, MC1-MC25: 29), drivers = `sender` / `source` (S1-S4, M1-M21: 25), pairs =
  every driver -> responder of one `interaction` programme (31), **hard negatives** = roles `spatial`, `marker`, `batch`, `condition` (SP1-10, CM1-3, bg, cg).
  Receptors MR16-25 (the receiver's own gate) are ordinary negatives.
- spiked CosMx (`programme_truth`): per receiver type the spiked genes; the null programmes' genes (chosen by the same rule, never spiked) as hard
  negatives; the planted sender type per programme.
- real CosMx: none; stability over seeds, flags (a responder that marks another type: `spillover_flags`; handling stress / ribosomal genes from
  `biology_resource.json`), known ligand-receptor pairs against expression-matched targets (`lr_support_by_type`).

### Decision rules (fixed 2026-11-22, before any run; `gene_rank_read.py`)

A bar is the best of Moran, variance, ridge and the arm's twin; means over seeds.

- **G1 responders** (synth): AUPRC >= best bar + 0.10 and AUROC against the hard negatives >= 0.80.
- **G2 drivers** (synth): AUPRC >= best bar + 0.10.
- **G3 pairs** (synth): precision at k (k = the planted pairs scored, 31) >= 0.5 and AUPRC above the ridge's.
- The ranking model for synth is the arm passing more of G1-G3 (tie: higher pair AUPRC); none passing: no ranking model, said so.
- **S1** (spiked): per spiked receiver type, AUROC of the planted genes >= 0.75 and above every bar and the twin, in every spiked type.
- **S2** (spiked): every null programme's genes outside the top 20 % of their receiver type's responders.
- **S3** (spiked): the planted sender type first for at least half of the spiked programmes. A programme is *detected* when its genes' median rank lies in
  the top 10 % of the receiver type's responders (the shortest range and smallest effect detected are reported).
- **C1** (CosMx): responder ranking stable over seeds, mean RBO (p = 0.9) over types and seed pairs >= 0.5 (drivers and pairs reported). **C2** reported:
  the flagged share (spillover, handling) of the top 50 responders per type.

### Jobs (`slurm/November/`, flat; results `MA/22_11_results/rank_{synth,cosmx,semi}/gene_rank.md`)

| Job | What | Runs | Per task |
| --- | --- | --- | --- |
| J5 `synth` | `make_rank_synth_yaml.py`: ir_ch (J1's ir2_rate), ctx_ch; channel masks 0.2; 300 epochs; standard programme evaluation (hiding J1's set) + rankings in folds; bars in ir_ch_s42 | 4 (+ twins of both at s42) | 1 GPU, train ~30 min, rankings minutes |
| J6 `cosmx` | J4 checkpoints, evaluation only (`MODE = "evaluate"`): typeres, ir_c, ir_c_ktype, ir_c_knone x 3 seeds, ir_c_same; twins of typeres, ir_c; clean pass, every cell a target; bars in ir_c_s42 | 13 | 1 GPU; ~960 sender passes per section (estimate 10-30 min per model) |
| J6b `semi` | J3 checkpoints on the spiked file, as J6 with the programme truth | 7 | as J6 |

Files: `gene_rank.py` (one script, `RANK_JOB=synth|cosmx|semi`, every setting of ir_synth.py / ir_cosmx.py taken over so the data, split and checkpoints
match), `gene_rank_common.py` (the hook), `gene_rank_read.py` (the rules), `make_rank_synth_yaml.py`, `rank_job.sbatch`, `rank_submit.sh`. Copy them with
`biology_resource.json` (slurm/new) to `MA/scripts/` (sbatch and submit to `MA/slurm/`), install the fork's main, then `bash slurm/rank_submit.sh synth`
(and `cosmx`, `semi`). cosmx / semi need the J4 / J3 sweep yamls in `MA/yaml/` and their checkpoints as ir_submit.sh left them.

### Checks run (no model on real data)

- Library: `tests/test_gene_ranking.py` (10 tests on a grid of known dependence: responders g0 / g2 found, g1 never moves, the same-gene responder g6 loses its
  score once its own gene is taken off, the pairs g3 -> g0 and g4 -> g2 on top, no self pair, drivers leave themselves out; folds cover every gene and merge
  by averaging where read; metrics by hand; truth builders; the ridge leaves its own neighbours out). Six mutants (self path kept, self pairs kept, ridge
  reading its own neighbours, no blind check, `hide_genes` without `mask`, merge dividing by all parts) all caught. `test_interpretation.py` unchanged and
  passing (`gene_flows` defaults are the old behaviour).
- Jobs: the synth arms through `load_sweep` / `arm_cfg` / `_validate` / `check_masking` (channel masks 20 % in every cell; names unique, <= 163 characters;
  run order = runs file); `configure()` for cosmx and semi; their run lists equal the J4 / J3 runs restricted to the ranked arms (13 / 7).
- The hook end to end on real GlobalModels at toy size on the CPU (a read-out and a stream context query under channel masks with folds and roles truth; a
  mode-c read-out and a typeres-like mode-a model on labelled data with flags and a toy resource; the programme truth), then the reader on their tables.
- Bugs these found, fixed: `hide_genes` left `mask` unset, which `apply_mask` asserts under gene masking (every synth fold would have failed); the reader
  failed on the empty `rank_scores.csv` of a run without truth; it named a "ranking model" when no arm passed a rule.

### Limits, said before the run

- The neighbour ridge does not tell a sender's type (diluted by other types: 0.2-0.3 on the test grid where the rule is exact); smooth targets pick up
  chance correlations with smooth features (~0.1).
- Responder scores are drops of a centred cosine under a scramble: comparable within a run and against its twin, not an effect size in expression units.
- Channel masking hardly fitted its own task in 2026-10-23's run (0.245 on the programme; notes.md). If J5's arms do not fit, the rankings say little; the
  standard programme evaluation in the same job shows the fit.
- On CosMx (clean pass) a responder's neighbours show the same gene, so the model may have learnt to read that instead of other genes; `responder_cross`
  is then a conservative cross-gene score. A channel-masked CosMx arm would remove the shortcut (not written).

### E7 read (2026-11-23)

J5 synth: no G passes. Fit 0.50 / 0.52 (ir_ch / ctx_ch) vs own+context 0.68. Responders AUPRC 0.33 / 0.36 vs ridge 0.41, Moran 0.34, AUROC_hard
0.32 (CM / SP genes above responders). Drivers 0.37 / 0.33 vs twins 0.30 / 0.31. Pairs ir_ch AUPRC 0.211 (twin 0.013, ridge 0.196), precision@k 0.26:
M8 -> MC10 / MC11 rank 1 / 2, M7 -> MC8 / MC9 / MC7 rank 4 / 10 / 11, M3 -> MC3 48; short range R1-R4 rank 120-990; 19 of 31 planted pairs scored.
J6 CosMx (11 of 13 runs; typeres and its twin absent): C1 ir_c 0.486 FAIL, ktype 0.609 / knone 0.610 PASS; LR AUC 0.485-0.503 (null); C2 5-26 %
flagged though the top lists are stress / ribosomal / acinar genes in other types. J6b semi: S1 FAIL (variance bar 0.98 / 0.94 AUROC), S2 PASS for
amici_like, ir_a, ir_c_rate (not ir_c), S3 FAIL; detected 2-4 of 8 programmes (exp kernels into Acinar; never Immune>Ductal@step20).

### E7: why typeres was missing from J6, and the re-read (2026-11-23)

typeres reads its own cell (raw own-cell skip); J6 read every CosMx run with TARGETS "all", which `scored_entries` refuses for such a model, so both typeres
tasks failed in the hook (the toy check had used a stand-in without the skip). Fixed in gene_rank_common: TARGETS "auto" (clean pass when blind to the own
cell, channel folds when trained with channel masks, else 20 % of the cells hidden whole). Re-read: `RANK_RUNS=typeres bash slurm/rank_submit.sh cosmx`
(not run). comparison.md "Round 2" has the details.
