# Notes: what worked and what does not

A log of the verdict of every test. **Append only:** new entries go at the end of this file; never
overwrite or edit an earlier entry (a correction is a new entry that says what it corrects). Each entry
is `## <date> — <what the test was about>`, then **What worked** and **What doesn't work**. Tables,
settings and the reasoning live in the plan files (`global_plan.md`, `attention_plan.md`, ...); this
file keeps only the points.

## Major improvements so far

Not an entry: a standing, updated index of the changes that actually moved a number, most recent first.
Everything below it stays append-only. Full numbers are in the dated entry named.

- **Distance prior on a global-only model** (2026-10-24): `pe.distance.kind: scales` + the raw own-cell
  skip reads synth_spot's short-range receivers at 0.94-0.98 (every earlier global-only arm: ~0), best
  programme mean so far (0.728). **Held over 3 seeds** (2026-10-25: 0.909 ± 0.003, programme mean 0.729).
- **Programme masking on legnini** (2026-10-24): every trained arm first reaches its bar (0.77-0.80
  against 0.75).
- **Context query + additive decoder** (2026-10-23): separates the dual model's parts with no cost to the
  score (0.842 against 0.843) -- the context part carries only the rings, the local part everything else.
- **Additive decoder alone** (2026-10-23): dual sum 0.820 -> 0.843, first arm to reach the programme bar
  on its mean.
- **Raw (full-rank) own-cell skip** (2026-10-23): closes the mirror-programme gap that the token-rank skip
  left open, programme mean 0.577 -> 0.611.
- **Isotropic Fourier attention bias** (2026-10-19/20): the first architecture change that made the
  transformer do more than a cell's own genes. Without position, `lin_none` = `ref_own` (0.405); with it,
  `lin_fourier` 0.764-0.775, `joint_fourier`'s global head 0.820, against the bar's 0.682 -- the gain is
  the ring programmes (M7/M8), which only the distance to a visible source gives. Beat RoPE in every arm.
- **Centred targets + more steps** (2026-10-14, round 3a): the first thing that broke the gene-mean floor
  every earlier round had converged to, before which no architecture change had anything to work with.

---

## 2026-10-20 — Programme masking on synth_spot (`response_masking`: 11 arms, 19 runs, 3 seeds, 100 epochs)

Arms: `local_gcn` is the GCN alone (`LocalModel`, no transformer); `lin_*` is the transformer alone
(`GlobalModel`, learned linear input); `joint_*` is GCN + transformer with the usual dual decoder, trained
jointly (no staged schedule, nothing frozen, long-range mask on); `*_pool500` runs the transformer on 500 µm
patches. The bar is `ref_own+context` (0.682).

**What worked**
- `joint_fourier`'s global head scores 0.820 ± 0.005 over three seeds against the bar's 0.682 and its own
  local head's 0.506. Seed spread is 0.003-0.017 overall, so differences above ~0.03 are real.
- The gain is in the long-range programmes: the rings M7 / M8 score 0.93 / 0.93 against 0.63 / 0.75 (bar) and
  0.44 / 0.49 (GCN), the long-range responses 0.884 against 0.692 (bar) and 0.471 (GCN alone).
- Against the GCN alone, the joint global head is better by more than 0.05 on 5 of the 25 programmes (M7 +0.47,
  M8 +0.42, M11 +0.18), ties on 16 and is worse on 3. The overall 0.820 against 0.510 is carried by the rings.
- Fourier bias beats RoPE in every arm (joint 0.820 against 0.748, transformer alone 0.764 against 0.709).
- A blend with the bar no longer adds to the joint head (0.804 < 0.820): it has absorbed what the references know.
- 100 epochs instead of 200 costs about 0.01 (`lin_fourier` 0.764 against 0.775).
- The pipeline is fast: a run trains in 2-3 min and evaluates in 30-60 s; the fitted references take 61 s once;
  a 19-run job array finishes in minutes.
- Pooling at 500 µm keeps the ring and long-range programmes (0.79 against 0.89 per cell) and runs 1.5x faster
  with 3.5-5.5x less GPU memory (at only 2,500 spots per slide).

**What doesn't work**
- Per programme, no model beats the bar. Equal weight over the 25 programmes: `joint_fourier` 0.628 against
  the bar's 0.685; it beats the bar on 3, ties on 12, loses on 9.
- Mirror programmes (MC1, MC2, MC4, MC6: the response copies the cell's own visible source gene) score 0.2-0.4
  below a plain ridge on the cell's own genes, in every neural arm, the GCN alone included.
- M11 (response within 500 µm of single source spots): 0.54 against 0.92 for a 150 µm kernel; the near-source
  scale is not learned.
- Directional programmes (M9, M10): about 0.1 in every arm and reference; the Fourier bias used is isotropic.
- Sparse threshold programmes (M16, M17, M20) only match `ref_own`; pooled over all cells the score mostly
  measures whether a cell is a receiver, so it cannot show whether the distance to the source was learned.
- Pooled 500 µm tokens lose the sparse responses (0.07-0.08 against 0.78): a patch token is shared, so the
  global head cannot know which cell is the receiver (the joint arm's local head, which reads cells, scores
  0.757 on the same genes). Pooled overall is 0.62 against 0.76 per cell.
- A PCA of the visible genes is a poor token (0.34-0.42 against 0.76 for the learned linear input).
- The transformer alone scores about 0 on the short-range receivers (0.01 against 0.6 for the GCN); only the
  joint model reaches them (0.64).
- Sender ranks are unreadable (0.35-0.57; the GCN ranks the senders of its own short-range programmes at
  0.41 while predicting them at 0.59): the gradient-magnitude measure is invalid and has to be replaced by an
  attribution (roadmap T4).

---

*Backfill, appended 2026-10-20 from `contrastive_plan.md`, `PE_plan.md`, `attention_plan.md`,
`global_plan.md` and the session notes. The entries below are in chronological order and keep the date the
result was read; the entry above was the first one written live.*

## 2026-09-18 — Probe battery on synth_data_0: can a value-reconstruction probe show the global component is needed?

**What worked**
- The controls separate pass-through from learning: the structure-free `noise_00` gene reads R² 0.33 from both
  embeddings unrestricted and ~0.02 on masked cells only, so gene probes are scored on masked cells only. A
  label (cell type) must not be masked: masking it held it at 0.22 macro precision against 0.46 on the clean pass.
- The split-independence check (`dataset.group_key`) and the attention-flow control were built and tested on
  synthetic flow matrices; the flow's sign convention is pinned by a test.

**What doesn't work**
- No target of the probe panel can show that the global component is needed: every long-range target is
  spatially smooth and readable from the cell's own ≤30 µm neighbourhood (correlation with the neighbour mean
  0.35-0.998: `int_long` 0.805, `hub_response` / `dist_to_hub` 0.998, the noise floor 0.46).
- A probe gap therefore says nothing about range. `int_long` is the gene the local component should do best
  on (the legend said the opposite), and `hub_response` / `dist_to_hub` are worse discriminators, not better.
- VICReg, context NCE and ring NCE are implemented, but no verified training run exists; the probe numbers of
  the one real run settled nothing.
- The attention-flow control has not been run on a real trained attention matrix.

## 2026-10-10 — What the synth_spot lattice does to the positional encodings (model-free measurements)

**What worked**
- Coordinate encodings (naive, sinusoidal, RoPE, Fourier) are the only ones that see direction (MC12 runs along
  x, MC13 along ±y); RoPE sees it as the direction between two cells.
- k-NN with k = 8 (used since 2026-10-10) has no interior ties: every interior spot gets its 4 neighbours and
  4 diagonals, each diagonal direction equally often; the symmetrised degree is 8 for 92% of spots. The
  2-layer GCN's reach, and so the long-range mask, is the 5 x 5 block (≤ 283 µm).

**What doesn't work**
- RWPE barely sees position: the 16-step RWPE takes 40 distinct values over 2,500 spots and the deep
  interior (46% of spots) shares one; it encodes distance to borders and holes, not where a spot is.
- LapPE is degenerate on square slides: on an intact 50 x 50 slide λ1 = λ2 to 1e-14, so its first two
  columns are an arbitrary rotation that sign flips do not cover; a crop or holes split the pair by ~4%, so
  intact and damaged slides get differently oriented encodings.
- LapPE, RWPE, the distance bias and the isotropic GCN are blind to orientation, and `rotate_train` erases it.
- Centring moves with a crop: one lost border line shifts a slide's centroid by 50 µm.
- k-NN with k = 6 is directed and picks tied diagonals (slide A_01: the (−,−) diagonal 1,544 times, the (+,+)
  one 890 times).

## 2026-10-12 — Round 1: attention backends x positional encodings on synth_spot (dense, Performer, Exphormer; 40 epochs, one seed)

**What worked**
- The checks exposed the null honestly (attention uniformity, planted-edge rank, comparison with the
  gene-mean floor) instead of letting single-seed differences be over-read.
- `naive` lifted position-derived readouts: source distances 0.28 → 0.40 (the PE alone reaches 0.33) — that is
  coordinates handed to the tokens, not interaction learning.

**What doesn't work**
- Attention was uniform in every backend: the Performer spread over ~2,460 of ~2,500 effective keys, Exphormer
  over 7.9 of its 8 partners, the span (2.6 mm) is the mean pair distance, relevance was flat with distance and
  planted senders ranked at chance. The backends were not really tested.
- The biases and RoPE frequencies barely learned (coefficients 0.01-0.03); several bias arms are the same
  model to three decimals.
- The decoders were worse than predicting each gene's training mean (val SCE 0.107-0.121 against 0.082), and
  raw SCE barely rewards context (a neighbour average 0.0811 against 0.0816).
- The 16-dim embeddings lose information that trivial baselines keep: the neighbour average beats every learned
  representation on region, R3/R4 state, the M9 stripe and source distances.
- Gene ranks barely depend on the run (Spearman 0.98-1.0 across 32 runs).
- Each sweep's `none` arm started from a different initialisation (the model was built before seeding;
  relative weight distance ~1.1 against ≤0.08 among the PE arms): fixed. The efficiency monitor's peak memory
  also included validation and the probes: fixed.

## 2026-10-12 — Round 2: long-range mask off, width 64, warm-up 8 (dense, Performer, Exphormer; 34 runs)

**What worked**
- The seeding fix worked: GCN weights are within 0.000-0.016 across all 34 runs, so comparisons are paired.
- Cost at S = 2,500 and width 64, measured properly: Performer 13-18.5 steps/s at 1.2 GiB; dense ~9 steps/s
  (2.6 with the distance bias) at 3.0 GiB; Exphormer 2.8-3.8 steps/s at 1.2 GiB (it rebuilds its pattern on
  the CPU every step).

**What doesn't work**
- Every run converged to one function: val SCE 0.08336-0.08342 in all 34 runs, still above the gene-mean floor
  (0.0816) and the neighbour floor (0.0811); gene ranks correlate at 1.00 and decoder loadings agree to three
  decimals across backends; the query / key norms stayed at their initial size (5.6 against 5.66).
- Attention stayed uniform with the mask off (dense and Performer ~2,480 of ~2,500 keys, Exphormer 15.95 of 16);
  RoPE heads attend farther, not nearer. The mechanism is not the problem.
- The biases stayed tiny (≤0.024): dense spectral, Fourier and distance biases rank immediate neighbours at
  the top but add ≤17% weight; none of it changes the loss.
- The probe gains (region 0.987, M9 0.80, source distances 0.43) all came through the GCN: local equals global,
  and there was no untrained-model control.
- Exphormer's short-range relevance AUROC of 1.0 is by construction (the k-NN edges sit in its graph).
- Under raw SCE the network settles near the mean profile; nothing rewards context, so the objective had to change.

## 2026-10-13 — Round 2 on legnini (dense, Performer, Exphormer x positional encodings)

**What worked**
- The diagnosis: legnini has a real local signal that the model misses. Masked SCE on the val slides: train gene
  mean 0.158, slide mean 0.142, a parameter-free mean of the visible entries within 55 µm 0.122, the model 0.141.
  The model underfits; the signal is not missing.
- RoPE sharpened attention more than on synth_spot (dense layer 0: 656 effective keys of ~1,880, span 231 against
  378 µm), and its q / k alignment grew.
- Cost per step at batch 1: dense 15-17 steps/s at 2.0 GiB, Performer 17-23 at 0.4 GiB, Exphormer 7-9 at 0.43 GiB.

**What doesn't work**
- Val SCE is 0.1408-0.1412 in every arm of every backend (`naive` worse, 0.1420-0.1425): the same single
  function as on synth_spot, at about the slide mean.
- GCN and decoder weights are within 0.3% of `none`, and |W_q|, |W_k| are still ~5.7-5.8 (init 5.66).
- Attention differences do not move the loss: even RoPE's sharpest attention averages almost the same vector
  as uniform attention (a 250 µm average of expression has cosine 0.995 to the slide mean).
- The condition probe is saturated on only 2 test slides; the legnini scripts log no per-head losses.

## 2026-10-14 — Round 3a: longer training and centred queries, keys and targets (synth_spot, dense and Performer, one seed)

**What worked**
- The model learns beyond the mean profile: raw masked SCE 0.0834 → 0.0701, below the gene-mean floor (0.0816) and
  the neighbour average (0.0811). The longer schedule alone gave 0.0727, centred targets 0.0706, centred queries
  and keys 0.0701 (only together with centred targets). Dense and Performer agree.
- Centred targets changed the representation, not just the readout: the token cosine fell from ~0.95 to
  0.69-0.74, the root cause found in the diagnosis (near-identical tokens).
- Probes improved with centred targets in both embeddings (R3/R4 state local 0.45 → 0.56, global 0.39 → 0.50;
  M9 stripe global ~0.80 → ~0.89; source distances global 0.41 → 0.45-0.48), and `val_global_loss` sat below
  `val_local_loss` in every arm (0.6184 against 0.6225 centred): a small, consistent global edge.

**What doesn't work**
- Attention is still far from selective (effective keys ~1,000-1,400 of ~2,500, span ~2.5 mm), and the
  encodings do not matter for the loss (`none`, `rope`, `fourier` agree to 0.0001; `naive` is worse by ~0.001).
- Planted edges are at chance (rank 0.47-0.52, lift 0.87-1.0).
- The neighbour input baseline still beats both embeddings on R3/R4 (0.67) and source distances (0.55); local is
  ahead of global on every probe and `local+global` never beats `local`.
- The gene ranks read an unmasked pass (`get_model_output`), so they measure pass-through, not context: the
  "global ranks far above local on sparse responses" of this round was not evidence for global context.

## 2026-10-14 — Round 3b: 500 µm discs as the only targets, residual decoder, long-range mask (first run invalid, rerun valid)

**What worked**
- The rerun (after the mask fix and the rescaled signal) passed its sanity check (local raw SCE 0.078-0.081 at
  the context floor, not 0.037) and dense and Performer agree to 0.001-0.003.
- The local head is blind beyond its reach, as designed: centred cosine 0.33-0.35 within 290 µm, 0.05 at 290-500 µm,
  about 0 beyond.
- The global head recovers coarse context in the blind area: 0.21 without an encoding (= a slide-mean predictor),
  0.29-0.31 with RoPE or Fourier (= a Gaussian kernel of visible spots, 0.30-0.33). Position matters for the first
  time (+0.1).

**What doesn't work**
- The first run was invalid: the disc arms trained as autoencoders (`masked_nodes` was `mask_percentage > 0` and
  the discs set the rate to 0): raw SCE 0.037 below the floor, flat across depth, noise genes predicted at
  r = 0.997. Fixed; the lesson is that a new mask mode needs a test through the real module and trainer and a
  check that hidden values cannot reach a prediction.
- Global never beats a smoother of the visible spots: equal on slide- and region-level programmes, below on
  long-range (0.70 against 0.77), spatially variable (0.64 against 0.79) and sparse responses (0.08 against 0.30).
- Fine context (the R3/R4 sender state) is at chance and attention shows no preference for planted senders.
- The residual decoder does not help (`rope_no_res` is equal or better in the blind area, +0.02 centred
  cosine), nor does the long-range mask (0.307 against 0.311).
- Global's magnitude is off: R² −1.4 to −3.4 on the high-variance programmes while the directions are right
  (the centred SCE is scale-free). `rope_gene` is not comparable on pooled numbers (different masking).

## 2026-10-18 — Global model alone, other embeddings, staged training (disc task; first read, one seed)

**What worked**
- The global head fills the GCN's blind area: 0.438 against the local head's 0.060 at 290-500 µm, equal to the best
  smoother beyond 500 µm; untrained twins sit at ~0.005, so it is learned. The round-2 floor problem is gone
  (0.46 against the slide mean 0.26).
- A fixed precomputed PCA is the strongest global-only input (0.442; scVI 0.40-0.41; learned linear 0.384).
- scVI tokens beat the smoother on source distances (0.459 against 0.435; 0.532 against 0.429 beyond 500 µm), and
  PCA + Fourier adds +0.066 over it for region beyond 500 µm.
- Training is bit-reproducible (two separate runs gave identical losses).

**What doesn't work**
- Nothing passes the best smoother (0.465; joint global head 0.459). Neither "the GCN's embedding limits
  global" nor "joint training holds global back" is supported: every variant, a frozen random GCN included,
  lands at global validation loss 0.545-0.57; no staged schedule beats joint (0.424-0.452 against 0.459) and they
  weaken the local head.
- Near visible spots the global head mostly copies the local one (CKA 0.93 at the rim).
- The residual decoder hurts the deep field (0.389 against 0.438 at 290-500 µm).
- The in-pipeline PCA is fitted on one masked validation slide: it misses slide- and condition-level genes
  (batch 0.02, condition 0.30 against 0.985 and 0.984 for a precomputed PCA) and halves throughput.
- The learned linear input without RoPE fails (0.153, below the slide mean).
- Bugs found: the combined validation metric averages both heads (misleading with a frozen head); planted-pair
  ranks were inflated by a 50 µm tolerance that mixed neighbours and diagonals (fixed to 10 µm). Two jobs ended
  during evaluation and one arm never ran.

## 2026-10-18 — Second read of the global sweeps (disc task; gene groups by depth, blend test, position baseline)

**What worked**
- The first gene group where a model beats the best reference: the response genes beyond the GCN's reach
  (scVI + Fourier +0.035 at 290-500 µm and +0.036 beyond 500 µm; the joint global head +0.015 and +0.033).
- A blend of PCA + RoPE with the 400 µm kernel beats both in the deep field (0.456 against 0.448; 0.429 against
  0.413): the model holds modest complementary information. Noise genes score ~0 everywhere (the control works).

**What doesn't work**
- One seed only; the gains are a few hundredths.
- The disc task cannot test the programmes built for global context: short-range receivers score ~0 for every
  model and reference (a hidden receiver loses its sender with it), sparse responses 0.04-0.09 against 0.19 (the
  receptor is hidden too). This is why programme masking was built.
- Position explains much of what looked like knowledge: on source distances the smoother rises 0.435 → 0.524 with
  position, and the embeddings add at most ~0.02 beyond that. Planted pairs stay ~0.5 (no sender selection).

## 2026-10-18 — legnini attention sweep: dense, distance + expression bias, entmax, top-k 16 / 64 (GlobalModel on PCA + RoPE, 125 µm discs, 400 epochs)

**What worked**
- The sparse backends work as built (effective keys 0.7-2% of the open keys against 5-7% dense).
- The Leiden clusters match Legnini et al. 2023's SHH-patterned neural tube (c13 floor plate with FOXA2 and SHH, c02
  p3 / pMN with NKX2.2 and OLIG2, c06 motor neurons, c07-c10 dorsal progenitors): a real long-range morphogen setting.
- The flow structure beyond distance is real (the untrained twins give ~0).

**What doesn't work**
- Every arm reconstructs at the slide mean (0.495-0.511 against 0.497; the 40 µm kernel reaches 0.594): the
  attention mechanism changes nothing. Predictions are broad averages (hidden cells at the disc rim take 92% of
  their gradient influence from more than 60 µm away).
- The first flow maps were not readable: the plotted slide was the control organoid, whether a key was visible
  explains only ~18% of the structure, and no rule (key type, same type, depth, density, token PCs, 2-D offset)
  fits both test organoids; in the SHH organoid the floor plate, the SHH source, is avoided. The flow was
  rewritten afterwards and has not been re-read.

## 2026-10-19 — Programme masking on synth_spot, first read (5 arms, one seed, 200 epochs)

**What worked**
- The first model to pass the bar on a task built for global context: `lin_fourier` 0.775, `joint_rope` global
  head 0.756, `lin_rope` 0.708 against the bar's 0.682 (GCN alone 0.516). The gain sits in the ring programmes
  (M7 / M8: 0.95 / 0.96 for `lin_fourier` against 0.63 / 0.75). Untrained twins are at ~0.
- The joint model's global head also reaches the short-range receivers (0.64-0.68); without position a
  global-only model reads only the cell itself (`lin_none` 0.407 = `ref_own` 0.405). Fourier beat RoPE for
  global-only models (0.775 against 0.708).

**What doesn't work**
- Global-only models score ~0 on the short-range receivers.
- M11 (response within 500 µm of single sources): 0.15 global-only, 0.57 joint, against 0.92 for a 150 µm kernel.
- The mirror programmes (response copies the cell's own source gene) are below `ref_own` (MC4 0.36 against 0.68).
- The directional programmes (M9, M10) sit at 0.1-0.3 in every arm and reference; sparse responses only match
  the bar.
- The sender ranks are not interpretable: the GCN ranks its own short-range senders at 0.40 while predicting
  them at 0.61 (a gradient magnitude, not a contribution).

## 2026-10-19 — legnini candidate programme genes (model-free look at the data: which hidden set could test SHH signalling)

**What worked**
- The SHH organoids are strongly ventralised: ventral fate genes are 17-27x higher than in controls (NKX2.2 1.6
  against 0.06 counts per cell, NKX6.1 18 against 1.1), ISL1 ~150x higher, PAX7 ~70x and MSX2 ~7x lower. In the
  SHH test organoid (slide4_B2-2) they sit around a compact SHH / FOXA2 source (the control has no ventral genes).
- They are more spatially clustered than the direct readouts (Moran's I in SHH organoids: NKX6.1 0.50, NKX2.2
  0.37, OLIG2 0.36 against GLI1 0.24 and HHIP 0.25).

**What doesn't work**
- The direct readouts barely separate SHH from control organoids on average (PTCH1 1.5x and detected in 89% of
  all cells, GLI1 1.9x, HHIP 0.6x; SHH and FOXA2 are not higher on average either): a weak main task.
- Whether a small hidden set gives itself away through the cell's other genes is not settled: a first attempt
  found the cell's own genes explain the three readouts (centred cosine 0.51, the surroundings adding 0.01), a
  second attempt with other settings disagreed on the surroundings (0.04 against 0.42), and the check was
  interrupted before a clean version ran.

## 2026-10-21 — Is attention computed correctly? The vendored attention and the relevance against torch (code and toy models; no data)

**What worked**
- The default attention (the vendored copy of torch's, edited to keep the map in the autograd graph) equals torch's
  stock `nn.MultiheadAttention` with the same weights to 1e-6: outputs, per-head maps and input gradients, under
  boolean and float masks, key padding and both together, in training (dropout 0) and evaluation, and with separate
  query/key and value tokens (`center_qk`). The custom encoder layer equals torch's stock layer. The gradient the
  hook stores is autograd's gradient with respect to the map the layer used.
- Chefer's rules 5 and 6 are implemented as in the paper (head mean of the positive gradient x attention, then
  R += Ā R per layer).

**What doesn't work**
- The relevance averaged the heads over all graphs of a batch together (only harmless because `evaluate()` runs
  one graph) and built its identity and masks on the CPU, so it failed for a model on a GPU. Fixed.
- Its target, the summed output, is still degenerate (known since 2026-10-17); a target can now be given.
- Left as they are, documented: `average_attn_weights` is ignored (maps are always per head); the map returned in
  training is the one after dropout; `evaluate()`'s raw matrix is head 0 of graph 0, not a head mean (unused); the
  input LayerNorm is applied twice (before and after the node encoding is added).
- Nothing here says which readout finds senders: that is the `attention_check` sweep, not run.

## 2026-10-21 — Disc geometry on a synth_spot-sized lattice: how deep the hidden spots are (model-free, 20 draws; for the regime loss, D2.2)

**What worked**
- The disc masks give the regime loss both regimes to work with. With discs over 30% of a 50 x 50 lattice (100 µm
  spacing) and the threshold at the 2-hop GCN's reach (290 µm), the share of hidden spots that are deep is 7% for
  300 µm discs, **34% for 500 µm** (+-3%), 50% for 750 µm and **64% for 1000 µm** (+-4%).
- At 500 µm the hidden spots fall in the depth bins 0-150 / 150-290 / 290-500 / 500+ µm as 38 / 28 / 32 / 2%; the
  evaluation's bins therefore cover them, and nearly none lies deeper than 500 µm. At 1000 µm: 19 / 17 / 29 / 35%.

**What doesn't work**
- 300 µm discs (and anything with a radius near the threshold) leave almost no deep spot: the global head would have
  nothing to learn from, which the config now refuses.
- My first estimate for 500 µm discs, "a fifth deep", was wrong by a third: discs overlap and merge into larger holes.

## 2026-10-21 — Attention check on the response_masking checkpoints: is it computed correctly, and which readout finds the planted senders (13 trained runs, 3 untrained twins, programme masking on synth_spot)

**What worked**
- The attention is computed as the definition says: for every layer of every run the stored map and the layer output
  equal the written-out definition to below the table's 4 decimals (reported as 0; scale of the values 0.3-2), and
  where torch's stock `nn.MultiheadAttention` can take the same inputs (no RoPE: `lin_none`, the Fourier arms) it agrees
  as well. So the vendored attention and its per-head, graph-kept maps are correct, also after training.
- Gradient x input at the tokens is the one readout above its untrained twin in every arm and seed that has a Fourier bias
  or RoPE: planted sender's matched rank 0.551 (`lin_fourier`, 3 seeds: 0.541 / 0.550 / 0.562) and 0.549 (`joint_fourier`:
  0.562 / 0.552 / 0.533) against twins at 0.511 and 0.498. Small (+0.04-0.05), but it reads no attention map.
- After training the readouts disagree, before it they do not: attention, attention mean and rollout correlate 0.98-1.0 in
  the untrained twins (uniform attention) and 0.70-0.93 trained; gradient readouts correlate 0.15-0.5 with the attention ones.
  Which readout is used therefore changes the picture; it is not one signal read many ways.

**What doesn't work**
- No readout finds the planted senders: the rank over all planted pairs is 0.47-0.55 for all seven readouts in all arms, the
  untrained twins 0.48-0.52. Chefer's relevance from a query's own target (0.535 for `lin_fourier`) is no better than the
  degenerate summed-output relevance (0.546), so this instrument cannot yet tell a valid target from an arbitrary one.
- Attention is still nearly uniform: the score-weighted distance to the query is 2.4-2.6 mm in every arm (twins 2.53 mm),
  the share within 300 um is 1.1% for the twins (= uniform). `lin_fourier` is below that (0.5%: it looks away from near
  cells), `lin_rope` far above (6.1% last layer, 5 x uniform: local heads, but still no sender selection).
- The RoPE arms rank the mid-range planted sender far *below* chance: `lin_rope` 0.18 (attn_last) and 0.09 (rollout),
  `joint_rope` 0.33 and 0.21 over 3 seeds (rollout per seed 0.235 / 0.195 / 0.185), `joint_fourier` 0.38 / 0.36; the twins
  sit at 0.4-0.5. Trained attention avoids those senders relative to cells as far away. The number of pairs behind a class is
  not in these files (the report dropped `n_pairs`; kept since), so it is a lead, not a result.
- Seeds scatter by 0.02-0.14 per class (the mid class up to 0.14), so single-class differences below that are noise.

## 2026-10-21 — The oracle bar on synth_spot: what a ridge on the planted drivers reaches (model-free; fresh synth_spot, 24 training / test slides, programme masking)

**What worked**
- The task is solvable from the planted truth: a ridge on the spot's own visible genes plus the true drivers (distances to the sources as 24
  Gaussian bumps each, receiver and cone gates, stripe and exposure flags, `R34_state`) reaches a mean centred cosine of **0.983** over
  the 24 programmes, against 0.685 for `ref_own+context` (the sweeps' bar, 0.682 there) and 0.580 for `ref_own`. At least 0.94 on
  every programme but one (the weakest after it: M4→MC4 at 0.949). So a programme the models miss is the model's limit, not the task's.
- Where the model-free bar is far below the oracle: the stripe M9→MC12 (0.27 → 1.00), the cone M10→MC13 (0.16 → 0.83), the short-range
  programmes (S1→R1 0.65, S2→R2 0.66, S3/S4→R3/R4 0.55 → 1.00 each once `R34_state` is a driver), the flat long-range M3→MC3 (0.46 → 0.98)
  and M5→MC5 (0.58 → 0.98), the rings M7 and M8 (0.63, 0.76 → 1.00). These are the programmes a model can still gain on.

**What doesn't work**
- The cone M10→MC13 stays at 0.83: its response is a plateau and an exponential decay inside a 45° cone, which 24 quantile bumps resolve only
  roughly. Without `R34_state` the S3/S4 programme read 0.88: R3 and R4 depend on both exposures, an interaction a linear ridge on the two flags
  cannot form.
- The oracle's ridge penalty sits at the lowest value of the grid (0.1), so noiseless drivers may want less regularisation than
  `FITTED_ALPHAS` offers; the numbers are a floor for what the truth reaches.
- The oracle is built on truth a model must infer (`is_receiver_*`, `R34_state`): it bounds the task, not what a model can learn.

## 2026-10-22 — Decoder arms of the global-only model under programme masking (`global_decoders`: `lin_fourier` and `lin_fourier_skip` at 3 seeds, nonlinear, LSE, pooled 500 µm with and without the skip at 1 seed; synth_spot)

**What worked**
- The own-cell skip does what it was built for, on the programmes it was built for. Over 3 seeds (seed spread 0.01-0.05) the
  mirror programmes rise: M1→MC1 0.54 → 0.63, M2→MC2 0.41 → 0.56, M4→MC4 0.27 → 0.49, M6→MC6 0.47 → 0.60, also M17→MC21
  0.44 → 0.53. Nothing else drops: the rings stay at 0.95 / 0.96, the sparse programmes at 0.91-0.98. Mean over the 24
  programmes 0.542 → 0.577, wins / ties / losses against the bar 4/10/10 → 5/10/9. On the joint model's global head the same
  mirror programmes read 0.21-0.28 (MC4) and 0.46-0.51 (MC6), so the skip does better on them than the GCN does.
- Pooled 500 µm tokens with the skip match the per-cell model at about 1/25 of the tokens: programme mean 0.582 (one seed)
  against 0.577, peak GPU memory 0.14 against 0.77 GiB. They do better on M3→MC3 (0.72 against 0.61) and pick up some
  short-range signal (S1, S2: 0.14, 0.21 against ~0), and lose some on the rings (0.80 / 0.87 against 0.95 / 0.96).
- The linear decoder is the right one for a global-only model; the other two are worse (below).

**What doesn't work**
- Every per-cell global-only arm is blind to the short-range programmes: S1→R1, S2→R2, S3/S4→R3/R4 at 0.00-0.03 against
  the bar's 0.55-0.66. Within their receivers they are strongly *negative* (−0.56 to −0.86). M11→MC14/15, which a 150 µm kernel
  solves (bar 0.92), reads 0.14-0.19. `lin_rope` in `response_masking` gives the same zeros, local heads or not. The joint model
  reads these programmes (0.59-0.68) only through its GCN. Short range is the largest single loss of the global-only model, and
  no decoder fixes it: the attention never selects the near neighbours (attention check: `lin_fourier` puts 0.5% of its weight
  within 300 µm, below uniform).
- The skip closes only about half the gap to `ref_own` on the mirror programmes (MC4 0.49 against 0.68, MC1 0.63 against 0.84),
  although `W_o e_own` is linear in the same visible genes that `ref_own`'s ridge reads. The cause is not known yet: optimisation
  (W_o starts at 0, 100 epochs, the transformer path already fits most of the loss), or the input layer's width as a rank limit.
  The all-hidden cosine does not move (0.764 against 0.761): it is dominated by the long-range genes.
- Without the skip, pooling destroys the sparse programmes (M12-M18 at 0.00-0.03 against 0.91-0.98): a patch token cannot say
  which of its cells is a receiver. A pooled model needs the skip.
- `linear-lse` loses the sparse programmes (M12→MC16 0.28, M16→MC20 0.08, M17→MC21 0.06; programme mean 0.324) and `nonlinear`
  loses M16 / M17 (0.06 / 0.03) and its magnitudes (raw SCE 0.071 against 0.039; mean 0.472). Drop both for global-only models.
- Sender ranks do not move with the decoder: 0.51-0.59 per class for `lin_fourier` (mid class seed spread 0.08), 0.52-0.56 with
  the skip, twin 0.49-0.54. The pooled arms rank short-range senders 0.70-0.75, but there the sender usually shares the
  receiver's patch, so this is not yet a sign of selection.
- These runs used a library and sweep script older than the oracle and D0 (committed 16:06 on 2026-10-21, report rebuilt 16:00):
  no `ref_oracle`, no attribution shares, so the skip's `own_cell` share is not measured. Re-evaluate the checkpoints
  (`MODE = "evaluate"`) to get them.

## 2026-10-22 — Each head's loss on its own regime (D2.2, `regime_loss`: `joint` and `regime` at 3 seeds, `regime_remask`, `joint_d1000`, `regime_d1000` at 1 seed; synth_spot, 500 µm discs in the evaluation)

**What worked**
- The heads specialise as assigned, and the effect is far larger than the seed spread (0.002-0.012). The local head trained on
  near cells only gains near the rim: 0.443 → 0.483 (0-150 µm) and 0.423 → 0.466 (150-290 µm). On 0-150 µm this ties
  `kernel_400um` (0.481) and is the best per-cell number at the rim so far. The global head trained on deep cells only loses near
  the rim (0.428 → 0.385) and stays the same deep (0.400 → 0.406, 0.382 → 0.390). CKA local~global over hidden spots
  falls 0.485 → 0.414, at the rim 0.888 → 0.848.
- Regime loss and re-masking go together without a cost (one seed): global 0.400 at every depth (regime alone 0.394), local
  0.490 / 0.472 near the rim, CKA over hidden spots 0.369, the lowest of the sweep.
- At depth the global head reads the label `region` better than the smoother does (balanced accuracy 0.900 against 0.863 at
  500+ µm), but only 0.03 better than smoother+position, on 135 spots.

**What doesn't work**
- No head beats the smoother in its own regime. The deep global head (0.406 / 0.390) is below `kernel_400um` (0.448 / 0.413)
  and `nearest_50` (0.444 / 0.413). Taking the local head near the rim and the global head deep gives
  0.483 / 0.466 / 0.406 / 0.390, against 0.481 / 0.467 / 0.448 / 0.413 for the kernel: a tie near the rim, a loss deep.
  So the regime loss separates the heads because of what each one is trained on, not because the global head learns
  something the local one cannot.
- Wider training discs (1000 µm, so 64% of the hidden spots are deep instead of 34%) make the global head worse at every depth,
  including the deep bins it now trains on much more: `joint_d1000` 0.345 / 0.340 / 0.315 / 0.301, `regime_d1000` about 0.34
  everywhere, against 0.40 for 500 µm discs (one seed each, the gap ~10× the seed spread). More deep training cells do not
  teach a long-range rule. Less visible context to smooth just makes the head worse.
- Specialising costs the global head the rim (−0.04) and nothing comes back elsewhere. The blend scores show the same thing:
  `blend_global` 0.460 → 0.448.
- Cost: the label and source-distance probes take 1,440-1,775 s per arm, about 25 min of a ~30 min evaluation. Cut or
  subsample them in the next disc sweeps.
- The redundancy table (D0) is missing: these runs predate it. Re-evaluate the checkpoints to get it.

## 2026-10-22 — Re-masking hidden cells before the transformer (task 3, `remask_test`: `joint` and `joint_remask` at 3 seeds with the long-range mask, `joint_nolrm` and `joint_nolrm_remask` at 1 seed without it; synth_spot, 500 µm discs)

**What worked**
- The mechanism works. With the long-range mask on, a rim cell's global head now draws 94% of its influence from beyond
  300 µm (was 7%), CKA local~global falls at the rim (0.888 → 0.776) and over all hidden spots (0.485 → 0.297), and hidden
  queries stop attending to hidden keys (visible share over key share 0.77-0.87 → 1.14 in layer 0, where above 1 means the
  visible keys get more than their share). With the long-range mask, a hidden cell's global embedding is now provably made
  from context beyond the GCN's reach, which is the property an interpretation of the global head needs.
- Without the long-range mask, re-masking costs nothing (global 0.417 against 0.418 at every depth) and still lowers the CKA
  over hidden spots (0.473 → 0.254).

**What doesn't work**
- With the long-range mask on, the global head loses 0.02-0.05 at every depth: 0.428 → 0.378 at the rim (expected: the near
  visible cells are now out of its reach), but also 0.400 → 0.376 at 290-500 µm and 0.382 → 0.361 beyond 500 µm, where nothing
  was copied (the local head reads 0.05 there). The likely reason: under plain masking the deep cells drew on the hidden
  rim cells' GCN tokens, which carry the rim's visible neighbours, so the hidden-token "sink" of K5 was partly a relay. After
  re-masking those tokens are all the same and attention avoids them. Long-range response genes: 0.813 → 0.720.
- Without the long-range mask the copying moves into attention rather than going away: the rim cells' global head attends
  nearer (span 0.66 of uniform's, was 0.83) and still draws 79% of its influence from within 300 µm. The global head then
  reads the same near cells as the local head, only through attention instead of its own token.
- Nothing beats the smoother: best head 0.418 against `kernel_400um` 0.465; planted senders at 0.47-0.54 in every arm.
- The local head loses a little when the GCN no longer feeds the hidden cells' global tokens (0.443 → 0.434 at the rim; in
  `nolrm` 0.421 → 0.403 at 150-290 µm).

## 2026-10-22 — Cost of the label probes on synthetic arrays shaped like a disc sweep's (no model: 18,000 hidden training spots, 6,000 test, 64-d embeddings, a 960-d smoother, 4 classes; 6 of the 12 feature sets)

**What worked**
- Fitting each probe on a seeded subset of the training rows cuts the cost in proportion: 6,000 rows took 8.9 s instead of 34.5 s
  (3.9 x), 3,000 rows 5.7 s (6 x). One draw serves every feature set and label, so the sets stay paired.
- The accuracies at 6,000 rows stay close: smoother 0.250 -> 0.260, smoother+global 0.275 -> 0.269, smoother+position+global 0.271 ->
  0.268; at 3,000 rows they move further (global 0.346 -> 0.288).

**What doesn't work**
- The synthetic fits do not reproduce the real cost. Here the logistic regressions converge in 16-140 iterations and a smoother
  fit takes 10 s, so 12 sets x 4 labels would be 8 minutes; the disc sweeps spent 1,440-1,775 s on this step. The real fits must run
  many more iterations (or on wider sets) than these, so the speed-up of `max_iter` 300 against 2000 cannot be measured here (on these arrays it
  changes nothing: no fit reached 300), and the row cap's 3 x is a ratio of rows, not a measured one on the cluster's features.
- A cap moves single-set accuracies by more than the gains the tables read: global alone fell 0.346 -> 0.315 at 6,000 rows on this one draw,
  while the real region gains over the smoother are 0.01-0.02. The smoother and smoother+embedding are cut to the same rows, so the
  gain rows are paired, but their noise from the subsample was not measured. Probe numbers from capped runs are not comparable to the
  uncapped ones of the first disc sweeps.

## 2026-10-22 — The context query (D1) on toy data: is a cell blind to itself, and can the read-out still learn from the context (no model run on synth_spot; stirred parameters, 16-100 cells, 2 layers, 4 heads, width 16)

**What worked**
- A cell's global embedding does not move, bit for bit, with its own expression: at one to three layers, with a node encoding, an attention bias or RoPE,
  for a pooled patch, and behind a 2-hop GCN with the long-range mask (without the mask it moves: the neighbours' GCN tokens hold the cell). The plain
  transformer's moves by 0.19-0.26 on the same perturbation. Through the instrument of the sender ranks (integrated gradients from a baseline cell) the
  receiver contributes exactly 0 to its own predicted response under the query stream and a nonzero amount under the plain transformer.
- The read-out equals its definition written out with plain tensor operations (to 1e-5, also centred); the attention of a cell has weight 0 on every blocked
  pair and on padding, rows sum to 1, the CLS key stays open; the queries follow the kept cells of a subsampled slide.
- A toy where the other cells determine the target (the mean of gene 0 over the others) is learned by the query stream, held-out correlation 0.53 and 0.74
  at two seeds after 300 steps (the plain transformer: 0.73 and 0.55); a target that copies the cell's own gene 0 is learned by the plain transformer
  (0.995, 0.993) and not by the query stream (-0.07, 0.03). So the stream has no way to the cell and keeps the way to the context.
- With `attention.center_qk` a cell enters its embedding through the mean it is centred on, and the dependence falls faster than 1 / S: 1.1e-3, 6.5e-4, 2.8e-4,
  9e-5 at 16, 36, 64 and 100 cells, against 0.19-0.26 for the plain transformer and 6e-3 to 2e-2 for the effect of the same perturbation on the other cells.

**What doesn't work**
- At the default initialisation every node-encoding projection and attention-bias table is zero and the layer norms are identities, so the first versions of the
  tests ran with the encoding and the bias off: six mutations survived (the encoding dropped from the queries, the memory left un-normalised, norm2 for norm1,
  the encodings not following the kept cells) until every parameter was moved by seeded noise (`tests-stir-the-init` in memory). Two tests that built their
  own module were still unstirred and showed it in a second run.
- A stirred start (noise 0.1 on every parameter) learned the "mean of the others" toy at -0.09 in 300 steps, from the default start 0.53: random bias tables
  concentrate the attention at the start, and the stream has to undo them. A trainability check must start from the default; it is also a small warning about
  the read-out's first steps.
- Exact blindness holds with `center_qk` off only, and the sweeps' default has it on (`add_ctx_fourier_nocenter` is the arm without).
- Nothing here says what the global part learns on synth_spot. The expectation that two linear parts (context plus the cell) cannot gate, so the sparse
  programmes that multiply the cell's receptor and a distance fall, is an expectation: the sweep `context_query` is its test.

## 2026-10-23 — Moran's I against the neighbourhood size: does the instrument read synth_spot's planted scales (model-free, the 8 test slides; toy models for the fold passes)

`evaluation.gene_scales` (roadmap.md, task 1), on `layers['log1p_norm']` of `/home/lehnerl/Arbeit/data/synth_spot.h5ad`'s test split (19,877 spots,
253 genes): shells of k = 4 ... 512 neighbours (100 µm ... 1.44 mm), 2,000 centres per slide, a 10-shuffle null. Figures and the per-gene table in
`/home/lehnerl/Arbeit/data/synth_spot_gene_scales/` (`design_scales.png`: every gene's curve coloured by its design class).

**What worked**
- With every cell a centre the estimate is the textbook Moran's I (row-standardised k-NN weights) to 1e-12, per shell and cumulative; float32 sums agree
  with float64 to 2e-7. Sampled centres: the ratio sum z_c * mean z_neighbours / sum z_c^2 over the centres was at most 0.035 off on a smooth field read
  from 300 of 900 lattice cells (20 draws); dividing by the slide's variance instead was up to 0.15 off (median 0.05) and read I = 1.11.
- The measured scale (where the shell curve halves after its peak) follows the planted one where the planted one is a scale of variation: SP9 265 µm
  (planted 200), SP10 310 (180), SP8 363 (250), SP7 452 (300), SP6 761 (350), SP5 765 (500), SP3 813 (600), SP2 996 (700), SP4 1,033 (1,000), SP1 beyond
  1.44 mm (900). Region markers stay up beyond 1.44 mm. Spearman with the planted scale 0.59 over the 53 variable genes that have one.
- The short-range receivers R1-R4 are found from the second shell: about 0.01 in the 4 nearest spots, 0.42-0.47 in the next 4 (their senders sit
  diagonally), halving at k = 12.5, 190 µm: local scale. The first version read the first shell only and called three of them "none".
- 168 of the 170 noise genes sit inside the null; batch and condition genes (slide-level) are flat within a slide.
- Cost on the 8 test slides: 0.9 s for the neighbours, 2 s per map, 19 s for the 10-shuffle null, 22 s for the data side (145 s at first: k up to 1,024,
  float64, 20 shuffles).
- A constructed check (rough fields on a "local part", smooth fields on a "global part"): Spearman of the scale with the global share 0.76, each part's
  curve and map as constructed. Through a real stirred additive dual model on toy slides, the fold passes give every cell one row, the parts add up to
  the prediction, a cell's own values never reach its row while a visible neighbour's do.

**What doesn't work**
- The planted scale is a generative parameter, not the scale of variation. Rings read at their width: MC7 726 µm, MC8 539, MC9 524 for radii of 1-3 mm;
  compact sources at about twice their radius (M12-M21 465-475 µm for 200). So the axis is how far a gene's variation reaches, not how far an interaction
  reaches: a long-range signal can make a locally varying gene.
- The senders S1-S4 (random spots) have no autocorrelation and no scale: Moran's I sees a one-spot interaction on the receiver side only.
- The sparse responses exist at receivers only: Moran's I 0.003-0.11, 6 of 10 "variable" at small effect sizes with short measured scales against
  planted 1-3 mm. They pull the planted Spearman down (0.72 over 48 genes before the receivers and these were read at their peak).
- Per-gene null SDs from 10 shuffles made 4 noise genes look structured (3 "anti", 1 "variable"); flooring each gene's SD at the median over genes and
  correcting the peak test for the 8 shells leaves 2 (NB40 anti at z = -3.6, POI5 variable at I = 0.017).
- Nothing here says what a trained model does. Under cell masking at 30% a hidden spot's neighbours are mostly visible, so the local part can read any
  smooth gene whatever its scale; a gene that varies only within the GCN's reach can only be the local part's. Expect the global share to rise with the
  scale (the Spearman) rather than pass 0.5. The sweep `gene_scales` is the test.

## 2026-10-23 — Orthogonality terms on constructed data and a toy dual model: what each measure reads, and dropout (D4; no synth_spot run)

The eight `orth_*` terms of decoder_plan.md, D4 (`train.orthogonality`), on constructed matrices, and through `train()` on ten toy
slides (dual additive model, n_embed 8, one transformer layer, 30% of the cells hidden, 15 epochs, weight 1).

**What worked**
- Each measure matches its definition: linear and kernel CKA the textbook formulas, the canonical overlap the canonical correlations
  from a QR (and, on loadings, the mean squared cosine of scipy's principal angles, exactly), OPL's term a double loop.
- The loopholes the plan names are real and the scale-free forms close the first: a copy hidden under an inflated unused direction
  reads linear CKA 0.0015 and canonical overlap 0.997; a nonlinear copy (y = x²) linear CKA 0.009, kernel CKA 0.43 (independent
  0.002); a copy moved into the orthogonal complement of a shared space OPL's term 0.0 with the overlap at 1.0.
- Optimised alone on a stirred toy dual model, every term falls fast: CKA 0.39 → 0.005, canonical overlap 0.24 → 0.02, gene
  correlation 0.06 → 0.003, kernel CKA 0.46 → 0.01 (300 Adam steps; most of the fall in the first 60).
- Trained without dropout, the term lowers its measure at evaluation: CKA 0.998 → 0.10 and canonical overlap 0.47 → 0.19 (80
  cells per slide), 0.51 → 0.29 (30 per slide), the run without the term against the run with it; the reconstruction converged
  in both.

**What doesn't work**
- With dropout (0.1 in the GCN and the transformer, the sweeps' default) the term is met by noise: the training value of `orth_cka`
  fell 0.97 → 0.02 within three epochs while the validation value stayed at 0.999, and the canonical overlap ended higher with the
  term than without (0.62 against 0.55). Dropout adds noise independent between the two parts in training only. Every term that
  reads cells now refuses dropout on its path; an orthogonality sweep has to train every arm, its baseline too, without dropout.
- A global-only model's own-cell part is constant on the loss's cells under cell masking (a hidden cell's own token is the fill),
  so `orth_gene_corr` is 0 there and does nothing; under gene masking it works.
- The toy runs say the terms can lower redundancy; not whether that separates the scales or costs the global part its long-range
  programmes. That is the sweep's question.

## 2026-10-23 — The additive decoder of the dual model under programme masking (D3.1, `additive_decoder`: `joint_fourier` and `add_fourier` at 3 seeds, `add_fourier_l01`, `add_fourier_l2g`, `add_fourier_pool500`, `joint_fourier_pool500` at 1 seed; synth_spot, results in `results/22_10_results/additive_decoder`)

**What worked**
- The sum beats the plain dual model's global head: centred cosine 0.843 against 0.820 (seed spread 0.005), programme mean 0.687
  against 0.628, wins / ties / losses against the bar 5/12/7 against 3/12/9. The first arm whose programme mean reaches the bar
  (`ref_own+context` 0.685). The gain is on the mirror programmes (MC1 0.55 → 0.73, MC4 0.26 → 0.48, MC6 0.48 → 0.60) and M11
  (0.54 → 0.65); the rings hold (0.96 / 0.96).
- A small local term costs the sum nothing and makes the local part a predictor: `add_fourier_l01` (weights 0.1 / 0.9, one seed)
  sum 0.833, programme mean 0.689, local part alone 0.522 (the plain model's local head 0.506), against 0.127 with the loss on the
  sum only.
- The additive decoder makes the pooled dual model work: `add_fourier_pool500` 0.772 against `joint_fourier_pool500`'s global head
  0.626, the sparse programmes 0.79 against 0.08 (the local part says which spot is a receiver), at 0.235 against 0.809 GiB. Still
  0.07 below the per-cell sum (rings 0.83 / 0.90 against 0.96).
- The integrated-gradients sender ranks (built 2026-10-20, first run on data here) find the near senders: the global head ranks
  the short-range senders at 0.98 and M11's (within 500 µm) at 0.96, the untrained twin 0.48-0.55; the local head 0.95-0.99
  short, 0.69 mid (out of its reach). The attention check's pooled ranks (2026-10-21) hid this: of 11,809 planted pairs, 273
  are short and 64 mid.

**What doesn't work**
- Without a context-only global embedding the global part repeats the local one: it carries 60% of the short-range receivers'
  variance, and on the mirror programmes the parts work against each other (attribution norms 1.38 global and 1.63 local on MC4,
  1.08 / 0.83 on MC1).
- The "part alone" rows (`local`, `global_part`) of an additive arm cannot be read as scores: each decoder keeps its own bias and
  the centred cosine centres on the gene's training mean, so a part alone carries an offset (`local` raw SCE 0.067-0.24). The
  attribution shares, which centre each part on its own mean, are the read.
- Staged (`add_fourier_l2g`): 0.750, 2/9/13, worse than joint.
- What no arm reaches: M11 0.65 against the bar's 0.92 (a 150 µm kernel), the stripe and the cone (MC12 / MC13 0.18 / 0.18
  against the oracle's 1.00 / 0.83), the distance part of the sparse responses (MC20 / MC21 0.63 / 0.50 against 1.00).
- Planted senders of the dense, sparse and directional programmes stay at 0.52-0.60: those predictions do not use the senders.
- An IG rank says the prediction depends on the sender's driver among cells as far, not that attention selected it: a fixed
  kernel that reads the driver would rank the sender first too. It has no model-free bar yet.

## 2026-10-23 — The context query under programme masking (D1.2, `context_query`: `ctx_fourier_skip`, `add_fourier`, `add_ctx_fourier` at 3 seeds, `ctx_fourier`, `ctx_fourier_skipraw`, `add_ctx_fourier_nocenter`, `ctx_joint_fourier` at 1 seed, 2 twins; synth_spot)

**What worked**
- Context query and additive decoder together (`add_ctx_fourier`) separate the parts at no cost: sum 0.842 against 0.843 for
  `add_fourier` (seed spread 0.002 / 0.005), programme mean 0.691, 4/14/6. The local part now carries all of the short-range
  receivers' variance (1.00, was 0.40), the sparse responses (1.00), M11 (0.99) and the mirror programmes (0.88-0.92); the
  context part the rings (0.76 for M7, 1.00 for M8) and 0.84 of the long-range group. The norms equal the shares (no cancelling,
  MC4 0.34 / 0.94 against 1.38 / 1.63 without the query).
- So the decomposition answers what the model learned from beyond the cell's neighbourhood on synth_spot: the rings, and nothing
  else that the cell and its GCN neighbourhood do not already give.
- A global-only context-only model loses exactly what the plan said: `ctx_fourier` keeps the rings (0.94 / 0.97) and drops the
  mirror programmes (0.02-0.09) and the sparse responses (~0); the own-cell skip brings them back (`ctx_fourier_skip` 0.743
  against `lin_fourier_skip` 0.761, sparse 0.92-0.98). The feared loss of the sparse programmes under two additive parts did not
  show: within their receivers every model, and `ref_own`, reads 0.93 / 0.93 / 0.95 / 0.99 on MC16-MC19.

**What doesn't work**
- The context query in the plain dual decoder (`ctx_joint_fourier`) is worse: the global head 0.709, 0.007 on the short-range
  receivers and 0.03 on the sparse responses, since it has to predict them alone. D1 needs D3.1.
- The context part still ranks the short-range senders at 0.85 and M11's at 0.83 although it cannot see the cell. A likely
  reason (not checked): the long-range mask blocks the keys within two hops, but keys three to four hops away carry GCN tokens
  that hold the receiver's neighbours. "Context" is everything but the cell, not everything beyond its neighbourhood. These
  ranks are also coarse: the completeness gap of the context part is 0.17-0.27 (over 0.1: too few integration steps).
- `add_ctx_fourier_nocenter` (one seed) 0.825 against 0.842: exact blindness costs 0.017, unexplained, one seed.
- Global-only models never pick the near senders, context query or not: short 0.51-0.53, mid 0.51-0.55.
- The mirror programmes stay below `ref_own` in the dual sum (MC1 0.73 against 0.84): the local part is the GCN token, rank at
  most `n_embed`, the limit the raw skip removed in the global-only model (below).

## 2026-10-23 — The full-rank own-cell skip and the re-evaluated global decoders (`global_decoders`: `lin_fourier`, `lin_fourier_skip`, `lin_fourier_skipraw` at 3 seeds, nonlinear, LSE and three pooled 500 µm arms at 1 seed; synth_spot)

**What worked**
- The raw skip closes the mirror gap: MC1 0.63 → 0.80 (`ref_own` 0.84), MC2 0.56 → 0.66 (0.68), MC4 0.49 → 0.66 (0.68), MC6
  0.60 → 0.66 (0.67). Programme mean 0.577 → 0.611, 5/10/9 → 6/13/5. This answers the 2026-10-22 question: the token skip's
  rank was the limit, not the training.
- Pooled 500 µm tokens with the raw skip are the best global-only arm (one seed): programme mean 0.625, 4/13/7; the short-range
  receivers 0.21-0.28 (the patch holds the neighbours), against 0.00-0.02 for every per-cell global-only arm.
- The re-evaluation adds what the 2026-10-22 read lacked: the oracle (all arms 21-23 losses against it) and the shares (own cell
  0.07 of the variance overall, 0.67-0.94 on the mirror programmes).

**What doesn't work**
- Per-cell global-only models stay blind to the short-range programmes (0.00-0.02, within receivers −0.2 to −0.8) and to M11
  (0.16-0.20), and rank every sender class at 0.50-0.59.
- The overall cosine hardly moves with the skip (0.761 → 0.769): it is carried by the long-range genes.

## 2026-10-23 — Channel masking on synth_spot, evaluated on the programme (T3, `channel_masking`: `lin_fourier_ch30` and `_ch50` at 3 seeds, `joint_fourier_ch30` and `_ch50` at 1 seed, a twin)

**What worked**
- The evaluation is the programme's: the programme mask is drawn explicitly (`EVAL_MASK_GENES`), the references equal the other
  sweeps' to three decimals, the untrained twin is at 0.
- The arms learned something above the twin (MC25 0.54-0.72, MC1 0.33-0.35, MC10/MC11 0.44).

**What doesn't work**
- Channel masking does not transfer to the programme: global-only 0.245 (30%) and 0.221 (50%) against 0.764 for the same model
  trained on the programme and 0.405 for `ref_own`; programme mean 0.16 / 0.15, 0/1/23. Even the rings (0.20 against 0.94)
  and the sparse responses, which only need the spot's own receptor (0.12 against `ref_own`'s 0.79). The joint model's GCN head
  reads 0.087 (0.506 under programme masking). Every sender class at 0.47-0.53.
- 50% is not closer than 30%, against the sweep's expectation.
- The model hardly fits its own task: training loss 0.76-0.83 (0.14-0.29 in the programme-masking sweeps; other hidden entries, so
  only roughly comparable), validation SCE 0.12-0.17 against 0.04-0.07. A likely cause, not checked: 170 of synth_spot's 253
  genes are noise, so about two thirds of each step's hidden entries are noise and dominate the per-cell SCE, while the 29
  programme genes are hidden in 30% of the steps. Not yet known whether the model fails its own random masks or only the transfer.

## 2026-10-24 — The local / global split on synth_spot, with a distance prior (`local_global_split`: cells and programme tasks; `add_ctx_fourier`, `add_ctx_fourier_l01`, `add_ctx_dist` at 3 seeds, `add_fourier`, `add_ctx_fourier_dist`, `add_ctx_dist_frozen` at 1, and in the programme task `lin_fourier_skipraw`, `lin_dist_skipraw` at 1; results in `results/local_global_split/synth_spot_*`)

The distance prior is `pe.distance.kind: scales` (four Gaussian windows, geometric over 290-3000 µm for the dual arms, which keep the
long-range mask, and 100-3000 µm for the global-only ones). Moran's I against the neighbourhood size (`evaluation.gene_scales`) per
gene, over all 253 genes in the cells task and over the 29 hidden genes in the programme task.

**What worked**
- A distance window gives a global-only model the short range. `lin_dist_skipraw` (one seed) reads the short-range receivers at
  0.97 / 0.97 / 0.94 (R1, R2, R3/R4; oracle 1.00), within their receivers 0.97 / 0.98 / 0.94, against 0.84 / 0.84 / 0.68 for
  `ref_own+context` and the dual models' GCN path, and 0.01 for the same model with the Fourier bias. It ranks the planted
  short-range senders at 0.94-1.00 (twin ~0.5; Fourier 0.43-0.55), and the context part carries those programmes whole (share
  1.00). Its programme mean, 0.728 with 8 wins / 14 ties / 2 losses against the bar, and the whole-cell 0.905 are the best of any
  model so far (dual sums 0.83-0.85). Everything else equals `lin_fourier_skipraw`.
- With the prior the dual model's split follows the genes' scales. Spearman of a gene's measured scale with its global share,
  per seed: cells task `add_ctx_dist` 0.30 / 0.40 / 0.52 against `add_ctx_fourier` -0.27 / -0.14 / -0.23 (twins 0.19 and 0.00);
  the share against the planted scale 0.55 / 0.71 / 0.60 against 0.09 / 0.26 / 0.21. Programme task 0.48 / 0.67 / 0.55 against
  0.23 / 0.35 / 0.32. The planted-global class gets a median global share of 0.14 (local class 0.01). Learned and frozen
  scales give the same (0.33 frozen, one seed).
- At no cost to the prediction: every dual arm's sum is within 0.03 of `add_ctx_fourier` on every programme (seed spread up to
  0.04), cells-task cosine 0.77 for all.

**What doesn't work**
- In the dual model the prior changes no score: the long-range mask blocks the 2-hop GCN's reach and the windows start at 290 µm,
  so the short range stays with the GCN (0.56-0.68), below what the global-only window reads. Its only visible moves are the cone
  (global share of M10→MC13 0.29 → 0.47-0.64; within receivers 0.42 → 0.49-0.63, one seed each) and the mid-range sender rank
  (M11 0.88 → 0.99).
- M11 (single sources within 500 µm) is unsolved by every arm: dual 0.64-0.67, global-only 0.21-0.22 (within receivers 0.01),
  bar 0.92. The stripe and the cone stay at ~0.2 against the oracle's 1.00 / 0.83.
- Under cell masking the global part carries little even with the prior (median share 0.07-0.10; without it 0.003): with 70% of
  the cells visible the GCN reads nearly everything. Every arm reads the spatially variable genes at 0.77, below a 150 µm kernel
  (0.82): the reconstruction is still under a smoother (K1).
- The integrated gradients do not add up for the global part of the distance arms (completeness gap 0.45-0.62, against 0.13-0.17
  without the prior): its sender ranks are not readable at RESPONSE_STEPS = 5.
- `lin_dist_skipraw` is one seed: the decisive result rests on one run. The prior costs ~1.4x time per epoch (3.2 against 2.2 s)
  and +0.3 GiB.

## 2026-10-24 — The local / global split on legnini, with a distance prior (`local_global_split`, the same arms; windows 55-1000 µm dual, 15-1000 µm global-only; results in `results/local_global_split/legnini_*`)

**What worked**
- Programme masking on legnini (identity code + SHH readouts hidden in every cell): every trained arm is at or above the best
  reference, a first on legnini. Whole cell 0.77-0.80 against 0.75 (`ref_own+context`); per program mean 0.74-0.77 against 0.728,
  1-2 wins, no loss. The gains: motor neuron 0.59-0.66 against 0.53, intermediate dorsal 0.84-0.85 against 0.79; ventral
  progenitor +0.01-0.04, BMP dorsal +0.02. Seed spread 0.000-0.012.
- With the prior the context part carries about half of the identity code (ventral progenitor 0.26 → 0.44, motor neuron
  0.29 → 0.50, intermediate dorsal 0.30 → 0.49, BMP dorsal 0.49 → 0.73), while the floor plate (FOXA2) stays wholly local; the
  scores do not move (within 0.01).

**What doesn't work**
- No arm separates from the others on legnini's scores: dual, global-only, prior or not, all 0.77-0.80. The SHH readouts tie the
  reference (0.57-0.63 against 0.61).
- Cells task: no split by scale (`add_ctx_dist` Spearman 0.00 / 0.15 / 0.03 against its twin's 0.09; `add_ctx_fourier`
  -0.16 / -0.08 / -0.18). legnini's genes vary over a narrow band (median scale 57-100 µm), about the GCN's 55 µm reach. Every arm
  reads the variable genes at 0.56, level with a 15-40 µm kernel (0.57).
- In the programme task the untrained twins already reach a Spearman of 0.29-0.42 on the 18 genes: that number cannot be read
  there.
- Two test organoids; locally the two legnini yamls name different data files (legnini23_pp.h5ad and legnini23_split_pp.h5ad), so
  the two tasks' absolute numbers are compared with care.

## 2026-10-25 — The distance window over seeds, a range aimed at M11, PCA and scVI tokens (`synth_spot_window`: `lin_dist_skipraw` 100-3000 µm, `lin_dist_m11_skipraw` 100-1000 µm, `pca_dist_m11_skipraw`, `scvi_dist_m11_skipraw` at 3 seeds, 3 twins; GlobalModel with the raw own-cell skip, programme masking; results in `results/distance_window`)

**What worked**
- The window result holds over seeds: `lin_dist_skipraw` 0.909 ± 0.003 (one seed on 2026-10-24: 0.905), programme mean 0.729,
  8 / 14 / 2 against the bar; short-range receivers 0.97 / 0.97 / 0.95 (seed SD at most 0.014), their planted senders ranked
  0.98. Still the best model (dual sums 0.83-0.85).
- 100-1000 µm costs nothing: 0.905 ± 0.001, 0.730, 8 / 14 / 2. With no window starting past 1 mm the rings hold (M7 0.95, M8
  0.93; per gene only MC11, 4 mm, drops, 0.93 → 0.89), and the mid-range sender rank rises (0.73 ± 0.10 → 0.83 ± 0.05).
- The windows hardly move: every head but the narrowest ends within ~20% of its start (215 → 208-237, 464 → 459-482,
  1000 → 819-843 µm); the narrowest goes 100 → 54-60 µm at strength ~1.5 in the linear arms, a nearest-neighbour head. The range
  is set by the start, not learned.
- What a token keeps decides what the context can read (model-free: the sweep's saved embeddings against synth_spot regenerated
  at signal_scale 6, which reproduces the cluster PCA at R² 1.00). The PCA carries the ZINB background, batch, condition and
  region markers at R² 0.99-1.00, S1 / S2 at 0.18 / 0.23, and M9, M10, M11 and the sparse sources and receptors MR16-MR22 at
  0.00. scVI carries S1 / S2 at 0.01 but M1 / M3 at 0.93 / 0.86, and it is the only arm that wins M3 (0.79 against 0.61-0.65).

**What doesn't work**
- M11 is not a question of reach: heads end at ~210 and ~470 µm, around its 150 µm kernel, and it stays at 0.25 (100-1000)
  against 0.22 (100-3000), `ref_own` 0.24, bar 0.92; the context part carries 0.17 of it. Not known yet whether the learning signal
  from 3-6 single source spots per slide is the limit (a softmax mean over a ~220 µm window gives one spot a few percent), or the
  competition with the other 28 hidden genes.
- PCA and scVI tokens: 0.539 / 0.385, programme mean 0.61 / 0.57, short-range receivers 0.26 / 0.22, rings 0.61 / 0.39. They also
  fit worse and generalise worse: train loss 0.23 / 0.27, validation 0.44 / 0.53, against 0.083 / 0.085 for the linear token.
  Not checked: whether the gap is memorised per-spot noise (the PCA is mostly the ZINB background) or slide-level batch.
- The within-receiver table rewards the receivers' level, not their variation: it centres on the training mean over all cells,
  so the untrained twins read S2>R2 at 0.99-1.00 within receivers (R1 and R2 are constant within their receivers: nothing else
  is there to score), and the seed SD on M11 within receivers is 0.17 (linear) and 0.39 (scVI).
- 3 of the 8 wins (M3, M5, M10) are the bar falling below `ref_own` (0.46 / 0.58 / 0.16 against 0.61 / 0.69 / 0.24); the model only
  matches `ref_own` there. The real wins are the short range and the rings.
- Against the oracle 0 / 5 / 19 (0.73 against 0.98): M9 / M10 (0.22 / 0.23), M11, the sparse responses' distance part (MC20 /
  MC21 0.67 / 0.58), the mirror programmes MC1-MC6 at `ref_own` (oracle 0.95-0.98).
- Gene scales: `rho_ell_share` is higher for the PCA / scVI arms (0.45 / 0.35 against 0.06-0.09) because their context fails on
  the short range, not because they split better; the summary's kernel columns read 0.001 under programme masking (they smooth
  genes hidden everywhere). Neither reads a global-only programme run.
- The PCA arm's sender ranks are not readable (completeness gap 0.21).

## 2026-10-25 — Exact fused attention for whole sections (attention_plan.md stage 6, `attention.kind: fused`; code and toy models, no data)

**What worked**
- The fused attention is the dense model: outputs, maps and every gradient (tokens, projections, the window's scales and
  strengths, spectral and Fourier coefficients) equal dense attention's to 1e-5, with the blocked diagonal, the long-range mask
  (1 and 2 hops), padding, a subsampled graph, RoPE and centred queries and keys, in one block, in blocks of 3-4 queries and of
  one; through the window sweep's GlobalModel too (per cell, subsampled, on pooled patches).
- No S x S: with more than one block no op of a pass, backward included, returns a tensor with two axes over the sequence (the
  dense model's does, the check's control); each block's mask is built when it runs and again for its backward pass, never kept.
- SDPA takes the additive mask and returns its gradient: the CPU math path and the local GPU's memory-efficient kernel (held to
  that kernel in the test) agree with explicit attention to ~5e-7; the window learns through it.
- The default model is unchanged (harness IDENTICAL); a fused model starts from the dense model's weights and leaves the RNG
  where dense does. 34 tests; 27 mutations, all caught after one test was strengthened.

**What doesn't work**
- A maskless SDPA, the first idea in roadmap T6, would not be the windowed model: without the long-range mask the dense model
  blocks each cell's own key, and with an open diagonal the narrowest window head (57 µm, strength 1.44, 100 µm lattice) would
  put about two thirds of its weight on the cell itself at equal content logits. The fused attention keeps the diagonal blocked,
  so each block builds its mask.
- Each block materialises its rows of the bias (heads x rows x tokens floats): bounded memory, but likely the dominant cost.
  Folding the window into q and k (exact, rank 4) would remove it, but its terms cancel at mm coordinates (fp32, centring, and
  the cluster kernel's accuracy to check), and the diagonal would still need a mask per block.
- A "dropout did act" check written as `not allclose(...)` at default tolerances passed without any dropout: rounding between
  the CPU kernels taken with and without a gradient already breaks `allclose` near zero. Ask for a difference of the effect's size.
- Not measured: speed, memory and the kernel's precision on the cluster's GPUs at 10^4-10^5 cells
  (`scripts/fused_attention_benchmark.py`). Not built for whole sections: an inference path without dense maps (the relevance in
  `get_model_output`), readouts on sampled queries, the CosMx pipeline (T0), `visible_keys_only` with the fused attention.

## 2026-10-25 — Why the windowed global-only model does not read M11: hidden alone, 300 epochs, entmax (5 arms, seed 42, one run each; results in `results/m11_tests`)

Arms: `lin_dist_m11_skipraw` (baseline, 100-1000 µm window, linear), `lin_dist_m11_e300` (the same for 300 epochs),
`lin_dist_m11_only` and `ent_dist_m11_only` (only MC14 / MC15 hidden), `ent_dist_m11_skipraw` (entmax attention). The sweep yaml
(`synth_spot_m11`) is no longer in the scripts folder (only a stale .pyc); the other arms run the window sweep's 100 epochs
(`synth_spot_window.yaml`, 2,400 steps), which is the base the sweep was built on.

**What worked**
- M11 was undertrained, not out of reach: 300 epochs read it at 0.955 pooled (receivers alone 0.962) against 0.285 / 0.312 for the
  same arm at the shorter run, bar 0.92 (0.95 in receivers), oracle 0.99. The five discs of MC14 / MC15 are drawn in the held-out
  slide's map (global part), and the planted sender ranks 1.00 (was 0.87) with a completeness gap of 0.09 (limit 0.1). The
  attribution is 31x / 16x uniform within 300 / 500 µm of the receiver and under 1 beyond: the 500 µm kernel, nothing else.
- The route changed, not only the score: the own-cell skip carried 0.75 of M11's variance at the shorter run and 0.05 at 300
  epochs; the context part went 0.25 → 0.95 (global share of MC14 0.29 → 0.98, MC15 0.17 → 0.84).
- The rest of the table moves with it (same seed): overall 0.904 → 0.952, programme mean 0.730 → 0.800, wins / ties / losses
  against the bar 9 / 13 / 2 → 11 / 13 / 0, against the oracle 0 / 5 / 19 → 0 / 8 / 16; train loss 0.039, val 0.040 (0.089 /
  0.085): no sign of overfitting. M19 / M20 (sparse responses) 0.65 / 0.55 → 0.94 / 0.90, S3/S4 0.93 → 0.98, short-range
  receivers 0.95 → 0.98. The windows hardly move again (100 → 45 µm at strength 1.66, 1000 → 800).
- Hiding M11's responses alone helps at the shorter run: 0.285 → 0.62 (ent 0.23 → 0.61), the context part 0.97 of the variance.
  Part of the old failure was the competition with the 28 other hidden genes and the own-cell skip; the rest is length.

**What doesn't work**
- Entmax buys nothing: 0.23 against 0.29 with every gene hidden, 0.61 against 0.62 alone, overall 0.899 against 0.904. Its alphas
  stay near the 1.5 start (1.41-1.66). It costs 2.1x the time per epoch (5.7 against 2.7 s) and 1.4x the memory (1.51 against
  1.07 GiB). Not worth carrying.
- M11 hidden alone at the shorter run stops at 0.62 (0.61 in receivers), below the bar's 0.91: the competition was not the whole
  story. Its sender ranks (0.93 / 0.88) are not readable: completeness gaps 0.97 / 1.40. The baseline's (0.87 / 0.63) are not
  either (0.22 / 0.62).
- Still at the floor with 300 epochs: M9 (0.23, within receivers -0.54), M10 (0.25, oracle 0.83), the sparse responses' distance
  part MC20 / MC21 (0.67 / 0.58, at `ref_own`), the mirror programmes M2-M6 (0.67-0.68 against oracle 0.95-0.98).
- One seed. The earlier 3-seed window sweep had an SD of 0.17 on M11 within receivers at the short run; the gap here (0.31 →
  0.96) is far beyond that, but the 300-epoch result is not replicated, and where in the 100-300 range M11 switches is unknown
  (the training section keeps only the last epoch).

**Consequence:** "M11 is unsolved by every arm" (notes 2026-10-23 to 2026-10-25) holds for 100 epochs only; the same may hold for
the other programmes that sit at `ref_own` (M9, M10, MC20 / MC21, mirror) and for the PCA / scVI arms. A 300-epoch linear
run costs about 13 minutes (2.6 s per epoch).

## 2026-10-26 — Hierarchical attention on synth_spot, Stage A, first read (20 of 24 runs; results in `results/26_10_results/hier_synth_spot_A`)

Programme task, 300 epochs, 3 seeds (hier_fac 2: s42 missing; hier_soft all 3 missing, cause unknown). Two coarse levels (16 regions
of ~156 spots, 256 clusters of ~10), dense candidates, readout concat, the distance windows (100-3000 µm) shared by every level.
Centred cosine on hidden entries, whole cell / programme mean (24 programmes) / wins-ties-losses against `ref_own+context` (±0.02):
full 0.947 ± 0.002 / 0.789 / 11-11.7-1.3; hier_none 0.939 ± 0.003 / 0.779 / 11-11-2; hier_fpc 0.864 ± 0.022 / 0.734; dual_add_ctx_dist
0.862 / 0.726; hier_fac 0.769 / 0.703; gcn 0.512 / 0.591; hier_soft_dir 0.398 / 0.561 (invalid, below). Bar 0.682, oracle 0.995.

**What worked**
- Levels without a prior, joined by the readout (`hier_none`), are within 0.01 of the dense windowed model everywhere (whole cell
  -0.008, programme mean -0.010), keep the short range (R1-R4 0.98-0.99), the rings (M7 / M8 0.96) and M21 (0.98), and run faster
  on 2,500 spots (2.0 against 2.5 s per epoch; 1.4 against 1.1 GiB). Its attention finds planted senders at least as well:
  pair AUPRC short 0.45 (full 0.33), mid 0.15 (full 0.03); integrated-gradients sender rank mid 1.00, short 0.97, dense 0.75 (full 0.81).
- Where a learned prior weight could switch the prior off, the model did: `hier_fpc` drove lambda of the narrowest head at the cell
  level to ~0 (0.06 / 0.16 / -0.02 in layer 0, -0.29 / -0.36 / -0.15 in layer 1), the other heads stayed at ~0.6-1.

**What doesn't work**
- Every prior costs, the more the harder: none 0.939 > fpc 0.864 > fac 0.769. The losses are the short range (S1-S3: fac 0.74-0.77,
  fpc 0.80-0.87 against 0.98), M11 (fac 0.36, fpc 0.53, none 0.81, full 0.90), the rings and M19 / M21. Likely cause: the cells'
  windows are shared with the coarse levels, so a narrow head's parent (cluster centroids ~300 µm apart against a ~50 µm window)
  reads only its own cluster, and the prior then confines the head to it; most spots of a 10-spot cluster sit at its boundary, so
  their nearest neighbours are cut off (the plan's S5). Not yet tested: `coarse_bias: own / none`, lambda starting at 0.
- `hier_soft_dir` is invalid: the direction prior divided by the length of the parent's coarse arrow, which is ~0 when a parent reads
  itself; gradients reached ~1e11 on a toy model, Adam froze the model (lambda, mu and the windows ended exactly at their start, short
  range ~0.02). Fixed 2026-10-26 (`direction_cosine`: arrow normalised by sqrt(|v|^2 + spread^2), a test); rerun it.
- The arrow angle (`arrow_angle_deg`, head mean) does not read: the untrained twin already scores 31° on dense / sparse, as wide heads
  pull every head-mean `read_vec` to the slide centre. Read arrows per head (the narrow ones) or by the top-k targets instead.
- CLS takes 15-35% of the head-mean weight (soft_dir most); the soft prior gives CLS its parent's weight undivided while a cluster's is
  split over its children -- not the cause of the collapse, but a bias to check once hier_soft is in.

## 2026-10-27 — Hierarchical attention, Stage A complete (soft, soft_dir rerun, fac s42) and CosMx fold 0 (results in `results/26_10_results/hier_synth_spot_A`, `hier_cosmx_fast`)

synth_spot, programme task, 300 epochs, 3 seeds each (all 24 runs + 2 twins). Centred cosine whole cell ± SD / programme mean / wins-losses
against `ref_own+context`: full 0.947 ± 0.002 / 0.789 / 11-1.3; hier_none 0.939 ± 0.003 / 0.779 / 11-2; hier_soft_dir 0.892 ± 0.009 / 0.740;
hier_soft 0.887 ± 0.015 / 0.736; hier_fpc 0.864 ± 0.022 / 0.734; dual_add_ctx_dist (global head) 0.862 / 0.726; hier_fac 0.771 ± 0.010 / 0.707;
gcn 0.512 / 0.591. Both twins -0.006.

**What worked**
- `hier_none` stays the hierarchical arm to carry: 0.01 below dense over 3 seeds, everything else 0.05-0.18 below.
- `soft` keeps the short range (S1-S4 0.96-0.98), unlike fac / fpc (0.75-0.87): the narrowest head switches its prior off at the cell level
  in layer 1 in every prior mode (lambda 1.0 → -0.02 soft / soft_dir, -0.27 fpc).
- The repaired direction prior trains (mu 0.3-0.9, no freeze): soft_dir is a valid run now.

**What doesn't work**
- Every prior still costs. The soft prior's loss is mid range and dense: M11 0.32 (none 0.81, full 0.90), M19 0.65 (0.84 / 0.89), M8 0.88
  (0.96). Train / val loss 0.066 / 0.099 against 0.041 / 0.057 for none: it fits less, not overfits.
- The attention is not where M11 fails: soft's head 1 puts more weight on M11's senders than full's (layer-1 sender lift 175 against 90,
  AUPRC 490x against 174x chance). M11 within receivers swings 0.10-0.83 by seed in the prior arms (fac 0.10-0.57, soft 0.10-0.42, fpc
  0.38-0.83) and the context share swings with it: the "M11 has switched or not" pattern of the m11 sweep. The priors slow the switch.
- The direction prior adds nothing: soft_dir - soft = +0.005 whole cell (inside the seed SD), and the directional class (M9 0.22, M10 0.25)
  stays at the floor in every arm, dense included.
- Most lambdas end near their 1.0 start (0.6-1.0): apart from the narrow head the prior weight hardly learns in 300 epochs.

CosMx fold 0 (one seed, 1,000 epochs, whole sections, routed: ~100 keys per query at level 1, ~300-330 at the cell level, 205-401 regions of
250 cells, clusters of 25; no twin, no hier_none, no dense arm). Centred cosine on hidden cells: hier_fac 0.189, hier_soft 0.183, hier_fpc
0.180, gcn 0.069; references knn8 0.172, knn32 0.220, knn128 0.244, section mean 0.209.

**What worked**
- Whole sections per pass are cheap: 6.5 GiB peak, 3.4-3.6 s per epoch (0.7 steps/s), ~1 h per 1,000-epoch run, not the 24 h guessed.
  never_selected 0 at every level.
- The hierarchical arms hold across depth (0.21 → 0.16 from 0-12 to >40 µm); the GCN falls below zero beyond 25 µm (-0.03, -0.11).

**What doesn't work**
- No arm beats a smoother: all below knn32, knn128 and the section mean at every depth; the prior modes are indistinguishable (0.180-0.189).
- Little was learned: attention lift over uniform 0.8-1.2 beyond 15 µm (5.7-14x at 0-15 µm), windows within ~10% of their start
  (7 / 43-57 / 238-243 / 974-1092 µm), lambdas 0.6-1.0 bar the narrowest head. Train 0.75 / val 0.83. Same signature as the
  underfitting reads (legnini round 2, synth_spot at 100 epochs). ~2,500 steps only.
- Test section 3_ND is ~9 separate tissue blocks over ~7 mm: the region level spans gaps between blocks; head-mean arrows are short and
  say nothing.

**Next:** CosMx first: train 10-20x longer (the time allows it), add `hier_none` and an untrained twin, read the loss curve before new arms.
Priors: only with lambda starting at 0 (soft = none at the start) and `coarse_bias: own`; drop the direction prior.

## 2026-10-27 — The field export per head, the own-window baseline and the dependence field (code, tests, a stub on generated synth_spot; no run)

Built for the next sweeps (decided 2026-10-26): `hierarchy_readouts.fields` exports per head cell mass, top-5 and `window_vec_heads`
(the head's trained position biases alone over the same keys, scaled to its cell mass), the planted programmes, and
`receptive_field.receiver_dependence` (IG per receiver, retried with more points over a 0.1 completeness gap); `fields_by_class.py`
draws them and runs after every export. Sweeps written, not run: synth_spot Stage B, the Stage A fields re-read, CosMx `long`.

**What worked**
- Under flat content (query and key projections zeroed) the window baseline equals the attention exactly, dense and hierarchical;
  in the stub figure the wide head's pull to the slide centre disappears once its own window is subtracted.
- The retry rule is pinned against direct calls; 7 mutations of the new code killed.

**What doesn't work**
- Completeness converges slowly where the response barely moves: a toy query at change -0.013 needed ~128 points for a gap under
  0.1, and on stirred toy models 54-98 of 170 queries still missed 0.1 after 20. Read `dep_gap` next to `dep_change`.
- The dependence is expensive: a backward pass per receiver and point (~0.05 s on the GPU); every receiver of a slide (~3,000)
  would cost ~40 min per run, so the sweep takes 50 per programme on one slide (~10 min, estimated, not measured).

## 2026-10-27 — Hierarchical attention, Stage B: the priors repaired (lambda from 0, a window per coarse level; results in `results/27_10_results/hier_synth_spot_B`)

synth_spot, programme task, 300 epochs, 4 arms x 3 seeds (`hier_fpc_l0_own_s43` missing: 11 runs) and 4 untrained twins (-0.006). Centred
cosine on hidden entries, whole cell ± SD: `hier_soft_l0_own` 0.944 ± 0.006, `hier_soft_l0` 0.943 ± 0.003, `hier_soft_own` 0.916 ± 0.010,
`hier_fpc_l0_own` 0.913 (2 seeds). Stage A for comparison: full 0.947 ± 0.002, hier_none 0.939 ± 0.003, soft 0.887 ± 0.015, fpc 0.864 ± 0.022.
No `results.md` in the folder; read from `sections/`. The fields (`fields/`, `fields_by_class/`) not read yet.

**What worked**
- Lambda starting at 0 removes the soft prior's cost: `soft_l0` and `soft_l0_own` reach dense level (0.943 / 0.944 against 0.947), with the short
  range at 0.98-0.99 (S1-S4) and M11 at 0.82-0.90 in every seed (Stage A soft 0.32, none 0.81, full 0.90); train / val loss 0.037-0.042 /
  0.047-0.056, as hier_none.
- A window of its own per coarse level helps when the prior starts on: `soft_own` 0.916 against Stage A soft's 0.887. Its narrowest level-1
  window weakens (seed 44: 100 → 185 µm, strength 1.0 → 0.28; read for s44 only), so the coarse level confines the narrow heads less.

**What doesn't work**
- The prior learns nothing when it may stay off: in the lambda-from-0 arms the median |lambda| ends at 0.05-0.10 (max 0.24-0.32, a few wide
  heads), so `soft_l0` is in effect `hier_none`. Where it starts on (`soft_own`) it stays on (median 0.70-0.91) and costs. No arm with a prior beats
  hier_none or dense: the level above carries no information the cells' own windows lack on synth_spot.
- `soft_own` still has the seed-dependent M11 switch (0.85 / 0.88 / 0.30); the own window adds nothing once lambda starts at 0 (0.944 against 0.943).
- `fpc_l0_own` still loses the short range (S1-S4 0.83-0.90 against 0.98) and fits less (train 0.070, val 0.078-0.088): the factorized per-cell
  prior hurts even from 0 (max |lambda| 0.53-0.92).
- Directional class unchanged (M9 0.21-0.23, M10 0.24-0.26). Peak memory 2.0 GiB (Stage A hier_none 1.4); epoch times 1.2-5.7 s scatter by node.

**Consequence:** on synth_spot carry `hier_none` (or soft from 0, which ends there); priors are not a lever here. The CosMx long run's
`hier_soft_l0_own` arm is safe (it cannot do worse than none) but will most likely read as hier_none.

## 2026-10-29 — Moran's I gene sets: a local and a global set at the GCN's reach, without labels (masking_plan.md M2; model-free, synth_spot's 24 training slides, k-NN 8, 2 hops)

Per gene, Moran's I within the GCN's reach (2 hops on the training graph: 24 cells, 283 µm, measured) and across the slide; the candidates
(above the shuffles, Bonferroni over genes, I_local >= 0.05: 53 of 253) ranked by the fall from one to the other; the smallest falls are
the global set, the largest the local set (n = 10). Read against the planted `length_scale_um`.

**What worked**
- The relative fall `(I_local - I_global) / I_local` ranks the planted scales (Spearman -0.75 to -0.83 over every global statistic and
  prefilter tried). Global set: the three region markers, SP1, M1 / M3 / M5, MC3 / MC5 / MC23 (median planted 1,200 µm); local set: R1-R4,
  SP8-SP10, MC6, MC14, M9 (180 µm). On 4 slides (a stub) the same picture at n = 20 (Spearman -0.76).
- Inverse-distance weights over every other cell of the slide give a global Moran's I with no bandwidth to choose; Gaussian kernels at a
  quarter or an eighth of the slide and the k = 512 shell or cumulative read about the same under the relative rule.
- The reach measured on the graph equals the known 2-hop reach (283 against 290 µm). 10 s for 60k spots.

**What doesn't work**
- The absolute difference `|I_local - I_global|` (the rule as first stated): small for any weakly patterned gene, whatever its scale. The
  four short-range receivers (planted 100 µm, I_local ~0.2, I_global ~0.01) were 4 of the 10 "global" genes (median planted 350 µm), and
  its Spearman with the planted scale swung from +0.12 to -0.55 with the global statistic and the prefilter. Kept as `rule: absolute`.
- "Across the slide" with every other cell weighted alike is degenerate: Moran's I of a z-scored gene is then -1 / (n - 1) for every gene.
- The sparse responses MC16-MC22 are no candidates (I_local 0.00-0.03): a Moran's I selection cannot pick rare-receiver programmes. Weak
  long-range sparse genes that pass the floor (MC23, MC24, I_local 0.06-0.08) land in the global set on a fall measured between two small
  numbers.

## 2026-10-30 — Randomized against pre-selected gene sets under programme masking, and the cost of whole CosMx sections (masking_plan.md M1: 7 arms, 13 runs, 300 epochs, results in `results/29_10_results/masking_sets`; `cosmx_feasibility`, H100)

The dense windowed global-only model on synth_spot; every arm scored on the planted programme (29 receiver / response genes hidden in
every spot) and on draws of its own mask against `ref_own`. Centred cosine, whole cell / programme mean / wins-ties-losses against
`ref_own+context`: fixed 0.947 / 0.786 / 11-12-1; sub50 0.875 / 0.710 / 7-15-2; ch30_svg 0.473 / 0.510 / 1-9-14; ch30_all 0.430 /
0.399 / 1-3-20; moran_local 0.290 / 0.196; moran_global 0.200 / 0.096; pool_k1 invalid (below). Bars: ref_own 0.405, own+context 0.682.

**What worked**
- Randomization within the programme is fine: `sub50` (a random half of the 29 genes, redrawn every epoch) reaches 0.875 on the whole
  programme and 0.89-0.95 on its own task against ref_own 0.56. The hidden unit need not be fixed; it must come from the programme.
- The Moran's I sets are learnable tasks: on their own masks moran_local 0.74-0.76 (ref_own 0.10), moran_global 0.41-0.45 (ref_own 0.18).
- 300 epochs lift channel masking over its 100-epoch read on the programme (ch30_all 0.245 -> 0.430; without noise genes 0.473).
- CosMx whole sections (largest, 3_T1D, 100,230 cells, H100): routed hierarchy 1.2-2.1 s per training step, 6.0 GiB; hier_none 1.15 h per
  1,000 epochs (upper bound), ~10,400 epochs in 12 h; GCN 0.02 s per step.

**What doesn't work**
- Random channels learn no communication even on their own task: ch30_all 0.32 against ref_own 0.34 (the cell's own visible genes),
  ch30_svg 0.57-0.67 against 0.63. Short-range receivers on the programme 0.006 / 0.007 (fixed 0.987).
- Transfer to the programme is not a fair test of an arm that never hid the programme's genes: under gene-set masking the decoder rows of
  never-hidden genes get no gradient. moran_local reads R1-R4 (in its set) at 0.908 and the long-range responses (not in it) at 0.055. Read
  the Moran arms by their own task.
- The Moran sets hold drivers: global = region markers, SP genes, sources M1 / M2 / M3 / M5 and some responses; local = R1-R4, sources M6 /
  M9 / M13 / M18 / M19 / M21 and some responses. Hiding a source removes what predicts its responses. moran_global overfits (train 0.06-0.07,
  val 0.49). MC23 / MC24 (I_local < 0.09) entered the global set on a fall between two small numbers.
- `pool_k1` is invalid: train loss 0.99-1.05, no evaluation section (crash, cause not in the results folder). 18 of the 24 planted sets are
  one gene, and the SCE of a single hidden entry per spot is a sign (+-1), with no gradient on the size. Needs several sets per draw (>= ~10
  genes) or an MSE loss.
- The exact dense windowed model on fused attention does not train on whole sections: 127 s per training step at 100k cells (41.7 s per
  evaluation pass, 5.6 GiB), 165 epochs in 12 h; the dual model the same. Memory is not the limit, the quadratic compute is.

## 2026-10-31 — CosMx whole sections at 300 epochs: the GCN against hier_none, folds 0 and 2 (`cosmx_run`: 5 of 6 runs; results in `results/30_10_results/cosmx_run`)

One whole section per forward pass; cell masking, 20% of the cells at random + 60 µm discs over 10% in training, discs alone in the disc evaluation;
SCELoss on centred targets; 300 epochs (~600 steps); one seed per fold; each fold tests the two sections of one physical slide (f0: 3_ND + 3_T1D,
f2: 1_ND + 1_T1D). `hier_none` = GlobalModel, routed coarse-to-fine without a prior (clusters of 250 and 25, readout concat, top-k 8 / 8), a distance
window per head (15-1000 µm), raw own-cell skip. **`hier_none_f1` is missing**: no section, visibility or field file, and no log in the folder. So:
gcn f0-f2, `hier_none` f0 and f2, and the untrained twin of `hier_none_f0`.

Centred cosine on the hidden spots, discs alone, all depths, **per fold**. The summary table prints fold 0's references and spot count once for every
arm, so fold 1 and 2 arms stand against the wrong bar there (the per-arm sections are right):

```text
fold (hidden spots)   gcn     hier_none  twin    knn8   knn32  knn128  section mean
f0 (15,221)           0.053   0.157     -0.057   0.172  0.220  0.244   0.209
f1 (12,786)           0.199   missing            0.173  0.225  0.252   0.224
f2 (10,984)           0.100   0.174              0.153  0.200  0.223   0.177
beyond 40 um, f0 / f1 / f2:   gcn -0.113 / 0.158 / 0.004;  hier_none 0.134 / - / 0.150;  knn128 0.236 / 0.246 / 0.211
```

**What worked**
- The run: 5 of 6 runs trained, evaluated and exported on whole sections. `hier_none` 2.6-3.2 s per epoch (13-16 min for 300 epochs), 6.5-8.2 GiB;
  the GCN 0.5-0.8 s per epoch, 4.9-7.9 GiB; the whole evaluation (head scores, visibility, fields) 1.5-3.5 min per run. The feasibility bound held
  (0.7-0.9 h per 1,000 epochs against 1.15 h): the request of 4 h and 192 G was ~10x generous, and 3,000 epochs of `hier_none` would take ~2.6 h.
- `hier_none` beats the GCN where the GCN is blind, on both folds: +0.104 (f0) / +0.074 (f2) over all hidden spots, +0.247 / +0.146 beyond 40 µm. The
  GCN falls to -0.113 (f0) and 0.004 (f2) there; `hier_none` is flat with depth (0.171 → 0.134, 0.195 → 0.150). The same on the training-style mask
  (0.182 vs 0.139, 0.199 vs 0.160), in the validation loss (0.840 vs 0.866, 0.820 vs 0.855) and per gene (better than the GCN on 91% / 94% of the 979 genes).
- It learned: the untrained twin scores -0.057 (f0), 0.21 below the trained model.
- Per gene it is a competitive smoother. Mean pooled centred cosine over the 979 genes (training-style pass, the only one with a per-gene table), f0 / f2:
  hier_none 0.100 / 0.118, gcn 0.062 / 0.097, knn32 0.101 / 0.105, knn128 0.117 / 0.127, section mean 0.082 / 0.098. Where the section mean scores below 0.15
  (825 / 877 genes, no large section offset) hier_none is 0.069 / 0.103 against knn32 0.062 / 0.083, knn128 0.080 / 0.105, gcn 0.042 / 0.085. Its gain over the
  section mean falls on the same genes in both folds (correlation 0.62; 49 genes gain > 0.05 in both, 5 lose): the islet / endocrine genes (TTR, GCG, CHGA,
  SCG5: +0.35 to +0.67) and exocrine ones (PRSS2, SPINK1, REG1A). It reads compact tissue structure from context.
- M0 (hidden against visible, masking_plan.md): hidden-cell scores are not flattered by the cell's own input. Under cell masking every model scores
  *lower* on the visible cells (training-style pass, centred cosine hidden / visible: hier_none f0 0.182 / -0.001, f2 0.199 / 0.129; gcn 0.139 / 0.038,
  0.230 / 0.188, 0.160 / 0.115), while every reference stays within ±0.011. Visible cells are no training targets, so a clean-pass score of these models
  means nothing; the masked pass is the right one. The genes a clean pass flatters most are section-offset genes (RGS2 +0.57 on f0; MEG3 +0.33, RPS4Y1 +0.26 on f2).
- The windows moved, unlike in the prior arms of 26-27 Oct (within ~10% of their start): 15 / 60.8 / 246.6 / 1000 µm → 11.2 / 49.4 / 234.8 / 984.1 (f0) and
  11.1 / 49.8 / 223.1 / 979.8 (f2), strength 1.0 → 1.28 / 1.20 / 1.05 / 1.02 and 1.29 / 1.20 / 1.10 / 1.02. Routing is healthy: never_selected 0 at every level,
  ~110 keys per query at the cluster level, 265-290 cell keys (0.3-0.6% of a section).

**What doesn't work**
- `hier_none_f1` is missing (cause unknown: check `cosmx_run_<jobid>_4.err`; `sbatch --array=4 slurm/cosmx_run.sbatch`). The folds are the replicates, so `hier_none`
  has two. f1 is the fold where the GCN is best (0.199, flat with depth), so the gap to the GCN would be smallest there.
- Still below the smoothers, as on 26-27 Oct. `hier_none` is below knn32 and knn128 on both folds (knn128 by 0.087 on f0, 0.049 on f2), below the section mean on f0
  (by 0.052), level with it on f2 (0.174 vs 0.177), and ahead of knn8 on f2 only. On raw SCE it is worse than every reference (0.493 against 0.413-0.465; 0.589
  against 0.438-0.491). Against the 26 Oct runs on the same f0 test sections (1,000 epochs, with priors: 0.181-0.189, gcn 0.069) the GCN is 0.016 lower at 300 epochs
  and `hier_none` 0.03 below the prior arms (another arm, so not a clean epoch comparison). The 27 Oct read asked for 10-20x more epochs first; no 10,000-epoch run
  is in the results folders, and this run went the other way (3x fewer). Undertrained or the model's limit is still open, and the validation curve is not in
  the folder (final values only), so it cannot be read here.
- The headline metric is mostly the section's own offset. Cosines are centred on the *training* mean, so whatever sets a test section apart from it (donor, sex, stress
  state) counts as signal: the section mean alone gives 0.209 / 0.224 / 0.177, 80-89% of knn128, and what neighbours add beyond it is only 0.03-0.05. So every arm
  sits in a band ~0.06 wide, and the GCN's deep score swings with the fold (-0.113 / 0.158 / 0.004), probably the match between its constant output (the training
  sections' offset) and the test sections' (not tested), not context. The genes the section mean wins by most are offset genes: MEG3, COL9A2, RGS2, RPS4Y1 (Y-linked),
  FOS, HSP90AB1. Of the 154 / 102 genes where the section mean scores >= 0.15, hier_none has 0.267 / 0.248 against knn32 0.305 / 0.299, knn128 0.319 / 0.315; on RPS4Y1
  0.23 / 0.07 against the section mean 0.38 / 0.45. The model does not read the state of the section, and on the islet genes the smoothers are still ahead (nine-gene
  mean f0 / f2: hier_none 0.417 / 0.467, knn32 0.506 / 0.572, gcn 0.335 / 0.431).
  Probable cause, not tested: nothing in `hier_none` reaches the whole section. Windows end at ~1 mm, the attention beyond 2 mm is 0.004-0.03 of uniform, a section is ~10
  tissue blocks of ~1 x 0.5 mm over ~7 x 10 mm (fields figure), and the coarse level reads ~4-5 of 205-246 regions (normalised entropy 0.26-0.34; twin 0.37).
- The attention is mostly the prior's. The untrained twin already reads locally (lift over uniform 4.6x at 0-15 µm, 0.70 at 60-120 µm, 0.23 at 500-1000 µm); trained 6.2-7.9x,
  0.72-0.76, 0.32-0.40; coarse entropy 0.37 → 0.26-0.34; keys per query 250 → 265-290. The fields show no readable arrows (two isolated ones on 3_ND; row entropy 4.0-4.9
  nats, ~55-130 keys). `hier_none` here is a distance-weighted smoother with a learned value path; no content-based choice of senders can be seen.
- Raw SCE and centred cosine disagree on f2 (SCE: gcn 0.525, hier_none 0.589; cosine 0.100 vs 0.174; val SCE 0.460 vs 0.480). The loss is SCE on centred targets and scale-free, so
  the size of the predicted deviation is never trained, while the references average and shrink; val_r2 is negative in every run (-0.24 to -0.71). Probably an over-large deviation
  norm (not tested): read the centred cosine, which the loss trains; a shrinkage fitted on the training sections would show it.
- The own-cell skip is dead under cell masking: the `own_cell` share is exactly 0 (a hidden cell's raw input is the constant -1 vector), and at visible cells it only adds junk.
  It came from the programme-masking arms; drop it for cell masking.
- 60 µm discs cannot separate a global model from a wider smoother: the smoothers lose nothing with depth (knn128 0.245 → 0.236, section mean flat); 128 neighbours cover the hole.
- Reporting: the summary's references (above); `visibility/*/maps.png` is solid black in every run (each hidden cell, 28% of 50-100k, gets a 0.3 pt ink ring drawn over the values, so
  truth and prediction cannot be read); no validation curve is written to the results.

**Next, in this order:** (1) rerun `hier_none_f1`; (2) the report: per-fold references in the summary, the maps, a within-section centred cosine (centred on each test section's visible
mean: the section mean scores 0 by construction and what is left is context), the validation curve as a csv -- before the lung sweep, whose donors will carry the same offsets; (3) one
fold-0 `hier_none` run at 3,000+ epochs (~2.6 h), to settle training length; (4) only then a way to read the section (a section token, a head without a distance window) and larger discs.

## 2026-10-31 — Correction to the CosMx read above: what the run measured, which part of CosMx each run used, and a first look at the genes (the user's questions)

**What was used.** The file holds 386,727 cells x 979 genes in 6 sections (3 slides x ND / T1D donor, 9-15 FOVs each). A fold column puts each section wholly in one split, one
slide (ND + T1D) per split; every slide is tested once:

```text
fold    train (2 sections)         val (2 sections)           test (2 sections)
split_0 slide 1:  108,711 cells    slide 2:  126,421 cells    slide 3:  151,595 cells
split_1 slide 3:  151,595 cells    slide 1:  108,711 cells    slide 2:  126,421 cells
split_2 slide 2:  126,421 cells    slide 3:  151,595 cells    slide 1:  108,711 cells
```

- Per forward pass it is the whole section: all its cells, all 979 genes, batch size 1, no windows, no cell removed (f0 test: 42,853 hidden + 108,742 visible = 151,595).
- But **each run trains on two sections (two donors, one ND and one T1D), 28-39% of the cells**: ~2 steps per epoch, ~600 for 300 epochs. The validation slide is not trained on, it
  picks the checkpoint (`train()` restores the best `val_loss` of the 30 validations, one per 10 epochs); only the test slide is scored: in the disc evaluation the hidden spots are the
  discs over 10% of its cells (15,221 of 151,595 on f0), in the visibility pass ~28% (20% random + discs). Everything is therefore a transfer to two unseen donors from two seen ones.
  I had not said this; "the folds are the replicates" holds, but each fold is a different pair of training donors.

**What the run measured and what I wrote.** Only gene-expression reconstruction was scored (head scores, the references, the visibility pass), plus the attention by distance. Statements in
the entry above that went beyond it, to be read as follows:
- "The model does not read the state of the section": it reconstructs the section-offset genes (RPS4Y1, MEG3, RGS2, FOS) worse than the section mean. What its embedding encodes was not
  measured. With two training donors a decoder cannot learn how a donor's state maps onto genes, whatever the embedding holds.
- "No content-based choice of senders can be seen" and "a distance-weighted smoother with a learned value path": only the attention's profile by distance was read (trained against the
  untrained twin). That profile cannot show or exclude a choice among cells at one distance. Sender type (the cell-type labels exist), integrated gradients and the decoder's genes were not run
  on CosMx. Untested.
- The ranking against the smoothers, and "mostly the section's own offset", are statements about reconstruction scores. They say nothing about what the global embedding represents.
Not measured, and what is needed (the design is in roadmap.md, task 2, "global-only form"): the decoder's top genes per embedding dimension, the importance of each dimension, and whether
the genes tied to a dimension vary in space.

**A first look at the genes, local and model-free (no checkpoint).** Moran's I curve and length scale of each gene on the fold's two test sections (`gene_scales`, k 4-512, 2,000 centres,
10 shuffles), joined with the per-gene hidden-cell centred cosines of the training-style pass:
- The genes `hier_none` ranks highest are spatially variable, like every method's: all of the top 100 of `hier_none`, the GCN and knn32 have a "variable" pattern, against 78-80% of all
  genes; Spearman of a method's per-gene score with the gene's first-shell Moran's I: hier_none 0.65 / 0.62 (f0 / f2), gcn 0.68 / 0.64, knn8 0.77 / 0.71, knn32 0.75 / 0.69. Top genes f0: MT2A,
  SOD2, TTR, MT1X, SERPINA3, CHGA, GCG, CD74, B2M, PRSS2; f2: GCG, TTR, CHGA, CPB1, PRSS2, SPINK1, CUZD1, SERPINA3, INS, RGS2 (islet, exocrine, metallothionein / immune genes). It does not
  separate the model from a smoother.
- The edge over an 8-neighbour smoother does not rise with a gene's spatial scale for `hier_none` (Spearman of score minus knn8 with the autocorrelation left at the widest shell: 0.13 / -0.02;
  with the length scale: -0.01 / -0.05), while it does for wider smoothers (knn32: 0.60 / 0.51 and 0.22 / 0.33; knn128: 0.51 / 0.36). Hint only: a pass where 70% of the hidden cells are
  isolated, and a reconstruction score, not the embedding.
- Median length scale of the variable genes 23 / 31 µm; of the top 100 by `hier_none` 31 / 80 µm (TTR, CHGA, GCG, INS 67-126 µm).

## 2026-10-31 — The CosMx field plots (`fields/hier_none_f0`, `_f2`, the twin): what they draw, why they look as they do, what the exported arrays say

`plot_fields` is the synth_spot lattice figure, unchanged. It draws 1,500 random cells (2.4-2.9% of the section) of the last layer's head mean, each an arrow true to scale (1 µm = 1 data unit)
of `read_vec_i = sum_j A_ij (x_j - x_i)`, the attention-weighted displacement to where the cell reads from, coloured by the entropy of its attention row, on the section's bounding box.
- **Blank space:** only 11% (f0, 7.5 x 9.4 mm) and 16% (f2, 7.8 x 8.2 mm) of that box holds cells (100 µm squares): a section is ~10 tissue blocks of ~1 x 0.5 mm. The "patches" are those blocks.
- **Two arrows:** the median `read_vec` is 28 µm (f2: 25), ~2 pixels at ~15 µm per pixel; 90% are under 70 µm and 54 of the 1,500 drawn exceed 100 µm. The two long ones (cells 15 and 25967: 471 and
  372 µm, entropy 1.85 and 1.75) put about half their attention on CLS and read a few far cells (mean distance ~500 µm); they are the same two outliers in the untrained twin (461 and 372 µm): not learned.
- **Entropy** is non-negative (nats; e^H keys read evenly): median 4.55 (95 keys; twin 4.84, 126), 90% of the cells within 4.1-4.7. The colour range is stretched by the 35 cells below 3 nats, so all else is one shade.
- **Few plots:** `EXPORT_GRAPHS = 1` in `cosmx_run.py` (the first test section only: 3_ND, 1_ND; the 100k-cell 3_T1D is not exported), only the hierarchical arms and the twin (the GCN has no attention), fold 1 missing,
  the PNG is the head mean of the last layer (per-head arrays are in the npz, undrawn), `fields_by_class` needs planted classes (synth_spot only).

What the arrays say (last layer, head mean, top-5 cell keys; attention only, not what the embedding holds):
- Trained reading is more lopsided than the twin's: median `read_vec` 28 vs 18 µm, 99% 130 vs 71. It is weakly tied to the tissue edge (Spearman with the offset of the 30-neighbour centroid 0.22-0.35) and partly follows the
  mask: for visible cells within 150 µm of >= 15 hidden cells the arrow leans towards the hidden cells (cosine +0.29 f0, +0.33 f2; twin +0.20; 68-69% of the arrows point towards them).
- Hidden cells read more sharply than visible ones: entropy 4.08 / 4.11 against 4.55 / 4.49 (f0 / f2); the twin 4.88 against 4.82.
- The twin's masked tokens attract each other: a hidden cell puts 91% of its top-5 weight on other hidden cells (58% of its 128 nearest are hidden). Training removed it: 51% (f0) / 58% (f2) against 58% / 56% by position,
  i.e. 49% / 42% of a hidden cell's top-5 weight on visible cells against 42% / 44% expected: at most a few points of preference for the informative keys. Visible cells put 2% of theirs on hidden cells (5% expected).
- This qualifies "the attention is mostly the prior's" (above): the profile by distance is close to the twin's, but the entropy and the treatment of masked keys changed with training.

**Verdict:** on CosMx there are no planted senders, so the arrows have nothing to be compared with, and what they show is geometry plus the mask layout. Do not read this figure for signalling. The replacement is the readout of
`interaction_readout_plan.md` (separate session) and the dimension analysis of roadmap.md task 2; a stopgap would crop to one tissue block with the discs outlined and export both test sections.

## 2026-10-31 — Attention flow maps for whole CosMx sections replace the quiver plots (code, tests, a layout check on 3_ND; not run on a model)

The user (2026-10-31): the field plots are "useless" and "don't belong to the cosmx run"; "a plot needs to show a single slide and its observations (or just the vector
field), not blank space. I want attention flow plots for cosmx."
- **Removed from the CosMx run:** the per-head fields npz and `plot_fields` (the CosMx hook now calls `hierarchy_tables(export_graphs=0, flow=True)`; `EXPORT_GRAPHS` left
  `cosmx_run.py`). synth_spot's fields and `fields_by_class` are untouched.
- **Built:** `evaluation/flow_maps.py`, `hierarchy_readouts(flow_dir=...)` (the `hier_flow` table), the CosMx hook and `cosmx_run.py` in `slurm/new/`. The net flow
  `F_ij = A_ij - A_ji` of the last layer's head-mean attention over cells (the manuscript's `M - M^T`), from the routed rows as a sparse matrix, mutual pairs only; each
  positive pair an arrow source -> receiver deposited at its midpoint, smoothed (30 µm) and divided by the cell count; streams per tissue block, one figure per test section
  (`flow/<run>/<section>.png`, and `.npz` to redraw), one tile per block at one scale, cells grey and the hidden ones tinted, nothing but tissue. No label, no gradient.
- **Checked:** 19 tests against hand-worked values (a planted "every cell reads its left neighbour more" gives +0.4 along +x; the deposit conserves the flow and sits at the midpoint;
  dense attention and routing over every parent give one flow; the figure has one axes per block, the tint on the hidden cells only, the packed tiles inside the figure), 24 of 26
  mutations caught (the two left are equivalent: a safety margin of the grid, and zero-flow pairs); 1,528 tests pass. 100k cells with 290 keys each: flow 5 s, figure 4 s, 4 GB RAM.
- **Layout on the real geometry** (the old 3_ND export, whose only attention is the top-5 keys per cell, a stand-in: the figure is not the result): 10 tiles for the 10 FOVs, each filled
  with tissue. The first version found 8 blocks: two pairs of FOVs touch at a corner and form an L, half of whose tile was empty slide; non-rectangular pieces are now cut into rectangles.
- **Not validated:** raw attention did not find planted senders (2026-10-21); the planted-pair check of the brief was not run on this form. Part of any net flow is geometry
  (density, the tissue's edge, the mask's layout: a hidden cell reads the visible ones around it, which read it little). The coarse levels and the first layer are not in it; routing
  truncates rows, which is why only mutual pairs count (`mutual_share` is 0.74 on the top-5 stand-in; read the real one in `hier_flow`).
- **To read:** the trained figure next to the twin's (`hier_none_f0_untrained`, same code); `hier_flow`: `to_hidden` against `hidden_cells` (its chance value), `visible_to_hidden`, `length_um`.
- **To get it:** the fork's main with `flow_maps.py` and the `flow_dir` option on the cluster, `hier_common.py` and `cosmx_run.py` from `slurm/new/`, then the runs redone from their
  checkpoints: `INTERSCALE_MODE=evaluate` with the same sbatch (no retraining).

## 2026-10-31 — Which readout finds the planted senders in trained windowed models: attention per head and head mean, integrated gradients, and the direction of the net flow (Stage B exports of synth_spot, model-free reading of arrays; 22 slides)

The user asked for alternatives to raw attention flow, to see which cell influences which and ideally sinks and sources. Read from `results/27_10_results/hier_synth_spot_B/fields` (windowed hierarchical models at 300 epochs, priors
from lambda 0; 11 trained runs, 3 untrained twins of seed 42): for the receivers whose integrated gradients were exported (752 per slide, by programme), the share whose top 5 holds one of their planted senders
(attention: the last layer's top-5 targets, per head and head mean; IG: the first 5 of the 32 cells with the largest |contribution|, `dep_top_cells`); chance is hypergeometric for 5 random cells. The three arms with a twin, seed 42:

```text
class (receivers)   chance   twin head mean   trained head mean   best head   trained IG
short (600)         0.002    0.69             0.97                1.00 (h0)   0.99
mid, M11 (150)      0.002    0.07             0.73                0.80 (h1)   1.00
dense (450)         0.035    0.01             0.01                0.02        0.09
sparse (756)        0.034    0.03             0.04                0.04        0.04
directional (300)   0.007    0.03             0.04                0.04        0.01
net flow F_rs = A_rs - A_sr (top-5 matrix), share of planted pairs with F > 0:  short 0.61 (twin 0.49), mid 0.94 (twin 0.42); a random cell at the same distance: 0.30 / 0.37
```
Over all 11 runs the numbers are lower where the prior costs (short 0.88 head mean, 0.97 IG; mid 0.45 and 0.93; direction short 0.47).

**What worked**
- Trained attention is a readout of who influences whom for the short and mid range: the head mean's top 5 holds a planted sender for 97% and 73% of the receivers, and the net flow `A - A^T` points from the sender to the receiver (61% and 94%
  of the pairs, against 49% and 42% for the untrained twin). The head mean costs little against the best head (97% against 100%, 73% against 80%): it does not wash the signal out.
- Integrated gradients are the stronger readout where the two differ: mid range 100% against 73% (the best head 80%), the dense class 9% (3x chance) against 1%. They need no attention map, are gene-resolved and cover the whole model.

**What doesn't work**
- The short range is mostly the distance window: the untrained twin's head mean already holds a planted sender for 69% of the receivers (316x chance); training adds 28 points. Only the mid range (twin 7%) shows the model reading something its prior does not.
- Nothing finds the dense, sparse or directional senders (attention at or below chance, IG 3x chance on dense only). The directional programmes are learned by no model, so that may be the model's blind spot and not the readout's.
- Net flow at short range is no direction signal by itself in an untrained model (0.49): neighbours read each other.
- Limits: the matrix is the top 5 per cell (truncated), the chance is over all cells (generous), the arms are priors from lambda 0 and not `hier_none`, one lattice, one seed for the twins; IG is only on the sampled receivers.

**For the goal (which cell influences which, sinks and sources in the tissue).** Both readouts are partial and complementary: attention flow is cheap, covers every cell of a whole section and gives the net direction; IG is the confirmation, on sampled receivers or
summed targets of a disc (it is linear in its target). A source is a cell that is read or drawn from by many (out-strength), a sink one that draws from many (in-strength); the net `out - in` per cell, or the divergence of the smoothed current, maps them. Since hidden cells
are the receivers by construction, sinks follow the mask layout: read the map against the untrained twin, and for a CosMx claim average over many masks (fold passes) and compare with the expression-permuted slide.
Plots (not built): a source-sink map (cells coloured by smoothed `out - in`, diverging), zoom panels on discs with the arrows from the top sources to sampled receivers, the streams with the divergence shaded, and the influence against distance for trained, twin and permuted.

## 2026-10-31 — Sources, sinks and who influences whom: three views on the net attention flow (code, tests, a layout check on 3_ND; not run on a model)

The user chose views 1-3 of the plan above. `flow_arrays` now also stores each cell's net flow out and in (`out - in` is the divergence of the net flow, summing to 0 over a slide) and the
top-5 sources of up to 2,000 sampled receivers (hidden cells first, seeded). Per section the CosMx hook draws `<section>.png` (the streams, with the smoothed `out - in` shaded behind: red sources,
blue sinks), `<section>_source_sink.png` (cells coloured by it, the strongest 1% ringed, hidden cells dotted) and `<section>_who_whom.png` (260 µm zoom panels on the 8 largest hidden patches,
arrows from each sampled receiver's top sources, coloured by the pair's net flow).

**What worked**
- 26 tests (a planted star, where every cell within 50 µm reads the centre, gives the centre `out - in` = 0.4 per reader and each reader -0.4; the smoothed value keeps the signs; bins of
  several cells average them; the sample takes hidden receivers and their strongest sources first and is seeded; a panel draws only the receivers in its view); all 11 mutations caught.
- On the real 3_ND geometry (top-5 stand-in, not the result) all three draw, 10 tiles and 8 panels, nothing but tissue.

**What doesn't work (yet)**
- On the stand-in the hidden discs are the sinks, as the mask makes them; the strongest sources and sinks also sit on tissue edges. Whether anything beyond the mask and the edges shows can only be
  read on the real rows against the untrained twin. View 4 (influence against distance, with the permuted slide) and IG on discs are not built.

## 2026-10-31 — The in- and outflow views made unbiased (the user: "You don't need to bias the plots to show sources or sinks, I just hope to find any in an unbiased fashion")

What selected by content before, and what replaced it: the source-sink map ringed the 1% strongest cells (a fixed share of "hubs" found even in noise) -> nothing ringed, every cell coloured on a
scale symmetric about 0, neutral wording ("is read more" / "reads more"); the who -> whom panels sat on the largest hidden patches (sinks by the mask's construction) -> seeded random places, non-overlapping,
at least 85% tissue by position alone; the receivers were drawn from hidden cells first -> uniformly from all cells; the streams carried the source-sink shading by default -> off unless asked.
27 tests (uniform receiver draws, no ring or pick in the map, panels placed by the seed and never overlapping, views mostly tissue, no shading by default), all mutations caught. Not run on a model.

## 2026-10-31 — Tier 1 for the real-data comparison: visible keys on fused / hierarchy, the context query on fused, the section offset (code and tests; not run on data)

Built on branch `tier1-real-data` (roadmap.md, "Tier 1"): `attention.visible_keys_only` for the fused attention (`StructuredMask.hidden_keys`)
and every level of the hierarchy; the context query on the fused attention (`FusedAttention.supports_cross`); `model.decoder.section_offset:
visible_mean` for GlobalModel and LocalModel, and `masked_probing.within_section` for scores within sections. Toy models only, CPU.

**What worked**
- Fused + visible keys equals dense + visible keys through the real GlobalModel (cell masking, distance window, centred q/k), predictions and every
  gradient to 1e-5, in one block, in blocks of 4 queries and one query per block on a subsampled slide.
- The fused context query equals the dense one (node encodings, Fourier, the window, RoPE, centred q/k; one block, blocks, subsampled): embeddings,
  maps and gradients to 1e-5; a cell stays blind to its own expression bit for bit, also behind a GCN under the long-range mask.
- Section offset: the part is exactly w x (the leave-one-out visible mean of the cell's section), hidden values never reach a prediction, a visible
  cell moves every cell of its own section and none of the other; w trains from 1; a gene hidden in every cell gets 0; inference reads the clean batch.
- Mutation checks: 9 / 9 (visible keys), 6 / 6 (context query), 18 / 18 (section offset; 3 survived first and led to a gene-masking test, a subsampled
  test and the removal of a redundant `where`). Harness IDENTICAL after each commit.

**What doesn't work / found on the way**
- **Under cell masking the singleton hierarchy was never the dense model.** The must-pass regression (hierarchical_attention_plan.md, test 2) runs
  under programme masking only, where no cell is hidden whole. With cells hidden whole the top level hands CLS the learned `pool_fill` of every hidden
  singleton where the dense model hands it the hidden cell's own token: predictions 2-4e-2 apart (toy model). With visible keys on both, 4e-7. In the
  CosMx `hier_none` runs (clusters of 25 and 250, 60 µm discs) the same applies only to a coarse cluster whose cells are all hidden, which is rare.
- **`attention.center_qk` leaks hidden tokens past `visible_keys_only`.** Queries and keys are centred on the mean of every cell token, hidden ones
  included (weight 1 / S), so moving a hidden token moves the visible cells' predictions even with every hidden key blocked (dense, fused and
  hierarchy alike). Small at 10^5 cells, but the option is not exactly "no hidden key" with centring on; a centring mean over visible cells only would
  close it (not built).
- With centring on, the singleton hierarchy and the dense model still differ by 1.8e-4 under visible keys (the top level centres on a mean holding
  `pool_fill`).
- Within sections the slide mean is the centre: its cosine is float32 rounding noise around 0, not exactly 0 or NaN.
- Nothing run on CosMx yet; `within_section` is not in `sweep_common.py` (cluster scripts) yet.

## 2026-10-31 — Tier 2: a bias for a set of heads, window scales from the data, distances in local units, the dimension readout (code and tests; not run on data)

Built on branch `tier1-real-data` (roadmap.md, "Tier 2"): `pe.bias_heads.<kind>` (attention_plan.md stage 10a) with `pe.distance.scales_um`
and `gene_scales.window_scales`; `pe.distance.units: local` (density-scaled distances); `evaluation.dimensions` (task 2, ED0-ED2), wired into
`slurm/new/hier_common.py` (`dim_*` tables). Toy models only, CPU.

**What worked**
- A head a bias does not serve gets exactly 0 from it in the dense, pair, row and factor forms and in the hierarchy's `offset_bias`; the served
  heads get what a bias built for them alone gives; with a free head the fused model equals the dense one (one block and blocks).
- Local units: the dense bias is the encoder at each query row's scaled distances (asymmetric, as meant); pairs and rows equal dense; at the
  reference density the bias is the µm one; fused = dense and the singleton hierarchy (dense and routed) = dense under local units.
- The dimension readout: decoded, the collected rows are the step's prediction (GlobalModel, LocalModel); on a planted readout the dimension
  that writes the signal tops all three importance readings, writes and tracks its genes, and lifts above an untrained twin; every reading is
  unchanged by a shift of `z` the bias takes back, and the singular directions by a rotation of the dimensions.
- Mutation checks: 16 / 16 (bias heads; 2 after stronger tests), 13 / 13 (local units; 4 after: a 0.5 µm unit, jittered lattices, a small
  readout block, a `level_radii` helper tested on its own), 11 / 11 (dimensions; 4 after: shifted `z`, a negative weight, the twin's mean).
  Full suite 1,623 passed; harness IDENTICAL after each commit.

**What doesn't work / found on the way**
- **On a regular lattice local units do almost nothing.** synth_spot's kNN radii are one value for nearly every spot (only edges, corners and
  holes differ), so `units: local` there is the µm model up to the border. It is meant for CosMx and the lung data; a synth_spot read of it says
  nothing about uneven tissue -- the same lattice symmetry first hid a misaligned-radius bug in the tests.
- **`W diag(std z)` is not rotation-free** when the embedding's dimensions are correlated, as task 2's design said it was; the readout uses
  `W Σ_z^{1/2}` (equal to it for uncorrelated dimensions).
- The window table of `hier_common.py` numbered heads by position: with `bias_heads` that names the wrong head; it now reads `served_heads`.
- Not built: ED3 (spatial variability against matched gene sets), ED4 (cell type, planted truth), the PCA control of ED2, an elbow plot,
  stage 10b; nothing run on data.

## 2026-10-31 — Tier 3: training on crops, an evaluation path without dense maps, and dropout at inference (code and tests; not run on data)

Built on branch `tier1-real-data` (roadmap.md, "Tier 3"): `dataset.crop` (`tl/crops.py`, `CropDataset`, `crop_kwargs`);
`reference_predictions(method="sparse" | "auto")`; `TransformerNodeEncoderHook.encode` and `get_model_output(attention=False)`; a fix to
`get_model_output` (eval mode). `slurm/new/hier_common.py` now takes the library's references. Toy models only, CPU.

**What worked**
- Crops: a window keeps exactly the cells inside it; every per-cell attribute, the masks of the whole section included, is sliced and the edges
  among the kept cells kept; new crops every read, from a private generator or the worker's per-epoch seed, repeated by the seed, the global
  generators untouched; a GlobalModel trains end to end on them (3 steps an epoch from one training section instead of 1).
- Sparse references equal the dense ones to rounding (1e-14) on random positions with cells hidden whole and in part, a gene hidden everywhere,
  far-away cells, labels, two graphs, chunks of 7 pairs; no distance matrix is formed.
- `encode` gives `evaluate`'s clean pass (no hidden key blocked, no map kept); fused = dense through it; a routed hierarchy encodes 400 cells
  without a tensor over the cells' pairs; `get_model_output(attention=False)` writes the same embeddings and predictions as with the maps.
- Mutation checks: 12 / 12 (crops; 2 after a test with lone cells and one cluster), 9 / 10 (sparse references; the survivor is an equivalent
  chunk-size mutation; 1 after counting `cdist` calls), 4 / 4 (encode; 2 after a masked graph under visible keys and a kept-map check).
  Full suite 1,655 passed; harness IDENTICAL after each commit.

**What doesn't work / found on the way**
- **`get_model_output` predicted with dropout on.** No model switched its module to eval mode for it, and Lightning leaves it in training mode
  after `fit`: every read of a trained model gave another output (the test fails on all three models without the fix). Every `.h5ad` written by
  `get_model_output` before 2026-10-31 -- gene ranks, loadings, `y_pred_*` -- carried dropout noise. The masked-pass evaluations
  (`collect_head_predictions`, the probes, the sweeps' tables) set eval mode themselves and are not affected.
- **A kernel cut relative to the query's largest weight is not enough to match the dense form.** The dense path falls back to the slide mean
  when a gene's denominator is below an *absolute* 1e-12; at a cut of exp(-12) a gene whose only visible cells lay many bandwidths away gave
  up to 1.2 apart (a 15 µm kernel, far-away cells). exp(-40) keeps every weight that could move that floor: rounding only, at ~3x the pairs.
- **Lightning creates the epoch's iterator before `on_train_epoch_start`**, so with a worker (or the fetcher's prefetch) the first batch of an
  epoch can be built before `NodeMaskResampleCallback` redraws the masks. Crops avoid it by drawing in `__getitem__`; the mask redraw itself is
  unchanged (pre-existing, not measured).
- Crops change what is taken per sequence (`center_qk`, centred node encodings, grid patches, CLS, the hierarchy): train-on-crops and read-on-
  sections is a shift the validator warns about, not something tested; a no-crop arm is the control.
- Not built: IG on sampled receivers for CosMx (the readouts that need it already sample queries); `visibility_common.py` still has its own
  k-NN references; nothing run on data.

## 2026-11-01 — CosMx `cosmx_run`, the missing fold: `hier_none_f1` (results in `results/30_10_results/cosmx_run`; first CosMx flow plots)

- **Scores (centred cosine on hidden cells, test slide of fold 1):** hier_none_f1 0.233, flat over depth (0.242 at 0-12 µm, 0.224 beyond 40 µm);
  GCN_f1 0.199, falling with depth (0.240 to 0.158); ref_section_mean 0.224, ref_knn32 0.225, ref_knn128 0.252. Fold 1 is the best fold for
  both models, and the only one where hier_none beats the section mean (+0.009) and knn32 (+0.008). It stays below knn128 (-0.019).
- **All three folds, hier_none minus the section mean:** -0.052 (f0), +0.009 (f1), -0.003 (f2). Minus knn128: -0.087, -0.019, -0.049. Beyond
  40 µm (the GCN goes blind) hier_none beats the GCN on every fold: 0.134 / 0.224 / 0.150 against -0.113 / 0.158 / 0.004. The
  conclusion from folds 0 and 2 holds: it beats the GCN at depth but sits at about the section mean, below the k-NN smoothers. The
  spread over folds (0.157 to 0.233) is larger than any difference between the arms.
- **Raw SCE is worse than every reference on every fold.** hier_none_f1 0.672 against 0.46 for the references (f0 0.493 against 0.41;
  f2 0.589 against 0.44). The GCN shows the same. The centred score is close to the references while the raw score is far off, so the
  models' mean profile (or scale) per section is off. This is what `section_offset: visible_mean` addresses (cosmx_tiers: gcn ->
  gcn_sec, hier_crop -> hier_tier).
- **Training:** 4.8 s an epoch, 0.49 steps/s, so about 2 steps an epoch and about 600 steps in total. val_loss 0.733 (the lowest of the three).
  The model is still undertrained, as on folds 0 and 2.
- **Attention (hier_profile):** strongly local. 0-15 µm gets 16% of the weight (7.8x uniform), 15-30 µm 2.2x, and beyond 30 µm less
  than uniform. The windows sit at 11 / 49 / 231 / 953 µm with strengths 1.3 to 1.05: the short heads are the strongest, and the windows
  barely moved from their initial spread. Coarse routing is sharp (coarse entropy 0.29 / 0.33) and never leaves a cluster unselected;
  110 / 280 keys per query at levels 1 / 2.
- **Flow plots (2_ND, 2_T1D; the first from `flow_maps.py`):** the net flow is organised by the masking discs, not the tissue. The
  strongest streams and the deepest sources and sinks sit on the hidden discs. Many discs are sinks (they read from their rim), as
  expected, but **several hidden discs are net sources**: the rest of the tissue reads hidden cells more than they read it, although a
  hidden cell's token holds only the mask fill and its position. Hidden cells are keys here (no `visible_keys_only`), so this is the leak
  that hier_tier closes. Away from the discs the flow is weak and has no visible tissue structure. Who-whom: the top-5 sources sit mostly
  within ~20-50 µm, plus a few long, faint links of ~200 µm. No untrained twin exists for fold 1, so these plots cannot be read against
  one yet (cosmx_tiers draws them for every run and for hier_none_f0's twin).
- **For cosmx_tiers:** (1) read per fold, because fold 1 alone would look like a win; (2) read raw SCE next to the within-section score,
  because the offset fix should show in raw SCE first; (3) compare the flow plots of hier_tier against hier_crop: hidden discs should stop
  being sources.

## 2026-11-01 — CosMx `cosmx_tiers`: Tier 1-3 on three folds (results in `results/31_10_results/cosmx_tiers`)

18 of 21 runs came back: gcn, gcn_sec, hier_none, hier_crop, hier_tier, hier_tier2 on folds 0-2, plus the twins of hier_none_f0 and
hier_tier2_f0. **dual_ctx (all three folds) is missing**: no checkpoint, no section, no twin (cluster logs not read).

- **Within sections (the score that counts on CosMx), centred cosine on hidden cells, f0 / f1 / f2:** gcn 0.052 / 0.060 / 0.072;
  gcn_sec 0.055 / 0.060 / 0.071; hier_none 0.089 / 0.103 / 0.122; hier_crop 0.085 / 0.097 / 0.120; hier_tier 0.087 / 0.096 / 0.117;
  hier_tier2 0.089 / 0.096 / 0.118. References: knn8 0.084 / 0.076 / 0.089, knn32 0.111 / 0.104 / 0.118, knn128 0.130 / 0.124 / 0.136,
  section mean ~0.005. Every hierarchical arm beats knn8 on every fold and is at about knn32 (below on f0, level on f1 / f2); none reaches
  knn128 at any depth (closest: f2 at 0-12 µm, 0.145 against 0.148). The GCN is below knn8 at every depth and falls to 0.02 beyond 40 µm.
  **The Tier 1-3 options do not move the within-section score** (all arms within ±0.006 of hier_none per fold).
- **The section offset does what it was built for and nothing else.** On the training-mean centred score gcn -> gcn_sec jumps 0.053 ->
  0.218 (f0), 0.100 -> 0.196 (f2), and is flat over depth; hier_none -> hier_tier 0.156 -> 0.213 (f0), 0.174 -> 0.200 (f2); hier_tier_f1
  0.257 is above knn128 (0.252), the only arm and fold that is. Within sections nothing changes. The twin of hier_tier2 scores 0.0000
  within sections (the offset alone is zero there, as it should be).
- **Raw SCE stays worse than every reference with the offset on** (hier_tier f0 / f1 / f2 0.481 / 0.800 / 0.644 against the section mean
  0.422; gcn_sec is the exception at 0.42-0.47). The 2026-11-01 entry above expected the offset to fix raw SCE: it does not, so the raw gap is
  not (only) the section offset. Centred cosine high and raw cosine low with negative val R² (-0.5 to -3.4; crops worst) points to the
  prediction's scale along the mean profile, not its direction. Not measured: the ratio of the prediction's norm to the target's. hier_crop
  has the worst raw SCE and val SCE (f1 0.944 / 0.821) but a better centred test score than hier_none, so val SCE and the centred test score
  disagree: check which metric picks the checkpoint.
- **Visible cells: Tier 1 makes the model usable on them.** Visible minus hidden centred cosine (f0): hier_none -0.183 (visible cells at
  -0.0002, i.e. useless; raw SCE 1.13), hier_crop -0.039, hier_tier -0.019, hier_tier2 -0.008 (f2 -0.002); gcn -0.094, gcn_sec -0.015.
  Matters for any readout on all cells (CCC maps, comparisons).
- **Crops: 8x the steps, 6x less memory, no gain.** hier_crop runs ~4,800 steps (16 an epoch) against ~600 on whole sections at the same
  epoch time, with peak memory 1.1-1.7 GiB against 6.5-10.5 GiB; the within-section score is the same or 0.005 lower. So "undertrained"
  (cosmx_run read) is unlikely to be the main reason for the gap to knn128: more steps did not buy it. Crops are the route for the lung
  (memory), not for accuracy.
- **Tier 2 (window on heads 0-2, local units) adds nothing over Tier 1** (within ±0.002 per fold). Windows after training (hier_tier2_f0):
  7.7 / 45 / 689 with strengths 1.6 / 1.9 / 1.4; Tier 1 moved the windows further than hier_none (hier_tier 7.4 / 34 / 193 / 629 against
  hier_none 11 / 50 / 235 / 959) and sharpened the profile a little (0-15 µm lift 7.8 against 6.2; twin 4.9).
- **Negative probes:** the 19 `NegPrb*` channels are in the 979 reconstructed features. hier_tier2 reconstructs them within sections at
  0.041-0.049 (knn128 0.030-0.040; twin 0.002), about 70% of its mean over real genes (0.062-0.066); hier_none 0.027, gcn 0.012. Ten
  NegPrb channels are among the genes where hier_tier2 beats knn128 on all three folds. So a large part of the per-gene gain is technical
  (FOV background, counts per cell), and per-gene "wins over knn128" are not evidence of cell-cell signal without a NegPrb floor. Real
  genes with the clearest gains over knn128 (f2): REG1A, PRSS2, ANXA4, TTR, MGP (acinar / islet markers, i.e. cell type composition).
- **Flow plots:** with visible keys the hidden discs are no longer sources (to_hidden 0 by construction); trained hier_tier2_f0 shows
  compact sinks ~50-100 µm across with flow about twice the twin's scale (±0.075 against ±0.04); whether they are islets is not checked
  (no cell labels in the plot). Who-whom arrows run 50-150 µm. Hidden-disc sources in hier_none: to_hidden 0.17 against the twin's 0.05.
- **Dimension readout (hier_tier2_f0):** importance spread over many dims (top dim 5.7% of the variance, single-dim ablations < 0.003),
  tracks stronger than the twin's; no dim with a clear role.
- **Next, proposed:** (1) drop the NegPrb channels from inputs and targets, or keep them only as a technical control score next to every
  per-gene table; (2) measure the prediction / target norm ratio to explain raw SCE, and check the checkpoint metric; (3) find out why
  dual_ctx did not run; (4) the gap to knn128 within sections is now the question: larger discs (60 µm do not separate a global model from
  a wider smoother) and a model-plus-knn128 blend to see whether the model holds anything knn128 lacks.

## 2026-11-01 — CosMx `cosmx_tiers`: the decoder readout (dim_directions / dim_direction_genes / dim_tracks)

- **The decoder writes into few directions, the same on every fold and arm.** SVD of `W Σ_z^{1/2}` on hidden cells: 5-9 directions carry 99%
  for every trained hier arm (gcn 10, gcn_sec 19; untrained twins 24-30 with random genes on top, NegPrb among them). The recurring axes:
  (A) acinar against endocrine: CPB1 / SPINK1 / REG1A / PRSS2 / CUZD1 against INS / GCG / TTR / CHGA (IAPP, SST, SCG5 in the GCN);
  (B) a one-signed axis of abundant genes: RPL21, RPL34, AZU1, NEAT1, MZT2A, MT1X, HSP90AB1, B2M, HLA-A / -C (every loading the same sign, so
  a size / quality factor, probably technical); (C) stress / ductal: SERPINA3, HSPB1, SOD2, OLFM4, CXCL2, HSPA1A; ANXA4, SOX9, KRT8, FGFR3,
  CLU; stromal (IGFBP7, VIM, COL6A2, DCN, TAGLN) further down. Tier 1 (section offset) makes A the first direction (0.40) instead of B.
- **Gene priority (hier_tier2, squared direction loadings over the listed top-10 genes, mean over folds):** CPB1, INS, TTR, GCG, SPINK1,
  CUZD1, AZU1, SERPINA3, PRSS2, REG1A, RPL21, RPL34, HLA-A, XBP1, TYK2 -- the first 10 on all three folds. The top-10 lists cover only
  35-43% of the squared total, and loadings are in expression units, so abundant markers dominate; a full per-gene ranking (absolute and
  per-gene standardised `||(W Σ_z^{1/2})[g]||`) is computed inside `effective_directions` but not exported.
- **Tracks:** trained dimensions track true expression of hidden cells about twice as well as the twin's (mean top-5 |rho| 0.31 against 0.16;
  hier_none 0.32 against 0.15); the tracked genes are the same markers (CPB1 / PRSS2 / REG1A -0.54 / -0.54 / -0.40 on dim 8).
- **Reading:** the prediction varies along tissue composition (acinar / islet / duct / stroma) plus one abundance axis. This is what a
  neighbourhood smoother would also produce; not checked: the same SVD on the knn128 prediction. Without that, the readout does not show
  anything only the model holds.

## 2026-11-02 — Label residuals before the Moran's I scale sets (yardsticks_plan.md Y0; results in `results/02_11_results/label_residuals`)

Run locally by mistake (whole datasets belong on the cluster; `slurm/new/label_residuals.sbatch` written for the full run). synth_spot (24
training slides) and CosMx fold 0 (training slide, 2 sections, 108,711 cells x 979 genes) complete; vannan25 only 8 samples balanced over
`sample_type` (430,261 cells) and only the `final_CT` stage (the run was stopped). k-NN 8, 2 hops, `fit: graph`, n = 20 per set.

**What worked**
- synth_spot, label `region` (positive control): the region markers CM1-CM3 lose their pattern (I_local 0.95 -> 0.001, label R² 0.98) and
  leave the global set; every planted programme keeps 95-100% of its local Moran's I; the Spearman of the fall with the planted scale is
  unchanged (-0.79 -> -0.78). M4, M7, M8 enter the freed global slots.
- vannan25, `final_CT` (8 samples): the global set before is airway identity (C20orf85, FOXJ1, MUC5B, SCGB1A1, WFDC2, SOX2); 16 of 20
  leave it once the cell type is out (kept I_global ~0.02-0.2). The global set after is ECM / fibrosis (COL1A2, COL3A1, POSTN, FN1, SFRP2),
  inflammation (CCL2, CCL18, CXCL13, ICAM1, SOD2, ATF3) and alveolar (SFTPC, NAPSA): tissue-scale variation within cell types. Local set
  before = markers of scattered types (mast TPSAB1 / CPA3, endothelial ACKR1 / PLVAP), 15 of 20 leave.
- CosMx, `slide_fov` as a region label: the global set (HSPA1A/B, HSP90s, HSPB1, BAG3, FOS, JUN, DUSP1/5, RPLs) loses 85% of its I_global
  (median kept 0.15), 15 of 20 leave; the cell type hardly touches them (kept I_global 0.99). A stress / immediate-early + ribosomal
  signature constant within ~1 x 0.66 mm fields: most likely handling or per-FOV technical, not tissue biology (the statistic alone cannot
  tell; an FOV residual also removes real mm-scale gradients).

**What doesn't work / caveats**
- CosMx, `cell_type_coarse`: the genes that enter the local set after the cell type is out (GCG, TTR, CHGA, CPB1, SST; R² 0.4-0.7) are
  **segmentation spillover**: non-endocrine cells with 0 / 1 / 2 / 3 / 4+ endocrine cells among their 8 nearest carry GCG at 0.02 / 0.05 /
  0.10 / 0.20 / 0.56 (alpha cells 2.77) and INS at 0.21 / 0.50 / 0.76 / 0.97 / 1.41 (log-normalised X, fold 0 training slide). A GCN
  predicts them from its neighbours trivially; they are not communication. The Shapley split shows it too (GCG own 0.38, near composition 0.32).
- CosMx label R² is small for most genes (median 0.004); the 110 significant candidates shrink to 65 after the cell type. 19 NegPrb: none in a
  set before or after.
- Composition explains interaction too, so the sets are rebuilt on own-label (or region) residuals only; composition stays a variance share.
- vannan25 TNiche / CNiche stages and the Shapley split are missing (run stopped); CosMx folds 1-2 not run.

## 2026-11-03 — Gene rankings on synth_spot: sender deletion, seed stability, remove and retrain (results in `results/03_11_results`)

`full` x seeds 42-44 + twin, 300 epochs, programme masking; ROAR arms at seed 42 only.

**What worked**
- Consensus context top 10 is 10/10 planted (S1-S4, M7, M8, M11, M19-M21). Removing them: centred cosine 0.946 -> 0.289; all 25 planted removed 0.280; bottom10 0.953; rand10 0.946 / 0.928; twin -0.006.
- Context edges above ΔR² 0.05: 14 of 14 planted. Precision@1 vs twin: `all` 0.62 vs 0.10, `context` 0.49 vs 0.07.
- Seeds: nine genes in every seed's context top 10 (Jaccard@10 0.82-1.0; twin RBO 0.02). Stage 1 per-gene cosine SD mostly <= 0.01.
- Removals agree with `all` ΔR² per target: MR22 -> MC22 0.84 -> 0.03, MR25 -> MC25 0.98 -> 0.10, M1 -> MC1 0.83 -> 0.37, M3 -> MC3 0.74 -> 0.31, M10 -> MC13 no change.

**What doesn't work**
- Conditional deletion understates smooth senders: the other genes rebuild them within a slide (corr with M7 0.63-0.81 on 3 test slides; S1, M11, noise 0.26-0.40). M7 ΔR² <= 0.25, yet cosine ~0.96 -> ~0.2 on removal. Importance is not comparable across sender classes. Cause inferred; `METHOD: permute` on the checkpoints is the test.
- `context` misses own-row reading: M1-M6 ΔR² ~0 there, 0.01-0.48 in `all`. Only 14 of 31 planted edges pass 0.05.
- Top 10 removal leaves MC1-MC6 at 0.52-0.65 (planted arm 0.12-0.28); the headline 0.289 vs 0.280 hides it.
- Collateral loss: top10 kept M12-M18 yet MC16-MC22 fell 0.04-0.14, MC1 0.31; train / val loss 0.20 / 0.60 vs 0.04 / 0.05. ROAR drops are not all lost information.
- rand arms are not neutral: they hit M1, M3 and receptors.
- MC16-MC22 cosine is mostly receptor on/off (sources removed 0.52-0.92), so it cannot show source reading. MC12, MC13 stay ~0.25 in every arm.
- V2 skipped (no ctx_ tables). V1 unique_global vs planted length: Spearman 0.12.

## 2026-11-04 — Gene rankings on synth_spot, follow-up: deletion by permutation, one class removed per arm, scores within receivers (results in `results/03_11_results/synth_ranking_permute`, `synth_followup`)

Stage 1's checkpoints re-evaluated (seeds 42-44 + twin); four retraining arms x seeds 42-43, 300 epochs. Check: `full_s42` reproduces 0.945.

**What worked**
- The diagnosis of 2026-11-03 holds: the conditional replacement hid smooth senders. Context ΔR², conditional -> permute: M7 on MC8 / MC9 0.10 / 0.21 -> 1.12 / 1.33; M8 on MC10 / MC11 0.03 / 0.14 -> 1.43 / 1.65; M19-M21 0.07-0.13 -> 0.27-0.57. Spot-like senders hardly move (S, M11).
- `context` + permute is the usable ranking: the same 11 genes lead in all three seeds, all planted (S1-S4, M7, M8, M11, M19-M21, M3); next gene <= 0.05, noise max 0.006. 17 of 31 planted edges above ΔR² 0.05. Seven off-target edges, all from M7 / M8 (largest M7 -> MC24 0.16).
- Own-row reading confirmed: M1, M2, M4, M5, M6 context ΔR² <= 0.003, `all` 0.45-1.51. `own_row` arm: MC1-MC6 0.67-0.83 -> -0.05-0.39, everything else unchanged.
- The sparse sources M12-M18 are not read: ΔR² <= 0.002 under both methods in both regions; `sparse_sources` arm leaves MC16-MC22 within 0.001 of `full`; receiver_corr -0.25 to 0.18 (full -0.23 to 0.21, twin -0.15 to 0.09). Their high cosine is receptor on/off: `receptors` arm takes MC16-MC25 to 0.02-0.23.
- Readings follow scores between seeds (3 seeds only): M3 context 0.05 / 0.47 / 0.13 with MC3 0.66 / 0.81 / 0.75; M20 0.42 / 0.31 / 0.10 with MC24 receiver_corr 0.60 / 0.60 / 0.37.
- `receiver_scores`: MC23-MC25 0.37-0.91 in full against twin ~0, so the dense programmes' level is followed.

**What doesn't work**
- Permutation in `all` gives false positives on collinear genes: CM1 ranks first (5.03; MC4 1.88, MC2 1.05, MC6 0.98), though bottom10 removed CM1 and CM2 with no loss and CM1's context sum is 0.015. Off-manifold: CM1 changed while CM2, CM3 stay. `all` precision@1 falls 0.62 -> 0.52. Neither method is right for `all`; a conditional pooled over slides is the proposed fix (not built).
- Headline cosine misses whole classes: `own_row` 0.946 vs full 0.947, though six programmes collapse. Read per programme.
- Removing rings damages programmes whose senders stayed: MC14 / MC15 ~0.90 -> 0.07-0.34 (M11 kept), MC23-MC25 receiver_corr -> 0.24-0.39; train / val loss 0.055 / 0.387. No other arm does this, so the top10 collateral of 2026-11-03 probably comes from here. Cause open: overfitting on targets that became unpredictable, or the rings teaching long-range reading.
- Tried and refuted: the hidden response leaking through library-size normalisation (raw counts, 8 test slides: within receivers its correlation with receptor, background mean and log library is ~0, |r| <= 0.11).
- Open: `receptors` arm raises receiver_corr of MC18, MC19, MC21, MC22 to 0.33-0.47. Level depends on distance to the source, so this hints the source is read once the receptor is gone; not tested. M19-M21 still have no retraining arm of their own. V2 reason still unknown.

## 2026-11-04 — Plan for the improved global model, a type-residual target, CosMx gene masking and a capacity sweep (code, tests, stub data; no run)

Review of the plan files with the user (roadmap.md, "Focus until Tuesday 10 November"): the hard date moved to 10 November; first a somewhat well working improved global-only model, then T7, T8 and T10, for which the user will
give details. What the user wants downstream: spatially variable genes among the most influential ones, better still a separation of local and global genes; expression itself matters less.

Built on branch `cosmx-programme`:
- `model.decoder.section_offset: label_mean` + `label_offset_key` (GlobalModel): the prediction gets a fixed part, the training mean of the cell's label; the loss reads prediction and target minus it. Inputs unchanged, so a sender's type stays readable. `tl._preprocessing.imitation_labels` gives clusters where no annotation exists.
- `masked_probing.within_groups`: every head and the truth on their own mean per group. It needs no visible entry, so it works for a gene hidden in every cell, where `within_section` does not. `within_type_*` tables centre per section and label.
- `FittedReferences(k_nearest=...)`: the ridge bars on neighbour lists for whole sections; `dataset.eval_mask_genes`: an arm's own evaluation set; a long `mask_genes` list is hashed in the checkpoint name.
- Sweeps in `slurm/new/`: `cosmx_masking` (8 arms, 21 runs + 2 twins) and `synth_capacity` (7 arms x 2 seeds).

**What worked**
- 14 new tests pass, the whole suite passes (1738), all 16 mutations are caught (4 only after stronger tests, among them that the label means come from the training graphs alone), the harness is IDENTICAL against `main` (5 cases / 15 epochs). On stub data: the gene-set step, a smoke training for 5 arms that cover every kind of mask and target, and the evaluation hook; the stub's `ref_own+context` lies above `ref_own` (0.27 against 0.19).

**What the existing tables already say (fold 0, label_residuals of 2026-11-02; no model)**
- Only 65 genes pass the default candidate threshold (I_local >= 0.05), 50 once the genes whose neighbours' label composition explains >= 5% are out; so the run lowers the threshold to 0.03 (107 clean genes on fold 0) and takes 30 per set.
- Removed as spillover: PRSS2, CPB1, REG1A, GCG, IAPP, CHGA, TTR. The most local clean genes: IGF2, CD74, CCL2, COL1A1, OLFM4, APOD, CXCL2, IL32, COL3A1, SOD2: cytokines, matrix, stress response.
- The most global: HSPA1A/B, BAG3, HSP90AA1, FOS, JUN, RPL22, HSPB1, DUSP5: stress, immediate-early and ribosomal, with the largest FOV shares (HSPA1A 0.22, against a median of 0.003). Possibly handling, not biology. Read `moran_global` with that in mind; it could still pass a score.

**Choices to read the run with**
- The main arms train on 1 mm crops (4,800 steps), since synth_spot's programmes needed ~7,200 steps; cosmx_tiers' "crops bring nothing" was read under discs. The space arms cannot use crops (a crop re-centres the coordinates), hence the whole-section control `rand50_whole`.
- `space_only` is scored on sections of donors never trained on, with coordinates centred per section: expect about 0 within sections; a score above that says how much of the target is place, or a bug.
- The bars use the k nearest cells' visible genes, not kernels: not the bar of the synth_spot runs.

## 2026-11-05 — Capacity sweep on synth_spot and CosMx gene masking (results of 04_11_results)

Run: `synth_capacity` (7 arms x 2 seeds, 14 of 14 done) and `cosmx_masking` (20 of 23 runs have results.md sections; see the two faults below). The checks marked "from the npz" are not in results.md: the arms' saved flow files, one T1D section per fold, 25,000 sampled cells, net flow smoothed over 30 µm, Spearman.

**Faults in the run, both in the sweep scripts of 2026-11-04 (not fixed yet)**
- **The Moran arms were evaluated on the wrong mask.** `make_cosmx_masking_yaml.py` leaves `dataset.eval_mask_genes` empty for them, and `sweep_common.eval_gene_selectors` then falls back on the script's `EVAL_MASK_GENES` (`rand_eval`) before `mask_genes`. So each Moran model, trained with 30 genes hidden, was scored with 480 hidden. Visible in the counts: moran_set=local on fold 0 has 2,728,710 hidden entries = 18 x 151,595 cells, of 30 genes. Its numbers are not the planned test: raw SCE above 1, pooled cosine -0.25 to 0.26, attribution shares and flow all under that shifted input. Only a re-evaluation can read them (`eval_mask_genes` = the arm's `mask_genes`; the ridge bars refit for it).
- **The type-residual target was not evaluated.** `rand50_type_f0-2` have a training section only. `sweep_common.eval_cfg` sets `extra_obs_keys = list(CONTEXT_LABELS)`, and this script has `CONTEXT_LABELS = []`, which removes `cell_type_coarse`; `GlobalModel` then needs it for `label_offset_attr` (`label_attribute` raises), and `one_trial` catches the error. Read from the code; the job log should show it. Evaluation only, if the checkpoints are kept.
- **`moran_local_type` is missing altogether** (no section for any fold, not even training). Cause unknown, the job log is not in the results folder. So decision 11's target has no result yet.
- Also: the CosMx summary's reference rows are fold 0's only (raw SCE own: 0.502 on f0, 0.577 on f1, 0.591 on f2), and it omits the arms without an evaluation without saying so. Read per fold from the sections.

**What worked**
- Capacity has room: centred cosine on the programme, mean of 2 seeds: w64_h4_l2 0.949 (reproduces `full`), w64_h8_l2 0.961, w64_h4_l4 0.962, e900 0.959, w256_h8_l4 0.960, at 0.91-1.29 s/epoch (e900 3x the epochs). The gain is in the dense and mid-range programmes: M20 0.70 -> 0.86 (h8, e900), M19 0.89 -> 0.93-0.95, M11 0.88 -> 0.93-0.96. Within receivers MC24 0.62 -> 0.81-0.83, MC23 0.81 -> 0.89-0.92.
- Width is the first thing that moved the sparse levels within receivers (the class `full` misses, notes of 2026-11-04), in both seeds: w64 -> w128 -> w256: MC18 0.06 / 0.47 / 0.80, MC22 0.06 / 0.34 / 0.81, MC16 0.05 / 0.16 / 0.46, MC17 0.09 / 0.27 / 0.42. Heads, layers and epochs do not (MC18 and MC22 stay <= 0.14). w256_h8_l4 keeps both: MC18 0.68, MC22 0.67, MC24 0.80, MC23 0.81. Not yet known whether that reads the source: a score, no deletion. Next: deletion of M12-M18 on it.
- CosMx rand50 is trained and clearly above its twin: within type 0.15-0.19 against -0.001, and ties the ridge from the own cell's visible genes over 3 folds (mean within type on rand_eval: model 0.166, ref_own 0.167, ref_own+context 0.168; within section 0.225, 0.230, 0.233).
- More steps help: crops (4,800 steps) over whole sections (600): within type +0.012 / +0.015, within section +0.017 / +0.028 (f0, f1).
- `space_only` behaves as predicted on held-out donors: within section 0.018 / 0.005, within type 0.011 / 0.006. The pooled 0.13 on f1 is level, not place (f0: -0.02).
- Flow: no cell is wholly hidden now, so the discs' sinks are gone by construction (`hidden_cells` 0). Against the twin the net flow is 2x larger per cell (0.29-0.37 vs 0.15), less mutual (0.72-0.77 vs 0.90), and no longer follows local density (from the npz: rho with cells within 30 µm 0.42 in the twin, -0.03 to 0.05 in rand50); the twin's tissue-edge band is much weaker.

**What doesn't work**
- Width alone costs the dense class and overfits: w256_h4_l2 0.927 (train 0.011, val 0.068), M20 0.46, M11 0.74, MC24 0.28. Raw SCE worsens with width and epochs although val_loss does not: 0.009 (base), 0.045 (e900), 0.087 (w256), 0.078 (w256_h8_l4).
- No arm moves the directional programmes (M9 0.17-0.23, M10 0.18-0.25) or the own-row ones (M2, M4, M5, M6 ~0.67-0.70), nor M16 / M17 (0.67 / 0.58); within receivers MC19 and MC21 stay <= 0.19. Directional is not a capacity problem: the distance windows carry no direction.
- 4 layers is seed-unstable on M20 (0.89 / 0.61, MC24 0.85 / 0.45). 2 seeds in all, read small gaps with that in mind.
- CosMx rand50 finds no context effect beyond own-cell genes: per gene within type (480 genes, 3 folds), the model beats ref_own by > 0.02 in every fold on 2 genes: TTR, GCG, both segmentation spillover (excluded from the Moran sets for that reason, but not from rand_eval). ref_own+context beats ref_own by > 0.02 in every fold on 5: GCG, TTR, CHGA (spillover), SERPINA3, HSPB1. Pooled, the ridge's own context gain is ~0 (-0.002 to +0.004), so the bar itself says little context is linearly readable from the 8-128 nearest cells' means. What the model reads cannot be said from these scores.
- Attribution says context supplies 50-60% of rand50's prediction, while the score equals an own-cell ridge. Shares say nothing about whether the part helps; a within-type score of the parts is missing.
- Coordinates add nothing: rand50_space against rand50_whole within type ±0.001, within section ±0.003. Expected on donors never trained on, so this does not test place within a donor.
- Pooled against the bar rand50 is below, by more than within section (0.246 vs own 0.275, context 0.284), and raw SCE well above (0.566 / 0.895 / 0.736 vs 0.502 / 0.577 / 0.591). Crop-trained models have much worse val R² on whole sections (-2.1 to -3.4 vs -0.5 to -0.6): a level or scale shift from crops to sections, unconfirmed.
- The flow hot spots are not yet stable (from the npz, smoothed net flow on the same section): rand50_whole vs rand50_space 0.57 / 0.74 (same training, same seed), rand50 vs rand50_whole 0.35 / 0.41, rand50 vs Moran arms -0.08 to 0.32 (those under the wrong mask), rand50 vs twin 0.04. No second seed on one fold, so noise and task are not separated. Not readable as biology before roadmap item 3's tests.
- The user's criterion (SVGs among the most influential genes, local / global separated) was not tested: this run has no influence ranking.

## 2026-11-05 — Net flow under expression permutation (attention_test.py, 2026-09-04) re-read while turning it into a notebook

Read against the code of three versions (the Aug-19 upstream checkout `MA/interscale`, upstream/main of 2026-10-28, the fork's main) and the CosMx file's metadata. Nothing was trained; the setup path of the new notebook was run up to `model.train` on a 12-FOV stub.

**What worked**
- The pipeline itself is sound: one model per level, `get_model_output`, `compute_hierarchical_net_flow`, heatmaps of reference / level / difference, the partial permutation curve. The permutation draws are nested over levels (one pool order, `SHUFFLE_LEVELS` 0, 0.1, 0.2, 0.5, 0.8, 1.0, seed 42).
- The same `CombinedModel` is built by all three versions for the notebook's config: 43 tensors, 323,814 parameters, identical names and shapes; no config value differs among shared keys. The fork's 184 extra config keys are switched off or inert at their defaults.
- `get_model_output` fills `_attn_matrix` identically in the older and the upstream code (the only difference is an identity gather in the local decoder).

**What doesn't work (in the Sep-4 scripts; the notebook corrects all of them)**
- `FOV_KEY = "fov"` is not unique: 24 of its 25 ids occur in several sections and 5 occur in both conditions, so `compute_hierarchical_net_flow` pooled FOVs of different slides and gave 5 ids a single condition by `first()`. `slide_fov` (71 values) is unique.
- The attention sequences follow the **first** sample key, not the last: `prepare_a2d_dataset` returns inside its loop (older checkout, upstream, fork alike; checked by counting the graphs it builds on a stub: 47 = `sliding_window_square`, not 64 = `sliding_window_v`). The script's `resolve_window_key` took the last key, so with a config listing (square, v) the flow was aggregated over windows other than those the attention is indexed by. Flows of that run are probably not the cell-type pairs they claim.
- `X` of `cosmx_pancreas.h5ad` is log-normalised (max 4.5, non-integer), `layers['counts']` holds the integer counts. `create_permuted_data.py` permuted `X` as if it were counts and normalised it a second time (`target_sum=1e4`), so the model trained on a log of a log. The manuscript's "normalised and log-transformed with scanpy, maximum normalised values of 4" matches `X`, not that layer.
- `long_range_attention: True` raises `NotImplementedError` in the older code and in upstream/main (only the fork implements the mask, with `local_mask_hops`), so the Sep-4 run must have used the default (off), although the manuscript describes the mask `1 - A`.
- With the package defaults (`lr_warmup` 20, `n_epochs` 100) and `MAX_EPOCHS = 20` the run never leaves the learning-rate warm-up, if cosmx.yaml kept them. The older loop also restores no best checkpoint (last epoch is used); newer versions restore the best epoch after the warm-up.
- `create_permuted_data.py` permutes only train and val by default: slide 3 (39% of the cells, the test split) stays intact at every level, and the flow is averaged over all cells.
- Not settled: the content of `MA/interscale/config_files/cosmx.yaml` (it uses `pct_mask_nodes`, which the current package no longer reads). The notebook's config is the package default plus radius 110 (manuscript's CosMx radii: 0, 60, 110, 150), dual decoder, batch size 8, warm-up of a tenth of the epochs.

## 2026-11-05 — Correction: the net flow of the permutation notebook reads the relevance (K10)

Corrects the entry above, whose "the pipeline itself is sound" did not consider which matrix the flow reads. No new run; read from the code.

**What doesn't work**
- `compute_hierarchical_net_flow` reads `_attn_matrix`, which `get_model_output` fills from `generate_relevance` of a summed transformer output, in the older code and in the fork (K10). That target is an arbitrary direction, so the flow heatmaps and the partial permutation curves of `attention_test.py`, of the notebook and of the manuscript's Fig. 4 rest on it. Each permutation level is a separately trained model, which makes differences between levels especially exposed.
- The notebook's text says only "relevance-weighted attention" and carries no caveat yet.

**What worked**
- The notebook keeps `checkpoints/level_<pct>/`, so raw attention can be compared on the same models without retraining (roadmap, "Added 2026-11-05", task 7).
- The CosMx flow plots (`flow_maps`) read the last layer's head-mean raw attention, not the relevance, and were checked on synth_spot (2026-10-31).

## 2026-11-05 — Script fixes for the mean-subtracted target and Moran's I after mean subtraction (code and stub only; no run)

The user: "normalized slide" means mean subtraction, per slide, region or cell type where annotations exist. Task 7 (which net flow) is parked; the notebook as it runs is fine for now.

**What worked**
- Checked without a model on stand-in gene lists: after the fixes only `moran_local_type` changes its checkpoint name, the Moran arms evaluate their own 30 genes, `rand50_type` keeps `cell_type_coarse` through `eval_cfg`.
- `label_residuals.py` takes a list of columns as one label; `cell_type_coarse+fov` runs on a 12-FOV stub (stub numbers mean nothing). An existing `cosmx_masking_genes.json` gains the spillover lists and keeps its masks.

**What doesn't work**
- `moran_local_type` is the only arm whose checkpoint name reaches 255 characters (242 the next), and the only one with no result in any fold. Likely the cause, not proven: 255 is still legal, and the job log has not been seen.
- `label_mean` cannot subtract a label unseen in training: FOVs of held-out donors get the overall mean, so it suits cell types and recurring regions, not FOVs.

## 2026-11-05 — Stability sweep and donor-independent CosMx splits (code and stubs only; no run)

The user: for regression the predefined donor split is not needed, CosMx splits can be shuffled independently of donor. Roadmap "Added 2026-11-05", task 4.

**What worked**
- FOVs shuffled within each section: 70 / 15 / 15 by cells on the real metadata, every section in every split, no FOV cut, repeatable. The sweep's list mode, the data path on a stub, the read script on relabelled outputs and the seeds job with a fake model all run.

**What doesn't work**
- Not yet known whether anything holds on data: nothing was trained. Runtimes are estimates.
- Scores on these splits cannot be compared with the donor-held-out ones, and say nothing about new donors, since test FOVs share a donor and tissue with training.
- Split pairs compare flows only on cells that are test cells in both draws: few FOVs, so that row may be thin or empty.

## 2026-11-05 — Net-flow permutation notebook: the out-of-memory crash, and first read of levels 0-50

Full run (seed 42, older checkout 662e864, 192 GB) finished levels 0, 10, 20, 50 and died in `get_model_output` at level 80. No traceback in the executed notebook: the kernel was killed. Plot cells come after training, so no figures. The Slurm .err was not read here.

**What worked**
- Cause found on toy data (360 cells, CPU), not measured on CosMx. Three things outlive a level: (1) `generate_relevance` does `I += matmul(I, attn_map)` with `attn_map` requiring grad, a C++ reference cycle that gc cannot see and that pins each window's whole graph; (2) `_setup_adata_manager_store` / `_per_instance_manager_store` keep every level's AnnData, with its attention and predictions; (3) a layer <-> hook cycle keeps the last window and the layer. Estimate for CosMx: ~45 GB per level, which reaches 192 GB in the fifth level, where it died.
- Fix in the notebook only (`relevance_without_graph`, `release_level`). Toy: 11.6 MB per level before, 0 after, on 662e864 and upstream. Attention bytes identical before/after on 662e864, so levels 0-50 stay valid next to new ones. Resume from a crash state tested on toy. The log now prints peak RAM per level: that is the check on real data.
- Also fixed: summary `last_*` were NaN and `epochs` 21 (the final validate/test passes add history rows); levels 0-50 keep the old values in the CSV. The sbatch restarts up to 3 times from finished levels.

**What doesn't work**
- Dropping `retain_graph=True` alone is not enough (88% on the toy, but the N² attention tensors stay, and they dominate at N ~ 650).
- Upstream still has the cycle; the fork's `generate_relevance` is no longer in place (not tested for the leak). Library fix not made.

**First read, levels 0-50 only, one seed, relevance not raw attention (K10)**
- Total absolute flow ND 4.99 / 5.11 / 4.71 / 2.40, T1D 4.34 / 4.13 / 3.68 / 2.47. Holds to 20%, about half at 50%.
- Correlation with level 0 stays high (ND 0.86 / 0.80 / 0.71, T1D 0.93 / 0.81 / 0.83). Level 0 is further from 10 (0.86) than 10 from 20 (0.99) in ND: separate-model noise may be as large as the small-level effect. Needs net_flow_seeds.
- Reconstruction does not drop with permutation: val_loss -0.5303 at level 0, -0.5328 at level 50; test cosine 0.668 to 0.678 at level 80. The model does not need the spatial arrangement to reach its loss after 20 epochs.

## 2026-11-06 — Net-flow permutation notebook: full run read (all six levels)

Results: local `results/net_flow_permutation/` (levels 0-50 from the 5 Nov job, 80 and 100 resumed 6 Nov; 0-50 unchanged to the digit against the partial folder). One seed per level, relevance not raw attention (K10), 20 epochs, 466 training windows.

**What worked**
- The memory fix holds on CosMx: peak RAM 15 GB after level 80, 16 GB after level 100 (only these two levels ran in the resumed process). About 4-5 min per level, so a full series costs under 30 min.

**Read**
- Total absolute flow: ND 4.99 / 5.11 / 4.71 / 2.40 / 0.97 / 0.60; T1D 4.34 / 4.13 / 3.68 / 2.47 / 1.20 / 0.67 (levels 0, 10, 20, 50, 80, 100).
- Cosine with level 0: ND 0.86 / 0.80 / 0.71 / 0.53 / -0.22; T1D 0.93 / 0.81 / 0.83 / 0.25 / -0.34. Slope on level 0 through the origin: ND 0.87 / 0.75 / 0.34 / 0.11 / -0.03; T1D 0.84 / 0.63 / 0.49 / 0.07 / -0.05.
- Reconstruction does not see the permutation: test loss -0.5297 at level 0, -0.5347 at level 100 (slightly better); test cosine 0.668 / 0.678 (80) / 0.662 (100). One seed, so these small differences are not read further.
- One score per cell type explains 93-98% of the squared flow at levels 0-50 (additive potential p_s - p_r), 85-89% at 80-100. The heatmaps carry about nine numbers (source/sink rank: Acinar and Endothelial sources, Endocrine and Immune sinks), not pair-specific flows.
- Mean across-FOV spread hardly moves (0.14 at level 0, 0.12-0.13 at level 100) while RMS mean flow drops from 0.16 to 0.02. What disappears is agreement between FOVs, not per-FOV flow size. Per-window max normalisation rescales noise to the same size as signal, so this spread is partly built in.

**What doesn't work (the inference, not the run)**
- The notebook keeps cell-type labels fixed while it moves the expression profiles, so permuting also breaks label <-> expression. Flow averaged by label must decay even if attention depended only on each cell's own expression and not on position at all. For additive potentials that predicts slope 1 - p (0.9, 0.8, 0.5, 0.2, 0); observed values are close. This is compatible with the confound, not proof of it; a spatial effect could look similar. The notebook text ("a flat line means the flow does not depend on where expression sits") claims more than the design can show.
- Decisive test, not run: from the saved checkpoints, recompute the attention at levels 80 and 100 and aggregate by transported labels (`labels[order]`, the type of the profile now at that position). Pattern returns: flow follows expression identity, the decay is label mixing. Pattern stays gone: it depends on arrangement.
- Stars are not a significance test (difference over across-FOV spread of one model, 72 x 5 entries uncorrected); no stars at 10-20%. Adjacent-level cosine in ND is 0.86 between levels 0 and 10 but 0.99 between 10 and 20, so model-to-model noise needs net_flow_seeds.

**Still open before the PR:** K10 raw attention comparison on the checkpoints, transported-label test, notebook wording, NaN `last_*` columns and `epochs` 21 vs 20 in the 0-50 rows of the summary table.

## 2026-11-06 — Fork synced with upstream; tasks 1, 2 and 4 as one job (code and stubs only; no run)

**What worked**
- `main` now contains upstream/main 4363aea (14 commits). 1748 tests pass. Every line upstream added is in the merge except the deliberate ones: `pp/__init__.py`, the masked metric signatures that gained `sce_gamma`, the seeding comment in `main_sweep.py`, the `CLAUDE.md` table reformatting.
- Metrics follow upstream: torchmetrics objects updated per batch and computed once per epoch, so R2, correlations and the masked metrics are exact epoch values, not means of batches. Fork additions moved onto them: SCE metric, loss-free runs, target centring, the masked cosine and SCE averaged over cells that have a masked gene. Pinned: `test_the_masked_metric_over_an_epoch_is_the_metric_of_all_its_batches`, `test_the_masked_metrics_skip_cells_without_a_masked_gene`.
- `slurm/new/tasks_1_2_4.sbatch`: one job, steps task1, task2_eval, task2_train, task4_flow, task4_runs, task4_read; estimate 8-10 h, unmeasured. A step that fails does not stop the next; `STEPS` runs a subset, so the same file can run tasks 1-2 and task 4 side by side. A preflight stops the job if the cluster's checkout is not the synced one. Checked: the order, environment and failure handling with a stub `python`; the stability yaml, split draws and the seed read on toy files. Nothing trained, no real data read.

**What doesn't work / what to watch**
- Upstream's `pp/__init__.py` imports `pp.background` and `pp.donor_scale`, which are not in its tree, so `interscale.main_sweep` does not import on upstream's main. The merge keeps the old `__init__`; nothing here uses those functions. Expect one small conflict when upstream adds the files.
- Upstream's best-epoch selection read `optim.lr_warmup`, wrong for a staged phase (scaled warm-up, epochs from 0). Fixed to the phase's value; mutation-checked. Upstream's classes refer to `tests/test_warmup_selection.py`, which it never committed; written here with a real `Trainer`, because a renamed private Lightning hook would otherwise fail silently.
- `moran_local_type` and the stability runs train on the synced code, the other Moran arms before it: the best checkpoint can no longer come from the warm-up. Expected not to matter at warm-up 10 of 100-300 epochs, not measured.
- Stability results moved to `06_11_results/cosmx_stability`, so runs on the synced code cannot mix with anything started on 2026-11-05.
- Task 1 still has no legnini arm: no type or region annotation there, clusters via `imitation_labels` would have to be added.

## 2026-11-06 — From a trained model to biology: interventions, gene sets, programme cards (code and toy only; no run)

The user needs readable downstream output by Tuesday: flows of information, not only attention, and embedding dimensions, tied to biology. Design: `.claude/interaction_readout_plan.md`, "Design 2026-11-06".

**What worked**
- `evaluation.interpretation`: cell-type flows by intervention (`identity` with a location-matched null, `state`, near / far), gene flows, known-pair support against expression-matched targets, attention sources and sinks within type, programme cards with a class. `evaluation.interpretation_report` builds a report per run and a summary over runs from the tables alone. 19 tests on a module of known dependence, 14 mutations caught, four of them only after the toy data was made harder (types in patches, a target detected in few cells).
- Biology resource, panel-filtered: 960 genes, 1,007 known pairs (OmniPath), Hallmark 46, KEGG 187, Reactome 647, GO BP 1,445 sets. The control sets are proposed by Claude from general knowledge and are for the user to check.

**What doesn't work / what to watch**
- Nothing has run on data. The synth_spot check of the readouts (`DOWNSTREAM=synth`) is the first thing to read.
- Every table fills whatever the model is: in the dry run a random stub got a pathway at q 0.04 and a full top-pairs table. Read against the twin.
- rand50 ties the own-cell ridge within type (notes.md 2026-11-05), so the context-use line may well be near 0. That would be the result, not a fault of the readout.
- The LIANA consensus URL is gone; the pairs are OmniPath's instead, which is not the resource `comparison_plan.md` names.

## 2026-11-06 — CellAgentChat: what its code allows on CosMx (task 6; code and synthetic data only, no run)

**What worked**
- `slurm/new/cellagent_run.py`, `cellagent_read.py`, `cellagent.sbatch` run the unmodified package: a fixed sample per section on global µm, log1p_norm in X, modes `none`, `far` (delta 0.25), `near` (delta 1), each distance mode again with coordinates shuffled among the same cells. On synthetic data with all 728 panel pairs, one planted co-located pair ranks first of 241 ligands and of 205 receptors by space gain, about 8x above the next gene.
- Measured on synthetic data (~58 active receptors per cell): 0.2 min at 400 cells, 0.8 min at 800: quadratic, ~11 min a distance run at 3,000 cells.

**What doesn't work**
- Whole sections in distance mode: an N x N table, then a Python loop per cell, active receptor, ligand and sender, ~10^12 operations at 10^5 cells; all cells at once also ~600 GB for the distances. So a sample, which loses a cell's real nearest neighbours (`near` is not contact range), or `--unit fov`, which has real density but reaches ~1 mm only.
- The August script reads `obsm['spatial']` with `batch_label=slide_condition`: FOV-local pixels, FOVs on top of each other, so its distances across FOVs are wrong. Its permutation test at threshold 10,000 on all cells would also build the model about 125 times.
- The package's significance test is not comparable between deltas: its null ignores delta. On random synthetic data 100% of the scores pass at delta 0.25, 3-9% at delta 1. Hence every score is saved and read against shuffled coordinates.
- A raw far-minus-near gain calls everything near: the planted pair gains 2.79 under the near kernel and 0.43 under the far one, as any clustered gene would. Compare ranks within a kernel.
- A far gain can come from tissue structure alone. It says a pair is co-located at that scale.


## 2026-11-07 — Tasks 1 and 4 from `tasks_1_2_4.sbatch`: label residuals on CosMx folds 0-2, and the first stability run (results in `results/cosmx_stability`, `results/label_residuals`)

What came back: `label_residuals` for synth_spot and CosMx folds 0-2 (no vannan25; labels `cell_type_coarse`, `CellTypes_max`, `fov`,
`cell_type_coarse+fov`); `cosmx_stability` with **one run of five** (`rand50_split0_s42`) and its twin, each flow read under both hidden
halves. No `stability.md`, no task 2 outputs, no `net_flow_seeds` folders in what was copied; the job log was not read here. The hidden-half
agreement below was computed locally from the saved flow npz files (test cells only, ~9k per section) with `cosmx_stability_read.py`'s own
`smooth` / `rho` (30 µm, 5,000 centres per section), not by the script's `main`.

**Task 1: what worked**
- synth_spot positive control holds again: `region` takes CM1-CM3 out (R² 0.96-1.00, I_local 0.93-0.97 -> 0.01-0.03), every planted programme
  keeps its I_local (local set 20/20 stays), M4 / M7 / M8 fill the freed global slots, Spearman(fall, planted scale) -0.72 -> -0.69.
- CosMx, the fold-0 read of 2026-11-02 reproduces on all three folds: the FOV takes out most of the global set (median kept I_global 0.15 /
  0.23 / 0.20) and hardly touches the local one (I_local kept 0.91 / 0.92 / 0.94); the cell type does the reverse (global I_global kept
  0.99 / 0.89 / 0.96, local I_local kept 0.55 / 0.75 / 0.41). `CellTypes_max` (14 levels) says the same as `cell_type_coarse`.
- Type + FOV together leave few candidates: 110 -> 31, 144 -> 64, 71 -> 19 (f0 / f1 / f2). Clean local genes in all three folds after it:
  IGF2, OLFM4 (REG1A, SST also, but spillover); CCL2, PRSS2, SERPINA1, MEG3, CPB1, SPINK1 in two.
- The Shapley split agrees across folds: the global set's largest block is the FOV (0.036 / 0.021 / 0.026), the local set's the own cell type
  (0.089 / 0.064 / 0.185). NegPrb in no set before or after, in any fold or label.

**Task 1: what doesn't work / caveats**
- **The sets depend on the fold (= which two donors)**, before any residual: only 3 of 20 global genes (HSP90AA1, HSP90AB1, RPL34) and 4 of
  20 local genes (ANXA4, COL18A1, FGFR3, IGF2) are in the set on all three training slides. Per-fold sets are not a property of the tissue.
- **The stress / immediate-early signature is fold 0's alone**: HSPA1A I_local 0.23 on f0, 0.05 on f1 and f2; HSPA1B 0.14 vs 0.03; FOS 0.13
  vs 0.02-0.03. The 2026-11-02 headline "global set = HSPA1A/B, FOS, JUN" describes one slide. Fits handling of that slide, not biology.
- **Type + FOV pushes spillover into the global set**: INS enters the global set on all three folds (R² 0.43 on f0), with CHGA, TTR, GCG,
  CUZD1 on some. FOV alone does it more (f2: 8 endocrine / acinar products global after). Islet-sized halos of spillover outlast the 2-hop
  reach. INS, CUZD1, SST, SPINK1, SCG5 are not in the spillover list of 2026-11-04 (PRSS2, CPB1, REG1A, GCG, IAPP, CHGA, TTR); a Moran arm
  built on type + FOV sets would mask them as "global" unless the list grows.
- The reach on CosMx is 37 µm (k-NN 8, 2 hops), so "global" here means beyond about two cell layers, not the ~290 µm of synth_spot.
- The map figures keep the blank space between FOVs and colour each panel on its own scale.
- vannan25 missing.

**Task 4 (stability), one run only: what worked**
- Training is cheap: 9.5 s / epoch, 1.5 GB GPU; evaluation ~8 min (ridge references ~5 min). So the four missing runs did not fail for time
  per run; the cause is in the job log.
- rand50 on FOV-shuffled splits (every donor in training, six sections) is trained: within type 0.156 vs twin 0.000; within section 0.217.
- Attention moves outwards with training: share beyond 250 µm 0.09 (twin 0.004), still lift 6.6-8.0 at 0-15 µm. The distance windows
  hardly move from init (15 / 61 / 247 / 1000 -> 12.5 / 61 / 231 / 1101 µm), as in the distance-window sweep.

**Task 4: what doesn't work**
- **The flow map of one checkpoint depends on the mask draw**: the same model, the same cells, the other random half of genes hidden: per-cell
  net flow Spearman 0.20-0.34, smoothed 0.13-0.41 over the six test sections. The twin agrees with itself more (smoothed 0.32-0.53),
  because its flow follows geometry. Trained vs twin 0.0-0.2. So a flow map from one mask draw is mostly that draw; this backs task 7's
  "mask average" before any seed or condition comparison of flow maps.
- Seeing the donors does not close the gap to the ridges: within type 0.156 vs ref_own 0.150 / ref_own+context 0.160; on spillover=no
  0.157 vs 0.161 / 0.164 (below both). Pooled raw SCE 0.715 vs 0.561 / 0.552, val R² -1.26. The raw SCE gap of 2026-11-05 was therefore not
  (only) unseen donors; the crop-to-section level shift stays the candidate.
- Per gene within type, the model beats ref_own by > 0.02 on 8 of 480 genes: GCG, TTR, CHGA, SST, CUZD1 (spillover), MEG3, SERPINA3,
  OLFM4, MT2A; ref_own+context matches all but MEG3. One seed, so not read further.
- Attribution again says context supplies 61% of the prediction's variance (56% on spillover=no) while the score ties an own-cell ridge.
- No seed or split spread exists yet: the task's output (a spread per readout as the bar for tasks 3 and 5) is still missing.
- Addendum, same day (the user): the job hit its 12 h limit inside `task4_runs`, after the first run. Tasks 1, 2 and `task4_flow` came
  before it in the job; their outputs (`MA/04_11_results/cosmx_masking`, `MA/net_flow_seeds`) were not copied here. Measured per stability
  run: ~16 min training, ~8 min evaluation (ridge fit ~5 min, cached per split draw), so the four missing runs need ~2-2.5 h on their own.

## 2026-11-07 — Downstream interpretation on CosMx (`downstream.sbatch`, step `cosmx`; results in `results/downstream_cosmx`)

Evaluation only, on the checkpoints of `04_11_results/cosmx_masking`: rand50_f0-2, rand50_type_f0-2, twins of both fold-0 arms. Predefined donor
folds, so each run is read on one test slide (one ND + one T1D donor). The synth check (`downstream_synth`) was not copied, so it is not yet
known whether the readouts find planted senders; everything below is read without that check.

**What worked**
- The job runs end to end on all 6 trained models and the rand50 twin. The twin's numbers are tiny (context ΔR² ~0.001, top gene flow 0.07),
  so the trained values are not readout noise.
- Programmes recur over folds: programmes 0-4 of rand50_f0 match rand50_f1 / f2 at loading cosine 0.79-0.94. Programme 1 (ANXA4, CLU, CCND1,
  COL18A1, FGFR3, SOX9, KRT8; "spatial state", 22% of the prediction's variance) holds the genes that were in the local Moran's I set on all three
  folds in task 1 (ANXA4, COL18A1, FGFR3). The most reproducible spatial signal of the day, in both analyses.
- The identity controls are found where they should be (identity_acinar, _ductal, _alpha, _beta at q 1e-7 to 5e-3 in programmes), and the
  handling sets flag the handling programmes (ribosomal, stress).
- Attention reads cell identity: sources are the endocrine types (Beta, Endocrine, Alpha; 25-30% in the top decile against 10% by chance) and
  ductal genes (identity_ductal q 1.5e-6, MHC-I q 1.2e-3); sinks are acinar genes (identity_acinar q 4e-5) and ribosomal genes.

**What doesn't work**
- **Nothing repeats over folds as a flow**: 0 of 81 type pairs keep one sign with a positive mean in every run, for state flows and for
  identity flows alike.
- **The large flows are segmentation spillover and handling, not communication**: Acinar -> every type (targets RGS2, CUZD1, RPL21, RPL34,
  SERPINA3: acinar products and ribosomal), Ductal -> Mast 0.47 (ANXA4, KRT8 in mast cells: ductal transcripts). Programme 1's top type is Mast
  too, so its ductal genes in mast cells may be spillover as well. Programmes are mostly `handling` (4-5 of 12) or `weak` (5-7).
- **Context ΔR² has both signs and is large**: mean over types -0.018 to -0.159; Mast -0.18 on f0 (shuffling other cells within their type
  *improves* the prediction). These models' R² is far off (val R² -1.3 to -3.4, raw SCE well above the ridges), so a change in R² mixes scale
  and level with context use. Gene-level ΔR² above 1 (LEFTY1 -> REG1A 1.34, top 7.16 on rand50_type_f2) says the same. **Readout fault, mine:**
  the interventions score by R², but these models need centred cosine within type (memory `metrics-cosine-not-r2`). Not fixed yet.
- **The known-pair test is biased**: AUC 0.449 / 0.487 / 0.503 against a null of ~0.48-0.49, but the untrained twin also gets 0.434 (p = 1).
  A random model sitting below the null means the expression matching or the null is off, not that known pairs are depleted. Unreadable until
  fixed.
- Attention does not follow use: Spearman of the type-by-type tables 0.00 over 81 pairs.
- Condition differences: 2 (state) and 4 (identity) of 81 pairs keep their sign over all six runs, fewer than chance with three donor pairs.
  The two arms share their donor pairs, so six runs are three independent comparisons. Programme condition effects flip sign between folds.
- `rand50_type_f0_untrained` wrote nothing ("no receiver was scored"; every step < 4 s). Silent failure; cause not read (job log).

## 2026-11-07 — Downstream readouts rescored by centred cosine within type; known-pair test within types; twin fix (code and toy data only; no run)

Fixes the readout faults found in the CosMx read above. Results of the next run go to `07_11_results/downstream_{cosmx,synth}`.

**What worked**
- Every intervention is now scored by the centred cosine within the receiver type: per receiver cell over its hidden genes (the flows, context use),
  per target across the type's receivers (flow targets, gene flows). Truth and the unchanged model's prediction are centred on the receiver type's
  means over the whole section, the changed prediction on the same point, and the per-gene sums are pooled over draws and sections. A flow is the
  drop, at most 2. A model 5x too large and shifted per gene gives identical flows (tested); under ΔR² it would have moved them ~25-fold.
- Known pairs are ranked within each receiver type, against targets matched on detection in that type, and pooled (`lr_support_by_type`). On random
  scores with type-specific receptors the old test (score maximised over types, matched on overall expression) gave AUC ~0.23 against a null of
  0.50; the new one 0.503 ± 0.060 with p < 0.05 for 6.5% and p < 0.01 for 1.0% of 200 null seeds; a planted pair is found (AUC > 0.8). This is
  the likely cause of the CosMx twin's 0.434 against 0.485: a receptor is often detected in few types, so its maximum is over fewer types.
- The untrained twin of a label-mean model now gets the training label means (`sweep_common.give_label_means`): checked on a real GlobalModel (it
  refused to predict before, the means equal the training means after, a model with means and an offset-free model are untouched).
- The readers load the copies shipped next to them before an installed `interscale.evaluation.interpretation` (checked with the editable fork
  installed), so an older copy on the cluster cannot win silently.
- 25 tests pass; 10 of 10 mutations caught, 4 of them only after stronger tests (the readable floor, the detection match, the 2-gene minimum per
  cell, the context-use ratio).

**What doesn't work / what to watch**
- Not run on data. Numbers from the 2026-11-07 CosMx read are not comparable with the next run's (another score, another pair test).
- A target's within-type score needs at least 3 receivers with it hidden; with few far receivers some far targets stay unscored (NaN, not 0).
- The cosine flow counts a level shift the perturbation causes at the receivers (the changed prediction is centred on the unchanged model's mean).
  That is intended (context moved the prediction), but it is not the same as `within_groups`, which centres every head on its own mean.
- Spillover genes are still scored as targets; `spillover` is not yet excluded from the flows.

## 2026-11-08 — Downstream readouts rerun with centred cosine within type: synth check and CosMx (`07_11_results/downstream_{synth,cosmx}`)

Evaluation only. Synth: `full_s42` of `03_11_results/synth_ranking` and its twin (2 region labels CM1/CM2, 224 senders, 15 planted targets).
CosMx: rand50_f0-2, rand50_type_f0-2 (checkpoints of `04_11_results/cosmx_masking`, predefined donor folds) and the two fold-0 twins. All 8
CosMx runs and both synth runs wrote complete reports; the summaries were rebuilt at the end of each step (synth 16:54, cosmx 18:17).
The two fold-0 twins are identical in every number (same init; the type target only changes the loss), so there is one twin.

**What worked**
- Synth: context use is large and learned: shuffling other cells' profiles within their type lowers the centred cosine within type by 0.51
  (among all cells 0.64); twin -0.002. Beyond 300 um still 0.18-0.20 (far CM2 has 14 receivers only).
- Synth: gene flows find the planted senders: best planted sender ranked first for 11 of 15 targets (MC1 rank 7, MC3 3, MC6 12), deltas up to
  0.88 (M8 -> MC11, M7 -> MC9, S1-S4 -> R1-R4); twin's top gene flow 0.010.
- CosMx: the cosine scoring removed the both-signed, oversized context numbers: context use is positive in every trained run, 0.0011-0.0043
  (state) and 0.0024-0.0052 (arrangement), against the twin's 0.00015 / 0.00035, so 10-30x the twin. In absolute terms it is ~1% of synth's:
  the CosMx models barely rest on other cells. `far` (beyond 60 um) is ~0 and both-signed.
- CosMx: flows now repeat over folds (last read: none). State: Acinar->Acinar 0.0018, Ductal->Endothelial 0.0017, Ductal->Ductal 0.0015,
  Acinar->Endothelial 0.0013, Acinar->Ductal 0.0013, all above the twin's largest (0.00032); Ductal->Alpha, Alpha->Fibroblasts, Beta->Beta,
  Endocrine->Alpha repeat too but sit at the twin's level. Identity: Acinar -> Acinar/Endothelial/Immune/Ductal/Fibroblasts 0.005-0.013
  (twin max 0.00043), Alpha->Alpha 0.0008. Six runs are three folds (the arms share test slides).
- Programmes 0-4 of rand50_f0 again match rand50_f1/f2 at loading cosine 0.67-0.94; the twin's programmes classify as "other", the trained ones
  as handling / spatial state / weak.

**What doesn't work / what to watch**
- Synth planted check: the twin also ranks a planted sender first for 5 of 15 targets (33%), against "chance about 1%" in the report. Its ranks
  are bimodal (1 or 200-224), so planted senders have extreme |effect| even at random weights: likely input scale (planted senders are the
  high-variance genes). The right bar is the twin's 5/15, not 1/224. Fix: rank against the twin's score per (sender, target), or standardise
  senders, and print the twin's rate next to the trained one.
- Synth: every programme classifies as `spatial state` in the trained model and the twin (12/12); the class thresholds do not separate on
  synth_spot. Attention biology and known biology are empty (no hierarchy readout in this model, no resource for synth genes). The over-runs
  tables are empty with one trained run.
- CosMx: the repeating flows are the ones the last read called spillover / identity: Acinar identity flows move CUZD1, SERPINA3, OLFM4, RGS2,
  RPL21 (acinar products, ribosomal) in every receiver type. State flows Acinar/Ductal -> Endothelial/Ductal are the only candidates that are not
  plainly identity, and their targets (XBP1, MT2A, CUZD1, RPL21) are still acinar/handling genes.
- CosMx known pairs: AUC 0.49-0.52 against null ~0.50; p = 0.37, 0.91, 0.52 (rand50) and 0.08, 0.23, 0.0145 (type). One of six below 0.05,
  not after correction. The twin is still below its null (0.470 vs 0.501, p = 1), less than before (0.434) but ~3 null SDs: the test is still
  not calibrated for a model with near-zero effects (ties at zero are the first suspect).
- CosMx gene flows: the top-15 table of rand50_type_f2 is INS in Alpha from 15 unrelated senders (ETV5, IL18, WIF1, IL6, GCG, ...) all at
  ~0.009: a spillover target that moves with any input, not 15 pathways. Trained top gene flow 0.006-0.013 vs twin 0.0057, so gene-level flows
  are at most twice the twin's.
- Condition: 2 (state) and 4 (identity) of 81 pairs keep their sign over all runs; with three donor pairs ~1/4 would by chance. Ductal ->
  Endothelial / Immune identity higher in T1D in all three folds is the only lead and is still one comparison per fold.
- Attention: Spearman with the intervention table -0.29 to 0.23 over runs (twin -0.22); Alpha / Beta are the top sources on f0 and the top
  sinks on f1 and type_f2, so the "endocrine sources" of the last read do not hold over folds. The twin's sources are also enriched for
  identity_acinar (q 4e-5): the enrichment reads identity, not training.

## 2026-11-08 — Downstream readout: known-pair p over shared genes, planted check against the twin, conditions over sections, stability config (code and saved tables only; no GPU run)

**Correction to the entry above:** the synth planted check is 12 of 15 targets first (80%), not 11; first here and not in the twin: 9 of 15 (60%).

**What worked**
- **The "twin below its null" was the null's spread, not a bias.** The 1,245 CosMx (pair, type) reads come from 124 receptors and 165 ligands
  (ITGB1 alone 144, FZD4/FZD5/ITGAV 56 each); the per-pair null draws every pair's target independently and gave SD 0.008. A bootstrap over
  ligand and receptor genes gives the observed mean SD 0.022, so the twin's 0.470 against 0.501 is z = -1.4. New `interpretation.lr_cluster_p`:
  `p` one-sided from z = (auc - null_mean) / sqrt(cluster_sd² + null_sd²); the old p kept as `p_independent`. Rebuilt from the 07_11 tables:
  rand50 f0/f1/f2 p = 0.44 / 0.74 / 0.51, rand50_type 0.24 / 0.38 / 0.15 (was 0.0145), twin 0.90. No known-pair support in any run.
  Test: random gene effects shared by a receptor's pairs, 12 seeds: per-pair p < 0.05 in 42%, the clustered p in 0%; cluster_sd 3-10x null_sd;
  distinct receptors keep cluster_sd ≈ null_sd and a planted effect is still found (p < 0.01).
- **Planted check against the twin** (`interpretation_report.twin_of`, `planted_against_twin`): the twin is `<run>_untrained`, else the twin of
  another seed of the same arm. Reports and the summary now give `planted_twin` and `planted_beyond_twin`. synth: 80% / twin 33% / beyond 60%.
- **Conditions over sections** (`condition_by_section`): when a run has ≥ 2 test sections per condition (the FOV-shuffled stability splits:
  3 ND + 3 T1D), per pair the difference of condition means, `separated` (every section of one condition beyond the other's) and an exact
  Mann-Whitney p (3 vs 3: 0.1 at best). Runs with one section per condition keep the over-runs table; its chance line now follows the run count.
- **Gene flows over runs** (`gene_flows_over_runs`): triples in the top 1000 of trained runs, with the twins' values. On the 07_11 folds no
  triple is in all six tops; the most repeated (HLA-A -> SERPINA3 in Mast, 5 of 6) is in the twin's top too (0.0023): an input property.
- **`DOWNSTREAM=stability`** in `downstream_read.py` and step `stability` in `downstream.sbatch`: the 5 cosmx_stability checkpoints + the twin
  of split0_s42, `cosmx_stability.prepare` for the FOV split columns, the training job's flow reused from `06_11_results/cosmx_stability/flow`,
  results `08_11_results/downstream_stability`. Checked without data: config import, run expansion (5 runs, 1 twin), selections.
- `--summary` rebuilds old results with all three fixes from their tables: no GPU rerun of cosmx or synth needed. Rebuilt in a scratch copy only.
- 30 tests pass (5 new); ruff clean.

**What doesn't work / what to watch**
- The bootstrap over genes is an approximation of the right null (a null that substitutes each receptor gene consistently would be exact); it is
  conservative where few genes carry many pairs.
- The stability step is unmeasured on data; it needs the four missing task 4 checkpoints (`tasks_1_2_4.sbatch`, `STEPS=task4_runs`).
- The stability runs share their test sections and donors: agreement over them is the model's stability, not a donor replicate.
- Programme classes still do not separate on synth_spot (all `spatial state`, the twin too); not touched.

## 2026-11-08 — Gene influence against spatial pattern (report section 7) and summary figures (code and toy data only; no run)

The user's success criterion (spatially variable genes among the most influential, local and global separated) as a readout. The user,
2026-11-08: important, but "just one way to measure relevance of results"; it sits next to the other readouts, not above them.

**What worked**
- `interpretation.gene_flows(far_radius=)`: the cells within the radius of a receiver keep their values, so only distant cells change.
  `gene_influence`: per sender gene the mean drop in centred cosine over every (receiver type, target) read (unbiased where nothing moves,
  so it can be set against the twin) and the largest. `moran_genes` (every gene at once, equal to `moran` per gene to 1e-12) and
  `gene_spatial`: per gene Moran's I within type (every (section, type) mean out) at 8 neighbours and across the section
  (`gene_scales.global_moran`), plus the raw local I (the usual SVG statistic).
- The read (`interpretation_report.influence_read`, from tables only): learned influence = influence minus the twin's; Spearman with each
  Moran's I, raw and inside expression deciles; the 20 most influential genes' share of spatially variable genes (top quarter by local or
  global I) against genes of the same expression (`expected`, permutation `p`); among spatially variable genes, whether global ones lean
  more to far influence than local ones (percentile far minus near, one-sided Mann-Whitney). Summary: the same per run, and the ranking's
  stability over runs (RBO, Spearman, top-20 Jaccard, genes in every top 20). Figure `influence_vs_moran.png` per run.
- `DOWNSTREAM_READOUTS` (default the four old readouts, so the queued stability job is unchanged); `influence` writes `gene_influence` into
  the same run folders. sbatch steps `synth_influence` (full_s42 + twin) and `stability_influence` (split0 s42-44 + twin).
- Summary figures `flows_over_runs.png` (state and identity, mean over runs; outlined: one sign in every run and beyond the largest twin
  flow) and `condition_over_runs.png` (outlined: one sign in every donor pair, or separated in every run with sections as replicates).
- 34 tests pass (9 new today); far region: a sender read within 150 is exactly unchanged beyond 160, a partner at 400 still lost.

**What doesn't work / what to watch**
- **Far receivers:** `pick_receivers` serves the rarest type first, and with zones the first types fill the 40% cover, leaving common types
  about one receiver per section. The far pass therefore uses the far per-type count (60, as `type_flows`); still few receivers, so far
  influence is noisier than near. Found while testing, on toy data.
- "Spatially variable" is relative (top quarter of the panel), so about a third to a half of the genes qualify; `expected` is the bar, not 0.
- Cost unmeasured: one pass per visible gene, section and region (~480 x 6 x 2 per CosMx run). synth first.
- Influence is read for the visible half only (the eval mask hides the other half in every cell); no complement pass.

## 2026-11-09 — Task 4 stability read: three seeds + twin on rand50, FOV-shuffled split 0, and the notebook's flow at seeds 43/44 (`results/cosmx_stability`)

What came back: `rand50_split0_s42/s43/s44` and `rand50_split0_s42_untrained`, each flow read under both hidden gene halves; `stability.md`
with the `net_flow_seeds` tables. **Not back: split draws 1 and 2 (seed 42)**, so no split spread; `stability.md` has no `split` rows.
`sections/rand50_split0_s44.md` holds two `training` sections (the run trained twice; scores identical, 7.4 vs 6.8 s/epoch). Extra checks
below were computed locally from the shipped tables and npz files (no data loaded).

**What worked**
- Scores are seed-stable: centred cosine on all hidden spots 0.244 / 0.243 / 0.231 (SD 0.007); within type, spillover=no 0.157 / 0.158 /
  0.156. Training is cheap (7-10 s / epoch, 1.5-1.8 GB).
- Per-gene within-type scores agree between seeds (Spearman 0.996, top-20 Jaccard 0.90-1.0), but also with the own-cell ridge (0.987): this
  stability is mostly *which genes are predictable*. The informative part: model minus ref_own per gene still agrees between seeds
  (0.79-0.82); per-gene seed SD 0.002. Beyond ref_own by > 0.02 in every seed: CHGA, GCG, SST, TTR (spillover) and MEG3; beyond
  ref_own+context: none.
- Decoder gene priority is learned and seed-stable: Spearman 0.97 between seeds, 0.04-0.05 with the twin; top-20 Jaccard 0.74-0.90 between
  seeds, 0.03 with the twin. The top genes are acinar / islet products and high-variance transcripts (CUZD1, CPB1, SPINK1, NEAT1, ANXA4,
  MT1X, MT2A, REG1A, PRSS2, INS, MALAT1), so it tracks expression variance and identity; whether it says more than gene variance is unchecked.
- The trained flow is not cell density: Spearman(net flow, cells within 30 µm) -0.03 to 0.09 per section; the twin 0.21-0.30. (rand50 hides
  genes, not cells: every cell is visible, so the blobs in the maps are not mask discs.)

**What doesn't work**
- **The per-cell flow map is not reproducible over seeds**: smoothed net flow Spearman -0.05 / 0.33 / 0.27 between seed pairs, at the level
  of a mask redraw on one checkpoint (0.32 / 0.28 / 0.48) and not far above trained vs twin (0.14). Visibly: in 2_T1D the large structure at
  the left of FOV 2 is a strong sink at s42 and a strong source at s43. Seed and mask draw are of similar size; split unknown.
- **The notebook's sender x receiver flow is not reproducible over seeds at level 0** (only the model seed differs): Spearman over the 36
  pairs -0.28 to 0.04, one outlier 0.61 (T1D s43/s44); at 100% -0.0 to 0.40. Every level's change from the 0% reference is below the
  seed SD (median change 0.05-0.13 vs SD 0.11-0.17); 3-14% of pairs beyond 2 SD, about what three seeds' SD gives by chance. So the
  permutation effect is not separated from seed noise at any level, and the high level-to-level similarity of 2026-11-06 (0.7-0.99) is the
  shared seed 42, not the data. Caveats: levels 10-80 have seed 42 only, set against a 3-seed mean (the table mixes a single run with a
  mean, so its change is not like for like at 10-80); an SD from 3 seeds is itself loose.
- **Cell-type net flow over seeds** (median over sections): Mast 0.26 ± 0.15, Immune 0.13 ± 0.07, Fibroblasts 0.11 ± 0.07, Acinar -0.05 ±
  0.04; SDs as large as the means for Alpha / Beta / Endocrine. No twin row (the reader skips twins in `cell_type_flow`), so how much of
  this is geometry is unknown.
- **Condition (T1D - ND), task 5's bar**: Fibroblasts 0.13 ± 0.03 and Mast 0.24 ± 0.11 are the only rows beyond 2 seed SD, but the SD is
  over seeds of the slide-averaged difference and omits the donor pairs. Per slide x seed: Mast > 0 in 8 of 9 (one 0.000), yet ND sections
  hold only 17 / 4 / 12 test mast cells; Fibroblasts > 0 on slides 1-2 in every seed, < 0 on slide 3 in two of three. Alpha > 0 in 8 of 9
  but 21 T1D alpha cells on slide 1. Nothing here survives donor pairs + twin + cell counts yet.

**Next**
- Copy back or rerun split draws 1 and 2 (job log first).
- Add the twin to `cell_type_flow` / `condition_bar`, and report the condition difference per slide with cell counts (min-count filter, e.g.
  >= 30 per section) instead of one seed SD.
- Make `net_flow_change` like for like: compare single seeds against the reference's seed spread (or run levels 10-80 at s43/s44 too).
- Per-cell flow maps need an average over seeds and mask draws before any reading; a single map is mostly its draw.

## 2026-11-09 — Does averaging stabilise the CosMx flow map? Mask halves and seeds of the three rand50 runs (local, from the shipped npz)

Computed with `cosmx_stability_read.py`'s own `load_flow` / `flow_agreement` (30 µm smoothing, 5,000 centres per section, median over the six
test sections); every map holds the same cells in the same order (checked).

**What worked**
- Averaging the two mask halves per seed raises the seed-pair agreement a little: -0.04 / 0.32 / 0.28 -> 0.15 / 0.35 / 0.26 (mean 0.19 -> 0.25).
- The mean of two seeds (mask-averaged) against the third: 0.25 / 0.20 / 0.45 (mean 0.30). A plain signal-plus-noise model with single-run
  reliability 0.25 predicts 0.32 for this, so the seeds' disagreement behaves like noise around one shared map, not like different solutions.
  By Spearman-Brown an average of k seeds would have reliability 0.50 (k = 3), 0.63 (5), 0.77 (10), 0.87 (20): two independent 10-seed ensembles
  should agree at about 0.77. Estimated from three seeds, so rough.

**What doesn't work / caveats**
- Part of the shared map is what the twin has too: the 3-seed mean against the twin 0.20 (single runs 0.14). An ensemble map is read against
  the twin like any other.
- A single run's map stays unreadable; only an ensemble (seeds x mask draws) is. Nothing here makes one model's attention identifiable.

## 2026-11-09 — CosMx flow ensembles (cosmx_ensemble), first read

Results: local `results/cosmx_ensemble/` (cluster `MA/09_11_results/cosmx_ensemble`). hier_tier, whole-cell masking (20 % + 60 um discs), FOV-shuffled split draw 0, 100 epochs on crops; flow from hidden queries (`fold_flow`, 2 fold draws). `cell_l1` (1 layer) seeds 42-51 + twin; `cell_l2` (2 layers) seeds 42-51 + split draws 1-2 + twin.

**Incomplete read:** `stability.md` holds cell_l1 with all 10 seeds but cell_l2 with only 7 (s42-s48), no split draws. Most likely the cell_l1 job's read ran while the cell_l2 job was still training, and the cell_l2 job's own read did not overwrite it (failed or not reached; its runs are complete: flow/ and sections/ for s49-s51, split1, split2 exist). Fix: `STEPS=read sbatch slurm/cosmx_ensemble.sbatch`; check JOB=A's log for the read step. cell_l2 numbers below are 7-seed numbers.

What worked:
- **Averaging seeds gives a reproducible map, as the signal-plus-noise model predicted.** Disjoint k-seed ensembles agree as Spearman-Brown says from the single-run agreement: cell_l1 k = 1..5: 0.489, 0.675, 0.758, 0.814, 0.840 (SB 0.489, 0.656, 0.741, 0.793, 0.827); cell_l2 k = 1..3: 0.456, 0.585, 0.724 (SB 0.456, 0.626, 0.715). Extrapolated two 10-seed ensembles: ~0.90 (l1), ~0.89 (l2).
- **Single-run reliability doubled against the rand50 gene-masking runs** (0.25 there, 0.46-0.49 here): whole-cell masking with the flow from hidden queries is the steadier readout. Mask draws hardly matter (two fold draws agree 0.98 / 0.99; twin 0.95).
- Scores are seed-stable: within-section hidden-cell cosine 0.129 +- 0.001 (l1), 0.134 +- 0.001 (l2); between knn32 (0.124) and knn128 (0.146). Gene readouts seed-stable (dim_gene_priority Spearman 0.994, top-20 Jaccard 0.94).

What does not work / limits:
- **Reproducible is not learned for the 1-layer model:** its ensemble map shares 0.38 with the untrained twin's (single run 0.34): a large part is geometry (with one layer the last-layer attention is the first-layer attention on input + PE). cell_l2: 0.09 (single 0.01), i.e. its map is learned.
- **The two depths disagree on cell types.** Mean net flow per cell (positive = source): Alpha -0.083 (l1) vs +0.099 (l2), Beta -0.079 vs +0.100, Ductal -0.038 vs -0.165, Immune -0.011 vs -0.132; only Acinar is a source in both (0.031 / 0.076). Ensemble SEs ~0.025, so the sign flips are real: which type "sends" depends on the architecture. No per-cell l1-vs-l2 ensemble agreement was computed (needed: add to the reader).
- **Condition differences:** none in cell_l1 (Acinar same sign on 3/3 slides but within the twin). cell_l2 (7 seeds): Acinar T1D - ND = -0.032, 3/3 slides, beyond the twin (max 0.021) — the only candidate; Endocrine (2 slides kept) and Immune (2/3) do not pass. Mast has no slide with 30 test cells in both sections.
- Maps: source / sink patches of ~50-150 um in both arms and in the twins; without type overlays they cannot be read as islets / ducts. Plot defect: the subtitle runs into the colour bar.

Decision pending (the user): ensembles vs a model-level fix. The data say averaging works and is predictable; it does not make the map biological (l1 = geometry, l1 vs l2 disagree on cell types).

## 2026-11-09 — Type-residual target on whole cells and a sparse-key penalty: built (code and toy data only; no run)

Asked by the user after the ensemble read: run the cell-type-mean target now (3 seeds, 300 epochs) and a second set where attention to sparse regions is penalised a bit, with a slide-dependent density radius.

What worked (checks, not results):
- The penalty is exactly the sparsity-weighted key mass of the attention the pass uses: equal to the hand sum over the exported cell-level attention for dense, routed and recomputed routed blocks; the gradient reaches the attention projections and not the decoder; the forward pass is unchanged bit for bit; off, nothing is computed. Three mutants (query weights for key weights, CLS penalised, recomputed blocks dropping their term) each fail two tests. Full suite 1792 passed.
- Density per section: relative to the section's own median count, radius 5 x its median nearest-neighbour distance, so two sections at different spacing read alike (tested); a grid corner reads ~0.3, an edge ~0.55, a hole rim ~0.8 of the interior.
- Both arms pass the config validator on `cosmx_hier.yaml`; `sparse_stats` matches a hand computation.

What doesn't work / open:
- The type-mean target was never read before: `rand50_type` (gene masking, donor folds) failed its first evaluation and its later scores were not copied here. So there is no earlier number to compare with.
- Only the cell level is penalised; a coarse cluster token in sparse tissue is not. The readout reports the last layer only (the penalty acts on both layers).
- Three seeds: the ensemble curve (needs 4) is not drawn; seed agreement and the per-arm SD are.

## 2026-11-10 — Within-type shuffle benchmark (celltype_shuffle), first read: Steamboat complete, GITIII 0-10 %, InterScale failed

Results: local `results/celltype_shuffle/` (report.md and figures rebuilt locally by `celltype_shuffle_read.py`; small tables only). CosMx, expression shuffled within section x `cell_type_coarse` (54 groups), nested levels 0/10/20/50/80/100 %, seed 42; every method on the same draw.

**Incomplete run:**
- **InterScale: nothing trained.** All six notebook runs died at the first training step with `CUDA error: no kernel image is available for execution on the device`: the job landed on an NVIDIA B300 node and the `interscale` env has torch 2.12.1+cu126, which has no kernels for that GPU. The `benchmark` env ran on the same GPU. Every job in the `interscale` env (cosmx_typeres, downstream, the ensemble read) can hit this.
- **GITIII:** 0 % and 10 % only; 20 % was cut by the 12 h limit, the seed bar (seed 43 at 0 %) never ran. 118 min for the first fit, 68 min for the second (20 epochs; my estimate was 30-60).

What worked:
- **Steamboat's cell-type interaction matrix (the package's symmetric one) hardly moves:** Spearman with its own 0 % matrix 0.999 / 0.993 / 0.977 / 0.965 / 0.945 at 10-100 %, against 0.970 between two seeds at 0 %. Only 80-100 % fall below the seed bar, and only a little. Its pair ranking is set by who sits next to whom (composition), not by the within-type arrangement of expression. The directed matrix moves more (0.870 at 100 %, bar 0.978).
- **What the shuffle does change in Steamboat is strength, not pattern:** the matrix's magnitude falls to 0.60 of 0 % (seed bar 0.88), and the local scale's marginal explained variance halves (0.101 -> 0.050; the second seed gives 0.125), while ego rises (0.174 -> 0.256) and the total stays (ev_full 0.240 -> 0.235). About half of what Steamboat's local attention explains is within-type spatial structure; the rest, and the whole pair ranking, is composition.
- GITIII at 10 %: edge matrix 0.947, net 0.924, share 0.995 against its 0 %; no seed bar, so unread. Its fit is weak overall (median per-gene Pearson of the state 0.080, 0.079 at 10 %).

What doesn't work / open:
- No InterScale curve, so no comparison yet. Steamboat has one seed bar (two seeds); GITIII none.
- Steamboat stopped at epoch 1742 at every seed-42 level (1748 at seed 43): the early-stopping trajectory is set by the seed; the levels' fits still differ (EV by scale).
- Next: keep the `interscale` env off B300 (a GPU constraint) or give it a CUDA 12.9+ torch; resume in the same folder with STEPS="interscale gitiii read" (finished runs are skipped; GITIII's four levels + seed bar ~6 h).

## 2026-11-10 — Type-residual target on whole cells (cosmx_typeres), first read: 2 of 3 seeds, sparse-key arm not run

Results: local `results/cosmx_typeres/`. cell_l2 (hierarchical, 2 layers, whole cells hidden, FOV-shuffled split 0, the same 6,051 hidden test cells as the ensemble), `label_mean` on section x `cell_type_coarse`, 300 epochs (4.9 s an epoch: ~25 min training, ~2 min evaluation a run).

**Incomplete run:** typeres_split0_s42, _s43 and the twin were evaluated. s44 trained to the end (best checkpoint at epoch 289, as s43) but has no saved model and no evaluation, and the sparse-key arm never started. The read step ran afterwards, so the training process ended early as a whole (sweep_common would have gone on to the next run after an ordinary error): likely killed (host memory?) during s44's save or evaluation. The job log decides it. Not the time limit (the whole job would have stopped, read included).

What worked:
- **Within section x type the model is at the best smoother for the first time on CosMx:** centred cosine 0.085 / 0.088 (seed SD 0.002) against ref_nearest_128 0.083, _32 0.069, _8 0.049; the untrained twin 0.0005 (it has the type means and nothing else). Small margin (+0.004), two seeds.
- **No over-attention to sparse tissue in this model:** hidden queries' last-layer attention on keys below 0.6 of the section's median density is 4.0-5.1 % of their mass, while 6.9 % of the cells are that sparse; weighted mass 0.008-0.010 against the twin's 0.010. The pull towards edges and sparse regions the user saw is not in these rows; the penalty arm has little to remove here (other models and the visible-query flow plots may differ).
- Flow learned, not geometry: trained against twin 0.046; two fold draws agree 0.98.
- **Cell-type roles that hold across three models** (cell_l1, cell_l2, typeres): Acinar a source (+0.031 / +0.076 / +0.091), Ductal, Fibroblasts, Endothelial sinks, twin ~0. With the type means out of the target Acinar is still the main source: the residual of other cells is read from acinar neighbours, which fits the segmentation spillover found 2026-11-07 (acinar products in every type) better than signalling. Needs the gene-level readout to decide.
- Condition: Immune T1D - ND +0.038, all 3 slides agree, beyond the twin (0.028); cell_l2's ensemble gave +0.039 on 2 of 3 slides. The only difference that repeats over two models. Two seeds, three donor pairs, a flow (not signalling).

What doesn't work / open:
- **Within section the type-residual model is lower than cell_l2:** 0.122 vs 0.134 (visible-mean offset), below ref_nearest_32 (0.124) and _128 (0.146). The cell's own type mean alone (the twin) gives only 0.064 within section, below an 8-neighbour average (0.093): on CosMx a hidden cell's profile is told better by its neighbours than by its own type template (FOV / handling / spillover structure, cf. task 1).
- Seed agreement of the per-cell flow 0.20 (one pair; cell_l2: 0.46 over 21 pairs, min 0.20). Seed 43 puts 23 % of its attention on CLS, seed 42 3 %.
- Endocrine, Immune, Mast, Alpha, Beta change sign between the two seeds (SD >= |mean|); the endocrine roles stay unreadable.
- Next: the job log; then RUNS=typeres_split0_s44,typeres_sparse_split0 (about 2.5 h).

## 2026-11-10 — cosmx_typeres second read: s44 added (3 seeds), sparse-key arm still missing

Results: local `results/cosmx_typeres/` (summary rebuilt 2026-11-10 02:47 from typeres_split0_s42/_s43/_s44 + the s42 twin; read 09:02). s44 trained to the end (val_loss 0.874, 8.4 s an epoch on its node) and was evaluated. **No `typeres_sparse_split0` rows, twin or flow folder anywhere:** the sparse arm either was not in this rerun's RUNS, is still queued/running, or failed; the job log decides it.

What worked:
- **Within section x type the model holds at the bar over three seeds:** 0.085 / 0.088 / 0.088 (mean 0.087, SD 0.002) against ref_nearest_128 0.083, _32 0.069, _8 0.049, twin ~0. Margin +0.004, about two seed SDs: at the bar, not clearly above it.
- **No over-attention to sparse keys (3 seeds):** sparse_key_share 4.9 % (SD 0.9) of hidden queries' last-layer cell attention vs 6.9 % sparse cells; weighted mass 0.010 vs twin 0.011. Per section the share tracks the section's sparse fraction (1_T1D: 9-12 % on 13.6 % sparse cells). The penalty has little to remove in this model.
- Gene readouts stable over seeds: dim_gene_priority Spearman 0.995 (top-20 Jaccard 0.90-1.0), heads_within_per_gene 0.975 (0.82).
- Cell-type roles over 3 seeds: Acinar source +0.076 (SD 0.035, twin -0.001), Ductal sink -0.123 (SD 0.022, twin -0.003), Fibroblasts sink -0.074 (SD 0.051, twin +0.016). The others have SD >= |mean|.
- Condition (T1D - ND, slides agree and beyond the twin; none beyond seed): Immune +0.065 (3/3 slides, twin max 0.028; first read +0.038 with 2 seeds; cell_l2 ensemble +0.039), Endothelial +0.049 (3/3, twin max 0.039, seed SD 0.051), Acinar -0.027 (3/3, twin max 0.021), Alpha -0.046 (2/2 slides, only 21 T1D cells).

What doesn't work / open:
- Within section 0.124 (SD 0.003): = ref_nearest_32 (0.124), below _128 (0.146) and below cell_l2 (0.134). Headline centred cosine (section offset in) 0.166 vs knn128 0.227; raw SCE 0.885, worse than every reference and the twin (0.548).
- **Per-cell flow not reproducible over seeds:** pairs 0.20 / -0.03 / 0.26 (mean 0.14; cell_l2 0.46 over 21 pairs), while two mask draws of one model agree 0.98 and trained vs twin 0.046. Spearman-Brown: ~0.33 for a 3-seed mean, ~0.62 at 10 seeds. Per-cell flow maps of this target are not citable; only type-level means are. s43 keeps 23 % of attention on CLS (cell_mass 0.77) vs 2-3 % for s42/s44.
- Next: the sparse arm (RUNS=typeres_sparse_split0 + its twin, ~1 h), after checking why this rerun skipped it.

## 2026-11-10 — Noise ceiling by count splitting and band-pass fields with the type-residual target: built (code and toy data only; no run)

The user's long-range redesign (`/home/lehnerl/Arbeit/long-range-interaction-plan.md`) compared against the fork and trimmed to
`/home/lehnerl/Arbeit/long-range-interaction-plan-v2.md`: its stage 1 (a linear kernel field on the residual) is `FittedReferences`'
`ref_own+context` with bands and the `label_mean` target; the attention helpers (attn_pool, per-head windows, raw skip) are `cell_l2`
already; only the receiver gate and a low-rank readable map are new. Steps 0-1 built as one model-free CPU job in the user's folder
`notebooks_and_scripts/slurm/new/fable/` (`fields_common.py`, `noise_ceiling.py`, `band_fields.py`, `fields.sbatch`; self-contained,
`fov_split` copied verbatim, only `interscale.tl.masking.sample_block_mask` imported). Results will go to `MA/10_11_results/fields/`.

What worked (toy data: 3 sections x 8 FOVs x 150 cells with 3 NegPrb channels; synth_spot at 5 slides per condition, 30 x 30):
- All three steps run end to end and write `ceiling.md`, `fields.md`, `synth_spot/fields.md` with the CSVs behind them.
- The disc draw mirrors `draw_eval_masks` (one torch generator over the test graphs in section-category order, seed 1, 60 um, 10 %);
  the script warns when the hidden count is not the reads' 6,051 -- the first thing to check in the log.
- Bands are **rank shells** of visible neighbours (1-8, 9-32, 33-128, 129-512, 513-2048), not um annuli: one k-NN query, the first
  three shells are the k-NN bars, and a 1.5 mm ball in dense tissue (~10k cells) over 50k queries would be 10^8-10^9 pairs.
- Sender identity on the toy synth_spot: the source-only ridge's largest loading names the planted sender for 64 % of the response
  genes (median rank 1 of 21) with only 6 training slides.

What doesn't work / decided while building:
- **"Best band vs planted scale" is a weak recovery score for any linear reader.** An all-gene band predicts a smooth response best
  from its own gene in the nearest band (a smoother); with source genes only, a single smooth peak is still read best in the nearest
  band whatever the response's range, because its local level says how far the peak is. The file reports sender identity and the
  far-band gain by programme (all bands minus the nearest: expected > 0 for long-range / sparse responses, ~0 for short-range and
  noise) and keeps the Spearman marked weak. On the toy the far gain is negative for long_range (6 training slides overfit the
  extra bands); read it on the real 24.
- On CosMx there is no "own" feature: whole cells are hidden, so the base is the section x type training mean (scores 0 within type,
  the twin's number). `own` features would need gene masking, not the setting the model reads were made in.
- A constant prediction (the base within its type) is scored 0, not NaN (`cosine_rows`): the twin's reading.
- The half-count ceiling is a lower bound (half the depth); thinning 0.3 is in the job to show the depth trend.

## 2026-11-10 — Within-type shuffle as two notebooks, 0 % vs 100 % (code and toy data only; no run)

Asked by the user: the Steamboat + GITIII shuffle experiment and the InterScale one as notebooks with plots, two runs, two envs, results to `MA/11_11_results`. Only 0 % vs 100 % (within section x `cell_type_coarse`), one seed, no arms. The user (mid-build): "No need to compare older fits or look at seeds, the notebook needs to be reproducible", so neither notebook reads earlier results or seed bars; everything is trained from the data file with fixed seeds.

Files (flat in `slurm/new/`): `celltype_shuffle_baselines.ipynb` + `.sbatch` (env `benchmark`, 256 GB, 12 h; Steamboat tutorial settings and GITIII 20 epochs, code ported from the 09 Nov scripts that ran on the cluster) and `celltype_shuffle_interscale.ipynb` + `.sbatch` (env `interscale`, older checkout on PYTHONPATH, 192 GB, 4 h; the 06 Nov net-flow notebook with only the shuffle, levels, paths and an added 0 vs 100 scatter changed). Both draw the shuffle with the same function and seed 42 and print each order's sha1; on a toy file both gave the same hash. Results `MA/11_11_results/celltype_shuffle_{baselines,interscale}/`.

Checks: the baselines notebook ran end to end on a synthetic toy (1,040 cells, 80 genes) on the local GPU, including resume (all four fits loaded on the second run). One bug found that way: the GITIII settings dict was shadowed by the imported `GITIII` class (renamed to `GITIII_SETTINGS`, class imported as `GITIIIModel`). The InterScale notebook's model cells were not run locally (rule); its data, shuffle and plotting cells ran on the toy with fabricated flow matrices, and a static check found no undefined names (it had found `CONFIG_FILE`, left from the removed YAML override). The interscale sbatch resubmits itself excluding the node if the env cannot run on the GPU (B300).

## 2026-11-11 — Within-type shuffle (InterScale + Steamboat + GITIII) and the fields job (noise ceiling, band fields): first read

Results: local `/home/lehnerl/Arbeit/results/11_11_results/` (`celltype_shuffle_baselines/`, `celltype_shuffle_interscale/`, `fields/` -- the fields job wrote to 11_11, not 10_11). Same shuffle draw in both notebooks (order sha1 0 % `791ba4fe0816`, 100 % `acabdf484822`, 386,665 profiles moved, 54 section x type groups). One seed (42) per level, no seed bars (the notebooks' design).

**Which InterScale ran:** the original `CombinedModel` at commit 4363aea (GCN 2 layers radius 110 px + `self-attn-transformer` 3 layers / 4 heads, n_embed 16, linear dual decoders, GaussianNLL, 20 % node masking, 20 epochs, warm-up 2, batch 8 windows, 466 training windows = 59 steps an epoch, V100, ~5 min a level). The transformer has **no spatial input**: no positional encoding, no distance bias, no coordinates; `long_range_attention` off (raises NotImplementedError in that version); only the diagonal is masked, so every cell attends to every other cell of its window (sliding_window_square, median 631 cells, max 1,600) as a set. Spatial information reaches it only through the GCN embedding (2-hop neighbourhood) and window membership.

What worked:
- **The shuffle removes what it should (model-free):** per-gene correlation of a cell's within-type state with its 8-NN mean state: mean 0.034 -> 0.000 (max 0.487 -> 0.008); top genes PRSS2, SST, HSPA1A, CPB1, REG1A (0.42-0.49) all go to ~0.
- **Steamboat (reproduces the 11-10 read):** total EV unchanged (0.239 -> 0.237) but local EV halves (0.108 -> 0.054), ego takes it over (0.132 -> 0.231), global 0.244 -> 0.198. Pair ranking holds (symmetric Spearman 0.93, directed 0.87, net 0.95; adjacency control 1.00); strength drops (magnitude 0.63; net 1.13). Its type x type map = composition; within-type structure lives in the strength.
- **GITIII responds most:** state PCC median 0.081 -> 0.051 (mean 0.113 -> 0.072); edge Spearman 0.71, magnitude 0.49; share (composition-weighted) 0.97 / 1.00; net 0.45 (net of a nearly symmetric edge matrix: noise-dominated, do not read). PCC stays > 0 at 100 %: probably section-level offsets (GITIII's state is relative to a type mean not taken per section, while the shuffle is within section) -- not checked.
- **Fields job checks pass:** 6,051 hidden cells, `knn<k>_full` within type 0.049 / 0.069 / 0.083 = the model reads' bars exactly.
- **CosMx band fields (within type):** cum4 0.085, cum5 0.085 vs knn128 0.083 (= the typeres model's 0.087). Far bands alone (ranks 129-2048, ~150-450 um) 0.080, but the far gain over cum3 is +0.002 in total. The genes that gain from far bands are heat-shock / stress (HSPA1A +0.056, FOS, HSPA1B, BAG3, HSP90AA1, HSPB1) and islet hormones (INS, GCG): handling gradients and hormone spillover more than signalling. **No linear long-range signal beyond rank 128 on CosMx.**
- **synth_spot band fields:** nearest band (rank 4, 283 um) best (within section 0.399 vs knn4 0.303); adding bands lowers it (cum5 0.346, the ridge overfits more bands). The source-only far-band gain separates programmes as designed: long_range +0.101 (median 0.064), sparse_response +0.019, noise programmes ~0 (-0.005 to +0.002), region_marker -0.067; it grows with the planted scale (MC8 / MC9 / MC10 / MC11 at 2-4 mm: +0.22 to +0.37; scales <= 1.2 mm ~0 except MC14/15).

What doesn't work / open:
- **The InterScale run never left the gene-mean floor:** masked R2 -0.007 (0 %) / -0.006 (100 %), masked cosine 0.134 / 0.134, val loss -0.5322 / -0.5317, flat from epoch ~6 on in both runs. Its flow keeps its ranking (Spearman ND 0.944 / T1D 0.965, magnitude 0.90 / 0.91) and its per-type signal moves little (largest drops Alpha, Ductal -23 to -30 %), but a model that learned no within-type structure at 0 % has nothing to lose under the shuffle: **this stability is not evidence about composition vs arrangement, and the InterScale arm is not comparable to Steamboat / GITIII.** The flow pattern itself (Acinar net sender to every type, Alpha / Beta net receivers from every type) is the same at both levels. The 20-epoch budget and one training slide are the obvious suspects; the fork models (cell_l2, typeres) get to the knn bars but are not "original-model" material ([[pr-uses-original-model-only]]).
- **The noise-ceiling read rule was wrong (mine, in noise_ceiling.py's header text).** `own_half` = cos(A half, B half) is a split-half *reliability* rho, not the ceiling of a predictor of B: a perfect predictor (the true profile) scores ~sqrt(rho) against a noisy half. Evidence in the table: for HSPA1A, HSPB1, HSPA1B, FOS the 128-NN mean beats own_half (0.567 vs 0.409 ...), impossible if own_half were a ceiling. Within type: rho = 0.106 at half depth -> ceiling ~0.33; Spearman-Brown to full depth rho ~0.19 -> ceiling ~0.44 (rough: per-cell cosine of log-normalised halves is only correlation-like). So knn128 / the models at ~0.085 explain roughly (0.085/0.44)^2 ~ 4 % of the reproducible within-type residual. Read: the residual is not mostly count noise; most of it is cell-intrinsic and not predicted by neighbours at any band. The "~0.1 = data are the limit" branch does not apply; nor does it show the models underfit, because the ceiling counts the cell's own state, which neighbours need not determine.
- synth_spot sender identity from the source-only ridge: 5 / 25 (20 %; the toy gave 64 %). Long-range responses 5 / 15 correct, planted sender median rank 3; sparse responses 0 / 10, median rank ~15 of 21 (chance). Sparse responses are not linearly recoverable (cosine <= 0.12).
- Ridge alphas sit at the top of the grid on CosMx (1e5-1e6, b1/b2/cum* at 1e6, the largest tried): the grid should go higher before reading small differences between fits.
- Next options: (a) disattenuated ceiling column in noise_ceiling.py (sqrt of own_half, Spearman-Brown to full depth); (b) InterScale shuffle rerun with a budget that leaves the floor, or drop it from the comparison and say why; (c) per-section state for GITIII to test the offset explanation.

## 2026-11-11 — Noise ceiling re-read with a depth model; FOV control and wider alpha grid for the band fields (built, reread run, cluster run pending)

The user asked to improve the ceiling script and add checks (no shuffle rerun). Files changed in `notebooks_and_scripts/slurm/new/fable/`: `noise_ceiling.py` (depth model, per-gene shares, `--reread` mode, corrected read text), `band_fields.py` (FOV control, alpha-edge flag), `fields_common.py` (alphas to 1e8), `fields.sbatch` (step `reread`, results `$FIELDS_RESULTS`, default `MA/11_11_results/fields_v2`).

What worked:
- **Depth model `r(f) = f / (f + c)` fitted to the split-half value holds on the real data.** Per gene, fitted at 0.5/0.5, it predicts the 0.3/0.7 split-half value with Pearson 0.923 and median observed/predicted 0.995 (874 genes); per cell within type it predicts 0.097 vs observed 0.098. So the sqrt / Spearman-Brown correction is justified, not just theory.
- **Ceiling within type at full depth ~0.44 per cell** (r_full 0.19); within section ~0.50. knn128 reaches 0.19 of the within-type ceiling (0.29 within section). Per gene (585 of 960 genes with r_full >= 0.1, median r_full 0.115): **knn128 explains a median 1.2 % of a gene's reproducible within-type variance (pooled 3.9 %); the best band ridge (cum5) median 2.7 %, pooled 4.2 %.**
- **What the neighbourhood explains is stress / quality, not communication-looking genes:** best shares HSPA1A 0.39-0.50, HSPA1B, HSPB1, FOS, JUN, HSP90AA1/AB1, BAG3, ribosomal RPL34/21/22/32, TNFRSF12A, MIF, MT1X. **The most reproducible genes are cell-intrinsic:** SST (r_full 0.70, share 0.004-0.008), IGF2, MALAT1, GCG, CHGA, TPSB2/TPSAB1 (mast), ANXA4 -- subtype / state identity within the coarse type that no neighbourhood predicts.
- `--reread` wrote `ceiling_reliability.md` + `ceiling_reliability_genes.csv` into the local `11_11_results/fields/` (no data needed).
- Toy checks (3 sections x 20 FOVs x 300 cells, planted FOV offset on 5 genes, smooth field on 5, intrinsic on 1): both scripts run end to end; for the FOV genes the far gain +0.007 becomes -0.002 after the FOV block (fov alone 0.355 vs cum3 0.372). With 8 FOVs per section each test graph is one FOV and the FOV block is constant (centred away) -- on CosMx a test graph holds 15 % of a section's FOVs, several each.

What doesn't work / open:
- The per-cell ceiling is approximate: the share at half vs full depth agrees per gene (Spearman 0.85) but not exactly (median ratio 0.87), and knn128 half/full (0.079 / 0.083) is closer than the cell-level model predicts (0.062): neighbour-predictable signal sits in high-reliability genes. Read the genes, not the per-cell number.
- On the old run the variance-weighted share is missing (no `truth_var` column); the next full run adds it.
- **Not run on the cluster:** the FOV control (`fov`, `cum3+fov`, `cum5+fov`, per gene `fov_gain`, `far_gain_after_fov`) and the alpha grid to 1e8 (the first run chose 1e6, the top, for most fits). Quickest: `STEPS="cosmx reread" FIELDS_RESULTS=$MA/11_11_results/fields` (overwrites the fields*.csv there), or the full job into `fields_v2`. Time of the first run not recorded; the alpha grid adds ~30 % to the ridge step, the FOV fits ~4 more solves.
- Idea not built: a finer cell-type key for the within-type residual (SST inside coarse types suggests subtypes are the largest reproducible part); needs a fine label in `cosmx_inputs.h5ad`.

## 2026-11-12 — Fields job run 2 (FOV control, wide alpha grid, variance-weighted ceiling shares): read

Results: `MA/12_11_results/fields/` (local `/home/lehnerl/Arbeit/results/12_11_results/fields/`). Full plan and decision: `.claude/ceiling_plan.md` (run 2 read).

What worked:
- Fully reproducible: every number run 1 also computed is identical (ceiling, bars 0.049 / 0.069 / 0.083, band fits, synth_spot); 6,051 cells.
- Alpha grid to 1e8: every fit still picks 1e5-1e6, held-out score falls above (cum5 0.102 / 0.095 / 0.091 at 1e6 / 1e7 / 1e8); no fit at the edge. Run 1's fits stand.
- Variance-weighted share of the reproducible within-type variance: knn128 0.067, cum5 0.062 (standardised 0.039 / 0.042; median gene 0.012 / 0.027). The plain 128-NN mean matches the ridge.
- **FOV control: the stress far gain is a FOV offset.** HSPA1A far gain 0.056 -> 0.004 after the FOV block (FOV alone 0.504 vs cum3 0.418), FOS 0.036 -> 0.002, BAG3, HSP90AA1, HSPB1 -> <= 0. 22 genes with far gain > 0.01, 6 after the FOV.
- Per cell within type: cum3 0.0828, cum3+fov 0.0837, cum5 0.0853, cum5+fov 0.0852, fov alone 0.048. Far gain +0.0025, of which the FOV takes 0.0010.

What doesn't work / open:
- What survives the FOV is islet-related: INS (far gain after FOV 0.025, FOV gain 0), GCG (0.011), small TIMP1 / RGS2 / INSR / CUZD1. Biology (peri-islet cells) and technique (hormone transcript spillover, ambient RNA) are not separable without the receiver's cell type; not checked.
- No spread on the third-decimal differences (one draw, per-cell scores not saved, no bootstrap).
- Decision: steps 2-3 of the long-range plan skipped on CosMx (rule in ceiling_plan.md); CosMx statement for the thesis written there.

## 2026-11-12 — Follow-up to fields run 2: far signal by receiver type and distance (far_by_type.py), built; what the ceiling means for InterScale

Built (toy-checked, not run): `notebooks_and_scripts/slurm/new/fable/far_by_type.py` + step `bytype` in `fields.sbatch` (not in the default STEPS; reads run 2's `fields_alpha.csv`, writes `$RESULTS/by_type/`); `load_cosmx(negprb_obs=)` keeps the NegPrb total per cell (default off, data unchanged). Part A: cum3 / cum5 / cum3+fov / cum5+fov refitted on the same hidden test cells, scored per receiver cell type, FOV-block bootstrap intervals for the target genes (INS, GCG, TIMP1, RGS2, INSR, CUZD1, HSPA1A, FOS), far-band loadings per target gene. Part B (every cell, no fit): within-type residual (SD units of its type) by distance to the nearest source cell (INS:Beta, GCG:Alpha; mirror controls PRSS2:Acinar, CPB1:Acinar), NegPrb / total counts by distance, and the partial correlation with the source count at 150-450 um among receivers beyond 50 um of any source cell.

What worked:
- Toy (3 sections x 20 FOVs, contact spillover of a T0 marker into cells within 15 um of a T0 cell): Part B shows the contact profile in every receiver type (+0.6 to +0.76 SD under 10 um, below 0 by 20 um), flat profiles for the genes without spillover, flat negative probes, and a far-annulus partial r ~0 with intervals across 0: a contact effect is not called a far effect.
- The control pairs' annulus counts are skipped (acinar is ~half of all cells; a 450 um ball count over every cell would dominate the run).

What doesn't work / open:
- Part A on the toy has 277 hidden test cells: its numbers are noise; only the code path is checked. On CosMx Beta / Alpha have ~100-200 hidden test cells each: expect wide intervals; Part B (all cells) carries the decision.
- The grand-scheme read (ceiling_plan.md, "What it means for InterScale"): on CosMx there is no long-range target for a global component (FOV offsets + composition + possibly islet hormones); the preprint's CosMx programmes (local INS / oxidative stress, global immune / stromal) need the within-type target, the FOV covariate and the smoother before they read as communication; the audit should run before any model; legnini (SHH) is the dataset where it could find a target (slot S10, proposed). S6 dropped on CosMx.

## 2026-11-12 — Run 2 tables re-read: the neighbour term split into FOV, near-beyond-FOV and far; the gene list it leaves

From `12_11_results/fields/` (ceiling_reliability_genes.csv + fields_per_gene.csv; local computation on the result tables, no data). Shares of the reproducible within-type variance, pooled over the 585 reproducible genes: FOV mean alone 2.7 %; neighbourhood within ~100 um given the FOV (cum3+fov minus fov) 1.4 %; far bands given the FOV 0.1 %. So of the "4-7 %" more than half is FOV-scale; the local part is ~1.4 % pooled.

What worked:
- Per gene, 30 genes have > 5 % of their reproducible variance explained by the near neighbourhood beyond the FOV, 3 above 10 %, none above 20 %: RPL34 0.13, RPL21 0.13, JAK1 0.12, MIF 0.10, GSTP1 0.10, TNFRSF12A 0.09, TUBB 0.09, EZR 0.08, CD63, AZU1, EIF5A, CD81, KRT18, CD44, MT1X, NEAT1, PRSS2, YBX3, HLA-C, RPL32, REG1A, CALM1, MZT2A, IFITM3, DUSP5, S100A6, CLU, AZGP1, CD164, LGALS1 (0.05-0.08).
- Scale per gene (knn8 share / knn128 share): acinar markers PRSS2, REG1A ~1.0-1.2 = contact scale (spillover-like); the ribosomal / housekeeping / inflammation genes 0.1-0.5 = regional (tens of um), a shared local tissue state rather than contact.

What doesn't work / open:
- The list is translation / cytoskeleton / stress-adjacent (RPL, EIF5A, TUBB, CALM1, YBX3, MT1X) plus a few inflammation-related genes (JAK1, TNFRSF12A, MIF, CD44, IFITM3, HLA-C, LGALS1, DUSP5): local tissue state and quality more than ligand-receptor biology. Whether any of it is expression-dependent (communication) rather than "which types are nearby" (composition) is untested: a composition block (type counts per band) in the band ridge decides it. Not built.

## 2026-11-12 — Composition control, spillover index, legnini audit and the conditional-reading sweep: built (toys and unit tests; no run)

The user's request after the run-2 read: items 1 and 2 into the fields job, item 3 noted for later, item 4 implemented, a legnini script. All plans and the status in `ceiling_plan.md`.

What worked:
- **Composition control** (`band_fields.py`, CosMx): `comp<k>+fov` = the visible neighbours' cell-type shares per band + the FOV mean; `cum<k>|comp<k>+fov` = the expression bands fitted on its out-of-fold residual (stagewise, because one ridge would shrink 27 composition columns with the expression bands' alpha ~1e6). Toy (3 sections x 20 FOVs, a smooth type-independent field on 5 genes): within type comp+FOV 0.094, expression on its residual 0.153, cum3+fov 0.155: the control credits expression that composition does not carry.
- **Spillover index** (`spillover_index.py`, step `spill`): per gene its source type (highest linear mean; on the log scale a 10-fold marker read 1.3-fold, fixed), the within-type residual in other types' cells at < 15 um vs 25-50 um vs farther. Toy: the planted 15 um contact spillover of a T0 marker gives +0.99 SD (SE 0.016), the only flag; the field genes lead the near-term list as "flat".
- **legnini audit** (`--dataset legnini` in `noise_ceiling.py` and `band_fields.py`, step `legnini`): counts from `layers['raw']` (the file's X is not used), coordinates x 0.138 um, evaluation = the usual val + test organoids (7), discs 60 um over 20 %, a per programme x condition table (SHH vs Ctrl organoids as the negative control). Stub (17 organoids with the real sample names, planted SHH readout gradients in SHH organoids): readouts 0.17 within SHH organoids, ~0.02 in Ctrl.
- **Item 4 in the fork** (branch `conditional-context`): `model.decoder.group_offset: visible_mean` + `group_offset_key` (the FOV's visible mean of the residual from the type means as a known part, weight per gene from 1, tag `-grp-<key>`), `draw_masks(kind="gene_set+blocks")` (a gene set hidden only in disc cells; at rate 0 the same cells as a disc draw), `FittedReferences(group_attr=, composition_attr=)` (+4 references). `tests/test_group_offset.py` 8 pass; two mutations (group mean ignoring the graph, no label residual) caught; 371 tests of the touched areas pass. `CLAUDE.md` documents the option and points at ceiling_plan.md.
- **The sweep** (`slurm/new/cosmx_conditional.py`, `make_cosmx_conditional_yaml.py`, `cosmx_conditional_read.py`, `cosmx_conditional.sbatch`; `sweep_common.py` gains `EVAL_GENE_SET_DISCS`, `FITTED_GROUP_KEY`, `FITTED_COMPOSITION_KEY`): arms cond_fov_w150 (seeds 42, 43), cond_w150, cond_fov_w1000, twin; all three configs pass the fork's validators against cosmx_hier.yaml; checkpoint names 235-244 characters (first draft 254-255, shortened: column `fovs`, names c1-c3); `prepare` checked on a minimal AnnData (FOV codes per section, split by FOV, gene groups).

What doesn't work / open:
- Nothing ran on real data. Run 3 of the fields job (`13_11_results/fields`) comes first; the conditional sweep reads its ceiling and near-term list.
- The validator warns that channels at a cell rate of 0.2 let a neighbour give a hidden gene away: intended (the band fields read neighbours' expression of every gene too, and the ridges get the same inputs), but it means the model can smooth the hidden gene from neighbours; the bars include that (`ref_own+context`).
- The 150 um window is soft (frozen distance windows, ~e^-4.5 at 450 um; coarse cluster tokens are windowed by their radius), not a hard cut.
- Crops (1 mm) cut FOVs during training, so the FOV offset is computed from part of a FOV there and from whole FOVs at evaluation.
- On the legnini stub the band ridges chose the largest alpha (1e8); watch the edge warnings on the real data.
- ruff is not installed in interscale_env: the fork changes are not linted.
- Item 3 noted for later (ceiling_plan.md, open ideas): per gene typeres (3 seeds) against the band ridge on the same cells.

## 2026-11-12 — Fields run 3 (composition control, spillover index): read; legnini step gave nothing

Results `MA/13_11_results/fields/` (local `/home/lehnerl/Arbeit/results/13_11_results/fields/`). Ceiling, bars and every run 2 fit identical. Details in ceiling_plan.md, "Run 3".

What worked:
- Composition control: within type per cell fov 0.0476, comp3+fov 0.0482 (composition adds ~0 beyond the FOV, also at alpha 1), expression bands on that residual 0.0785 (+0.030), joint cum3+fov 0.0837. Per gene comp_gain > 0.02 only for type markers of the neighbours (GCG, PRSS2, REG1A, TTR, CPB1, INS, GC); expr_beyond_comp > 0.02 for 192 genes; far beyond composition ~0 (RGS2 alone > 0.02). The within-type neighbourhood term is neighbours' expression within ~100 um, not composition.
- Spillover index: 21 type markers flagged (GCG 0.71 SD, TTR, CHGA, INS, IAPP, SST, TPSB2/TPSAB1, CD74, HLA-DR, C1Q, CD163, CD68, LYZ, ANXA4); none of the 30 near-term genes; MIF, TUBB, HLA-C, MZT2A regional around islet cells.

What doesn't work / open:
- `fields/legnini/` and `cosmx_conditional/` are empty: legnini step failed or did not run; read the job log.
- PRSS2 / REG1A have contact-shaped profiles (index 0.27 / 0.21) but fail the specificity gate (1.4 / 1.3): acinar spillover the flag misses.
- INS spreads to ~50 um (+0.71 < 15 um, +0.46 at 25-50, +0.08 at 50-150 um): wider than contact; far_by_type.py not run.
- The all-gene band ridge fails synth_spot's control (from run 2's tables): long_range cum1 0.732 -> cum5 0.637, far_gain_all_genes < 0 for all 15 long-range genes incl. 2-4 mm (source-only reader +0.22 to +0.37). The 2026-11-10 "toy's 6 slides" explanation does not hold on the real 24. The CosMx far null needs a reader that passes the control (stagewise far|cum3 with its own alpha, or far bands as PCs) before it counts; `synth` was not in run 3's steps.

## 2026-11-12 — Bilinear part (readable cross-cell gene map) built; tests and mutations only, no run

The user's point 1 for next week (roadmap.md): genes that co-vary across space. Built on main: `model.decoder.bilinear.rank` / `.head` (decoder_plan.md D6; CLAUDE.md). `y_i += sum_j a_ij (x_j V) U^T`: a_ij the last layer's attention (head mean or one head, kept with its gradient), x_j the neighbours' visible genes (hidden entries 0), B = U V^T the sender-gene -> receiver-gene map. `evaluation/bilinear.py`: `gene_map`, `components` (SVD / raw), `planted_recovery`, `compare_maps`, `pair_weights` (per-gene flow arrows). Validator `_validate_bilinear`, tag `-bil<r>[h<k>]`, `TransformerNodeEncoderHook.last_attention_map()`, `encode` keeps the map for a bilinear model.

What worked:
- `tests/test_bilinear_decoder.py`: 10 tests (the part by hand from the returned map for the head mean and one head; no hidden value under cell and gene masking while visible values do reach it; off = no new state, no random draw, same prediction; the part trains U, V and the last layer's in_proj and not the context decoder; the clean pass reads its own map; refusals; the readouts on known matrices). Mutations caught: hidden values passed through (2 fail), the first layer's map (3), a right-padded gather (2).
- U zero at the start: a bilinear model's first prediction equals the plain model's bit for bit.

What doesn't work / open:
- Dense attention only: the part multiplies the explicit map, so CosMx whole sections (fused / routed) are out until a read-out head of its own exists (decoder_plan.md D6, last point). synth_spot and legnini run as they are.
- The attention serves two paths (tokens and the bilinear read); whether it compromises is a question for the run (compare the sum and the context part's shares against rank 0).
- Sweep `bilinear_decoder` (synth_spot planted pairs + legnini, ranks 8 / 16 / 32 vs 0, 3 seeds) not yet written; per-gene flow maps from `pair_weights` not yet wired into `flow_maps`.
