# Notes: what worked and what does not

A log of the verdict of every test. **Append only:** new entries go at the end of this file; never
overwrite or edit an earlier entry (a correction is a new entry that says what it corrects). Each entry
is `## <date> — <what the test was about>`, then **What worked** and **What doesn't work**. Tables,
settings and the reasoning live in the plan files (`global_plan.md`, `attention_plan.md`, ...); this
file keeps only the points.

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
