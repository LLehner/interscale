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
