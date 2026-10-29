# Separating local and global: decoder structure and objectives

**Proposed 2026-10-19, nothing built.** The question (the user, 2026-10-19): restructure the
decoder and the dual model, and use losses that separate the local and the global embedding. Builds
on `global_plan.md` (what the sweeps showed), `contrastive_plan.md` (auxiliary losses that exist) and
`background.md` (why the dual decoder exists). Same rules as the other plans: every option is a
config switch whose default leaves the model unchanged (harness IDENTICAL), with a test through the
real model and trainer that hidden values cannot reach a prediction; the status table is updated in
the same commit as the work.

## Status

| stage | state | verified by | date |
|---|---|---|---|
| D0 — yardsticks: redundancy and attribution readouts | **implemented, unverified on data** -- `representations.redundancy_table`, `masked_probing.attribution_shares`, `StepOutput.parts` (roadmap.md, T2); not run on checkpoints | 15 tests, 25 mutations (with the oracle), harness IDENTICAL | 2026-10-21 |
| D1.0 — re-masking before the transformer (hidden cells only) | **implemented, unverified** -- `remask: hidden` (roadmap.md, T2, task 3); sweep `remask_test` written, not run | 16 tests, 8 mutations, harness IDENTICAL | 2026-10-21 |
| D1 — a context-only global embedding | **D1.2 implemented, unverified** -- `model.global_component.parameters.context_only: query` (2026-10-22): the transformer's queries from one learned token plus the node encoding, reading the cells' input tokens; sweep `context_query` (13 runs + 2 twins) **run and read 2026-10-23** (notes.md): with D3.1 (`add_ctx_fourier`, 3 seeds) the sum ties `add_fourier` (0.842 / 0.843) and the parts separate -- the local part carries the short-range, sparse, M11 and mirror programmes (shares 0.88-1.00), the context part the rings, no cancelling; in the plain dual decoder it costs (0.709). D1.1 not built (below) | 36 tests, 38 mutations, harness IDENTICAL; synth_spot | 2026-10-23 |
| D2.2 — regime-assigned losses | **implemented, unverified** -- `optim.regime_loss` (roadmap.md, T2); sweep `regime_loss` written, not run | 34 tests, mutation-checked, harness IDENTICAL | 2026-10-21 |
| D2 — different targets per head | D2.1 and D2.3 proposed; **D2.4 (gene-set targets from Moran's I: per-part losses, a set-loading term, read on held-out genes) proposed 2026-10-29**, after masking_plan.md M2 | | 2026-10-29 |
| D3.0 — own-cell skip in a global-only model | **run and read 2026-10-22** (`global_decoders`, 3 seeds; notes.md): the mirror programmes rise (MC4 0.27 → 0.49, MC2 0.41 → 0.56, MC6 0.47 → 0.60), about half the gap to `ref_own`; nothing else drops. The full-rank read `own_cell_skip_input: raw` (2026-10-22) tests whether the other half is the skip's width: **run and read 2026-10-23** -- it was the width: MC1 0.63 → 0.80, MC4 0.49 → 0.66 (`ref_own` 0.84, 0.68), programme mean 0.577 → 0.611 (3 seeds) | 21 + 12 tests, mutation-checked, harness IDENTICAL; synth_spot | 2026-10-23 |
| D3.1 — additive linear decoder in the dual model | **implemented, unverified** -- `model.decoder.additive` (2026-10-22): `y = W_l h_local + W_g h_global + b`, the loss on the sum; sweep `additive_decoder` (10 runs) **run and read 2026-10-23** (notes.md): the sum 0.843 against the plain global head's 0.820 (3 seeds), programme mean 0.687 at the bar; without D1 the global part repeats the local one (0.60 of the short-range variance) and the parts cancel on the mirror programmes; weights 0.1 / 0.9 keep the sum and give a readable local part; a part scored alone carries its decoder's bias (read the shares) | 22 tests, 41 mutations caught (with the raw skip and the probe cost), harness IDENTICAL; synth_spot | 2026-10-23 |
| D3 — the rest | D3.2 (gated mixture) and D3.3 (FiLM) proposed | | |
| D4 — orthogonality terms (rewritten 2026-10-23: eight scale-free forms as auxiliary losses) | **implemented, unverified on synth_spot** (2026-10-23): `train.orthogonality`, the `orth_*` terms; a term lowers its measure on a toy dual model only without dropout, so terms that read cells refuse dropout; the sweep to design | 55 tests, 43 mutations, harness IDENTICAL | 2026-10-23 |

## Why the two components do not separate today

`DualDecoderCombinedModule` gives each component a decoder and asks **both to predict the same full
target**; the loss is the mean of the two (`optim.loss_weights` 0.5 / 0.5). What the sweeps showed:

- **Near visible cells the global head copies the local one** (CKA 0.93 at the rim, 0.13 at
  290–500 µm), **and deep in a hole it acts as a smoother** of the visible cells. Copying is the
  optimum of two heads on one target, and it is free: the global token is the transformer's
  output, whose residual stream carries the cell's own GCN token. The long-range mask hides the
  GCN's neighbourhood from attention, but not the cell's own token from its residual stream.
- **Joint training barely moves the GCN** (local head 0.333 vs 0.327 trained alone), staged
  schedules gain nothing, and every variant ends at the same global validation loss (0.545–0.57).
- **The residual decoder** (global predicts the local head's error) **hurt the deep field** (0.389
  vs 0.438 at 290–500 µm): in the blind area the local prediction is a constant the residual first
  has to undo.
- **The interpretability contract** (`background.md`): separate, linear decoders so an effect can be
  attributed to local or global origin, with gene loadings. Any restructuring keeps a linear,
  attributable read-out per component.

Separation should come first from **what each embedding can see** and **what each head is asked to
predict**; penalties on the embeddings come after. A penalty alone (decorrelate the two) can be
met by the global embedding encoding noise or slide identity -- `contrastive_plan.md` documents the
slide-classifier collapse for the global contrastive pairing.

## D0 — yardsticks (no behaviour change)

What decides between the options below, all on the masked pass:

- **Communication:** programme masking (`global_plan.md`) -- the global or combined head against
  `ref_own+context` on long-range and sparse responses, and the planted senders' matched rank in
  their receivers' predicted responses (`response_influence`); the rewritten patch flow against its
  untrained twin.
- **Redundancy:** linear CKA local~global by depth (exists, `representations`); new (**built 2026-10-21**,
  `redundancy_table`): how much of the global embedding a ridge reads from the local one (R² of `h_global` from `h_local`,
  fit on train, scored on test, each query group on its own cells). Low is separated; with D1 it should drop well below
  today's ~0.9 at the rim. Next to it `global <- smoother` and `global <- pe`, the same read from a spatial average of the
  visible expression and from the position alone.
- **Attribution:** with D3, each component's share of the prediction (norm of its centred part over
  the norm of the whole) by depth and by programme. A separated model gives global the deep field and
  the long-range programmes, local the rim and the short-range ones. **Built 2026-10-21** for any head that is a sum of
  named parts (`attribution_shares`; `norm_ratio` and `cov_share`): today the own-cell skip and the residual decoder, D3.1
  when it exists. A plain dual decoder's heads each predict the whole target and have no parts.
- **No regression:** whole-cell reconstruction of the combined prediction must not drop below
  today's joint model.

## D1 — what the global embedding can see

**A context-only global embedding.** The global head should read what the cell's *context* says, not
the cell itself. Two ways, from the cheaper:

1. **Context read-out.** Decode the global head from the attention updates only: the transformer's
   output minus the cell's own input token (the residual stream's starting point), i.e. what the
   layers added. Exact for one layer; for two layers the second layer's update still sees the cell's
   token through the first layer's residual, so:
2. **Context query.** Run the transformer's query stream for the read-out from a position-only token
   (the positional encoding plus a learned constant), while keys and values are every other cell's
   GCN token -- cross-attention from "this place" to the tissue. The diagonal of the attention mask is
   already blocked, so the read-out never sees its own cell. This is the hidden cell's situation
   under discs, made the rule for every cell.

Either way the global head cannot copy the local one: its input carries no trace of the cell's own
token.

**Built 2026-10-22 (implemented, unverified on data): D1.2, `model.global_component.parameters.context_only: query`.**
   D1.1 (decode from the transformer's output minus the cell's input token) was not built: it does not remove the cell. The layer's
   feed-forward block can re-encode the token it is given (256 ReLU units hold a 64-dimensional `x` as `relu(x) - relu(-x)`), and the
   attention query is the cell's own token, so subtracting the residual path leaves two ways for the cell to reach its embedding.
   The query stream has none.
   - *The stream.* Each cell, and CLS, starts as `norm_input(context_token + node encoding)`: one learned token and the position half of
     the token the cell would have had, laid out like the tokens. (`GlobalInput.queries`, an attribute of the input and not a field of the
     tuple, which is unpacked by position in the module and some thirty tests; zeros without a node encoding, in which case the position enters through
     the attention biases and RoPE alone.) In every layer it reads the cells' **input** tokens (after `norm_input`, CLS included) through that
     layer's own weights, `read_context`: `q + attention(norm1(q), norm1(memory))`, then `q + feed_forward(norm2(q))`; the output is the
     encoder's final norm of the stream. The memory is the same at every layer and nothing of the stream is written into it, so there is no
     second-order path from a cell back to itself through another cell's updated token either (XLNet's query stream would have one). The
     cost is the encoder's: one cross-attention per layer in place of one self-attention. No second pass (the open question below).
   - *The mask* is the tokens' own: the diagonal, the long-range mask (behind a GCN none of the cells its neighbourhood mixed the cell
     into are keys), `visible_keys_only`, padding, the attention biases and RoPE, unchanged.
   - *What is exact* (tests): a cell's global embedding does not move, bit for bit, with its own expression, at one to three layers,
     with a node encoding, an attention bias or RoPE, for a pooled patch, and behind a GCN with the long-range mask (without the mask it
     does: the neighbours' GCN tokens hold the cell). The plain transformer's moves by 0.2. One exception: `attention.center_qk` centres
     queries and keys on the mean over all cells' tokens, the cell's own included at weight 1 / S; the dependence falls with S (tested), and
     the arm `add_ctx_fourier_nocenter` has none.
   - *What it asks of the rest.* The global embedding alone says nothing about the cell, so something else has to: the own-cell skip of a
     global-only model, the local part of the additive dual model. Two linear parts add; they do not gate, so a response that is the
     *product* of the cell's own receptor and the distance to a source (the sparse programmes) is out of reach of the pair, and is what D3.2
     (a gate) or D3.3 (FiLM) are for.
   - *Not defined:* attention rollout and relevance (they compose attention over layers as token mixing, and here every layer reads the same
     tokens; the readouts refuse the model, the single-layer attention maps and the gradient readouts work), other attention backends
     (self-attention only), graph-level prediction. The attention maps a context-only model returns are the read-out's: a cell queries, the
     other cells are keys.
   - *Sweep `context_query`* (slurm folder; programme masking on synth_spot): global-only `ctx_fourier`, `ctx_fourier_skip` (3 seeds),
     `ctx_fourier_skipraw`; dual `add_fourier` (3 seeds, the arm to beat), `add_ctx_fourier` (3 seeds, the main bet: D1.2 + D3.1),
     `add_ctx_fourier_nocenter`, `ctx_joint_fourier` (D1 alone). **Read:** `ctx_fourier` against `lin_fourier` (the long-range and ring
     programmes should hold, the mirror, short-range and sparse ones fall); the skip arms against `lin_fourier_skip` (what the skip brings
     back, what gating loses); the attribution shares of `add_ctx_fourier` against `add_fourier` per programme; the planted senders' rank.

0. **Re-masking, for hidden cells only (built 2026-10-21, `model.global_component.parameters.remask: hidden`).**
   The cheapest step towards 2: a cell hidden whole enters the transformer as one learned token instead of its
   GCN token (GraphMAE's re-mask), while visible cells keep theirs. It removes the copying where the disc task
   measured it -- a hidden rim cell's GCN token holds its visible neighbours -- but leaves visible cells, and
   every cell under programme masking (none is hidden whole there), as they are. Design and checks in
   roadmap.md, T2. Tests: a change to a cell's own input never changes its global head's prediction (only its
local one); hidden values never reach a prediction; the default model unchanged.

## D2 — what each head is asked to predict

Give the heads different targets, so they are not competing for one signal:

1. **Scale split.** `y = s(y) + (y - s(y))`, with `s` a Gaussian smoothing over the slide at a
   bandwidth `σ_g` (computed from the true expression of all cells: a training target only, never an
   input). The global head predicts the tissue-scale field `s(y)`, the local head the cell's
   deviation from it; the prediction is the sum. Attribution is exact by construction. `σ_g` is a
   knob (synth_spot ~300–1000 µm, legnini ~50–150 µm); a smoother of the *visible* cells is the bar
   for the global part.
2. **Regime-assigned losses.** One step carries two kinds of hidden cells: random single cells
   (their neighbourhood visible: the local head's job) and cells deep in discs (beyond the GCN's
   reach: the global head's job). The local loss is read on the first, the global loss on the
   second; the combined prediction on both. Cheapest of all to build -- per-cell loss weights from
   the depth the evaluation already computes.
3. **Programme targets.** The global head learns the programme task (the response genes hidden in
   every cell, predicted from their drivers), the local head whole-cell reconstruction: two masks per
   step, one loss each.
4. **Gene-set targets from Moran's I (D2.4; asked 2026-10-29, point 4 of the focus list).** The local
   and global gene sets of masking_plan.md M2 (Moran's I inside the GCN's reach vs at the largest range,
   from the training slides, no labels) say which part *should* carry a gene. The question is whether
   they can push the two embeddings apart. Three forms, cheapest first, all on the additive decoder
   (D3.1) with the context query (D1.2), which already separates by construction, so the read is
   whether the sets add to that:
   - *D2.4a, per-part losses.* The sum keeps its loss on every hidden entry. Each part gets an extra loss on
     its own set only: `w_p * L(part_local, y)[local set] + w_p * L(part_global, y)[global set]`. A part
     is thereby rewarded for carrying its set alone, and nothing forbids it the other genes. Config:
     `optim.part_targets: {local: [selectors], global: [selectors], weight: w_p}` (default off), the
     selectors resolved on `var` as `mask_genes` are (`scale_set=local`). The per-part loss reads the
     rows `StepOutput.parts` already exposes.
   - *D2.4b, a loading term.* A scale-free share, as D4's terms: the fraction of the global decoder's
     squared loadings that falls on the local set, plus the reverse, weighted by `orth_set_loadings`.
     It works in gene space, so dropout does not affect it (unlike the terms that read cells, D4).
   - *D2.4c, a probe against the wrong set* (adversarial, gradient reversal): not now, since it trains
     less stably and needs a probe in the loop.

   **Circularity.** A model trained on the sets cannot be read by the Moran's I yardstick on the same
   genes (task 1's Plot A and its Spearman). So each set is split at random into a training half and a
   held-out half (seeded, recorded in `var`), and only the training half enters D2.4a/b. The read is
   on the held-out half: if the separation generalises, held-out local genes land on the local part
   (`attribution_shares` per gene) and held-out global genes on the global part. Next to it, quantities
   no term trained: `redundancy_table` and CKA between the embeddings, the label probes per embedding,
   and the score of the sum (separation must not cost reconstruction).

   **Where it can run.** The context query needs torch's dense attention (refused with fused attention
   and with the hierarchy), so whole sections (CosMx, the lung data) are out until the fused attention
   takes separate queries (comparison_plan.md, "Whole sections", route a). First synth_spot under
   the `cells` task with discs (the planted scales are the truth: short-range receivers local, rings and
   region genes global), then legnini. Arms: `add_ctx_dist` (base), `+part_targets` at two weights,
   `+orth_set_loadings`, 3 seeds for the base and the better term; 300 epochs. Build after M2's
   selection has passed its synth_spot check.

## D3 — combining the heads into one prediction

0. **Own-cell skip in a global-only model (built 2026-10-21, `model.decoder.own_cell_skip`).** `ŷ = decoder(h_global) +
   W_o e_own`: D3.1 without a GCN, the cell's own token (before the transformer, from its visible genes) reaching the
   prediction around it. `W_o` starts at zero. It gives the own-versus-context split by construction and tests
   whether the global-only model's misses on the mirror programmes are a read-out problem. For the dual model D3.1
   below is the same idea with the GCN's token as the cell's own part. **Read 2026-10-22** (notes.md): it closes about half
   the gap to `ref_own` on the mirror programmes, although `W_o e_own` is linear in the same genes `ref_own` reads. The token
   is the output of the learned input layer, so `W_o` has rank at most `n_embed`; **`own_cell_skip_input: raw` (built
   2026-10-22)** reads the visible genes themselves (`W_o` `[n_output, n_input]`, full rank, hidden entries at the fill as in
   every other input). If it reaches `ref_own`, the limit was the skip's width; if it lands where the token read did, it was
   the training (`W_o` starts at zero, 100 epochs, the transformer path already carrying the own cell's token).

1. **Additive linear decoder.** `ŷ = W_l h_l + W_g h_g + b`, the loss on `ŷ` (optionally small
   per-head terms). Neither part predicts the full target alone, so duplicating a signal earns
   nothing; each part keeps its own gene loadings. Unlike today's residual decoder, the global part
   is not asked to undo a constant in the blind area -- and with D1 it cannot copy.

   **Built 2026-10-22 (implemented, unverified on data): `model.decoder.additive`.**
   - *The halves.* The stacked output keeps its layout (the training plan, the metrics and every head consumer read it as before):
     the **local half is the local decoder's output alone**, the **global half is the sum**, the prediction the model reports -- as
     under `global_residual`, whose gradient cut it does not have. `combine_global` is the one place that sums, in training and in
     `get_model_output`. Not with `global_residual`; needs the dual decoder at node level.
   - *The loss.* `optim.loss_weights` weigh the halves: `{local: 0, global: 1}` is the loss on the sum only (neither part is asked to
     predict the target alone), `{0.1, 0.9}` a small term for the local part alone (the "optional per-head term"). The default 0.5 / 0.5
     would also train the local part to predict the whole target alone, so the validator refuses it under the `joint` schedule. A
     staged schedule sets its own weights per phase: `local_then_global` trains the local part alone first and then the global part on
     the sum with the local component frozen -- a stagewise additive model.
   - *The readouts.* `StepOutput.parts["global"] = {"local", "global"}` (the sum's own tensors, not detached: the influence readouts differentiate the
     head `global_part`), so `attribution_shares` takes the sum apart per
     programme and depth; the evaluation adds the head `global_part` (the global decoder alone) next to `local` and `global`, so
     the head tables score what each part reads alone. Each decoder keeps its bias (their sum is the one bias).
   - *What it does not do.* The global component's tokens are built from the local ones, so the sum is a reparametrisation the
     optimiser may use unevenly -- a part that repeats the other earns nothing but is not punished. The shares measure it; D1 (a
     context-only global embedding) is what removes the possibility.
   - *Sweep `additive_decoder`* (slurm folder; programme masking on synth_spot; `joint_fourier`'s model): `joint_fourier` (the plain dual
     decoder) and `add_fourier` (loss on the sum) at 3 seeds, `add_fourier_l01` (weights 0.1 / 0.9), `add_fourier_l2g` (staged), and the
     pooled pair `add_fourier_pool500` / `joint_fourier_pool500` (a cell's prediction is its patch's global output plus its own local
     output: the un-pooling of roadmap.md T1b) at one seed. **Read:** the attribution shares per programme (short-range on the local
     part, long-range on the global part?), the sum's programme scores against `ref_own+context`, the oracle and the plain model's
     global head (first the short-range programmes and M11, then that the rings do not drop), `local` and `global_part` alone, and the
     senders' ranks.
2. **Gated mixture with a competitive loss** (Jacobs et al. 1991). Each head predicts `ŷ_k`; a gate
   `g_k(cell)` from depth and the embeddings weighs them; the loss is
   `-log Σ_k g_k exp(-L_k / τ)`. Unlike a cooperative mixture (loss on `Σ g_k ŷ_k`, which rewards
   both heads for every cell and so copying), the competitive form rewards a head on the cells it
   explains best: specialisation is the optimum. The gate is a per-cell attribution.
3. **FiLM** (global modulates the local decoder per gene programme: "context sets competence").
   Biologically apt, but the separation is weaker; last.

## D4 — orthogonality terms (rewritten 2026-10-23)

The user (2026-10-23) asked for real orthogonality losses (pointing to the Orthogonal Projection Loss, Ranasinghe et al.
2021) and wants one sweep over a variety of them; the contrastive losses get their own rework for global-only models
(`contrastive_plan.md`). The first version of this section proposed raw Frobenius penalties. They were the right *kind*
but gameable, and they did not say which of several orthogonalities they meant.

**Which orthogonality constrains what.**
- *Between two representations of the same cells, over cells* -- Domain Separation Networks' soft subspace
  orthogonality (Bousmalis et al. 2016, `||H_c^T H_p||_F^2`): no linear function of one part predicts the other across
  cells. This is the disentangling kind and what the description's "orthogonality between embedding subspaces" means.
  Written raw it is gameable: the additive decoder rescales a part freely, so shrinking it zeroes the penalty. Every term
  below is scale-free.
- *Per cell, between two vectors in a shared feature space* (`z_l(i) ⟂ z_g(i)`, or OPL's cross-class term with the
  component as the class): with two decoders a rotation of one part meets it without removing any information. It is
  kept as a control, to show that.
- *In gene space, between the parts' decoder loadings*: the two parts' outputs live in orthogonal gene subspaces --
  orthogonal gene programmes.
- *Within one embedding or decoder*: decorrelated dimensions, orthogonal loading vectors -- readable dimensions (task 2).
  The only kind a global-only model without parts has.
- *Between classes within one space* (OPL itself): needs labels and separates classes, not components. Not part of this
  stage; an arm for a classification track (roadmap T10 A) if one starts.

**The terms.** Each is an auxiliary loss (`train.aux_losses`) with its own weight `optim.aux_loss_weights.orth_<name>`
(default 0: never built, every run unchanged), so a sweep can switch any mix on. Each value lies in [0, 1].

| term | what it penalises | pair | invariant to | caveat |
|---|---|---|---|---|
| `orth_cka` | linear CKA (Kornblith et al. 2019) of `h_local` and `h_global` over cells: DSN's loss, normalised | dual | scale, rotation | dominated by high-variance directions: an unused inflated direction lowers it |
| `orth_cca` | mean squared canonical correlation of `h_local` and `h_global` | dual | any invertible linear map (what a linear decoder can undo) | ridge `optim.orthogonality.ridge`: directions below it escape |
| `orth_kernel_cka` | RBF kernel CKA (normalised HSIC) of the same, on a subsample | dual | scale, rotation; nonlinear dependence | bandwidth from the median distance |
| `orth_opl` | OPL's term with the component as the class: mean `|cos|` between every local and every global vector | dual | -- | the control: met by moving one part into the other's orthogonal complement |
| `orth_gene_corr` | per gene, the squared correlation over cells of the two parts' predictions | any decomposable head | per gene | makes each gene's prediction an orthogonal sum, so the attribution shares have no cross term |
| `orth_loadings` | mean squared cosine of the principal angles between the two parts' gene-space loadings | any decomposable head, linear decoders | any invertible linear map | a full-rank part (raw skip) spans every programme: refused |
| `orth_decorrelation` | mean squared correlation between distinct dimensions of the global embedding | any model with a global part | per-dimension scale | a basis choice: readable dimensions, not separation |
| `orth_loading_ortho` | mean squared cosine between distinct loading vectors of the global decoder | linear global decoder | per-column scale | each dimension a distinct gene programme |

"Dual" is the dual decoder (`local` = the GCN embedding, `global` = the transformer token); "decomposable head" is the
additive or residual dual decoder (parts `local`, `global`) or a global-only model with the own-cell skip (parts
`context`, `own_cell`).

**Rules.**
- Every term reads the cells whose predictions the loss reads (rows with a hidden entry), each graph centred on its own
  (OPL reads raw vectors, as the paper does). The pairwise terms (kernel CKA, OPL) read at most
  `optim.orthogonality.subsample` of them, drawn from a private generator.
- **No dropout on the path of a term that reads cells** (found while building, below): with it, training meets the term
  with noise that evaluation removes. Such a term refuses a model with encoder dropout (or, for `orth_gene_corr`, a
  nonlinear decoder's dropout); the arms it is compared with must drop dropout too. The loading terms read weights and
  do not care.
- A term whose model lacks what it needs refuses at construction, with the reason.
- Checkpoint names carry every non-zero auxiliary weight (`aux-<name><weight>`); until now even the contrastive terms
  left no trace, so arms differing only in one would have overwritten each other.

**Cautions.**
- Independence is a strong target. Local and far-field context are genuinely correlated (autocorrelation beyond two
  hops), so a penalty makes the global part the *residual* of the local one; shared variation goes to whichever part is
  cheaper, likely the local one.
- Every linear measure has a loophole (above), so a run is never read by the quantity it trained: the redundancy table,
  the attribution shares, the gene-scale split (`evaluation.gene_scales`: Spearman of each gene's scale with its global
  share), the programme scores against the bar, and the reconstruction, which must not drop much.
- With the context query (D1.2) the global part already cannot see the cell or, under the long-range mask, its
  neighbourhood. A penalty then only removes far-field information the near field also carries.

**Not in this stage.** An information bottleneck on the global embedding (KL to a prior, weight β), only together with
D3: it then encodes what the local part does not already supply. The contrastive losses (VICReg, context NCE, ring NCE)
moved to `contrastive_plan.md`'s rework for global-only models.

**Built 2026-10-23 (implemented; checked on toy models, unverified on synth_spot).** `train.orthogonality` (the measures),
the eight `orth_*` terms in `train.aux_losses`, `optim.orthogonality` (`subsample` 1024, `ridge` 1e-3), the checkpoint tag.
- *The measures, checked against their definitions* (notes.md, 2026-10-23): linear CKA and kernel CKA against the textbook
  formulas, the canonical overlap against the canonical correlations from a QR and against scipy's principal angles
  (exact), OPL's term against a double loop. The loopholes hold as stated: a copy under an inflated unused direction reads
  CKA 0.0015 and canonical overlap 0.997; a copy moved into the orthogonal complement of a shared space reads OPL 0.0 and
  overlap 1.0; a nonlinear copy (y = x²) reads linear CKA 0.009 and kernel CKA 0.43.
- *Dropout meets the terms with noise* (found while building). On a toy dual additive model trained with `orth_cka` at
  weight 1 and dropout 0.1, the training value fell 0.97 → 0.02 within three epochs while the validation value stayed at
  0.999. Without dropout the term works at evaluation: CKA 0.998 → 0.10, canonical overlap 0.47 → 0.19 (the same run
  without the term against with it, toy slides). Hence the rule above.
- *The canonical overlap scales each column to unit size before whitening* (an invertible map: no canonical correlation
  changes), so one inflated column cannot raise the ridge over the others; directions that are combinations of columns
  with little variance still escape below the ridge.
- *The own-cell skip under cell masking:* a hidden cell's own token is the fill, so the own-cell part is constant on the
  loss's cells and `orth_gene_corr` is 0 there; it needs gene masking.
- *Tested:* 55 tests (the measures and their invariances, the plumbing on hand-built step outputs, every term through a real
  dual and global-only step reaching exactly the parameters it should, the refusals, the state dict and a deep copy, the
  name, the settings, a real `train()` run in which the term lowers its measure); 43 mutations, all caught; harness
  IDENTICAL.
- *Not built:* a nonlinear redundancy read for the evaluation (the redundancy table is a ridge); the sweep (to be designed
  with the user).

## Recommended order

1. **D2.2 (regime-assigned losses)** -- an afternoon: per-cell loss weights from depth; it tests
   whether "global copies local" disappears once global is only rewarded where local is blind.
2. **D1.2 (context query) + D3.1 (additive linear decoder)** -- removes copying by construction and
   keeps a linear, attributable read-out per component; the main bet.
3. **D2.1 (scale split)** as the target alternative, compared with 2.
4. **D4** -- the orthogonality terms (rewritten 2026-10-23), one sweep over a variety of them on top of whichever of 2–3
   wins; the contrastive losses separately, reworked for global-only models.

Each arm on synth_spot under programme masking and discs, 3 seeds for the arms a decision rests on;
then legnini.

## Open questions

- ~~D1.2 needs a second query stream through the transformer (cost ~1 extra pass with the keys and
  values cached): acceptable, or start with the one-layer-exact D1.1?~~ Answered 2026-10-22: the stream
  replaces the encoder and reads the input tokens, so there is no second pass; and D1.1 is not exact at one layer.
- D2.1's `σ_g`: fixed per dataset, or several heads at several scales (a multi-scale decomposition)?
- Should the combined prediction or each component be what the downstream attribution analyses read?
