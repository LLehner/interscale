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
| D1 — a context-only global embedding | proposed | | |
| D2.2 — regime-assigned losses | **implemented, unverified** -- `optim.regime_loss` (roadmap.md, T2); sweep `regime_loss` written, not run | 34 tests, mutation-checked, harness IDENTICAL | 2026-10-21 |
| D2 — different targets per head | D2.1 and D2.3 proposed | | |
| D3.0 — own-cell skip in a global-only model | **run and read 2026-10-22** (`global_decoders`, 3 seeds; notes.md): the mirror programmes rise (MC4 0.27 → 0.49, MC2 0.41 → 0.56, MC6 0.47 → 0.60), about half the gap to `ref_own`; nothing else drops. The full-rank read `own_cell_skip_input: raw` (2026-10-22) tests whether the other half is the skip's width: arms `lin_fourier_skipraw` (3 seeds), `lin_fourier_pool500_skipraw`, **not run** | 21 + 12 tests, mutation-checked, harness IDENTICAL | 2026-10-22 |
| D3.1 — additive linear decoder in the dual model | **implemented, unverified** -- `model.decoder.additive` (2026-10-22): `y = W_l h_local + W_g h_global + b`, the loss on the sum; sweep `additive_decoder` (10 runs) written, **not run** | 22 tests, 41 mutations caught (with the raw skip and the probe cost), harness IDENTICAL | 2026-10-22 |
| D3 — the rest | D3.2 (gated mixture) and D3.3 (FiLM) proposed | | |
| D4 — penalties on the embeddings; run the existing contrastive losses | proposed | | |

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

## D4 — penalties on the embeddings, on top of D1–D3

1. **Cross-covariance penalty** `||Cov(z_l, z_g)||_F^2` over a batch (the Barlow Twins cross term):
   no linear redundancy, cheap. Two stronger forms for the description's "orthogonality between
   embedding subspaces": on uncentred embeddings, `||Z_l^T Z_g||_F^2` (orthogonal column spaces, not
   just uncorrelated), and on the decoder weights, `||W_l^T W_g||_F^2` (a gene's loadings on the two
   scales point apart). HSIC for the non-linear case; an adversarial predictor (gradient reversal) or
   a CLUB mutual-information bound if needed -- more fragile.
2. **The existing contrastive losses** (`contrastive_plan.md`, stages 1, 2, 2b: VICReg, context NCE,
   ring NCE) are implemented and **were never evaluated** on a task that can show communication.
   Ring NCE is scale separation as a loss (positive: the outer ring; negative: the GCN's own
   neighbourhood). Run them under D0's yardsticks before building anything new in this stage.
3. **Information bottleneck** on the global embedding (KL to a prior, weight β), only together with
   D3: it then encodes what the local part does not already supply.

## Recommended order

1. **D2.2 (regime-assigned losses)** -- an afternoon: per-cell loss weights from depth; it tests
   whether "global copies local" disappears once global is only rewarded where local is blind.
2. **D1.2 (context query) + D3.1 (additive linear decoder)** -- removes copying by construction and
   keeps a linear, attributable read-out per component; the main bet.
3. **D2.1 (scale split)** as the target alternative, compared with 2.
4. **D4.2** -- the existing ring NCE and context NCE under the new yardsticks; **D4.1** cross-covariance
   as an add-on to whichever of 2–3 wins.

Each arm on synth_spot under programme masking and discs, 3 seeds for the arms a decision rests on;
then legnini.

## Open questions

- D1.2 needs a second query stream through the transformer (cost ~1 extra pass with the keys and
  values cached): acceptable, or start with the one-layer-exact D1.1?
- D2.1's `σ_g`: fixed per dataset, or several heads at several scales (a multi-scale decomposition)?
- Should the combined prediction or each component be what the downstream attribution analyses read?
