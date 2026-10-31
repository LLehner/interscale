# Staged plan: contrastive objectives for the global module

Working document for adding contrastive (and other auxiliary) losses to the global
component. Read [`background.md`](background.md) first for why the local/global split
exists — the design below is constrained by the fact that the global embedding is also
the interpretability substrate (gene loadings, net attention flow), not just a
prediction intermediate.

Trimmed 2026-10-31: the Stage 0 specification (built 2026-09-13; the code and CLAUDE.md describe it), the build
diaries and the per-stage file list were removed; git history has them. The decisions they recorded are kept below.

## Status

Last updated 2026-10-31. **Read 'What the probes cannot tell you' before interpreting any probe result.** Update this
table in the same commit as the work it describes. "Implemented, unverified" is a real state — a stage is only `done`
when something external says so (a test, a reproduced number, a run that was actually looked at).

Where it stands: none of Stages 1-2b has been read on a task that can show communication; they were built for the
dual model on synth_data_0 (legacy, see the focus on synth_spot). The orthogonality terms moved to decoder_plan.md D4;
the contrastive terms are to be reworked for global-only models (decoder_plan.md, D4, 2026-10-23). Stage 3 is parked
(roadmap.md, "If we fall behind").

| stage | state | date |
|---|---|---|
| 0 — plumbing (`StepOutput`, `gather_tokens`, composite loss, step collapse, dataset fields) | **done** (equivalence harness IDENTICAL against the pre-refactor baseline) | 2026-09-13 |
| 0b — probe battery: online probes, split-independence check, attention-flow control | **done**; the flow control never run on a trained attention matrix (and `_attn_matrix` is degenerate: roadmap K10) | 2026-09-16 |
| 1 — VICReg (variance, covariance, invariance; beside or instead of reconstruction) | **implemented, unverified** — smoke runs only | 2026-09-18 |
| 2 — context NCE, composition-matched negatives | **implemented, unverified** — no training run | 2026-09-18 |
| 2b — ring NCE (positive: outer ring; negative: the GNN's own neighbourhood) | **implemented, unverified** — no training run | 2026-09-21 |
| 3 — two views (NT-Xent / VICReg invariance) | not started; parked | |
| 4 — interaction-destroying negatives | not started | |
| rework for global-only models | proposed (decoder_plan.md D4) | 2026-10-23 |

### What is built (Stages 0 and 0b)

- **`StepOutput`** (`module/base/_step_output.py`) replaced the 6-tuple: `y_pred`, `y_true`, `entry_mask`, `views` (one
  `ViewOutput` per pass: `global_embedding` `[S, B, E]` padded with CLS last, `local_embedding`, `src_padding_mask`,
  `padded_node_idx`, `attn`). `padded_node_idx` is the one bridge between padded token order and batch node order; every
  auxiliary loss uses it rather than re-deriving it.
- **`gather_tokens(view)`**: the real cell tokens, CLS and padding removed, in `padded_node_idx` order. It exists so CLS (similar
  to everything) never enters a negative pool and padding (identical rows) never inflates a statistic.
- **The composite loss** (`train/aux_losses.py`): `optim.aux_loss_weights.<name>` (flat floats, so sweeps can set them); a weight
  of 0 builds nothing and runs no extra pass. Each term is an `AuxLoss` with `requires_views` and `forward(out, batch) ->
  {name: scalar}`; the module runs `max(requires_views)` passes, so a two-view term turns on two-view mode by itself. Auxiliary heads live on the module (in `module.state_dict()`, so a reload
  restores them), built at model construction.
- **One `_step(batch, mode)`** for training, validation and test.
- **Online probes** (`evaluation/online_probes.py`; CLAUDE.md "Online probes"), **split independence**
  (`tl.check_split_independence`, `dataset.group_key`: reports groups whose cells straddle splits; unset means no claim), and the
  **attention-flow control** (`evaluation/flow_control.py`: null pairs scored on |flow|, signal pairs signed; `null_percentile`
  uses a midrank, since flow matrices are sparse and a plain `<=` reads a clean zero as violated).
- **The orientation trap, pinned by a test:** `compute_hierarchical_net_flow` inverts the sign, so A attending to B is flow
  *from* B; `"senderA>receiverA"` claims information flows senderA → receiverA while attention runs the other way.
- **`dataset.extra_obs_keys`** attaches any obs column under its own name; reserved names (e.g. `mask`) are rejected.
  `tl.warn_missing_categories` warns per (column, split) on categories a split lacks.
- **The gate:** `scripts/equivalence_harness.py` runs short deterministic trainings over every `_common_step` and loss path and
  compares metric histories exactly. Read its docstring before the next refactor.

Pre-existing issues found while building Stage 0, not fixed: `TrainingPlan(lr_scheduler=None)` returns a scheduler dict
Lightning rejects; `prepare_geome_dataset` reads `type_gex_embedding` unconditionally, so a local-only config raises.

### Decided

- Node-level (cell token) contrast, not graph-level: one graph per slide gives no batch of graphs to contrast.
- Augmentation is transcriptional, at the input. **Topological augmentation is excluded** — invariance to adjacency is
  self-defeating for a model that measures interaction. Structure destruction appears in Stage 4 as a *negative*.
- No hard-negative mining: the hardness ranking and the false-negative ranking are the same ranking. τ on the high side (~0.5).
- Positives are never defined by transcriptomic similarity — a sender and a receiver in one programme are dissimilar cells.
- A corruption is only useful if the encoder can *see* it. `GCNConv`'s symmetric aggregation is near-invariant to permuting
  expression within a neighbourhood, so Stage 4 permutes globally within cell type. Check every proposed negative against the
  encoder's invariances: a no-op corruption gives a flat loss and looks wired.

### Open questions

- Expander width for VICReg: 256 is a guess, unswept.
- Whether the SimCSE floor (two passes, dropout-only) is strong enough to change the ordering of Stages 1–3.
- What the rework for global-only models keeps: the ring and context contrasts are defined by the GCN's reach.

## Design principles

1. **Every term is optional and weighted.** A weight of `0.0` means the term is never instantiated, no extra forward pass is run,
   and the run is byte-identical to today.
2. **Losses consume a named structure, not a positional tuple.** Adding a loss must not change a function signature.
3. **Augment at the input, contrast at the projected embedding.** The corruption enters `batch.x` before the GEX embedding so it
   propagates through both components; the contrastive loss reads a projection head discarded at inference.
4. **Condition never enters the objective.** Not in the positive definition, not in the negative restriction. It stays an
   evaluation-only variable so a probe on it means something.
5. **Each stage is independently shippable and measurable.**

Projection heads are not shared between loss families: VICReg's `expander` (`n_embed` → 256+, not normalised; VICReg Table 8:
L2 costs 3.5%), NT-Xent's and context NCE's `projector` (→ 64, L2-normalised; SimCLR Table 5). With `n_embed = 16` the covariance
term has only 120 off-diagonal entries, so the expander is not decoration (VICReg Table 12).

## Stage 0b — the yardstick

The battery every run is read with:

| probe | wanted direction | why |
|---|---|---|
| interaction-program membership | **up** | the actual objective |
| niche | up | expected, but not sufficient — composition is the easy part |
| **slide identity** | **down** | the batch-effect alarm. If this rises, the contrastive term learned the slide signature |
| condition | up | only interpretable because condition never entered the objective |
| reconstruction (`val_masked_*`) | flat or up | the hybrid must not cost accuracy |

**Restrict probes to masked cells.** An unmasked cell's own expression is in the encoder input, so a linear readout recovers the
target by inverting the embedding: on synth_data_0 the structure-free `noise_00` probed at R2 0.33 unrestricted, ~0.02 restricted.
`probe.masked_cells_only` stays True. The online probe fits on train and scores on val; `linear_probing.py` does the donor-grouped
CV for the post-hoc version.

Plus a synthetic control: a sender next to a receiver with no interaction between them; net attention flow on that pair must stay
flat while genuine sender→receiver pairs strengthen. (synth_data_0's `medulla` niche was built for it; on synth_spot the planted
edges and their distance-matched cells play this role.)

## What the probes cannot tell you — measured 2026-09-18

**On `synth_data_0`, a value-reconstruction probe cannot show that the global component is needed, for any target.** Every
long-range target is spatially smooth, and every smooth target is locally readable: the correlation of a target with the mean of
its own ≤30 µm neighbourhood is 0.81 for `int_long`, 0.99-1.00 for the distance and hub fields, against a 0.46 floor
(`noise_00`, shared library effects). Consequences:

1. **Predicting a long-range gene and detecting the interaction that produced it are different claims.** A probe gap says
   nothing about range.
2. **`hub_response` / `dist_to_hub` are not the fix** — smoother still, worse discriminators (measured and retracted; do not
   re-suggest).
3. **What tests it instead:** an ablation (local vs combined, long-range mask on vs off), and an attribution question (does
   influence run from the distant sender, against a matched null).
4. **A benchmark where it is falsifiable** needs a long-range target that is **non-smooth**: dependence on a specific, rare,
   distant source rather than a gradient. synth_spot's programmes and programme masking were built for this.

## Stage 1 — VICReg

Variance and covariance need one pass and no pairs; the invariance term needs two views, which dropout supplies (the SimCSE floor:
`_forward_views` runs `_common_step` once per view).

| `optim.loss` | `vicreg_lambda` | what it is | passes |
|---|---|---|---|
| a criterion | 0 | collapse regulariser beside reconstruction | 1 |
| a criterion | > 0 | the hybrid | 2 |
| `none` | > 0 | VICReg as the whole objective | 2 |

- **The variance hinge is per slide, then averaged.** A batch-level hinge can be met by between-slide variance (the batch effect)
  while every cell within a slide collapses. The covariance term stays batch-level.
- Weights: VICReg's λ=μ=25, ν=1 (`optim.contrastive.vicreg_*`), scaled against reconstruction by `optim.aux_loss_weights.vicreg`.
- **`<mode>_loss` logs what is optimised**, not only the reconstruction half (under `optim.loss: none` it was a constant zero,
  and `val_loss` drives EarlyStopping).
- **Dropout-only views vanish under evaluation:** `val_vicreg_inv` is identically 0 in `eval()`. With every encoder dropout at 0
  the views are identical in training too (`build_aux_losses` warns).
- **`optim.loss: none` leaves every decoder untrained** (the terms read tokens upstream of the decoders), so the run has no gene
  loadings — the interpretability substrate. If a contrastive-only run is wanted: freeze the encoder and fit the decoder in a
  second pass (not built).
- **Gate:** reconstruction holds, per-dimension std stops shrinking (log it every epoch), probes move the right way. Smoke runs:
  the variance term starts near 15 of 25 and dominates; `vicreg_cov` rose. The λ/μ/ν balance is the first thing to sweep.

## Stage 2 — context contrast with composition-matched negatives

One forward pass, node-level, aimed at *interaction* rather than *composition*. For anchor `i`: `z_i = projector(token_i)`; the
positive `c⁺_i` the pooled projection of `i`'s k-hop neighbours (excluding `i`); negatives other cells' actual k-hop
neighbourhoods on the **same slide**, selected to match the cell-type histogram of `c⁺_i`; InfoNCE over them. With the histogram
held fixed, composition carries no information; the model wins only by encoding which states are present and how they are
arranged.

Decisions (`train/context_nce.py`):
- **Negatives are neighbourhoods built by the same operator as the positive**, not scattered cell sets: a scattered set pools
  toward the slide mean and is separable from a contiguous patch without any interaction learned. `negative_context: scattered`
  is that ablation.
- **A per-slide bank** of `n_candidates` contexts (must exceed `n_negatives`, or matching selects nothing; the constructor
  refuses).
- **Three rejection rules:** a candidate overlapping the anchor's context, containing the anchor, or empty.
- **`anchors_masked_only: true`:** the GCN puts an unmasked anchor's own expression inside its neighbours' embeddings, so the
  positive would be identifiable by self-detection.
- **The index mapping** from `edge_index` (batch node order) to tokens (kept order) is built once from `padded_node_idx`; edges
  with an endpoint dropped by padding are removed.
- **Diagnostics:** `_context_nce_clean` (fraction of anchors whose negatives were all genuinely non-overlapping; low = false
  negatives in the denominator, with a normal-looking loss), `_context_nce_hist_gap` (L1 between positive and negative histograms;
  read against a `match_composition: false` run), `_context_nce_acc` (1.0 from step 0 means a leak).
- Knobs: `temperature` ≈ 0.5, `n_negatives` 8–32, `context_hops` matched to the local component's layers, `n_anchors` (e.g. 512 per
  slide).
- **Gate:** the no-interaction control stays flat; the slide-identity probe stays flat.

## Stage 2b — scale-matched pairing: the ring

The proposal was to give each component the pairing of its scale — the local one "same neighbourhood", the global one "two
distant cells on the same slide" with cross-slide negatives. **The distant-same-slide positive was dropped before
implementation:** its exact optimum is a slide classifier, and mixing within-slide negatives in does not cancel that — the optimum
becomes slide identity *plus* a near/far component, and NT-Xent's similarity weighting teaches the slide first. Cross-slide
negatives buy nothing the within-slide ones do not carry. If collapse prevention is the motive, Stage 1's variance and covariance
terms give it without rewarding slide identity.

**What shipped** (`train/ring_nce.py`, `RingNCE`): the positive is the anchor's pooled **hop annulus** (beyond the attention
mask's reach, within `ring_outer_hops`), the negative its own pooled **inner neighbourhood** (what the GCN aggregates);
`ring_n_negatives > 1` adds other anchors' cores. It is `M = 1 − A` restated as an objective: the mask forbids attending inside the
GNN's reach; the term asks the embedding to describe what lies outside it.
- **No slide shortcut** (both sides from the same anchor) and **no smooth-field shortcut** (the annulus is centred on the anchor).
- **The inner radius is derived** (`ring_inner_hops: 0` → `tl.local_reach_hops`, the mask's reach); an explicit different value
  warns.
- **`_ring_nce_gap`** (cos to ring minus cos to core) starts negative and is supposed to; the question is whether it rises.
  `_ring_nce_kept` is the share of anchors with both a ring and a core.
- **`ring_outer_hops` has no safe default** (a hop is a density-dependent distance); the first thing to sweep.
- Condition stays out: cross-slide positives would either span conditions or put condition in the objective. Batch effect is
  handled by neutrality and measured by the slide-ID probe.

## Stage 3 — two views (parked)

Costs 2× compute; only worth it if Stages 1–2 move the probes.
- **`ViewSampler`** (`tl/augment.py`): composable input-space ops with their own probabilities (`gene_mask`, `expression_noise`,
  `neighbour_mask`), applied to `batch.x` before the GEX embedding.
- **Both views from the same sampler with independent draws** — neither is the clean original; an asymmetric pair aligns on
  "corrupted vs clean" and reintroduces the identity shortcut. The reconstruction term reads one of the views.
- **The anchor's own features are corrupted in both views** (hard invariant), or the pair aligns on the cell's own transcriptome.
- **The mask draw moves into the module** (per step, per view) when `requires_views > 1`; today `NodeMaskResampleCallback` redraws
  once per epoch, so two passes differ only by dropout (the SimCSE baseline everything has to beat). Per-step draws change the
  reconstruction term's schedule too: run that control first.
- **The loss:** NT-Xent over `(p₁[i], p₂[i])`, negatives all other tokens on the same slide minus the anchor's k-hop neighbours;
  subsample anchors, not the pool. VICReg's invariance term is the negative-free alternative.

## Stage 4 — interaction-destroying negatives (optional)

A pass over a copy whose *composition is preserved and interaction destroyed*: expression vectors permuted **within cell type**
across the slide. Every neighbourhood keeps its composition; the correlation between a cell's state and its neighbours' is gone.
Global within-type, not per-neighbourhood (a symmetric aggregator is near-blind to the latter), and not across all cells (that
destroys composition too, so the model could win on niche identity).

## Config

```yaml
optim:
  loss: SCELoss                        # the reconstruction criterion; `none` for contrastive-only
  aux_loss_weights:                    # all 0.0 by default → nothing is built
    vicreg: 0.0                        # outer scale; vicreg_lambda / _mu / _nu weigh its three terms
    context_nce: 0.0
    ring_nce: 0.0
  contrastive:
    temperature: 0.5
    negatives: within_slide
    exclude_khop: 2
    projector_dims: [64]
    expander_dims: [256]
    vicreg_lambda: 25.0                # invariance (0 = one pass, no views)
    vicreg_mu: 25.0                    # variance
    vicreg_nu: 1.0                     # covariance
    context_hops: 2
    n_anchors: 512                     # 0 = all
    n_negatives: 16
    n_candidates: 256                  # per-slide bank; must exceed n_negatives
    negative_context: neighbourhood    # neighbourhood | scattered (the ablation)
    match_composition: true            # needs dataset.celltype_key
    anchors_masked_only: true
    ring_outer_hops: 6
    ring_inner_hops: 0                 # 0 = the mask's reach
    ring_n_negatives: 1
```

The exact key names are in `config/optim_config.py`.

## Invariants checklist

Things that fail silently rather than loudly:

- [ ] CLS token (last sequence position) excluded from every token gather
- [ ] padded positions excluded from every statistic and every negative pool
- [ ] negatives restricted to the same slide
- [ ] anchor's k-hop neighbours excluded from the denominator
- [ ] both views drawn independently from the *same* sampler — neither is a clean pass
- [ ] anchor's own features corrupted in **both** views
- [ ] VICReg variance hinge computed **per slide**, then averaged
- [ ] condition absent from positive definition and negative restriction
- [ ] projector/expander discarded at inference; the embedding the linear decoder reads is never directly constrained by a
      contrastive term
- [ ] `celltype` used only as a negative-sampling stratifier, never as a target
- [ ] a negative context is built by the *same operator* as the positive, so spatial coherence is held fixed alongside composition
- [ ] a candidate that *contains the anchor* is rejected, not only one that overlaps the anchor's context
- [ ] the candidate bank is larger than `n_negatives`, or the composition match selects nothing
- [ ] anchors restricted to masked cells
- [ ] `_context_nce_clean` read before any conclusion
- [ ] `_context_nce_hist_gap` compared against a `match_composition: false` run
- [ ] a proposed corruption is actually visible to the encoder (the `GCNConv` note above)
- [ ] the ring's inner radius equals the attention mask's reach — derived, not configured
- [ ] `_ring_nce_gap` read as a *trend*, not a sign
- [ ] no cross-slide positive or negative unless donor and condition are held fixed across the pair
