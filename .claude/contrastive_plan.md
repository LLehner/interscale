# Staged plan: contrastive objectives for the global module

Working document for adding contrastive (and other auxiliary) losses to the global
component. Read [`background.md`](background.md) first for why the local/global split
exists — the design below is constrained by the fact that the global embedding is also
the interpretability substrate (gene loadings, net attention flow), not just a
prediction intermediate.

## Status

Last updated 2026-09-13. Update this table in the same commit as the work it describes.
"Implemented, unverified" is a real state — a stage is only `done` when something external
says so (a test, a reproduced number, a run that was actually looked at).

Work happens on the `contrastive-learning` branch.

| stage | state | verified by | date |
|---|---|---|---|
| 0 — plumbing (`StepOutput`, `gather_tokens`, composite loss, step collapse, dataset fields) | **done** | 119 tests pass; `scripts/equivalence_harness.py` reports IDENTICAL against the pre-refactor baseline | 2026-09-13 |
| 0b — probe battery | not started (deliberately skipped for now) | | |
| 1 — VICReg var/cov, no views | not started | | |
| 2 — context NCE, composition-matched negatives | not started | | |
| 3 — two views (NT-Xent / VICReg invariance) | not started | | |
| 4 — interaction-destroying negatives | not started | | |

### The gate

`scripts/equivalence_harness.py` runs four short deterministic trainings covering every
`_common_step` implementation and all three loss paths, and compares per-epoch metric histories
exactly. Every Stage 0 commit was checked against a baseline captured before the first change.
Read its docstring before the next refactor; the tests passed at every intermediate point too,
so they are not the thing that tells you nothing moved.

### How Stage 0 differed from this plan

- **No transitional `__iter__` on `StepOutput`.** The plan allowed one so the four producers
  could migrate one at a time. All four, and all three consumers, turned out to be a single
  commit's worth of work, so the shim was never needed — and the plan already said to delete it
  at exactly that point.
- **The step collapse (0.4) was done before the composite-loss scaffolding (0.3)**, since
  otherwise the aux wiring would have been written three times and two copies deleted.
- **Projection heads (0.5) are deferred to Stage 1.** The extension point exists — an `AuxLoss`
  owns its own head, and `configure_optimizers` collects it — but a shared head-builder with no
  consumer would be untested dead code. Stage 1's expander is its first user.

**Decided so far**

- Node-level (cell token) contrast, not graph-level: with one graph per slide there is no
  meaningful batch of graphs to contrast across.
- Augmentation is transcriptional, applied at the input. **Topological augmentation is
  deliberately excluded** — making the model invariant to which cells are adjacent is
  self-defeating for a model that measures interaction. Structure destruction appears in
  Stage 4 as a *negative*, with the opposite sign.
- No hard-negative mining: here the hardness ranking and the false-negative ranking are the
  same ranking. τ on the high side (~0.5) instead.
- Positives are never defined by transcriptomic similarity — a sender and a receiver in the
  same program are different cell types with dissimilar transcriptomes.

**Open questions**

- Expander width for VICReg: 256 is a guess against `n_embed = 16`, unswept.
- Source of the `celltype` stratifier for Stage 2 — existing annotation vs. a stored
  expression clustering. `cfg.dataset.celltype_key` now attaches an existing annotation.
- Whether the SimCSE floor (two passes, dropout-only) is strong enough to change the
  ordering of Stages 1–3.
- **Auxiliary-loss heads are not persisted.** They live on the `TrainingPlan`, so
  `BaseModel.save` (which writes only `module.state_dict()`) does not store them. Correct for
  inference, where the head is discarded anyway, but resuming a run would silently reinitialise
  the expander. Decide in Stage 1, when there is a real parameter at stake.

**Found while implementing Stage 0** (all pre-existing, none fixed here)

- `val` never logged `combined_loss` although `train` and `test` did, and `test` logged
  `kl_loss` without `sync_dist` while logging everything else with it. Both are reproduced in
  `_step` so the refactor changed no number; each deserves its own `Fix` commit.
- `TrainingPlan(lr_scheduler=None)` returns a scheduler dict whose `"scheduler"` entry is
  `None`, which Lightning rejects — so that documented option cannot actually be used.
- `prepare_geome_dataset` reads `cfg.model.global_component.parameters.type_gex_embedding`
  unconditionally, so a local-only config (no global component name) raises `AttributeError`
  before any training starts.
- Categoricals were being narrowed three times before one-hot encoding (split subsetting,
  `transforms.Subset`, then `preserve_categories` preserving only what was left). Fixed for the
  new optional fields via `_RestoreCategories`; the same exposure exists for a classification
  `prediction_obs` whose label is absent from a split.

## Design principles

1. **Every term is optional and weighted.** A weight of `0.0` means the term is never
   instantiated, no extra forward pass is run, and the run is byte-identical to today.
2. **Losses consume a named structure, not a positional tuple.** Adding the sixth loss
   must not change a single function signature.
3. **Augment at the input, contrast at the projected embedding.** The corruption enters
   `batch.x` before the GEX embedding (where masking already sits) so it propagates
   through both components; the contrastive loss reads a projection head that is
   discarded at inference.
4. **Condition never enters the objective.** Not in the positive definition, not in the
   negative restriction. It stays an evaluation-only variable so a probe on it means
   something.
5. **Each stage is independently shippable and measurable.** No stage requires the next
   one to be worth running.

---

## Stage 0 — plumbing (no behaviour change)

Nothing here adds a loss. The point is that Stages 1–4 then become additive.

### 0.1 `StepOutput` replaces the 6-tuple

`_common_step` currently returns
`(local_emb, global_emb, y_pred, y_true, attn, entry_mask)`, unpacked positionally at
[`_trainingplans.py:434`](../src/interscale/train/_trainingplans.py#L434),
`:506` and `:570`. Four modules implement it: `LocalModule`, `GlobalModule`,
`CombinedModule`, `DualDecoderCombinedModule`.

```python
# src/interscale/module/base/_step_output.py
@dataclass
class ViewOutput:
    """One forward pass over one corrupted copy of the batch."""
    global_embedding: torch.Tensor      # [S, B, E] padded, CLS at the LAST position
    local_embedding: torch.Tensor | None
    src_padding_mask: torch.Tensor      # [B, S]  True = pad
    padded_node_idx: torch.Tensor       # [N_kept] batch-global node id of each kept token
    attn: torch.Tensor | None

@dataclass
class StepOutput:
    y_pred: torch.Tensor
    y_true: torch.Tensor
    entry_mask: torch.Tensor | None
    views: list[ViewOutput]             # len 1 today; len 2+ from Stage 3

    @property
    def view(self) -> ViewOutput:       # the reconstruction view
        return self.views[0]

    def __iter__(self):                 # TRANSITIONAL — see note
        v = self.view
        yield from (v.local_embedding, v.global_embedding, self.y_pred,
                    self.y_true, v.attn, self.entry_mask)
```

`__iter__` exists so the four producers can be migrated one commit at a time while the
three consumers keep working. **Delete it once all four return `StepOutput`** — a
dataclass that is also a tuple is a trap waiting for someone to add a field in the
middle.

**The important new field is `padded_node_idx`.** It is already computed inside
`_process_batch_for_metrics` and then thrown away, but it is the only bridge between the
two coordinate systems in play:

- `global_embedding` is `[S, B, E]` in *padded* order,
- `local_embedding`, `batch.x`, `batch.batch` and `edge_index` are in *batch node* order.

Every auxiliary loss needs to go from a token to its slide (`batch.batch[padded_node_idx]`)
and to its spatial neighbours (`edge_index`). Without this field on the output, each loss
would re-derive it and they would drift apart.

### 0.2 `gather_tokens` helper

`predict()` already does the padded → flat conversion for the node-level branch
([`_base_global_module.py:279-284`](../src/interscale/module/base/_base_global_module.py#L279)).
Extract it so every loss uses the same one:

```python
def gather_tokens(view: ViewOutput) -> torch.Tensor:
    """[N_kept, E] real cell tokens, CLS and padding removed, in padded_node_idx order."""
    h = view.global_embedding[:-1]                      # drop CLS (last position)
    h = h.permute(1, 0, 2)                              # [B, S, E]
    return h[~view.src_padding_mask[:, :-1]]            # [N_kept, E]
```

Two failure modes this exists to prevent, both silent:

- **CLS leaking into the negative pool.** CLS attends to every cell, so it is similar to
  everything; as a negative it produces a large, meaningless repulsive gradient for every
  anchor.
- **Padding leaking into the statistics.** Padded rows are identical to each other, so
  they inflate covariance and fake a variance the VICReg hinge would then report as
  satisfied.

### 0.3 Composite loss

Config (full schema in the appendix):

```yaml
optim:
  loss: SCELoss                  # unchanged — the reconstruction criterion
  aux_loss_weights:              # all 0.0 by default → nothing is built
    vicreg_var: 0.0
    vicreg_cov: 0.0
    context_nce: 0.0
    nt_xent: 0.0
```

Flat float keys rather than a list of nested nodes, so `main_sweep.py` can set
`optim.aux_loss_weights.context_nce` through `merge_from_list` without special handling.

Uniform interface — this is what makes point 2 true:

```python
class AuxLoss(nn.Module):
    requires_views: int = 1
    def forward(self, out: StepOutput, batch) -> dict[str, torch.Tensor]:
        """Named scalar terms. The plan applies the weight, sums, and logs each by name."""
```

`build_aux_losses(cfg) -> nn.ModuleDict` instantiates only the terms with weight > 0. The
module then runs `max(t.requires_views for t in aux)` forward passes — so enabling a
two-view loss turns on two-view mode automatically. There is no second switch to forget
to set, which is the kind of mistake that produces a run that looks fine and trains the
wrong thing.

### 0.4 Collapse the three duplicated step bodies

`training_step`, `validation_step` and `test_step` are ~60 near-identical lines each,
including three copies of the `compute_separate_losses` logging block. Fold them into
`_step(batch, mode)`. Do this in Stage 0, before there is a fourth thing to keep in sync.

At the same time, retire the `if self.loss_type == "SCE_EntropyATT_Loss"` special case at
[`_trainingplans.py:351`](../src/interscale/train/_trainingplans.py#L351): the attention
entropy term is an auxiliary loss in the new sense (it reads `attn`, not `y_pred`), so it
becomes `attn_entropy` in `aux_loss_weights` and the branch disappears. That is the
migration that proves the interface works.

### 0.5 Projection heads

Built lazily, only when a term needs one, and **not shared between loss families**:

| head | used by | output | normalised? |
|---|---|---|---|
| `expander` | VICReg | `n_embed` → 256+ | no (VICReg Table 8: L2 costs 3.5%) |
| `projector` | NT-Xent, context NCE | `n_embed` → 64 | yes, L2 (SimCLR Table 5: 64.4 vs 57.2) |

With `cfg.model.n_embed = 16`, VICReg's covariance term has only 120 off-diagonal entries
and its variance hinge acts on 16 dimensions that the decoder also needs for
reconstruction. The paper's own sweep (Table 12: 256-d → 55.9%, 8192-d → 68.6%) says this
matters, so the expander is not optional decoration.

New parameters appear as new state-dict keys; `BaseModel.load` uses `strict=False`, so old
checkpoints still load.

### 0.6 Fields the data pipeline must carry

`prepare_geome_dataset` currently attaches only `x`, `edge_index`, `obs_names` (+`y`,
`embeddings`). Add:

- `pos` — spatial coordinates. Not needed until Stage 4, but free to add now.
- `slide` / `donor` / `condition` as graph-level attributes. `slide` gates the negative
  pool; the other two are for the probe battery and for *auditing* that condition never
  reaches the objective.
- `celltype` (or a stored expression clustering) — used **only** as a stratifier by the
  composition-matched negative sampler in Stage 2, never as a target.

---

## Stage 0b — the yardstick (do this before Stage 1)

You cannot evaluate Stage 1 without it. A script over
[`evaluation/linear_probing.py`](../src/interscale/evaluation/linear_probing.py) (which
already does donor-grouped CV) reporting, for every run:

| probe | wanted direction | why |
|---|---|---|
| interaction-program membership | **up** | the actual objective |
| niche | up | expected, but not sufficient — composition is the easy part |
| **slide identity** | **down** | the batch-effect alarm. If this rises, the contrastive term learned the slide signature |
| condition | up | only interpretable because condition never entered the objective |
| reconstruction (`val_masked_*`) | flat or up | the hybrid must not cost accuracy |

Plus the synthetic control: in `medulla`, `senderB` sits next to `receiverA` with no
interaction between them. Net attention flow on that pair must stay flat while genuine
sender→receiver pairs strengthen. That is the one measurement that distinguishes "learned
composition" from "learned interaction", and it is the reason the synthetic generator
already has that niche.

---

## Stage 1 — VICReg variance + covariance, no views

The cheapest real term: one forward pass, no pairing, no negatives, so none of the
false-negative problems apply.

```python
class VICRegReg(AuxLoss):
    requires_views = 1
    def forward(self, out, batch):
        z = self.expander(gather_tokens(out.view))          # [N_kept, D]
        slide = batch.batch[out.view.padded_node_idx]       # [N_kept]
        return {"vicreg_var": self._variance(z, slide), "vicreg_cov": self._covariance(z)}
```

**Compute the variance hinge per slide and average.** A batch-level hinge can be satisfied
entirely by *between*-slide variance — i.e. by the batch effect — while every cell within a
slide collapses to the same point. The within-slide variance is the one that means "cells
in this tissue differ from each other". The covariance term can stay batch-level; it is a
marginal statistic and mixing slides is harmless there.

Weights: VICReg uses λ=μ=25, ν=1, so relative to a reconstruction weight of 1.0 the
starting point is `vicreg_var: 1.0`, `vicreg_cov: 0.04`. Their Table 7 shows Inv+Var alone
reaches 57.5 and Inv+Var+Cov 68.6 — both terms earn their place.

**Gate:** reconstruction metrics hold, embedding per-dimension std stops shrinking, probes
move the right way. Log per-dimension std every epoch (VICReg Fig. 4 uses exactly this to
catch slow collapse).

---

## Stage 2 — context contrast with composition-matched negatives

One forward pass, node-level, and the term that actually targets *interaction* rather than
*composition*.

For each anchor cell `i`:

- `z_i` = `projector(token_i)`
- `c⁺_i` = pooled projection of `i`'s k-hop neighbours from `edge_index`, excluding `i`
- `c⁻_i,k` = pooled projection of a random cell set from the **same slide**, sampled to
  match the cell-type histogram of `c⁺_i`
- `L = -log exp(sim(z_i, c⁺_i)/τ) / (exp(sim(z_i, c⁺_i)/τ) + Σ_k exp(sim(z_i, c⁻_i,k)/τ))`

Because the type histogram is held fixed across positive and negative, composition carries
zero information about which is which. The model can only win by encoding *which specific
states are present and how they are arranged*.

### The index mapping that will break if you write it twice

`edge_index` is in batch node order; tokens are in kept order. Build the inverse once:

```python
inv = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=dev)
inv[view.padded_node_idx] = torch.arange(len(view.padded_node_idx), device=dev)
src, dst = edge_index
keep = (inv[src] >= 0) & (inv[dst] >= 0)     # both endpoints survived padding
edges_tok = torch.stack([inv[src[keep]], inv[dst[keep]]])
```

Cells whose neighbours were dropped by padding end up with truncated contexts. With
sequences that fit inside `max_seq_len` nothing is dropped and `keep` is all-True — but the
guard has to be there for the day someone lowers `max_seq_len`.

### Knobs

`temperature` ≈ 0.5 (not 0.1 — see below), `n_negatives` 8–32, `k_hops` matched to the
local component's `num_layers`, anchors subsampled (e.g. 512 per slide) to bound cost.

**On temperature.** NT-Xent's softmax denominator already weights negatives by similarity,
which *is* soft hard-negative mining (SimCLR Table 2 names this as why it beats margin and
logistic losses). Here the hardness ranking and the false-negative ranking are the same
ranking — the cells most similar to the anchor are the ones most likely to share its
interaction program. τ is the knob that flattens that weighting, so it belongs on the high
side, and the anchor's own k-hop neighbours come out of the denominator entirely.

**Gate:** the medulla control stays flat. If it moves, the term learned co-occurrence.

---

## Stage 3 — two views

The first stage that costs 2× compute. Only worth it if Stages 1–2 have moved the probes.

### 3.1 `ViewSampler` (`tl/augment.py`)

A composable list of input-space ops, each with its own probability, applied to `batch.x`
*before* the GEX embedding — generalising what `apply_mask` already does:

```yaml
contrastive:
  augment:
    gene_mask: 0.3          # independent Bernoulli draw per view
    expression_noise: 0.1
    neighbour_mask: 0.1     # corrupt the CONTEXT, not only the anchor
```

**Both branches are augmented, symmetrically.** SimCLR draws `t ~ T` and `t' ~ T` and
applies one to each branch — *neither view is the uncorrupted original*. The tempting
shortcut here is to reuse the batch that `_common_step_masking` already produced as view 1
and generate only view 2 freshly; that gives an asymmetric pair, and the model can then
align the two views on "corrupted vs. clean" rather than on content, while the clean branch
quietly reintroduces the identity shortcut. Both views must come from the same sampler with
independent draws, and the reconstruction term should read one of them rather than a third,
uncorrupted pass.

**Hard invariant, not a knob: the anchor's own features are corrupted in both views.**
Otherwise the two embeddings of cell `i` share the component that encodes `i`'s own
transcriptome, the model aligns on that, and you have trained an autoencoder with a
contrastive-shaped loss. Topological augmentation is *deliberately absent* — telling the
model to be invariant to which cells are adjacent is self-defeating for a model whose
purpose is to measure interaction. Structure destruction belongs in Stage 4 with the
opposite sign.

### 3.2 The per-epoch → per-step mask draw

`apply_mask` does not sample; it reads a `batch.mask` that the dataloader writes at setup
and `NodeMaskResampleCallback` redraws **once per epoch**. So two calls to `_common_step`
on the same batch today produce *identical* corruption, differing only by dropout.

Two consequences:

- **Free baseline.** Two passes with dropout-only differences is SimCSE's unsupervised
  setup and needs no new augmentation code. Run it first; everything else has to beat it.
- **The draw must move into the module** (per step, per view) when `requires_views > 1`.
  Keep the dataloader path when `requires_views == 1` so existing runs are untouched — and
  note that per-step draws change the effective corruption schedule for the reconstruction
  term too, so the single-view-with-per-step-draws control needs running before attributing
  anything to the contrastive term.

### 3.3 The loss

NT-Xent over `(p₁[i], p₂[i])` — index identity, since structure and token set are untouched
across views, so no alignment bookkeeping. Negatives: all other tokens **in the same
slide** (`batch.batch[padded_node_idx]` equality), minus the anchor's k-hop neighbours.

A 4300² logit matrix is ~74 MB in fp32 before gradients; subsample anchors against the full
negative pool rather than shrinking the pool. VICReg's invariance term (plain MSE, no
normalisation) is the negative-free alternative on the same two views.

---

## Stage 4 — interaction-destroying negatives (optional)

A third pass over a corrupted copy whose *composition is preserved and whose interaction is
destroyed*: permute expression vectors **within cell type** across the slide. Every
neighbourhood keeps an identical cell-type composition; what is destroyed is the
correlation between a cell's state and its neighbours' states.

Contrast with the label-free global permutation (the standard DGI corruption), which shuffles
across all cells: that destroys composition too, so the model can win on niche identity alone
and never learn interaction. The within-type restriction is the whole point.

---

## Appendix A — config schema

```yaml
optim:
  loss: SCELoss                        # unchanged: reconstruction criterion
  aux_loss_weights:
    vicreg_var: 0.0
    vicreg_cov: 0.0
    context_nce: 0.0
    nt_xent: 0.0
    attn_entropy: 0.0                  # migrated from SCE_EntropyATT_Loss
  contrastive:
    temperature: 0.5
    projector_dims: [64]
    expander_dims: [256]
    negatives: within_slide            # within_slide | within_batch
    exclude_khop: 2                    # match local_component.num_layers
    n_negatives: 16
    n_anchors: 512                     # 0 = all
    augment:
      gene_mask: 0.3
      expression_noise: 0.0
      neighbour_mask: 0.0
```

## Appendix B — files touched per stage

| stage | new | modified |
|---|---|---|
| 0 | `module/base/_step_output.py`, `train/aux_losses.py` | 4× `_common_step`, `_trainingplans.py`, `optim_config.py`, `tl/geome_utils.py` |
| 0b | `evaluation/probe_battery.py` | — |
| 1 | — | `aux_losses.py`, `_base_global_module.py` (expander) |
| 2 | — | `aux_losses.py`, `tl/geome_utils.py` (celltype field) |
| 3 | `tl/augment.py` | `_base_global_module.py` (multi-pass), `geome_dataloader.py` |
| 4 | — | `tl/augment.py`, `aux_losses.py` |

## Appendix C — invariants checklist

Things that fail silently rather than loudly:

- [ ] CLS token (last sequence position) excluded from every token gather
- [ ] padded positions excluded from every statistic and every negative pool
- [ ] negatives restricted to the same slide
- [ ] anchor's k-hop neighbours excluded from the denominator
- [ ] both views drawn independently from the *same* sampler — neither is a clean pass
- [ ] anchor's own features corrupted in **both** views
- [ ] VICReg variance hinge computed **per slide**, then averaged
- [ ] condition absent from positive definition and negative restriction
- [ ] projector/expander discarded at inference; the embedding the linear decoder reads is
      never directly constrained by a contrastive term
- [ ] `celltype` used only as a negative-sampling stratifier, never as a target

## Appendix D — pre-existing issue worth a separate look

`apply_mask` clones the batch and rewrites only `.x`
([`masking.py:178`](../src/interscale/tl/masking.py#L178)); `.embeddings` passes through
untouched. `_common_step` prefers `batch_masked.embeddings` when present
([`_base_global_module.py:310`](../src/interscale/module/base/_base_global_module.py#L310)).
So under `type_gex_embedding: "Precomputed"` the encoder receives embeddings derived from
the **uncorrupted** expression while the loss asks it to predict the masked entries of
`batch.x`. The PCA/NMF paths embed `batch_masked.x` and are unaffected. Unrelated to
contrastive learning, but it is the one configuration where input-space augmentation is a
no-op and latent-space corruption would be forced.
