# Staged plan: contrastive objectives for the global module

Working document for adding contrastive (and other auxiliary) losses to the global
component. Read [`background.md`](background.md) first for why the local/global split
exists — the design below is constrained by the fact that the global embedding is also
the interpretability substrate (gene loadings, net attention flow), not just a
prediction intermediate.

## Status

Last updated 2026-09-18. **Read section 'What the probes cannot tell you' before interpreting any probe result.** Update this table in the same commit as the work it describes.
"Implemented, unverified" is a real state — a stage is only `done` when something external
says so (a test, a reproduced number, a run that was actually looked at).

Work now happens on `main`. The `contrastive-learning` branch was fast-forwarded into it on
2026-09-14 (main was a strict ancestor, so no merge commit) because Stage 0b needs `StepOutput`
and the optional dataset fields, and there was no reason to keep those off main once Stage 0 was
verified. The branch still exists but is behind; do not commit to it.

| stage | state | verified by | date |
|---|---|---|---|
| 0 — plumbing (`StepOutput`, `gather_tokens`, composite loss, step collapse, dataset fields) | **done** | 119 tests pass; `scripts/equivalence_harness.py` reports IDENTICAL against the pre-refactor baseline | 2026-09-13 |
| 0b — probe battery | **done** — online probes (`b88b76f`), split-independence check (`ddfccdc`), attention-flow control (`cd15021`, `11a2622`) | 201 tests pass; `noise_00` reads ~0.02 R2 on a real synth_data_0 run; the flow control's sign convention is pinned against `compute_hierarchical_net_flow`. **Not yet run against a real trained attention matrix** — see below | 2026-09-16 |
| 1 — VICReg | **implemented, unverified** — all three terms, selectable beside *or instead of* reconstruction, on the local or the global embedding | 237 tests pass; equivalence harness IDENTICAL; runs end to end through `GlobalModel` and `CombinedModel` (dual and single decoder) and on a local-only GCN. **No verified training run yet**: a real run was launched and produced probe numbers, but see 'What the probes cannot tell you' — they do not settle anything | 2026-09-16 → 09-18 |
| 2 — context NCE, composition-matched negatives | **implemented, unverified** — one pass, contiguous composition-matched negatives, `scattered`/`match_composition` ablations | 275 tests pass; equivalence harness IDENTICAL across 4 cases / 12 epoch records; runs end to end through the training plan in the hybrid. **No verified training run yet** — and see the two predicted failure modes below, neither of which a falling loss would rule out | 2026-09-18 |
| 2b — scale-matched pairing (local: same-neighbourhood; global: distant-same-slide) | not started | | |
| 3 — two views (NT-Xent / VICReg invariance) | not started | | |
| 4 — interaction-destroying negatives | not started | | |

### What Stage 0b ended up being

Three pieces, all driven by config and none naming a cell type, niche or column:

* **Online probes** (`evaluation/online_probes.py`) — local-vs-global readouts during training,
  so attribution is a curve. Targets are config lists; adding one needs no code change.
* **Split independence** (`tl.check_split_independence`, `dataset.group_key`) — the "donor-grouped
  protocol", generalised. `group_key` names the unit of statistical independence (donor, patient,
  mouse, batch — *not* hardcoded as donor), and `prepare_geome_dataset` reports any group whose
  cells straddle splits. It reports rather than enforces: a straddling split is sometimes
  unavoidable, and the repair depends on study design. Unset means no check **and no claim** —
  not a false all-clear.
* **Attention-flow control** (`evaluation/flow_control.py`) — the instrument the probes cannot be.
  Null pairs (co-occurring, non-interacting) are scored on |flow|, since either direction is a
  violation; optional signal pairs are scored signed, so `"a>b"` and `"b>a"` are different
  claims. `null_percentile` uses a **midrank**, which matters: flow matrices are sparse, and under
  a plain `<=` a perfectly clean null pair sitting at 0 alongside every other zero ranks at the
  *top* of its tie group and reads as violated exactly when it passes.

**The orientation trap, now pinned by a test.** `compute_hierarchical_net_flow` deliberately
inverts the sign ("information as opposite of attention"), so A attending to B is reported as
flow *from* B. A configured `"senderA>receiverA"` therefore claims information flows senderA →
receiverA while the attention runs the other way. Getting this backwards inverts every
interpretation without changing a single magnitude — no probe, and no other test, would catch it.

**What is not verified.** The control has only been run against synthetic flow matrices and a
hand-built attention matrix, not against `_attn_matrix` from a trained model. The arithmetic and
the orientation are pinned; what a real flow matrix's `null_percentile` looks like is unknown
until Stage 2 needs it.

**Also generalised while here:** `dataset.extra_obs_keys` attaches any obs column under its own
name, so `OPTIONAL_FIELDS` now holds only the roles the code reads *by name* (`slide` for negative
sampling, `group` for the split check, `celltype` for the flow control). A new probe target or
stratifier is a config line. Reserved names are rejected — an obs column called `mask` would
otherwise replace the corruption mask with a label and train against its own annotation silently.

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
- A corruption is only useful if the architecture can *see* it. `GCNConv` aggregates
  `Σ_j a_ij W x_j` with `a_ij = 1/√(d_i d_j)`, so permuting expression *within* a neighbourhood is
  invariant up to degree differences — and on a near-regular kNN graph that is invisible. The
  Stage 4 permutation is therefore global-within-cell-type (which changes each neighbourhood's
  vector multiset), never within-neighbourhood. Check every proposed negative against the
  encoder's invariances before building it: a no-op corruption gives a flat loss and looks wired.

**Open questions**

- Expander width for VICReg: 256 is a guess against `n_embed = 16`, unswept.
- Source of the `celltype` stratifier for Stage 2 — existing annotation vs. a stored
  expression clustering. `cfg.dataset.celltype_key` now attaches an existing annotation.
- Whether the SimCSE floor (two passes, dropout-only) is strong enough to change the
  ordering of Stages 1–3.
- ~~Auxiliary-loss heads are not in the `.pt`.~~ **Resolved** — see below.

**Resolved during Stage 0**

- **Auxiliary-loss heads now live on the module, not the training plan.** They were outside
  `module.state_dict()`, which is all `BaseModel.save` writes — so the `.pt` carried 0 aux keys
  while Lightning's `.ckpt` carried them, and `BaseModel.load` reads the `.pt`. Harmless at
  inference, where the head is discarded anyway, but a resumed run silently reinitialised the
  expander: no error, just a loss jump. `BaseModel._attach_aux_losses` now builds the composite
  at model construction — early enough for `load_state_dict(strict=False)` to reach it — and
  `TrainingPlan.aux_losses` is a read-only view of the module's. Verified by a save/reload round
  trip restoring a trained head, and in both backward-compatible directions (a checkpoint with no
  aux keys, and aux keys loaded into a model with no terms).

  **The cost is cosmetic and worth stating: a training-only object now hangs off the model.** The
  terms and their heads are part of `module.state_dict()` and show up in a parameter count, while
  being unused at inference. That was the price of having one placement answer all three
  questions — optimiser, checkpoint, reload — instead of three mechanisms that can disagree.

**Found while implementing Stage 0** (pre-existing)

- `val` never logged `combined_loss` although `train` and `test` did, and `test` logged
  `kl_loss` without `sync_dist` while logging everything else with it. Both are reproduced in
  `_step` so the refactor changed no number; each deserves its own `Fix` commit.
- `TrainingPlan(lr_scheduler=None)` returns a scheduler dict whose `"scheduler"` entry is
  `None`, which Lightning rejects — so that documented option cannot actually be used.
- `prepare_geome_dataset` reads `cfg.model.global_component.parameters.type_gex_embedding`
  unconditionally, so a local-only config (no global component name) raises `AttributeError`
  before any training starts.
- Categoricals are narrowed three times before one-hot encoding — split subsetting drops unused
  categories on the *view*, `transforms.Subset` drops them again with an empty `key_value`, and
  only then does `preserve_categories` run, preserving what is left rather than what was there.
  Fixed for the new optional fields via `_RestoreCategories`, which had no other guard.

  A classification `prediction_obs` missing a label from one split **is exposed to the same
  narrowing but fails loudly rather than silently**, so it needs no fix: `n_output` comes from
  `summary_stats["n_prediction_obs"]` on the *full* object while `y` is built per split, so a
  dropped category is a width mismatch and `_compute_and_log_metrics` asserts. Verified by
  running it (4 classes, `d` absent from train and `b` from val: both splits width 3 with column
  1 meaning `b` in one and `c` in the other, and the run raises before any of that is used).
  Note this is an *accidental* guard — it holds only because the two numbers come from different
  objects — and the message, "y_true and y_pred must have the same shape", names nothing that
  would lead anyone to a missing cell type. **`tl.warn_missing_categories`, called from
  `prepare_geome_dataset`, now covers both cases**: one warning per (column, split) naming the
  absent categories, and saying either which failure it is about to cause (`prediction_obs`) or
  that the encoding is fine but the data is empty there (the optional annotations).

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

**Status: partly built.** [`evaluation/online_probes.py`](../src/interscale/evaluation/online_probes.py)
now runs probes *during* training as a Lightning callback, gated on `cfg.probe.use`. It reads the
same target out of the local and the global embedding with one readout, fit on train and scored on
val, and logs each pair as flat `val_probe_<target>_<metric>_<embedding>` scalars plus a wandb
chart carrying both lines. Targets are any attached annotation (`probe.classification_targets`,
so `slide` and `condition` are available once their `dataset.*_key` is set) and any gene
(`probe.regression_genes`).

What that covers and what it does not:

- **Covered**: interaction-program membership (as per-gene regression on `int_short`/`int_mid`/`int_long`),
  slide identity and condition (as categorical probes), reconstruction (already logged).
- **Not covered**: `niche` is not in `tl.geome_utils.OPTIONAL_FIELDS`, so it cannot be attached yet.
  The **net-attention-flow control on the `medulla` `senderB`→`receiverA` pair is not implemented at
  all**, and it is the measurement that separates "learned composition" from "learned interaction" —
  Stage 1 cannot be called evaluated without it.
- **Different protocol**: the online probe fits on train and scores on val. It does *not* do the
  donor-grouped CV that `linear_probing.py` does, so its absolute numbers are not comparable with
  that script's. Use `linear_probing.py` for the post-hoc, publishable version and the online probe
  for watching a run.

One finding from building it, which applies to any probe added here: **restrict probes to masked
cells**. An unmasked cell's own expression is in the encoder input, so a linear readout recovers
the target by inverting the embedding. On synth_data_0 at `mask_percentage` 0.3 the structure-free
`noise_00` control probed at **R2 0.33 from both embeddings**; restricted to masked cells it sits
at **~0.02**. A negative control that does not read as negative invalidates every other number in
the battery. `probe.masked_cells_only` defaults to True and should stay there.

The full battery, from the original plan — a script over
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

### How Stage 1 differed from this plan

The plan scoped Stage 1 as *variance and covariance only, no views*. It shipped with all three
terms, because "VICReg selectable instead of the reconstruction loss" needs the invariance term —
and invariance needs two views, which was Stage 3 plumbing. What made that affordable was that
**dropout already makes two passes differ**: `_forward_views` runs `_common_step` once per view
and the encoder's own stochasticity supplies the difference, so the SimCSE floor arrived for free
and Stage 3's view sampler now only has to replace *how* the views differ, not build the
machinery.

Three configurations, all from one term:

| `optim.loss` | `vicreg_lambda` | what it is | passes |
|---|---|---|---|
| a criterion | 0 | collapse regulariser beside reconstruction — Stage 1 as planned | 1 |
| a criterion | > 0 | the hybrid | 2 |
| `none` | > 0 | VICReg as the whole objective | 2 |

**Two things found while building it, both now fixed and tested:**

* `<mode>_loss` logged only the reconstruction half, not what was actually optimised. Harmless
  until now; fatal under `optim.loss: none`, where it is a constant zero — and `val_loss` is what
  EarlyStopping and ModelCheckpoint monitor for a regression run, so every such run would have
  stopped at `patience` with a flat curve.
* **Dropout-only views vanish under evaluation.** `validation_step` runs in `eval()`, where the
  encoder is deterministic, so the two views coincide exactly and `val_vicreg_inv` is identically
  0.0 — confirmed on a real run. It means "no stochasticity in eval", not "the views agree". The
  Stage 3 sampler corrupts the *input*, which does not depend on training mode, and fixes it.

**Two more findings from using it (both now guarded and tested):**

* **`optim.loss: none` leaves every decoder untrained** — verified for the local one under
  `dual_decoder: true`, whose parameters do not move across a run. The mechanism is not specific
  to it: a decoder is a leaf downstream of the embedding, and the contrastive terms read tokens
  *upstream* of it through their own projector head, so with no reconstruction criterion nothing
  reads a decoder's output and none of them receive gradient. The *encoders* still train,
  including the local one on a `CombinedModel`, since the global tokens are built from local
  embeddings. What is left at the end of the run is a representation with no readout attached:
  `val_local_loss` / `val_global_loss` are numbers about a randomly-initialised head — not NaN,
  not obviously broken — and `val_loss` is what EarlyStopping and ModelCheckpoint monitor.

  **For InterScale that also costs the interpretability substrate.** Standardised gene loadings
  come straight out of the linear decoder's weights (see `background.md`), so a contrastive-only
  run produces nothing to interpret, which is the point of the model.

  **The remedy, if a contrastive-only run is wanted anyway: freeze the encoder and fit the decoder
  in a second pass.** That is the standard linear-probe protocol — train the representation with
  `optim.loss: none`, then run a short second training with the encoder parameters frozen and only
  the decoder(s) optimised against the reconstruction criterion. It gives an honest
  reconstruction number *and* a set of gene loadings for a representation that never saw the
  reconstruction objective, which is a cleaner attribution than the hybrid can offer: under the
  hybrid, loadings are shaped by both terms at once. **Not implemented** — it needs a freeze flag
  and a second `Trainer` invocation, not new loss machinery. Use the hybrid until someone wants
  the ablation badly enough to build it.
* **A two-view run with every encoder dropout at 0 has no task at all.** Dropout is currently the
  only thing making two passes differ, so the views are identical and the invariance term is
  exactly 0.0 *in training*, not just in eval. `build_aux_losses` warns on that combination.

**What is not verified.** No real training run: the numbers above come from 3-epoch smoke runs on
the harness's synthetic data. Two things to watch on the first proper run — at defaults the
variance term sits near 15 of its maximum 25 (embedding std ~0.4 against `gamma` 1), so the
objective is initially almost entirely "increase variance"; and `vicreg_cov` *rose* over those
epochs, which is what a `mu`-dominated balance does. The λ/μ/ν balance is the first thing to
sweep, not the last.

## What the probes cannot tell you — measured 2026-09-18

**The finding: on `synth_data_0`, a value-reconstruction probe cannot show that the global
component is needed, for any target in the panel.** This is not a tuning problem and not a bug;
it follows from the data.

Correlation of each target with the mean of its own <=30u neighbourhood — i.e. how much of it is
already inside the 2-hop GCN reach:

| target | kind | r(cell, neighbour mean) |
|---|---|---|
| `lr_tone` | obs | 0.348 |
| `int_short` | gene | 0.384 |
| `n_senderA_short` | obs | 0.438 |
| `noise_00` | gene | 0.462 ← the floor, from shared library/normalisation effects |
| `int_mid` | gene | 0.451 |
| `int_long` | gene | **0.805** |
| `kern_senderB_mid` | obs | 0.989 |
| `dist_to_center` | obs | 0.997 |
| `dist_to_hub` | obs | 0.998 |
| `hub_response` | obs | 0.998 |

**Every long-range target is spatially smooth, and every smooth target is locally readable.** A
distant influence that varies slowly in space produces a field whose value at a cell is almost
exactly its neighbours' average, so the local component reads it off without understanding the
mechanism at all. The only locally-hard targets are high-frequency — which are short-range by
construction. There is no target here whose *value* requires long range.

Consequences, in order of importance:

1. **Predicting `int_long` and detecting the long-range interaction that produced it are different
   claims.** An observed "R2 0.4 from both embeddings" is equally consistent with "the transformer
   added nothing" and "the transformer learned the interaction and the probe cannot see it". Do
   not read a probe gap as evidence about range.
2. **`int_long` is the gene the LOCAL component should do best on** (0.805, far above the 0.46
   floor). The expectation printed by `linear_probing.py`'s legend — "int_long: global should
   lead" — is wrong for the reason it gives, and the legend still states it.
3. **`hub_response` / `dist_to_hub` are not the fix.** At 0.998 they are smoother than `int_long`,
   so they are worse discriminators, not better. (Recommended in this session, then measured and
   retracted — do not re-suggest them.)

### What can test it instead

* **An ablation.** `LocalModel` vs `CombinedModel` on identical config, comparing `int_long`
  reconstruction; and `long_range_attention` on vs off. If adding the transformer does not
  improve it, the global component contributes nothing to that target — a finding about the
  architecture rather than about the probe. Cheap, available today, and the obvious next
  measurement.
* **The attention-flow control** (`evaluation/flow_control.py`, Stage 0b). It asks an
  *attribution* question rather than a prediction one: does flow run senderA -> receiverA at long
  range while staying flat on the composition-matched null pair? A model reading a smooth field
  locally has no reason to produce that; a model using the distant sender does. This is why the
  control exists as a separate instrument from the probes.

### For the next benchmark dataset

A long-range target that is genuinely locally hard must be **non-smooth**: dependence on a
specific, rare, distant source rather than a gradient. A field with a 400-unit length scale will
always be readable from a 60-unit neighbourhood. Building one in is what would make the
local/global split falsifiable by probes at all.

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

### How Stage 2 differed from this plan

The plan's negative was "a random cell set from the same slide, sampled to match the cell-type
histogram". It shipped as **another cell's actual k-hop neighbourhood**, selected for histogram
match, because composition is not the only nuisance variable:

**Spatial coherence is the second one, and the plan did not account for it.** A scattered cell set
pools toward the slide mean while a true context is a contiguous patch. Given the smoothness
measured on 2026-09-18 — `dist_to_center` at 0.997 with its own neighbourhood mean — "which of
these is spatially coherent, or sits at my value of the smooth field" fully separates positive
from negative without any interaction being learned, and it would look like a healthy loss curve
the whole way down. Building the negative with the *same operator* as the positive holds
contiguity fixed alongside composition. `negative_context: scattered` keeps the plan's original
version as the ablation that says how much of the loss was ever about arrangement.

**The negatives are a per-slide bank, not per-anchor.** Negatives are contexts rather than cells,
so drawing them per anchor would cost `n_anchors × n_negatives` k-hop expansions. `n_candidates`
contexts are built once per slide and every anchor on it selects from the same bank. This is why
`n_candidates` must exceed `n_negatives` — with no surplus every candidate is taken regardless of
its histogram and the matching silently does nothing, so the constructor rejects it.

**Three rejection rules, not one.** A candidate is dropped if its context overlaps the anchor's,
if its context *contains the anchor*, or if it is empty. The second is not implied by the first:
the anchor is excluded from its own context, so a candidate context consisting only of the anchor
overlaps nothing while being the most direct false negative available.

**`anchors_masked_only` defaults to True**, for the graph-shaped form of the identity shortcut.
An unmasked anchor's own expression is in the encoder input *and*, because the GCN aggregates over
neighbourhoods, inside its own neighbours' embeddings — so the positive is identifiable by
detecting the anchor's own transcriptome in the pooled context. This is the single-view analogue
of Stage 3's "anchor's own features corrupted in both views", and the same reasoning that put
`probe.masked_cells_only` at True.

**Matching is *selection*, not construction — and that needed its own diagnostic.** Candidate
centres are drawn uniformly from the slide, so the bank's composition coverage is whatever chance
supplies; an anchor with a rare local composition takes the closest of a bad bank. `clean` reports
whether a negative was *false*, not whether it was *matched*, and the loss falls either way, so
`_context_nce_hist_gap` reports the mean L1 between the anchor's positive histogram and its chosen
negatives'. It lies in `[0, 2]`, and it has no scale on its own: read it against the same number
from a `match_composition: False` run, which is what no matching at all looks like on that data.
If the two are close, the bank is too small or too uniform and `n_candidates` is the knob.

**`_context_nce_clean` is the number to watch.** It is the fraction of anchors whose negatives
were all genuinely non-overlapping. On a small slide, or with `context_hops` high enough that every
neighbourhood touches every other, rows get filled from rejected candidates and the denominator
quietly fills with false negatives. The loss looks entirely normal while that happens. A low value
means `context_hops` is too large for the slide, or the slide is too small to contrast within.

**Found while implementing:** `composition` derived its one-hot dtype from the boolean membership
matrix, so the histogram matmul had no float operand. It raised rather than returning a wrong
number, and a test caught it, but the general shape — deriving a dtype from a mask — is the kind
that promotes silently in another arrangement.

**Also done here:** `slide_codes` in `aux_losses.py` is now shared by the VICReg variance hinge and
the contrastive negative pool, instead of VICReg's private `_groups`. Both need exactly the
"prefer the attached `slide` over the graph index" mapping, and two copies would drift the moment
one learned about a new field. Equivalence harness IDENTICAL across the refactor.

### What is not verified

No training run, real or synthetic, beyond the 1-epoch smoke fits in the test suite. Two things to
measure on the first proper run, in this order:

1. **`_context_nce_clean`, `_context_nce_hist_gap` and `_context_nce_acc`.** If `clean` is low the
   term is training against false negatives; if `hist_gap` is no better than a
   `match_composition: false` run the composition matching is not actually happening and the term
   has degenerated into the naive version the stage exists to avoid; if `acc` is at 1.0 from step 0
   the task is trivial and something is leaking — check `anchors_masked_only` first.
2. **The medulla control and the slide-identity probe**, as the plan's gate already says. A term
   designed to be composition-blind moving the flow control means the matching did not work.

`distant_min_um` and the Stage 2b knobs are untouched; 2b remains not started.

---

## Stage 2b — scale-matched pairing (local vs. global)

An add-on to test, not a replacement for Stage 2. The idea: give each component the pairing that
matches the scale it is supposed to encode, instead of one pairing policy for both.

| component | positive | negative |
|---|---|---|
| local | two cells from the **same neighbourhood** | cells **not** from the same neighbourhood |
| global | two **distant** cells on the **same slide** | a *mixture*, sampled per negative: with probability `p` a random cell from **another slide**, otherwise one from the anchor's **nearest neighbourhood** |

The mixture is per-negative, not two separate runs: each negative slot independently draws which
kind it is. `p = 0` is pure within-slide.

The global row's second negative option is the appealing one: "a long-range program is not the
same thing as a local niche" is the loss-side statement of the architecture's own `M = 1 - A`
attention mask, which already forbids the transformer from attending inside the GNN's receptive
field. Making that complementarity an objective rather than only a constraint is a genuinely
new thing to try, and it is cheap — one forward pass, no views.

Every field it needs already exists from Stage 0: `pos` for distance, `edge_index` for
neighbourhood, `batch.batch` / `slide` for slide identity. It is a pairing *policy* over the
existing `AuxLoss` interface, not new plumbing.

### Two predicted failure modes — measure them, do not assume they are avoided

**The local positive may be satisfied at initialisation.** A GCN already averages over the
neighbourhood, so two cells from one neighbourhood share most of their input before any training;
under anchor masking they share even more. "Same neighbourhood ⇒ close" is then close to a
Laplacian smoothness penalty on something already smoothed. Not worthless — it is the DGI /
proximity-embedding objective — but check the loss actually falls from a non-trivial starting
value before concluding it taught the model anything. A flat-from-step-0 curve is the tell.

**The global pairing, taken literally, is a recipe for a slide classifier.** If *every* pair of
distant cells on a slide is a positive and *every* cross-slide pair is a negative, the exact
optimum is "encode which slide you are on": all cells of slide *k* collapse to one point, all
slides pushed apart, loss zero, nothing about interaction learned. That is the batch-effect
failure this plan restricts negatives to one slide to avoid — reached from the other direction,
and it would look like a *good* loss curve the whole way down.

**Mixing the two negative kinds does not neutralise this, and the mixture is what is proposed.**
It is worth being precise about what mixing changes, because the intuition that the within-slide
half "cancels" the cross-slide half is wrong in a specific way:

* A slide-encoding representation *perfectly* solves the cross-slide fraction — anchor and
  positive share a slide, the negative does not, so slide identity separates them exactly.
* It *maximally fails* the within-slide fraction — anchor, positive and near negative are all on
  one slide, so slide identity gives that fraction zero discriminative power.

So the optimum stops being pure slide encoding, but it becomes **slide identity *plus* a near/far
component** — you get the thing you wanted *and* a batch component, not one instead of the other.
With `n_embed = 16` shared with reconstruction, that capacity is genuinely spent.

The training dynamics make it worse than the static argument suggests. NT-Xent weights negatives
by similarity, so early on the cross-slide negatives are similar to the anchor, carry large
gradient, and the cheapest way to reduce that term is to separate slides. Once separated they are
dissimilar, their weight collapses, and they fall silent — having already spent the first epochs
teaching the model the one variable the probes are supposed to test for.

And the cross-slide negatives buy nothing: all the discriminative work wanted here — *distant
cells on this slide belong together, my immediate neighbours do not* — is carried entirely by the
within-slide negatives. Cross-slide ones are easy negatives pointed at the wrong variable.

**If the motive for `p > 0` is collapse prevention**, that is a real worry with a better answer:
Stage 1's variance and covariance terms provide uniformity pressure with no negatives at all, so
nothing about slide identity gets rewarded. Reach for that first.

**If `p > 0` is still wanted, constrain what "another slide" means.** The harmful variable is not
the slide, it is the technical covariate the slide carries. Sample cross-slide negatives from the
**same donor and the same condition**, so "different slide" means a different section rather than
a different batch. That also keeps the objective clear of condition, per the rule above. This
needs a `dataset.donor_key`, which does not exist yet — a one-line extension of the Stage 0
`OPTIONAL_FIELDS` mechanism, not new plumbing.

**Measure the cost rather than argue about it.** Sweep `p` (0, 0.25, 0.5) with everything else
fixed and read the slide-identity probe. If `p > 0` raises it, that is the price, quantified —
and a `p = 1` run is a free positive control that makes the slide-ID alarm a number rather than a
rule of thumb. If even `p = 0` drives slide-ID up, then the distant-same-slide *positive* is
degenerate on its own and needs a constraint making two distant cells plausibly related rather
than merely co-resident (matched niche, or matched local embedding), at the cost of reintroducing
a label. Decide that after seeing the probe, not before.

### Knobs

```yaml
optim:
  aux_loss_weights:
    local_neighbourhood_nce: 0.0
    global_distant_nce: 0.0
  contrastive:
    near_hops: 1              # what counts as "same neighbourhood" for the local positive
    distant_min_um: 200       # how far apart a global positive must be, in `pos` units
    cross_slide_fraction: 0.0 # p: per-negative probability of drawing from another slide
    cross_slide_match: [donor, condition]   # hold these fixed when crossing slides, so
                              # "another slide" is another section and not another batch
```

`distant_min_um` wants setting against the neighbour-graph radius (`spatial_neigbors_kwargs`),
not guessed: "distant" has to mean well outside the local component's reach, or the two rows of
the table are contrasting the same thing.

### What would count as success

Beyond the usual probe directions: the local and global embeddings should become **less**
redundant, not more. The online probes already report both separately, so the readout is whether
the gap between them widens on targets that are scale-specific — niche composition readable from
local, tissue-scale program readable from global — while slide identity stays flat for the
`near_negative` arm.

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

This is deliberately a *global* within-type permutation, not a per-neighbourhood one: a
symmetric aggregator is near-blind to a shuffle inside one neighbourhood (see **Decided so
far**), so that variant would be a no-op the loss could never reduce.

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
    context_hops: 2                    # hops defining the POSITIVE context
    n_negatives: 16
    n_anchors: 512                     # 0 = all
    n_candidates: 256                  # per-slide bank; must exceed n_negatives
    negative_context: neighbourhood    # neighbourhood | scattered (the ablation)
    match_composition: true            # needs dataset.celltype_key
    anchors_masked_only: true
    augment:
      gene_mask: 0.3
      expression_noise: 0.0
      neighbour_mask: 0.0
```

## Appendix B — files touched per stage

| stage | new | modified |
|---|---|---|
| 0 | `module/base/_step_output.py`, `train/aux_losses.py` | 4× `_common_step`, `_trainingplans.py`, `optim_config.py`, `tl/geome_utils.py` |
| 0b | `evaluation/online_probes.py`, `config/probe_config.py` | `config/__init__.py` (validation), `train/_training.py` (callback), `evaluation/__init__.py` |
| 1 | — | `aux_losses.py`, `_base_global_module.py` (expander) |
| 2 | `train/context_nce.py` | `aux_losses.py` (`ContextNCE`, `build_projector`, `slide_codes`), `optim_config.py` (weight + 8 knobs) |
| 2b | — | `aux_losses.py`, `optim_config.py` (three knobs); needs `pos` and `slide` from stage 0 |
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
- [ ] a negative context is built by the *same operator* as the positive, so spatial coherence is
      held fixed alongside composition — a scattered negative pools toward the slide mean and is
      separable without any interaction being learned
- [ ] a candidate that *contains the anchor* is rejected, not only one that overlaps the anchor's
      context — the anchor is excluded from its own context, so the two are different tests
- [ ] the candidate bank is larger than `n_negatives`, or the composition match selects nothing
- [ ] anchors restricted to masked cells: the GCN puts the anchor's own expression inside its
      neighbours' embeddings, so the positive is otherwise identifiable by self-detection
- [ ] `_context_nce_clean` read before any conclusion — a denominator full of false negatives
      produces an entirely normal-looking loss curve
- [ ] `_context_nce_hist_gap` compared against a `match_composition: false` run — selecting the
      nearest candidate from a uniformly drawn bank is not the same as finding a matched one, and
      only this comparison says which happened
- [ ] a proposed corruption is actually visible to the encoder (see the `GCNConv` note above)
- [ ] a distant-same-slide positive is paired with *within-slide* negatives — any cross-slide
      fraction makes slide identity a rewarded direction, and mixing does not cancel it, it just
      adds a near/far component alongside it
- [ ] if cross-slide negatives are used anyway, donor and condition are held fixed across the
      pair, so "another slide" is another section rather than another batch

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
