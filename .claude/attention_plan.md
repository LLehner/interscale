# Staged plan: attention mechanisms for the global module

Adds alternative attention mechanisms ("backends") to the transformer, selectable from config, so
that many combinations of attention, encodings and model widths can be tested on several datasets.
The papers below were inspiration, not specifications: a combination one paper happened to use is
not a constraint, and anything that can be combined should be combinable.

- Efficient-transformer survey (Tay et al., arXiv 2009.06732v3): the taxonomy.
- Long Range Arena (LRA; Tay et al., ICLR 2021): how to compare mechanisms fairly, and the
  "required attention span" measure.
- Dimitrov (arXiv 2508.17175): dense vs. sparse graph transformers (Graphormer, SAN, GraphGPS,
  Exphormer).
- SpaFormer (Wen et al., arXiv 2302.03038): a Performer over cells as tokens.

Read [`background.md`](background.md) and [`PE_plan.md`](PE_plan.md) first: this plan builds on PE
Stages 5–8 and keeps every invariant listed there. Code paths are relative to `src/interscale/`.

## Status

Last updated 2026-10-11. Same rules as the other plans: update this table in the same commit as the
work, and a stage is `done` only when something external verified it. "Implemented, unverified" is
a real state.

| stage | state | verified by | date |
|---|---|---|---|
| 0 — plumbing: backend registry, free head width, compatibility validator, maps only when consumed | **done** | 517 tests pass; harness IDENTICAL across 5 cases / 15 epochs after each of the two commits; training steps keep no map and hook nothing unless the loss reads attention; the skeleton's dense core reproduces torch's attention (outputs and maps, with mask, padding and RoPE) and draws its initial weights; a non-default backend leaves every weight built after it unchanged (mutation-checked: without the replayed draws it fails); every validator rejection tested | 2026-10-11 |
| 1 — attention diagnostics (span in µm, sink mass, effective keys) | **implemented, unverified** — planted maps read their values (a query attending to a cell 300 µm away reads 300; mass beyond radii; CLS mass; effective keys 1 and 3); left padding and graph-major heads pinned on two graphs of different size; a real pass's numbers are those of the maps the step returns, each token's coordinates placed by the step's own token-to-cell index; the Performer's reference equals the dense model's first-layer map at one seed, and its TV falls with the feature count (0.19, 0.041, 0.014 at m = 8, 512, 8192); no TV for dense, linear or Exphormer; a pass leaves every global generator, the training flag and the module's flags as they were, and dense numbers unchanged while recording. Six mutations (CLS as a key, right padding, head-major rows, a reference without the mask, flags not restored, no RNG guard), each caught. 623 tests pass; harness IDENTICAL. **Not yet run in training** | 2026-10-11 |
| 2 — kernel attention: Performer (FAVOR+) and linear (elu+1); generated sweep arms | **Performer, linear and the generated arms implemented, unverified** — `tl.blocked_pairs` equals the dense mask at hops 0–3 (two graphs of different size, one subsampled). Performer: the orthogonal features estimate exp(x·y) without bias (found biased by ~7% before the Haar sign fix); the map's error against softmax falls with the feature count (TV 0.10 at m = 16, 0.009 at 4096); the float64 subtraction stays exact where blocked keys hold 99.7% of a query's mass (float32: ~1e-5 off); shares every initial weight with the dense model at one seed; per-layer features from their own seeded generator, redrawn only in training and only when asked. Linear: equals its definition; each head's map has rank exactly `head_dim` before the mask (dense softmax at the same width: near full rank); adds no state, so a dense checkpoint loads strictly; refuses RoPE and biases, and the validator says why; warns below a head width of 32. Both: factorised and explicit paths agree to 1e-5 with mask and padding; block exactly the dense mask's pairs; finite on the complete-graph test at hops 1–3 and through 5 training steps; relevance runs; train through `_common_step`. Five mutations, each caught. 570 tests pass; harness IDENTICAL. Generated arms: the validators decide what is kept, each rejection with its reason; every arm sets the same keys; an arm must validate at every grid point; the file round-trips through `build_sweep_config`, `load_arms` and `apply_sweep_config` into configs the validators accept; trials that would share a checkpoint are named. Seven mutations, each caught; 646 tests pass; harness IDENTICAL. **No real-data run yet** | 2026-10-11 |
| 3 — low-rank biases (spectral bias into q/k; Fourier distance bias) | **implemented, unverified** — spectral: `U Wᵀ` equals the dense bias (two graphs of different size, one subsampled; CLS and padding zero); a zero bias leaves one side 0 and the other small (mutation-checked); a Performer's map converges to dense softmax with the same bias (TV 0.047 → 0.014 from m = 1024 to 16384 at a bias of ±1.2). Fourier (`pe.bias: [fourier]`): zero at init; each entry is the learned function of `p_key − p_query` (mutation-checked: a flipped sine sign fails); translation-invariant; isotropic is symmetric and invariant to a turn by 180°/directions, anisotropic is not; the default grid fits a ring at 1 mm with R² > 0.9; derived `max_wavelength`; through Performer and dense attention end to end, gradient reaching the coefficients. The directional variant is tagged `fourier-aniso` in checkpoint names (fixed 2026-10-12; both variants had shared one). 555 tests pass; harness IDENTICAL. **No real-data run yet** | 2026-10-11 |
| 4 — global tokens and a null key | not started | | |
| 5 — Exphormer-style sparse attention | **implemented, unverified** — with the pattern of every open pair it equals dense attention (outputs and maps; long-range mask, padding, a subsampled graph, distance + spectral + Fourier biases per pair, RoPE), and an Exphormer model on a complete graph equals the dense model at one seed, distance bias included; factorised and explicit paths agree; weights only along the pattern, every row with a key; edge types neutral at init, then steering; blocks exactly the long-range mask's pairs, the graph edges gone under it; finite on the complete-graph test at hops 1–3 and through 5 training steps; relevance runs; trains through `_common_step`; adds only its edge-type weights and moves the global RNG as dense does; the pattern is redrawn only in training with `resample`, never from the global RNG. The bias pair form and the pattern builder (earlier commits) checked against the dense bias and mask. Nine mutations across the three commits, each caught. 596 tests pass; harness IDENTICAL. **No real-data run yet** | 2026-10-11 |
| 6 — fused exact attention (SDPA) | not started | | |
| 7 — inducing points (optional) | not started | | |
| 8 — beyond the window (whole slides; FlexAttention) | not planned in detail | | |

## In short

- **Two new attention types come first: Performer (Stage 2) and Exphormer-style sparse attention
  (Stage 5).** Linear attention rides along with the Performer because they share all of their
  machinery. Exphormer was built before global tokens (Stage 4), with CLS as its one virtual node;
  global tokens join its attention graph once they exist.
- **Everything else is a component that combines freely.** That covers node encodings, attention
  biases, RoPE, global tokens, the null key, the long-range mask, and the model's width
  (`n_embed`, `n_heads`, `head_dim`, `num_layers`).
- **Compatibility is declared, not discovered.** Each backend lists what it can carry, and each
  bias lists the forms it can be evaluated in. The validator rejects a combination that cannot work
  when the config loads, and a generated arms file keeps only the combinations that pass.
- **Dense attention stays the default and the reference.** Every backend is also run at a size
  where dense still runs, so a difference is the mechanism and not the length. That is LRA's method.
- **Interpretation needs per-layer maps with gradients at evaluation.** Every backend therefore has
  an explicit path that computes the same function and materialises its map. Training may use the
  cheap path; `evaluate()` uses the explicit one.

## What the code already decides

- **Every layer materialises its attention, in training too.** `_sa_block` calls the vendored MHA
  with `need_weights=True, average_attn_weights=False`
  (`module/global_modules/transformer_encoder_layer.py:87-97`). The explicit branch
  (`transformer_utils.py:391-423`) therefore always runs, and the fused SDPA branch (`:425-447`)
  never does. Each layer holds several `[B·H, S+1, S+1]` fp32 tensors (softmax and dropout outputs
  kept for backward, plus a float mask built per layer at `:234-241, 381`). One such tensor at
  B = 4 graphs and H = 4 heads is 0.4 GB at S = 2.5k, 6.4 GB at 10k and 160 GB at 50k. Its size
  depends on the head count, not the head width.
- **Training hooks, stacks and keeps maps that nothing reads.** `forward`'s `register_hook`
  defaults to True (`transformer_encoder.py:200, 232`). Every training step therefore hooks every
  layer's map for its gradient and stacks all layers into `final_attn` (`:283`). Each layer also
  keeps its last map between steps. The only training consumer of the maps is
  `SCE_EntropyATT_Loss` (`train/losses.py:183`); the gradients are only read in `evaluate()`.
- **Interpretation reads dense maps with gradients.** `evaluate()` runs one graph, then
  `SelfAttentionRelevance.generate_relevance` backpropagates and rolls out `I += Ā·I` over the
  layers (`tl/self_attn_relevance.py:62-95`): dense `S × S`, O(L·S³). `get_model_output` stores `I`
  as a dense `[n_cells, max_seq_len]` obsm (`model/combined_model.py:131-149`), which net flow and
  the flow control read.
- **Biases are merged into one dense float mask.** `AttentionBias` returns `[B·H, S+1, S+1]`
  (`positional_encodings.py:735`), and `common_step_local_to_global` merges the boolean mask in
  with `masked_fill` (`transformer_encoder.py:184`). That is right for dense attention and the wrong
  shape for anything else.
- **The spectral bias is already low rank.** `SpectralBias.forward` is `(V·diag(h_h))·Vᵀ`
  (`positional_encodings.py:333`), so it can move into the queries and keys (Stage 3). The distance
  bias is a function of `cdist` and is not low rank.
- **RoPE is applied to q and k before the explicit/fused split** (`transformer_utils.py:339-355`).
- **Head width is tied to model width.** torch's MHA fixes `head_dim = n_embed / n_heads`, and
  `n_embed` (`config/model_config.py:10`) is shared with the local module, the decoders and the
  probes. Widening the heads today widens the whole model; Stage 0 decouples the two.
- **The k-hop closure is densified per full graph** (`tl/masking.py:296`): `N × N` booleans before
  `pad_batch` subsamples. The sparse product just before it already holds the closure as pairs.
- **Init order is load-bearing.** `CustomTransformerEncoderLayer` lets
  `TransformerEncoderLayer.__init__` build and initialise a stock MHA, then replaces it
  (`transformer_encoder_layer.py:46`), so the init RNG is drawn twice. A refactor that drops the
  first draw or moves the second changes every later initial weight.
- **`batch_first=False` keeps torch's encoder fast path off.** That path would otherwise bypass
  `_sa_block`, and RoPE with it, in no-grad eval. Keep it.

## Triage

| family | examples | verdict | why |
|---|---|---|---|
| kernels | Performer (FAVOR+), linear (elu+1) | **yes** (2) | O(S); every pair stays reachable; SpaFormer's backbone, LRA's best on Pathfinder. Linear attention needs wide heads, see Stage 2 |
| graph-sparse | Exphormer (graph ∪ expander ∪ virtual nodes), BigBird random + global | **yes** (5) | the sparse side of Dimitrov's comparison, and carries per-pair biases |
| global tokens | ETC/Longformer global tokens, Exphormer virtual nodes, registers | **yes** (4) | CLS is already one; more can absorb attention sinks; Exphormer needs them |
| exact, memory-efficient | SDPA, FlashAttention, FlexAttention | **yes** (6, 8) | same function, no approximation |
| latent bottleneck | Set Transformer ISAB, Luna, Perceiver IO | optional (7) | tissue-wide latents as a summary |
| local patterns | local window, blockwise, Image Transformer, Swin | no | the GNN already does local; the long-range mask forbids it |
| strided / axial | Sparse Transformer, Longformer dilation, Axial, CCNet | no | need a 1D order or a grid, and token order here is arbitrary node order |
| low-rank length projection | Linformer, Synthesizer | no | weights indexed by token position: not permutation-equivariant, and N varies per graph |
| learnable patterns | Reformer, Routing, Sinkhorn | not now | fallback if kernels and Exphormer both fail sharp long-range attention |
| recurrence, sparse MoE | Transformer-XL; Switch, GShard | no | sequence recurrence; FFN-level, not attention |
| MPNN in every layer | GraphGPS | out of scope | an architecture change, not an attention mechanism; it would also mix local message passing into the global stream |

## Components and how they combine

✓ works; ✗ cannot work (the validator rejects it); ~ works with the caveat noted. A stage number
means "from that stage on".

| backend | node PE / SE | rope | distance (profile, linear) | fourier | spectral | SPD bias (future) | long-range mask | maps + relevance at eval | `SCE_EntropyATT_Loss` |
|---|---|---|---|---|---|---|---|---|---|
| dense | ✓ | ✓ | ✓ | ✓ 3 | ✓ | ✓ | ✓ | ✓ exact | ✓ |
| performer (2) | ✓ | ✓ rotate, then feature map | ✗ | ✓ 3 | ✓ 3 | ✗ | ✓ by subtraction | ✓ implied map | ✗ |
| linear (2) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ by subtraction | ✓ explicit map | ✗ |
| exphormer (5) | ✓ | ✓ | ✓ on edges | ✓ | ✓ on edges | ✓ with an SPD table | ✓ removed from the edges | ~ zero off the edges | ✗ |
| dense, fused (6) | ✓ | ✓ | ✓ float mask | ✓ | ✓ | ✓ | ✓ | ✓ explicit path | ✗ |
| inducing (7) | ✓ | ~ latents need positions | ✗ | ✗ | ✗ | ✗ | ✗ | ~ rank-m surrogate | ✗ |

- **Global tokens and the null key (Stage 4)** combine with every row except inducing points.
- **Head width, head count, depth and `n_embed`** combine with every row.
- **Linear attention needs `head_dim` well above 4** to be worth running, and carries neither biases nor
  RoPE: both act through an exponential that elu+1 does not have (Stage 2).

**Bias forms.** These are the extension point that makes the table derived rather than maintained:

- `dense(ctx) → [B, H, S, S]`: today's form.
- `pairs(ctx, b, i, j) → [H, P]` on any list of pairs: used by the dense backend (all pairs),
  Exphormer (its edges) and FlexAttention (`score_mod`).
- `factors(ctx) → U, W [B, H, S, r]` with `bias = U·Wᵀ`, moved into the queries and keys:
  `q̃ = [q/√d, U]`, `k̃ = [k, W]`, logit `q̃ᵀk̃` at scale 1. This works in every backend that sees q
  and k through an exponential: for Performer, FAVOR+ estimates `exp(q̃ᵀk̃) = exp(qᵀk/√d + b)`.
  Linear attention has no exponential, so a bias has no `exp(b)` factor to become there, and it
  takes none.

| encoding | dense | pairs | factors |
|---|---|---|---|
| distance, `profile` / `linear` | ✓ | ✓ per-pair norm | ✗ |
| fourier (3) | ✓ | ✓ | ✓, 2R dims |
| spectral | ✓ | ✓, `Σᵢ h(λᵢ)·N·vᵢ[j]·vᵢ[l]` per pair | ✓, k dims |
| SPD (Graphormer; future) | ✓ | ✓ from an SPD table (uint8, S² bytes) | ✗ |
| RRWP (relative random-walk probabilities; future) | ✓ | ~ local only | ✗ |

Node-level encodings enter the token before any attention, so they work with every backend. That
covers the four node PEs and any future node SE: degree, SignNet, SAN's learned LPE, local density,
niche composition.

Under the long-range mask, local pair encodings carry nothing. RRWP, SPD within k hops and the
distance bias below the mask radius only ever touch blocked pairs.

## Design

**Registry.** `module/global_modules/attention.py` holds
`ATTENTION_REGISTRY[name] = AttentionSpec(build, validate, mask_forms, bias_forms, rotary, training_maps, eval_maps)`.
Bias `PESpec`s gain `forms` (today both are `{"dense"}`). `_validate_attention` joins the folded
validator list from PE Stage P. It checks that:

- every enabled bias shares a form with the backend;
- `pe.rotary` has a backend that rotates;
- `long_range_attention` has a mask form;
- `SCE_EntropyATT_Loss` gets `training_maps`.

Each rejection names the backends or encodings that would fit.

**One skeleton for every new backend.** Each new backend runs the same steps:

1. Project `n_embed → H × head_dim` for q, k and v.
2. Rotate q and k (RoPE).
3. Run the core: dense, kernel, edge-list or latent.
4. Merge the heads.
5. Project `H × head_dim → n_embed`.

The default dense path keeps today's `MultiHeadAttentionWithEdits`, so the harness stays
IDENTICAL. The skeleton's dense core, with weights copied from that module, must reproduce its
outputs and maps.

**Free head width.** `attention.head_dim` (0 = `n_embed / n_heads`, today) sets the attention width
without touching the local module, the decoders or the probes. `model.n_embed` stays free too, for
when the whole model should be wider.

- Head width is cheap: q, k and v are `O(S·H·head_dim)`.
- Head count is not: every map is `[B·H, S, S]`.
- RoPE's multiple-of-4 check moves to the attention head width, and so do the RoPE shape checks in
  `forward` (`transformer_encoder.py:225`).

**Per-batch context.** Inputs other than the masks reach the layers the way RoPE's do today: set on
each layer for one `forward` call and cleared in a `finally` (`transformer_encoder.py:260-268`).
`layer.rotary` becomes a general `layer.context` when the second consumer arrives (Stage 2's blocked
pairs).

**Explicit path.** On request, every backend materialises `[B·H, S', S']` maps with autograd, where
S' = S + 1 + global tokens. `evaluate()`, relevance, Stage 1's diagnostics and the equivalence tests
all use it. Maps are requested only where something consumes them.

**Randomness.** Performer's features and Exphormer's resampled edges draw from a backend generator
seeded from `optim.seed` and built under `fork_rng`, and only in training forward passes. Eval is
deterministic per graph, and an off backend draws nothing.

**Checkpoint tag.** `attn-<kind>[-hd<d>][-fused][-g<R>][-null]_` after the PE tag, only when not
default. Without it, a sweep's arms overwrite each other's checkpoints.

**Config.** `model.global_component.parameters.attention`, beside `pe`. Each stage adds its own
sub-block together with its registry entry, so no key exists before something reads it:

```yaml
attention:
  kind: dense          # dense | performer | linear | exphormer | inducing
  head_dim: 0          # 0 = n_embed / n_heads
  fused: false         # dense only (Stage 6)
  global_tokens: 0     # Stage 4
  null_key: false      # Stage 4
  performer: {num_features: 128, redraw_interval: 0}   # 0 = draw once
  exphormer: {degree: 8, graph_edges: true, resample: false, edge_type: bias}
  inducing:  {num_latents: 16}
```

Most of what the previous version left as questions are knobs here: `redraw_interval`, `head_dim`,
and the row normalisation of net flow (both variants are reported). The sweep decides.

## Stages

Every stage: the tests pass; it runs end to end through `GlobalModel`, `CombinedModel`, the dual
decoder and long-range masking, at more than one width; `get_model_output` runs; and the
equivalence harness stays IDENTICAL with the new option off. Model runs happen on the cluster. New
tests go in `tests/test_attention_backends.py`, and the existing long-range mask tests get
parametrised over backends.

### 0 — plumbing, free head width

- **Registry** with `dense` = today's `MultiHeadAttentionWithEdits`, built at the same point in the
  init order.
- **The skeleton**, with a dense core and `attention.head_dim`. A non-default `head_dim` takes the
  skeleton; the default keeps today's module.
- **Maps only when consumed.** `register_hook` splits into `return_maps` and `relevance_hooks`.
  `return_maps` stacks the per-layer maps and is on only when the loss consumes them;
  `relevance_hooks` is on only in `evaluate()`. Numbers cannot move, and each training step stops
  holding a stacked copy and a stored gradient per layer.
- `forms` on bias specs, `_validate_attention`, checkpoint tag.
- **Verify:**
  - Harness IDENTICAL on all five cases. `global_sce_entropy_node` proves the entropy loss still
    gets its maps; `combined_longrange_mse_node` covers the mask path.
  - The skeleton's dense core, with weights copied from MHA, equals it in outputs and maps.
  - Every validator rejection has a test, using test-registered backends that lack a form.
  - Checkpoint names are unchanged by default.

#### How Stage 0 was built

- **`register_hook` became three-valued** instead of two flags. True (what `evaluate` and the tests
  pass) keeps and returns every layer's map and hooks it for its gradient. False keeps nothing. None,
  the default of every training step, returns the stacked maps only when
  `TransformerNodeEncoderHook.return_attention_maps` is set; `from_config` sets it when `optim.loss`
  is in `train.losses.LOSSES_READING_ATTENTION`. Torch's attention still takes its explicit path
  every time, so its numbers cannot move. Other backends compute a map only when one is kept.
- **The skeleton is `attention.ProjectedAttention`**, with `DenseAttention` as its first core. A
  backend's module replaces `self_attn` in `CustomTransformerEncoderLayer`. Its per-batch extras
  arrive as `layer.attention_context`, set like `layer.rotary`, and the layer tells backends from
  torch's attention by `is_backend`.
- **`attention.head_dim`** reaches RoPE through `build_positional_encodings(..., head_dim=)`. The
  config-time RoPE check reads it too (`_validate_pe`), and the transformer keeps it as `head_dim`.
- **Bias forms** are `PESpec.forms`; `AttentionBias.factor_dim()` sums the factor widths of the
  enabled biases (0 until Stage 3).
- Code: `module/global_modules/attention.py`, the layer and transformer changes, `_validate_attention`
  in `config/__init__.py`, `_attention_tag` in `tl/utils.py`; tests in
  `tests/test_attention_backends.py`.

### 1 — attention diagnostics

Per layer and head, from explicit maps on a fixed set of val graphs, with CLS and padding left out
of the keys:

- **Span (µm)**: `Σⱼ Aᵢⱼ‖pᵢ − pⱼ‖ / Σⱼ Aᵢⱼ`, the mean and quartiles over cells. This is LRA's
  required attention span in physical units.
- **Mass beyond radii** from config (`probe.attention_radii_um`).
- **Mass on CLS**, and on global tokens and the null key after Stage 4: the attention sinks.
- **Effective number of keys** `exp(H(Aᵢ))`: how peaked a head is.
- **For approximate backends**, the row-wise total-variation distance to exact attention on the
  same weights.

These are logged as `probe/attn_*` through the probe callback when `probe.attention_diagnostics` is
set, under its RNG rules, and are callable post hoc on any checkpoint. Coordinates come from
`dataset.spatial_key`; without them span and radii are skipped with a warning. Everything is
dataset-agnostic.

**Verify:**

- Planted maps read their known values: a query attending only to a cell 300 µm away reads 300.
- Mutations are caught: CLS counted as a key, the wrong padding side, the wrong head order.
- The global RNG states are unchanged after a pass.

#### How Stage 1 was built

- **`TransformerNodeEncoderHook.recording_attention(reference=False)`** is a context manager. Inside
  it every pass keeps its maps and records `{index_nodes, src_padding_mask, maps, reference}`. Kernel
  backends then take their explicit path, as `evaluate()` does: same function, rounding apart. Its
  flags are restored on exit, even if a pass raises.
- **The Performer's reference** is the exact softmax of the inputs its features see,
  `x · y = q · k / √d + bias` after RoPE, with blocked pairs and padding excluded, in float64. It is
  formed in `_feature_maps` while `record_reference` is set, and feeds nothing.
- **`evaluation/attention_diagnostics.py`**: `head_statistics` turns one pass's maps into per-cell
  values, `summarize` into scalars, and `attention_diagnostics(module, data_list, ...)` runs the first
  `max_graphs` graphs one at a time: clean (nothing masked), in eval, under `_rng_untouched`.
  Coordinates are `pos × dataset.spatial_unit_um`, laid out with `pad_like` from the recorded
  `index_nodes`.
- **Definitions.** Every statistic is over cell queries; CLS and padding rows are left out. Span,
  radii and effective keys renormalise each row over its cell keys and skip a row whose cell mass is
  below `MIN_CELL_MASS` (1e-6). CLS mass and TV read every cell row, TV over the whole row. A head's
  value is the mean over every cell read; `_l<layer>_mean` averages a layer's heads and `_mean` all
  heads.
- **Names**: `probe/attn_<metric>_l<layer>_h<head>`, metrics `span_um`, `span_um_q25` / `q50` /
  `q75`, `beyond_<r>um` (a dot becomes `p`), `cls_mass`, `effective_keys`, and `tv_softmax` for the
  Performer only. The callback's charts put a layer's heads on one plot.
- **Config**: `probe.attention_diagnostics` (counts as a probe target), `probe.attention_graphs` (4;
  0 reads every val graph), `probe.attention_radii_um` (`[]`).
- **Measured on an untrained toy model** at head width 4: the Performer's TV to softmax is 0.19 at
  m = 8, 0.041 at 512, 0.020 at 4096 and 0.010 at 16384, about 1/√m.
- Code: the recorder in `transformer_encoder.py`, the reference in `attention.py`, the callback hook
  in `online_probes.py`, `_validate_probe`; tests in `tests/test_attention_diagnostics.py`.

### 2 — kernel attention: Performer and linear

Both compute `out = D⁻¹·φ(Q)(φ(K)ᵀV)` with `D = diag(φ(Q)φ(K)ᵀ1)`, at O(S·r·head_dim), where r is
the feature count. They differ only in the feature map φ:

- **Performer (FAVOR+, Choromanski et al., ICLR 2021):** positive orthogonal random features,
  `φ(x) = exp(Wx − ‖x‖²/2)/√m` with `x = q/head_dim^{1/4}`, so that
  `E[φ(q)ᵀφ(k)] = exp(qᵀk/√head_dim)`. That is an unbiased estimate of the softmax kernel, and each
  head's map has rank m whatever the head width. Knobs: `num_features` m and `redraw_interval`.
- **Linear (Katharopoulos et al., 2020):** `φ(x) = elu(x) + 1`. It is deterministic, with no
  features to draw, but it is a different kernel, not an approximation of softmax.
  - φ has as many features as the head has dims, so each head's map has rank ≤ `head_dim`.
  - Every query's attention is then a mixture of the same `head_dim` key profiles, and the whole
    sequence reaches a query through `head_dim` summary vectors. At 4 dims per head that is a
    4-slot bottleneck per head.
  - Softmax escapes the bound because exp of a low-rank logit matrix is generally full rank: a
    query can still single out its best-aligned key.
  - So linear attention is only run with `head_dim` ≥ 32 or so, and the width sweep shows how much
    the bound costs.

What both share:

- **Long-range mask by subtraction.** Every blocked pair's term comes out of both sums:
  `numᵢ −= Σ_{j∈B(i)} φ(qᵢ)ᵀφ(kⱼ)·vⱼ`, and `denᵢ` likewise.
  - The pairs come from a new `tl.blocked_pairs`: the sparse k-hop closure mapped through
    `index_nodes` to padded positions and never densified. It is checked equal to the dense mask,
    including subsampled graphs and paths through dropped nodes.
  - The subtraction is exact with respect to the kernel: both feature maps are positive, so what
    remains is a sum of positive terms over the open keys, and CLS is always open.
  - Risk: cancellation when a query's mass sits almost entirely on its blocked neighbourhood. Test
    that case.
- **Padding:** φ(k) is zeroed at padded keys.
- **RoPE** (Performer only): rotate q and k, then apply the feature map. That estimates
  `exp(qᵀR_{n−m}k)`, the RoPE softmax kernel, with positive, normalised weights. Linear attention
  does not take RoPE: turned before elu+1, the similarity depends on absolute positions, not
  offsets. RoFormer's alternative, rotating the features after φ in the numerator only, keeps the
  offset but gives weights that can be negative and do not sum to 1, which the flow pipeline cannot
  use.
- **Biases** (Performer only) arrive with Stage 3 (`factors`).
- **Performer features** are buffers, so they are in the checkpoint. They are drawn from the backend
  generator and fixed at eval.
- **Maps:** the explicit path normalises `φ(Q)φ(K)ᵀ` with the blocked entries set to 0, which gives
  the same numbers as the factorised path.

**Generated sweep arms** (built with this stage, the first second backend).
`scripts/make_attention_arms.py` takes lists of backends, node PEs, biases, rotary options and
widths. It runs every combination through `load_config` and writes the ones the validator accepts
as an `arms:` file, logging each rejected combination with its reason.

**Verify:**

- Performer: `φ(q)ᵀφ(k)` averaged over draws matches `exp(qᵀk/√d)` within its standard error, and
  the error against exact attention falls like 1/√m.
- Explicit and factorised paths agree to 1e-5.
- The subtraction equals the explicit masked computation, including nearly fully blocked rows, CLS
  rows and padding.
- The long-range mask tests pass for both backends: `test_blocked_pairs_receive_no_attention`,
  `test_paths_through_dropped_nodes_are_still_blocked` and the complete-graph NaN test.
- Under RoPE with Performer, translating the coordinates leaves the expected attention unchanged.
- Enabling either backend leaves the other initial weights and the batch order unchanged.
- **Cross-evaluation** (cluster): a dense checkpoint evaluated with Performer attention, and the
  reverse, measures the approximation on trained weights. Register the feature buffer under
  `_OPTIONAL_STATE_PREFIXES` so either direction loads.

#### How Stage 2 was built

- **The mask slot carries a `StructuredMask`.** It holds the blocked pairs plus the bias factors
  (Stage 3), and stands in `GlobalInput.attention_mask` where the dense mask stands; every caller
  only passes that slot on, so no signature changed. `forward` hands it to the layers as
  `attention_context` and gives `nn.TransformerEncoder` no mask. A mismatch between the backend's
  declared mask form and what arrives raises.
- **The sums are float64.** The features, the full sums and the subtracted blocked terms are float64,
  and only the result is cast back. With 35 of 40 keys blocked and aligned with the query, float32
  cancellation was off by ~1e-5 and float64 by ~1e-14 (the precision test, mutation-checked). The
  extra memory is the `[B·H, S, m]` features and `[P·H, m]` gathered pairs.
- **The orthogonal features needed the Haar sign fix.** Plain `torch.linalg.qr` of a Gaussian
  matrix is not uniform over the orthogonal group; without multiplying Q by the signs of R's
  diagonal, the estimate of exp(x·y) came out ~7% high.
- **No attention dropout.** The factorised path has no map to drop entries from; dropout elsewhere
  (FFN, residual) is unchanged.
- **Features** are the buffer `favor_projection`, drawn per layer in layer order from one generator
  seeded with `optim.seed` (`attention_seed`). The layers are deep copies of one, so without the
  per-layer draw every layer would hold the same features.
- **Cross-evaluation** (dense checkpoint, Performer attention) loads with `allow_partial_load=True`.
  `_OPTIONAL_STATE_PREFIXES` stays narrow on purpose (`test_gene_masking.py`): a Performer
  checkpoint that lost its features must not load silently with new ones.
- **Default `num_features` 128.** It is a knob; the error measurements above say 128 leaves a mean
  TV distance of a few percent on unit-scale queries.
- **Biases** came with Stage 3, through the `factors` form.
- **One kernel base.** `KernelAttention` holds everything the two share: the mask as blocked
  pairs, the float64 sums and subtraction, padding and the explicit path. `PerformerAttention` and
  `LinearAttention` only supply their feature map (`_feature_maps`). The Performer was moved onto it
  in its own commit, with its tests unchanged.
- **Linear attention** (`attention.kind: linear`) is `φ(x) = elu(x) + 1` on the raw queries and keys,
  as in Katharopoulos et al. It has no settings, buffers or randomness: its state dict is the dense
  model's, so a dense checkpoint loads into it strictly (as a different kernel, not an estimate).
  `supports_rotary = False` makes it refuse a rotation even when built by hand. Its registry entry
  carries a `note` that the validator appends to every rejection (no exponential for a bias or RoPE
  to act through; the Performer carries both), and it warns below a head width of 32
  (`LINEAR_RANK_WARNING`).
- **Rank, measured.** At head width 4 and 8 a linear head's map (before the mask) has exactly that
  many singular values above ~3e-2 of the largest; the rest sit at ~1e-8, float32 noise. Dense
  softmax at width 4 on the same 40 tokens has rank 37 at a relative tolerance of 1e-4. The mask's
  zeroed pairs are a sparse correction on top, so a masked map's numerical rank is higher without
  carrying more long-range structure.
- **What linear attention is for:** a control for the Performer (the same O(S) structure without
  the softmax kernel or random features; compare at `num_features` = `head_dim` for equal rank), a
  probe of how much attention rank a task needs (sweep `attention.head_dim`), and the cheapest
  backend.
- **Generated arms** (`config.sweep.generate_arms`, `scripts/make_attention_arms.py`). Backends are
  `[label=]kind[:key=value,...]`, encoding levels `none` or `a+b`, and `--grid key=v1,v2` adds a plain
  sweep parameter crossed with every arm (the widths). Every combination goes through `load_config`
  on the base config (`--cfg` or a registry pair) at every grid point. The kept ones become arms
  named `<label>_<node>_<bias>_<rotary>`, with empty axes left out. All arms set the same keys; a
  backend without a setting gets the base value. The header lists the rejections, grouped by reason,
  and the trials whose checkpoint names would coincide. The name carries the PEs, the backend and
  its head width, but not feature counts, degree or widths, so parallel agents would overwrite each
  other's checkpoints. Sweeps never re-validate a trial (`apply_sweep_config`), so this is the only
  check before the model build.
- **A width only one backend reads goes into its settings** (`wide=linear:attention.head_dim=64`),
  not into the grid: in the other arms a grid key that nothing reads varies nothing.

### 3 — low-rank biases

- **The `factors` form**, starting with the spectral bias: `U = √N·V·diag(h_h(λ))` and `W = √N·V`,
  zero at CLS and padding. That is exactly today's bias as k extra q/k dims per head.
- **`pe.bias: [fourier]`** (new; its own bias rather than a `distance.kind`, since bias forms are
  declared per encoding): `b_h(Δ) = Σ_r a_{h,r}·cos(ω_rᵀΔ)`, optionally plus
  `c_{h,r}·sin(ω_rᵀΔ)` for a directional prior.
  - The frequencies sit on a polar grid: directions over 180°, radii geometric over
    `[1/max_wavelength, 1/min_wavelength]`.
  - `isotropic: true` ties the coefficients across directions, giving a near-radial bias.
  - It costs 2R dims, starts at zero, and is also usable by dense and Exphormer as an alternative
    to `profile`.
- **Payoff:** the kernel backends get the spectral bias and a distance-like one, and fused attention
  stops materialising a float bias.
- **Verify:**
  - The augmented logits equal the dense spectral-bias logits to 1e-5.
  - CLS and padding stay zero.
  - `fourier` is zero at init and translation-invariant, and an isotropic fit reproduces a target
    radial profile within its band limit.
  - Performer plus spectral is unbiased as in Stage 2.

#### How Stage 3 was built

- **`AttentionBias.factors`** returns `(U, W)`, each `[B·H, S+1, r]`, laid out like the dense
  bias. An encoder brings its low-rank form as a `factors` method; the spectral bias's has width k
  (`SpectralBias(k=)`), declared as `factor_dim`.
- **Balanced split.** `balanced_factors` splits each weight between the two sides with a detached
  scale `sqrt(|w| + 1e-3)`: exact in the product and its gradient, and at the zero init one side is
  0 and the other small, so the Performer's estimator starts with almost no extra variance.
- **A bias's size costs Performer accuracy.** The estimator's variance grows with `exp(|x + y|²)`, and
  folded biases add to those norms. Measured at m = 16384: TV 0.014 against dense with the same bias
  at a bias range of ±1.2, and 0.05 at ±3. Large learned biases want more features; watch the TV
  diagnostic (Stage 1) on Performer arms with biases.

- **The Fourier bias is `pe.bias: [fourier]`**, with its `pe.fourier` block: `num_directions` 8
  × `num_scales` 16 over `[min_wavelength 200 µm, max_wavelength]`, so 256 factor dims. Its
  `max_wavelength` 0 is derived as twice the largest slide diameter (`resolve_fourier_range`,
  chained in `BaseModel.__init__` after RoPE's). `isotropic` (the default) has one cosine
  coefficient per head and scale, averaged over the directions; off, there are a cosine and a sine
  coefficient per head and frequency. Phases are float64; the frequencies are a buffer.
  `FourierBias.bias_at(offsets)` reads the learned function out.
- **Grid chosen by fit.** With radial targets within 3 mm (a ring at 1 mm, an exponential decay, a
  step at 2.5 mm), 8 directions × 16 scales over 200 µm–12 mm fit with R² 0.94–0.99. 16 directions
  reach 0.96–0.998 at twice the width. Isotropy only holds while a wave's phase across the
  distance is resolved by the directions, so short wavelengths at long range average out rather
  than being radial.
- **Coordinates, not distances.** A bias with `reads_coordinates` gets the padded positions
  (input key `pos_coords`) rather than the `cdist` the distance bias takes. The registry still says
  `requires="pos"`, so the spatial-key check applies.

### 4 — global tokens and a null key

- **`global_tokens: R`**: R learned tokens built like CLS, placed between the cells and CLS inside
  `forward` only.
  - They are never blocked, get zero bias rows and columns, and sit at the origin for RoPE.
  - They are stripped from everything `forward` returns (outputs, maps, relevance), so the `[S+1]`
    layout outside `forward` is unchanged. Relevance is rolled out over the full internal maps, then
    stripped.
  - These are Exphormer's virtual nodes, ETC's global tokens and ViT registers (Darcet et al., ICLR
    2024).
- **`null_key: true`**: torch's `add_zero_attn`, a zero key (logit 0) with a zero value, i.e.
  softmax "off by one".
  - Relax the RoPE guard at `transformer_utils.py:343`: the zero key is appended after the rotation
    (`:358`).
  - Delete `evaluate()`'s unused diagonal statistics (`transformer_encoder.py:314-325`), which would
    raise on the wider map.
- **New per-cell readouts:** mass on the global tokens, and null mass, i.e. how much a cell draws
  from long-range context at all, as a map.
- **Why:** background.md names attention sinks as a reason raw attention is not interpreted, and
  CLS is the only sink today.
- **Backends:** works with dense, kernel (where `φ(0)` estimates `exp(0) = 1`) and Exphormer.
- **Verify:**
  - R = 0 with the null key off is IDENTICAL.
  - Stripped rows sum to 1 minus the global-token and null mass.
  - CLS is never blocked, and the long-range mask tests pass with R > 0.
  - RoPE with the null key runs, and relevance keeps its shapes.

### 5 — Exphormer-style sparse attention

Each token attends only along an attention graph built from three edge sets (Shirzad et al., ICML
2023):

- **graph edges:** the neighbour graph the GNN uses (`graph_edges`);
- **expander edges:** a random ~d-regular graph (`degree: d`, even). Two alternatives that are not
  in Exphormer are not built yet: `uniform` (d sampled keys per query) and `distance` (keys
  stratified in log distance so far cells are not crowded out);
- **virtual nodes:** CLS (and Stage 4's global tokens once they exist), linked both ways to every
  cell.

The long-range mask removes its blocked set from this graph, whatever an edge's source. Under
`CombinedModel` with the mask on, the graph edges therefore vanish by themselves and the attention
is purely long range; under `GlobalModel` they stay, as in Exphormer.

- **Mechanics:** an edge list in padded positions, built from `index_nodes`, with a per-head scatter
  softmax over each query's incoming edges (`torch_geometric.utils.softmax`). O(E·head_dim).
- **Edge types:** each edge carries its type (graph, expander, virtual). With `edge_type: bias` that
  is a learned per-head bias, zero at init. With `edge_type: key`, the keys are scaled by a
  per-type embedding, as Exphormer does.
- **Fixed or resampled edges:** `resample: false` draws the expander on the full graph from a seed
  the graph itself determines, so it is the same every time the graph is seen, and restricts it to
  the kept tokens. `resample: true` draws a fresh one each training step from the backend
  generator. Eval always uses the fixed one, so maps reproduce.
- **Encodings:** biases through `pairs` (distance per edge, spectral per edge, SPD from a table);
  RoPE rotates before the gather; node PEs as usual.
- **Maps:** scattered to dense at eval, zero off the attention graph.
- **Reach:** one layer reaches d + G keys per cell, and information travels further through the
  virtual nodes and through more layers (expanders mix in O(log S) hops). Direct pairwise attention
  between two given distant cells is mostly unobserved at small d. `degree` and `num_layers` are the
  knobs.
- **Verify:**
  - The complete graph (every open pair an edge) reproduces dense to 1e-5.
  - Blocked and padded keys are never attended, whatever the edge source.
  - The precomputed expander is d-regular before restriction and identical across calls and
    epochs; resampling uses only the backend generator.
  - `pairs` biases equal the dense bias on the edges.
  - The edge-type bias is zero at init.

#### How Stage 5 was built

- **Before Stage 4, with CLS as the only virtual node.** The plan put global tokens first because
  Exphormer's virtual nodes are global tokens; CLS already is one, linked both ways to every cell,
  so Exphormer runs with one virtual node now. Global tokens join the pattern when Stage 4 adds
  them.
- **The expander is not precomputed.** Instead of an attribute on each `Data`, it is drawn on the
  fly from `tl.graph_seed`, a SplitMix64 mix of `optim.seed`, the graph's size and its edges. The
  same graph -- in training, in evaluation, rebuilt by `get_model_output` -- gets the same expander
  whatever batch it lands in, with no dataset hook. Python's `hash` was avoided because its tuple
  hashing may change between versions. `tl.random_expander` builds it from `degree / 2` random
  permutations in both directions (so `degree` is even), dropping self-loops and repeats, so a few
  nodes end up with fewer than `degree` edges.
- **`tl.attention_pattern` builds the attention graph** as `(graph, query, key)` positions with edge
  types `("graph", "expander", "virtual")`. A graph edge `src -> dst` is the query `dst` attending to
  `src` (PyG's message direction). Every row, padding rows included, attends to CLS, so no softmax
  is ever empty. Blocked pairs are removed by code, and a pair found twice keeps the graph type.
- **A third bias form, `pairs`.** Distance (profile and linear), spectral and Fourier read off any
  list of pairs, so Exphormer carries every bias. The distance biases take each pair's distance
  from the padded coordinates, with no `cdist` and nothing `S x S`.
- **The backend** (`ExphormerAttention`) scatters a softmax over each query's pairs
  (`torch_geometric.utils.softmax`). Its edge types act as a per-type, per-head bias
  (`edge_type: bias`, zero at init) or as a per-type scaling of each key dimension (`edge_type: key`,
  ones at init), Exphormer's form; both start as plain attention along the pattern. Attention
  dropout is applied to the pair weights, as dense attention applies it to its map.
- **`AttentionSpec.pattern`** holds the pattern builder, so the transformer stays backend-agnostic:
  for `mask="pattern"` it builds the blocked pairs, asks the spec for the pattern, and computes the
  biases on it.
- **Verified by reduction to dense.** With every open pair as the pattern it is dense attention,
  to 1e-5 at the attention level (mask, padding, subsampling, all three biases, RoPE). End to end, an
  Exphormer model on a complete graph without the long-range mask equals the dense model at one
  seed, distance bias included.

### 6 — fused exact attention

`attention.fused: true` (dense only): training and validation steps take SDPA (`need_weights=False`),
and `evaluate()` keeps the explicit path. Key padding is merged into the mask once per batch rather
than once per layer.

- **Kernel eligibility.** In fp32 only the memory-efficient kernel can be chosen (FlashAttention
  needs fp16/bf16). If it refuses the head width, SDPA silently falls back to the math kernel and
  saves nothing. Check it in a GPU test with `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`, and
  zero-pad the heads if needed (logits unchanged, pass `scale`).
- **SDPA's boolean mask means "attend"**, the opposite of MHA's. A test hands it the MHA-convention
  mask and must fail.
- **Dropout:** the fused kernel has its own RNG stream, so fused runs are paired with explicit runs
  by seed, not identical.
- **Verify:**
  - Outputs and gradients equal the explicit path to 1e-5 at dropout 0, with the mask, both biases,
    RoPE, padding and a subsampled graph.
  - The validator rejects fused together with the entropy loss.
  - On the cluster: memory and steps/s against the explicit path over a range of S.

### 7 — inducing points (optional)

Each layer runs two cross-attentions (ISAB, Luna's pack/unpack): m learned latents attend over the
cells, then the cells attend over the latents. O(S·m).

- **Maps:** the product of the two maps is a row-stochastic, rank-m cell × cell surrogate.
  Chefer et al. give relevance rules for cross-attention.
- **The long-range mask cannot hold:** a latent pools the neighbourhood and hands it back, so the
  validator rejects the combination.
- **Positions:** biases and RoPE need latents with positions. A learned or gridded set of anchors
  would enable both, as a follow-up.
- **Verify:**
  - Permutation equivariance.
  - The composed map is row-stochastic.
  - The validator rejections have tests.
  - It trains finite.

### 8 — beyond the window (not planned in detail)

Larger windows or whole slides, so interactions are no longer cut at window boundaries. Three
blockers sit outside the attention code:

- **Evaluation is S² by design.** The dense `_attn_matrix` obsm and the O(L·S³) relevance rollout
  need their own plan: top-k pairs, cell-type flow, or CLS-only relevance.
- **The k-hop closure is densified.** Stage 2's `blocked_pairs` replaces it.
- **Exact attention at 10k–50k tokens** needs FlexAttention, with `score_mod` biases and a
  block-sparse mask, plus torch ≥ 2.5 on the cluster and `torch.compile` with dynamic shapes. A
  Euclidean-radius mask would be simpler but contradicts the 2026-10-03 decision that the mask
  covers exactly what the GNN mixed.

## Test matrix

The axes, all independent:

| axis | values |
|---|---|
| backend | dense, performer, linear, exphormer, (fused), (inducing) |
| add-ons | `global_tokens` 0 / 1 / 4, `null_key` |
| node encodings | none, naive, sinusoidal, lap, rw, and combinations |
| biases | none, distance (`profile`, `linear`), fourier, spectral |
| rotary | none, rope (`axial`, `mixed`) |
| width | `model.n_embed`, `n_heads`, `attention.head_dim`, `num_layers`, plus `performer.num_features` and `exphormer.degree` |
| model | `GlobalModel`, `CombinedModel` (single or dual decoder), long-range mask on / off |
| dataset | each with its own base config |

**Datasets.** Each dataset brings one base config: data path, `spatial_key`, `sample_key`, graph,
masking, `max_seq_len` or windows, and its probe targets. The arms files are shared across datasets.
Nothing in the arms names a dataset, and evaluation reads its targets from config.

**Protocol for a matrix this large:**

1. **Screen** with the generated arms at one seed per combination (`scripts/make_attention_arms.py`,
   run against each dataset's base config, since that is the config it validates).
2. **Confirm** the promising combinations at ≥ 3 seeds, compared paired by seed.
3. **Compare efficient backends with dense at a size where dense still runs**, then take the
   survivors to sizes where it does not.

Width axes (`n_embed`, `head_dim`, `n_heads`, `num_layers`, `num_features`, `degree`) are plain
dotted sweep parameters (the generator's `--grid`). List-valued and backend-dependent choices go
through arms.

**What to read**, the same on every dataset:

- masked reconstruction;
- the configured probe targets (local vs. global vs. `pe`);
- Stage 1's span, sink mass and effective keys;
- steps/s, cells/s and peak GPU memory per training epoch (`optim.log_efficiency`, logged under
  `efficiency/` by `train._utils.EfficiencyMonitor`), and the largest S that fits;
- the flow control's null pairs where configured;
- ground-truth scoring where the dataset has it (planted edges on the synthetic ones).

A probe gap alone settles nothing about range (the contrastive plan's finding); the ablation is the
instrument.

**First runs: synth_spot, one backend per sweep (2026-10-12; results read the same day).** The run scripts are
`notebooks_and_scripts/performer_test.{py,yaml}` + `performer_sweep.yaml` and
`notebooks_and_scripts/exphormer_test.{py,yaml}` + `exphormer_sweep.yaml`. Each is pe_test_2's
setup with the attention changed and 40 epochs instead of 100, still SCELoss, with the accelerator
fixed to `gpu`. The arm names are pe_sweep_2's. The sweeps cover every single PE the backend carries
plus the no-PE baseline: the Performer gets 10 arms (no distance bias), Exphormer 12. Both add
`fourier` and `fourier_aniso`. The scripts validate every arm before queueing, because sweeps do not
re-validate. They only write files; comparing backends is left to reading those files. They drop
pe_test_2's gene-vector probe, both online (too slow every epoch) and post hoc; the gene ranks and
decoder loadings cover reconstruction per gene. Besides the rest of pe_test_2's evaluation they
write:

- Stage 1's diagnostics on the test slides, with the Performer's TV to softmax;
- per layer, the raw attention on the planted pairs: `coverage` (with `open_share` as the
  edge-blind reference), `rank_open` and `lift_open`;
- the training summary (`efficiency/*`, final `val_*`).

Under the long-range mask, Exphormer's k-NN edges all drop out, which leaves the expander (degree 8)
and CLS. Its `coverage` of long-range pairs should therefore sit near `open_share`, about 8 / 2500.

**What round 1 showed** (with the dense PE sweep, `PE_test_sce`), one seed per arm:

- **Attention stayed uniform in every backend.**
  - The Performer spread over ~2,460 of ~2,500 keys, and Exphormer over 7.9 of its 8 expander
    partners. The span (2.6 mm) is the mean distance between two spots.
  - Planted senders ranked at chance, and relevance was flat with distance. The Performer's TV to
    softmax (0.006) only says both maps are uniform.
  - So neither backend was really tested.
- **The biases never learned.** Their coefficients stayed at 0.01–0.03, and several bias arms are
  the same model to three decimals. Only `naive` had an effect, on position-derived readouts.
- **The decoders scored below the gene-mean floor.** The val SCE of the attention runs was
  0.107–0.121; predicting each gene's training mean scores 0.082.
- **Raw SCE barely rewards context.** A neighbour average scores 0.081, against 0.082 for the gene
  mean.
- **Every sweep's `none` arm was a different initialisation.** `main_sweep` built the model before
  seeding, so the first trial started from another random state than all later ones (relative weight
  distance ~1.1 vs ≤0.08 among PE arms). The same held across sweeps. Fixed 2026-10-12: each trial now
  seeds before it builds its model.
- **The efficiency monitor's peak memory included validation and the probes.** Fixed the same day:
  it is now read at the end of each training step.

**Round 2 (2026-10-12, not yet run):** `notebooks_and_scripts/{dense,performer,exphormer}_test_2.{py,yaml}`
+ `*_sweep_2.yaml`. The same arms and readouts, with three changes:

- the long-range mask off, as a sanity check that attention can learn structure at all;
- `optim.lr_warmup: 8`, the PE sweep's schedule shape at 40 epochs;
- width 64 with four 16-wide heads (`model.n_embed`, `n_heads`, `attention.head_dim` in the config).

Dense runs the same 12 arms as Exphormer. Each backend and round has its own W&B project, and
checkpoints and results go to `13_10_results/attention_<backend>_2`.

## Why attention stays uniform, and what to change (2026-10-12)

Round 2 (results in `attention_{dense,performer,exphormer}_2`) changed three things: the long-range
mask off, width 64 with 16-wide heads, and a warm-up of 8. Thanks to the seeding fix, all 34 runs
started from the same weights. The outcome:

- **Every run converged to the same function.** Val SCE is 0.08336–0.08342 everywhere, still above
  predicting each gene's training mean (0.0816) and a neighbour average (0.0811).
- **The backends ended as the same network.** Gene ranks correlate at 1.00, decoder loadings agree
  to three decimals, and the weights differ by 0.2–8%.
- **Attention stayed uniform with the mask off.** Dense and Performer spread over ~2,480 of ~2,500
  keys, Exphormer over 15.95 of 16. RoPE heads attend farther, not nearer.
- **The biases learned only a faint preference.** Their parameters stay ≤ 0.024. Dense spectral,
  Fourier and distance biases rank immediate neighbours at the top (rank up to 0.998), but add only
  ≤ 17% weight. `naive` ranks planted senders at 0.63–0.68, with ~1% extra weight. Neither changes
  the loss.
- **The probes improved, all through the GCN.** Region 0.987 (neighbour baseline 0.989), M9 stripe
  0.80, source distances 0.43, R3/R4 0.38. Local equals global and is the same in every run.
- **Exphormer's relevance AUROC of 1.0 on short-range pairs is by construction.** Those pairs are
  its k-NN edges.

Uniform attention in three independent implementations is a design problem, not an attention bug.
The optimizer gets every parameter, and the global decoder's loss reaches the transformer.

### Diagnosis

1. **The tokens are nearly identical across spots.**
   - From the data alone: 84% of each spot's `log1p_norm` vector is the shared mean profile; two
     random spots have cosine 0.844. Only ~15% is spot-specific, and the slide mean adds 0.5%.
   - Masking 30% of entries to −1 adds random per-spot noise larger than the biological
     differences: cosine drops to 0.435.
   - The GCN averages neighbours, so its outputs keep the shared component. The per-token
     LayerNorm (`norm_input`) does not remove a component shared across tokens.
   - **Evidence:** RoPE rotates queries and keys by position, whatever the content. It is the only
     arm whose spread matches random attention on distinct tokens: ~1,950 effective keys, against
     ~1,950 predicted for a logit spread of ~0.5 at xavier init. Every content-based arm sits at
     ~2,480, a logit spread of ~0.08.
   - With identical keys the attention output is the slide mean whatever the weights, so W_q and
     W_k get no useful gradient. That is a fixed point. Their norms stayed at about the
     initialisation value: 5.6, against 5.66 expected for xavier.
2. **Training is tiny.** 24 training slides at batch size 4 give 6 optimizer steps per epoch, so 40
   epochs is 240 steps, about 1–2 min of GPU time. The decoders only reached the mean profile.
3. **The task never needs the global model.**
   - 30% gene masking leaves each spot's other genes and its neighbours visible, so the GCN suffices.
   - Raw SCE is ~85% satisfied by the mean profile: a neighbour average beats the gene mean by only
     0.0005.
   - Under the dual decoder the global head predicts the same target as the local head, so copying
     local is optimal for it.

### Levers, by expected impact

Every code change goes in as an opt-in config flag with tests, so defaults stay unchanged.

1. **More optimizer steps (config only).**
   - Batch size 1 and ~200 epochs give ~4,800 steps, 20× round 2 and still minutes per run.
   - Set `lr_warmup` to ~20 (10% of the run).
   - Set `probe.every_n_epochs` to ~10, or the probes dominate the runtime.
2. **Centre the tokens per slide before attention (small).**
   - After `norm_input`, subtract each graph's mean token from its spot tokens, leaving CLS alone;
     optionally rescale.
   - Queries and keys then compare what differs between spots, which breaks the fixed point.
   - Add a diagnostic: mean pairwise cosine of the tokens entering each layer.
   - **Implemented 2026-10-13, unverified on data:** `attention.center_qk`. Queries and keys come
     from `center_tokens`: each sequence's tokens minus the mean of its cells (padding and CLS left
     out), layer-normalised without affine weights. Values and the residual stream keep the raw
     tokens, so the slide-level information stays. Works in every backend; the backends now accept
     a separate value tensor. Checkpoint tag `-cqk`. The diagnostics log `attn_token_cosine_l<l>`:
     the mean pairwise cosine of the cell tokens entering each layer, before centring.
   - Verified by tests:
     - centred attention is unchanged by a shared shift of every token, and uncentred is not;
     - the values are the raw tokens;
     - near-identical tokens give uniform attention without centring and ≥10% fewer effective keys
       with it, in dense, Performer and Exphormer.
     Six mutations, each caught; harness IDENTICAL.
3. **Spatial block masking (moderate).**
   - Hide whole spots in discs wider than the GCN's reach (radius > 283 µm). The local model can't
     fill them in from neighbours, so the global model must use distant context.
   - This is the main lever for "global learns what local can't". Expect the gap on the
     long-range and spatially variable programmes.
   - Trade-off: receptors are hidden too, so the sparse responses can't be learned inside discs.
     Keep gene masking selectable, or combine both.
   - Touches mask drawing in `GraphAnnDataModule` (train and the fixed val mask), `entry_mask` and
     the probes.
4. **Residual global decoder (moderate).**
   - The global head predicts the local head's error, with no gradient into local; the final
     prediction is local plus global.
   - The global component is then rewarded only for what local misses, which is the local-vs-global
     question itself.
   - Touches `DualDecoderCombinedModule` and how the losses and metrics are reported.
   - **Implemented 2026-10-13, unverified on data:** `model.decoder.global_residual` (needs
     `dual_decoder` and node level). The global head's prediction is `combine_global`: the local
     prediction, detached, plus the global decoder's output, for the same cells in the same order.
     Training (`_common_step`) and inference (`get_model_output`, which decodes outside the step)
     both report that sum, so `global_loss`, the metrics, `y_pred_global` and the gene ranks all read
     the combined prediction. The local decoder answers to the local loss alone; the GCN still gets
     the global loss through the tokens, as before. Composes with `optim.center_targets` (the sum is
     centred like any prediction). Checkpoint tag `var-res_`.
   - How to read it: `global_loss` below `local_loss` is what the global component adds. The global
     decoder starts from random weights, so the gap starts negative and must first recover.
     Zero-initialising its last layer would start the gap at 0; not done, add it if the curves are
     hard to read.
   - Verified by tests:
     - with the same weights, the global half equals the local half plus the plain global output;
     - a loss on the global half leaves every local-decoder gradient at zero and reaches the global
       decoder;
     - inference on a subsampled slide reports each kept cell's own local prediction plus the global
       output;
     - the validator requires the dual decoder at node level.
     Eight mutations, each caught; harness IDENTICAL.
5. **SCE on gene-centred targets (small).**
   - Subtract the training gene means from prediction and target before the cosine, so
     context-dependent deviations drive the gradient.
   - Z-scoring the *input* instead would put the −1 mask value inside the data range (CLAUDE.md,
     masking).
   - **Implemented 2026-10-13, unverified on data:** `optim.center_targets`. `train()` takes each
     gene's mean over the training graphs (`train_gene_means`), and the plan subtracts it from
     prediction and target before the reconstruction loss and the dual decoder's split losses,
     never before the metrics. So `<mode>_loss` is centred while `val_scaled_cosine_error` stays the
     raw SCE. The validator allows it only for regression with a row-normalising loss. Checkpoint
     tag `var-ctr_`.
   - Verified by tests:
     - the loss is the SCE of the centred values over the same masked entries;
     - predicting the mean profile scores 1.0 instead of < 0.05 raw;
     - the split losses are centred and the metrics stay raw.
     Six mutations, each caught; harness IDENTICAL.

### Plan

- **Round 3a:** levers 1, 2 and 5 on a reduced grid (dense and Performer × none / naive / rope /
  fourier), plus ablation arms on `none` that switch off lever 2, lever 5 or both, to tell them
  apart. Question: does attention leave uniform once the tokens are centred, the loss rewards
  deviations from the mean profile, and training is long enough? Read the effective keys, the token
  cosine, `planted_edges` lift and rank, and whether val SCE drops below 0.0816. Lever 5 belongs
  here, not in 3b: it changes what the whole model is asked to learn, not what global must do that
  local can't.
- **Round 3b:** add levers 3 and 4, with the long-range mask back on. This is the actual "global
  learns something the local can't" experiment: the local-vs-global gap per programme, the probes
  on hidden discs, and planted long-range edges.
- **Controls:**
  - evaluate an untrained checkpoint through the same script, to see what a random GCN already
    preserves;
  - run several seeds once differences appear.

Status: diagnosed 2026-10-12 (the user called it urgent). On 2026-10-13 levers 2, 4 and 5 were
implemented, each unverified on data; lever 3 and the round-3a configs are in progress.

## Predictions to check, not assume

1. **Kernel attention matches dense where the long-range signal is smooth, and loses where a few
   distant cells set a response.** synth_spot's sparse programmes are one designed instance.
   - Performer's relative variance is `(exp(‖x + y‖²) − 1)/m`, largest for exactly the high-logit
     pairs a sharp attention needs.
   - Linear attention loses more than Performer at equal width.
   - Both close part of the gap with width: `head_dim` for linear, `num_features` for Performer.
2. **Head width matters little for dense attention and a lot for linear attention**, because exp
   already makes dense maps full rank.
3. **Exphormer with virtual nodes matches dense on prediction, but loses planted-edge AUC as
   `degree` falls**, because most pairs are unobserved in any one layer. Extra layers buy back reach
   but not pairwise interpretability.
4. **Global tokens or a null key take mass off CLS without costing reconstruction.** If Stage 1
   finds CLS mass small, they are cosmetic.

## Invariants

- **One mask in every backend.** The blocked set is the k-hop closure the local module mixed,
  computed on the full graph and then restricted to the kept tokens. CLS is never blocked, and
  padding contributes nothing.
- **Booleans:** MHA's True means blocked; SDPA's and FlexAttention's True means attend. Convert in
  exactly one place per backend, by indexing.
- **Biases:** the same additive logit on every open pair in every backend. CLS and padding rows and
  columns are zero. The dense layout is left-padded, CLS last, graph-major.
- **Rotation** is applied to q and k before any kernel, feature map or gather. CLS, padding and
  global tokens sit at the origin.
- **Explicit path:** every backend has one, computing the same function and its `[B·H, S', S']`
  maps, and `evaluate()` uses it.
- **Global tokens and the null key never leave `forward`.**
- **Backend randomness** happens only in training forward passes and only from the backend
  generator.
- **Defaults** (dense, default `head_dim`, not fused, no global tokens, no null key) give the same
  parameters, init order, RNG draws, checkpoint names and numbers as today.

## What the backends change about the measurements

- **Kernel** maps are smoother than softmax maps: Performer's are an estimate, linear's a different
  kernel altogether. Read a flow difference against dense together with Stage 1's TV distance and
  the flow control's null pairs.
- **Exphormer** reads 0 off its attention graph. Score planted edges over the pairs the graph
  contained, and report coverage alongside.
- **Global tokens and the null key** leave cell rows summing to less than 1. Net flow is reported
  from the raw rows and from rows renormalised over cells.
- **Fused** changes nothing.
- **Inducing** flow comes from a rank-m surrogate.

## Open questions

- **Whole-slide evaluation format** (Stage 8): top-k pairs, cell-type flow, or CLS-only relevance?
- **Inducing points:** free latents, or spatial anchors?
- **Exphormer's edge-type handling:** additive bias or Exphormer's key scaling? Both are knobs; keep
  both only if they differ.
