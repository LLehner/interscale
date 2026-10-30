# Hierarchical Coarse-to-Fine Attention: Implementation & Sweep Plan

## Status

The plan below is the user's (2026-10-26); this section and the next record how it was built. The rules are the other
plans': a stage is `done` only when a run has been read; "implemented, unverified" is a real state.

| part | state | verified by | date |
|---|---|---|---|
| T0 input sources | **exists, no new key**: `raw` = GlobalModel with the `Linear` token; `pca` = `Precomputed` on `obsm['X_pca']`; `gnn_joint` = CombinedModel; `gnn_frozen` = CombinedModel with `optim.schedule: local_then_global` (or a GCN embedding through `Precomputed`). The hierarchy is an option of the global component, so it runs in every model type | -- | 2026-10-26 |
| T1 hierarchy builder | **implemented, unverified on data**: `kdtree`, `random`; top level `tree` or `patches` (with `min_cells` merging); `shift` (kdtree on coordinates turned 45 degrees, the grid moved half a patch); levels as counts or cells per cluster; cached per slide by a fingerprint of its positions; build time and cluster sizes recorded. **2026-10-30, branch `niches`, unverified on data:** `obs_label` with `obs_key` and `split_by_space` none / components / kdtree (components on the slide's own graph, `min_cells` merging into the neighbour with most shared edges, kdtree pieces of at most the first level's size; `tl/niches.py`, shared with pooling and the niche mask; lung_plan.md L2), the cache keyed by labels and edges, a warning when routing's cap cuts the chosen parents. The plan's `kdtree` split is "within each label"; built as within each connected patch (compact pieces of one place). Not built: `quadtree`, `metis`, `leiden_spatial_expr` | tests 1, 10-11 (`test_hierarchical_attention.py`); plan test 10 and the singleton regression under `obs_label` in `test_niches.py` | 2026-10-30 |
| literature, second round | **collected 2026-10-29** (section 10): H1-H9 (NSA, its point-cloud version, Erwin, MoBA, Focal / PVT, FMMformer / H-Transformer-1D, Point Transformer V3, Routing Transformer / Combiner, spatial-omics multi-level models); stages H-α (window branch + per-cell gate, focal candidates), H-β (contiguous tree order, per-layer rotation), H-γ (additive near / far read-out) proposed, after the CosMx `long` read | -- | 2026-10-29 |
| T2 pooling | **implemented, unverified on data**: `mean` and `attn_pool` (a learned query per level, starting as the mean), over visible cells only; `count` / `spread` features. Not built: `multi_token` | tests | 2026-10-26 |
| T3 layer, prior modes | **implemented, unverified on data**: `none`, `soft` (size correction toggle), `factorized`, `factorized_percell`, `direction_prior`; lambda / mu per head, global or fixed, per level; `prior_refresh` per_layer / once; `share_level_params`; a distance bias per level (`coarse_bias`: shared, own, none). The must-pass regression (lambda 0, singletons, dense) equals the dense transformer in predictions and every gradient | tests 2-4, 7 | 2026-10-26 |
| T4 routing | **implemented, unverified on data**: per parent, the children of its top-k targets (max over heads), of its spatial neighbours (itself included) and of exploration parents (uniform or prior-sampled, training only, private generator), capped by `cap_per_query`; a padded gather a block of parents at a time, the cell level recomputed in the backward pass. With every parent selected it equals the dense model in every prior mode; no tensor over the cells' pairs is formed. Not built: FlexAttention, `threshold`, `gumbel`, the per-cell kNN LSE merge | test 5 (and no-matrix) | 2026-10-26 |
| T5 local path | not built (the windowed narrow heads and the routed neighbour clusters are the local path) | | |
| T6 readout | **implemented, unverified on data**: `none`, `concat` (zero-initialised: starts as `none`), `transformer` | tests | 2026-10-26 |
| T7 regularisers | not built; the diagnostics (coarse entropy, regions never selected, lambda / mu) are in the evaluation | | |
| T8 export | **implemented**: `read_vec`, its exact coarse + refinement split, entropy, mean distance, top-5 targets, per slide as npz; quiver plots with planted senders; **2026-10-27, unverified on data:** per head cell mass, top-5 targets and `window_vec_heads` (the head's own trained window alone over the same keys, scaled to its cell mass: the baseline in place of an untrained twin), the planted pairs' programmes, and the receivers' integrated-gradients dependence (`receptive_field.receiver_dependence`, retried with more points over a 0.1 completeness gap); `fields_by_class.py` draws them (rows: arrows, minus own window + dependence, top-5 per head + top-5 by contribution) | tests 8-9, tests (window = attention under flat content, per-head top-5, dependence retry; 7 mutations killed) | 2026-10-27 |
| T9 scenarios | **synth_spot as MIX** (decided 2026-10-26); its range classes stand for S1 (`short`), S2 (`mid`), S3 / S4 (`dense` / `sparse`), directional; the loss caveat is met by programme masking. Not built: the scenario generator (S0, S5-S7 as separate slide families, a parametric S2, SCALE) | | |
| T10 metrics | **implemented** (`evaluation.hierarchy_readouts`): sender lift, pair AUPRC, arrow angle, routing recall per level, distance profile, routing health; the dense model is read by the same code from its maps. CosMx: hidden-cell scores by disc depth against k-NN smoothers (no dense distance matrix). Not built: LR plausibility, replicate consistency | tests | 2026-10-26 |
| T11 sweeps | **Written 2026-10-27, not run:** synth_spot Stage B (`HIER_PART=B`, 4 arms x 3 seeds: soft lambda 0, soft `coarse_bias: own`, both, fpc both; twins of all four; results `27_10_results/hier_synth_spot_B`), the fields re-read of Stage A's full_s42 / hier_none_s42 (`HIER_PART=fields`, `INTERSCALE_MODE=evaluate`), CosMx `long` (fold 0, 10,000 epochs: gcn, hier_none, hier_soft_l0_own, hier_none's twin; `27_10_results/hier_cosmx_long`). Checked: every arm validates, checkpoint names distinct, a stub evaluation on generated synth_spot. **Stage A complete, CosMx fold 0 read 2026-10-27** (notes.md): synth_spot full 0.947 > hier_none 0.939 > soft_dir 0.892 ≈ soft 0.887 > fpc 0.864 ≈ dual 0.862 > fac 0.771 > gcn 0.512 (3 seeds); priors slow M11 (seed-dependent switch), direction prior adds nothing. CosMx (routed, ~300 cell keys per query): fac / soft / fpc 0.18-0.19, gcn 0.07, all below knn128 0.244 and the section mean 0.209; 6.5 GiB, ~1 h per run; near-uniform attention, windows and lambdas at their start: undertrained. Next: CosMx 10-20x longer with hier_none and a twin; priors only with lambda_init 0 and `coarse_bias: own`. Earlier: **Stage A first read 2026-10-26** (20 of 24 runs, notes.md): `hier_none` within 0.01 of `full` (0.939 against 0.947, programme mean 0.779 / 0.789), faster; every prior costs (fpc 0.864, fac 0.769), most likely through the windows shared with the coarse levels confining narrow heads to their own cluster; `hier_soft_dir` invalid (gradient explosion in the direction prior, fixed the same day, to rerun); `hier_soft` missing. Submitted 2026-10-26: two job-array sbatch files (`slurm/synth_spot_hier.sbatch`, `slurm/cosmx_hier.sbatch`; yamls and scripts flat in `yaml/`, `scripts/`; local copy in `notebooks_and_scripts/slurm/hierarchical/`). synth_spot Stage A (8 arms x 3 seeds = 24 runs, 300 epochs, 8 h per run) started first; the one-seed screen (18 arms, `HIER_PART=screen`) and CosMx (fold 0: gcn, hier_soft, hier_fac, hier_fpc, 1,000 epochs, 24 h per run; builds `inputs.h5ad` itself) are queued or to follow. Checked only on stub data | stub runs | 2026-10-26 |

## Implementation in the repo (2026-10-26)

Decisions of the user (2026-10-26): synth_spot as the MIX scenario today, the separate scenario generator later; split
sweeps (one job per run); baselines GCN, the dual model (`add_ctx_dist`) and the dense windowed model, no Performer /
Exphormer; 300 epochs on synth_spot; on CosMx one section (one donor) per forward pass, in training too; cell masking with
discs only; ~2,000 steps (1,000 epochs, 2 training sections per fold) and a lean evaluation, the comparison readouts later.

- **Config:** `model.global_component.parameters.hierarchy` (`kind: none` by default: nothing built, harness unchanged), the
  plan's keys flattened (`lambda_init`, `lambda_learnable`, `candidates.topk`, ...). `levels` (clusters per coarse level) or
  `cells_per_cluster`. Checkpoint tag `hier-...` with every non-default setting. `optim.check_val_every_n_epoch` (default 1)
  came with it, for whole-section runs.
- **Code:** `module/global_modules/hierarchy.py` (builder, cache), `hierarchical_attention.py` (`HierarchicalEncoderLayer`,
  a `CustomTransformerEncoderLayer` whose attention block is the coarse-to-fine pass, with the dense layer's projections),
  `transformer_encoder.py` (the context in place of the S x S mask, `recording_hierarchy`), `evaluation/hierarchy_readouts.py`.
- **Deviations from the plan, and why:**
  - CosMx sections hold 47k-100k cells, not 0.5-1M: the Stage 0 grid is scaled down (regions of ~250-1000 cells, clusters of
    25-64). Exact dense attention fits there (`attention.kind: fused`), so CosMx has a `full` arm too, at O(cells^2) compute.
  - Routing by a padded gather, not FlexAttention: exact, tested on the CPU, no `torch.compile` on the cluster. The routed
    keys of a parent are chosen by the parents' weights maximised over heads (one key set per parent for every head).
  - CLS queries at level 0 only and stays an open key with bias 0 at every level (in the factorized modes its own
    one-member cluster), so the singleton hierarchy is the dense transformer exactly.
  - The cell level blocks a cell's own key, as the dense transformer does; coarse levels do not.
  - With `prior_mode none` the coarse levels reach a cell only through a readout: the validator asks for one.
- **Invariants:**
  - The hierarchy depends on positions alone (no hidden value decides a cluster); hidden cells weigh 0 in every pooled
    token; a cluster without a visible cell is a learned token.
  - Every parameter the hierarchy adds is zero or comes from a private generator: the dense weights are the dense model's at
    the same seed, and exploration draws nothing from the global generator.
  - Refused with the hierarchy: other attention backends and head widths, RoPE, spectral / expression biases, the long-range
    mask, the context query, `visible_keys_only`, `remask`, pooling, losses reading attention maps.
  - Readouts that index attention cell by cell get the dense mode's cell-level maps; routed models give none, so read them by
    `hierarchy_readouts`, integrated gradients and head scores.

### Reading the first results (written 2026-10-26, before any run was read)

Results: `MA/26_10_results/hier_synth_spot_A/` (also `_screen`, `hier_cosmx_fast`, `hier_cosmx_slow`); `results.md` per sweep, rebuilt from the
finished runs with `INTERSCALE_MODE=report HIER_PART=<A|screen|fast|slow> python scripts/<synth_spot_hier|cosmx_hier>.py`; fields (npz and quiver
plots of two test slides) in `fields/<run>/`. Run index in an array = arms x seeds in the yaml's order (gcn 0-2, dual_add_ctx_dist 3-5, full 6-8,
hier_none 9-11, hier_soft 12-14, hier_fac 15-17, hier_fpc 18-20, hier_soft_dir 21-23); a failed run is rerun with `sbatch --array=<k>`.

Read, in this order (synth_spot Stage A, 3 seeds, paired by seed):
1. Programme mean and wins / ties / losses against `ref_own+context`, per mode, against `full` (0.952 whole cell at 300 epochs in the M11 sweep, one seed, programme mean
   0.800), `gcn` and the dual model. The oracle bar is 0.983. Judge by centred cosine, not R^2.
2. Per range class (`short`, `mid`, `sparse`, `dense`, `directional`): `hier_planted` -- `sender_lift` (1 is chance), `pair_auprc` against `pair_auprc_chance`, `arrow_angle_deg`
   (90 is random) -- hierarchical modes against `full`. The twins (`full_untrained`, `hier_soft_untrained`) are the floor.
3. `fields/*.png`: do receivers point at their sources; coarse part (which cluster) against refinement (where inside it).
4. `hier_health`: `coarse_entropy`, `never_selected`, the learned lambda and mu per head and level; `hier_windows`: where the distance windows ended; `hier_stats`: cluster sizes and build time.
5. Then the screen (one seed, so only differences far above the seed spread of Stage A count): `scr_random` is the floor for the hierarchy; routed top-k against dense by
   `routing_recall_l1/_l2` and `sender_in_keys`.
CosMx: `heads_by_depth` / `heads_cells` against the k-NN smoothers (8, 32, 128) and the section mean, time and peak memory in the `training` table (efficiency/*).

Known caveats to remember when reading: the 8 h / 24 h limits are guesses; CosMx time per epoch on a whole section was never measured (the feasibility script `cosmx_feasibility.py`
stayed local, unused); `visible_keys_only`-style sinks are not addressed; routed attention gives no cell-level maps, so rollout / relevance readouts do not apply; the CosMx masking
rates (20% random cells, 60 um discs over 10%) are my choice of the comparison plan's proposal.

**Field plots for the next sweeps (decided 2026-10-26; built 2026-10-27, see the end of this paragraph).** Read the fields with
`notebooks_and_scripts/slurm/hierarchical/scripts/fields_by_class.py` (`python fields_by_class.py <results>/fields/<run> <out>
[--data <synth_spot.h5ad>] [--twin <results>/fields/<run>_untrained]`; since 2026-10-27 the sweep runs it after every export, into
`RESULTS_DIR/fields_by_class/<run>/`):
per range class, receivers coloured by programme with their own senders, one panel per head (narrow heads scaled up), a top-5
panel with precision@5 against chance, and with `--twin` the trained-minus-untrained row. Head-mean arrows alone mostly show slide
geometry (wide heads point to the slide centre); read the twin row and precision@5 instead. First read (full_s42): mid (M11)
~40x chance at precision@5 and a learned direction in heads 0-1; dense at chance; sparse mixed. For the next sweeps:
- give every arm whose fields are read an untrained twin (`UNTRAINED_ARMS`; the twin must be the same architecture and seed);
- extend the export (`hierarchy_readouts.fields`) with each head's cell mass and top-k targets per head, and the programme of each
  planted pair (so no local h5ad is needed), then subtract a baseline from each head's own trained window instead of the twin's;
- export the integrated-gradients contributions per receiver (`receptive_field._program_attribution`, today used for ranks only)
  for all receivers of the plotted slides, with more steps where the completeness gap exceeds 0.1, as the dependence panel next to
  the attention panels.

Built 2026-10-27, all three: twins for every arm of Stage B and of the re-read; the export per head (cell mass, top-5,
`window_vec_heads`) and the programmes; the dependence (`dep_*` in the npz) on the first exported slide, for at most 50
receivers per programme (`DEPENDENCE` in synth_spot_hier.py: a backward pass per receiver and point, ~0.05 s on the GPU; all
~3,000 receivers of a slide at 5-20 points would take ~40 min per run, the cap ~10 min; the sparse programmes stay whole), from
the graph's mean token, 5 points, 20 where the gap exceeds 0.1, not for twins. On stirred toy models over half the queries still
missed 0.1 after 20 points: read the figure's count and raise `retry_steps` if a trained run shows the same. A
relative gap stays large where the response barely changes from the baseline (a toy query at change -0.013 needed ~128 points):
read the gap next to `dep_change`. The figure's rows: where each head reads; minus its own window (last panel: the dependence
arrows, senders' share of |contribution| over chance); top-5 per head (last panel: top-5 by |contribution|); `--twin` adds a row.

Not built (the next steps if the results warrant): the scenario generator (S0, S5-S7 as separate slide families, a parametric S2, SCALE), `metis` / `quadtree` / `obs_label` clusters,
`multi_token` pooling, the local path and merges (T5), FlexAttention, `gumbel`, the regularisers (T7), LR plausibility and replicate consistency on CosMx, the exact dense CosMx arms (slow part) and
folds 1-2. Append a dated, titled entry to notes.md after reading (it is append-only), update this file's status table, and the roadmap's T6b row.

---

> **For Claude Code:** read the repo before writing any code. Find:
> - the global Transformer module (it runs on GNN node embeddings plus a CLS token),
> - the existing spatial/edge encodings,
> - the loss,
> - the data loaders for the synthetic spot data and CosMx,
> - the sweep/logging tooling.
>
> Add everything below as **new options** alongside the existing attention mechanisms, and don't change the defaults of existing configs. Names, paths and config keys here are suggestions; adapt them to the repo's conventions. Where this plan conflicts with the codebase, ask before restructuring.

---

## 1. Context

- **Model:** a GNN encodes local neighbourhoods. A Transformer then runs over the GNN node embeddings plus a CLS token and outputs global embeddings per node (one node = one cell or spot). The project builds on Interscale.
- **Spatial encoding:** real coordinates with a Euclidean distance bias (Graphormer-style), not shortest-path distance.
- **Goal of the global module:** capture long-range interactions (paracrine, endocrine, signalling cascades) without cell-type annotation.
- **Data scale:**
  - CosMx: slides of about 0.5–1M cells, spanning several conditions, donors and replicates.
  - Synthetic spot data: small enough for dense attention, which serves as an oracle baseline.
- **Current loss:** reconstruct a cell's expression from its neighbours. Keep the loss fixed during the attention sweeps (see the caveat in §5).
- **Hard requirement for CosMx:** process **one full slide per forward pass, with no sliding windows**.

---

## 2. Core idea and notation

- Levels are `ℓ = 0 … L`. `ℓ = 0` is the coarsest level (regions) and `ℓ = L` is individual cells/spots.
- `p(i)` is the parent of node `i` at level `ℓ−1`, and `n_c` is the number of children of cluster `c`.
- `A^ℓ` is the attention at level `ℓ`. Each row sums to 1 over the query's candidate set `C_i^ℓ`.
- `B^ℓ_ij` is the existing Euclidean distance bias. At coarse levels it is computed between cluster centroids.
- `x_i` is the coordinate of a node (the centroid, for clusters).

Each **hierarchical attention layer** makes one coarse-to-fine pass:

1. Pool tokens bottom-up: cells → level `L−1` → … → level 0 (the "vertical block").
2. At level 0, run dense attention among the region tokens. The CLS token attaches at this level.
3. At levels 1…L, restrict attention to the candidate set and bias or factorize it by the parent attention.
4. Readout: fuse each cell's output with the outputs of its ancestors.
5. Merge the result with the local path (residual or gated).

### Prior modes (`hier.prior_mode`)

| mode | definition |
|---|---|
| `none` | Levels are independent (an HSGT-like baseline). |
| `soft` | `logits_ij = q_i·k_j/√d + B_ij + λ_h · log(P_ij + ε)`, with `P_ij = A^{ℓ−1}_{p(i),p(j)} / n_{p(j)}`. The size correction `/n_{p(j)}` is a toggle. |
| `factorized` | `A^ℓ_ij = A^{ℓ−1}_{p(i),p(j)} · softmax_{j' ∈ children(p(j))}(q_i·k_j'/√d + B_ij')`. This is a hierarchical softmax: rows sum to 1 whenever the parent rows do. |
| `factorized_percell` | Each cell reweights the clusters itself: `w_i(c) = softmax_c(q_i·k̄_c/√d + B_ic + λ_h·log A^{ℓ−1}_{p(i),c})`, followed by a within-cluster softmax. This fixes the problem that "all cells in a cluster share one prior". `k̄_c` is the level-`ℓ−1` cluster token passed through the level-`ℓ` key projection. |

**Orthogonal flags:**

- `direction_prior`: adds `μ_h · conf_{p(i)} · cos∠(x_j − x_i, v_{p(i)})` to the logits, where
  - `v_c = Σ_{c'} A^{ℓ−1}_{c,c'} (x_{c'} − x_c)` is the parent's coarse arrow, and
  - `conf_c = 1 − H(A^{ℓ−1}_{c,:}) / log|C_c|` down-weights uncertain parents.
- `prior_channels`:
  - `row` encodes "where my cluster reads from".
  - `row+col` also encodes "who reads from my cluster". It adds a second bias from `Aᵀ` (row-normalized) with its own `λ_col`.
- `λ`, `μ`: learnable per head (default), learnable globally, or fixed. Head `h` at level `ℓ` uses the prior from head `h` at level `ℓ−1`.
- `prior_refresh`:
  - `per_layer` (default) recomputes the coarse attention in every layer.
  - `once` reuses the first layer's coarse attention in all later layers.

---

## 3. Implementation tasks

### T0: Input representation (global-only vs. staged combined model)

The hierarchical encoder must accept any of the following node inputs. This makes both a **global-only** model and a **staged GNN → global** model possible.

| `input.source` | description |
|---|---|
| `raw` | Normalized expression from `adata.X` or a layer (`input.layer`, e.g. log-normalized counts), optionally restricted to HVGs, then a linear projection to `d_model`. |
| `pca` | `adata.obsm[input.obsm_key]` (default `X_pca`). If missing, compute it in preprocessing with `input.n_pcs`, fitted on the training slides only, and cache it. |
| `gnn_frozen` | Embeddings from a pretrained GNN, loaded from a checkpoint or from `adata.obsm`. The GNN is not updated (staged training). |
| `gnn_joint` | The existing GNN runs in the same forward pass and is trained end-to-end with the global module (the current combined setup). |

**Notes:**
- `input.source` and `local.path` (T5) are independent switches. Avoid double-counting: with `input.source=gnn_*`, the default is `local.path=none` or `local_attn`. `gnn + gnn` is allowed but must be explicit.
- The reconstruction target always stays the expression, whatever the input. With `pca`/`gnn_*` input, make sure the masked cell's own information can't leak in through its input embedding (see the loss caveat in T9).
- **Acceptance:** all four sources run through the same encoder and config with no other code changes.

### T1: Hierarchy builder (preprocessing, cached per slide)

- **Methods:**
  - `kdtree`: recursive median splits on coordinates, balanced by cell count. This is the default; it avoids empty clusters at CosMx FOV gaps.
  - `quadtree`: area-based splits.
  - `metis`: partitioning of the spatial kNN/radius graph (the HSGT choice; it scales to millions of nodes in minutes).
  - `leiden_spatial_expr`: optional.
  - `random`: a negative control. HSGT saw a large performance drop with random partitions.
  - `obs_label`: the **coarsest level** is defined by a categorical column in `adata.obs` (`hier.top_level.obs_key`, e.g. a cell type or niche label). Finer levels are then built *within* each label with `kdtree`/`metis`.
  - `patches`: the coarsest level is a regular spatial grid of patches (`hier.top_level.patch_size_um`, or the number of patches per axis). Empty patches are dropped and tiny ones merged into a neighbour (`min_cells`). Finer levels are built within each patch as above.
- **The top level is configured separately from the finer levels:** `hier.top_level.mode` ∈ {`tree` (same builder as below), `obs_label`, `patches`}. This allows combinations such as niche labels → kdtree subclusters → cells.
- **Non-contiguous labels:** a cell-type label is usually scattered across the slide, so its centroid and arrows are not meaningful. Provide `hier.top_level.split_by_space`:
  - `none`: one cluster per label.
  - `components`: one cluster per spatially connected component of the label on the kNN graph, with tiny components merged.
  - `kdtree`: a kdtree split within each label down to a target size.
  - Whenever clusters are not spatially compact (log the ratio of spatial spread to the median spread), disable `direction_prior` and the coarse-level arrows for those clusters and log a warning.
- **Annotation caveat:** label-defined top levels use annotation, which conflicts with the annotation-free goal. Treat `obs_label` as an ablation or upper bound (e.g. "does a niche prior help?"), not as the default. For annotation-free niches, labels can come from an unsupervised clustering such as Leiden.
- **Config:** the number of levels plus the target number of clusters per level, either absolute or as coarsening ratios.
- **Outputs per level:**
  - parent index array and children (as CSR),
  - cluster sizes, centroids and spatial spread,
  - cluster-to-cluster spatial neighbour lists.
- **`shift` option:** builds a second hierarchy with offset split boundaries (shifted median thresholds, alternate axis order, or a half-cell quadtree offset). Layers alternate between the two hierarchies, Swin-style, so that partition boundaries are not always in the same place.
- **Acceptance:** deterministic for a given seed. Log the build time and the cluster-size histogram. A 1M-cell hierarchy should build in minutes, not hours.

### T2: Cluster token pooling (vertical block)

- **Pooling options:**
  - `mean`
  - `attn_pool`: a learned query attends over the cluster's members. HSGT found attention-based vertical aggregation beats mean pooling.
  - `multi_token`: `m` learned queries produce `m` tokens per cluster, with `m ∈ {2,4,8}`. This keeps rare senders from being averaged away.
- Pool level `ℓ−1` from level-`ℓ` tokens, not directly from cells, so the cost stays `O(N)`.
- Cluster tokens get positional features: the centroid (through the existing spatial encoding), log cluster size and spatial spread.
- With `multi_token`, a cluster's attention mass for prior projection is the sum over its tokens.

### T3: Hierarchical attention layer

- Level 0 uses dense attention over region tokens plus the CLS token, with the distance bias.
- Levels ≥ 1 use the prior modes from §2.
- **`share_level_params`:** shares attention weights across levels. HSGT found sharing barely changed accuracy while cutting parameters.
- **Must-pass regression test:**
  - **Setup:** `prior_mode=soft`, `λ=0`, dense candidates, and a single level in which every cluster is a singleton.
  - **Expected:** the module reproduces the existing global attention (given identical weights).

### T4: Candidate sets / sparse routing (required for CosMx)

**Candidate set:**

```
C_i^ℓ = children(top-k_ℓ targets of p(i) in A^{ℓ−1})
      ∪ children(e_ℓ exploration parents)              # uniform, or sampled ∝ A^{ℓ−1} outside top-k
      ∪ children(spatial neighbours of p(i))           # guaranteed local, always on
      [∪ children(parents with A^{ℓ−1} > τ)]           # optional threshold selection
```

**Kernels and memory:**

- The routed part of the candidate set is shared by all queries with the same parent. This makes the computation block-sparse:
  - **Query block:** the members of `p(i)`.
  - **Key block:** the union of the selected children.
- **Implementation options:**
  - PyTorch **FlexAttention** with a block mask (preferred),
  - padded batched gather,
  - or segment softmax (`torch_scatter`) as a fallback.
- **Never materialise an `N×N` matrix.**

**Local neighbours at the cell level:** use a per-cell kNN/radius. Two ways to combine them with the routed keys:

- join them in the same softmax by merging the partial attention results via log-sum-exp (deduplicate overlaps with the routed set), or
- route them as a separate path (see T5).

**Prior inside the candidate set:**

- `soft`: the prior is applied within the candidate set.
- `factorized`: the parent weights are renormalized over the selected parents.

**Gradients through selection:**

- **Default:** hard top-k. Gradients still reach the coarse level through the prior bias or the factorized product.
- **Optional:** `gumbel_topk`, a straight-through estimator.

**Candidate modes:**

- **`dense` (synthetic only):** all nodes are candidates. Use it to measure the effect of the prior without any pruning.
- **`routed`:** the candidate set above. Your notes recommend proving the soft prior first and adding pruning only afterwards.

**Per-query cap:** `cap_per_query` sets the maximum number of routed keys per query (e.g. 512). It bounds memory.

### T5: Local path and merge (also answers the "standalone without GNN" question)

- **`local_path`:**
  - `gnn` (existing),
  - `local_attn` (kNN attention with the distance bias),
  - `none`.
- **`merge`:**
  - `sum`: `H_hier + H_local`.
  - `gate`: `g = σ(W[h_hier; h_local])`, applied per node and per feature.
  - `joint_softmax`: local neighbours go inside the hierarchical softmax (the LSE merge from T4).

### T6: Readout block

- **`readout`:**
  - `none`
  - `concat` → linear
  - `transformer`: the cell token attends to `[h_i, h_{p(i)}, h_{p(p(i))}, …]`. HSGT's attention-based readout beat concatenation.

### T7: Regularisers and diagnostics (optional; low priority)

- **Regularisers:**
  - an entropy penalty `β · H(A^0)` on the coarse attention (optionally also at level 1),
  - `sparsemax`/`entmax` at level 0.
- **Log only (not used as losses):**
  - the fraction of regions that appear in no query's top-k (routing collapse),
  - the mean coarse entropy,
  - the learned `λ`/`μ` values per head and level.

### T8: Attention export and vector fields

Export for selected layers, heads and levels, streamed in chunks so it works at 1M cells.

**Fields per cell:**

| field | definition |
|---|---|
| `read_vec_i` | `Σ_j A_ij (x_j − x_i)`. Points toward **where i reads from**, i.e. the sources. The signal flows the opposite way, so name both directions explicitly in the plots. |
| `send_vec_j` | `Σ_i Â_ij (x_i − x_j)` with `Â` column-normalized ("where j's signal goes"). |
| top-k arrows | Target indices and weights (default k = 5). Needed because averaging cancels opposite directions. |
| entropy | Entropy of the attention row. |
| mean distance | Attention-weighted mean distance. |

**Factorized decomposition** (gives literal coarse-to-fine arrows):

`read_vec_i = Σ_c w_i(c)(x̄_c − x_i)` [coarse] `+ Σ_c w_i(c) Σ_{j∈c} a_{ij|c}(x_j − x̄_c)` [refinement]

Export both components for every level.

**Plotting script:**

- quiver plots per level (region → subcluster → cell), per head and averaged over heads,
- arrows coloured by entropy,
- on synthetic data, an overlay of the ground-truth sender locations.

Save outputs as npz or parquet, one file per slide.

### T9: Synthetic scenarios (extend the existing synthetic spot generator)

In each scenario, planted senders up-regulate a set of response genes in receivers. Store the ground truth:

- per receiver: the true senders and their weights,
- the true source direction: the weighted mean of `x_s − x_r`.

**Scenario parameters:** effect size, noise, number of response genes, and slide size.

| id | scenario | tests |
|---|---|---|
| S0 | null (no communication) | false-positive arrows; high entropy expected |
| S1 | juxtacrine (direct neighbours only) | local path; does hierarchy-only lose the local signal? |
| S2 | paracrine kernel `exp(−d/ℓ)`, with ℓ ∈ {3, 10, 30} spot spacings | the distance range the model can capture |
| S3 | long-range: a fixed sender region drives a receiver region ≥ ⅓ of the slide away, with no distance decay | coarse routing |
| S4 | rare senders: 1–2% of spots, scattered inside a region | mean-pool washout; `multi_token`, `attn_pool` |
| S5 | sender–receiver pairs straddling partition boundaries | `shift`; the guaranteed-neighbour candidates |
| S6 | cascade A → B → C | depth (number of layers), `prior_refresh` |
| S7 | receivers between two opposing sender regions | arrow cancellation; top-k arrows and entropy |
| MIX | all mechanisms in one slide | final comparison |
| SCALE | the generator tiled up to ~500k spots, with S3 + S4 planted | routing recall at CosMx scale (routed only; dense is infeasible here) |

**Loss caveat (check this first):** the planted signal must be recoverable **only** from the senders. Mask each receiver's own response genes in its input. If the current loss reconstructs a cell only from a local neighbourhood, S3/S6 are unsolvable by construction. In that case, add a masked-node reconstruction variant in which the global module sees the whole slide, and use it for the sweeps.

### T10: Metrics

**Synthetic data** (ground truth available):

- **Task performance:** reconstruction loss and R² on response genes, reported separately for receivers and non-receivers.
- **Sender attention lift:** the attention mass on true senders divided by the chance level `|senders ∩ C_i| / |C_i|`.
- **Pair-level AUPRC:** attention weight as the score for true sender–receiver pairs. Report it for the final layer, both averaged over heads and for the best head.
- **Routing recall@ℓ:** the fraction of receivers whose true sender clusters are inside their candidate set at level `ℓ`. This is the key diagnostic for hard routing.
- **Arrow angular error:** the angle between `read_vec` and the true source direction, over receivers only and weighted by magnitude. In S0, report the normalized `|read_vec|` and the entropy instead.
- **Distance profile:** the distribution of attention-weighted distances.
- **Compute:** peak GPU memory and seconds per step.

**CosMx** (no ground truth):

- **Held-out slide:** reconstruction loss on held-out slide(s), plus any downstream tasks already defined in the repo.
- **Ligand–receptor plausibility (a sanity check only):**
  - **Pairs tested:** top-attention pairs where `i` reads from `j`.
  - **Measure:** enrichment of ligand(`j`) × receptor(`i`) co-expression for LR pairs on the panel (OmniPath or CellChatDB).
  - **Baseline:** distance-matched random pairs.
  - **Reporting:** separately per distance bin, local vs. long-range.
- **Replicate consistency** (no image registration needed):
  - correlation of aggregated attention matrices across replicates and donors, at cell-type-pair level (or cluster-pair level if no annotation is available),
  - similarity of the attention-distance profiles.
- **Routing health:** coarse entropy, the fraction of regions never selected, and the learned `λ`/`μ`.
- **Compute:**
  - peak GPU memory, seconds per step, and full-slide inference time,
  - an explicit assertion that no windowing was used.

### T11: Sweep configs

Implement the sweeps in §6 and §7 with the repo's existing sweep tooling. If there is none, use Hydra multirun or W&B sweeps.

- Every run logs its full config, all T10 metrics, the learned `λ`/`μ` and the hierarchy statistics.

---

## 4. Unit tests

1. Hierarchy:
   - every node has exactly one parent;
   - children partition each parent;
   - `kdtree` cluster sizes are balanced within tolerance;
   - the build is deterministic for a given seed.
2. `soft`, dense candidates, `λ=0`, singleton hierarchy → output equals the existing global attention (fp32, atol 1e-5).
3. The prior `P` with size correction has rows summing to 1 (dense mode).
4. `factorized` rows sum to 1, including after renormalization over the selected parents in routed mode.
5. Block-sparse/FlexAttention output equals masked dense attention on a small graph.
6. The log-sum-exp merge of two partial attentions equals one joint softmax, with overlaps deduplicated.
7. Output is invariant to permuting the cell order.
8. **Arrow sign convention:** with a single planted sender, `read_vec` points toward the sender.
9. The factorized decomposition sums exactly to the full `read_vec`.
10. `obs_label` with `split_by_space=none` yields exactly one top-level cluster per label present. With `components`, every cluster is spatially connected on the kNN graph.
11. `patches`: every cell lands in exactly one patch, and no top-level cluster has fewer than `min_cells` cells.
12. Every `input.source` produces node inputs of shape `[N, d_model]`. `gnn_frozen` receives no gradients, while `gnn_joint` does.

---

## 5. Example config sketch

```yaml
model:
  global_encoder:
    type: hierarchical            # new option, next to the existing types
    input:
      source: gnn_joint           # raw | pca | gnn_frozen | gnn_joint
      layer: null                 # for raw: adata layer, null = adata.X
      hvg_only: true
      obsm_key: X_pca             # for pca (computed if missing)
      n_pcs: 50
      gnn_checkpoint: null        # for gnn_frozen
    hier:
      top_level:
        mode: tree                # tree | obs_label | patches
        obs_key: null             # e.g. niche or cell_type column in adata.obs
        split_by_space: components  # none | components | kdtree
        patch_size_um: null       # for patches
        min_cells: 50
      builder: kdtree             # finer levels: kdtree | quadtree | metis | leiden_spatial_expr | random
      levels: [1000, 16000]       # clusters per coarse level (or ratios)
      shift: false
      pooling: attn_pool          # mean | attn_pool | multi_token
      tokens_per_cluster: 1
      prior_mode: soft            # none | soft | factorized | factorized_percell
      direction_prior: false
      prior_channels: row         # row | row+col
      size_correction: true
      lambda: {init: 1.0, learnable: per_head}   # per_head | global | fixed
      mu:     {init: 0.5, learnable: per_head}
      prior_refresh: per_layer    # per_layer | once
      share_level_params: true
      candidates:
        mode: routed              # dense | routed
        topk: [8, 8]
        explore: {n: 1, strategy: prior_sampled}  # uniform | prior_sampled
        include_spatial_neighbors: true
        threshold: null
        cap_per_query: 512
        gumbel: false
      local:
        path: gnn                 # gnn | local_attn | none
        knn: 16
        merge: gate               # sum | gate | joint_softmax
      readout: transformer        # none | concat | transformer
      reg: {coarse_entropy: 0.0, sparsemax: false}
    export:
      layers: [-1]
      heads: all
      levels: all
      topk_arrows: 5
```

---

## 6. Sweeps: synthetic spot data

Run the stages in order and carry the best configuration of each stage forward. Use 3 seeds per stage and report mean ± sd.

**Stage A: mechanism** (dense candidates, all scenarios S0–S7)

- **attention:**
  - `full` (the existing dense global attention, used as the oracle),
  - hierarchical: `none`, `soft`, `factorized`, `factorized_percell`, `soft+direction`,
  - any other attention mechanisms already in the repo (e.g. linear/Performer) as baselines.
- Size: about 6–8 modes × 8 scenarios × 3 seeds.

**Stage B: prior details** (best two modes from A)

- `λ_init` ∈ {0.5, 1, 2}
- `λ` ∈ {per_head, global, fixed}
- `prior_channels` ∈ {row, row+col}
- `size_correction` ∈ {on, off}
- `prior_refresh` ∈ {per_layer, once}
- `μ_init` ∈ {0, 0.5, 1}

**Stage C: hierarchy and pooling** (focus on S4 and S5)

- `builder` ∈ {kdtree, quadtree, metis, random}
- `top_level.mode` ∈ {tree, patches, obs_label (planted region/niche label from the generator)} × `split_by_space` ∈ {none, components}
- number of levels ∈ {2, 3}
- cluster sizes: 2–3 settings
- `shift` ∈ {off, on}
- `pooling` ∈ {mean, attn_pool, multi_token with m=2, 4, 8}

**Stage D: routing** (compare against the dense run with the same config to isolate the cost of pruning)

- `topk` ∈ {2, 4, 8, 16}
- `explore.n` ∈ {0, 1, 2}
- `explore.strategy` ∈ {uniform, prior_sampled}
- `gumbel` ∈ {off, on}
- Main metric: routing recall@ℓ.

**Stage E: local path** (focus on S1 vs. S3; this answers whether the model works standalone)

- `input.source` ∈ {raw, pca, gnn_frozen, gnn_joint}
- `local.path` ∈ {gnn, local_attn, none} × `merge` ∈ {sum, gate, joint_softmax}
- Don't run the full factorial. The key cells are:
  - global-only: `raw`/`pca` + `local_attn`
  - global-only, no local path: `raw`/`pca` + `none`
  - staged: `gnn_frozen` + `none`
  - combined: `gnn_joint` + `none`
  - plus the best `merge` for each of these.

**Stage F: regularisers**

- `coarse_entropy` β ∈ {0, 1e-3, 1e-2}
- `sparsemax` ∈ {off, on}

**Stage G: final**

- the best 3 configurations on the MIX scenario, with 5 seeds,
- vector-field plots,
- the SCALE scenario with routed attention, as a dress rehearsal for CosMx.

---

## 7. Sweeps: CosMx (full slide, no sliding windows)

### Stage 0: feasibility gate (run before any training sweeps)

**Test:** a full-slide forward and backward pass on the largest slide.

**Log:** peak GPU memory and step time against these settings:

- number of levels and cluster sizes,
- `topk`, `cap_per_query` and local kNN,
- `d_model`, number of heads and number of layers.

**Starting grid** for N ≈ 1M cells:

- level-0 regions ∈ {1k, 4k}
- level-1 clusters ∈ {16k, 64k} (about 64 or 16 cells each)
- `cap_per_query` ∈ {256, 512, 1024}
- local kNN ∈ {16, 32}

**Calibration from the papers:**

- HSGT worked best with coarsening ratios around 0.005–0.01 on large graphs.
- ANS-GT found that adding more super-nodes helps only up to a point, after which they add noise. Don't make level 0 excessively fine.

**Cost sanity check:**

- 1M cells × ~550 keys ≈ 5.5e8 scores per head per layer.
- This is fine with block-sparse or FlexAttention kernels, provided it is never materialised.

**Fallbacks if memory is short** (in this order):

1. activation checkpointing per layer,
2. bf16,
3. smaller `topk` / `cap_per_query`,
4. `share_level_params`,
5. historical embeddings for coarse tokens (as in HSGT),
6. sliding windows, only as a last resort, and flagged explicitly.

**Practical points:**

- Use one slide per training step and shuffle the slide order.
- Express the distance bias and kNN radii in µm.
- Check FOV stitching and gaps. The cell-count-balanced `kdtree` should keep empty space from creating empty clusters.

### Stage 1: port the synthetic winners

- Take the top 2–3 configurations from Stage G.
- **Sweep:**
  - `topk` ∈ {4, 8}
  - `explore.n` ∈ {0, 2}
  - `λ_init` ∈ {1, 2}
  - `pooling` ∈ {attn_pool, multi_token m=4}
  - `shift` ∈ {off, on}
  - `local` ∈ {gnn+gate, local_attn+gate}
- **Setup:** 2 seeds; train on slides from several donors and conditions; validate on held-out slide(s).

### Stage 2: the standalone question and coarse-cluster definition

- **Inputs:** compare global-only (`pca` or `raw` + `local_attn`) with staged (`gnn_frozen`) and combined (`gnn_joint`). Give all of them the same compute budget.
- **Top level:** compare `tree`, `patches` (2 patch sizes) and `obs_label` (niche labels if available; cell type only as an ablation), each with `split_by_space=components`.
- **Coarse-cluster sizes:** for CosMx, `patches` and label-defined top levels must still yield a top level of roughly 1k–4k clusters. Adjust `patch_size_um`, `split_by_space` or `min_cells` accordingly.

### Stage 3: interpretation

- Export the per-level vector fields for all slides from the best model.
- Run the LR-plausibility and replicate-consistency analyses.
- Compare the fields across conditions.

---

## 8. Priorities

- **P0:**
  - T0 (all four input sources; `raw` and `pca` are the simplest to implement first),
  - T1 (`kdtree`, `random`, plus `top_level.mode=patches`),
  - T2 (`mean`, `attn_pool`),
  - T3 with `none` and `soft` in dense mode,
  - T8 `read_vec` export,
  - T9 scenarios S0–S4 and the loss caveat,
  - T10 synthetic metrics,
  - unit tests 1–3 and 7–8,
  - then Stage A.
- **P1:**
  - `top_level.mode=obs_label` with `split_by_space`,
  - `factorized`, `factorized_percell` and `direction_prior`,
  - T4 block-sparse routing,
  - T5 and T6,
  - scenarios S5–S7 and MIX,
  - Stages B–E.
- **P2:**
  - SCALE scenario,
  - CosMx Stage 0 feasibility,
  - then CosMx Stages 1–3.
- **P3 (optional):**
  - `metis`/`leiden`, `multi_token`, T7 regularisers, `gumbel`, `row+col`,
  - DiffPool-style learned soft assignment at the coarsest level, on synthetic data only:
    - **Regularisers:** add an entropy term plus a spatial-adjacency link-prediction term; DiffPool needed both for stability.
    - **Scale:** this is not feasible at CosMx scale.

---

## 9. References

- Zhu et al., *Hierarchical Transformer for Scalable Graph Learning* (HSGT), 2023. METIS hierarchy; horizontal, vertical and readout blocks; historical embeddings.
- Zhang et al., *Hierarchical Graph Transformer with Adaptive Node Sampling* (ANS-GT), NeurIPS 2022. Coarse super-node attention; coarsening-rate and super-node-count sensitivity.
- Ying et al., *Hierarchical Graph Representation Learning with Differentiable Pooling* (DiffPool), 2018. Learned soft assignments; entropy and link-prediction regularisers.
- Tang et al., *QuadTree Attention for Vision Transformers*, ICLR 2022. Coarse-to-fine top-k attention refinement.
- Liu et al., *Swin Transformer*, ICCV 2021. Shifted partitions.
- Rampášek et al., *GraphGPS*, 2022. Local message passing combined with global attention.
- Shirzad et al., *Exphormer*, 2023. Sparse attention baseline for graph transformers.
- Own design notes: `hierarchical_attention.txt`. Soft log-prior, residual and gated local path, sparse routing with an exploration budget; recommends adding the soft prior first and pruning afterwards.

---

## 10. More ideas from the literature (collected 2026-10-29)

Asked for on 2026-10-29 (point 5 of the focus list, roadmap.md). What Stages A and B already showed narrows which ideas are worth
building. On synth_spot the levels add nothing the cells' own windows lack: `hier_none` is within 0.01 of dense, and every prior
costs or ends switched off. On CosMx the model runs on whole sections in ~1 h, but undertrained. So the ideas are ranked by
three things a prior did not give: **(i)** cost and robustness at 10^5-10^6 cells, **(ii)** a readable split of each cell's output
into near and far parts (the decomposition question of decoder_plan.md), **(iii)** a reason for the coarse level to carry
information (content, not position). Papers marked *checked* were looked up on 2026-10-29; the others are from memory: read them
before building on them.

| # | source | the idea | what it would mean here | serves | priority |
|---|---|---|---|---|---|
| H1 | Yuan et al., *Native Sparse Attention* (NSA), arXiv 2502.11089, 2025 (*checked*) | three branches per query: attention to **compressed** block tokens, to the fine tokens of the **top-n blocks selected by the compressed scores**, and a **sliding window**; per query a gate `g ∈ [0,1]^3` (MLP + sigmoid) mixes them. The window has its own branch because local patterns learn faster and would otherwise shortcut the other two | our routed mode is compression + selection already. Add the window branch (the narrow windowed heads as their own path) and the per-cell gate. The gates are a per-cell near / selected-far / coarse share: a decomposable read-out for free, and the separate window branch is NSA's answer to "global copies local" | (ii), (i) | **high** |
| H2 | *Natively Trainable Sparse Attention for Hierarchical Point Cloud Datasets*, arXiv 2508.10758, 2025 (*checked*: Erwin + NSA) | NSA's branches on a ball-tree hierarchy of points in space, not on sequences | the closest prior work to whole sections (points in 2-D, 10^5+); read first, before H1 is built | (i), (ii) | **high (read)** |
| H3 | Zhdanov, Welling, van de Meent, *Erwin*, ICML 2025, arXiv 2502.17019 (*checked*) | ball tree; attention within fixed-size balls; the tree stored **contiguously** (pooling is a `view` + `mean`); a **rotated** tree in alternating layers for cross-ball mixing; U-Net coarsen / refine; positions relative to the ball centre + a distance bias with learned sigma | (a) order the cells by tree once per slide, so our padded gather becomes contiguous slices: speed on CosMx; (b) the rotation per layer in place of our one-off `shift` twin; (c) a U-Net variant: coarse layers carry the long range, fine layers refine, instead of every layer doing coarse-to-fine | (i) | medium |
| H4 | Lu et al., *MoBA: Mixture of Block Attention*, arXiv 2502.13189, NeurIPS 2025 (*checked*) | parameter-free top-k gating **per query**: the score of a block is `q · mean(k_block)`; the own block is always in; layers can switch between full and sparse, with hybrids (some layers full) | routing per cell instead of per parent (today a parent's top-k, maximised over heads, is shared by its children), at the cost of one `q · mean(k)` per cluster; and a hybrid model: the last layer fused-dense, earlier layers routed | (i) | medium |
| H5 | Yang et al., *Focal Transformer*, NeurIPS 2021; Wang et al., *Pyramid Vision Transformer*, ICCV 2021 (spatial-reduction attention) | each query sees fine tokens near it and **coarser pooled tokens the farther away**, by fixed rule; or keys / values pooled while queries stay fine | a routing-free candidate set: own fine window + every cluster token of the next level + every region token. Fixed cost, no top-k that can miss (routing recall), and it is exactly a near + far decomposition | (i), (ii) | medium (the baseline routed must beat) |
| H6 | Nguyen et al., *FMMformer*, NeurIPS 2021; Zhu and Soricut, *H-Transformer-1D*, ACL 2021 | the fast multipole view: near field exact (banded), far field low rank or coarsened, the far interaction of two distant clusters computed once and handed to all children | a cell's output as an explicit sum `near(i) + far(i)`, the far part from unrefined coarse levels only. With the additive read-out it is the local / global decomposition by construction, inside one attention model (decision 13) | (ii) | medium |
| H7 | Wu et al., *Point Transformer V3*, CVPR 2024 | serialise points by space-filling curves (Z-order, Hilbert and their transposes), attend within patches of the serial order, shuffle the curve between layers | a third hierarchy builder (`serialize`), nearly free to build at 10^6 cells, with nested patches as levels | (i) | low |
| H8 | Roy et al., *Routing Transformer*, TACL 2021; Ren et al., *Combiner*, NeurIPS 2021 | clusters by **content** (online k-means on queries and keys) instead of position; Combiner: every key reachable either directly or through a structured summary of its group | a coarse level of expression clusters, so a receiver can read every sender-like cell wherever it sits. It is the one idea that gives the coarse level information the window lacks (iii). Risk: it is same-type attention (T5), and the clusters must come from visible values only | (iii) | low now; with T5 |
| H9 | Spatia (arXiv 2507.04704), CellTransformer (ICML 2025 workshop; OpenReview h2bMZzUsUr), TERRA (preprint) (*checked*: abstracts only) | spatial-omics models with explicit levels: cell, niche (patches of cells), tissue; separate embeddings per level, fused at the end (TERRA: cell-intrinsic, spatially contextualised, neighbourhood) | the domain's own precedent for the lung niche roles (lung_plan.md L2, L3), and the papers to cite for them | write-up | cite |

**Proposed stages, after the CosMx `long` read** (each one arm against `hier_none` and dense, synth_spot 3 seeds at 300 epochs, then CosMx fold 0):
- **H-α (H1 + H5):** a window branch and a per-cell gate on top of `hier_none`, with the focal candidate set as the routing-free
  variant. Read: the score, the gate shares per programme class (short-range receivers should take the window branch, rings the
  coarse one), routing recall against focal.
- **H-β (H3 a, b):** contiguous tree order and a per-layer rotation. Read: seconds per epoch and peak memory on CosMx at equal
  score.
- **H-γ (H6):** the additive near / far read-out, its parts into `StepOutput.parts`, so `attribution_shares` and the gene-scale
  plots read it like the additive decoder.
- Later: H4 (per-cell routing) if routing recall is the limit on CosMx; H8 with T5.

**Niche labels as the top level (lung data, 2026-10-29).** `top_level.mode: obs_label` with `split_by_space: components` (T1 above)
becomes P0 for lung_plan.md L2. Its annotation caveat (T1) decides how it is read: an upper-bound arm, not the default. Built 2026-10-30
(branch `niches`); for routed models use `split_by_space: kdtree` (lung_plan.md L2: one wide parent starves every query's candidates).

**Sources of section 10** (looked up 2026-10-29; the rest of the table is from memory, to check before building):
- H1 NSA: https://arxiv.org/abs/2502.11089
- H2 NSA on hierarchical point clouds: https://arxiv.org/abs/2508.10758
- H3 Erwin: https://arxiv.org/abs/2502.17019 (code: https://github.com/maxxxzdn/erwin)
- H4 MoBA: https://arxiv.org/abs/2502.13189 (code: https://github.com/MoonshotAI/MoBA)
- H9 Spatia: https://arxiv.org/abs/2507.04704; CellTransformer: https://openreview.net/forum?id=h2bMZzUsUr; TERRA (preprint):
  https://www.researchgate.net/publication/411408062_Multi-scale_modeling_of_human_tissues_from_spatial_transcriptomics_with_TERRA
- From memory, not looked up: Focal Transformer (Yang et al., NeurIPS 2021), PVT (Wang et al., ICCV 2021), FMMformer (Nguyen et al.,
  NeurIPS 2021), H-Transformer-1D (Zhu and Soricut, ACL 2021), Point Transformer V3 (Wu et al., CVPR 2024), Routing Transformer
  (Roy et al., TACL 2021), Combiner (Ren et al., NeurIPS 2021).
