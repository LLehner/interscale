# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Sender attribution: `receptive_field.response_influence(measure="attribution")` ranks a planted sender
  by its *contribution* to its receiver's predicted response -- integrated gradients (Gauss-Legendre,
  `steps` points) from a baseline cell (`baseline`: a `[G]` expression, e.g. the training mean; default the
  graph's mean; a model without a GCN moves from its graph's mean token), the net contribution of each
  cell summed over its visible entries, on the summed response genes of the programme. A hidden entry does
  not move, and the passes along the path replay one random state (`pad_batch` subsamples long graphs with
  Python's `random`). The new `check` table gives `completeness_gap`: the share of the response's change from
  the baseline that the contributions fail to add up to, per programme and over all. The squared gradient
  stays as `measure="gradient"` (reach, not use).
- Scoring within the cells that respond: `HeadPredictions.cells` (the graphs' `obs_names` of each row),
  `masked_probing.group_receivers` (the rows that are receivers of the programme a gene group holds, from
  the planted edges) and `score_heads(group_cells=...)`, whose `by_group_cells` table is the pooled centred
  cosine over a group's hidden entries in those cells alone (the response where there is one to predict,
  apart from telling who responds).
- The summary counts, per programme and with every programme weighing the same, the wins, ties and losses
  of each head against a bar (`write_report(bar="ref_own+context", programmes="interaction=")`, or
  `bar="best"` for the highest reference on each programme; ties within 0.05) for all cells and within
  receivers, names the best and worst programme, and shows the attribution's completeness gap; the pooled
  score is carried by the programmes with the most hidden entries.
- Pooled tokens (`model.global_component.parameters.pooling`, default `kind: none`): the transformer runs
  on patches of cells instead of single cells. `kind: grid` groups cells into squares by position alone
  (`size_um`, or `n_patches` for about that many per slide; `offset` shifts the grid); a patch token is the
  mean of its *visible* cells' tokens at the patch's centroid (a learned token where none is visible; the
  optional `features` `count` and `std` start at zero); every cell's global embedding is its patch's, laid
  out as the per-cell output is, so decoders, losses and probes read one row per cell. S drops 10-1000x,
  so whole sections fit dense attention. New `GlobalModule.step` (input step + transformer, pooling
  included) replaces the two calls the three modules made; module `global_modules/pooling.py`. Checkpoint
  tag `pool-grid<side>um` / `pool-grid-n<patches>`. The config refuses what a patch cannot carry (the
  long-range mask, Laplacian/random-walk encodings, spectral/expression biases, Exphormer, an attention
  loss); `evaluate()`, `attention_by_query`, `attention_diagnostics` and the rollout of `patch_flow` refuse
  a pooled model (they read attention per cell).
- `masked_probing.FittedReferences.state()` / `from_state(state, position)`: the ridge references of the
  programme evaluation can be saved and loaded (the `position` function is handed back in), so jobs that
  evaluate different models on the same training slides fit them once.
- The results file reads seeds: runs named `<arm>_s<seed>` are shown once in the summary as the mean over
  their seeds (`<arm> [n seeds]`), with the standard deviation over seeds and every run's own value in
  tables beside it (`evaluation.report.average_over_seeds`); the sections keep every run. Jobs that run
  at the same time can each write their own file of sections (`<folder>/<arm>.md`) and
  `write_report(..., parts=folder)` merges them, the newest winning; a `timing` analysis (seconds per
  evaluation step) gets a summary table.
- `type_gex_embedding: Precomputed` under programme masking: allowed with `mask_strategy: gene` when
  `dataset.mask_genes` hides a gene set, because an embedding computed from the other genes holds none
  of the hidden values. No cell is replaced by the mask fill then (every cell keeps its visible genes).
  `interscale.tl.record_embedding_genes(adata, key, genes)` stores which genes an embedding was made
  from (`adata.uns["embedding_genes"]`); `prepare_geome_dataset` refuses an embedding that saw a hidden
  gene or whose genes are unrecorded (`tl.check_embedding_genes`). Single hidden entries stay refused.
- `model.global_component.parameters.pca_fit: train`: a GlobalModel's PCA tokens are fitted once, before
  training, on the masked inputs of every training slide (deterministic solver), instead of on the first
  batch the module sees (the first, masked validation slide). Default `first_batch` is unchanged;
  checkpoint tag `emb-pca-train`.
- Programme masking: `dataset.mask_genes` (gene names or `var` selectors such as `"role=response"`)
  hides exactly that gene set in the selected cells under `mask_strategy: gene`
  (`mask_percentage` is then the share of cells; discs select cells for it); resolved by
  `geome_dataloader.gene_set_mask_kwargs`, checkpoint tag `gs<rate>-<selectors>`. For its evaluation:
  `masked_probing.FittedReferences` (ridge references fitted on training graphs -- the cell's own
  visible genes, plus kernel means of the visible genes around it, plus position -- penalty chosen on
  held-out graphs), `draw_masks(kind="gene_set")`, `collect_head_predictions(fitted_references=...)`
  and `receptive_field.response_influence` (a planted sender's rank in its receiver's predicted
  response genes among cells as far away).
- `evaluation.attention_flow`: where the cells of a patch draw their predictions from -- gradient
  influence (own input left out) and attention rollout with the residual path -- against the same
  model on the slide with the visible cells' expression permuted; summary metrics, flow fields,
  source maps and label heatmaps (`plot_flow_field`, `plot_source_maps`, `plot_label_flow`).
- Attention backends `entmax` (alpha-entmax, alpha learned per head; needs the `entmax` package) and
  `topk` (softmax over each query's k best keys), and the attention bias `pe.bias: [expression]`
  (a learned profile over each key's rank by expression distance, read from the masked input).
- Evaluation additions: a position feature set for the probes (`masked_probing.position_features`,
  `combined_features`; the bar becomes `smoother+position`), head scores per gene group and depth
  (`score_heads` → `by_group_depth`), and blends of each head with a reference
  (`masked_probing.blend_heads`).
- `main_sweep(..., prepare_adata=...)`: a function called on the AnnData once a sweep trial has read
  it, before the model is built -- e.g. to compute an embedding into `obsm` that the model reads through
  `type_gex_embedding: Precomputed`.
- Staged runs record each phase's end (`model.phase_summaries_`, and in the W&B summary as
  `schedule/phase<k>_<metric>`): how far each component's weights moved (`drift_<component>`, 0 when
  frozen) and the phase's last validation losses. The results file shows them as a `phases` section.
- One results file per evaluation run (`interscale.evaluation.report`). `append_sections` appends an
  arm's analyses to a Markdown file as `## <arm> | <analysis>` sections, each a compact CSV table;
  `read_sections` reads them back, a later section replacing an earlier one; `write_report`
  rewrites the file with a summary across arms on top. Missing analyses are simply absent.
- Masked-pass evaluation, every analysis read with the input corrupted as in training (overview:
  `.claude/linear_probing_docs.md`). Label and covariate probes are scored by depth into the hidden
  area (`depth_bins_um`) beside a model-free `smoother` feature set (`masked_probing.smoother_features`).
  `attention_diagnostics.attention_by_query` gives the attention of hidden queries against uniform
  attention, distance profiles and planted pairs against distance-matched cells.
  `evaluation.receptive_field` measures how much each visible cell influences a hidden cell's
  prediction (Hutchinson estimates of Jacobian norms). `evaluation.representations` gives linear
  CKA and participation ratios. `score_heads(gene_metrics=...)` can score genes by centred cosine.
- Staged training of the local and global components (`train/_schedule.py`): per-head loss weights
  (`optim.loss_weights`, default 0.5/0.5, the previous stacked loss), frozen components
  (`optim.frozen`), and schedules of phases (`optim.schedule`: `joint`, `local_then_global`,
  `local_then_global_ft`, `alternating`, `alternating_coupled`, or `custom` with
  `optim.schedule_phases`) in one W&B run. `CombinedModel.copy_local_from` starts from a trained
  `LocalModel`.
- `precomputed_mask_fill` (`learned` | `minus_one`): a hidden cell's precomputed embedding
  (`type_gex_embedding: Precomputed`) is replaced by a learned vector or -1, so it cannot hand the
  model the values it is scored on.
- `type_gex_embedding: Linear`, a learned linear input for the global-only `GlobalModel`, trained
  with the transformer.
- Head scores on hidden cells for every model type (`masked_probing.collect_head_predictions`,
  `score_heads`): per cell, gene, gene group and depth, beside model-free references
  (`reference_predictions`: slide mean, label mean, Gaussian kernels, nearest visible cells).
- Spatial block masking: `dataset.mask_block_radius_um` and `dataset.mask_block_fraction` hide whole
  cells in discs, alone or beside node or gene masking.
- `attention.center_qk` (queries and keys from per-slide centred tokens), `optim.center_targets` (the
  loss on gene-centred targets) and `model.decoder.global_residual` (the global head predicts the
  local head's error).
- Attention backends behind a registry (`attention.kind`, free head width): dense softmax, Performer
  (FAVOR+), elu+1 linear attention and Exphormer-style sparse attention. Also attention diagnostics
  (`probe.attention_diagnostics`: span, CLS mass, effective keys, TV distance to softmax), an
  efficiency logging callback (`optim.log_efficiency`) and a generator for sweep arms.
- Positional encodings (`pe.node`, `pe.bias`, `pe.rotary`): naive, sinusoidal, Laplacian
  eigenvectors, random walk, relative distance, spectral and Fourier distance biases, and RoPE; a
  `pe` control column in the probes.
- `synth_spot`, a spot-lattice benchmark with planted short- and long-range programmes
  (`evaluation/synth_spot.py`, `signal_scale`).
- Masked post-hoc probing (`masked_probing`): gene-vector, label and covariate probes from the
  embeddings of hidden cells.
- Auxiliary losses (`optim.aux_loss_weights`): VICReg (variance, invariance and covariance terms),
  context NCE with composition-matched negatives, and ring NCE (the anchor's long-range ring against
  the GNN's neighbourhood); multi-view forward passes.
- Online probes during training (`probe.use`): label, gene and gene-vector probes per epoch on
  hidden cells, logged as `probe/*` with W&B charts.
- Long-range attention masking (`long_range_attention`): the transformer cannot attend inside the
  local component's receptive field.
- `synth_data_0`, a synthetic dataset with known niches and a ligand-receptor interaction
  (`evaluation/synthetic_data.py`), with post-hoc probes (`linear_probing`, `downstream_regression`)
  and an attention-flow control (`flow_control`).
- The scaled cosine error of GraphMAE (Hou et al., KDD 2022, eq. 2), `(1 - cos(pred, true)) ** gamma`
  per cell, is logged for every regression run whatever its loss, as `{train,val,test}_scaled_cosine_error`
  over all cells and `..._masked_scaled_cosine_error` over the masked ones. With `SCELoss` as the
  criterion the masked one is the training loss itself. The exponent is the new
  `optim.sce_gamma` (default 3.0, the value `SCELoss` always used; `>= 1`, checked at config
  load), which `SCELoss` and `SCE_EntropyATT_Loss` now read too. `gamma = 1` is the plain cosine
  error, `1 - cosine_similarity`. The per-cell function is `train.losses.scaled_cosine_error`.

- Anchor-point / distance-zone analysis of a latent dimension (`interscale.tl.anchors`,
  plotted by `interscale.pl.anchor_plots`): `find_anchor_cells` localises the foci a dimension
  is reading (per-sample tail quantile, then a spatial-coherence filter),
  `anchor_signed_distance` gives every cell a signed distance to the anchor border,
  `anchor_zones` cuts that into `anchor core` / `near` / `far` at the local component's reach
  (`local_reach_um` = `num_layers` x `radius`), and `profile_by_distance` /
  `anchor_enrichment` profile expression and composition against it. This is the evidence a
  *global* dimension carries a long-range interaction rather than a neighbourhood effect:
  every distance plot marks the local reach, past which no cell's neighbourhood extends.

- Gene-wise (per-entry) masking for the node-level reconstruction task, alongside the existing
  whole-cell masking, selected by `dataset.mask_strategy` (`"node"` | `"gene"`). Under `"gene"`
  a Bernoulli subset of `(cell, gene)` entries is blanked in every cell and the loss and every
  regression metric are restricted to those entries. Registered as the `node_reg_genemask` task
  for `legnini23` and `melton25`, with `config_files/sweeps/mask_granularity_legnini.yaml`
  running the granularity/rate ablation.

### Changed

- `receptive_field.response_influence` defaults to `measure="attribution"` (it was the squared gradient).
  Ranks from the squared gradient were not a measure of use -- a mean-aggregating GCN weighs its neighbours
  alike, and ranked its own planted senders at 0.41 while predicting their responses at 0.59 -- and are not
  comparable with the attribution's. `measure="gradient"` gives the old numbers.

- Attention records carry the mask's open keys (`open`), and `attention_by_query` and the online
  attention diagnostics read open keys from it rather than from weights above 0, which a sparse
  normaliser breaks. `_spatial_weights.graph_edges` carries every further column of the edges.
- A precomputed embedding's checkpoint tag carries its `obsm` key (`emb-pre-<key>-<fill>_`, letters and
  digits only), so two embeddings never share a checkpoint.
- The in-training attention diagnostics read the val graphs under their own masks
  (`probe.attention_masked`, default `true`); `false` restores the clean pass.
- `_common_step` returns a `StepOutput` (named fields, with `ViewOutput`, `gather_tokens` and
  `gather_cls`) instead of a tuple, and the train, val and test steps share one body.
- The transformer keeps attention maps only when something reads them.
- `SCELoss` and `SCE_EntropyATT_Loss` take `gamma` (the paper's name) instead of `alpha`, and
  validate it is `>= 1`. `1 - cos` is clamped at 0 before the power, so a non-integer `gamma` no
  longer turns two identical rows into NaN.
- `tl.masking.MASK_VALUE` is now **-1**, was 0. Masked positions must be distinguishable from
  real measurements: ~62% of legnini23's `log1p_norm` entries are already exactly 0 and 648 of
  its cells are all-zero, so a zero fill made the corruption invisible under gene masking.
  Measured at gene rate 0.25 over 3 seeds, fill 0 scored `val_concordance_corr` 0.0696 ±0.0113
  against -1's 0.0799 ±0.0046 — better on every seed, with 2.5x less run-to-run spread. A
  configurable fill (`zero`/`sentinel`/`learned`, including a trainable per-gene `[MASK]` token)
  was trialled and removed: the learnable token tied with the fixed -1 to three decimals, so the
  option carried no information. **This also changes the cell-masking path**, whose published
  runs used 0.
- `dataset.pct_mask_nodes` and `dataset.pct_mask_genes` are replaced by a single
  `dataset.mask_percentage` (and `GraphAnnDataModule`/`BaseModule` take one `mask_percentage`
  argument). `mask_strategy` already says whether a masked unit is a cell or an entry, so a
  second rate was always the inert half of a pair — and setting the wrong one silently produced
  a run with no masking. `GraphAnnDataModule.mask_rate` and `BaseModule.mask_rate` are gone with
  it. **Every config using `pct_mask_nodes` must be renamed**, including saved wandb sweeps.
- `tl.masking.apply_mask` takes `mask_strategy` explicitly instead of inferring it from whether
  the batch carries a `gene_mask`, and `GlobalModule._process_batch_for_metrics` gates on the
  strategy the same way. A stale `gene_mask` can no longer hijack a cell-masking run, so
  `GraphAnnDataModule._clear_gene_mask` is removed.
- `tl.masking.apply_mask` and `BaseModule._common_step_masking` now return a third value, the
  entry mask; every `_common_step` returns a sixth value carrying it through to the training plan.
  `GlobalModule._process_batch_for_metrics` returns it as a third value.
  `DualDecoderCombinedModule.compute_separate_losses` takes it as an optional fifth argument.
- `LocalModule._common_step` returned a 4-tuple where the training plan unpacked 5; it now
  matches the 6-tuple contract of the other modules.

### Deprecated

- Relevance-based readouts. `tl/self_attn_relevance.py` backpropagates the summed transformer output
  after its affine LayerNorm: the gradient is zero while the norm's scale is uniform and points in an
  arbitrary direction after. `_attn_matrix` and `cls_horizontal`/`cls_vertical` from
  `get_model_output`, and everything built on them (`downstream_regression`, `net_streams`,
  `flow_control`, `graph_classification`), should not be interpreted. `evaluation.receptive_field`
  measures what a prediction depends on.

### Fixed

- A dual decoder now logs each head's masked scaled cosine error (`<mode>_masked_scaled_cosine_error_local` /
  `_global`). The combined metric is the mean of both halves, so with one head frozen it said nothing about
  the head that trains.
- The planted-pair matched rank compares senders with cells within 10 µm of their distance to the
  receiver (was 50 µm), so lattice neighbours no longer rank against the diagonals.
- Spatial blocks alone (`mask_percentage` 0) left the model input unmasked.
- Sweep trials seed before building the model; the efficiency monitor measures training memory only;
  the directional Fourier bias has its own checkpoint tag.
- The long-range mask's reach follows the local module's hops (`resolve_local_mask_hops`).
- `load_config` skipped the objective check for config files, and a `GlobalModel` was built from the
  underived config.
- Auxiliary loss heads were lost on save and reload.
- Online probes masked their label targets, and the probe pass moved the global random generators;
  it now restores every one.
- The masked per-cell cosine (`..._masked_cosine_similarity`) averaged over every cell, and a cell
  without a masked entry entered as a cosine of 0. Under cell masking that is every unmasked cell,
  so the logged value was scaled by the fraction of cells masked: a near-perfect reconstruction
  read 0.35 at a 30% masking rate. It now averages over the cells with a masked entry. The other
  masked metrics were not affected.
- `SCE_EntropyATT_Loss` under cell masking averaged its cosine term over every cell, with the
  unmasked ones zeroed into a constant error of 1: the loss was `f * SCE(masked) + (1 - f)` for a
  masked fraction `f`, and its gradient `f` times too small. It now averages over the masked cells,
  as `SCELoss` already did. **This changes training under that loss.**
- Under gene masking, the row-normalising losses (`SCELoss`, `SCE_EntropyATT_Loss`,
  `BalancedPearsonCorrelationLoss`) counted a cell with no masked gene as an all-zero row. Such
  cells are now left out; at typical rates and panel sizes they almost never occur.
- `optim.monitor` minimised only metrics ending in `loss`, so monitoring an error -- `val_mse`, or
  the new scaled cosine error -- made early stopping and checkpointing keep the *worst* epoch. Names
  ending in `error` or `mse` are now minimised too.

## [0.0.1]

initial release
