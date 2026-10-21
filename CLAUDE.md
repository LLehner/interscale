# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

InterScale is a Python package for multi-scale cell interaction analysis in spatial transcriptomics data. It combines a local, graph-based component (cell-neighborhood scale) with a global, transformer-based component (whole-sample scale) into a single trainable model. It is built on top of `geome`/PyTorch Geometric, AnnData, scanpy/squidpy, `scvi-tools` (for AnnData registration/data management), and `lightning`/`pytorch_lightning` for training.

For the scientific motivation, model design rationale, and terminology behind this architecture (from the InterScale preprint — Drummer, Jiménez et al., bioRxiv 2026), see [`.claude/background.md`](.claude/background.md). Read it before working on model/module architecture, loss functions, or the interpretability/evaluation pipeline — it explains *why* the local/global split and dual-decoder design exist, not just what the code does.

The three-week roadmap (2026-10-19 to 2026-11-08) is in [`.claude/roadmap.md`](.claude/roadmap.md):
the project's goals (disease-specific interactions, how cell-type driven attention is, masking
strategies, complexity vs tissue coverage, disentangling local and global, NicheNet priors) and the
known problems of the global model, as tracks with a schedule and decision points. The designs stay
in the plan files below. Read it before starting work on any of its tracks, and update its status
table in the same commit as the work it describes.

What each test showed -- what worked and what does not -- is logged in [`.claude/notes.md`](.claude/notes.md).
It is append-only: after reading a new test, add a dated entry at the end with a short title saying what the
test was about, then its **What worked** and **What doesn't work** points; never overwrite or edit an earlier
entry (a correction is a new entry).

Work on contrastive / auxiliary losses for the global component is planned in
[`.claude/contrastive_plan.md`](.claude/contrastive_plan.md). It carries the staged design, a
status table, and an invariants checklist — read it before touching `train/losses.py`,
`_trainingplans.py`, or any `_common_step`, and update its status table in the same commit as
the work it describes.

Positional encodings for the global component are planned in
[`.claude/PE_plan.md`](.claude/PE_plan.md), with the same staged layout, status table and
invariants. Read it before touching the transformer's input path (`common_step_local_to_global`,
`pad_batch`, the attention mask), and update its status table in the same commit as the work it
describes.

Alternative attention mechanisms (Performer, Exphormer-style sparse attention, entmax and top-k
normalisers, an expression-similarity bias, global tokens) are planned in
[`.claude/attention_plan.md`](.claude/attention_plan.md), with the same layout. Read it
before touching the attention layers (`transformer_encoder_layer.py`, `transformer_utils.py`), the
per-batch attention inputs of `common_step_local_to_global`, or the bias forms, and update its
status table in the same commit as the work it describes.

Why the global component struggles to learn, and the experiments that separate the causes
(global model alone, other local embeddings, separate training of local and global), are planned in
[`.claude/global_plan.md`](.claude/global_plan.md), with the same layout. Read it before touching
`GlobalModel`'s input, the local modules' embeddings, the dual decoder's loss weighting or the
training loop, and update its status table in the same commit as the work it describes.

How to separate the local and the global component -- the decoder structure of the dual model
(a context-only global embedding, different targets per head, one prediction from two parts) and
losses on the embeddings -- is proposed in [`.claude/decoder_plan.md`](.claude/decoder_plan.md),
with the same layout. Read it before touching the decoders, `DualDecoderCombinedModule` or the
per-head losses, and update its status table in the same commit as the work it describes.

## Common commands

This project uses `hatch` as the primary project manager (also works with `uv` or `pip`; see `docs/installation.md`).

```bash
# Install dev dependencies (uv)
uv sync --all-extras

# Install dev dependencies (pip)
pip install -e ".[dev,test,doc]"

# Run the full test suite
hatch test            # or: uv run pytest
hatch test --all      # across the full Python version matrix

# Run a single test file / test
uv run pytest tests/test_package.py
uv run pytest tests/test_package.py::test_import

# Lint / format (ruff + biome are also run via pre-commit)
uv run ruff check .
uv run ruff format .
pre-commit run --all-files

# Build docs
hatch run docs:build
hatch run docs:open
```

Training entrypoints are run as scripts, not via a CLI command:

```bash
python src/interscale/main.py --cfg config_files/legnini_example.yaml --model_type CombinedModel
python src/interscale/main_sweep.py --cfg <cfg.yaml> --sweep_cfg <sweep.yaml> --model_type CombinedModel --sweep_goal hyperparmeter
```

`--model_type` is one of `LocalModel`, `GlobalModel`, `CombinedModel`. `main_sweep.py` additionally requires `wandb` and a sweep config with `sweep_goal` in `{robustness, segmentation, hyperparmeter, loss}`.

Ruff config lives in `pyproject.toml` (`[tool.ruff]`): line length 120, numpy docstring convention, many rules relaxed for legacy code (see `lint.ignore`). Tests are exempt from docstring rules (`D`).

## Architecture

### Config-driven design (yacs)

Everything is driven by a single `yacs` `CfgNode` (`cfg`), built via `interscale.config.load_config(cfg_path)`:
- `get_cfg_defaults()` assembles defaults from `config/{wandb,model,optim,dataset}_config.py`.
- `load_config` additionally loads component-specific defaults based on `cfg.model.local_component.name` / `cfg.model.global_component.name` (from `local_component_config.py` / `global_component_config.py`) *before* merging the user's YAML file, then freezes the config.
- User-facing YAML files (see `config_files/legnini_example.yaml`) only need to override values relevant to their run — everything else falls back to defaults.
- The `cfg` object is threaded through nearly every layer (model, module, dataloader, geome dataset prep) rather than passed as individual kwargs.

### Model hierarchy (`interscale.model`)

`BaseModel` (`model/base/_base_model.py`) mimics the `scvi-tools` model API pattern:
- Uses an `scvi.data.AnnDataManager` (registered via `_setup_anndata`/`register_fields`) to validate and access fields on the `AnnData` object (expression layer, sample/library keys, prediction obs, split key). This is why models require `ModelClass._setup_anndata(...)` to be called before instantiation — it registers the manager keyed by an AnnData UUID stored in `adata.uns`.
- `prediction_task` is `"classification"` or `"regression"`; `prediction_level` is `"node"` or `"graph"` — these gate a lot of branching throughout the codebase (loss selection, decoder output shape, evaluation output storage).
- `save`/`load` persist only the module `state_dict`, with filenames derived from `tl.utils.get_model_filename_prefix(cfg, local_component, global_component)`. `load` includes legacy state-dict key remapping (`global_module.module.` → `global_module.`, stripped `module.` prefixes, and `tl.utils.detect_and_remap_state_dict_keys`) to stay compatible with older/wandb-saved checkpoints.

Three concrete models subclass `BaseModel`, differing in which components they instantiate and which module class they wrap:
- `LocalModel` — local (graph) component only.
- `GlobalModel` — global (transformer) component only.
- `CombinedModel` — both; wraps either `CombinedModule` or `DualDecoderCombinedModule` (when `cfg.model.decoder.dual_decoder` is `True`, both local and global components get their own decoder instead of sharing one).

`LocalModule.from_config` / `GlobalModule.from_config` (`module/base/`) look up `cfg.model.local_component.name` / `cfg.model.global_component.name` and instantiate the matching module class (`"GCN"`, `"GIN"`, `"SCVI"`; `"self-attn-transformer"`) — adding a new local/global component means adding a branch there plus a corresponding config file. `GlobalModule.from_config` requires `local_mask_hops`: the combined modules pass their local module's `receptive_field_hops` (one hop per message-passing layer) and `GlobalModel` passes 0, so the long-range mask always covers exactly what the local component mixed. The config only switches the mask on or off (`long_range_attention`). (`BaseModel._register_local_component` is dead code.)

### Pooled tokens (`module/global_modules/pooling.py`)

`model.global_component.parameters.pooling` (default `kind: none`, the per-cell model, no extra state)
makes the transformer run on patches of cells (`.claude/roadmap.md`, T1b). `GlobalModule.step(batch_masked,
cell_tokens)` is the one place that does the input step and the transformer: unpooled it is
`common_step_local_to_global` + `forward`; pooled it builds the patch tokens (`PatchPooling.plan` / `pool`),
runs the same two calls on a `PooledBatch` (the patches as nodes: centroids, graph index, no edges), and
hands every cell its patch's output in the per-cell layout (`cell_layout`: left-padded, CLS last, cells in
batch order, `index_nodes` = every cell). Everything outside it -- decoders, loss, `padded_node_idx`,
probes, the receptive field -- therefore reads one row per cell, pooled or not. Invariants:

- **A patch token holds visible cells only.** Cells hidden whole (`GlobalModule.wholly_hidden`: the node
  mask, or a fully hidden row under gene masking) weigh 0; a patch with none gets one learned token. The
  grid depends on positions alone, so a hidden value cannot decide a cell's patch; any other assignment
  (`ASSIGNMENTS`) must read visible entries only.
- **`max_seq_len` bounds patches, not cells** (`PatchPooling.plan` raises when a slide has more patches).
- **Not available pooled:** the long-range mask, Laplacian/random-walk node encodings, spectral and expression
  biases, Exphormer, the attention-entropy loss (the validator says so), and every readout that indexes
  attention by cell (`evaluate()`, `attention_by_query`, `attention_diagnostics`, the rollout of `patch_flow`).
  Head scores, probes, `receptive_field`, `response_influence` and the influence flow work.

### Module hierarchy (`interscale.module`)

Mirrors the model hierarchy at the `pytorch_lightning`/`LightningModule` level:
- `BaseModule` (`module/base/_base_module.py`) owns the decoder (`interscale.nn`: `LinearDecoder`, `LinearLSEDecoder`, `NonLinearDecoder`, or `None` when a wrapping combined module owns decoding), and shared masking logic (`tl.masking.apply_mask`, via `_common_step_masking`) used for self-supervised/robustness training. `_common_step_masking` returns `(batch_masked, mask_idx, entry_mask)`; every `_common_step` returns `(local_emb, global_emb, y_pred, y_true, attn, entry_mask)`, and `entry_mask` is what restricts the loss and metrics under gene masking.
- `module/local_modules/` — local (graph) encoders: `GCN`, `GIN`, `Precomputed` (precomputed embeddings), `SCVI` (SCVI-based encoder).
- `module/global_modules/` — transformer-based global encoder (`TransformerNodeEncoderHook`) plus supporting transformer encoder/layer/utils, encoding whole-sample (long-range) context across cells.
- `module/combined_module/` — `CombinedModule` and `DualDecoderCombinedModule` compose a local module + global module sequentially: local embeddings feed into the global (transformer) component, and predictions/attention/CLS tokens are extracted from there (see `CombinedModel.get_model_output` in `model/combined_model.py` for the full inference/evaluation flow, including how attention matrices and horizontal/vertical CLS tokens are extracted and padded to `max_seq_len`).

### Data pipeline: AnnData → PyG graphs

`interscale.tl.geome_utils.prepare_geome_dataset` / `prepare_a2d_dataset` bridge AnnData and PyTorch Geometric using `geome`:
- Iterates per-sample/library (`cfg.dataset.sample_key`), builds a spatial neighbor graph per sample (`squidpy`-style `spatial_neigbors_kwargs`, converted to an edge index via `geome.transforms.AddEdgeIndex`), and yields one PyG `Data` object per sample.
- Splits data by `cfg.dataset.split_key` (an `adata.obs` column that must contain `train`/`val`, optionally `test`) — this must exist in the AnnData before calling `prepare_geome_dataset`.
- Handles both classification (one-hot encodes `prediction_obs`) and regression prediction tasks, and optionally attaches precomputed embeddings (`cfg.model.global_component.parameters.type_gex_embedding == "Precomputed"`) from `adata.obsm`.
- `interscale.geome_dataloader.GraphAnnDataModule` wraps the resulting `list[Data]` splits into a `LightningDataModule`. For node-level learning it corrupts the reconstruction input for each dataloader construction (redrawn every epoch by `NodeMaskResampleCallback`) — this is the masking scheme used for self-supervised node reconstruction/robustness experiments referenced in `config_files/legnini_example.yaml`'s `pct_mask_nodes` and `dataset.segmentation_robustness`.

### Masking (`tl/masking.py`)

Two config axes, both under `cfg.dataset`, plus optional spatial blocks. Read the module docstring of `tl/masking.py` before changing any of them.

| key | values | meaning |
| --- | --- | --- |
| `mask_strategy` | `node` (default) / `gene` | Granularity. `node` blanks whole cells (`data.mask` `[N]`) and scores all G genes of them. `gene` blanks individual `(cell, gene)` entries (`data.gene_mask` `[N, G]`, every cell a target) and scores **only those entries**. |
| `mask_percentage` | float | The Bernoulli rate, per cell under `node` and per entry under `gene` — one key, because the strategy already says what a unit is. Not comparable across strategies at equal values. `_validate_masking` rejects a regression config with a rate outside (0, 1], except 0 under `node` with spatial blocks. |
| `model.global_component.parameters.precomputed_mask_fill` | `learned` (default) / `minus_one` | Only with `type_gex_embedding: Precomputed` (embeddings from `obsm`): a hidden cell's whole embedding is replaced by this fill, since masking only touches `x` and the embedding was made from the expression being reconstructed. Gene masking is refused with `Precomputed`. Any constant fill leaves the transformer's input LayerNorm as its bias (`.claude/global_plan.md`, Stage A). |
| `mask_genes` | list of gene names or `var` selectors (`"role=response"`) | Programme masking, `gene` strategy only: hide exactly this gene set in the selected cells and nothing else. `mask_percentage` is then the share of cells (1.0: every cell), and discs select cells for it. Resolved on `adata.var` by `geome_dataloader.gene_set_mask_kwargs`; checkpoint tag `gs<rate>-<selectors>`. Hidden in every cell, a programme can only be predicted from its drivers (`.claude/global_plan.md`, programme masking). |
| `model.global_component.parameters.remask` | `none` (default) / `hidden` | Re-masking before the transformer (`GlobalModule.remask_tokens`, called by `step`): every cell hidden whole hands the transformer one learned token instead of the local component's token, which mixed in its visible neighbours. Combined models only (a `GlobalModel` refuses it), with cell masking or discs; refused with `mask_genes` (no cell is hidden whole), random entries alone, or pooling. Checkpoint tag `rmk-hidden`. |
| `mask_block_radius_um`, `mask_block_fraction` | float, float | Spatial blocks on top of either strategy (default fraction 0: off, no random draws): whole cells in discs of the radius until the fraction of each graph is covered — masked cells under `node`, whole masked rows under `gene`. A disc wider than the local reach cannot be filled in from neighbours (`.claude/attention_plan.md`, lever 3). |

Masked positions are always filled with `tl.masking.MASK_VALUE`, which is **-1** and is not
configurable. Two things are easy to get wrong here:

- **The strategy is passed explicitly, never inferred.** `apply_mask` and `_process_batch_for_metrics` both take/consult `mask_strategy` rather than checking whether a `gene_mask` attribute exists, so a stale mask on a reused `Data` object cannot turn a cell-masking run into a gene-masking one. That is why the dataloader has no clear-the-attribute helper.
- **The loss must be restricted to the masked entries under `gene`.** That is what `entry_mask` threads through `_common_step` → `TrainingPlan._regression_metrics` → `tl.masking.masked_loss`, and metrics go through `train._trainingplans.masked_regression_metrics` (same metric *names*, computed over masked entries only, verified against torchmetrics in `tests/test_gene_masking.py`). Without it the unmasked entries — which were handed to the model as input — make the objective the identity map. **Metric names are shared but the entry sets differ, so a `val_r2` from a `node` run and one from a `gene` run are not comparable as absolute numbers.**
- **`MASK_VALUE` must stay outside the layer's range; do not set it back to 0.** ~62% of legnini23's `log1p_norm` entries are already exactly 0 (and 648 cells are all-zero), so a zero fill makes a masked position indistinguishable from a real measurement — under `gene` the corruption becomes invisible and the model cannot tell which entries it is being asked for. Cell masking was less exposed because an all-zero *row* is still mostly a pattern. Measured at gene rate 0.25 over 3 seeds: fill 0 gave `val_concordance_corr` 0.0696 ±0.0113, fill -1 gave 0.0799 ±0.0046 — better on every seed and with 2.5× less run-to-run spread, below the cell-masking baseline's own CV. A learnable per-gene `[MASK]` token (GraphMAE's design) was tried and tied with the fixed -1 to three decimals — it drifts only ~0.02–0.035 from any init — so it was removed rather than kept as an option.

The registered ablation pairs are `<dataset>/node_reg` (cell) vs `<dataset>/node_reg_genemask` (gene) — note each carries its own `dataset.name` so the two arms do not share a checkpoint prefix. `config_files/sweeps/mask_granularity_legnini.yaml` runs the granularity/rate grid.

### Staged training (`train/_schedule.py`)

`optim.schedule` splits a run into phases (`.claude/global_plan.md`, Stage C): `joint` (default,
one phase, unchanged), `local_then_global`, `local_then_global_ft`, `alternating`,
`alternating_coupled`, or `custom` (`optim.schedule_phases`). Each phase can freeze components
(`frozen`) and weight the dual decoder's two losses (`loss_weights`). `optim.loss_weights`
(default 0.5 / 0.5 = the old mean of both heads) and `optim.frozen` apply to a joint run.

- **The config key is `frozen`, not `freeze`:** a yacs `CfgNode` has a `freeze()` method, so
  `cfg.optim.freeze` would return the method.
- **`train()` runs the phases in one process,** with the weights carrying over. Only the last phase
  checkpoints and saves, and all phases log to one W&B run with a continuing step.
- **Each phase's end is recorded** (`model.phase_summaries_`, and the W&B summary as
  `schedule/phase<k>_<metric>`): `drift_<component>`, the relative change of that component's
  weights (0 when frozen), and the phase's last validation losses.

### Regime losses and the own-cell skip (`tl/depth.py`, `.claude/decoder_plan.md`)

`optim.regime_loss` (default `kind: none`, every run as before) gives each head of the dual decoder the hidden cells of
its regime: `near` when the nearest visible cell of the slide lies within `depth_um` (290: the 2-hop GCN's reach on
synth_spot), `deep` beyond; `local_cells` / `global_cells` name each head's regime (`near`, `deep`, `all`).
Invariants:

- **Only the loss is cut.** `DualDecoderCombinedModule` returns `StepOutput.loss_mask` (the entry mask, each half cut to
  its cells); `TrainingPlan._step` hands that to `compute_separate_losses` and keeps `entry_mask` for every metric, so
  `*_masked_*` metrics and `_log_head_sce` mean what they always did.
- **The heads are weighed, not the rows.** Under regimes the plan always uses the weighted path (`loss_weights`,
  default 0.5 / 0.5), because one mean over the stacked rows would weigh a head by its regime's cell count. A head
  with no cell in a batch gets a zero (`_head_loss`), never a NaN.
- **Depth is `tl.depth.hidden_depth`** over every cell of the slide (also those `pad_batch` left out), in the units of
  `pos`; the threshold is divided by `dataset.spatial_unit_um`. It needs cells hidden whole (cell masking or discs, not
  `mask_genes`), and `deep` needs discs wider than `depth_um`; the validator says so. A model without two heads refuses
  the option in `train()` (`_refuse_unsupported_options`).

`model.decoder.own_cell_skip` (GlobalModel only; default off) adds the cell's own token to its prediction around the
transformer: `y = decoder(h_global) + W_o e_own` (`GlobalModule.predict(..., own_embedding=, node_idx=)`, which refuses to
run without them). `e_own` is `gex_embedding` of the *masked* batch, so no hidden entry reaches it; `W_o` starts at zero
and draws no random number. Any new caller of `predict` on such a module must pass both. The decoder type and the skip
are in the checkpoint name (`dec-<type>[-skip]_`; the plain linear decoder adds nothing).

### Preprocessing / robustness utilities (`interscale.pp`)

`pp.segmentation_noise.apply_segmentation_noise` simulates cell-segmentation errors (used when `cfg.dataset.segmentation_robustness` is set in `main.py`/`main_sweep.py`) for robustness sweeps.

### Datasets (`interscale.datasets`)

- `datasets/_legnini.py` provides `legnini()`, a loader for the Legnini et al. 2023 molecular cartography dataset used in the manuscript/tutorials — downloads from Zenodo and caches to `~/.cache/interscale/legnini_2023.h5ad`.


### Evaluation (`interscale.evaluation`)

Post-hoc analysis utilities operating on the `adata` produced by `save_evaluation_results`/`get_model_output`: gene loadings, gene-rank analysis, gene-set covariance, latent-space analysis, graph classification metrics, and network/attention stream visualization (`net_streams.py`).

What each evaluation task answers, which pass it reads, and which ones not to base conclusions on:
[`.claude/linear_probing_docs.md`](.claude/linear_probing_docs.md). In short, reconstruction from
context is read on a **masked pass** (the input corrupted as in training, seeded discs hidden):
`masked_probing` (head scores and model-free references by depth, label/covariate probes by depth
with a `smoother` feature set), `attention_diagnostics.attention_by_query` (attention of hidden
queries against uniform attention, distance profiles, planted pairs against distance-matched cells),
`receptive_field` (which visible cells a hidden cell's prediction depends on, by gradient) and
`representations` (CKA, participation ratio). A sweep writes all of it to one Markdown file
(`evaluation.report`: sections appended per arm, a cross-arm summary on top). The relevance rollout (`tl/self_attn_relevance.py`)
backpropagates the summed output after an affine LayerNorm, a degenerate target, so `_attn_matrix`,
`cls_horizontal`/`cls_vertical` and everything built on them (`downstream_regression`,
`net_streams`, `flow_control`, `graph_classification`) are not interpretable. Two yardsticks read how well the two
components separate: `representations.redundancy_table` (ridge R² of one embedding from another, fitted on the training
slides per query group) and `masked_probing.attribution_shares` (each named part's share of a head's variance; the parts
come from `StepOutput.parts`, set by a global-only model with the own-cell skip and by a residual dual decoder -- a module
that makes a head a sum of parts must put them there, in the rows of that head's `y_pred`). `ref_oracle`
(`oracle_features`, sweep settings `ORACLE_KEYS` / `ORACLE_GATES`) is a ridge on the true drivers of a synthetic dataset:
a bar for the task, listed beside the references but never counted as "the best reference". `generate_relevance(output,
target=...)` traces a given target instead; `evaluation.attention_check` checks the attention against torch's
stock modules on real inputs (`check_attention`) and compares seven readouts of it against planted pairs
(`attention_readouts`); a masked pass hooks its maps inside `TransformerNodeEncoderHook.hooking_attention()`.

#### Online probes (`evaluation/online_probes.py`)

`linear_probing.py` probes a finished run's `.h5ad`. `online_probes.py` probes *during* training,
as a Lightning callback, so a local-vs-global comparison is a curve rather than one end-of-run
number. It is off unless `cfg.probe.use` (see `config/probe_config.py`); when off, no callback is
built and the trainer is unchanged.

Every probe reads the **same target out of both embeddings** with the same readout — the gap
between the two columns is the point, not either number alone. With a node positional encoding a
third, control column `pe` (the summed node PE as added to the tokens) is probed too: a global gain
on a position-derived target counts only where it beats `pe` (`.claude/PE_plan.md`, Stage 6).
Targets are `probe.classification_targets`
(an optional annotation attached to the PyG graphs, so `celltype` requires `dataset.celltype_key`;
scored with macro precision/recall via logistic regression) and `probe.regression_genes` (a column
of `adata.var_names`; scored with MSE and R2 via ridge). `probe.gene_vector: true` adds the whole
gene vector as one target: a ridge per gene, scored by the cosine of predicted and true vectors over
each cell's hidden entries (`masked_probing.probe_expression`, the post-hoc scoring), logged as
`probe/gene_vector_cosine_{raw,centred}_<embedding>` with a `mean` floor column for the raw cosine.
Results are logged as flat
`probe/<target>_<metric>_<embedding>` scalars — monitorable by EarlyStopping/ModelCheckpoint/a
sweep — plus, under wandb, one custom chart per target carrying both embeddings as two lines. The
`probe/` prefix is load-bearing: wandb sections split on the text before the first `/`, so it keeps
these out of the default panel section where InterScale's own `train_`/`val_` metrics live.
`probe.attention_diagnostics: true` adds per-layer, per-head attention diagnostics over the first
`probe.attention_graphs` val graphs, under their own masks (`probe.attention_masked`, default; false
reads a clean pass): span in µm and relative to uniform attention, the share beyond each of
`probe.attention_radii_um`, CLS mass, effective keys (also over the keys a cell can attend to), and
the Performer's TV distance to softmax, as `probe/attn_<metric>_l<layer>_h<head>`
(`evaluation/attention_diagnostics.py`; Stage 1 of `.claude/attention_plan.md`).

Numeric `obs` targets reach the graphs as an **obsm matrix** (`dataset.probe_obsm_key`), because
geome cannot attach a numeric obs column — it returns a pandas Series and dies in `torch.cat`.
`evaluation.build_probe_obsm` stacks them and must run *before* `prepare_geome_dataset`; every
entrypoint (`main.py`, `main_sweep.py`) already calls it.

Three invariants, each guarding a failure that does not raise:

- **Gene targets are masked; label targets are not.** These are opposite failure modes and the
  probe runs a separate pass for each. A *gene* is part of `batch.x`, so scoring it on a cell the
  encoder was given measures invertibility, not learning: on synth_data_0 at `mask_percentage` 0.3
  the structure-free `noise_00` control probed at **R2 0.33 from both embeddings** unrestricted and
  **~0.02** restricted to masked cells — `probe.masked_cells_only` governs this and must stay True.
  A *label* (`classification_targets`, `regression_obs`) is never a model input, so there is nothing
  to leak and masking only deletes signal: masking cell type held it at **0.22** macro precision,
  which is the Bayes ceiling given only the niche (0.19), while the clean pass reaches **0.46** after
  three epochs against a raw-expression ceiling of 0.74.
- **The probe consumes no RNG.** It builds its own `shuffle=False` loaders over `datamodule.train_data`
  / `val_data` rather than iterating `train_dataloader()`, whose shuffle draws from the same torch
  generator the training batch order comes from — so enabling the probe would reorder training
  batches and change the model being measured. Dropout is off, masks are read not redrawn, and the
  subsample uses its own seeded numpy generator. Even an unshuffled `DataLoader` iterator seeds
  itself off the global torch generator, and `pad_batch` subsamples long graphs with Python's
  `random`, so `collect_features` restores every global generator afterwards (`_rng_untouched`).
- **Both halves see identical rows.** The local embedding is gathered by `ViewOutput.padded_node_idx`,
  the same index that brings global tokens into cell order; row sets are built once per target and
  shared by both embeddings. Misalignment scores at chance, which reads like a real negative result.

The readout is fit on train and scored on val (standard linear probing), which costs one extra
no-grad pass over each split — `probe.every_n_epochs` is the knob for that. The val mask is fixed
for the whole run (`NodeMaskResampleCallback` only redraws train), so the scored rows are stable
across epochs and the curve is comparable.

## Repository conventions

- Numpy-style docstrings (see `docs/contributing.md`); many docstring lint rules are intentionally disabled in `pyproject.toml` for legacy modules, but new public APIs should still be documented in numpy style since `sphinx-autodoc-typehints`/napoleon render them for the docs site.
- Code style is enforced by `pre-commit` (ruff-check/format, biome-format, pyproject-fmt, whitespace/merge-conflict hooks) — run `pre-commit install` once locally.
- Tests live under `tests/`, use `pytest` with `--import-mode=importlib`; shared fixtures (e.g. a small synthetic `AnnData`) are defined in `conftest.py` at the repo root.
- Coverage is measured over the `interscale` package only (`[tool.coverage] run.source = ["interscale"]`), excluding `test_*.py` files.
