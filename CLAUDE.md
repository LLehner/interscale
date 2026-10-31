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

Hierarchical coarse-to-fine attention for the global component (a cluster hierarchy per slide, each level's attention
steered by the level above, routed key sets for whole CosMx sections) is planned in
[`.claude/hierarchical_attention_plan.md`](.claude/hierarchical_attention_plan.md): the user's plan, with a status table and
how it was built on top. Read it before touching `hierarchy.py`, `hierarchical_attention.py` or the hierarchy's path through
`common_step_local_to_global`, and update its status table in the same commit as the work it describes.

Three plans written 2026-10-29 for the focus until Tuesday 3 November (roadmap.md, the section after the status table), each with a
status table: [`.claude/masking_plan.md`](.claude/masking_plan.md) (hidden vs visible scores, randomized vs pre-selected gene sets,
Moran's I gene sets, niches as a mask unit; read it before touching `tl/masking.py` or the masking keys),
[`.claude/lung_plan.md`](.claude/lung_plan.md) (the lung fibrosis data with niches, compared with the other tools), and
[`.claude/interaction_readout_plan.md`](.claude/interaction_readout_plan.md) (a brief for a separate session: integrated gradients
downstream, uncluttered sender-receiver figures, flow without cell types).

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
  (`ASSIGNMENTS`) must read visible entries only. The one deliberate exception is `kind: label` (niche patches,
  below): the label is an annotation computed from expression, so it is an annotation arm, never the default.
- **`max_seq_len` bounds patches, not cells** (`PatchPooling.plan` raises when a slide has more patches).
- **Not available pooled:** the long-range mask, Laplacian/random-walk node encodings, spectral and expression
  biases, Exphormer, the attention-entropy loss (the validator says so), and every readout that indexes
  attention by cell (`evaluate()`, `attention_by_query`, `attention_diagnostics`, the rollout of `patch_flow`).
  Head scores, probes, `receptive_field`, `response_influence` and the influence flow work.

### Niche patches (`tl/niches.py`; `.claude/lung_plan.md` L2-L3, `.claude/masking_plan.md` M3)

A patch is one connected piece of a categorical label (a niche) on the slide's own graph (`edge_index`): the components left once
every edge between two labels is cut, so two places of one niche are two patches. `min_cells` merges smaller patches into the
neighbour they share most edges with (a merged patch can hold two labels), `max_cells` cuts larger ones into kdtree pieces; an
unlabelled cell (NaN) is a label of its own. Three options read them: the hierarchy's top level (`hierarchy.top_level.mode:
obs_label`, `obs_key`, `split_by_space` none / components / kdtree), pooled tokens (`pooling.kind: label`, `label_key`,
`min_cells`, `max_cells`) and the mask unit (`dataset.mask_block_kind: label`, `mask_block_label_key`, `mask_block_max_cells`).
Invariants:

- **The label reaches the graphs through `dataset.extra_obs_keys`** (one-hot), resolved by `geome_utils.label_attribute` (a named
  role such as `celltype_key` arrives under the role's name); `niche_codes` reads it back. The validators refuse a column that
  does not arrive.
- **It is an annotation arm.** A niche label is computed from expression, hidden cells included, so it breaks "positions alone"
  (the hierarchy, the grid, discs): a hidden cell's patch says something about it. Read every such arm as an upper bound.
- **Routed hierarchies need `split_by_space: kdtree`.** Routing gives each parent `cap_per_query // (widest parent's children)`
  candidate parents: one large niche patch on top starves every query of the slide (the layer warns). `kdtree` cuts patches to the
  first level's size, `min_cells` keeps level 0 (dense) small.
- **The disc draw is unchanged.** Niche blocks share the discs' loop (`tl.masking._draw_blocks`); a disc run draws the same random
  numbers as before, and a niche block takes no radius.

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
| `mask_channel_fraction` | float in (0, 1), 0 = off | Channel masking, `gene` strategy only: the label-free form of programme masking. A random share of the genes (`round(fraction * G)`, `tl.masking.sample_channel_columns`, never none and never all) is hidden in every selected cell -- the same set for the whole graph, so no neighbour shows a hidden gene -- and drawn again whenever the graph's mask is redrawn: every epoch for the training graphs (`NodeMaskResampleCallback`), once at setup for validation and test. `mask_percentage` is the share of cells (1.0: every cell; a lower one warns, since the other cells show every gene), discs select cells as for `mask_genes`; not together with `mask_genes`. No cell is hidden whole (`tl.masking.hides_gene_set` is true for it and for `mask_genes`, which re-masking, the regime loss and visible-keys-only attention refuse), and a `Precomputed` embedding is refused: every gene is hidden at some step. Checkpoint tag `ch<fraction>c<cell rate>`. Default 0: no random number is drawn. |
| `mask_channel_genes` | selectors as for `mask_genes`, empty = every gene | The pool channel masking draws from (`.claude/masking_plan.md`, M1): `round(fraction * pool)` of its genes per draw, every other gene always visible (`tl.masking.sample_channel_columns(pool=)`; without a pool the draw is the old one, random number for random number). Needs `mask_channel_fraction`. Tag `-p<selectors>` after the channel tag. |
| `mask_gene_sets`, `mask_sets_per_draw` | a list of sets (each a list of selectors), int (default 1) | A pool of whole gene sets (M1): at every draw `mask_sets_per_draw` distinct sets are chosen (`sample_set_columns`, `sample_sets_mask`) and their union is hidden in every selected cell of a graph, redrawn as channels are. Not with `mask_genes` or channels; a flat list of strings is refused (it would be read as sets of characters). `hides_gene_set` is true, so the options that need cells hidden whole refuse it, as `Precomputed` does. Tag `sets<k>of<n>c<cell rate>-<sha1 of the sets>`. Empty (default): off, no random number drawn. |
| `model.global_component.parameters.remask` | `none` (default) / `hidden` | Re-masking before the transformer (`GlobalModule.remask_tokens`, called by `step`): every cell hidden whole hands the transformer one learned token instead of the local component's token, which mixed in its visible neighbours. Combined models only (a `GlobalModel` refuses it), with cell masking or discs; refused with `mask_genes` (no cell is hidden whole), random entries alone, or pooling. Checkpoint tag `rmk-hidden`. |
| `mask_block_radius_um`, `mask_block_fraction` | float, float | Spatial blocks on top of either strategy (default fraction 0: off, no random draws): whole cells in discs of the radius until the fraction of each graph is covered — masked cells under `node`, whole masked rows under `gene`. A disc wider than the local reach cannot be filled in from neighbours (`.claude/attention_plan.md`, lever 3). |
| `mask_block_kind`, `mask_block_label_key`, `mask_block_max_cells` | `disc` (default) / `label`, obs column, int (0) | What a block is (`.claude/masking_plan.md`, M3): `label` hides whole niche patches (`tl/niches.py`; the column in `extra_obs_keys`) drawn by the discs' loop (`sample_unit_mask`), no radius; `max_cells` > 0 cuts larger patches into kdtree pieces. The evaluation draws them with `masked_probing.draw_masks(label=, max_cells=)`. Tag `blklab-<key>[-mx<max>]f<fraction>`. |

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

### Distance prior per head and visible keys only (`.claude/roadmap.md`, T1)

`pe.bias: [distance]` with `pe.distance.kind: scales` (`ScaledDistanceBias`) gives head `h` the window
`-strength * phi(d / l_h)` (`shape` gaussian: `phi = u²/2`, or exponential: `phi = u`) with the scales `l_h` spaced geometrically
over `[scale_min_um, scale_max_um]`. It is a prior: **not zero at step 0**, unlike every other bias (the "a bias starts at zero"
convention of the other kinds does not hold). Scales and strength are learned unless `frozen`. Dense and pair forms only (not the
Performer). With the long-range mask on, `scale_min_um` should be about the local reach.

`attention.visible_keys_only` (default off) blocks, in the dense mask of `common_step_local_to_global`, every key that is a cell hidden
whole (`GlobalModule.wholly_hidden`); a hidden cell still queries, CLS stays open, a clean pass (`eval_step`) is untouched. The key
vector is laid out with `pad_like` from this call's `index_nodes` -- never a second `pad_batch` -- so it follows left padding and
subsampling. It must run *before* the bias is merged into the mask. Needs a dense-mask backend (the validator names which) and
cells hidden whole (not with `mask_genes`, pooling, or gene masking without discs). Checkpoint tag `-vk` in the attention part.

### Exact fused attention (`attention.kind: fused`; `.claude/attention_plan.md`, stage 6)

`attention.FusedAttention` is dense softmax attention, exactly, computed by `F.scaled_dot_product_attention` a block of queries at a
time, so a sequence of ~10^5 cells (a whole CosMx section) forms no `S x S` tensor. Invariants:

- **It is the dense model.** It takes its mask as a `StructuredMask` of blocked pairs (`mask="pairs"`): without the long-range mask
  that is each cell's own key (`attn_mask_diagonal`'s diagonal), with it the k-hop pairs -- never an open diagonal. Biases come a block
  of rows at a time from `AttentionBias.rows` (`StructuredMask.bias_rows`), rows of exactly the dense bias: every bias with a pair form
  (distance in every kind, Fourier, spectral); the expression bias, `visible_keys_only` and the entropy loss are refused.
- **No block is kept.** `rows = chunk_pairs // (B * L)` queries per block (`attention.fused.chunk_pairs`, default 2^24); each block's
  additive mask is built when it runs and, in training with more than one block, rebuilt in the backward pass (non-reentrant
  `torch.utils.checkpoint`). Anything a block computes must stay inside the block function; a tensor of all blocks kept outside it is the
  `S x S` memory again. One block (synth_spot's 2,501 tokens) is plain SDPA with the whole mask.
- **Exact distances.** `rows` takes `cdist` without the matrix-product shortcut, as the dense form: at a section's global coordinates
  (tens of mm) the shortcut loses whole µm, which a 15-20 µm window turns into wrong logits.
- **Maps only when read** (`need_weights`): the explicit path, the output computed from the returned map (relevance hooks it).
- **Dropout** is SDPA's own stream: fused and dense agree at dropout 0 or in evaluation, and pair by seed in training.

### Hierarchical coarse-to-fine attention (`hierarchy.kind: coarse_to_fine`; `.claude/hierarchical_attention_plan.md`)

`model.global_component.parameters.hierarchy` (default `kind: none`: nothing built) replaces every encoder layer by a
`HierarchicalEncoderLayer`: the dense layer (same projections, FFN, norms, residuals) with an attention block that pools the
cells into a per-slide cluster hierarchy (`hierarchy.py`: kdtree / random, a tree or patch top level, cached by the slide's
positions), attends densely among the coarsest clusters and CLS, then level by level with a prior from the level above
(`prior_mode`: none, soft, factorized, factorized_percell; `direction_prior`), over every token (`candidates.mode: dense`) or
the children of each parent's top-k targets and spatial neighbours (`routed`, a padded gather a block of parents at a time,
never an `S x S` tensor), and fuses each cell with its ancestors (`readout`). Invariants:

- **No `S x S` mask.** `common_step_local_to_global` returns a `HierarchicalContext` (per graph: the kept cells' positions in
  µm, which are hidden whole, the hierarchy, laid out from this call's `index_nodes`) as `GlobalInput.attention_mask`; `forward`
  hands it to the layers as `hier_context`, set and cleared around the call like `rotary`.
- **The dense model is a special case.** Prior soft, lambda 0, dense candidates and singleton clusters (`cells_per_cluster:
  [1]`) or no coarse level give the dense transformer's predictions and gradients exactly (tested), because every added
  parameter is zero or drawn from a private generator, CLS stays an open key with bias 0 at every level, and the cell level
  blocks a cell's own key. Keep all three when changing the layer.
- **Positions alone build the hierarchy; hidden cells weigh 0 in every pooled token** (a cluster with no visible cell is a
  learned token). Exploration draws only from the module's own generator. The exception is the `obs_label` top level (niche
  patches, above): an annotation arm, cached by labels and edges as well as positions.
- **Readouts.** `recording_hierarchy()` exports each layer's per-level attention (dense, or `SparseRows` when routed);
  `evaluation.hierarchy_readouts` reads hierarchical and dense models alike (sender lift, pair AUPRC, arrow angle, routing
  recall, health, fields, and the net attention flow of `evaluation.flow_maps`). Dense candidates also hand `forward` cell-level maps; routed models hand none. The exported fields
  carry per head the cell mass, the top-k targets and `window_vec_heads` (where the head's own trained position biases alone
  would read over the same keys, scaled to its cell mass: the baseline of a trained head, in place of an untrained twin), the
  planted pairs with their programmes, and with `dependence_genes` each receiver's integrated-gradients dependence
  (`receptive_field.receiver_dependence`: a query over the completeness limit is integrated again with more points).
- Refused (validator): other backends or head widths, RoPE, spectral / expression biases, the long-range mask, the context
  query, `visible_keys_only`, `remask`, pooling, losses reading attention maps; prior none without a readout.

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
are in the checkpoint name (`dec-<type>[-skip|-skipraw]_`; the plain linear decoder adds nothing).

### The additive decoder and the raw skip (`.claude/decoder_plan.md`, D3.1)

`model.decoder.additive` (dual decoder only; default off) makes the dual model's prediction the sum of its two decoders,
`y = W_l h_local + W_g h_global + b`. Invariants:

- **The halves.** The stacked `[2N]` output keeps its layout: the local half is the local decoder's output **alone**, the global
  half is the **sum** -- the prediction the model reports, as under `global_residual`. `DualDecoderCombinedModule.combine_global` is
  the one place that sums (training, `get_model_output`), `global_is_sum` is true for `additive` and `global_residual` (the two
  are exclusive: one cuts the gradient into the local decoder, the other does not). No gradient is cut.
- **The loss weights are a choice.** The default 0.5 / 0.5 also trains the local part to predict the whole target alone, which is what
  the option is for not doing; the validator refuses it under the `joint` schedule. `{local: 0, global: 1}` is the loss on the sum only,
  a small local weight a per-head term; a staged schedule sets its own per phase.
- **Parts and heads.** `StepOutput.parts["global"]` is `{"local", "global"}`, the sum's own tensors (not detached: the influence readouts differentiate
  the head `global_part`), what `attribution_shares` takes apart. `masked_probing._model_heads` adds the head `global_part` (the global decoder's output alone) for
  an additive module, so it appears in every head table and every consumer that loops over heads (`response_influence` runs a backward
  per head).
- **It separates nothing by itself.** The global component's tokens are built from the local ones, so the sum is a reparametrisation the
  optimiser may use unevenly; the attribution shares say how much each part carries. A context-only global embedding (D1) is what stops
  the global part repeating the local one.

`model.decoder.own_cell_skip_input` (`embedding` by default, or `raw`): what the own-cell skip reads. `raw` is the cell's visible genes
(`GlobalModule.own_cell_input`: the masked batch's `x` in training, hidden entries at `MASK_VALUE`; the clean batch at inference), `W_o`
is `[n_output, n_input]`, so the read has full rank; `embedding` is the token, rank at most `n_embed`. Both start at zero. Any caller of
`predict` must pass `own_embedding=module.own_cell_input(batch, embedding)` and the same batch that was tokenised.

The label and distance probes of a disc evaluation are the slowest step (one logistic regression per label and feature set):
`probe_labels` / `probe_covariates` take `max_train_rows` and `joint_embeddings`, the sweeps set them through `PROBE_*` in
`sweep_common.py` (not under version control). The defaults fit everything.

### The context query (`.claude/decoder_plan.md`, D1)

`model.global_component.parameters.context_only: query` (default `none`) makes a cell's global embedding a function of the *other* cells'
tokens and its position. The transformer is then not run as an encoder: every cell (and CLS) is a **query stream** starting as
`norm_input(context_token + node encoding)`, which in each layer reads the cells' *input* tokens (`norm_input` of the tokens, CLS appended;
the same memory at every layer, never updated) through that layer's own weights (`CustomTransformerEncoderLayer.read_context`, built on
`_attend(x, memory, ...)`, which is the self-attention block with `memory is x`), and the output is the encoder's final norm
(`TransformerNodeEncoderHook._read_context`). Invariants:

- **A cell is blind to itself.** Its embedding does not move, bit for bit, with its own expression (tested at 1-3 layers, with node encodings,
  biases, RoPE, pooled patches, and behind a GCN under the long-range mask, which blocks the neighbours' tokens that hold the cell). Keep
  the memory the layer-0 tokens and keep the stream out of it: reading the stream's own state, or the updated tokens of other cells, would open
  a path back to the cell. `attention.center_qk` is the one exception (the mean it centres on holds the cell at weight 1 / S).
- **The queries ride on the input.** `GlobalInput.queries` is an *attribute* of the returned tuple (a subclass), not a sixth field: `common_step_local_to_global`'s
  five fields are unpacked by position in many places. It is the very node encoding added to the tokens of this pass (the encodings draw their
  random signs and rotations once), taken through the selection `pad_batch` made (`pad_like`), zeros without an encoding. `forward` needs it
  exactly when the option is on (`GlobalModule._query_kwargs` passes it); a caller that unpacks the five fields and calls `forward` itself must
  pass `queries=`.
- **Zero-initialised encodings.** Every node-encoding projection and attention-bias table starts at zero, so a test of anything positional must
  set them to nonzero values first (the context-query tests do), or it checks nothing.
- **Torch's dense attention only** (the vendored one takes separate queries, keys and values; the backends are self-attention only). The
  readouts that compose attention as token mixing (`attention_check`, the rollout flow, relevance) refuse a context-only model; the
  single-layer attention maps (`attention_diagnostics`) and the gradient readouts work. The returned attention maps are the read-out's.
- The learned token is built from a private generator: the option draws nothing from the global one.

### Orthogonality terms (`train/orthogonality.py`, `.claude/decoder_plan.md`, D4)

Eight auxiliary losses, each its own weight `optim.aux_loss_weights.orth_<name>` (default 0: never built): between the dual
model's local and global embedding over cells (`orth_cka`, `orth_cca`, `orth_kernel_cka`, and `orth_opl` as a control),
between the two parts of a decomposable prediction (`orth_gene_corr` on their predictions per gene, `orth_loadings` on
their gene-space loadings), and within the global part (`orth_decorrelation`, `orth_loading_ortho`). Invariants:

- **Scale-free, in [0, 1].** The additive decoder rescales each part freely, so a raw Frobenius penalty can be met by
  shrinking a part. Which other changes a term ignores decides how it can be gamed: CKA is variance-weighted (an inflated
  unused direction lowers it), the canonical overlap reads through any invertible linear map down to `optim.orthogonality.ridge`,
  OPL's term is met by a rotation into the orthogonal complement. A run is read by quantities it did not train.
- **No dropout on the path.** With dropout, training meets a term that reads cells with noise evaluation removes (toy: train
  CKA 0.02, validation 0.999). Every such term refuses encoder dropout at construction (`orth_gene_corr` also a nonlinear
  decoder's); compare it only with arms trained without dropout too.
- **Rows and centring.** A term reads the cells whose predictions the loss reads (`entry_mask` rows; the local embedding
  gathered by `padded_node_idx`), each graph centred on its own; the pairwise terms subsample with a generator of their own.
- **The owner.** A loading term reads the decoders of the module that holds it, kept in a tuple that `nn.Module` does not
  register: not a child, not in the state dict, and a deep copy points at the copy. Add no registered back-reference.
- Checkpoint names carry every non-zero auxiliary weight (`aux-<name><weight>`, since 2026-10-23, the contrastive terms too).

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

#### Gene scales (`evaluation/gene_scales.py`)

Which genes vary on a local and which on a tissue scale, and which part of a decomposable prediction carries each
(`.claude/roadmap.md`, task 1). Per gene, Moran's I against the `k` nearest other cells of the same slide (shells `k_prev+1..k`
and cumulative), its length scale (`ell_k`, `ell_um`: where the shell curve halves after its peak) against a permutation null;
the model side is each part's share of a gene's prediction (`part_shares`). Invariants:

- **The statistic.** Each gene is z-scored within its slide, neighbours never cross a slide, the sums are pooled over slides.
  With sampled centres it is a ratio estimator (`sum z_c * mean z_nb / sum z_c^2` over the centres), which with every cell a
  centre is the textbook statistic (tested to 1e-12); dividing by the slide variance instead is noisier (up to 0.15 off on a smooth
  field). Every draw (centres, shuffles, folds) uses a private numpy generator.
- **Only a decomposable prediction is split** (`StepOutput.parts`: the additive decoder, the residual decoder); `collect_fold_parts`
  refuses a head without parts. The plain dual decoder's two heads both predict every gene and are not read here.
- **The parts are read where the model was trained to predict.** `collect_fold_parts` hides the cells of each slide fold by fold
  (`cell_folds`, `n_folds = round(1 / mask_percentage)`) and keeps each cell's row from the fold that hid it, then puts every graph's
  own `mask` / `gene_mask` back. On a fully visible pass every output sits where it never got a loss.
- **Every gene must have been a training target** for its share to mean anything: under programme masking the decoder rows of a gene
  outside the programme get no gradient. The sweep `gene_scales` trains with cell masking (`ALLOW_CELL_MASKING` in `sweep_common.py`).
- `pattern` is `anti` when the first shell is below the null, `variable` when the highest shell is above it (Bonferroni over the
  shells); each gene's null SD is floored at the median over genes (10 shuffles underestimate some).

**Scale sets** (`gene_scales.scale_sets`, `.claude/masking_plan.md` M2): a local and a global gene set without labels. `I_local` is
Moran's I with the graph's own hop neighbourhoods (`hop_neighbourhoods`: the adjacency the local component reads, made symmetric, never
across slides; the largest over 1 .. `hops`), `I_global` Moran's I across the slide with inverse-distance weights `1 / max(d, d0)` over
every other cell (`global_moran`; uniform weights would give `-1 / (n - 1)` for every gene). Candidates are significant against
shuffles (Bonferroni over genes) with `I_local >= min_local_i`; they are ranked by the **relative** fall `(I_local - I_global) / I_local`
(`rule: absolute` is `|I_local - I_global|`, which puts weak local genes into the global set: notes.md 2026-10-29); the smallest falls
are the global set, the largest the local set. Pass the training slides only. `write_scale_sets` puts `<prefix>_set` into `var`, so
`mask_genes: ["scale_set=global"]` hides the global set. The cluster tool is `notebooks_and_scripts/slurm/new/scale_sets.py`.

**Hidden against visible cells** (`masked_probing.visibility_scores`, `visible_view`; M0): the same masked pass scored on the visible
cells too -- every entry of each cell with no hidden entry -- with `HeadPredictions.near_hidden_um` for the bins. Every reference of
`reference_predictions` leaves the query cell out, the slide mean included since 2026-10-29 (a hidden cell's numbers are unchanged:
its entries were never in the sums), so at a visible cell a reference is its context alone. Under programme masking no cell is
visible as a whole and only the `groups` table comes back. In the cluster sweeps it is the `VISIBILITY` setting of `sweep_common.py`
(`{}` = defaults; skipped for a run trained without cell masking).

#### Attention flow maps (`evaluation/flow_maps.py`; `.claude/roadmap.md`, task 5)

One figure per slide of the **net attention flow**, annotation-free, for whole sections (routed or dense attention, no `S x S` map). `A` is the last layer's
head-mean attention over cell keys (`hierarchy_readouts.cell_rows`; CLS, padding and a cell's own key left out); a pair's net flow is `F_ij = A_ij - A_ji`,
positive when `i` reads `j` more, so information goes `j -> i` (the manuscript's `M - M^T`). Each such pair is an arrow deposited at its midpoint, summed on a
grid (`flow_grid`; stored in the npz, so a plot is redrawn with another bandwidth), Gaussian-smoothed and divided by the smoothed cell count: the net current per
cell. Invariants:

- **Mutual pairs only** by default: a routed row holds its parent's candidate keys, so a pair known in one row only would show as a net flow of its full
  weight. `mutual_share` (table `hier_flow`) is the attention mass on pairs known in both rows; with dense attention it is 1.
- **It is where the model reads, not signalling.** Checked on synth_spot's trained windowed models (notes.md, 2026-10-31): the head mean's top 5 holds a planted
  short-range sender for 97% of the receivers (untrained twin 69%: the distance window) and a mid-range one for 73% (twin 7%), and the net flow points from sender to
  receiver for 61% / 94% of the pairs (twin 49% / 42%); dense, sparse and directional senders are not found, and integrated gradients are stronger (mid range 100%).
  Part of any net flow is geometry (density, the tissue's edge, the layout of the mask: hidden cells read the visible ones around them). Read it against the untrained twin.
- **Nothing but tissue is drawn:** `tissue_blocks` (bins joined across gaps under `link_um`; a piece that is not a rectangle, two fields touching at a corner,
  is cut by `rectangles`) gives a tile per block, `tile_layout` packs them at one scale. `hierarchy_readouts(flow_dir=...)` writes `<graph>.npz` and the
  `hier_flow` table (off by default); the CosMx hook (`slurm/new/hier_common.py`) draws `flow/<run>/<section>.png`. The quiver plots (`plot_fields`) are not
  drawn on CosMx.
- **Sources and sinks** (2026-10-31): the npz also holds each cell's net flow out and in (`cell_strengths`; `out - in` is the exact divergence, summing to 0)
  and the top-5 sources of up to 2,000 sampled receivers, hidden cells first (`sample_pairs`, a private seeded generator). Three views per section from it:
  the streams with the smoothed `out - in` shaded behind (`smoothed_net`; red sources, blue sinks), the source-sink map (`plot_source_sink`, the strongest 1%
  ringed) and zoom panels on the largest hidden patches with arrows from each receiver's sources (`plot_who_whom`). Hidden cells are the receivers by
  construction, so sinks follow the mask: read against the twin. An npz from before has no strengths: the streams skip the shading, the other two refuse.

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
