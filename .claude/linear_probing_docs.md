# Evaluation tasks: what each one answers

The question behind all of them: does the global (transformer) component know something about a
cell that the local GCN and a plain average of the neighbours don't? Every task reads a model under
one of three passes:

- **masked**: the input is corrupted as in training. In the scripts, seeded discs of 500 µm hide
  30% of each test slide, and the hidden spots are what gets scored.
- **clean**: nothing hidden. Every cell sees its own expression, a regime training never shows.
- **unmasked output**: `get_model_output` hides nothing, so a head is scored on values it was given.

Status as of 2026-10-18. *Current* means used by `slurm/global_only.py` and the sweeps after it.

## Current

| Task | What it answers | Where | Pass |
|---|---|---|---|
| Head scores by depth | How well each decoder head fills in hidden spots: centred cosine and raw SCE per cell, per gene (cosine), per programme, and by depth (µm to the nearest visible spot) | `masked_probing.collect_head_predictions`, `score_heads(gene_metrics=("cosine",))` | masked |
| Head scores per gene group and depth | The same centred cosine pooled over each gene group's hidden entries, per depth bin: whether the genes that carry context (receivers, responses) are filled in deep in the hidden area, apart from noise genes no model can predict. Groups from config (`GENE_GROUP_KEYS`: var columns or `a/b` combinations, named `<key>=<value>`) | `score_heads(gene_groups=..., depth_bins_um=...)` → `by_group_depth` | masked |
| References | The bar a head must clear: averages of the visible spots (slide mean, Gaussian kernels, k nearest, oracle label mean). No model involved | `masked_probing.reference_predictions` | masked |
| Blend test | Each model head averaged with one reference (`BLEND_REFERENCE`, the 400 µm kernel), as `blend_<head>`. Above both: the head holds something the reference lacks. A plain average, so the one with larger deviations from the centre counts for more | `masked_probing.blend_heads` | masked |
| Label and distance probes by depth | What a hidden spot's embedding says about its context (region, states, distances to sources), per depth bin | `probe_labels`, `probe_covariates` with `depth_bins_um` | masked |
| Smoother feature set | The same probes on the references used as features. `smoother+<embedding>` minus `smoother` is what the embedding knows beyond a smoother of the visible spots | `masked_probing.smoother_features` (via `collect_features(extra_features=...)`) | masked |
| Position feature set | The same probes on where a spot lies and nothing else: its offset from its slide's centroid, its distance from it, and a one-hot of the patch it falls in on grids of several sizes (`POSITION_PATCHES_UM`). What position alone says, which an encoding hands the model without learning. The model-free bar is `smoother+position`; `smoother+position+<embedding>` minus it is what the embedding knows beyond both | `masked_probing.position_features`, combined with the smoother by `combined_features` | masked (positions are never hidden) |
| Representations | Linear CKA between embeddings and the smoother (is the global token a smoother? a copy of the local one?), and each set's participation ratio (do hidden tokens collapse?), by depth | `representations.representation_table` | masked |
| Attention by query | Where hidden spots look, per layer, head and depth bin, against uniform attention: attention on visible spots over their share of the keys (`visible_lift`), `span_rel`, `effective_rel`, mass beyond radii, CLS mass, the Performer's TV to softmax | `attention_diagnostics.attention_by_query` → `by_query` | masked |
| Attention profile | Attention per distance bin over uniform weights (1 = uniform): the shape of what a head looks at | same → `profile` | masked |
| Planted pairs, attention | Does a hidden receiver attend to its visible sender more than to visible spots at the same distance? (`matched_rank`, 0.5 = blind to the edge; `coverage`) | same → `edges` | masked |
| Receptive field | Which visible spots a hidden spot's prediction depends on: squared Jacobian norms (Hutchinson), by depth and distance, per head. Replaces relevance. The GCN alone must read 0 where it cannot reach, which checks the instrument | `receptive_field.receptive_field` → `by_query`, `profile` | masked |
| Planted pairs, influence | The same matched rank on the receptive field: does the prediction use the sender? | same → `edges` | masked |
| Patch and label flow | Where the cells of each patch draw their predictions from: `influence` (gradient flow on their predicted hidden entries, through visible entries, own input left out) and `rollout` (attention composed over the layers with the residual path), each against the same model on the slide with the visible cells' expression permuted -- everything geometric survives the permutation, so observed vs permuted is what depends on the tissue's arrangement. Reach, own-patch share, KL and centre-of-mass shift against their permutation floors; flow fields, evenly spread source maps, label heatmaps (observed, permuted, log2 ratio with stars). Read against the untrained twin: random networks depend on content too | `attention_flow.patch_flow`, `plot_flow_field`, `plot_source_maps`, `plot_label_flow`; scripts: `FLOW_*` in `sweep_common.py` (on for `legnini_attention.py`, `response_masking.py`) | masked (discs, several draws) or gene-set |
| Programme scores (gene-set evaluation) | With a gene set hidden in every cell (`dataset.mask_genes`, `EVAL_MASK = "gene_set"`): each head's centred cosine per programme and interaction, next to references fitted on the training slides -- the cell's own visible genes, plus kernel means of the visible genes around it, plus position. The bar is `ref_own+context`: a model beyond it knows more than fixed kernels of its inputs | `masked_probing.FittedReferences`, `draw_masks(kind="gene_set")`, `collect_head_predictions(fitted_references=...)` | gene-set mask |
| Sender influence on responses | For each planted programme, the sender's rank in its receiver's predicted response genes (that programme's only) among cells as far away; a cell counts through its visible entries. 0.5 is a prediction blind to the sender | `receptive_field.response_influence` → `response_edges`, `response_profile` | gene-set mask |
| Untrained twin | Every task again on the arm's model at initialisation (same seed). Only a gain over it is learned | `slurm/sweep_common.py` (`EVAL_UNTRAINED`, `UNTRAINED_ARMS`, `<arm>_untrained`) | masked |
| Phases (staged training) | How far each component's weights moved in each phase (0: frozen) and the phase's last validation losses | `TrainingMixin.phase_summaries_`, W&B `schedule/phase<k>_*` → results `phases` | training |
| Online probes (training curves) | Label probes (clean pass: labels are not inputs), gene probes on masked cells, the gene vector, attention diagnostics per epoch | `online_probes.OnlineProbeCallback` | mixed; attention masked by default (`probe.attention_masked`) |

### Reading them

- The bar is the best reference, or `smoother+position` for probes, never zero.
- Judge the context genes, not the average gene: on synth_spot 160 of 253 genes are noise, which
  dilutes every pooled number. Read `heads_group_by_depth` for `role=receiver`, `role=response`
  (and `role=sender`/`role=source`) in the deep bins. `is_spatial=False` is not pure noise: it holds
  the slide-level condition and batch genes, which the slide mean predicts; `role=noise` is the
  negative control.
- Depth: about 80% of a 500 µm disc's spots lie within 290 µm of a visible spot, so pooled numbers
  mostly describe the rim. Read the deep bins.
- "Beyond 290 µm the GCN is blind" holds in the interior only. On the 100 µm lattice, k-NN 8 edges
  reach 283 µm at slide edges, so two hops can span about 570 µm there. The receptive field shows
  this directly.
- Relative metrics (`*_rel`, `visible_lift`, profile `lift`) read 1 for uniform weights. A matched
  rank of 0.5 is a distance kernel, i.e. blind to the edge.
- Influence magnitudes compare only within one input type: tokens for a GlobalModel, `batch.x`
  for models with a GCN.
- A depth bin holding a single class scores NaN in the label probes.
- With few test slides, collapsed hidden tokens give a participation ratio of at most
  (number of slides − 1).

### The results file

A sweep writes one file, `RESULTS_DIR/results.md` (`interscale.evaluation.report`). Each arm's
analyses are appended as sections when it finishes, `## <arm> | <analysis>`, each with a one-line
note and a compact CSV table (index columns counted in the fence, ```` ```csv index=2 ````); the
untrained twin is `<arm>_untrained`. A later section replaces an earlier one with the same header,
and a missing analysis is simply absent. At the end of a sweep (or `MODE = "evaluate"`) the file
is rewritten with `# Summary` on top: one text table per question, arms as rows, counts and
references once. To read it, start with the summary (`grep -n '^### '` lists its tables) and go
to `## <arm> |` sections for detail; in Python, `read_sections(path)` returns
`{(arm, analysis): DataFrame}`. Per-gene head scores are not written (the per-group table
summarises them).

## Kept, valid, not in the sweeps

| Task | What it answers | Where | Note |
|---|---|---|---|
| Gene-vector probe | How much of a hidden cell's expression a linear readout recovers from each embedding | `masked_probing.probe_expression` | valid; off in A2 (heads are scored directly) |
| Attention diagnostics, clean pass | Span, CLS mass, effective keys with nothing hidden | `attention_diagnostics(masked=False)` | for comparison with rounds 1–3b only |
| Post-hoc probes on an `.h5ad` | Labels from saved embeddings, grouped CV | `linear_probing.py` (library) | labels fine; gene targets would be pass-through; skip its CLS feature set (relevance) |
| Decoder loadings, dim importance, top genes per dim | Which embedding dims each decoder reads | `_gene_loadings`, `_latent_analysis` | descriptive, says nothing about context |
| Gene-set covariance | Variance per embedding dim per gene set | `_gene_set_covariance` | descriptive |

## Not meaningful: don't base conclusions on these

| Task | Why | Where |
|---|---|---|
| Relevance rollout and everything reading it | Its gradient target, the summed output after an affine LayerNorm, is zero while the norm's scale is uniform and arbitrary after: two random scales on the same weights give maps correlating at 0.14. It also runs unmasked | `tl/self_attn_relevance.py`; `get_model_output`'s `_attn_matrix`, `cls_horizontal`/`cls_vertical`; `downstream_regression` (attention regression, relevance AUROC, distance profile); `net_streams`; `flow_control`; `graph_classification` (reads the CLS relevance columns) |
| Gene ranks | Unmasked pass: they measure pass-through. Also R² | `_gene_rank_analysis.calculate_gene_ranks` |
| R² generally | Not what we judge reconstruction by: cosine and SCE | `score_heads` default per-gene columns |
| `expression` probe baseline on hidden spots | Constant (all −1), so it equals the floor | `collect_features(input_baselines=True)`; keep it as a leak check |
| Probes pooled over all hidden spots | Mostly the rim, inside the GCN's reach | any probe without `depth_bins_um` |

Round 1's "relevance flat with distance" and round 2's Exphormer relevance AUROC were relevance
numbers. No conclusion rested on them.

## Older scripts (in `notebooks_and_scripts/`)

- **Rounds 1–3b** (`dense_test*`, `performer_test*`, `exphormer_test*`, `*_legnini`, `pe_test_*`):
  head scores with references (3b), probes pooled over hidden spots, clean-pass attention
  diagnostics and planted pairs, plus relevance analyses, gene ranks and loadings. Only the head
  scores and the raw-attention diagnostics hold up.
- **synth_data_0 era** (legacy): `linear_probing.py` (online probe curves), `track_training.py`
  (shared information between local and global embeddings per epoch), `expression_perturbation.py`
  (gene ranking under expression shuffling: pass-through), `attention_test.py` (net-flow
  permutation tests: relevance), `plot_ranking.py` (gene ranks and loadings).

## Not built

- Content versus position: how much of the variance of log-attention distance explains, per head.
- CLS: a `use_cls` ablation, or a slide-level task. For node-level reconstruction the CLS output is
  never decoded.
- Calibrated magnitude, e.g. ‖pred − mean‖ / ‖true − mean‖ by depth (global_plan 0c, deferred).
