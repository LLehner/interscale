# Yardsticks for a gene's scale: model-free (Y) and model readouts (V) (plan, 2026-11-02)

Asked 2026-11-02: "start implementing the model-free yardsticks ... the point about residualising labels sounds important so
we should test that today." The sources and reasons are in [gene_ranking_literature.md](gene_ranking_literature.md),
section 1 (S1-S5) and P1. The Y stages read data only (no model). Whole-dataset runs go to the cluster (the user,
2026-11-02); locally only tests on toy arrays and toy modules.

Asked 2026-11-02, second request: "continue with the unique and shared R² per gene, the per-gene test for global-only
models and per-gene split for one-head model. The training changes ... we keep in mind when the first test results are in."
These are literature notes M2, M3, M5: the V stages below. M6-M8 (scale-blocked embedding, asymmetric penalty, branch
balancing) wait for the V results.

## Status

| stage | what | state | date |
|---|---|---|---|
| Y0 | label residuals: residualise expression on labels / neighbourhood composition before the scale sets; order-free R² shares | **built 2026-11-02** (`evaluation/residuals.py`, 17 tests, 20/20 mutations caught); first read in notes.md 2026-11-02 (synth_spot, CosMx fold 0, vannan25 8 samples / `final_CT` only); **cluster run written, not run** (`slurm/new/label_residuals.{py,sbatch}`: synth_spot, CosMx folds 0-2, vannan25 all samples) | 2026-11-02 |
| Y1 | scale bands: each gene's variance share in fine / medium / broad bands of the slide graph's Laplacian eigenvectors (MEM / SpaGFT); the same for model parts | proposed | |
| Y2 | big-small patch ratio (BSP) at the GCN's reach vs a window | proposed | |
| Y3 | predictive views (MISTy-style): per gene, R² gained by the near neighbourhood's and the far context's expression over the cell's own other genes, cross-fitted | proposed; reuses Y0's neighbourhood blocks with expression in place of labels | |
| Y4 | structure-preserving null for map-to-map comparisons (Moran spectral randomisation) | proposed | |
| Y5 | external per-gene scale rankers (nnSVG, Spacelink; R) as references for `ell_um` and the sets | proposed (cluster, R) | |
| V1 | unique / shared R² per gene: cross-fitted ridge from frozen features (own visible genes, the visible neighbours within the reach, the local embedding, the global tokens) to the hidden truth; unique-local, unique-global, shared; Shapley over more blocks | **built 2026-11-02** (`evaluation/partition.py`, 13 tests; 16 mutations, 15 caught, 1 equivalent); in the hook `readout_common.scale_readouts` (`part_*`); **not run** | 2026-11-02 |
| V2 | per-gene context test: the context beyond (or within) r of sparse receivers replaced by a shifted copy of the section; Δloss per gene, rank p-value, BH; label none / local / global / both | **built 2026-11-02** (`evaluation/context_test.py`, 10 tests; 15 mutations caught); `ctx_*` in the hook; **not run** | 2026-11-02 |
| V3 | per-gene split of one prediction: SVD components of the prediction, each component map's length scale; a gene's prediction variance in components wider than the reach (global), narrower (local), unstructured | **built 2026-11-02** (`evaluation/component_scales.py` with `collect_fold_predictions`, 4 tests; 7 mutations caught); `comp_*` in the hook; **not run** | 2026-11-02 |

## Y0 — label residuals

**Question.** Moran's I of a cell-type marker is mostly the cell type's own clustering: on real tissue a "global" gene can be
a region or cell-type marker, which is not communication (ctSVG literature: C-SIDE, Celina, Spacelink's cell-type form). How
much of each gene's local and global pattern is left once each cell's label is taken out, and which genes change set?

**Blocks** (the extension point). A block is a design matrix over the cells: `label_block` (one-hot of a label: the cell's
own type, or a region / niche label), `composition_block` (the share of each label among a cell's neighbours: any 0/1
neighbourhood matrix, the GCN's own hop neighbourhood or `knn_neighbourhood`). Y3 adds expression blocks over the same
neighbourhoods.

**Fits.** Every fit has one intercept per graph (slide): stage `graph` is each gene centred within its graph, and every later
stage adds one block to the design (cumulative). `fit: graph` fits the coefficients per graph (a label's mean per slide:
donor-specific type means), `pooled` once over all cells with graph intercepts. Least squares (`lstsq`, minimum norm: one-hot
columns are collinear with the intercept). R² per gene against the within-graph variance, adjusted for the parameters, and an
**order-free split** (`shapley_r2`: Shapley over the blocks, the LMG decomposition; 2^B fits) so the share of a block does not
depend on where it was put in the sequence.

**Composition removes interaction too.** A gene that a cell expresses *because of* the types around it (NCEM's niche effect)
is explained by composition, and that is the signal we look for. So the composition stages are read as a decomposition, not
as the residual the scale sets are rebuilt on: the scale sets after the **own label** (and, separately, after a **region /
niche** label) are the yardstick. On CosMx and Xenium near-composition also catches segmentation spillover (a neighbour's
transcripts assigned to the cell), which looks like a very local gene.

**Readout.** `residual_scale_sets` runs `gene_scales.scale_sets` (I_local at the GCN's reach, I_global across the slide,
the relative fall, the two sets) on the residuals of each chosen stage and lines them up per gene; `label_summary` gives per gene
`I_local` and `I_global` before and after, their kept share, the set before and after, and the label's R². Cross-tables of set
membership before vs after.

**Run** (`notebooks_and_scripts/slurm/new/label_residuals.py`): k-NN 8 graph, 2 hops (the GCN's reach, as in the runs),
training slides only, `log1p_norm` of counts at 1e4.
- synth_spot: label `region` (CM1 / CM2). Positive control: the region markers (CM1-CM3) lose their pattern; planted
  programmes that do not follow the region keep theirs.
- CosMx (fold 0 training slide): `cell_type_coarse`, then `CellTypes_max`; composition at the reach for the decomposition.
- vannan25 (lung): `final_CT`, `final_lineage`, and the niche `TNiche` as a region label.

**Tests** (`tests/test_residuals.py`): exact OLS against statsmodels-free closed forms (a label's residual is the value minus
its per-graph label mean), graphs never mixed, pooled vs per-graph coefficients, R² of a planted block, Shapley shares adding
to the full R² and order-free, a perfect marker becoming constant (Moran's I NaN, not noise), composition rows summing to 1,
neighbourhoods never crossing graphs, no global RNG touched.

## V1 — unique and shared R² per gene (`evaluation/partition.py`)

**Question.** What does the global embedding know about each gene beyond what the cell and its neighbourhood show -- and, for a
dual model, beyond the local embedding? `attribution_shares` (cov_share) splits a prediction's shared variance by how training
went; variation partitioning does not.

**Features** (`collect_feature_blocks`, a masked pass; with `folds` every cell hidden once, as `collect_dimensions`): per kept row
the truth, the hidden entries, graph, position and the blocks
- `own` -- the cell's input with its hidden entries at 0 (under cell masking a hidden cell's row is all 0: no own information);
- `near` -- per gene the mean of the *visible* entries of the cells within `hops` hops of the batch's graph (the local
  component's reach; for a global-only model the GCN reach of the runs, a choice), the graph mean where none;
- `local` -- the local embedding (combined / dual / local models), gathered by `padded_node_idx`;
- `global` -- the global tokens (`gather_tokens`; the context query's read-out under D1.2).
Model-free blocks need no model: `own` + `near` is the local information any model could have.

**Fit** (`partition_r2`). Targets are the genes hidden in every kept row (cell folds: all; programme masking: the programme);
anything else is refused. Target and features centred within each graph (within-section R², as `within_section`); ridge
(`RidgeCV`, an alpha per gene, features standardised on the training rows) for every subset of the named groups of blocks,
cross-fitted over folds by graph when there are enough graphs, else spatial blocks (`spatial_folds`, `block_um`). Out-of-fold
R² per gene. Two groups: `r2_<A>`, `r2_<B>`, `r2_both`, `unique_<A>` = R²(both) − R²(B), `unique_<B>`, `shared` = R²(A) + R²(B)
− R²(both). More groups: Shapley shares (the helper `shapley_from_subsets`, shared with Y0's `shapley_r2`).

**Read.** Genes ranked by `unique_global` with L = own + near: the model's context knowledge per gene beyond the visible local
neighbourhood -- a gene ranking that works for global-only models. Dual: L = local embedding, G = global embedding.
Against the untrained twin (a twin's unique_global is what the architecture gets for free).

## V2 — per-gene context test (`evaluation/context_test.py`)

**Question.** Is gene g's prediction at a hidden cell better with the true context beyond r than with a context of the same
tissue that has no relation to the cell? A conditional randomisation test (HRT, Tansey et al. 2022) with the model as a black box.

**Surrogate** (`shift_sources`). Per graph a random translation δ (at least `min_shift` of the graph's extent) with wrap-around in
the bounding box; each cell takes the *whole row* of the cell nearest to its shifted position: `x`, `mask`, `gene_mask` (the
`fields`), so hidden entries stay hidden wherever they go and a receiver's own hidden values never appear elsewhere. Positions,
edges and positional encodings stay. Composition and autocorrelation are kept, the link to the receivers is broken.

**Receivers** (`choose_receivers`). Cells with a hidden entry, in batches with pairwise distance > 2r and the union of their
r-zones at most `max_cover` of the graph (default 0.1), so for the `far` test at most ~10% of a receiver's far context stays true
(other receivers' zones: lowers power, keeps validity). `far`: the zones (receiver + cells within r) keep their true rows, every
other cell takes its surrogate row. `near`: the cells within r of a receiver (not the receiver) take surrogate rows.

**Statistic.** One true pass (the unmodified graphs) and, per receiver batch, `n_shifts` surrogate passes. Per gene the mean
squared error over the receivers' hidden entries: `loss_true`, `loss_k` for shift k (all batches pooled). `p = (1 + #{loss_k <=
loss_true}) / (n_shifts + 1)`, BH over genes; effect `delta_r2 = (mean_k loss_k − loss_true) / var_g` (the gene's variance over
the scored entries). `context_labels(far, near)`: global / local / both / none at q and a minimum effect.

**Caveats.** Shifts are liberal at a section's edge (nearest-cell mapping repeats edge cells); a model with `Precomputed` tokens
or expression-derived per-cell attributes needs them in `fields` (refused otherwise). Cost `1 + n_batches * n_shifts` passes.

## V3 — per-gene split of one prediction (`evaluation/component_scales.py`)

**Question.** A global-only model has one head and no parts. Which genes does it predict with patterns wider than the local
reach? NSFH's spatial importance score (a gene's loading share on spatial factors), MEFISTO's per-factor smoothness.

**Components.** Over the kept rows of a fold pass (every cell once, so the maps cover the sections), each gene centred within
its graph; SVD `P = C S U^T` (as `prediction_readout`), the first `n_components`. Each component's cell map `C_k S_k` gets
Moran's I curves and a length scale (`gene_scales.moran_curves`, `permutation_null`, `length_scales`, components as "genes").
A component is **global** when it is `variable` and `ell_um` > the reach (the median distance to the farthest cell within `hops`
hops), **local** when `variable` and within it, **unstructured** otherwise. A gene's shares: `sum_k S_k² U_gk² [class k] / sum_k
S_k² U_gk²` over all components, and `ell_um` = the loading-weighted median of the structured components' scales (`inf` allowed).
Any prediction: a model head, a reference (knn smoother), the truth (the data's own split, the comparison).

## V1-V3 as built (2026-11-02) -- deviations from the design above and what was found while building

- **V2, the near test takes one receiver per graph and pass** (`per_batch`, `"auto"`). With several, receiver j's replaced zone is
  far context for receiver i, so a gene that reads only far context shows a near effect: a false positive, not lost power (a test
  pins it). The far test keeps several receivers per pass (the leak keeps *true* context: less power, still valid). Cost of the near
  test: `near_batches x n_shifts` passes with as many receivers as graphs per pass.
- **V2, p-values.** The exact rank p cannot go below `1 / (n_shifts + 1)`, which BH over hundreds of genes never passes (979 CosMx genes
  at q 0.05 would need ~20,000 shifts). `p` / `q` come from `z = (mean_k loss_k - loss_true) / sd_k` (surrogate losses taken as normal,
  each a mean over many receivers); `p_rank` stays as the exact check. Ties (a surrogate that does not reach the gene) are `p = 1`,
  with a relative tolerance for float sums in another order (`TIE_TOLERANCE`).
- **V2, the shift** wraps on a torus (`cKDTree(boxsize=...)`): without it a point shifted into the gap past the last row maps to an
  edge cell, not the first row, and the shift is no bijection on a lattice.
- **V2, the normaliser** of `delta_r2` is each graph's variance of the gene over all its cells, weighted by the graph's scored
  entries (over the receivers alone it is 0 with one receiver per graph).
- **V2, fields.** Per-cell graph attributes are either moved with the row (`fields`: x, the masks, labels, embeddings) or declared
  positional (`POSITIONAL`, `stay`); any other refuses the test.
- **V3, the gene's scale** is the loading-weighted *median* of its components' scales: a wide component often never halves within
  the sizes (`ell_um = inf`), which a mean drops or is swamped by.
- **V1** predicts the training mean when a block has nothing that varies (own under cell masking); single-target predictions reshaped.

**Cluster files** (not run): `slurm/new/readout_common.py` (the hook `scale_readouts(settings)` and `combine`), `synth_spot_readouts.{py,sbatch}`
(Stage A checkpoints of the hierarchical sweep evaluated again, readouts only, programme mask, near test on; 24 runs + twins of
full and hier_none), `cosmx_tiers.py` (`SCALE_READOUTS`, near test off; re-evaluate gcn / hier_none / hier_tier2 with
`--array=0-2,6-8,15-17 --export=ALL,INTERSCALE_MODE=evaluate`).

**Read first (synth_spot):** Spearman of `v1_unique_global` and `v2far_delta_r2` with the planted `length_scale_um` over the programme
genes; long-range responses high, short-range receivers low, the twins as the floor; `v2_label` against the planted role;
`v3_global_share_global` against `v3_truth_share_global`.

**2026-11-02, later:** `synth_spot_readouts.sbatch` and `label_residuals.sbatch` are single jobs now (every run / dataset in turn, no
arrays; the user asked for single runs with multiple arms). The CosMx re-evaluation with the readouts: one job,
`INTERSCALE_MODE=evaluate INTERSCALE_RUNS=gcn_f0,hier_none_f0,hier_tier2_f0 python scripts/cosmx_tiers.py`.
