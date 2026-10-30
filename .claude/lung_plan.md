# Lung fibrosis: the SOTA model with niches, compared with the other tools

Written 2026-10-29, at the user's request (point 1 of "Focus until Tuesday 3 November", roadmap.md): run the best model on the lung
fibrosis data, use its niche annotation, and evaluate it against the tools of comparison_plan.md. The data is **not ready yet**
(the user, 2026-10-29). Every data-specific value below is a placeholder (`TBD`) to fill in when it arrives. The code stays
dataset-agnostic (memory: evaluation stays dataset-agnostic): the dataset enters through config and the input file, never by name.

The user's choice for the niches (2026-10-29): **all four roles** -- evaluation labels (L4), the hierarchy's top level (L2), pooled
tokens (L3) and a masking unit (masking_plan.md, M3).

The rules are the other plans': `done` only when a run has been read; "implemented, unverified" is a real state; nothing runs
locally; results here, verdicts appended to notes.md.

## Status

| stage | state | verified by | date |
|---|---|---|---|
| L0 — inputs: the data file, folds, niche columns, checks | proposed; **waits for the data** | | 2026-10-29 |
| L1 — the arms: SOTA model, baselines, masks | proposed | | 2026-10-29 |
| L2 — niches as the hierarchy's top level (`top_level.mode: obs_label`) | **implemented, unverified on data** (2026-10-30, branch `niches`): `hierarchy.top_level.mode: obs_label`, `obs_key`, `split_by_space` none / components / kdtree (below) | tests/test_niches.py (patches, top level, through the model, singleton = dense, config, names), 20 mutations caught, harness IDENTICAL | 2026-10-30 |
| L3 — niche-pooled tokens (`pooling.kind: label`) | **implemented, unverified on data** (2026-10-30, branch `niches`): `pooling.kind: label`, `label_key`, `min_cells`, `max_cells` (below) | tests/test_niches.py (one token per patch, no leakage, a label per cell = the per-cell model, config, names) | 2026-10-30 |
| L4 — readouts with niches as labels | proposed | | 2026-10-29 |
| L5 — the comparison with the other tools (comparison_plan.md C3-C4 on this data) | proposed; the comparison runners are to be written again (below) | | 2026-10-29 |

## The data (to fill in)

| field | value |
|---|---|
| dataset, platform, paper | TBD |
| cluster path | TBD (`MA/...`) |
| samples (the attention sequence) and their key | TBD; one tissue section = one sequence, never two donors in one (the CosMx rule) |
| cells per sample (min / median / max) | TBD: decides dense-fused vs routed (L1) |
| coordinates: `obsm` key, unit, pixel size | TBD; µm in the input file. Check that no two FOVs overlap (the CosMx trap: FOV-local pixels stacked) |
| genes on the panel | TBD |
| cell types: key, coarse and fine | TBD |
| niches: key(s), how they were computed | TBD: from cell-type composition of a neighbourhood, from transcripts, or from a model? This decides the leakage (below) |
| condition (IPF / other ILD / control), donor, region | TBD |
| folds | TBD; proposed: by donor, each test fold holds both conditions (as CosMx) |

## L0 — inputs

- `make_inputs.py` (now `notebooks_and_scripts/slurm/hierarchical/scripts/make_inputs.py`; written for CosMx) generalised by
  config: the coordinate key and unit, the sample key, label and niche keys, the fold rule. Same checks: nearest neighbours 3-30
  µm, no overlapping FOVs, every sample's bounding box plausible.
- `layers['counts']`, `layers['log1p_norm']`, the M2 `var` columns (`scale_set`, from the training slides; masking_plan.md M2), and
  the niche columns as categorical `obs`.
- The LR resource cut to the panel (the comparison's `make_resource.py`, to be written again, below) and its coverage reported first.

## L1 — the arms

**The SOTA model.** On synth_spot the dense windowed global-only model (`lin_dist_skipraw`: GlobalModel, linear token, distance
window per head, raw own-cell skip) is best (0.947 at 300 epochs, 3 seeds). `hier_none` is within 0.01 of it (0.939) and is the
only one shown to run on whole sections (CosMx: 6.5 GiB, ~3.5 s per epoch). A prior from lambda 0 ends where `hier_none` is
(Stage B), so priors are not arms here.

| arm | model | role |
|---|---|---|
| `gcn` | GCN alone (k-NN graph, 2 hops) | the local floor |
| `hier_none` | GlobalModel + hierarchy without prior, routed, the window per head, raw skip | **the SOTA arm** on whole samples |
| `global_window` | `lin_dist_skipraw` on fused attention, dense and exact | the exact reference, only if the fused benchmark passes at this data's sample size |
| `hier_niche` | `hier_none` with the niches as the top level (L2) | does a niche prior help (annotation arm) |
| `pool_niche` | GlobalModel on niche-pooled tokens + raw skip (L3) | the cheap route: a niche token per patch |
| twin of `hier_none` | untrained | the floor of every attribution |

**Masks.** Cell masking with discs wider than the GCN's reach (the CosMx decision of 2026-10-25), plus as second tasks: the M2
global set as a programme (`mask_genes: ["scale_set=global"]`), and the niche mask (M3). Each mask is its own run of `hier_none`;
the other arms take discs only.

**Training length.** CosMx fold 0 at 1,000 epochs was undertrained (every arm below the k-NN smoothers, windows at their start).
The CosMx `long` run (10,000 epochs) was written 2026-10-27; set this data's steps from its loss curve once it is read. With many
samples per fold the steps per epoch are many more than CosMx's 2, so epochs can be fewer for the same steps.

**Scale knobs** (TBD from the data, by the CosMx rules): k = 8 neighbours; discs ~4 cell spacings wide over 10% of cells plus 20%
random cells; windows from the cell spacing to ~1 mm; hierarchy regions ~250 cells, clusters of 25.

## L2 — niches as the hierarchy's top level

hierarchical_attention_plan.md T1 already specifies it: `top_level.mode: obs_label` with `hier.top_level.obs_key`, finer levels
built within each label by `kdtree`, and `split_by_space: components` (two patches of one niche are two clusters). Neither is built.
To build: the label's components on the spatial graph as the top-level clusters (with `min_cells` merging), kdtree inside each, the
cache keyed by positions **and** labels, the validator (the key must exist, categorical), tests (a component never spans two labels;
the singleton regression still equals dense).

**Leakage (decide with the user once the data arrives).** The hierarchy's invariant is "positions alone build it; no hidden value
decides a cluster". A niche label computed from every cell's expression breaks it: a hidden cell's cluster then says something
about what it expresses. The hierarchical plan's line holds: treat `obs_label` as an annotation arm, an upper bound ("does a niche
prior help?"), never the default. The fair variant, if it matters: niches recomputed from what the model may see (k-means on the
cell-type composition of each cell's neighbours, fitted on training slides, the cell itself left out).

**Built 2026-10-30 (branch `niches`).** The patches are `tl/niches.py`, shared with L3 and M3: the connected components of the
label on the slide's own graph (`edge_index`, the edges the GCN reads) after every edge between two labels is cut; patches under
`min_cells` merge into the neighbour they share the most edges with (then the larger, then the lower id; one without an edge to
another into the nearest centroid), so a merged patch can hold two labels; patches over `max_cells` are cut by the kdtree. A cell
with no label (NaN) is a label of its own (-1). The label reaches the graphs through `dataset.extra_obs_keys` (one-hot, read back as
codes); `geome_utils.label_attribute` resolves the column. Config: `hierarchy.top_level.mode: obs_label`, `obs_key: <niche column>`,
`split_by_space`: `none` (one cluster per label), `components` (one per patch, `min_cells` merging), `kdtree` (patches cut to the
size of the first level, `levels[0]` / `cells_per_cluster[0]`, unused otherwise). Finer levels as before, inside each top cluster.
The hierarchy cache keys by labels and edges too. Checkpoint tag `-ob-<key>-<all|cc|kd>`.

**Routed models need `kdtree` (found 2026-10-30).** Routing gives every parent `cap_per_query // (the widest parent's children)`
candidate parents (`hierarchical_attention._candidates`): one niche patch of 20,000 cells with clusters of 25 below it makes the widest
parent 800 children wide, so with the cap at 512 every query of the slide reads its own parent alone. `split_by_space: kdtree` keeps the
top level balanced (patches cut to ~250 cells), `min_cells` keeps noisy labels from making thousands of one-cell clusters (level 0 is
dense). The layer now warns when the cap cuts the chosen parents. **The `hier_niche` arm:** `hier_none` with
`top_level: {mode: obs_label, obs_key: <niche>, split_by_space: kdtree, min_cells: ~50}` and the same `cells_per_cluster`.

## L3 — niche-pooled tokens

**Built 2026-10-30 (branch `niches`):** `pooling.kind: label`, `label_key`, `min_cells` (1: none), `max_cells` (0: none), on the
patches of `tl/niches.py` (L2); no `size_um`, `n_patches` or `offset`. The raw own-cell skip is config
(`model.decoder.own_cell_skip: true`, `own_cell_skip_input: raw`), not forced. The patches are built every step (no cache):
for the user's coming small-niche algorithm only `label_key` changes; `max_seq_len` bounds the patches per slide, `min_cells` lowers
them. Checkpoint tag `pool-label-<key>[-mc<min>][-mx<max>]`. The plan as written:

T1b's pooling (`pooling.kind: grid`) with a new assignment: a patch is a connected component of one niche label, split by kdtree
into pieces of at most `max_cells` (so one large niche does not become one token). Its invariants stay: a patch token holds visible
cells only, a patch with none gets the learned token, `max_seq_len` bounds patches. The raw own-cell skip is required: pooled
tokens without it lose the sparse responses (0.00-0.03 on synth_spot). The same leakage note as L2 applies. `ASSIGNMENTS` in
`pooling.py` gets `label`. Tests as for `grid`, plus: components never cross labels; hidden cells weigh 0.

## L4 — readouts with niches as labels

Per arm and fold, into comparison_plan.md's tables (niche as one more label column):
- **Reconstruction:** hidden-cell centred cosine and SCE by niche, against the k-NN smoothers and the section mean; M0's hidden vs
  visible table.
- **Niche classification:** a logistic probe from each embedding (local, global, both; pooled: the patch token) to the niche,
  against expression alone and the smoother features; per-niche F1. Also cell type (coarse, fine), as on CosMx.
- **Interactions between niches:** integrated-gradients attribution of a receiver niche's predicted expression, summed over the
  senders of each niche (receivers sampled per niche and tile, IG is linear in its target), against the expression-permuted section
  and a distance-matched null; its distance profile. The same by cell type. **The design of this readout and its plots is the
  separate session's (interaction_readout_plan.md: points 7-9)**; this plan only says that the lung run must feed it.
- **Gene scales:** M2's local and global sets of this data, per niche (which niches carry the global genes); the global part's
  share where the model is decomposable.
- **Condition:** every table split by condition (IPF vs control), the differences with their fold spread (T7's first look).

Positive controls (draft from the IPF literature, **to check against the panel and with the user**): TGFB1 → TGFBR1 / TGFBR2 into
fibroblasts; PDGFB → PDGFRB; SPP1 (profibrotic macrophages) → CD44 / integrins; CXCL12 → CXCR4; aberrant basaloid cells (KRT17+,
KRT5-) next to fibroblastic foci (CTHRC1+ / COL1A1+ fibroblasts); each as (sender, ligand, receptor, receiver, condition).

## L5 — the comparison with the other tools

comparison_plan.md C0-C4 on this data, unchanged in design: the same output tables, baselines LIANA+ (rank aggregate and spatial
bivariate), squidpy `ligrec`, AMICI, Steamboat, NicheCompass (learned CCC), Moran's I (SVG, also within types and within niches),
logistic regression (cell type and niche). New here: every CCC table also by niche pair; SVG also within niches; classification of
niches.

**2026-10-29:** the user removed the comparison scripts of 2026-10-25 (`notebooks_and_scripts/slurm/comparison/`: they did not like
the code); only `make_inputs.py` remains (in `hierarchical/scripts/`). The runners are to be written again from comparison_plan.md's
design (the tables' schemas are in C0), in the fork (`scripts/comparison/`), after agreeing with the user how. AMICI, Steamboat and
NicheCompass runners also need the user's environment names.

## Order (until Tuesday 2026-11-03)

1. Before the data: build L2 and L3 (code and tests on synth_spot with a planted region label as the "niche"), M3's mask, M0; the comparison
   runners written again (agreed with the user first); write the sweep with placeholders.
2. Data day: L0 (inputs, checks, M2 sets, the niche maps), a one-sample feasibility run of `hier_none` and `global_window` (time,
   memory per sample).
3. Then the sweep: fold 0 first, all arms, one job per run; the baselines in parallel.
4. Read: reconstruction and classification first, then the interaction tables against the nulls and controls.

## Open decisions (for the user)

1. The data fields above.
2. Whether the given niche labels are acceptable as hierarchy level / pooling unit / mask (annotation arms), or should be
   recomputed from training-visible data (L2).
3. Which tissue samples form a fold, and whether conditions are balanced per fold.
4. The positive-control list.
