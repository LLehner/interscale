# synth_data_0 — what is in the synthetic dataset

Written by `synthetic_data.py`. This file describes, in words, what that script builds and why,
so the dataset can be read without reading the code.

```bash
python src/interscale/evaluation/synthetic_data.py --out data/synth_data_0.h5ad
```

The point of the dataset is architecture comparison. Every pattern the model is meant to find is
also written into the object as ground truth, so a change to the transformer can be scored instead
of eyeballed.

## The design

Two conditions, **healthy** and **diseased**. Six donors per condition, two slides per donor, so
**24 slides of 1500 cells each — 36,000 cells in total**. Each slide is a 1000 × 1000 unit square
with cells scattered across it.

Condition is a property of the *donor*, not of the slide. The two slides of a donor are technical
replicates of each other; different donors of the same condition are biological replicates.

**One slide is one graph and one transformer sequence. There are no sliding windows.** This is the
main simplification compared to the real datasets, and it means
`cfg.model.global_component.parameters.max_seq_len` has to be at least 1500. If it is smaller,
`pad_batch` keeps a random subset of the cells as tokens and the attention analysis can no longer
tell which token was which cell.

### Train / validation / test

The split is assigned **per donor**: per condition, 4 donors go to train, 1 to validation, 1 to
test — 8 / 2 / 2 donors, or 16 / 4 / 4 slides.

It has to be per donor for two reasons. The pipeline subsets the object by split *before* it builds
the neighbour graph, so a cell-level split would chop every graph into three partial graphs. And
holding out whole donors is what makes "predict the condition" a real question rather than a
question about which slide a cell came from.

## What is in a slide

**Three niches** — `cortex`, `medulla` and `stroma_rich` — are laid out per slide as three soft,
noisy territories, so their borders interdigitate instead of being clean polygons.

**Six cell types**, drawn per cell from its niche's composition:

| cell type | role |
|---|---|
| `senderA`, `receiverA` | the short-range interaction pair |
| `senderB`, `receiverB` | the mid-range interaction pair |
| `hub` | a single compact cluster of 15 cells per slide; the source of the tissue-scale program |
| `stroma` | filler, involved in nothing |

Niche and cell type are deliberately entangled: which cell types you find somewhere depends on the
niche. That is a confounder on purpose — the downstream regression has to separate "these two cells
attend to each other" from "these two cell types just happen to sit in the same neighbourhood".

There is also a **negative control**: in the `medulla` niche, `senderB` and `receiverA` sit right
next to each other and never interact. Any method that reports an interaction there is picking up
co-occurrence, not signalling.

Cell positions are not perfectly uniform. Each slide gets its own smooth density modulation, which
feeds through into neighbour counts and QC statistics — another nuisance the regression has to
account for.

## The 40 genes

All counts come from one zero-inflated negative binomial model. Every effect below is an additive
term in the log of the ZINB mean, so effect sizes are comparable across programs, and
`--effect-scale` scales all of them at once to find where an architecture stops detecting things.

- **24 noise genes** (`noise_00` … `noise_23`) — no structure at all. They are the majority of the
  panel on purpose: a method that highlights them is reporting noise.
- **6 cell-type markers** (`mark_senderA` …) — one per cell type. These make cell-type
  classification solvable.
- **4 gradient genes** — expression rises or falls with distance from the slide centre.
  `grad_up_broad` and `grad_down_broad` change gradually across the whole slide;
  `grad_center_sharp` and `grad_edge_sharp` turn over on a much shorter length scale (150 and 300
  units). Together they give a spread of spatial length scales to recover.
- **3 interaction genes** — the ones that matter:
  - `int_short` — expressed by `receiverA` cells, proportional to how many `senderA` cells sit
    **within 30 units**. Build the neighbour graph with `spatial_neigbors_kwargs.radius: 30` and
    this sits inside the local GCN's two-hop reach, so the local component should be able to
    explain it. At 1500 cells per slide that radius gives each cell about 4 neighbours.
  - `int_mid` — expressed by `receiverB` cells, driven by a Gaussian kernel of `senderB` density
    with a **width of 150 units**. That is beyond the local component's reach but well inside a
    slide, so it is the transformer's job.
  - `int_long` — expressed by every cell, falling off with distance from the slide's hub cluster on
    a **400-unit scale**, and **present only in diseased slides**. It needs long-range context
    *and* the condition; geometry alone will not produce it. The rule is identical on every
    diseased slide, so it is also a test of whether patterns transfer across slides.
- **3 condition genes** (`cond_up_1`, `cond_up_2`, `cond_down_1`) — simply shifted in diseased
  slides, with no spatial structure. They make condition classification solvable without any
  interaction being found, which is what makes them a useful control for the CLS-token analysis.

Noise genes drop out more often (5–35% of entries) than program genes (2–10%): a signal that is
zero-inflated away is not a detectable pattern.

## Nuisance variation

Library size varies per cell (lognormal) and, on top of that, per slide — that slide factor is the
batch effect the attention regression is asked to remove. Combined with the density modulation and
the niche/cell-type entanglement, there is enough structure that "explains attention" is a real
competition rather than a formality.

## What ships with the object

| where | what |
|---|---|
| `X`, `layers['counts']` | raw ZINB counts |
| `layers['log1p_norm']` | median-normalised, log1p — the layer to train on |
| `obsm['spatial']` | coordinates |
| `obs` | `condition`, `donor`, `slide`, `split`, `niche`, `cell_type` |
| `obs` (exposures) | `dist_to_center`, `dist_to_hub`, `hub_response`, `n_senderA_short`, `kern_senderB_mid` — the exact quantities that drove each interaction gene |
| `obs` (QC) | `total_counts`, `n_genes_by_counts`, `lib_factor` |
| `var` | `program`, `effect_size`, `true_length_scale`, `target_cell_type`, `is_spatial` |
| `uns['synthetic']['interaction_edges']` | every pair of cells that actually influenced each other, with weight and range class |
| `uns['synthetic']['interactions']` | the four programs in one small table, negative control included |
| `uns['synthetic']['params']` | every parameter the run used |

The edge list is the important one. Cells are referred to by **position** in the object, which
survives the `obs_names` renaming that `prepare_geome_dataset` does. At the default size it holds
roughly 5,000 short-range, 142,000 mid-range and 85,000 long-range pairs.

## Roughly what to expect

At default settings, on the training layer: cell-type markers separate their types cleanly, the
gradient genes correlate with distance-from-centre at about ρ ≈ 0.4, `int_short` correlates with
the local `senderA` count at ρ ≈ 0.37 in `receiverA` cells and at ≈ 0 everywhere else, `int_mid` at
ρ ≈ 0.51 in `receiverB` cells, and `int_long` at ρ ≈ 0.27 with the hub response in diseased slides
and ≈ 0 in healthy ones. Noise genes correlate with nothing. Median library size is about 235
counts per cell.

The long-range program is the weakest of the three on purpose — it is the one a better architecture
should be able to win on.

## Using it with the downstream tasks

`downstream_classification.py` probes how well an `.obs` label can be read out of the embeddings.
The informative comparisons here are `cell_type` and `niche` at node level — local embedding versus
global embedding versus raw expression — and `condition` at graph level, which should live in the
global embedding and the CLS token rather than the local one.

`downstream_regression.py` asks how much of the attention is left once distance, cell-type pair
identity, niche, counts and batch have been accounted for, and then ranks that residual against
`interaction_edges`. A residual AUC near 0.5 means the attention was a distance kernel and nothing
more. The three range classes are scored separately, which is the number to watch when comparing
transformer variants: short-range should be easy, mid-range is the honest test, long-range is hard.

## Knobs worth turning

- `--effect-scale` — shrink every effect to find an architecture's detection floor.
- `--n-cells-per-slide` — trades detection (denser sampling of each interaction kernel) against
  sequence length.
- `--n-donors-per-condition` — more donors make the held-out condition question harder and the
  grouped cross-validation more stable.
