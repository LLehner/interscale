# Ranking genes: effect sizes, sender deletion, stability, remove-and-retrain (plan, 2026-11-02)

Asked 2026-11-02: "continue implementing part 2 ranking genes in general ... rank by effect size, not p-value, sender genes by in
silico-deletion (This one is particularly interesting!), stability across seeds and folds (important! In particular when it comes to
seeds). And then we validate rankings via remove and retrain on synth as well. Overwrite the previous scripts and yamls for the lung
data if necessary. I want two runs: lung and synth." Sources and reasons: [gene_ranking_literature.md](gene_ranking_literature.md),
section 4 (R1-R5). Builds on [yardsticks_plan.md](yardsticks_plan.md) (V1-V3, whose effect sizes it ranks).

## Status

| stage | what | state | date |
|---|---|---|---|
| R1 | rank by effect size: robust z within expression-matched bins, a control floor (NegPrb), ranks | **built 2026-11-02** (`evaluation/ranking.py` `rank_by_effect`; tests/test_ranking.py) | 2026-11-02 |
| R2 | sender genes by in-silico deletion: conditional replacement (all genes' conditionals from one precision matrix per graph), regions `all` / `context` / `far`, a sender x target ΔR² matrix with per-receiver z | **built 2026-11-02** (`evaluation/deletion.py`; tests/test_deletion.py, 7 tests on a module of known dependence) | 2026-11-02 |
| R3 | remove and retrain: `dataset.drop_genes` (genes removed at load, every entry point, checkpoint tag) and the synth stage-2 arms | **built 2026-11-02** (`tl.drop_genes`, main.py, main_sweep.py, slurm/new/sweep_common.load_adata; tag `drop<n>-<sha1>`); stage-2 arms from `make_synth_ranking_yaml.py --roar` | 2026-11-02 |
| R4 | stability: RBO, Spearman, Jaccard@k between runs, top-k frequency, robust rank aggregation (Kolde 2012) | **built 2026-11-02** (`evaluation/ranking.py`; `slurm/new/ranking_stability.py` over a sweep's runs) | 2026-11-02 |
| runs | `synth_ranking` (stage 1: `full` x 3 seeds + twin; stability; stage 2: ROAR arms) and `lung_ranking` (vannan25: `hier_none` x 3 seeds x 2 folds + twin) | **written 2026-11-02, not run** (`submit_synth_ranking.sh`, `submit_lung_ranking.sh`); checked: configure on both scripts and stages, the hooks and the stability step on toy inputs, the lung input builder on a stand-in | 2026-11-02 |

## R1 -- effect sizes, matched floors (`evaluation/ranking.py`)

A p-value ranks by sample size as much as by effect. Every ranking here is by an effect size (ΔR² of a deletion or of the context
test, unique R²), and two adjustments say whether a gene's effect is more than its expression level buys:
- **expression-matched z**: genes binned by mean expression (quantile bins); a gene's effect minus its bin's median over the bin's
  MAD (1.4826 MAD: an SD for normal data). Spatially variable / predictable genes are the abundant ones; within a bin they compete
  with genes of their own level.
- **control floor**: named control genes (CosMx NegPrb); the floor is a quantile of their effects, a gene ranks only above it.

## R2 -- sender genes by in-silico deletion (`evaluation/deletion.py`)

**Question.** Which input genes does the model read from *other* cells to predict each hidden gene? Geneformer ranks genes by the
shift a deletion causes; the holdout randomisation test (Tansey et al. 2022) says to replace a feature by a draw from its
conditional given the rest, not by zero (off the data's manifold).

**Replacement** (`conditional_draws`). Per graph, over its visible cells: the regularised covariance of the genes and its inverse Θ;
the conditional mean of gene j given the cell's other genes is `x_j - (X Θ)_j / Θ_jj`, for every j from one inverse. A draw is that
mean plus the residuals permuted among the visible cells. Also `permute` (the gene's own values permuted: breaks its link to the
other genes too) and `mean`.

**Regions.** `all`: the gene replaced in every cell, receivers included (own and context). `context`: in every cell but the receivers
(sparse receivers, `max_cover` of a graph; the other receivers' rows stay true: power, not validity). `far`: only beyond `radius` of a
receiver (the V2 zones). Under cell masking a receiver is hidden whole, so `all` = `context` and every hidden cell is a receiver.

**Statistic.** One true pass; per sender and draw one pass. Per (sender j, target g) the per-receiver difference of squared errors on
the hidden entries, summed over receivers and draws: ΔR² (over the graph-weighted variance of g, as V2), its z from the receivers'
spread. Per sender: `importance` (ΔR² summed over targets, negatives at 0), `n_targets` (ΔR² > `min_effect` and z > 3). Senders are
groups of genes too (deleted together). Cost: senders x draws passes.

## R3 -- remove and retrain

`dataset.drop_genes` (selectors as `mask_genes`; default empty: nothing changes): `tl.drop_genes(adata, cfg)` removes the genes from
the AnnData right after it is read (main.py, main_sweep, the sweeps' `load_adata`), so neither input nor target holds them; refused
when it would drop a gene of `mask_genes` (a target). Checkpoint tag `drop<n>-<sha1>`.

**Synth stage 2** (ROAR, Hooker et al. 2019): the consensus sender ranking of stage 1 (RRA over seeds of `importance`, region
`context`) gives the top-k; arms `top<k>`, `rand<k>` (3 draws of k from the senders that are not top-k), `bottom<k>`, `planted` (the
planted senders/sources: the bar), and `none` (stage 1). Retrained at the same epochs and seeds; read: the programme score drop
against `rand<k>`. A ranking is worth something when `top<k>` drops much more than `rand<k>` and `bottom<k>` drops nothing.

## R4 -- stability (`evaluation/ranking.py`)

`rbo` (Webber et al. 2010, extrapolated, `p` 0.9), `rank_stability` (pairwise RBO, Spearman, Jaccard of the top-k for every pair of
runs, and per gene the share of runs that put it in the top-k), `robust_rank_aggregation` (Kolde et al. 2012: each gene's normalised
ranks over runs, the minimum over k of the Beta(k, m - k + 1) CDF at the k-th smallest, Bonferroni over m). Seeds first (the same
fold, different initialisations: what the model learns vs what it happens on), then folds (different donors / slides).

## Runs

- **synth_ranking** (`slurm/new/synth_ranking.{py,sbatch}`, `make_synth_ranking_yaml.py`, `ranking_common.py`, `ranking_stability.py`,
  `submit_synth_ranking.sh`): stage 1 trains `full` (dense windowed global-only, the SOTA) at seeds 42-44 under programme masking (300
  epochs) + its twin; the evaluation: deletion (every visible gene a sender, regions `context` and `all`, 3 draws), scored against the
  planted edges (`interaction` column: precision@1, mean reciprocal rank of the planted sender per target; twin as the floor), R1
  tables, V1-V3. `ranking_stability.py` then aggregates the seeds and writes the stage-2 yaml; stage 2 trains the ROAR arms.
- **lung_ranking** (`make_lung_inputs.py`, `make_lung_ranking_yaml.py`, `lung_hier.yaml`, `lung_ranking.{py,sbatch}`): vannan25 into
  inputs (µm, sample = sequence, 5 patient groups stratified by disease, folds rotate test / val), `hier_none` at seeds 42-44 on
  folds 0 and 1 + twin; evaluation under a fold mask (one fifth of cells hidden): deletion (every gene, `all`, 2 draws, on
  `MAX_TEST_GRAPHS` samples), V1-V3, R1; `ranking_stability.py` over seeds and folds.

## As built (2026-11-02)

Library: 13 tests (tests/test_ranking.py, tests/test_deletion.py), 31 mutations all caught (9 only after stronger tests: the matched z,
Jaccard, the top-k share, RRA's Bonferroni, the residual draw, `context` ignoring the radius, averaging over draws, importance leaving
out negative effects, `n_targets` needing z); full suite 1723 passed.

- **Conditional replacement is what makes a deletion mean "unique information"** (test: a near copy of sender 3 in the same cell makes
  the conditional deletion of 3 almost harmless, the permutation not; deleting the pair as a group restores the effect). Read a
  sender's ΔR² as what the model loses when that gene's information beyond the cell's other genes is gone. Correlated genes (one
  pathway, one cell type's markers) can then share an effect no single one of them shows: delete them as groups (`senders` as a dict).
- **`context` vs `all`:** on the programme mask every cell is a receiver; `context` keeps each receiver's own row (an own-cell sender,
  e.g. a receptor marker, scores exactly 0 there). On the lung data (cell masking) `all` = `context`.
- A gene the model reads as noise gets a negative ΔR² under the mean replacement (the noise removed); `importance` sums positive
  effects only.

Cluster files (flat in `slurm/new/`): `ranking_common.py` (hooks `gene_rankings`, `programme_scores`), `ranking_stability.py` (+ `.sbatch`,
CPU; partition guessed), synth: `synth_ranking.{py,sbatch}`, `synth_ranking_sweep.yaml`, `make_synth_ranking_yaml.py` (stage 2 via
`--roar`), `submit_synth_ranking.sh`; lung: `make_lung_inputs.py`, `lung_hier.yaml`, `make_lung_ranking_yaml.py`, `lung_ranking_sweep.yaml`,
`lung_ranking.{py,sbatch}`, `submit_lung_ranking.sh`. Lung data facts: 45 samples of 35 patients (9 control, 26 disease), one status
per patient; 5 patient groups dealt by condition, so every test fold holds both.

**Read first.** Synth: `del_recovery_summary` (precision@1 and MRR of the planted senders, `context`) against the twin; stability.md
(seed pairs' RBO and Jaccard@20); then synth_roar: programme scores of `top<k>` against `rand<k>` and `none`, `planted` as the bar.
Lung: heads_within_* first (did the model learn), then the seed pairs' agreement of `sender_all`, then the consensus senders.

## Runs as single jobs (2026-11-02, the user: "Why can't we do single runs with multiple arms as initially?")

The first version spread the runs over job arrays (one task per run, chained submit scripts). Replaced by one job per experiment,
every run in turn (`ARRAY_TASK = False`, sweep_common's default), sized to stay under 12 h:
- **synth** (`sbatch slurm/synth_ranking.sbatch`): stage 1 `full` x seeds 42-44 + twin -> `ranking_stability.py --roar-yaml` -> stage 2
  at seed 42 only: `top10`, `bottom10`, `rand10_d0`, `rand10_d1`, `planted` (5 runs; k = 20 and the `none` arm dropped: stage 1's
  `full_s42` is the baseline). ~5-9 h.
- **lung** (`sbatch slurm/lung_ranking.sbatch`): fold 0 only (the seeds are the replicates; no fold stability in this run), seeds 42-44,
  100 epochs, deletion with 1 conditional draw on 3 test samples, + twin, then the stability step. ~9-10 h, unmeasured.
`submit_*.sh` and `ranking_stability.sbatch` are gone; the earlier lines of this plan that name them, `rand<k>` x 3 draws, two folds,
seeds 42-43 in stage 2 or 2 draws describe the first version.

## Follow-ups of the synth read (2026-11-03; verdicts in notes.md)

`sbatch slurm/synth_followup.sbatch`, written, not run: stage `permute` (stage 1's checkpoints, `METHOD: permute`, `receiver_scores`: Pearson within each slide's receivers) and stage `followup` (one class removed per arm: sparse_sources, receptors, rings, own_row; seeds 42-43). Read: did M7, M8, M1-M6 rise under `permute`, and does `sparse_sources` lower `receiver_corr` for MC16-MC22 against `full`? A pooled-over-slides conditional is built only if `permute` confirms the diagnosis.

Follow-ups read 2026-11-04 (notes.md): `permute` confirms the diagnosis; rank communication by `context` + permute. `all` needs the pooled conditional (not built): permutation flags collinear genes (CM1). Next candidates: deletion of M12-M18 on the `receptors` checkpoints; the rings arm's collateral.
