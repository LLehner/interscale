# Implementation ideas for the weekend of 2026-11-28/29 (written 2026-11-27, while the user starts writing the thesis)

## Status (2026-11-28; the user: "implement A, B and C now")

| item | status | where |
|---|---|---|
| A1 frozen attention | BUILT + tested; cluster job `and_eval` written, not run | `attribution.AttentionTape`, `pairwise_interactions(freeze_attention=)`; tests/test_attention_tape.py; attr_common `AND_FROZEN` |
| A2 AND-capable arms, stronger AND | BUILT; jobs `and_mlp`, `and_strong` written, not run | make_and_synth_yaml.py variants; `make_synth_spot(and_rate=, and_high=)` (CLI `--and-rate`, `--and-high`) |
| A3 paired R5 - R6 | BUILT; read today from 11-27 tables (unpaired): 0.000 +- 0.005 in every run | attr_common table `and_receivers`; attr_read.py rule v2 |
| A4 closed-form control | BUILT as tests (linear heads 0, bilinear head exact) | tests/test_attention_tape.py |
| B5 seed stability | BUILT + read locally (synth 0.86, CosMx 0.76, legnini 0.48 fails) | allmap_checks.py |
| B6 outliers / edges | BUILT + read locally (ir_c heavy tail; legnini top senders not at the edge) | allmap_checks.py |
| B7 references | BUILT; job `allmap_synth_refs` written, not run (old maps carry one reference) | allmap_checks.py; attr_common keeps `out_<ref>` / `in_<ref>` |
| B8 type null | BUILT + read locally on CosMx (labels by position) | allmap_checks.py |
| B9 figure | BUILT (common scale, gaps collapsed); in the map job and the checks | `subtract_fields(normalise="common")`, `allmap_checks.draw_panels` |
| C10 SHH truth scaffold | BUILT; list extracted from Legnini et al. 2023 (slurm/November/shh_targets.txt, user to confirm) and scored on the 25_11 rankings (notes.md 2026-11-28) | legnini_truth.py; compare_job.sbatch report step (copy the list to MA/shh_targets.txt) |
| C11 SHH region | BUILT + read locally: the map's block is not at the source | legnini_shh_region.py |
| C12 semi-synthetic legnini | NOT BUILT (~1 day; user decides after the SHH list) | - |

Results of the local reads: notes.md, 2026-11-28.

Nothing here is built; the user asked for ideas to be noted and implementation to continue in the next session. Constraints that shape the list:
no model runs or heavy jobs locally (cluster only, the user submits); light checks on the saved maps / csv tables are fine; new job files flat in
slurm/November/; every item states what it needs. Order = my recommendation (thesis value per hour). The deadline Tue 2026-12-01 (vannan + IMC
analysis) is not on this list; it stays first in the next session (see reminder-next-session).

## What the 2026-11-27 reads left open (the source of the list)
- S13: no model learned the AND, and the additive twin reads as large as the AND (0.17 / 0.17 ref, 0.105 / 0.105 ir2_rate). The instrument is
  unvalidated; the floor is probably softmax re-weighting (untested).
- All-cell maps: synth references disagree on net / in (Spearman 0.3 / -0.5); CosMx type pattern = abundance; legnini hot patches = seed-specific edge
  outliers; the trained-minus-twin figure is speckle; per-cell seed stability never computed.
- legnini: ir_ch and the prototypes name different drivers (Spearman 0.1-0.2); no truth. The user will provide SHH targets (reminder).

## A. Close S13 (cluster needed for A1/A2; A3 is local-ish) -- highest value, the user's must-pursue
1. **Frozen-attention interaction test (eval only, existing ir2_rate / ref checkpoints in 27_11_results/attr_and).** Recompute the planted-pair and random-pair
   interaction with the softmax weights detached (attention computed from the unperturbed input, values perturbed). Prediction if softmax re-weighting
   is the floor: interaction on R6 falls to ~0.01 and the trained/twin ratio disappears. If it stays at 0.1 the floor is the decoder / log non-linearity.
   Implementation: a `freeze_attention` context in `evaluation/attribution.py` (hook on the attention module returning cached weights), a flag
   `AND_FREEZE_ATTN=1` in `attr_common._and_control`, one extra table `attr_and_pairwise_frozen`. Test: on a toy model with a known softmax, frozen
   interaction of an additive value map is exactly 0, unfrozen > 0 (mutant: not detaching is caught).
2. **A model that can express the AND.** Add an `ir2_rate_mlp` arm: the read-out's per-sender values summed to a context vector, then a small MLP (2 x width)
   instead of the linear decoder, so S5 and S6 can interact after the sum. Train on synth_spot_and, 300 epochs, two seeds; rule: AND index within 0.2 of
   1 for R5 *before* the interaction is read. If it still does not learn it (the data's AND is 2 senders of 8 % each among 4 neighbours: rare), raise the
   amplitude (R5 both-state 10 -> 25 counts) and the sender fraction (8 % -> 15 %) in `make_synth_spot(and_programme=True, and_scale=)`, opt-in, default
   unchanged. Needs: sweep entry in `attr_job.py` (`ATTR_JOB=and`, new arm), make_and_synth_yaml.py, job time ~3-6 h per arm.
3. **Paired additive contrast as the reported number.** Report per receiver R5 minus R6 (same receivers, senders, endpoints: only the rule differs), the
   median paired difference with a bootstrap interval over receivers, instead of the absolute relative interaction against a fixed 0.25 / 0.1 rule.
   The absolute rule failed for a reason unrelated to the AND. Local change in `attr_read.py` (`and` report), re-read of the existing tables
   (attr_and_pairwise) gives it today: expected ~0 for both models (0.170 vs 0.169). Cheap and honest; makes the negative result a clean sentence.
4. **Closed-form control (no training).** A hand-built model on the synth data: fixed-sum context, no softmax, then (a) a linear head, (b) an exact AND
   head. The pairwise measure must read 0 / ~0.5. Validates the instrument independent of training; also a unit test at the library level. Local, light.

## B. All-cell maps: make them readable or retire them (local on the saved npz in 27_11_results/allmap_*/allcell_maps, plus small cluster jobs)
5. **Per-cell seed stability** (never computed): Spearman of out / in / net between the three CosMx ktype seeds per section, and the two seeds on legnini
   and synth; then the seed-averaged map (Spearman-Brown reading as in flow-ensemble-averaging). A map is only drawn from the average if the seeds agree
   (rule: Spearman >= 0.5). Local; script `allmap_seed_stability.py` reading the npz fields (`at.map_fields`).
6. **Outlier and edge diagnostics.** Per section: top-1 % share of |out|, the per-cell maximum, out against the number of neighbours within 100 um and against
   the distance to the section border; legnini's corner patches (p0_0, p7_6) and CosMx ir_c's Endothelial cells are the test cases. If out scales with low
   degree, add a degree-normalised map (out per attended edge) as the reported one. Local; same npz plus coordinates.
7. **Reference disagreement on synth.** Why is `in` anti-correlated between graph-mean and sampled references (-0.5)? Check on the existing maps whether the
   sign flip sits in the lesion region, and run the sampled reference with 10 draws (the average over references is the proper expected IG; 3 draws were
   used). Cluster job, minutes (synth maps cost 1-2 s). Report `out` only for synth until it is settled.
8. **Type-level null for the CosMx map.** Permute the type labels within spatial neighbourhoods (preserves abundance and local composition) and report
   mean out / in per type with the null band; abundance-normalised (share of |out| / cell share). Replaces eyeballing "Acinar sends". Needs labels:
   the npz carry positions; the labels come from cosmx_inputs.h5ad (~100 MB, a light read is allowed) or a small cluster job.
9. **Fix the trained-minus-twin figure**: common colour scale (the trained map's 99th percentile for both) and the twin as its own panel; both maps in one figure.
   Small edit in `plot_source_sink` / `_scatter_map`; redraw from the npz locally.

## C. legnini (when the user provides the SHH targets; prepare the scaffold now)
10. **Truth scaffold:** `data/shh_targets.txt` (one gene per line, optional second column: expected sign / distance class) read by `compare_rank.py` for
    COMPARE_JOB=legnini: AUROC / AUPRC of the list among the 78 responders and the 88 drivers per model, rank of each list gene, against the twin and
    the prototypes. Unit-tested on a made-up list. Also deduplicate the drivers csv (receiver `*` and `all` rows) in the reader.
11. **SHH-region proxy for the map:** define the induced region from the data (cells with high SHH-pathway readout in slide4_B2-2: GLI1 / PTCH1 / HHIP
    mean in a 100 um window, or FOXA2 if present), then report where the SHH-readout all-cell map's hot block lies relative to it (overlap of the top 5 %
    cells, distance of the map's centroid to the region), control section as the null. Light read of legnini23_pp.h5ad; eval-only.
12. **Semi-synthetic legnini (bigger, optional):** plant a known sender -> response dependency in the real legnini expression (as J6b does on CosMx)
    so the ir_ch-vs-prototype driver disagreement has an answer: which family recovers a planted driver at a known distance. Cluster training, ~1 day
    to build; only if the SHH list is thin.

## D. Thesis support (the user is writing now; these save time)
13. **Claims ledger** `.claude/claims_ledger.md`: one row per claim the thesis may make (S13 additive read-out; typeres 4x twin; all-cell map validity on
    synth; CosMx type pattern; Beta -> Alpha; legnini rankings; A4 distance), with status (supported / unsupported / not validated / retired), the
    evidence (notes.md entry, table, seed count), the caveat in one sentence and the wording to use (and not to use). Built from notes.md; no code.
    Highest value for the writing; I recommend doing this first.
14. **Figure / table scripts from the saved tables** (local, no models): S13 control table (learned AND index by arm; paired R5 - R6), all-cell map
    AUROC bars (trained vs twin by range class), the A6 per-run table with twins, the A2 heat map (trained minus permuted), one slide per figure and
    tissue only (see plots-show-tissue-only). `thesis_figs/` flat, one script per figure, reads results/*/attr_read.md or the csv.
15. **Method paragraph drafts** for 4.7 / 5.2 in the fork's .claude (not in the thesis files): IG with type-mean reference and completeness; all-cell IG for blind
    models; pairwise interaction and its floor; the planted-AND control and why it failed. Short, with the numbers from notes.md, so the user can paste.

## Suggested order for the next session
(1) deadline items (vannan / IMC); (2) 13 claims ledger, 3 paired contrast, 5 + 6 map stability and outliers (all local, a few hours); (3) 1 + 2 + 4
(the S13 repair; 1 and 2 need cluster time, write the jobs first); (4) 10-11 when the SHH list arrives; (5) 14 as the figures are needed.
Not recommended now: 12 (semi-synthetic legnini), new architectures, more arms on CosMx.
