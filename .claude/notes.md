# Notes: what worked and what does not

A log of the verdict of every test. **Append only:** new entries go at the end of this file; never
overwrite or edit an earlier entry (a correction is a new entry that says what it corrects). Each entry
is `## <date> — <what the test was about>`, then **What worked** and **What doesn't work**. Tables,
settings and the reasoning live in the plan files (`global_plan.md`, `attention_plan.md`, ...); this
file keeps only the points.

---

## 2026-10-20 — Programme masking on synth_spot (`response_masking`: 11 arms, 19 runs, 3 seeds, 100 epochs)

Arms: `local_gcn` is the GCN alone (`LocalModel`, no transformer); `lin_*` is the transformer alone
(`GlobalModel`, learned linear input); `joint_*` is GCN + transformer with the usual dual decoder, trained
jointly (no staged schedule, nothing frozen, long-range mask on); `*_pool500` runs the transformer on 500 µm
patches. The bar is `ref_own+context` (0.682).

**What worked**
- `joint_fourier`'s global head scores 0.820 ± 0.005 over three seeds against the bar's 0.682 and its own
  local head's 0.506. Seed spread is 0.003-0.017 overall, so differences above ~0.03 are real.
- The gain is in the long-range programmes: the rings M7 / M8 score 0.93 / 0.93 against 0.63 / 0.75 (bar) and
  0.44 / 0.49 (GCN), the long-range responses 0.884 against 0.692 (bar) and 0.471 (GCN alone).
- Against the GCN alone, the joint global head is better by more than 0.05 on 5 of the 25 programmes (M7 +0.47,
  M8 +0.42, M11 +0.18), ties on 16 and is worse on 3. The overall 0.820 against 0.510 is carried by the rings.
- Fourier bias beats RoPE in every arm (joint 0.820 against 0.748, transformer alone 0.764 against 0.709).
- A blend with the bar no longer adds to the joint head (0.804 < 0.820): it has absorbed what the references know.
- 100 epochs instead of 200 costs about 0.01 (`lin_fourier` 0.764 against 0.775).
- The pipeline is fast: a run trains in 2-3 min and evaluates in 30-60 s; the fitted references take 61 s once;
  a 19-run job array finishes in minutes.
- Pooling at 500 µm keeps the ring and long-range programmes (0.79 against 0.89 per cell) and runs 1.5x faster
  with 3.5-5.5x less GPU memory (at only 2,500 spots per slide).

**What doesn't work**
- Per programme, no model beats the bar. Equal weight over the 25 programmes: `joint_fourier` 0.628 against
  the bar's 0.685; it beats the bar on 3, ties on 12, loses on 9.
- Mirror programmes (MC1, MC2, MC4, MC6: the response copies the cell's own visible source gene) score 0.2-0.4
  below a plain ridge on the cell's own genes, in every neural arm, the GCN alone included.
- M11 (response within 500 µm of single source spots): 0.54 against 0.92 for a 150 µm kernel; the near-source
  scale is not learned.
- Directional programmes (M9, M10): about 0.1 in every arm and reference; the Fourier bias used is isotropic.
- Sparse threshold programmes (M16, M17, M20) only match `ref_own`; pooled over all cells the score mostly
  measures whether a cell is a receiver, so it cannot show whether the distance to the source was learned.
- Pooled 500 µm tokens lose the sparse responses (0.07-0.08 against 0.78): a patch token is shared, so the
  global head cannot know which cell is the receiver (the joint arm's local head, which reads cells, scores
  0.757 on the same genes). Pooled overall is 0.62 against 0.76 per cell.
- A PCA of the visible genes is a poor token (0.34-0.42 against 0.76 for the learned linear input).
- The transformer alone scores about 0 on the short-range receivers (0.01 against 0.6 for the GCN); only the
  joint model reaches them (0.64).
- Sender ranks are unreadable (0.35-0.57; the GCN ranks the senders of its own short-range programmes at
  0.41 while predicting them at 0.59): the gradient-magnitude measure is invalid and has to be replaced by an
  attribution (roadmap T4).
