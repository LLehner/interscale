# legnini programme masking: the identity code, with the SHH readouts as a control

Decided 2026-10-20 (the user's call). Part of roadmap T1 (does the global model learn communication,
on real tissue) and T8 (SHH as the positive control). The method is `global_plan.md`, "programme
masking"; the synth_spot run is `response_masking`, this one is `legnini_programme` (scripts in
`notebooks_and_scripts/slurm/`, which is not under version control).

## The task

Hide a fixed gene set in **every** cell (`dataset.mask_genes`, `mask_strategy: gene`, nothing else
hidden), so a hidden gene can only be predicted from what remains: the cell's other genes, the cells
around it, the SHH source, and position. The loss and the evaluation are on the hidden entries alone.

| set | genes | role |
|---|---|---|
| **main task** | the dorsoventral **identity code**, 15 genes: NKX2.2 NKX6.1 NKX6.2 OLIG1 OLIG2 · FOXA2 · MNX1 ISL1 CHAT · PAX6 PAX7 IRX3 DBX1 DBX2 · MSX2 | where a cell sits between the SHH source and the dorsal side |
| **positive control** | the SHH **readouts**, 3 genes: PTCH1 GLI1 HHIP | the textbook signal -> response relation |
| visible, signal and machinery | SHH FOXA1 (the source cells), SMO GLI2 GLI3 (the cell's receiving machinery), BMP4 (the dorsal signal) | what a response can be explained by |
| visible, everything else | the other 64 genes | context |

Both hidden sets are hidden in the same training run (18 of 88 genes) and scored apart, by `var['role']`
(`identity`, `readout`) and `var['program']` (`ventral_progenitor`, `floor_plate`, `motor_neuron`,
`intermediate_dorsal`, `bmp_dorsal`, `shh_readout`), both written by the script when the data is read.
Hiding the readouts together with the identity code also takes away the shortcut that a cell's identity
genes would give for its readouts (the first model-free look found the cell's own genes explaining
the three readouts, centred cosine 0.51), so the control asks for the source and the neighbourhood.

## Why each gene

Sources are keyed below; "L23" is the dataset paper, "M" the InterScale draft. *Fold* is the mean
log1p_norm in SHH-induced over control organoids, *rho* the Spearman correlation of the gene with the
distance to the nearest SHH+ cell in the SHH **test** organoid `slide4_B2-2` (1,779 cells, 145 SHH+;
SHH+ by the Otsu threshold on SHH, as the manuscript's "SHH rings"); negative means higher near the source.
Both from the data on 2026-10-20 (`legnini23.h5ad`, 43,762 cells).

### Hidden: the identity code (main task)

| gene | `program` | biological reasoning | where | fold / rho |
|---|---|---|---|---|
| NKX2.2 | ventral_progenitor | Progenitor TF of the p3 domain, the most ventral progenitor; it needs the highest SHH dose of the class II factors (~4 nM against ~0.25 nM for NKX6.1) and is a direct Gli target. Induced by SHH activation in the organoids; its domain is depleted of PAX6 and IRX3. | B00, V07; L23 (results "Optogenetic patterning of neural organoids", SHH and p3 modules in the methods); M p. 8 | 11.3x / -0.19 |
| NKX6.1 | ventral_progenitor | Class II factor that needs SHH at a low dose: the wide ventral region (floor plate to p2). In the SHH, floor plate, p3, pMN and p0-2 modules; one of three genes stained in the organoids (Fig. 3g). | B00; L23; M p. 8 | 9.7x / -0.20 |
| NKX6.2 | ventral_progenitor | Named with NKX6-1 among the ventral markers "induced on SHH activation"; in the SHH and p0-2 modules. | L23 | 17.9x / -0.07 |
| OLIG2 | ventral_progenitor | bHLH factor of the pMN domain (motor-neuron and oligodendrocyte precursors), induced by SHH. In the SHH and pMN modules; stained in the organoids. | N01; L23; M p. 8 | 16.8x / -0.15 |
| OLIG1 | ventral_progenitor | OLIG2's paralog; the paper and the manuscript name "OLIG1/2" together. | L23; M p. 8 | 13.7x / -0.10 |
| FOXA2 | floor_plate | Floor-plate factor, at the highest SHH exposure. The paper finds FOXA2 spreading "in the immediate vicinity" of SHH+ cells "in a noncell-autonomous manner" (Fig. 3i): a response to a neighbour's signal, which is what the task asks for. **Changed from the first proposal, where it was listed beside SHH as a source marker**: FOXA1 is the source-cell marker, FOXA2 the response. | L23; R10 | 0.85x (controls express it at the surface too) / **-0.41** |
| MNX1 (HB9) | motor_neuron | Postmitotic motor-neuron identity, downstream of OLIG2 and NKX6.1; the paper stains HB9/ISL1 at day 30 as the ventral fates SHH induces. | A99; L23 (Extended Data Fig. 7e) | 4.4x / -0.04 |
| ISL1 | motor_neuron | LIM-homeodomain factor required for motor neurons; the manuscript's example of a local-embedding gene ("a motor neuron differentiation factor"). | P96; M p. 8; L23 | 63.6x / -0.13 |
| CHAT | motor_neuron | Cholinergic motor-neuron marker. General knowledge only: **neither L23 nor M mentions it** (it is on the panel). 10x higher in induced organoids. | - | 10.2x / +0.04 |
| PAX6 | intermediate_dorsal | Class I factor, repressed by SHH: absent from the NKX2-2, OLIG2 and FOXA domains, present farther away. In the p0-2 and dp modules. | B00; L23; M p. 8 | 0.79x / **+0.33** |
| IRX3 | intermediate_dorsal | Class I, as PAX6 (the study that defined the code gave it a role in the ventral third). | B00; L23; M p. 8 | 0.47x / **+0.27** |
| DBX1, DBX2 | intermediate_dorsal | Class I (p0/p1): "confined to spatially distinct regions farther away" than FOXA, NKX2-2 and OLIG2. The paper's sentence counts them among the ventral markers induced by SHH, but in this panel they are *lower* in induced organoids (0.67x, 0.50x); they sit between the two halves of the code either way. | B00; L23; M p. 8 | 0.67x, 0.50x / +0.26, +0.26 |
| PAX7 | intermediate_dorsal | The dorsal-most class I factor: "almost completely depleted" in induced organoids; with MSX2 it is the paper's own evidence that SHH ventralises. | B00; L23 | 0.03x / -0.08 |
| MSX2 | bmp_dorsal | Dorsal and roof-plate marker, depleted by SHH and raised by BMP signalling (the paper's BMP4 line raised MSX1 about fourfold). In the dp1-6 and roof-plate modules. | L23 | 0.13x / -0.04 |

### Hidden: the SHH readouts (positive control)

| gene | `program` | biological reasoning | where | fold / rho |
|---|---|---|---|---|
| PTCH1 | shh_readout | The SHH receptor and its own transcriptional target (the negative feedback); the standard readout of Hedgehog activity in the neural tube. In the paper's SHH module. | MT96; L23 | 1.09x / +0.13 |
| GLI1 | shh_readout | Transcriptional target and effector of the pathway, made only while it is on. The manuscript's example of a local-embedding gene. | L97; M p. 8 | 2.3x / +0.14 |
| HHIP | shh_readout | A Hedgehog-binding protein that is itself a target of the pathway (Hip): another negative feedback. In the paper's SHH module; in the manuscript a global-embedding gene. | CM99; L23; M p. 9 | 0.82x / +0.03 |

### Visible

| gene | why it stays visible | where |
|---|---|---|
| SHH | The ligand, and the source: "SHH+" cells are the optogenetically induced ones (the manuscript's "SHH rings"). The model must see where the source is. | L23; M (methods, appendix) |
| FOXA1 | "Mostly confined to SHH+ cells": a marker of the source cells, so it travels with SHH (rho -0.48 in the test organoid). | L23 (Fig. 3i) |
| SMO, GLI2, GLI3 | The receiving machinery (transducer, Gli activator and repressor). They belong to the cell that responds, not to the response: a readout is then explained by the source plus the cell's ability to receive. They are 0.6-0.8x in induced organoids and correlate positively with distance (+0.02 to +0.26). | B09, D08 (the pathway); L23 mentions SMO only (upregulated near SHH in the hiPSC time course, Extended Data Fig. 3m), GLI2 and GLI3 not at all |
| BMP4 | The dorsal signal, a second morphogen axis; the manuscript's list does not include it. | L23 |
| OTP, OTX2 | Regional markers the paper places "farther away" (OTX2: "a forebrain and midbrain marker ... in a narrow and spatially distinct subset of cells"); not part of the spinal dorsoventral code. Rho in the test organoid +0.22 and -0.08. | L23 |
| NKX2.1, FOXG1 | Forebrain markers the paper tests in hiPSCs; detected in 0.1-0.2% of the organoid cells, so there is nothing to predict. | L23 |
| the rest | Neurogenesis and neuron genes (SOX2, HES6, NEUROG2, NEUROD1, TUBB3, STMN2, ...), cortical and other neuron markers, glia, immediate-early genes and housekeepers: the cell's own state, which is what `ref_own` reads. The manuscript's other embedding genes (PROM1, NEUROG2, TUBB3, CUX2, FOXO6) are in here, not identity. | M p. 8-9 |

### Sources and how far each was checked

- **L23** Legnini et al., *Spatiotemporal, optogenetic control of gene expression in organoids*, Nat Methods
  20:1544 (2023). **Read in full on 2026-10-20** (open-access copy): the 88-transcript Molecular Cartography
  panel "including known SHH targets and markers for distinct DV neural tube cell populations", the ventral
  and dorsal marker statements above, and the module gene lists in the methods (SHH module: SHH FOXA2 NKX2-2
  OLIG2 NKX6-1 NKX6-2 PTCH1 HHIP; floor plate FOXA2 NKX6-1 SHH; p3 NKX6-1 NKX2-2; pMN NKX6-1 OLIG2; p0-2 IRX3
  PAX6 DBX2 DBX1 NKX6-2 NKX6-1; dp1-6 MSX2 PAX6 PAX7 IRX3 DBX1 DBX2; roof plate MSX2), taken from the mouse
  (Delile et al. 2019) and human (Rayon et al. 2021) spinal cord atlases.
- **M** `InterScale_Manuscript.pdf`, the draft: results section "The local and global role of the SHH morphogen
  along the dorsoventral axis" (pp. 7-9) and the dataset methods (p. 14). The draft's reference list is not in
  the PDF, so its reference numbers (39-43) are not resolved here.
- **B00** Briscoe, Pierani, Jessell, Ericson, *A homeodomain protein code specifies progenitor cell identity and
  neuronal fate in the ventral neural tube*, Cell 101:435 (2000): class I factors (PAX7, DBX2, IRX3, PAX6)
  repressed by SHH, class II (NKX6.1, NKX2.2) requiring it; cross-repression. Abstract and a figure legend
  confirmed by search on 2026-10-20.
- **V07** Vokes et al., *Genomic characterization of Gli-activator targets in sonic hedgehog-mediated neural
  patterning*, Development 134:1977 (2007): NKX2-2 and NKX2-1 confirmed as direct Hh targets (search).
- **MT96** Marigo & Tabin, PNAS 93:9346 (1996) (Ptc as a readout of Hh activity, with negative feedback);
  **L97** Lee, Platt, Censullo, Ruiz i Altaba, Development 124:2537 (1997) (Gli1 a target of Shh);
  **CM99** Chuang & McMahon, Nature 397:617 (1999) (Hip binds Hh and is a general transcriptional target).
  Titles and findings confirmed by search.
- **A99** Arber et al., Neuron 23:659 (1999) (Hb9 consolidates motor-neuron identity); **P96** Pfaff et al.,
  Cell 84:309 (1996) (Isl1 required for motor neurons). Confirmed by search (abstracts).
- **N01** Novitch, Chen, Jessell, Neuron 31:773 (2001) (Olig2 at the nodal point of pMN fate): the work is
  confirmed in citing sources; the volume and pages are from memory.
- **R10** Ribes et al., Genes Dev 24:1186 (2010) (distinct SHH signalling dynamics specify floor plate and ventral
  progenitors): title confirmed by search, pages from memory. **B09** Ribes & Briscoe, Cold Spring Harb Perspect
  Biol 1:a002014 (2009), the review L23 cites for the pathway; **D08** Dessaud, McMahon, Briscoe, Development
  135:2489 (2008), title confirmed, authors from memory.
- Not sourced: CHAT (above), and the class I/II labels of IRX3 and DBX1 beyond B00's list, from the field's
  standard picture.

## What the data say about this choice (read before trusting the control)

- The identity genes are strongly induced and ordered around the source in the test organoid: FOXA2 -0.41,
  NKX2.2 -0.19, NKX6.1 -0.20, OLIG2 -0.15 near; PAX6 +0.33, IRX3 +0.27, DBX1/2 +0.26 far. This matches the paper's
  description and is why the code is a task with a spatial answer. They are also spatially clustered (Moran's I in
  SHH organoids: NKX6.1 0.50, NKX2.2 0.37, OLIG2 0.36; `notes.md`, 2026-10-19).
- **The readouts are a weak control in this data.** PTCH1 is detected in 88% of cells and is 1.09x in induced
  organoids; GLI1 is 2.3x, HHIP 0.82x. In the test organoid PTCH1 and GLI1 correlate slightly *positively* with
  the distance to SHH+ cells (+0.13, +0.14), not negatively: the responding cells are not the producing cells,
  and the gradient is not a clean function of in-plane distance (a section need not cut the induced pole; the
  paper says so). Averaged over the 11 induced organoids the correlations are +0.03, +0.06 and +0.01. So a model
  that fails the control has not necessarily failed to learn communication; **FOXA2** (floor plate) and NKX2.2 are
  the cleaner source-dependent genes, and `program=floor_plate` is scored alone for that reason.
- Much of the code can be read from a cell's own other genes (neurogenic and neuronal genes move with the
  condition: HES6 3.6x, NEUROD1 4.8x, HEY1 3.4x, NR4A2 5.9x). `ref_own` and `ref_own+context` are the bar;
  the question is what the global model adds beyond them, not whether it beats zero.
- The test set is two organoids (`slide4_A2-3` control, `slide4_B2-2` SHH; 3,763 cells), so a difference of a few
  hundredths between arms is within what swapping the test organoids could move (roadmap K9: rotate them).

## Run

`legnini_programme.py` with `legnini_programme.yaml` and `legnini_programme_sweep.yaml`, the machinery of
`sweep_common.py` as for `response_masking`: `MODE` smoke, then sweep (or list / job array / report), evaluate to
redo the evaluation from the checkpoints.

- **Data:** the preprocessed file the attention sweep wrote (`legnini23_attention_pp.h5ad`; made from the raw file
  when missing): log1p of counts scaled to 1e4, the file's split by sample (10 / 5 / 2), radius-200-pixel graph,
  4,299 cells at most, 0.138 um per pixel.
- **Arms (7 runs):** `local_gcn`; `lin_fourier` (global only, Fourier bias); `joint_fourier` (GCN + transformer,
  dual decoder, long-range mask; **3 seeds**, the provisional base model of the synth_spot read); and the pooled
  `lin_fourier_pool100` and `joint_fourier_pool100` (100 um squares: 35 cells per patch on average, 73 tokens per
  organoid (39-112) instead of 1,640-4,318; roadmap T1b). The Fourier wavelengths start at 25 um, two cell spacings. No PCA arms.
- **Training:** 240 epochs over the 10 training organoids at batch size 1 = 2,400 optimiser steps, as the 100
  epochs of the synth_spot sweep (24 slides). Untrained twin of `joint_fourier`.
- **Evaluation** (`EVAL_MASK = "gene_set"`): the 18 genes hidden in every test cell; the centred cosine of each
  head next to references fitted on the training organoids (the cell's own visible genes; plus kernel means of
  the visible genes at 15, 40, 100, 250 and 500 um; plus position), per `role` and `program`, with each program
  counted as a win, a tie or a loss against the best reference on it (`PROGRAMME_BAR = "best"`,
  `PROGRAMME_PREFIX = "program="`; on a stub `ref_own` beat `ref_own+context` for the identity genes, 0.64 against
  0.60, so a fixed reference would be a soft bar). There are no
  planted edges, so no sender ranks and no within-receiver table.

## Not built

- **Attribution to the SHH+ cells.** The natural communication test here is the synth_spot one: the weight a
  hidden gene's prediction puts on the SHH+ cells (the Otsu set), against cells as far away, by the integrated
  gradients of `response_influence`. It needs "edges" (SHH+ sender, receiver within some range) built from the
  data, and a range, which is a choice (the gradient has no known length scale in the organoid).
- FOXA2 as a second control, or a different readout set, if the user prefers: change `mask_genes` in the yaml and
  `GENES` in the script together (the script refuses a mismatch).
- Rotating the test organoids as folds (K9).
