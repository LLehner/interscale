# Literature review: local vs global genes, and ranking genes (2026-11-02)

Asked 2026-11-02: "check whether there are more worthwhile strategies to 1) disentangle local from globally relevant genes (in the
dual model or the global-only model, can be different strategies) and 2) how to rank genes in general. Take notes of what is
worth doing (with references)."

**What we already have (so the items below say what is new):** Moran's I curves over k-shells, `ell_um`, the M2 scale sets
(relative fall of I from the GCN's reach to the slide); the additive decoder D3.1 + context query D1.2 (`add_ctx_fourier`)
with `attribution_shares` (cov_share); the own-cell skip in the global-only model; D4 `orth_*` terms; per-head distance
windows (`pe.distance.scales_um`, `pe.bias_heads`); IG sender attribution against the untrained twin; the ED0-ED2 dimension
readout, `gene_priority`, `prediction_tables`, varimax; reference predictors (knn smoothers, `ref_own+context`, section mean).

**Labels.** Priority **high / medium / low** = worth doing for InterScale now. Cost: **cheap** (post hoc on saved
predictions or embeddings, hours), **medium** (a new evaluation module, a day or two), **large** (a training change or a sweep).
`[checked]` = paper or docs read at abstract or methods level in this review; `[memory]` = cited from memory, not re-checked.

---

## Short version

1. **Replace the Moran's I *fall* with a per-gene variance partition across scales** (S1/S2): MEM / graph-Fourier bands or a
   multi-kernel mixed model (Spacelink). This gives each gene a *share of variance* per scale band instead of an
   autocorrelation, which cannot be compared across genes. The same bands then project each model part's prediction (M1).
2. **Take the cell's own label out before calling a gene local or global** (S4; built as Y0 in `yardsticks_plan.md`).
3. **Use a predictive, model-free yardstick, not only autocorrelation** (P1, MISTy views at the GCN's reach and beyond): per
   gene, how much R² the near neighbourhood and the far context add beyond the cell itself. That is "interaction-shaped"
   where Moran's I is not, and it already has a Python implementation (liana-py).
4. **Report variation-partitioning fractions for the model** (M2): unique-local, unique-global and shared R² per gene from
   frozen embeddings (three ridge fits). It removes the arbitrariness of how cov_share splits shared signal. It is the gene-level
   form of the D0 redundancy table and of SIMVI's double-ML spatial effect.
5. **For the global-only model, test global relevance per gene with a conditional randomisation test** (M3): replace the
   context beyond r by a spatially shifted copy (whole cells, structure kept), measure Δloss per gene on hidden cells,
   p-value by rank. Do the same within r. Every gene gets a four-way label (none / local / global / both) without needing parts.
6. **Rank sender genes by in-silico deletion** (R2), with conditional replacement values. Read the stability of every ranking
   over seeds and folds with RBO and RRA (R4), always against expression-matched nulls and the NegPrb floor (R1).

---

## 1. Data-side yardsticks for a gene's spatial scale (model-free; replace or extend M2)

**S1. Multiscale variance partitioning with spatial eigenvectors (MEM / graph Fourier bands).** high, cheap
- Literature: Moran's eigenvector maps (MEM / PCNM) are the orthogonal eigenvectors of the doubly centred spatial weight
  matrix. Each eigenvalue is proportional to its vector's Moran's I. Ecology groups them into broad, medium and fine
  "submodels" and partitions a response's variance among them by (adjusted) R² [Dray, Legendre & Peres-Neto 2006; Peres-Neto
  et al. 2006; adespatial tutorial]. SpaGFT is the same idea on a kNN graph Laplacian: low-frequency Fourier modes = smooth
  patterns, GFTscore = weight on low frequencies [Chang et al. 2024]. FlashS tests kernels at several scales in the frequency
  domain and combines the per-scale p-values with the Cauchy rule [FlashS 2026].
- Why it fits: a gene's Moran's I at a given k is a spectrum-weighted average of these eigenvalues. Our I(k) curve is a summary
  of the gene's spectrum, and the spectrum itself is the fuller readout. Band energy is a **fraction of the gene's variance**,
  comparable across genes, while I(k) mixes level and shape. The bands can be cut at physical wavelengths matched to the
  model: fine = within the GCN's reach (~2 hops), medium = up to the distance windows, broad = section scale.
- How: per section, Lanczos for the first few hundred low-frequency eigenvectors of the kNN-graph Laplacian (the fine band is
  the residual). Per gene, the share of its centred variance in each band, or R² of a ridge on each band's vectors. The same
  projection applies to model outputs (M1). Caveats: adjusted-R² partitions between orthogonal submodels still show non-zero
  overlaps [Dray et al., "Variation partitioning involving orthogonal spatial eigenfunction submodels"]. The synth_spot lattice
  has degenerate eigenvalues, so group by band rather than reading single vectors.

**S2. Per-gene multi-kernel mixed model (variance components per bandwidth).** high as a reference ranker on CosMx, medium cost
- Literature: SpatialDE splits each gene's variance into a spatial GP term and noise. FSV is the fraction of spatial variance,
  with one fitted length scale per gene [Svensson et al. 2018]. SpatialDE2 splits the covariance into **several spatial
  components** plus noise (a GLMM with Poisson likelihood) [Kats et al. 2021, preprint]. **Spacelink** chooses data-adaptive
  exponential kernels from a "kernel store" of bandwidths and fits a linear mixed model whose spatial variance is a **mixture of
  bandwidth-specific variance components**. It has a cell-type form that removes the effects of co-localising cell types, and
  it was validated on CosMx (R package) [Spacelink 2025, preprint]. nnSVG fits a per-gene length scale with a nearest-neighbour
  GP; its effect size `prop_sv` is the spatial share of variance. It beat Moran's I for small length scales [Weber et al. 2023].
- Why: these are the established answers to "what scale does this gene vary at, and how much". Run Spacelink or nnSVG once per
  dataset and compare our `ell_um` / scale sets with them (Spearman, scale-set overlap). They replace "Moran's I says" with
  a recognised per-gene effect size in the manuscript.

**S3. Big-small patch ratio (BSP).** medium, cheap
- Literature: for each spot, the variance of local means over a big radius divided by that over a small radius, fitted to
  a distribution for p-values. It is robust to the big radius, works in 2D and 3D, and scBSP scales it [Wang et al. 2023].
- Why: it is a calibrated version of our M2 "local vs global" rule (small radius = the GCN's reach, big = the window). It is a
  sanity check on the relative-fall rule that the user questioned (`|I_local - I_global|` vs relative fall).

**S4. Remove cell-type / composition effects before calling a gene "global".** high for real data, cheap-medium
- Literature: SVGs are dominated by cell-type markers because composition varies in space. C-SIDE, CTSV, spVC, Celina and
  STANCE test within cell type [Cable et al. 2022; Celina 2025]. A 2025 benchmark finds severe false-discovery inflation in most
  of them (Celina best, still ~41% FDP), and that random-effect models hold up better for rare cell types [ctSVG benchmark 2025].
  GASTON separates continuous gradients from discontinuities between domains via an isodepth [Chitra et al. 2024].
- Why: on CosMx / lung a "global" gene is often a region or cell-type marker (islet vs exocrine), which is not communication.
  Residualise on the cell's own label (or a region / niche label) before drawing the scale sets. **Correction 2026-11-02:**
  do *not* rebuild the sets on a residual of the neighbourhood composition. A gene a cell expresses because of the types
  around it is explained by composition, and that is the interaction signal (NCEM's niche effect). Read composition as a
  share of the variance instead (built as Y0, `yardsticks_plan.md`; the `scale_sets --residual-of` tool no longer exists in
  the scripts folder). Optionally residualise on a GASTON isodepth to separate "region identity" from smooth gradients.

**S5. Use structure-preserving nulls when two maps are compared.** high, cheap
- Literature: a plain permutation keeps no autocorrelation, so correlations between two smooth maps look significant.
  Moran spectral randomisation keeps a map's Moran's I and spectrum on irregular samples [Wagner & Dray 2015]. Toroidal
  random shifts are liberal at the edges [arXiv 1911.00240]. Neuroimaging spin tests serve the same purpose
  [Alexander-Bloch et al. 2018, `[memory]` beyond the search snippet].
- Why: ED3 (Lee's L between a dimension's map and its genes' maps) and any gene-map vs part-map comparison need such a null.
  The per-gene permutation null in `gene_scales` is fine for one gene's own I. It is not fine for map-to-map association.

Benchmark context [Li et al. 2025, Genome Biology]: across 14 SVG methods, SPARK-X first, SpaGFT second, **Moran's I third**,
with good ranking and the best calibration in the preprint. Moran's I is a sound base. **Most methods miss "high in a small
area, absent elsewhere"**, which matches our synth_spot finding that the sparse responses are barely autocorrelated. So
sparse interactions need the predictive yardsticks (P1, M3), not autocorrelation.

---

## 2. Predictive yardsticks: is a gene explained by its neighbours, and at what range? (model-free baselines)

**P1. MISTy-style multiview per target gene.** high, cheap (liana-py)
- Literature: MISTy fits, per target, one model per view: intraview (the cell's other markers), juxtaview (direct
  neighbours), paraview (Gaussian-weighted broad tissue, bandwidth l, optional zone of indifference `zoi`). A meta-model
  combines them. Outputs per target are `intra_R2`, `multi_R2`, `gain_R2` and each view's contribution [Tanevski et al. 2022].
  It is in liana-py as `li.mt.genericMistyData(..., bandwidth, zoi, n_neighs)` with linear or random-forest models
  [LIANA+, Dimitrov et al. 2024].
- Why: this is the model-free form of our question at gene level, and it is about prediction from neighbours (interaction-
  shaped), not autocorrelation. Use views matched to the model: juxta = the GCN's reach, para with `zoi` = beyond the reach
  (the "deep" region), a second para at the window scale. Read on the same cell folds as `collect_fold_parts`, then
  Spearman of MISTy's per-gene para share against the model's global share. Refinement: replace the meta-model coefficients by
  a group Shapley (LMG) R² split over the views, which does not depend on order [Grömping 2006].
- Caveat: the intraview includes the target's own co-expressed genes, so for programme masking build it from visible genes only.

**P2. NCEM radius curve.** medium, cheap
- Literature: NCEMs predict a cell's expression from its type and its niche composition at a radius. Linear models were as
  good as GNNs. The gain in held-out R² over the type-only model is the niche effect, read over radii [Fischer, Schaar et al. 2022].
- Why: the per-gene ΔR²(radius) curve is the predictive counterpart of our I(k) curve, and it comes from the lab's own method,
  which is easy to cite in the manuscript. It needs labels (fine for CosMx / lung), so P1 is the label-free version.

**P3. SVCA variance components (intrinsic / environmental / cell-cell).** low (slow GP), cite as a caution
- Literature: SVCA writes `K_int + K_env + K_c-c + noise` per gene. **The environmental term absorbs the cell-cell signal
  when the latter is not modelled** [Arnol et al. 2019].
- Why: this is the same identifiability problem as our additive parts (without D1 they "repeat and cancel"). Cite it in the
  write-up, and prefer M2's explicit unique / shared split.

**P4. Niche-DE / niche-LR.** low-medium (comparison track T10)
- Literature: per index cell type, genes up- or down-regulated near a niche cell type (NB GLM), with niche-LR linking them to
  ligand-receptor pairs [Mason et al. 2024].
- Why: a labelled list of context-dependent genes on CosMx / lung, to compare with our global-relevant set (M3).

**P5. Prior ranges from LR databases (sender side).** medium, cheap
- Literature: CellChat v2 restricts **contact** interactions to nearest neighbours and lets **secreted** ones (and, since
  v2.1.2, ECM-receptor ones) act within `interaction.range` (default 250 µm). CellChatDB v1 is ~60% secreted, 21% ECM, 19%
  contact [Jin et al. 2023/2025]. SpatialDM uses 100-200 µm for secreted signals and 6 nearest neighbours for contact
  [Li et al. 2023].
- Why: a weak label for the **sender** genes on real panels. Ligands of contact pairs should get their IG / deletion
  attribution (R2) at short distance, secreted ones further out. This checks whether the model's distance profile per sender
  gene is biologically ordered, with no planted truth.

---

## 3. Inside the models

### 3a. Readouts (no retraining)

**M1. Project each part's prediction onto the scale bands of S1.** high, cheap
- Per gene and per part (local / context / own-cell skip), the variance share in fine / medium / broad bands, next to the
  truth's own band shares. Global-only model: project the prediction and the own-cell skip part separately. This is a
  readout that "puts the data's scale and the model's split on one axis", and it replaces Plot A/B's reliance on I(k).

**M2. Variation partitioning of the decomposable prediction: unique local, unique global, shared.** high, cheap
- Literature: variation partitioning with fractions [a] unique to X, [c] unique to W and [b] shared, from three regressions
  [Peres-Neto et al. 2006]. Frisch-Waugh-Lovell / double ML: residualise both the outcome and the "treatment" on the covariate,
  then regress residual on residual. SIMVI does exactly this per gene with the spatial latent as treatment and the intrinsic
  latent as covariate, and reports intrinsic-R² vs spatial-R² per gene [Dong et al. 2025]. fANOVA purification moves effects
  between main and interaction terms until they are unique [Lengerich et al. 2020].
- How: on frozen embeddings, with cross-fitting over cell folds and held-out cells, ridge per gene on `h_local`, `h_global`,
  and both. Then `unique_global = R²(L,G) - R²(L)`, `unique_local = R²(L,G) - R²(G)`, `shared = rest`. For the global-only
  model, use L = the own-cell skip input (visible own genes) and G = the global embedding.
- Why: cov_share splits the shared fraction by how training went, so two arms with the same function can give different
  shares. The unique fractions do not depend on that. This is the gene-level form of the D0 redundancy table (ridge R² of
  global from local) and gives a ranking that is defensible: **genes by unique-global R²**. With D1.2 the shared fraction
  becomes interpretable as "neighbours that resemble the cell" (cell-type clustering), which S4 then explains.
- Note: for the **linear** additive decoder, the exact two-player Shapley values with a mean baseline are the centred parts
  themselves, so `attribution_shares` already is the Shapley answer. Shapley over parts only adds something for
  non-additive decoders (D3.2 gated mixture, nonlinear).

**M3. Conditional randomisation test of context beyond r, per gene (works without parts).** high, medium cost
- Literature: the holdout randomisation test (HRT) gives each feature a valid p-value for `Y ⊥ X_j | X_-j` from one fitted
  model. It compares held-out loss with the true feature against draws from its conditional distribution [Tansey et al. 2022;
  model-X framework, Candès et al. 2018 `[memory]`].
- How: on hidden cells of a fold, compare the per-gene loss with the true context against the loss after replacing every key
  cell **beyond r** with the cell at the same position in a randomly **shifted** copy of the section (whole expression vectors,
  so gene-gene structure, composition and autocorrelation survive; only the link to the receiver breaks). 50-100 shifts give
  a rank p-value and Δloss as the effect size, with BH across genes. Do the same **within r** (near context) for local
  relevance. Every gene gets a label: none / local / global / both, with effect sizes. Use r = the GCN's reach, or the
  290 µm "deep" threshold on synth_spot.
- Why: this is the most direct answer to "is gene g globally relevant", it needs no decomposable decoder (the user's
  global-only line, CosMx `hier_none`), and it tests the hypothesis rather than describing reconstruction. Validate on
  synth_spot first: planted long-range programmes must be "global", short-range "local", senders-only "none".
  Caveats: shifts are liberal at section edges (exclude a margin or use the variance-corrected shift). Absolute-position
  encodings see a shifted copy at the same coordinates, which is fine, because the test is about content.

**M4. Per-gene learned bandwidth from heads.** medium, cheap
- Literature: MGWR gives each covariate (process) its own bandwidth instead of one average one [Fotheringham et al. 2017].
- How: we already learn one window per head. IG through heads gives each gene a weighting over heads, so it inherits a
  weighted bandwidth, which can be compared with nnSVG / Spacelink `ell` (S2) and with S1 bands.

**M5. Scale of embedding dimensions → per-gene global share for a one-headed model (NSFH-style).** high for global-only, cheap
- Literature: NSFH mixes spatial (GP) and non-spatial factors. A gene's **spatial importance score** is its share of
  loading on the spatial factors [Townes & Engelhardt 2023]. MEFISTO learns a per-factor length scale and a smoothness `s_k`
  (0 = non-smooth, 1 = smooth) [Velten et al. 2022].
- How: ED3 already measures a dimension's (or varimax direction's) map scale. Turn it around: classify each direction by
  band (S1 on its map), then gene g's broad share = `sum_k W_gk² var(z_k) [k broad] / sum_k W_gk² var(z_k)`. A global-only
  model then gets a per-gene local / global split from its decoder alone, readable next to `gene_priority`.

### 3b. Training changes (sweeps)

**M6. Scale-blocked embedding (NSFH / MGWR as an architecture).** medium, large
- Idea: split the global-only embedding into blocks, each fed only by heads of one window range (narrow / wide / section),
  with an additive decoder over the blocks. The block split is identifiable up to block-wise transforms for additive
  decoders [Lachapelle et al. 2023]. This is hierarchical_attention_plan.md H-γ (additive near / far read-out) at the
  embedding level. It gives the global-only model parts by construction without a GCN.

**M7. Asymmetric independence penalty (SIMVI).** medium, cheap to add
- Literature: SIMVI penalises dependence between the intrinsic latent z and the spatial latent s (Gaussian mutual
  information from covariances, or MMD), **on z only**, because otherwise the intrinsic latent takes all information
  [Dong et al. 2025].
- How: a variant of the D4 `orth_*` terms with the global side detached, so only the local / own-cell part is pushed away
  and the context part keeps what it explains. This is a one-flag change to the existing terms (dropout 0, as D4 requires).

**M8. Balance the branches' learning speeds.** medium, cheap
- Literature: in multimodal nets the faster branch dominates and the slower one under-trains. OGM-GE scales each branch's
  gradient by its contribution and adds noise [Peng et al. 2022]. Wu et al. 2022 balance learning speeds in the same way.
  NSA gives the window its own branch for the same reason (hierarchical plan H1).
- Why: the GCN / own-cell path learns the short range first and the context path may stay under-used. Note that *path
  dropout* does the opposite of disentangling (each branch learns everything), so use gradient modulation, not dropout.

**M9. Causal / identifiable intra- vs inter-cellular models.** low (comparison only)
- Celcomen learns identifiable intracellular and intercellular gene-gene matrices and predicts counterfactual perturbations
  [Megas et al. 2026, ICLR 2025]. COVET / ENVI describes a niche by its gene-gene covariance [Haviv et al. 2024]. NicheCompass
  uses masked linear decoders over prior communication programmes [Birk et al. 2025]. These are candidates for the T10
  comparison, and NicheCompass's programme-masked decoder is an option if dimension = programme is wanted (a large change).

---

## 4. Ranking genes in general

**R1. Rank by effect size against an expression-matched floor.** high, cheap
- Rank by an effect size (variance share, unique-global R², Δloss in M3), not by p-values. Compare against random gene sets
  matched on mean and detection rate (ED3 plan). Select or normalise with analytic Pearson residuals so that variable means
  "biologically variable", not "highly expressed" [Lause et al. 2021]. On CosMx, set a NegPrb floor: a gene ranks only if it
  beats the negative probes' score (noted 2026-11-01).

**R2. Sender genes by in-silico deletion, with a conditional replacement.** high, medium cost
- Literature: Geneformer ranks genes by the embedding shift when a gene is deleted [Theodoris et al. 2023]. Later benchmarks
  show that such rankings mix direct and indirect effects and depend on the kind of perturbation [TEDDY 2025; CellxPert 2026].
  The HRT argues for replacing a feature with a draw from its conditional distribution rather than zero [Tansey et al. 2022].
- How: for sender gene j, replace its value in every visible context cell (optionally only beyond r) with a conditional
  draw, such as the mean of j over the cell's expression kNN. Read Δprediction on hidden cells per target gene g. The result
  is a sender × target matrix `D[j, g]`, optionally per distance band. Restrict j to the panel's ligands / receptors on CosMx
  to keep passes low. Compare with IG (a local linearisation) and with LR databases. Mind the caveat that indirect effects
  through the cell's type are included.

**R3. Validate a ranking by removal, not only by sanity checks.** medium, large
- Literature: ROAR removes the top-k features by an attribution, retrains, and compares the drop with random-k. Many
  attribution methods were no better than random [Hooker et al. 2019]. Pitfalls: redundant features, imputation bias
  [ROAD; arXiv 2304.13836].
- How: once on synth_spot, mask the top-k sender genes by IG vs by R2 vs random k from the input, retrain at 100 epochs, and
  read the programme score drop. One sweep, then trust the cheaper readouts.

**R4. Stability across seeds, folds and arms.** high, cheap
- Literature: rank-biased overlap (top-weighted, for lists that need not share items, p ≈ 0.9-0.95) [Webber et al. 2010].
  Robust rank aggregation: a parameter-free merged list with p-values per gene [Kolde et al. 2012]. Stability selection: keep
  what is selected in most resamples [Meinshausen & Bühlmann 2010, `[memory]`].
- How: report RBO between seeds and folds for every top-gene table (`gene_priority`, `dim_*`, M2, M3). Aggregate with RRA
  for the manuscript list, and state top-k membership frequency. CosMx folds train on different donors, so RBO between folds
  is also a donor-robustness read.

**R5. Modules instead of single genes.** medium, cheap
- Literature: Hotspot finds genes with local autocorrelation on a kNN graph and clusters them by pairwise local
  correlation [DeTomaso & Yosef 2021].
- How: group the top local and top global genes into modules at their scale, and make programme-level claims (CosMx islet
  vs exocrine). Combine per-scale tests with the Cauchy rule for one p-value per gene [FlashS 2026].

---

## Suggested order

| # | item | why first | cost |
|---|---|---|---|
| 1 | S1 + M1 (bands for data and parts) | one readout for data and model, replaces the I(k) fall | cheap |
| 2 | M2 (unique / shared R² per gene) | a defensible gene ranking from runs we already have | cheap |
| 3 | P1 (MISTy views at the reach / beyond) | model-free predictive bar per gene; liana-py | cheap |
| 4 | M3 (shift test beyond / within r) | the hypothesis test for global-only models; synth_spot check first | medium |
| 5 | R4 + R1 (RBO / RRA, matched floors) | makes every top-gene table trustworthy | cheap |
| 6 | R2 (deletion sender × target) | the communication ranking for CosMx / lung, against LR DBs (P5) | medium |
| 7 | S2 (Spacelink / nnSVG reference) and S4 (composition residuals) | real-data credibility | medium |
| 8 | M5, M4 (dimension / head scales → genes) | one-headed models get a split | cheap |
| 9 | M7, M8, M6 (training changes) | only if the readouts show the split is the problem | large |

---

## References

- Alexander-Bloch A. et al. On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage* 2018 `[memory: title]`.
- Arnol D., Schapiro D., Bodenmiller B., Saez-Rodriguez J., Stegle O. Modeling cell-cell interactions from spatial molecular data with spatial variance component analysis. *Cell Reports* 29:202 (2019). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6899515/ `[checked]`
- Birk S. et al. Quantitative characterization of cell niches in spatially resolved omics data (NicheCompass). *Nat Genet* (2025). https://www.nature.com/articles/s41588-025-02120-6 `[checked]`
- Cable D. et al. Cell type-specific inference of differential expression in spatial transcriptomics (C-SIDE). *Nat Methods* 2022. `[checked via review]`
- Celina: Statistical identification of cell type-specific spatially variable genes in spatial transcriptomics. *Nat Commun* (2025). https://www.nature.com/articles/s41467-025-56280-4 `[checked]`
- ctSVG benchmark: Benchmarking cell-type-specific spatially variable gene detection methods using a realistic and decomposable simulation framework. bioRxiv 2025.11.26.690782. `[checked]`
- Chang Y., Liu J. et al. Graph Fourier transform for spatial omics representation and analyses of complex organs (SpaGFT). *Nat Commun* (2024). https://www.nature.com/articles/s41467-024-51590-5 `[checked]`
- Chitra U. et al. Mapping the topography of spatial gene expression with interpretable deep learning (GASTON). *Nat Methods* (2024). https://www.nature.com/articles/s41592-024-02503-3 `[checked]`
- DeTomaso D., Yosef N. Hotspot identifies informative gene modules across modalities of single-cell genomics. *Cell Systems* 12:446 (2021). `[checked]`
- Dimitrov D. et al. LIANA+ provides an all-in-one framework for cell–cell communication inference. *Nat Cell Biol* (2024), doi 10.1038/s41556-024-01469-w; MISTy in liana-py: https://liana-py.readthedocs.io/en/latest/tutorials/notebooks/misty.html `[checked]`
- Dong M., Su D.G., Kluger H., Fan R., Kluger Y. SIMVI disentangles intrinsic and spatial-induced cellular states in spatial omics data. *Nat Commun* 16:2990 (2025). https://www.nature.com/articles/s41467-025-58089-7 `[checked: regulariser, double-ML spatial effect]`
- Dray S., Legendre P., Peres-Neto P. Spatial modelling: a comprehensive framework for principal coordinate analysis of neighbour matrices (PCNM). *Ecol Model* 196:483 (2006); adespatial tutorial https://cran.r-project.org/web/packages/adespatial/vignettes/tutorial.html `[checked]`
- Fischer D.S., Schaar A.C., Theis F.J. Modeling intercellular communication in tissues using spatial graphs of cells (NCEM). *Nat Biotechnol* (2022). https://www.nature.com/articles/s41587-022-01467-z `[checked]`
- FlashS: Multi-scale spatial testing recovers gene programs missed by existing detection methods. bioRxiv 10.64898/2026.03.12.711372 (2026). `[checked]`
- Fotheringham A.S., Yang W., Kang W. Multiscale geographically weighted regression (MGWR). *Ann Am Assoc Geogr* 107:1247 (2017). `[checked]`
- Grömping U. Relative importance for linear regression in R: the package relaimpo. *J Stat Softw* 17 (2006). `[checked]`
- Haviv D. et al. The covariance environment defines cellular niches for spatial inference (COVET / ENVI). *Nat Biotechnol* 43:269 (2025; online 2024). `[checked]`
- Hooker S., Erhan D., Kindermans P.-J., Kim B. A benchmark for interpretability methods in deep neural networks (ROAR). NeurIPS 2019. `[checked]`
- Jin S. et al. CellChat for systematic analysis of cell-cell communication from single-cell and spatially resolved transcriptomics. bioRxiv 2023.11.05.565674 (CellChat v2). `[checked]`
- Kats I., Vento-Tormo R., Stegle O. SpatialDE2: fast and localized variance component analysis of spatial transcriptomics. bioRxiv 2021.10.27.466045. `[checked]`
- Kolde R., Laur S., Adler P., Vilo J. Robust rank aggregation for gene list integration and meta-analysis. *Bioinformatics* 28:573 (2012). `[checked]`
- Lachapelle S., Mahajan D., Mitliagkas I., Lacoste-Julien S. Additive decoders for latent variables identification and Cartesian-product extrapolation. NeurIPS 2023 (oral). arXiv 2307.02598 `[checked]`
- Lause J., Berens P., Kobak D. Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data. *Genome Biol* 22:258 (2021). `[checked]`
- Lengerich B., Tan S., Chang C.-H., Hooker G., Caruana R. Purifying interaction effects with the functional ANOVA. AISTATS 2020. arXiv 1911.04974 `[checked]`
- Li Z. et al. (Pinello lab) Systematic benchmarking of computational methods to identify spatially variable genes. *Genome Biol* 26:285 (2025). https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03731-2 `[checked]`
- Li Z. et al. SpatialDM for rapid identification of spatially co-expressed ligand–receptor... *Nat Commun* (2023). `[checked via snippet]`
- Mason K. et al. Niche-DE: niche-differential gene expression analysis in spatial transcriptomics data identifies context-dependent cell-cell interactions. *Genome Biol* 25:14 (2024). `[checked]`
- Megas S. et al. Celcomen: spatial causal disentanglement for single-cell and tissue perturbation modeling. *Nat Commun* (2026); arXiv 2409.05804; ICLR 2025. `[checked]`
- Meinshausen N., Bühlmann P. Stability selection. *JRSS-B* 72:417 (2010). `[memory]`
- Peng X., Wei Y., Deng A., Wang D., Hu D. Balanced multimodal learning via on-the-fly gradient modulation (OGM-GE). CVPR 2022. `[checked]`
- Peres-Neto P., Legendre P., Dray S., Borcard D. Variation partitioning of species data matrices: estimation and comparison of fractions. *Ecology* 87:2614 (2006). `[checked via citation]`
- Spacelink: Mapping disease critical spatially variable gene programs by integrating spatial transcriptomics with human genetics. bioRxiv 2025.09.24.678397; https://github.com/hanbyul-lee/spacelink `[checked]`
- Svensson V., Teichmann S., Stegle O. SpatialDE: identification of spatially variable genes. *Nat Methods* 15:343 (2018). `[checked]`
- Tanevski J., Ramirez Flores R.O., Gabor A., Schapiro D., Saez-Rodriguez J. Explainable multiview framework for dissecting spatial relationships from highly multiplexed data (MISTy). *Genome Biol* 23:97 (2022). `[checked]`
- Tansey W., Veitch V., Zhang H., Rabadan R., Blei D. The holdout randomization test for feature selection in black box models. *JCGS* 31:151 (2022). `[checked]`
- Theodoris C.V. et al. Transfer learning enables predictions in network biology (Geneformer). *Nature* 618:616 (2023). `[checked]`
- Townes F.W., Engelhardt B.E. Nonnegative spatial factorization applied to spatial genomics (NSF / NSFH). *Nat Methods* 20:229 (2023). `[checked]`
- Velten B. et al. Identifying temporal and spatial patterns of variation from multimodal data using MEFISTO. *Nat Methods* 19:179 (2022). `[checked]`
- Wagner H.H., Dray S. Generating spatially constrained null models for irregularly spaced data using Moran spectral randomization methods. *Methods Ecol Evol* 6:1169 (2015). `[checked]`
- Wang J. et al. Dimension-agnostic and granularity-based spatially variable gene identification using BSP. *Nat Commun* 14:7367 (2023). `[checked]`
- Webber W., Moffat A., Zobel J. A similarity measure for indefinite rankings (RBO). *ACM TOIS* 28(4):20 (2010). `[checked]`
- Weber L.M. et al. nnSVG for the scalable identification of spatially variable genes using nearest-neighbor Gaussian processes. *Nat Commun* 14:4059 (2023). `[checked]`
- Wu N., Jastrzębski S., Cho K., Geras K. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. ICML 2022. `[memory: authors]`

Not read (title only): "SORT disentangles multilevel gene programs across heterogeneous spatial transcriptomic atlases", bioRxiv
10.64898/2026.10.04.756488 (the PDF did not parse). Worth a look if multilevel means scales.
