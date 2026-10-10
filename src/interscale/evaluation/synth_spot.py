"""Generate ``synth_spot``, a spot-based synthetic dataset on a square 100 µm lattice.

Same purpose as ``synth_data_0`` -- every pattern a model is meant to find is written into the
object as ground truth -- but with the geometry of a spot-based assay instead of scattered cells:
spots sit on a regular lattice, 100 µm centre to centre, and every spatial programme is defined
in µm. It is the benchmark for the positional encodings (``.claude/PE_plan.md``), which differ in
exactly what this dataset varies: a regular lattice, its border and holes, and directions.

Layout (defaults): conditions A and B with 20 slides each. A slide is a 50 x 50 grid centred on
the origin with no spot on either axis -- rows and columns sit at ±50, ±150, ..., ±2450 µm. Five
slides per condition lose one border row or column (50 x 49), ten get 1-5 holes of 2x2, 3x3 or
3x5 spots. Everything is simulated on the spots that remain, so a receiver next to a hole never
responds to a sender that is not in the data. One slide is one graph and one transformer
sequence, so ``max_seq_len`` must be >= 2500. The split is per slide, 12/4/4 train/val/test in
each condition (60/20/20), and val and test each hold both values of every slide-level factor.

Expression levels are named: ``0`` is a structural zero, ``down`` a mean of 1, ``up`` and
``medium`` a mean of 5, ``high`` a mean of 10. Genes defined by such levels or by a spatial field
("structured" genes) scatter by about a count around their mean -- a ``down`` gene reads 1 about
two times in three and 0 or 2 otherwise. Background genes follow their own distributions.

Programmes, with the item of the original specification in brackets:

- condition [6]: cg_1, cg_2, cg_3 -- down, 0, down in A; high, up, high in B.
- batch [7]: bg_1..bg_3 -- one level from {0, down, up, high} per slide; the triple is unique.
- short range [8]: S1..S4 on random spots, R1..R4 respond in the 4 nearest neighbours.
- long range [9]: sources M1..M11 and responses MC1..MC15, from 200 µm to 4 mm.
- spatially variable [10]: SP1..SP10 -- one or several Gaussians, some elongated or skewed.
- highly variable [11]: HV1..HV10 -- 0-100 counts per spot, no spatial structure.
- background [12-15]: NB1..NB40, ZINB1..ZINB40, POI1..POI40 (1-10 spots per slide), H1..H40.
- region [17]: CM1..CM3 -- the slide halved along the x-axis, the y-axis or a diagonal.
- sparse response [18]: compact sources M12..M21; only 20-50 (or 200-500) random receiver spots
  respond, with MC16..MC25 set by their distance to the source in 1 mm steps or beyond a threshold,
  and mark themselves with a receptor, MR16..MR25.

The rules in detail sit next to the code that implements them. Coordinates are ``(x, y)`` in µm
with y growing with ``array_row``, as in a plain scatter plot; image-based spatial plots draw y
downwards, which flips "+y" and "-y" on screen.

Run as a script::

    python src/interscale/evaluation/synth_spot.py --out data/synth_spot.h5ad

which also writes maps of every structured gene on one slide next to the file
(``synth_spot_maps_<slide>_expected.png`` and ``..._counts.png``; ``--no-plot`` skips them). The
same figure is :func:`plot_slide_maps`, for any slide and layer.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from scipy.special import ndtr

SPACING_UM = 100.0
CONDITIONS = ("A", "B")

#: Mean count of each named level. "0" is a structural zero, so such a gene reads exactly 0.
#: "medium" appears only as the R3/R4 default, between 0 and "high".
LEVELS = {"0": 0.0, "down": 1.0, "medium": 5.0, "up": 5.0, "high": 10.0}

#: Condition genes [6]: level in condition A, level in condition B. Constant within a slide.
CONDITION_GENES = {"cg_1": ("down", "high"), "cg_2": ("0", "up"), "cg_3": ("down", "high")}

#: Batch genes [7]. Each slide draws one level per gene, and no two slides share the triple.
BATCH_GENES = ("bg_1", "bg_2", "bg_3")
BATCH_LEVELS = ("0", "down", "up", "high")

#: Fraction of spots carrying each short-range sender [8]. The S3, S4 and S3+S4 entries are
#: disjoint: co-expression is a rule of its own (8v), and two independent 5% draws would make it a
#: 0.25% event -- about six spots per slide, too few to learn from.
SENDER_RATES = {"S1": 0.05, "S2": 0.05, "S3": 0.04, "S4": 0.04, "S3+S4": 0.02}

#: Geometry of the long-range programmes [9]. Lengths in µm; the full slide spans ±2450.
LONG_RANGE = {
    "M1_sigma": 700.0,  # (i) broad cluster
    "M2_sigma": 350.0,  # (ii), (iv), and the sources of (vii)/(viii): a smaller cluster
    "M3_sigma": 1200.0,  # (iii), (v): wide, and only "up" at the peak, i.e. flat
    "M6_sigma": 200.0,  # (vi) several small clusters
    "M6_n_clusters": (3, 5),
    "M7_bands": ((0.0, 1000.0), (1000.0, 2000.0), (2000.0, 3000.0)),  # MC7, MC8, MC9
    "M8_bands": ((0.0, 2000.0), (2000.0, 4000.0)),  # MC10, MC11
    "M9_block": (2, 3),  # rows and columns of the M9 block are each drawn from this range
    "M9_plateau": 300.0,
    "M9_decay": 800.0,
    "M10_half_angle_deg": 22.5,  # a 45° cone
    "M10_plateau": 200.0,
    "M10_decay": 600.0,
    "M11_n_sources": (3, 6),
    "M11_range": 500.0,
    "M11_steepness": 3.0,
    "M11_min_separation": 1000.0,
}

#: Spatially variable genes [10]: (number of Gaussians, major-axis sigma in µm, aspect ratio,
#: skewness). The shape is fixed per gene so that a gene means the same thing on every slide;
#: positions, orientation and the direction of the skew are redrawn per slide.
SP_GENES = {
    "SP1": (1, 900.0, 1.0, 0.0),  # one broad, round blob
    "SP2": (1, 700.0, 2.0, 0.0),  # elongated
    "SP3": (1, 600.0, 1.5, 4.0),  # elongated and skewed
    "SP4": (1, 1000.0, 3.0, 0.0),  # a long ridge
    "SP5": (1, 500.0, 1.0, 6.0),  # round but skewed
    "SP6": (2, 350.0, 1.0, 0.0),
    "SP7": (3, 300.0, 2.0, 0.0),
    "SP8": (3, 250.0, 1.5, 4.0),
    "SP9": (4, 200.0, 2.5, 0.0),
    "SP10": (5, 180.0, 1.0, 5.0),
}

#: Holes [16], in spots; 3x5 is laid either way round.
HOLE_SHAPES = ((2, 2), (3, 3), (3, 5))

#: Background genes [11-15]. Per-gene parameters are drawn once for the whole dataset, so a gene
#: means the same thing on every slide.
HV_MAX_COUNT = 100  # HV counts are BetaBinomial(100, a, b) with a, b ~ U(0.5, 2) per gene
NB_MEAN_RANGE = (1.0, 10.0)
ZINB_MEAN_RANGE = (1.0, 50.0)
ZINB_DROPOUT_RANGE = (0.1, 0.5)
THETA_RANGE = (1.0, 10.0)  # NB and ZINB inverse dispersion, log-uniform per gene
POI_N_SPOTS = (1, 10)
POI_MEAN = 5.0
HK_MEAN_RANGE = (5.0, 10.0)


@dataclass(frozen=True)
class _SparseProgramme:
    """One sparse-response programme [18]: its source, its receivers, and their response."""

    source: str  # M gene, expressed by the source clusters
    response: str  # MC gene, expressed by receivers only
    n_clusters: tuple[int, int]  # source clusters per slide, inclusive range
    n_receivers: tuple[int, int]  # receiver spots per slide, inclusive range
    start: float  # mean response in the first band, 0-1 mm from the nearest cluster
    step: float  # change of the mean per further 1 mm band; never below 0
    threshold: float = 0.0  # receivers closer than this to the nearest cluster stay at 0

    @property
    def receptor(self) -> str:
        """The receptor gene, expressed by the receivers and nobody else."""
        return "MR" + self.response.removeprefix("MC")

    @property
    def density(self) -> str:
        """``sparse`` (tens of receivers per slide) or ``dense`` (hundreds)."""
        return "dense" if self.n_receivers[0] >= 100 else "sparse"


#: Sparse-response programmes [18], in the order of the specification. The specification names
#: both (ii) and (iii) MC17; here (iii) is MC18 and every later response moves up by one, so (viii)
#: is "like MC20", (ix) "like MC21" and (x) "like MC22".
SPARSE = (
    _SparseProgramme("M12", "MC16", (1, 1), (20, 50), start=0.0, step=5.0),  # (i)
    _SparseProgramme("M13", "MC17", (1, 3), (20, 50), start=0.0, step=5.0),  # (ii)
    _SparseProgramme("M14", "MC18", (1, 1), (20, 50), start=0.0, step=20.0),  # (iii)
    _SparseProgramme("M15", "MC19", (1, 1), (20, 50), start=100.0, step=-20.0),  # (iv)
    _SparseProgramme("M16", "MC20", (1, 1), (20, 50), start=5.0, step=0.0, threshold=2500.0),  # (v)
    _SparseProgramme("M17", "MC21", (1, 1), (20, 50), start=25.0, step=0.0, threshold=3000.0),  # (vi)
    _SparseProgramme("M18", "MC22", (1, 3), (20, 50), start=10.0, step=0.0, threshold=1000.0),  # (vii)
    _SparseProgramme("M19", "MC23", (1, 1), (200, 500), start=5.0, step=0.0, threshold=2500.0),  # (viii)
    _SparseProgramme("M20", "MC24", (1, 1), (200, 500), start=25.0, step=0.0, threshold=3000.0),  # (ix)
    _SparseProgramme("M21", "MC25", (1, 3), (200, 500), start=10.0, step=0.0, threshold=1000.0),  # (x)
)
#: Source clusters are compact on purpose: M is down to ~0.1 counts at 600 µm, so a receiver past
#: the first band has no trace of the source at its own spot or its neighbours'.
SPARSE_SIGMA = 200.0
SPARSE_BAND = 1000.0
SPARSE_MIN_SEPARATION = 1000.0  # between the clusters of one programme


def _rng(seed: int, *keys: str) -> np.random.Generator:
    """A generator of its own for every (slide, programme) or (slide, gene) key.

    Keyed rather than one shared stream, so adding a programme later -- or changing how many draws
    one makes -- leaves every other gene in the dataset unchanged.
    """
    return np.random.default_rng([seed, *(zlib.crc32(k.encode()) for k in keys)])


# ------------------------------------------------------------------------------------------------
# Gene table
# ------------------------------------------------------------------------------------------------


def _gene_table(spacing: float, n_hv: int, n_nb: int, n_zinb: int, n_poi: int, n_hk: int) -> pd.DataFrame:
    """One row per gene: programme, role, which genes it interacts with, and its length scale.

    ``length_scale_um`` is the distance over which the gene's expected value changes under its
    rule: sigma for a Gaussian, the outer edge of a ring, plateau + decay length for a directional
    response, the range for M11, one spot for the short-range programmes, the 1 mm band or the
    threshold for a sparse response. NaN for genes without spatial structure, for single-spot
    sources and for receptors. ``slides`` says where a programme's source exists.
    """
    lr = LONG_RANGE
    rows = []

    def add(gene, program, role, noise_model, interaction="", length=np.nan, slides="all"):
        rows.append(
            {
                "gene": gene,
                "program": program,
                "role": role,
                "interaction": interaction,
                "length_scale_um": float(length),
                "slides": slides,
                "noise_model": noise_model,
            }
        )

    for gene in CONDITION_GENES:
        add(gene, "condition", "condition", "low_noise")
    for gene in BATCH_GENES:
        add(gene, "batch", "batch", "low_noise")

    for s, r in (("S1", "R1"), ("S2", "R2")):
        add(s, "short_range", "sender", "low_noise", f"{s}>{r}", spacing)
        add(r, "short_range", "receiver", "low_noise", f"{s}>{r}", spacing)
    for gene in ("S3", "S4"):
        add(gene, "short_range", "sender", "low_noise", "S3/S4>R3/R4", spacing)
    for gene in ("R3", "R4"):
        add(gene, "short_range", "receiver", "low_noise", "S3/S4>R3/R4", spacing)

    gaussian = (
        ("M1", "MC1", lr["M1_sigma"], "all"),
        ("M2", "MC2", lr["M2_sigma"], "all"),
        ("M3", "MC3", lr["M3_sigma"], "all"),
        ("M4", "MC4", lr["M2_sigma"], "every_2nd"),
        ("M5", "MC5", lr["M3_sigma"], "every_3rd"),
        ("M6", "MC6", lr["M6_sigma"], "all"),
    )
    for m, mc, sigma, slides in gaussian:
        add(m, "long_range", "source", "low_noise", f"{m}>{mc}", sigma, slides)
        add(mc, "long_range", "response", "low_noise", f"{m}>{mc}", sigma, slides)
    for m, responses, bands in (
        ("M7", ("MC7", "MC8", "MC9"), lr["M7_bands"]),
        ("M8", ("MC10", "MC11"), lr["M8_bands"]),
    ):
        label = f"{m}>{'/'.join(responses)}"
        add(m, "long_range", "source", "low_noise", label, lr["M2_sigma"])
        for gene, (_, outer) in zip(responses, bands, strict=True):
            add(gene, "long_range", "response", "low_noise", label, outer)
    add("M9", "long_range", "source", "low_noise", "M9>MC12")
    add("MC12", "long_range", "response", "low_noise", "M9>MC12", lr["M9_plateau"] + lr["M9_decay"])
    add("M10", "long_range", "source", "low_noise", "M10>MC13")
    add("MC13", "long_range", "response", "low_noise", "M10>MC13", lr["M10_plateau"] + lr["M10_decay"])
    add("M11", "long_range", "source", "low_noise", "M11>MC14/MC15")
    for gene in ("MC14", "MC15"):
        add(gene, "long_range", "response", "low_noise", "M11>MC14/MC15", lr["M11_range"])

    for gene, (_, sigma, _, _) in SP_GENES.items():
        add(gene, "spatially_variable", "spatial", "low_noise", length=sigma)
    for i in range(n_hv):
        add(f"HV{i + 1}", "highly_variable", "noise", "beta_binomial")
    for i in range(n_nb):
        add(f"NB{i + 1}", "background_nb", "noise", "negative_binomial")
    for i in range(n_zinb):
        add(f"ZINB{i + 1}", "background_zinb", "noise", "zero_inflated_nb")
    for i in range(n_poi):
        add(f"POI{i + 1}", "lowly_expressed", "noise", "sparse_poisson")
    for i in range(n_hk):
        add(f"H{i + 1}", "housekeeping", "noise", "poisson")
    for gene in ("CM1", "CM2", "CM3"):
        add(gene, "region_marker", "marker", "low_noise")
    for p in SPARSE:
        label = f"{p.source}>{p.response}"
        add(p.source, "sparse_response", "source", "low_noise", label, SPARSE_SIGMA)
        add(p.response, "sparse_response", "response", "low_noise", label, p.threshold or SPARSE_BAND)
        add(p.receptor, "sparse_response", "receptor", "low_noise", label)

    genes = pd.DataFrame(rows).set_index("gene")
    genes.index.name = None
    genes["is_spatial"] = genes["program"].isin(
        ["short_range", "long_range", "spatially_variable", "region_marker", "sparse_response"]
    )
    return genes


def _draw_gene_params(seed: int, genes: pd.DataFrame) -> None:
    """Draw the per-gene parameters of the background genes, in place, once for all slides.

    ``param_mean`` is the expected count of every background gene: the NB mean for NB and H, the
    mean after dropout for ZINB, the mean of an expressing spot for POI and ``100 a / (a + b)``
    for HV.
    """
    for col in ("param_mean", "param_theta", "param_dropout", "param_beta_a", "param_beta_b"):
        genes[col] = np.nan
    log_theta = np.log(THETA_RANGE)
    for gene, model in genes["noise_model"].items():
        r = _rng(seed, "gene", gene)
        if model == "beta_binomial":
            a, b = r.uniform(0.5, 2.0, size=2)
            genes.loc[gene, ["param_beta_a", "param_beta_b", "param_mean"]] = a, b, HV_MAX_COUNT * a / (a + b)
        elif model == "negative_binomial":
            genes.loc[gene, "param_mean"] = r.uniform(*NB_MEAN_RANGE)
            genes.loc[gene, "param_theta"] = np.exp(r.uniform(*log_theta))
        elif model == "zero_inflated_nb":
            mu, pi = r.uniform(*ZINB_MEAN_RANGE), r.uniform(*ZINB_DROPOUT_RANGE)
            genes.loc[gene, "param_theta"] = np.exp(r.uniform(*log_theta))
            genes.loc[gene, ["param_dropout", "param_mean"]] = pi, (1.0 - pi) * mu
        elif model == "sparse_poisson":
            genes.loc[gene, "param_mean"] = POI_MEAN
        elif model == "poisson":
            genes.loc[gene, "param_mean"] = r.uniform(*HK_MEAN_RANGE)


# ------------------------------------------------------------------------------------------------
# Slide design
# ------------------------------------------------------------------------------------------------


def _design(
    seed: int,
    n_slides_per_condition: int,
    n_cropped: int,
    n_with_holes: int,
    n_val: int,
    n_test: int,
) -> pd.DataFrame:
    """Per-slide flags: which programmes occur, which slides are damaged, batch levels, split."""
    rows = []
    for cond in CONDITIONS:
        r = _rng(seed, "design", cond)
        cropped = set(r.choice(n_slides_per_condition, n_cropped, replace=False).tolist())
        holed = set(r.choice(n_slides_per_condition, n_with_holes, replace=False).tolist())
        for i in range(n_slides_per_condition):
            k = i + 1
            rows.append(
                {
                    "slide": f"{cond}_{k:02d}",
                    "condition": cond,
                    "number": k,
                    # Counted within a condition, so both conditions carry M4 and M5 equally often
                    # and neither programme is confounded with the condition.
                    "has_M4": k % 2 == 0,
                    "has_M5": k % 3 == 0,
                    "cropped": i in cropped,
                    "has_holes": i in holed,
                }
            )
    design = pd.DataFrame(rows)

    combos = list(itertools.product(BATCH_LEVELS, repeat=len(BATCH_GENES)))
    if len(design) > len(combos):
        raise ValueError(f"{len(design)} slides cannot each get a unique batch triple; only {len(combos)} exist")
    pick = _rng(seed, "design", "batch").choice(len(combos), len(design), replace=False)
    for j, gene in enumerate(BATCH_GENES):
        design[gene] = [combos[p][j] for p in pick]

    design["split"] = _assign_split(_rng(seed, "design", "split"), design, n_val, n_test)
    return design


def _assign_split(r: np.random.Generator, design: pd.DataFrame, n_val: int, n_test: int, tries: int = 500):
    """Split per slide, stratified by condition, balanced over the slide-level factors.

    One slide is one graph, so the split has to be per slide. Among ``tries`` stratified draws the
    first one in which every split sees both values of each factor wins (or the best one, when the
    design is too small for that): a val set without an M4 slide could not ask about M4 at all.
    """
    factors = ["has_M4", "has_M5", "cropped", "has_holes"]
    best, best_score = None, -1
    for _ in range(tries):
        split = pd.Series("train", index=design.index, dtype=object)
        for _, sub in design.groupby("condition"):
            perm = r.permutation(sub.index.to_numpy())
            split.loc[perm[:n_val]] = "val"
            split.loc[perm[n_val : n_val + n_test]] = "test"
        score = sum(design.loc[split == s, f].nunique() for s in ("train", "val", "test") for f in factors)
        if score > best_score:
            best, best_score = split, score
        if best_score == 3 * 2 * len(factors):
            break
    return best


# ------------------------------------------------------------------------------------------------
# Lattice
# ------------------------------------------------------------------------------------------------


@dataclass
class _Grid:
    """The spots of one slide that survived cropping and holes."""

    pos: np.ndarray  # [n, 2] (x, y) in µm
    row: np.ndarray  # [n] lattice row, grows with y
    col: np.ndarray  # [n] lattice column, grows with x
    nbr: np.ndarray  # [n, 4] index of the 4 nearest neighbours, -1 where there is none
    index_map: np.ndarray  # [n_side, n_side] spot index at (row, col), -1 where removed
    half: float  # half the side of the full lattice, centre to centre


def _crop(r: np.random.Generator, n_side: int, n_lines: int) -> tuple[np.ndarray, str]:
    """Mask of kept lattice positions after cutting ``n_lines`` border lines off one side."""
    keep = np.ones((n_side, n_side), dtype=bool)
    if not n_lines:
        return keep, ""
    side = str(r.choice(["left", "right", "bottom", "top"]))
    if side == "left":
        keep[:, :n_lines] = False
    elif side == "right":
        keep[:, -n_lines:] = False
    elif side == "bottom":
        keep[:n_lines, :] = False
    else:
        keep[-n_lines:, :] = False
    return keep, side


def _punch_holes(
    r: np.random.Generator, keep: np.ndarray, n_holes: int, margin: int = 2, gap: int = 1, tries: int = 1000
) -> list[tuple[int, int, int, int]]:
    """Remove ``n_holes`` rectangular holes from ``keep``, in place; return ``(row, col, h, w)``.

    Holes stay ``margin`` spots inside the border and ``gap`` spots apart, so each one is a real
    hole -- enclosed by spots on every side -- rather than a notch in the border or a merger of two.
    """
    rows, cols = np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0))
    r_lo, r_hi = rows[0] + margin, rows[-1] - margin
    c_lo, c_hi = cols[0] + margin, cols[-1] - margin
    taken = np.zeros_like(keep)
    holes = []
    for _ in range(n_holes):
        for _ in range(tries):
            h, w = HOLE_SHAPES[r.integers(len(HOLE_SHAPES))]
            if r.random() < 0.5:
                h, w = w, h
            if r_hi - r_lo + 1 < h or c_hi - c_lo + 1 < w:
                continue
            r0 = int(r.integers(r_lo, r_hi - h + 2))
            c0 = int(r.integers(c_lo, c_hi - w + 2))
            if taken[r0 : r0 + h, c0 : c0 + w].any():
                continue
            keep[r0 : r0 + h, c0 : c0 + w] = False
            taken[max(r0 - gap, 0) : r0 + h + gap, max(c0 - gap, 0) : c0 + w + gap] = True
            holes.append((r0, c0, h, w))
            break
    return holes


def _make_grid(keep: np.ndarray, spacing: float) -> _Grid:
    """Coordinates and 4-neighbourhood of the kept lattice positions, in row-major order."""
    n_side = keep.shape[0]
    row, col = np.nonzero(keep)
    offset = (n_side - 1) / 2
    pos = np.column_stack([(col - offset) * spacing, (row - offset) * spacing])
    index_map = np.full(keep.shape, -1, dtype=np.int64)
    index_map[row, col] = np.arange(len(row))
    nbr = np.full((len(row), 4), -1, dtype=np.int64)
    for k, (dr, dc) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        rr, cc = row + dr, col + dc
        inside = (rr >= 0) & (rr < n_side) & (cc >= 0) & (cc < n_side)
        nbr[inside, k] = index_map[rr[inside], cc[inside]]
    return _Grid(pos=pos, row=row, col=col, nbr=nbr, index_map=index_map, half=offset * spacing)


# ------------------------------------------------------------------------------------------------
# Shapes
# ------------------------------------------------------------------------------------------------


def _gauss(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (d / sigma) ** 2)


def _band(d: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """A sin² bump: 0 at both edges of ``[inner, outer]`` and outside it, 1 in the middle."""
    t = (d - inner) / (outer - inner)
    return np.where((t >= 0) & (t <= 1), np.sin(np.pi * t) ** 2, 0.0)


def _plateau_decay(d: np.ndarray, plateau: float, decay: float) -> np.ndarray:
    """1 up to ``plateau``, then exponential decay with length ``decay``."""
    return np.exp(-np.maximum(d - plateau, 0.0) / decay)


@lru_cache
def _skew_peak(skew: float) -> float:
    """Peak of ``exp(-z²/2) Φ(skew z)``, which normalises a skewed Gaussian to a peak of 1."""
    z = np.linspace(-6.0, 6.0, 120_001)
    return float(np.max(np.exp(-0.5 * z**2) * ndtr(skew * z)))


def _uniform_point(r: np.random.Generator, half: float, frac: float) -> np.ndarray:
    return r.uniform(-frac * half, frac * half, size=2)


def _spread_points(r: np.random.Generator, n: int, half: float, frac: float, min_sep: float) -> np.ndarray:
    """``n`` points in the central ``frac`` of the slide, ``min_sep`` apart where possible."""
    pts: list[np.ndarray] = []
    for _ in range(2000):
        if len(pts) == n:
            break
        p = _uniform_point(r, half, frac)
        if all(np.hypot(*(p - q)) >= min_sep for q in pts):
            pts.append(p)
    while len(pts) < n:  # no room for the separation; place the rest anyway rather than fail
        pts.append(_uniform_point(r, half, frac))
    return np.array(pts)


def _spread_spots(r: np.random.Generator, grid: _Grid, n: int, frac: float, min_sep: float) -> np.ndarray:
    """Indices of ``n`` spots in the central ``frac`` of the slide, ``min_sep`` apart."""
    eligible = np.flatnonzero((np.abs(grid.pos) <= frac * grid.half).all(1))
    chosen: list[int] = []
    for _ in range(2000):
        if len(chosen) == n:
            break
        i = int(r.choice(eligible))
        if all(np.hypot(*(grid.pos[i] - grid.pos[j])) >= min_sep for j in chosen):
            chosen.append(i)
    return np.array(chosen, dtype=np.int64)


def _pairs(senders, receivers, weights, range_class: str, program: str) -> pd.DataFrame:
    """Every sender paired with every receiver; ``weights`` holds one value per receiver."""
    s = np.repeat(senders, len(receivers))
    rcv = np.tile(receivers, len(senders))
    w = np.tile(weights, len(senders))
    keep = s != rcv
    return pd.DataFrame(
        {"sender": s[keep], "receiver": rcv[keep], "weight": w[keep], "range_class": range_class, "program": program}
    )


# ------------------------------------------------------------------------------------------------
# Programmes
# ------------------------------------------------------------------------------------------------


def _any_neighbour(nbr: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """Whether any of a spot's (up to four) nearest neighbours carries ``flag``."""
    has = nbr >= 0
    return (has & flag[np.where(has, nbr, 0)]).any(axis=1)


def _short_range(r: np.random.Generator, grid: _Grid, rates: dict[str, float]):
    """Item 8: senders on random spots, responses in their 4 nearest neighbours.

    - R1 is "up" next to an S1 spot and 0 everywhere else.
    - R2 is 0 next to an S2 spot and "high" everywhere else.
    - R3/R4 follow the S3 and S4 signals a spot receives: S3 alone gives high/0, S4 alone 0/high,
      and both -- whether from one spot expressing S3 and S4 (8v) or from two different
      neighbours -- cancel to the default medium/medium, as does neither.

    A response is all-or-nothing: two S1 neighbours raise R1 no further than one.
    """
    n = len(grid.pos)
    s1 = r.random(n) < rates["S1"]
    s2 = r.random(n) < rates["S2"]
    u = r.random(n)
    a, b, c = rates["S3"], rates["S4"], rates["S3+S4"]
    s3 = (u < a) | ((u >= a + b) & (u < a + b + c))
    s4 = (u >= a) & (u < a + b + c)

    e1, e2, e3, e4 = (_any_neighbour(grid.nbr, f) for f in (s1, s2, s3, s4))
    only3, only4 = e3 & ~e4, e4 & ~e3
    up, high, medium = LEVELS["up"], LEVELS["high"], LEVELS["medium"]
    mean = {
        "S1": high * s1,
        "R1": np.where(e1, up, 0.0),
        "S2": high * s2,
        "R2": np.where(e2, 0.0, high),
        "S3": high * s3,
        "S4": high * s4,
        "R3": np.select([only3, only4], [high, 0.0], default=medium),
        "R4": np.select([only3, only4], [0.0, high], default=medium),
    }
    obs = {
        "is_S1": s1,
        "is_S2": s2,
        "is_S3": s3,
        "is_S4": s4,
        "exposed_S1": e1,
        "exposed_S2": e2,
        "exposed_S3": e3,
        "exposed_S4": e4,
        "R34_state": np.select([only3, only4, e3 & e4], ["S3", "S4", "both"], default="none"),
    }

    # Every sender -> nearest-neighbour pair. A co-expressing S3+S4 sender is listed under both
    # genes: its neighbours sit at the default, but predicting that requires seeing both signals.
    frames = []
    for flag, program in ((s1, "S1>R1"), (s2, "S2>R2"), (s3, "S3>R3/R4"), (s4, "S4>R3/R4")):
        snd = np.flatnonzero(flag)
        rcv = grid.nbr[snd]
        ok = rcv >= 0
        frames.append(
            pd.DataFrame(
                {
                    "sender": np.repeat(snd, 4)[ok.ravel()],
                    "receiver": rcv[ok],
                    "weight": 1.0,
                    "range_class": "short",
                    "program": program,
                }
            )
        )
    return mean, obs, frames


def _long_range(seed: int, slide: str, grid: _Grid, has_m4: bool, has_m5: bool, cutoff: float):
    """Item 9: morphogen sources M1..M11 and their responses MC1..MC15.

    Each programme draws from its own generator. Sources are Gaussian clusters with a continuous
    centre (i-viii) or single spots (x-xii); every response is a function of position relative to
    its source, so it is defined on every spot, holes or not.
    """
    lr = LONG_RANGE
    high, up, down = LEVELS["high"], LEVELS["up"], LEVELS["down"]
    n = len(grid.pos)
    idx = np.arange(n)
    mean: dict[str, np.ndarray] = {}
    obs: dict[str, np.ndarray] = {}
    sources: list[dict] = []
    edges: list[pd.DataFrame] = []

    def dist(c):
        return np.hypot(*(grid.pos - c).T)

    def cluster(name, sigma, frac):
        c = _uniform_point(_rng(seed, slide, name), grid.half, frac)
        sources.append({"program": name, "x": c[0], "y": c[1], "sigma": sigma})
        obs[f"dist_{name}"] = dist(c)
        return _gauss(obs[f"dist_{name}"], sigma)

    # (i) M1 peaks at "high" and never falls below "down", so every spot expresses it. MC1 mirrors
    # it: "down" -- not 0 -- under the M1 peak, "high" where M1 has fallen off.
    g = cluster("M1", lr["M1_sigma"], 0.6)
    mean["M1"] = down + (high - down) * g
    mean["MC1"] = high - (high - down) * g

    # (ii) A smaller cluster. M2 decays to 0; MC2 rises from 0 at the M2 peak to "high".
    g = cluster("M2", lr["M2_sigma"], 0.75)
    mean["M2"], mean["MC2"] = high * g, high * (1 - g)

    # (iii) Flat: a wide cluster peaking at only "up", MC3 rising from 0 to "up" just as slowly.
    g = cluster("M3", lr["M3_sigma"], 0.5)
    mean["M3"], mean["MC3"] = up * g, up * (1 - g)

    # (iv), (v) Copies of (ii) and (iii) on every 2nd and every 3rd slide. The response is a
    # function of the local M level alone, so where there is no M at all nothing holds it down:
    # MC4 sits at its plateau over the whole slide, and so does MC5.
    for m, mc, present, sigma, peak, frac in (
        ("M4", "MC4", has_m4, lr["M2_sigma"], high, 0.75),
        ("M5", "MC5", has_m5, lr["M3_sigma"], up, 0.5),
    ):
        if present:
            g = cluster(m, sigma, frac)
        else:
            g = np.zeros(n)
            obs[f"dist_{m}"] = np.full(n, np.nan)
        mean[m], mean[mc] = peak * g, peak * (1 - g)

    # (vi) Several small clusters. Combined as a union, 1 - prod(1 - g), not a sum, so overlapping
    # clusters still peak at "high" and MC6 still bottoms out at 0.
    r = _rng(seed, slide, "M6")
    k = int(r.integers(lr["M6_n_clusters"][0], lr["M6_n_clusters"][1] + 1))
    centres = _spread_points(r, k, grid.half, 0.85, 4 * lr["M6_sigma"])
    ds = np.stack([dist(c) for c in centres])
    g = 1 - np.prod(1 - _gauss(ds, lr["M6_sigma"]), axis=0)
    mean["M6"], mean["MC6"] = high * g, high * (1 - g)
    obs["dist_M6"] = ds.min(0)
    sources += [{"program": "M6", "x": c[0], "y": c[1], "sigma": lr["M6_sigma"]} for c in centres]

    # (vii), (viii) Pathway responses: a cluster like (ii), and its responses switched on one after
    # another in rings around it as M falls off -- each a sharp sin² rise and fall from 0 to "high"
    # and back to 0 across its band, so neighbouring rings meet at 0 rather than overlap. The rings
    # are defined by distance: beyond ~1 mm M itself is all but 0 and carries no information.
    for m, responses, bands in (
        ("M7", ("MC7", "MC8", "MC9"), lr["M7_bands"]),
        ("M8", ("MC10", "MC11"), lr["M8_bands"]),
    ):
        mean[m] = high * cluster(m, lr["M2_sigma"], 0.5)
        for gene, (inner, outer) in zip(responses, bands, strict=True):
            mean[gene] = high * _band(obs[f"dist_{m}"], inner, outer)

    # (x) A small block of spots, all at "high", sends MC12 along its own rows to the left and the
    # right: "high" for the first few spots past the block, then a slow exponential decay to 0.
    # Spots outside those rows are untouched, however close.
    r0, c0, h, w = _m9_block(_rng(seed, slide, "M9"), grid)
    in_rows = (grid.row >= r0) & (grid.row < r0 + h)
    in_block = in_rows & (grid.col >= c0) & (grid.col < c0 + w)
    x = grid.pos[:, 0]
    x_lo, x_hi = x[in_block].min(), x[in_block].max()
    dx = np.maximum(np.maximum(x_lo - x, x - x_hi), 0.0)
    kern = np.where(in_rows, _plateau_decay(dx, lr["M9_plateau"], lr["M9_decay"]), 0.0)
    mean["M9"], mean["MC12"] = high * in_block, high * kern
    obs["in_M9_stripe"] = in_rows
    obs["dist_M9"] = np.where(in_rows, dx, np.nan)
    block = np.flatnonzero(in_block)
    sources += [{"program": "M9", "x": grid.pos[i, 0], "y": grid.pos[i, 1], "sigma": np.nan} for i in block]
    rcv = np.flatnonzero((kern >= cutoff) & ~in_block)
    edges.append(_pairs(block, rcv, kern[rcv], "directional", "M9>MC12"))

    # (xi) A single spot. Inside a 45° cone opening towards +y or -y, MC13 rises from its "down"
    # baseline to "high" near the source and decays back to the baseline with distance. The source
    # sits in the half the cone points away from, so the cone has at least ~3 mm to run.
    r = _rng(seed, slide, "M10")
    sign = float(r.choice([-1.0, 1.0]))
    y = grid.pos[:, 1]
    away = -sign * y
    eligible = np.flatnonzero((np.abs(x) <= 0.6 * grid.half) & (away >= 0.2 * grid.half) & (away <= 0.8 * grid.half))
    s = int(r.choice(eligible if len(eligible) else idx))
    v = grid.pos - grid.pos[s]
    d = np.hypot(*v.T)
    cos = np.divide(sign * v[:, 1], d, out=np.zeros(n), where=d > 0)
    in_cone = cos >= np.cos(np.radians(lr["M10_half_angle_deg"]))
    kern = np.where(in_cone, _plateau_decay(d, lr["M10_plateau"], lr["M10_decay"]), 0.0)
    mean["M10"] = high * (idx == s)
    mean["MC13"] = down + (high - down) * kern
    obs["in_M10_cone"] = in_cone
    obs["dist_M10"] = d
    sources.append(
        {
            "program": "M10",
            "x": grid.pos[s, 0],
            "y": grid.pos[s, 1],
            "sigma": np.nan,
            "direction": "+y" if sign > 0 else "-y",
        }
    )
    rcv = np.flatnonzero(kern >= cutoff)
    edges.append(_pairs(np.array([s]), rcv, kern[rcv], "directional", "M10>MC13"))

    # (xii) Several single spots, at least 1 mm apart. Within 500 µm of the nearest one MC14 falls
    # exponentially from "high" to 0 and MC15 rises exponentially from 0 to "high"; beyond it they
    # stay at 0 and "high".
    r = _rng(seed, slide, "M11")
    k = int(r.integers(lr["M11_n_sources"][0], lr["M11_n_sources"][1] + 1))
    src = _spread_spots(r, grid, k, 0.85, lr["M11_min_separation"])
    ds = np.stack([dist(grid.pos[i]) for i in src])
    reach, steep = lr["M11_range"], lr["M11_steepness"]
    t = np.minimum(ds.min(0) / reach, 1.0)
    mean["M11"] = high * np.isin(idx, src)
    mean["MC14"] = high * (np.exp(-steep * t) - np.exp(-steep)) / (1 - np.exp(-steep))
    mean["MC15"] = high * np.expm1(steep * t) / np.expm1(steep)
    obs["dist_M11"] = ds.min(0)
    for j, i in enumerate(src):
        sources.append({"program": "M11", "x": grid.pos[i, 0], "y": grid.pos[i, 1], "sigma": np.nan})
        rcv = np.flatnonzero(ds[j] < reach)
        edges.append(_pairs(np.array([i]), rcv, 1.0 - ds[j][rcv] / reach, "mid", "M11>MC14/MC15"))

    return mean, obs, sources, edges


def _m9_block(r: np.random.Generator, grid: _Grid, tries: int = 2000) -> tuple[int, int, int, int]:
    """A fully present block for M9, centred within the middle half in x so both arms have room."""
    lo, hi = LONG_RANGE["M9_block"]
    n_side = grid.index_map.shape[0]
    offset = (n_side - 1) / 2
    spacing = grid.half / offset
    for _ in range(tries):
        h, w = (int(v) for v in r.integers(lo, hi + 1, size=2))
        r0, c0 = int(r.integers(0, n_side - h + 1)), int(r.integers(0, n_side - w + 1))
        xc = (c0 + (w - 1) / 2 - offset) * spacing
        yc = (r0 + (h - 1) / 2 - offset) * spacing
        if abs(xc) > 0.5 * grid.half or abs(yc) > 0.8 * grid.half:
            continue
        if (grid.index_map[r0 : r0 + h, c0 : c0 + w] >= 0).all():
            return r0, c0, h, w
    raise RuntimeError("found no room for the M9 block")


def _spatially_variable(seed: int, slide: str, grid: _Grid):
    """Item 10: each SP gene is "high" at its blobs and decays to 0, affecting nothing else.

    A blob is a rotated, possibly elongated Gaussian; a skewed one is multiplied by
    ``Φ(skew · z)`` along its major axis and renormalised to peak at 1, which pushes the mass to one
    side. Several blobs combine as a union, like M6.
    """
    mean, blobs = {}, []
    for gene, (n_blobs, sigma, aspect, skew) in SP_GENES.items():
        r = _rng(seed, slide, gene)
        fields = []
        for c in _spread_points(r, n_blobs, grid.half, 0.8, 3 * sigma):
            angle = r.uniform(0.0, np.pi)
            dx, dy = (grid.pos - c).T
            z1 = (dx * np.cos(angle) + dy * np.sin(angle)) / sigma
            z2 = (-dx * np.sin(angle) + dy * np.cos(angle)) / (sigma / aspect)
            f = np.exp(-0.5 * (z1**2 + z2**2))
            s = 0.0
            if skew:
                s = skew * float(r.choice([-1.0, 1.0]))
                f = np.minimum(f * ndtr(s * z1) / _skew_peak(s), 1.0)
            fields.append(f)
            blobs.append(
                {
                    "gene": gene,
                    "x": c[0],
                    "y": c[1],
                    "sigma_major": sigma,
                    "sigma_minor": sigma / aspect,
                    "angle_deg": np.degrees(angle),
                    "skew": s,
                }
            )
        mean[gene] = LEVELS["high"] * (1 - np.prod(1 - np.stack(fields), axis=0))
    return mean, blobs


def _region(seed: int, slide: str, grid: _Grid):
    """Item 17: halve the slide along the x-axis, the y-axis or a diagonal.

    One half is CM1+ ("up", CM2 at 0, CM3 "down"), the other CM2+ (CM1 at 0, CM2 and CM3 "up").
    No spot lies on an axis, but a diagonal runs through a line of spots; those join the side
    where ``y - x`` (or ``y + x``) is non-negative.
    """
    r = _rng(seed, slide, "region")
    split = str(r.choice(["x_axis", "y_axis", "diagonal"]))
    if split == "diagonal":
        split = str(r.choice(["diagonal", "anti_diagonal"]))
    x, y = grid.pos.T
    f = {"x_axis": y, "y_axis": x, "diagonal": y - x, "anti_diagonal": y + x}[split]
    cm1_side = str(r.choice(["positive", "negative"]))
    is_cm1 = (f >= 0) if cm1_side == "positive" else (f < 0)
    up, down = LEVELS["up"], LEVELS["down"]
    mean = {"CM1": up * is_cm1, "CM2": up * ~is_cm1, "CM3": np.where(is_cm1, down, up)}
    return mean, np.where(is_cm1, "CM1", "CM2"), split, cm1_side


def _sparse_responses(seed: int, slide: str, grid: _Grid):
    """Item 18: compact sources, and a few random receiver spots whose response depends on them.

    Every other long-range field is smooth at 100 µm, so a masked spot's unmasked neighbours
    predict most of it. These responses are not fields: only receivers express them, at a level
    set by the distance to the nearest source cluster -- ``start + step`` per further 1 mm band,
    never below 0, and 0 for a receiver closer than ``threshold``. With 20-50 receivers on ~2500
    spots another receiver is rarely within two hops, and the compact source leaves no trace past
    the first band, so nothing local says how strong the response is. The 200-500-receiver
    programmes are the control: there a neighbouring receiver in the same band usually gives the
    answer away.

    Receivers express the programme's receptor ("high"). Without it no model could tell a
    receiver from the other 98% of spots, and the best prediction of a masked response would be
    ~0 everywhere. With it, under gene masking, a spot shows *that* it responds while *how much*
    still needs the distant source. Node masking only scores a response while the whole spot,
    receptor included, is masked, so no model knows it is looking at a receiver: the best guess is
    P(receiver) x level, and knowing the source removes under 1% of the response's loss there
    (66-100% under gene masking).
    """
    high = LEVELS["high"]
    n = len(grid.pos)
    mean: dict[str, np.ndarray] = {}
    obs: dict[str, np.ndarray] = {}
    sources: list[dict] = []
    edges: list[pd.DataFrame] = []
    for p in SPARSE:
        r = _rng(seed, slide, p.source)
        k = int(r.integers(p.n_clusters[0], p.n_clusters[1] + 1))
        centres = _spread_points(r, k, grid.half, 0.8, SPARSE_MIN_SEPARATION)
        ds = np.stack([np.hypot(*(grid.pos - c).T) for c in centres])  # [k, n]
        g = _gauss(ds, SPARSE_SIGMA)
        d, nearest = ds.min(0), ds.argmin(0)

        receivers = r.choice(n, size=min(int(r.integers(p.n_receivers[0], p.n_receivers[1] + 1)), n), replace=False)
        is_receiver = np.isin(np.arange(n), receivers)
        level = np.maximum(p.start + p.step * np.floor(d / SPARSE_BAND), 0.0) * (d >= p.threshold)

        # Several clusters combine as a union, like M6, so overlapping ones still peak at "high".
        mean[p.source] = high * (1 - np.prod(1 - g, axis=0))
        mean[p.response] = np.where(is_receiver, level, 0.0)
        mean[p.receptor] = high * is_receiver
        obs[f"is_receiver_{p.response}"] = is_receiver
        obs[f"dist_{p.source}"] = d
        sources += [{"program": p.source, "x": c[0], "y": c[1], "sigma": SPARSE_SIGMA} for c in centres]

        # Edges run from the core of the cluster that sets a receiver's level (its spots at half
        # the peak or more) to that receiver -- receivers held at 0 by the distance included,
        # since telling that apart also takes the source.
        for j in range(k):
            core = np.flatnonzero(g[j] >= 0.5)
            rcv = receivers[nearest[receivers] == j]
            if len(core) and len(rcv):
                edges.append(_pairs(core, rcv, np.ones(len(rcv)), p.density, f"{p.source}>{p.response}"))
    return mean, obs, sources, edges


def _background(seed: int, slide: str, n: int, genes: pd.DataFrame):
    """Items 11-15: counts and expected counts of the background genes on one slide."""
    counts, expected = {}, {}
    for gene, p in genes.iterrows():
        r = _rng(seed, slide, gene)
        model = p["noise_model"]
        mu = np.full(n, p["param_mean"])
        if model == "beta_binomial":
            c = r.binomial(HV_MAX_COUNT, r.beta(p["param_beta_a"], p["param_beta_b"], size=n))
        elif model == "negative_binomial":
            c = r.poisson(r.gamma(p["param_theta"], p["param_mean"] / p["param_theta"], size=n))
        elif model == "zero_inflated_nb":
            nb_mean = p["param_mean"] / (1.0 - p["param_dropout"])
            c = r.poisson(r.gamma(p["param_theta"], nb_mean / p["param_theta"], size=n))
            c = c * (r.random(n) >= p["param_dropout"])
        elif model == "sparse_poisson":
            # Expressed on 1-10 spots of the slide, Poisson(5) there, 0 everywhere else.
            k = min(int(r.integers(POI_N_SPOTS[0], POI_N_SPOTS[1] + 1)), n)
            on = r.choice(n, size=k, replace=False)
            c = np.zeros(n, dtype=np.int64)
            c[on] = r.poisson(POI_MEAN, size=k)
            mu = np.zeros(n)
            mu[on] = POI_MEAN
        elif model == "poisson":
            c = r.poisson(p["param_mean"], size=n)
        else:
            raise ValueError(f"unknown noise model {model!r} for {gene}")
        counts[gene], expected[gene] = c, mu
    return counts, expected


def _low_noise_counts(r: np.random.Generator, mean: np.ndarray, noise_scale: float) -> np.ndarray:
    """Integer counts scattering around ``mean`` by about ``noise_scale * sqrt(mean)``.

    A Gaussian of that width, clipped at 0 and stochastically rounded, which keeps the expected
    count at ``mean`` (the clip only adds a few hundredths below a mean of ~0.3). At the default
    0.4 a mean of 1 ("down") reads 1 about two times in three and 0 or 2 otherwise, and a mean of 0
    reads exactly 0. ``noise_scale=1`` gives a variance about the size of Poisson's.
    """
    y = np.maximum(mean + noise_scale * np.sqrt(mean) * r.standard_normal(mean.shape), 0.0)
    base = np.floor(y)
    return base + (r.random(mean.shape) < y - base)


# ------------------------------------------------------------------------------------------------
# Slides and dataset
# ------------------------------------------------------------------------------------------------


def _simulate_slide(
    seed: int, info, genes: pd.DataFrame, *, n_side, spacing, n_crop_lines, max_holes, noise_scale, sender_rates, cutoff
):
    """Simulate one slide: lattice, every programme, counts, and its ground-truth records."""
    slide = info.slide
    r = _rng(seed, slide, "geometry")
    n_holes = int(r.integers(1, max_holes + 1)) if info.has_holes else 0
    keep, crop_side = _crop(r, n_side, n_crop_lines if info.cropped else 0)
    holes = _punch_holes(r, keep, n_holes)
    grid = _make_grid(keep, spacing)
    n = len(grid.pos)

    in_b = CONDITIONS.index(info.condition)
    mean = {gene: np.full(n, LEVELS[levels[in_b]]) for gene, levels in CONDITION_GENES.items()}
    mean |= {gene: np.full(n, LEVELS[getattr(info, gene)]) for gene in BATCH_GENES}
    sr_mean, sr_obs, sr_edges = _short_range(_rng(seed, slide, "short_range"), grid, sender_rates)
    lr_mean, lr_obs, sources, lr_edges = _long_range(seed, slide, grid, info.has_M4, info.has_M5, cutoff)
    sv_mean, blobs = _spatially_variable(seed, slide, grid)
    rg_mean, region, region_split, cm1_side = _region(seed, slide, grid)
    sp_mean, sp_obs, sp_sources, sp_edges = _sparse_responses(seed, slide, grid)
    mean |= sr_mean | lr_mean | sv_mean | rg_mean | sp_mean
    sources += sp_sources

    counts = {gene: _low_noise_counts(_rng(seed, slide, "noise", gene), mu, noise_scale) for gene, mu in mean.items()}
    bg_counts, bg_expected = _background(seed, slide, n, genes[genes["noise_model"] != "low_noise"])
    counts |= bg_counts
    mean |= bg_expected
    if set(mean) != set(genes.index):
        raise RuntimeError(f"gene table and simulation disagree: {sorted(set(mean) ^ set(genes.index))}")

    obs = pd.DataFrame(
        {
            "condition": info.condition,
            "slide": slide,
            "split": info.split,
            "region": region,
            "array_row": grid.row,
            "array_col": grid.col,
            "n_neighbors": (grid.nbr >= 0).sum(1),
            "has_M4": info.has_M4,
            "has_M5": info.has_M5,
            "cropped": info.cropped,
            "has_holes": n_holes > 0,
            **sr_obs,
            **lr_obs,
            **sp_obs,
        },
        index=[f"{slide}_r{rr:02d}_c{cc:02d}" for rr, cc in zip(grid.row, grid.col, strict=True)],
    )
    edges = pd.concat(sr_edges + lr_edges + sp_edges, ignore_index=True)
    record = {
        "n_spots": n,
        "n_rows": int(keep.any(1).sum()),
        "n_cols": int(keep.any(0).sum()),
        "crop_side": crop_side,
        "n_holes": len(holes),
        "region_split": region_split,
        "cm1_side": cm1_side,
        "M10_direction": next(s["direction"] for s in sources if s["program"] == "M10"),
    }
    return {
        "obs": obs,
        "pos": grid.pos,
        "counts": np.column_stack([counts[g] for g in genes.index]).astype(np.float32),
        "expected": np.column_stack([mean[g] for g in genes.index]).astype(np.float32),
        "edges": edges,
        "record": record,
        "holes": [{"row": h[0], "col": h[1], "n_rows": h[2], "n_cols": h[3]} for h in holes],
        "sources": sources,
        "blobs": blobs,
    }


def make_synth_spot(
    *,
    n_slides_per_condition: int = 20,
    n_side: int = 50,
    spacing: float = SPACING_UM,
    n_cropped_per_condition: int = 5,
    n_crop_lines: int = 1,
    n_with_holes_per_condition: int = 10,
    max_holes: int = 5,
    noise_scale: float = 0.4,
    sender_rates: dict[str, float] | None = None,
    n_hv: int = 10,
    n_nb: int = 40,
    n_zinb: int = 40,
    n_poi: int = 40,
    n_hk: int = 40,
    n_val_per_condition: int = 4,
    n_test_per_condition: int = 4,
    graph_radius: float | None = None,
    edge_weight_cutoff: float = 0.1,
    seed: int = 0,
) -> AnnData:
    """Simulate the full synth_spot dataset.

    Parameters
    ----------
    n_slides_per_condition
        Slides in each of conditions A and B. With 64 possible batch triples, at most 32.
    n_side, spacing
        Spots per side of the full lattice, and centre-to-centre distance in µm. The programme
        geometry (``LONG_RANGE``, ``SP_GENES``) is in µm and does not rescale with them.
    n_cropped_per_condition, n_crop_lines
        Slides per condition that lose border lines, and how many lines (all on one side).
    n_with_holes_per_condition, max_holes
        Slides per condition with holes, and the most holes one gets (at least 1).
    noise_scale
        Scatter of the structured genes around their mean, in units of ``sqrt(mean)``. The
        default 0.4 reproduces the spec's "down" level; 1.0 gives Poisson-sized variance.
    sender_rates
        Fraction of spots carrying each short-range sender; defaults to :data:`SENDER_RATES`.
    n_hv, n_nb, n_zinb, n_poi, n_hk
        Number of highly variable, NB, ZINB, lowly expressed and housekeeping genes.
    n_val_per_condition, n_test_per_condition
        Slides per condition held out for val and test; the rest are train. The defaults give
        12/4/4 slides per condition, 60/20/20 overall, the same in A and B.
    graph_radius
        Radius of the reference neighbour graph in ``obsp``. Defaults to 1.1 x ``spacing``, which
        is the 4-neighbourhood the short-range programmes use; any value between ``spacing`` and
        ``sqrt(2) x spacing`` gives the same graph. Training does not read it -- geome rebuilds the
        graph from ``cfg.dataset.spatial_neigbors_kwargs`` -- so set the config to match.
    edge_weight_cutoff
        Response strength above which a directional pair enters the ground-truth edge list.
    seed
        Root seed; every slide and programme derives its own generator from it.

    Returns
    -------
    anndata.AnnData
        Median-normalised log1p in ``X`` and ``layers['log1p_norm']``, raw counts in
        ``layers['counts']``, the noise-free expected counts in ``layers['expected']``, a
        Freeman-Tukey layer, coordinates in ``obsm['spatial']`` (µm), a reference graph in
        ``obsp``, and the ground truth in ``obs``, ``var`` and ``uns['synthetic']``.
    """
    if n_val_per_condition + n_test_per_condition >= n_slides_per_condition:
        raise ValueError("need at least one train slide per condition")
    if max(n_cropped_per_condition, n_with_holes_per_condition) > n_slides_per_condition:
        raise ValueError("cannot damage more slides than there are")
    rates = dict(SENDER_RATES if sender_rates is None else sender_rates)
    # Rounded: 1.1 * 100 is 110.00000000000001 in floating point, and a config's `radius: 110` is
    # compared against this value exactly.
    graph_radius = round(1.1 * spacing, 6) if graph_radius is None else float(graph_radius)

    genes = _gene_table(spacing, n_hv, n_nb, n_zinb, n_poi, n_hk)
    _draw_gene_params(seed, genes)
    design = _design(
        seed,
        n_slides_per_condition,
        n_cropped_per_condition,
        n_with_holes_per_condition,
        n_val_per_condition,
        n_test_per_condition,
    )

    results, records, holes, sources, blobs = [], [], [], [], []
    offset = 0
    for info in design.itertuples(index=False):
        res = _simulate_slide(
            seed,
            info,
            genes,
            n_side=n_side,
            spacing=spacing,
            n_crop_lines=n_crop_lines,
            max_holes=max_holes,
            noise_scale=noise_scale,
            sender_rates=rates,
            cutoff=edge_weight_cutoff,
        )
        # Edges refer to cells by position in the final object, which survives the obs_names
        # renaming prepare_geome_dataset does.
        res["edges"][["sender", "receiver"]] += offset
        res["edges"]["slide"] = info.slide
        offset += len(res["obs"])
        results.append(res)
        records.append(res["record"])
        holes += [{"slide": info.slide, **h} for h in res["holes"]]
        sources += [{"slide": info.slide, **s} for s in res["sources"]]
        blobs += [{"slide": info.slide, **b} for b in res["blobs"]]

    obs = pd.concat([res["obs"] for res in results])
    counts = np.concatenate([res["counts"] for res in results])
    expected = np.concatenate([res["expected"] for res in results])
    obs["total_counts"] = counts.sum(1)
    obs["n_genes_by_counts"] = (counts > 0).sum(1)
    if (obs["total_counts"] == 0).any():
        # Cannot happen with the housekeeping genes present. Raise rather than drop: dropping would
        # shift every positional index in the edge list.
        raise RuntimeError("some spots have no counts at all; keep the housekeeping genes")
    categories = {
        "condition": list(CONDITIONS),
        "slide": design["slide"].tolist(),
        "split": ["train", "val", "test"],
        "region": ["CM1", "CM2"],
        "R34_state": ["none", "S3", "S4", "both"],
    }
    for col, cats in categories.items():
        obs[col] = pd.Categorical(obs[col], categories=cats)

    var = genes.copy()
    adata = AnnData(X=counts, obs=obs, var=var)
    adata.obsm["spatial"] = np.concatenate([res["pos"] for res in results]).astype(np.float64)
    adata.layers["counts"] = counts.copy()
    adata.layers["expected"] = expected

    # Normalised the same way as synth_data_0 and the legnini preprocessing: scanpy's median
    # normalisation and log1p. Non-negative, which tl.masking's MASK_VALUE = -1 relies on. The
    # simulation has no library-size variation, so this only rescales spots by their total --
    # dominated by the HV genes -- and `counts`/`expected` remain the clean reference.
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    raw = adata.layers["counts"]
    adata.layers["norm_ftsqrt"] = np.sqrt(raw) + np.sqrt(raw + 1.0)

    # Reference graph: the 4-neighbourhood, built per slide. Training does not read it (see
    # graph_radius above); the radius is recorded so a run can check its config against it.
    # The call squidpy >= 1.8 deprecates, kept on purpose: the cluster environment may predate its
    # replacement, and geome's AddAdjMatrix makes the same call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=graph_radius, library_key="slide")

    edges = pd.concat([res["edges"] for res in results], ignore_index=True)
    edges = edges.astype({"sender": np.int32, "receiver": np.int32, "weight": np.float32})
    for col in ("range_class", "program", "slide"):
        edges[col] = pd.Categorical(edges[col])

    slides = pd.concat([design, pd.DataFrame(records)], axis=1).set_index("slide")
    adata.uns["synthetic"] = {
        "params": {
            "dataset": "synth_spot",
            "n_slides_per_condition": n_slides_per_condition,
            "n_side": n_side,
            "spacing_um": spacing,
            "n_cropped_per_condition": n_cropped_per_condition,
            "n_crop_lines": n_crop_lines,
            "n_with_holes_per_condition": n_with_holes_per_condition,
            "max_holes": max_holes,
            "noise_scale": noise_scale,
            "sender_rates": rates,
            "levels": dict(LEVELS),
            "long_range": {k: np.asarray(v, dtype=float) for k, v in LONG_RANGE.items()},
            "sparse_sigma": SPARSE_SIGMA,
            "sparse_band": SPARSE_BAND,
            "sparse_min_separation": SPARSE_MIN_SEPARATION,
            "graph_radius": graph_radius,
            "edge_weight_cutoff": edge_weight_cutoff,
            "seed": seed,
        },
        "slides": slides,
        "holes": pd.DataFrame(holes, columns=["slide", "row", "col", "n_rows", "n_cols"]),
        # Only M10 has a direction; "" elsewhere, since h5ad cannot store a str/NaN mixed column.
        "sources": pd.DataFrame(sources).fillna({"direction": ""}),
        "sp_genes": pd.DataFrame(SP_GENES, index=["n_blobs", "sigma_major", "aspect", "skew"]).T,
        "sp_blobs": pd.DataFrame(blobs),
        "sparse_programmes": pd.DataFrame(
            [
                {
                    "source": p.source,
                    "response": p.response,
                    "receptor": p.receptor,
                    "min_clusters": p.n_clusters[0],
                    "max_clusters": p.n_clusters[1],
                    "min_receivers": p.n_receivers[0],
                    "max_receivers": p.n_receivers[1],
                    "start": p.start,
                    "step": p.step,
                    "threshold": p.threshold,
                    "density": p.density,
                }
                for p in SPARSE
            ]
        ).set_index("response"),
        "interaction_edges": edges,
    }
    return adata


def _summary(adata: AnnData) -> str:
    """One-screen description of what was generated, printed by the CLI."""
    slides = adata.uns["synthetic"]["slides"]
    edges = adata.uns["synthetic"]["interaction_edges"]
    lines = [
        f"spots x genes: {adata.n_obs} x {adata.n_vars}",
        f"slides: {len(slides)}  spots per slide: {slides['n_spots'].min()}-{slides['n_spots'].max()}",
        "split (slides): " + ", ".join(f"{k}={v}" for k, v in slides["split"].value_counts().items()),
        f"cropped: {int(slides['cropped'].sum())}  with holes: {int(slides['has_holes'].sum())} "
        f"({int(slides['n_holes'].sum())} holes)  M4: {int(slides['has_M4'].sum())}  M5: {int(slides['has_M5'].sum())}",
        "programmes: " + ", ".join(f"{k}={v}" for k, v in adata.var["program"].value_counts().items()),
        "ground-truth edges: " + ", ".join(f"{k}={v}" for k, v in edges["range_class"].value_counts().items()),
        f"median counts/spot: {np.median(adata.obs['total_counts']):.0f}",
    ]
    if "sparse_programmes" in adata.uns["synthetic"]:
        # Receivers held at 0 by the distance respond to nothing visible, so count the ones that do.
        expected = adata.layers["expected"]
        per_slide = pd.DataFrame(
            {g: expected[:, adata.var_names.get_loc(g)] > 0 for g in adata.uns["synthetic"]["sparse_programmes"].index}
        ).groupby(adata.obs["slide"].to_numpy())
        lines.append(
            "responding receivers per slide (median): "
            + ", ".join(f"{g}={int(v)}" for g, v in per_slide.sum().median().items())
        )
    return "\n".join(lines)


#: Sequential blue ramp of the maps, light at 0 to dark at "high".
_MAP_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _showcase_slide(slides: pd.DataFrame) -> str:
    """The first slide with the most of: a crop, holes, M4 and M5."""
    return slides[["cropped", "has_holes", "has_M4", "has_M5"]].sum(axis=1).idxmax()


def plot_slide_maps(adata: AnnData, slide: str | None = None, layer: str = "expected", path=None):
    """Draw every structured gene of one slide on its lattice, one small panel per gene.

    The quickest check that the programmes came out as intended -- rings, the stripe and the cone
    are only convincing by eye. Every gene whose rule is a level or a spatial field gets a panel
    (the background genes have no shape to show), on the shared scale from 0 to "high", so panels
    compare directly and sampled counts above "high" saturate. The exceptions are genes whose rule
    goes past "high" (the larger sparse responses): they keep their own range, from 0 to their
    largest expected value anywhere in the dataset, and their title states it. Removed spots,
    holes and cropped lines alike, stay blank, and y points up.

    Parameters
    ----------
    adata
        Output of :func:`make_synth_spot`.
    slide
        Slide to draw. Defaults to the first one with the most of a crop, holes, M4 and M5.
    layer
        ``"expected"`` for the noise-free means, ``"counts"`` for what was sampled.
    path
        Where to save the figure as PNG; nothing is written when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        Built without pyplot, so it changes no global backend state: it renders on a headless
        node and displays inline in a notebook.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from scipy.sparse import issparse

    syn = adata.uns["synthetic"]
    slides, params = syn["slides"], syn["params"]
    slide = _showcase_slide(slides) if slide is None else slide
    info = slides.loc[slide]
    n_side = int(params["n_side"])
    half = n_side * float(params["spacing_um"]) / 2

    on = (adata.obs["slide"] == slide).to_numpy()
    rows, cols = adata.obs["array_row"].to_numpy()[on], adata.obs["array_col"].to_numpy()[on]
    genes = adata.var_names[(adata.var["noise_model"] == "low_noise").to_numpy()]
    values = adata.layers[layer][on][:, adata.var_names.get_indexer(genes)]
    values = values.toarray() if issparse(values) else np.asarray(values)
    # One range per gene for the whole dataset, so a gene's panels compare across slides and layers.
    reference = adata.layers["expected"] if "expected" in adata.layers else adata.layers[layer]
    top = np.asarray(reference[:, adata.var_names.get_indexer(genes)].max(axis=0)).ravel()
    vmax = np.maximum(top, LEVELS["high"])

    cmap = LinearSegmentedColormap.from_list("synth_spot", _MAP_RAMP)
    cmap.set_bad("#ffffff")  # removed spots show the background: a hole reads as missing tissue
    n_col = 12
    n_row = -(-len(genes) // n_col)
    fig = Figure(figsize=(1.75 * n_col, 1.8 * n_row + 0.6), facecolor="#ffffff", layout="constrained")
    axes = fig.subplots(n_row, n_col, squeeze=False)
    shared = None  # a panel on the 0-"high" scale, for the colour bar
    for j, (ax, gene) in enumerate(zip(axes.flat, genes, strict=False)):
        img = np.full((n_side, n_side), np.nan)
        img[rows, cols] = values[:, j]
        im = ax.imshow(
            img,
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax[j],
            extent=(-half, half, -half, half),
            interpolation="nearest",
        )
        if vmax[j] == LEVELS["high"]:
            shared = im
        title = gene if vmax[j] == LEVELS["high"] else f"{gene}  (0-{vmax[j]:.0f})"
        ax.set_title(title, fontsize=9, color="#222222", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d9d9d9")
    for ax in axes.flat[len(genes) :]:
        ax.axis("off")
    cbar = fig.colorbar(im if shared is None else shared, ax=axes, shrink=0.35, pad=0.01)
    label = "expected count" if layer == "expected" else f"count ({layer})"
    cbar.set_label(f"{label}; titled panels: their own range", color="#444444")
    cbar.outline.set_visible(False)
    fig.suptitle(
        f"synth_spot, slide {slide} ({layer}): condition {info['condition']}, crop {info['crop_side'] or 'none'}, "
        f"{info['n_holes']} holes, regions split on {info['region_split']}, M10 cone {info['M10_direction']}  (y up)",
        color="#222222",
        fontsize=11,
    )
    if path is not None:
        fig.savefig(path, dpi=110, facecolor="#ffffff")
    return fig


def main() -> None:
    """Write the dataset to an ``.h5ad`` file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synth_spot.h5ad"))
    parser.add_argument("--n-slides-per-condition", type=int, default=20)
    parser.add_argument("--n-side", type=int, default=50)
    parser.add_argument("--noise-scale", type=float, default=0.4)
    parser.add_argument("--graph-radius", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-slide", type=str, default=None, help="slide to map; default: the most damaged one")
    parser.add_argument("--no-plot", action="store_true", help="skip the per-gene maps")
    args = parser.parse_args()

    adata = make_synth_spot(
        n_slides_per_condition=args.n_slides_per_condition,
        n_side=args.n_side,
        noise_scale=args.noise_scale,
        graph_radius=args.graph_radius,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # gzip: 90 MB instead of 296 for the default design, read back in ~1 s instead of 0.2 s.
    adata.write_h5ad(args.out, compression="gzip")
    print(_summary(adata))
    print(f"\nwritten to {args.out}")
    print(f"set cfg.model.global_component.parameters.max_seq_len >= {args.n_side**2}")
    print(f"set cfg.dataset.spatial_neigbors_kwargs.radius = {adata.uns['synthetic']['params']['graph_radius']}")
    if not args.no_plot:
        slide = args.plot_slide or _showcase_slide(adata.uns["synthetic"]["slides"])
        for layer in ("expected", "counts"):
            png = args.out.with_name(f"{args.out.stem}_maps_{slide}_{layer}.png")
            plot_slide_maps(adata, slide, layer, png)
            print(f"maps written to {png}")


if __name__ == "__main__":
    main()
