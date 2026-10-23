"""Which genes vary on a local and which on a tissue scale, and which part of a decomposable prediction carries each.

roadmap.md, task 1 ("Moran's I split"). Per gene, Moran's I of its expression against a neighbourhood that grows: the
``k`` nearest other cells of the same slide. A gene that varies on a local scale is autocorrelated with its nearest
neighbours and with nobody farther away, so its curve drops; one that varies on the scale of the tissue stays correlated
with neighbours far out, so its curve stays up. Moran's I measures spatial autocorrelation, not interaction: it is the
axis along which a model with a local and a global part should split the genes, and it needs no labels.

**The curve** (:func:`moran_curves`). With row-standardised k-NN weights and each gene z-scored within its slide,
Moran's I is the mean over cells ``i`` of ``z_i`` times the mean ``z`` of ``i``'s neighbours -- the mean of the local
Moran's I's. It is estimated from ``n_centres`` cells drawn per slide (:func:`sample_neighbours`), the same for every
gene and every ``k``, at a cost of centres x k x genes whatever the slide's size, as the ratio of
``sum z_i * mean z_neighbours`` to ``sum z_i ** 2`` over the centres; with every cell a centre it is the textbook
statistic. The sums are kept per neighbour rank (:func:`moran_by_rank`), so every ``k`` of the list comes out of one
pass, in two forms:

* ``shell`` -- the neighbours ranked ``k_prev + 1 .. k`` (``k_prev`` the size before ``k`` in the list): a correlogram,
  the correlation with the cells at that distance. A local gene drops to about 0 beyond its scale. The main read.
* ``cumulative`` -- the neighbours ``1 .. k``: Moran's I with k-NN weights as it is usually computed (the manuscript's
  Figs. 3C, 4I). Every curve falls, since the near neighbours stay in.

Slides share no coordinates, so neighbours never cross a slide, and the sums are pooled over the slides into one curve
per gene. A gene constant within a slide counts nothing there. The floor is a permutation null
(:func:`permutation_null`): the values shuffled among the cells of each slide, the same neighbours.

**The length scale** (:func:`length_scales`): ``ell_k``, the ``k`` at which the shell curve, after its peak, falls to
half the peak (interpolated on log k; ``inf`` when it never does), and ``ell_um``, the same through the median distance
to the ``k``-th neighbour. ``pattern``: ``anti`` (the first shell more than ``z`` null SDs below the null: neighbours
that exclude each other, e.g. alternating senders and receivers -- not global variation), ``variable`` (the highest
shell above the null by ``z``, corrected for the number of shells), ``none``, ``constant`` (no curve). The peak rather
than the first shell: on a lattice the nearest shell can miss what the next one holds.

**The model** (:func:`collect_fold_parts`, :func:`part_shares`). Only a prediction that is a sum of parts can be split:
the additive decoder (``local`` = W_l h_l, ``global`` = W_g h_g), the residual decoder. The parts are read where the
model was trained to predict, in hidden cells: the cells of each slide are split into folds (:func:`cell_folds`) and
each fold is hidden in turn, so every cell is predicted once from its context with the rest of the slide visible as in
training. On a fully visible input every output would sit where it never got a loss, and the local part could copy the
visible gene. A gene's share of a part is that part's share of the variance of the summed prediction over the cells
(``cov_share`` of :func:`~interscale.evaluation.masked_probing.attribution_shares`, per gene). It means something only
for a model trained to predict every gene: under programme masking the decoder rows of a gene outside the programme get
no gradient and keep their initial values.

:func:`gene_scales` puts it together, :func:`scale_tables` gives the tables of a run's sections, and the plots draw the
curves (:func:`plot_scale_curves`: every gene's curve of its expression, coloured by the part that carries it), each
part's curve next to the expression's (:func:`plot_part_curves`), and their maps (:func:`plot_part_maps`). Everything is
dataset-agnostic: genes, the planted scales and classes of a synthetic dataset, the list of ``k`` and the units come
from the caller.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import norm, spearmanr

from interscale.evaluation.masked_probing import _centred_parts, collect_head_predictions

#: The neighbourhood sizes of the curves; doubling, each shell holds the neighbours ranked ``k/2 + 1 .. k``. On
#: synth_spot's 100 µm lattice 512 neighbours reach about 1.4 mm; the cost grows with the largest.
DEFAULT_KS = (4, 8, 16, 32, 64, 128, 256, 512)
#: The two forms of a curve.
FORMS = ("shell", "cumulative")
#: Elements of the ``[centres, k, genes]`` block one step of :func:`moran_by_rank` holds (float32: 40 MB).
CHUNK_ELEMENTS = 10_000_000
#: A gene's ``pattern`` in :func:`length_scales`.
VARIABLE, ANTI, NONE, CONSTANT = "variable", "anti", "none", "constant"
#: The map of the expression itself among the maps of :func:`gene_scales`.
TRUTH = "true"


# ------------------------------------------------------------------------------- neighbours


@dataclass
class Neighbours:
    """Sampled centres and their nearest other cells of the same graph.

    Attributes
    ----------
    centre
        ``[C]`` the row of each centre among the ``N`` cells.
    neighbour
        ``[C, K]`` the rows of its ``K`` nearest other cells of its graph, nearest first; -1 where the graph has fewer.
    distance
        ``[C, K]`` their distances, in the units of the positions; ``inf`` where the neighbour is -1.
    """

    centre: np.ndarray
    neighbour: np.ndarray
    distance: np.ndarray

    @property
    def k_max(self) -> int:
        return int(self.neighbour.shape[1])

    def reach(self, ks, unit: float = 1.0) -> np.ndarray:
        """``[len(ks)]`` the median distance from a centre to its ``k``-th neighbour, times ``unit``.

        NaN where no centre has a ``k``-th neighbour.
        """
        out = []
        for k in ks:
            d = self.distance[:, int(k) - 1]
            d = d[np.isfinite(d)]
            out.append(float(np.median(d)) * float(unit) if len(d) else float("nan"))
        return np.asarray(out)


def sample_neighbours(pos, graph, *, k_max: int, n_centres: int | None = 2000, seed: int = 0) -> Neighbours:
    """``n_centres`` random cells per graph and their ``k_max`` nearest others (a smaller graph: all; None: all).

    Drawn from a private generator, so the global ones are not touched; the same arguments give the same centres. The
    centre itself is dropped by its index, not by distance, so cells at the same position stay each other's neighbours.

    Parameters
    ----------
    pos
        ``[N, D]`` positions.
    graph
        ``[N]`` the graph (slide) of each cell; neighbours never cross it.
    k_max
        Neighbours per centre; a graph with fewer cells pads with -1.
    n_centres
        Centres per graph.
    seed
        Seed of the draw.
    """
    pos = np.asarray(pos, dtype=np.float64)
    graph = np.asarray(graph)
    if pos.ndim != 2 or len(pos) != len(graph):
        raise ValueError(f"pos must be [N, D] with one graph per row, got {pos.shape} and {graph.shape}.")
    if int(k_max) < 1:
        raise ValueError(f"k_max must be at least 1, got {k_max}.")
    if n_centres is not None and int(n_centres) < 1:
        raise ValueError(f"n_centres must be at least 1 or None, got {n_centres}.")
    k_max = int(k_max)
    rng = np.random.default_rng(seed)
    centres, neighbours, distances = [], [], []
    for g in pd.unique(graph):
        rows = np.flatnonzero(graph == g)
        n = len(rows)
        take = (
            rows
            if n_centres is None or int(n_centres) >= n
            else np.sort(rng.choice(rows, size=int(n_centres), replace=False))
        )
        nb = np.full((len(take), k_max), -1, dtype=np.int64)
        dist = np.full((len(take), k_max), np.inf)
        k = min(k_max, n - 1)
        if k > 0:
            own = np.searchsorted(rows, take)  # each centre's place among its graph's cells
            d, j = cKDTree(pos[rows]).query(pos[take], k=k + 1)
            d, j = np.asarray(d).reshape(len(take), k + 1), np.asarray(j).reshape(len(take), k + 1)
            is_self = j == own[:, None]
            # The centre's own column, or the farthest when ties at distance 0 pushed it out of the k + 1.
            drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k)
            keep = np.ones_like(is_self)
            keep[np.arange(len(take)), drop] = False
            nb[:, :k] = rows[j[keep].reshape(len(take), k)]
            dist[:, :k] = d[keep].reshape(len(take), k)
        centres.append(take)
        neighbours.append(nb)
        distances.append(dist)
    return Neighbours(np.concatenate(centres), np.concatenate(neighbours), np.concatenate(distances))


# ------------------------------------------------------------------------------- Moran's I


def _standardise(values: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """``values`` z-scored per gene within each graph (population SD). A gene whose SD there is below 1e-6 of its mean's
    size is constant there: z = 0, so it adds nothing to the sums of :func:`moran_by_rank`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(graph):
        raise ValueError(f"values must be [N, G] with one graph per row, got {values.shape} and {np.shape(graph)}.")
    graph = np.asarray(graph)
    z = np.zeros_like(values)
    for g in pd.unique(graph):
        rows = graph == g
        block = values[rows]
        mean, sd = block.mean(axis=0), block.std(axis=0)
        ok = sd > 1e-6 * np.maximum(np.abs(mean), 1e-6)
        z[rows] = np.where(ok, (block - mean) / np.where(ok, sd, 1.0), 0.0)
    return z


@dataclass
class RankSums:
    """Per neighbour rank (rows, nearest first) and gene: the sum over centres of ``z_centre * z_neighbour`` (``total``)
    and of ``z_centre ** 2`` over the same terms (``norm``).
    """

    total: np.ndarray
    norm: np.ndarray


def moran_by_rank(values, graph, neighbours: Neighbours, *, dtype=np.float32) -> RankSums:
    """The sums behind Moran's I at every neighbour rank, pooled over the graphs (:class:`RankSums`).

    ``values`` ``[N, G]`` are z-scored per gene within each graph first (in float64); a term counts where the neighbour
    exists (a gene constant in a graph is 0 there, so it adds nothing to either sum). Moran's I over a set of ranks is
    ``total / norm`` summed over them: with every cell a centre ``norm`` is the number of cells per rank and this is the
    textbook statistic; with sampled centres the centres' own ``z ** 2`` fluctuate with the draw, and dividing by them
    rather than by the slide's variance cancels most of it (a ratio estimator: a smooth field read from 300 of 900
    lattice cells was up to 0.15 off, median 0.05, the other way, and at most 0.035 this way, over 20 draws). The
    products are summed in ``dtype`` within a block of centres -- float32 halves the time, the step that dominates, at a
    relative error near 1e-6 -- and the blocks in float64.
    """
    z = _standardise(values, graph).astype(dtype, copy=False)
    k, g = neighbours.k_max, z.shape[1]
    total = np.zeros((k, g))
    norm = np.zeros((k, g))
    step = max(1, CHUNK_ELEMENTS // max(1, k * g))
    for start in range(0, len(neighbours.centre), step):
        centre = neighbours.centre[start : start + step]
        nb = neighbours.neighbour[start : start + step]
        valid = nb >= 0
        z_nb = z[np.where(valid, nb, centre[:, None])]
        z_nb[~valid] = 0.0
        total += np.einsum("cg,ckg->kg", z[centre], z_nb).astype(np.float64)
        norm += valid.T.astype(np.float64) @ (z[centre].astype(np.float64) ** 2)
    return RankSums(total, norm)


def _check_ks(ks, k_max: int) -> np.ndarray:
    ks = np.asarray([int(k) for k in ks])
    if not len(ks) or ks[0] < 1 or np.any(np.diff(ks) <= 0) or ks[-1] > k_max:
        raise ValueError(f"ks must increase from 1 up to the neighbours sampled ({k_max}), got {ks.tolist()}.")
    return ks


def _ratio(total: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """``total / norm``; NaN where ``norm`` is not above float noise (a gene constant there, or no term)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 1e-9, total / np.where(norm > 1e-9, norm, 1.0), np.nan)


def curves_from_ranks(sums: RankSums, ks) -> dict[str, np.ndarray]:
    """``{"shell": [J, G], "cumulative": [J, G]}`` at the sizes ``ks``.

    ``ks`` increase and go up to the ranks summed at most; NaN where a gene has no term.
    """
    ks = _check_ks(ks, sums.total.shape[0])
    total, norm = np.cumsum(sums.total, axis=0), np.cumsum(sums.norm, axis=0)
    hi = ks - 1
    lo = np.concatenate([[-1], hi[:-1]])
    below_total = np.where(lo[:, None] >= 0, total[np.maximum(lo, 0)], 0.0)
    below_norm = np.where(lo[:, None] >= 0, norm[np.maximum(lo, 0)], 0.0)
    return {
        "shell": _ratio(total[hi] - below_total, norm[hi] - below_norm),
        "cumulative": _ratio(total[hi], norm[hi]),
    }


@dataclass
class MoranCurves:
    """Moran's I of every gene at each size of ``ks``: ``shell`` and ``cumulative`` ``[J, G]``; ``reach_um`` ``[J]`` the
    median distance to the ``k``-th neighbour, in µm.
    """

    ks: np.ndarray
    reach_um: np.ndarray
    shell: np.ndarray
    cumulative: np.ndarray

    def form(self, name: str) -> np.ndarray:
        if name not in FORMS:
            raise ValueError(f"the form must be one of {FORMS}, got {name!r}.")
        return getattr(self, name)


def moran_curves(
    values, graph, neighbours: Neighbours, ks=DEFAULT_KS, *, unit_um: float = 1.0, dtype=np.float32
) -> MoranCurves:
    """Moran's I of every gene of ``values`` ``[N, G]`` against the neighbourhood size (see the module docstring).

    ``unit_um`` is the µm per unit of the positions the neighbours were found in (``dataset.spatial_unit_um``);
    ``dtype`` as for :func:`moran_by_rank`.
    """
    curves = curves_from_ranks(moran_by_rank(values, graph, neighbours, dtype=dtype), ks)
    ks = _check_ks(ks, neighbours.k_max)
    return MoranCurves(ks, neighbours.reach(ks, unit_um), curves["shell"], curves["cumulative"])


@dataclass
class MoranNull:
    """The curves of :func:`permutation_null`, per form ``[J, G]``: their mean and SD over the shuffles."""

    mean: dict[str, np.ndarray]
    sd: dict[str, np.ndarray]


def permutation_null(
    values, graph, neighbours: Neighbours, ks=DEFAULT_KS, *, n_permutations: int = 10, seed: int = 0, dtype=np.float32
) -> MoranNull:
    """Each gene's curve with its values shuffled among the cells of each graph, ``n_permutations`` times.

    The neighbours stay, so what remains is the statistic's spread without any spatial structure: the floor a curve must
    clear. Its mean sits near ``-1 / (n - 1)``; its SD widens for heavy-tailed (sparse) genes, hence one per gene. Each
    shuffle costs as much as the curve itself.
    """
    if int(n_permutations) < 2:
        raise ValueError(f"a standard deviation needs at least 2 permutations, got {n_permutations}.")
    values = np.asarray(values, dtype=np.float64)
    graph = np.asarray(graph)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(graph == g) for g in pd.unique(graph)]
    draws: dict[str, list[np.ndarray]] = {form: [] for form in FORMS}
    for _ in range(int(n_permutations)):
        order = np.arange(len(graph))
        for rows in groups:
            order[rows] = rng.permutation(rows)
        curves = curves_from_ranks(moran_by_rank(values[order], graph, neighbours, dtype=dtype), ks)
        for form in FORMS:
            draws[form].append(curves[form])
    stacked = {form: np.stack(found) for form, found in draws.items()}
    with np.errstate(invalid="ignore"):
        return MoranNull(
            mean={form: np.nanmean(s, axis=0) for form, s in stacked.items()},
            sd={form: np.nanstd(s, axis=0, ddof=1) for form, s in stacked.items()},
        )


# ------------------------------------------------------------------------------- length scales


def _log_interpolate(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """``y`` at ``x`` on the log-log line through the positive, finite ``(xs, ys)``; NaN without two of them."""
    use = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if use.sum() < 2 or not np.isfinite(x) or x <= 0:
        return float("nan")
    return float(np.exp(np.interp(np.log(x), np.log(xs[use]), np.log(ys[use]))))


def half_point(curve: np.ndarray, ks: np.ndarray) -> float:
    """The ``k`` at which ``curve``, after its peak, falls to half the peak (on log k); ``inf`` when it never does.

    The peak, not the first value: on a lattice the nearest shell can miss a pattern the next one holds (synth_spot's
    receivers sit diagonally to their senders: about 0 in the 4 nearest neighbours, 0.45 in the next 4). Undefined (NaN)
    without a positive peak; missing values are skipped.
    """
    curve, ks = np.asarray(curve, dtype=np.float64), np.asarray(ks, dtype=np.float64)
    use = np.isfinite(curve)
    curve, ks = curve[use], ks[use]
    if not len(curve) or curve.max() <= 0:
        return float("nan")
    peak = int(np.argmax(curve))
    target = curve[peak] / 2.0
    for j in range(peak + 1, len(curve)):
        if curve[j] <= target:
            a, b = curve[j - 1], curve[j]
            t = (a - target) / (a - b)
            return float(np.exp(np.log(ks[j - 1]) + t * (np.log(ks[j]) - np.log(ks[j - 1]))))
    return float("inf")


def _mean_over_log(values: np.ndarray, ks: np.ndarray) -> float:
    """The trapezoid mean of ``values`` over log k (finite ones); the value itself for one point, NaN for none."""
    use = np.isfinite(values)
    x, y = np.log(ks[use].astype(np.float64)), values[use]
    if not len(y):
        return float("nan")
    if len(y) == 1:
        return float(y[0])
    return float(((y[1:] + y[:-1]) / 2.0 * np.diff(x)).sum() / (x[-1] - x[0]))


#: The columns of :func:`length_scales`.
SCALE_COLUMNS = ["pattern", "i_first", "z_first", "i_peak", "k_peak", "ell_k", "ell_um", "i_last", "auc"]


def null_z(curves: MoranCurves, null: MoranNull, form: str = "shell") -> np.ndarray:
    """``[J, G]``: each point of the curves in null SDs from the null's mean.

    A gene's SD is floored at the median SD of all genes at that size: a gene's own SD from a few shuffles is a noisy
    estimate, and an estimate that came out low made noise genes look structured (4 of synth_spot's 170 noise genes at
    10 shuffles). The floor costs little: for slides of thousands of cells the randomisation SD of Moran's I hardly
    depends on the gene (its kurtosis enters at order 1 / N); a heavy-tailed gene whose own SD is wider keeps it.
    """
    sd = null.sd[form]
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        floor = np.nanmedian(sd, axis=1, keepdims=True)
        sd = np.fmax(sd, floor)
        return np.where(sd > 0, (curves.form(form) - null.mean[form]) / np.where(sd > 0, sd, 1.0), np.nan)


def length_scales(curves: MoranCurves, null: MoranNull, genes, *, z: float = 3.0) -> pd.DataFrame:
    """Per gene (index ``gene``): the length scale of its expression, from its shell curve against the null.

    Each point is read in null SDs (:func:`null_z`).

    * ``pattern`` -- :data:`ANTI` when the first shell lies more than ``z`` null SDs below the null (neighbours that
      exclude each other); else :data:`VARIABLE` when the curve's highest shell lies above it by more than ``z``
      corrected for the ``J`` shells looked at (Bonferroni: the one-sided level of ``z`` divided by ``J``; 3.55 for 3 SD
      and 8 sizes); :data:`NONE` otherwise; :data:`CONSTANT` without a curve;
    * ``i_first`` and ``z_first`` -- the first shell's Moran's I and its distance from the null in null SDs;
    * ``i_peak`` and ``k_peak`` -- the highest shell and its ``k``;
    * ``ell_k`` and ``ell_um`` -- where the shell curve falls to half its peak after it (:func:`half_point`; ``inf``: it
      never does within the sizes, the gene varies beyond them), in neighbours and in µm; only for variable genes;
    * ``i_last`` -- the last shell's Moran's I: how much is left at the widest size;
    * ``auc`` -- the cumulative curve's mean over log k (the manuscript's AUC, on the scale of Moran's I).
    """
    genes = list(genes)
    shell, cumulative = curves.shell, curves.cumulative
    if shell.shape[1] != len(genes):
        raise ValueError(f"the curves hold {shell.shape[1]} genes, the names {len(genes)}.")
    zs = null_z(curves, null)
    corrected = float(norm.isf(norm.sf(z) / len(curves.ks)))
    rows = []
    for j, gene in enumerate(genes):
        curve = shell[:, j]
        finite = np.isfinite(curve)
        peak = int(np.nanargmax(np.where(finite, curve, -np.inf))) if finite.any() else 0
        if not finite.any():
            pattern = CONSTANT
        elif zs[0, j] < -z:
            pattern = ANTI
        elif np.nanmax(np.where(np.isfinite(zs[:, j]), zs[:, j], -np.inf)) > corrected:
            pattern = VARIABLE
        else:
            pattern = NONE
        ell_k = half_point(curve, curves.ks) if pattern == VARIABLE else float("nan")
        if np.isinf(ell_k):
            ell_um = float("inf")
        else:
            ell_um = _log_interpolate(ell_k, curves.ks.astype(np.float64), curves.reach_um)
        last = curve[finite]
        rows.append(
            {
                "gene": gene,
                "pattern": pattern,
                "i_first": float(curve[0]),
                "z_first": float(zs[0, j]),
                "i_peak": float(curve[peak]) if finite.any() else float("nan"),
                "k_peak": int(curves.ks[peak]) if finite.any() else -1,
                "ell_k": ell_k,
                "ell_um": ell_um,
                "i_last": float(last[-1]) if len(last) else float("nan"),
                "auc": _mean_over_log(cumulative[:, j], curves.ks),
            }
        )
    return pd.DataFrame(rows, columns=["gene", *SCALE_COLUMNS]).set_index("gene")


# ------------------------------------------------------------------------------- the model's parts


def part_shares(parts: dict[str, np.ndarray], keep: np.ndarray | None = None) -> pd.DataFrame:
    """Per gene (rows) and part: the part's share of the variance of the summed prediction over the kept entries.

    ``parts`` ``{name: [N, G]}`` add up to a prediction. Each is centred per gene over the kept entries
    (``keep`` ``[N, G]``, default all), so the centred parts add up to the centred whole ``w``; ``share_<name>`` is
    ``<c, w> / ||w||^2`` (sums to 1 over the parts, negative for a part that works against the others) and
    ``norm_<name>`` is ``||c|| / ||w||`` (parts that cancel sum to more than 1), the per-gene form of
    :func:`~interscale.evaluation.masked_probing.attribution_shares`. NaN for a gene whose prediction does not vary.
    """
    if not parts:
        raise ValueError("no parts to take a prediction apart into.")
    parts = {name: np.asarray(value, dtype=np.float64) for name, value in parts.items()}
    shape = next(iter(parts.values())).shape
    if any(value.shape != shape for value in parts.values()):
        raise ValueError(f"the parts differ in shape: { {k: v.shape for k, v in parts.items()} }.")
    keep = np.ones(shape, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    centred = _centred_parts(parts, keep)
    whole = sum(centred.values())
    size = (whole**2).sum(axis=0)
    defined = size > 0
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, c in centred.items():
            out[f"share_{name}"] = np.where(defined, (c * whole).sum(axis=0) / np.where(defined, size, 1.0), np.nan)
        for name, c in centred.items():
            out[f"norm_{name}"] = np.where(defined, np.sqrt((c**2).sum(axis=0) / np.where(defined, size, 1.0)), np.nan)
    return pd.DataFrame(out)


def gene_cosine(prediction: np.ndarray, true: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """``[G]`` per gene, the cosine of the predicted and the true values over the cells, both minus ``center`` ``[G]``
    (the training mean: predicting it scores 0; None: raw); NaN where either does not move.
    """
    prediction, true = np.asarray(prediction, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if center is not None:
        prediction, true = prediction - np.asarray(center), true - np.asarray(center)
    norm = np.linalg.norm(prediction, axis=0) * np.linalg.norm(true, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, (prediction * true).sum(axis=0) / np.where(norm > 0, norm, 1.0), np.nan)


def cell_folds(data_list, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Per graph ``[N_g]``: each cell's fold, ``0 .. n_folds - 1``, as equal in size as can be and in random order.

    From a private generator; the same arguments give the same folds. Hiding one fold at a time hides every cell once,
    each time a share ``1 / n_folds`` of the slide -- the training rate, for ``n_folds = round(1 / mask_percentage)``.
    """
    if int(n_folds) < 2:
        raise ValueError(f"at least 2 folds are needed (one hidden, the rest visible), got {n_folds}.")
    rng = np.random.default_rng(seed)
    return [rng.permutation(np.arange(int(data.num_nodes)) % int(n_folds)) for data in data_list]


@contextmanager
def _masks_kept(data_list):
    """Put back every graph's ``mask`` and ``gene_mask`` as they were (or remove them where there were none) on exit."""
    saved = [(getattr(data, "mask", None), getattr(data, "gene_mask", None)) for data in data_list]
    try:
        yield
    finally:
        for data, values in zip(data_list, saved, strict=True):
            for name, value in zip(("mask", "gene_mask"), values, strict=True):
                if value is not None:
                    setattr(data, name, value)
                elif name in data:
                    del data[name]


def _hide_fold(data_list, folds: list[np.ndarray], fold: int) -> None:
    """Hide every cell of ``fold`` whole, written as ``draw_masks(kind="cells")`` does: ``gene_mask`` and ``mask``."""
    for data, assigned in zip(data_list, folds, strict=True):
        hide = torch.as_tensor(assigned == fold)
        data.gene_mask = hide[:, None].expand(int(data.num_nodes), data.x.shape[1]).clone()
        data.mask = hide.clone()


@dataclass
class FoldParts:
    """Every cell's prediction from the fold it was hidden in, rows the cells, columns the genes.

    Attributes
    ----------
    true
        ``[N, G]`` the expression.
    prediction
        ``[N, G]`` the head's prediction, the sum of the parts.
    parts
        ``{part: [N, G]}`` what the prediction adds up from.
    graph
        ``[N]`` each cell's graph: its index in the list collected from.
    pos
        ``[N, D]`` its position (the graph's ``pos``, in its units).
    cells
        ``[N]`` its name, as the graphs carry it (``obs_names``).
    references
        ``{name: [N, G]}`` the model-free references of the same passes, if asked for.
    """

    true: np.ndarray
    prediction: np.ndarray
    parts: dict[str, np.ndarray]
    graph: np.ndarray
    pos: np.ndarray
    cells: np.ndarray
    references: dict[str, np.ndarray]


def collect_fold_parts(
    module,
    data_list,
    *,
    n_folds: int,
    prediction_task: str,
    prediction_level: str,
    batch_size: int,
    device,
    head: str = "global",
    seed: int = 0,
    spatial_unit_um: float | None = None,
    references: dict | None = None,
) -> FoldParts:
    """Hide each fold of :func:`cell_folds` in turn and keep every hidden cell's prediction of ``head``, taken apart.

    ``module`` must make ``head`` a sum of parts (``StepOutput.parts``: the additive decoder, the residual decoder; a
    global-only model's own-cell skip too, though its parts are the cell and its context, not two scales). The graphs'
    own masks are put back afterwards. Each graph must carry ``pos`` and ``obs_names``; a cell that ``pad_batch`` left
    out of its fold's pass has no row. ``references`` (as for
    :func:`~interscale.evaluation.masked_probing.collect_head_predictions`, with ``spatial_unit_um``) adds the
    model-free references of the same passes.
    """
    for data in data_list:
        if getattr(data, "pos", None) is None or not isinstance(getattr(data, "obs_names", None), torch.Tensor):
            raise ValueError("every graph needs `pos` and `obs_names` to place a prediction back at its cell.")
    where = {}
    for g, data in enumerate(data_list):
        for j, name in enumerate(np.rint(data.obs_names.detach().cpu().double().numpy()).astype(np.int64)):
            where[int(name)] = (g, j)
    if len(where) != sum(int(data.num_nodes) for data in data_list):
        raise ValueError("the graphs' `obs_names` are not unique across the list.")
    folds = cell_folds(data_list, n_folds, seed)
    found: dict[str, list[np.ndarray]] = {"true": [], "prediction": [], "cells": []}
    part_rows: dict[str, list[np.ndarray]] = {}
    reference_rows: dict[str, list[np.ndarray]] = {}
    with _masks_kept(data_list):
        for fold in range(int(n_folds)):
            _hide_fold(data_list, folds, fold)
            pred = collect_head_predictions(
                module,
                data_list,
                prediction_task=prediction_task,
                prediction_level=prediction_level,
                mask_strategy="gene",
                batch_size=batch_size,
                device=device,
                spatial_unit_um=spatial_unit_um,
                references=references,
            )
            if not pred.parts or head not in pred.parts:
                raise ValueError(
                    f"the head {head!r} is not a sum of parts (heads with parts: {sorted(pred.parts or {})}); only a "
                    "decomposable prediction can be split (model.decoder.additive, model.decoder.global_residual)."
                )
            if pred.cells is None:
                raise ValueError("the predictions carry no cell names; the graphs need `obs_names`.")
            rows = pred.hidden.all(axis=1)
            found["true"].append(pred.true[rows])
            found["prediction"].append(pred.heads[head][rows])
            found["cells"].append(pred.cells[rows])
            for name, value in pred.parts[head].items():
                part_rows.setdefault(name, []).append(value[rows])
            for name, value in pred.heads.items():
                if name.startswith("ref_"):
                    reference_rows.setdefault(name, []).append(value[rows])
    cells = np.concatenate(found["cells"])
    names = np.rint(cells).astype(np.int64)
    if len(np.unique(names)) != len(names):
        raise RuntimeError("a cell was predicted in two folds; the folds do not partition the cells.")
    placed = np.array([where[int(name)] for name in names], dtype=np.int64).reshape(-1, 2)
    pos = np.zeros((len(placed), data_list[0].pos.shape[1]))
    for g in np.unique(placed[:, 0]):
        rows = placed[:, 0] == g
        pos[rows] = data_list[g].pos.detach().cpu().double().numpy()[placed[rows, 1]]
    return FoldParts(
        true=np.concatenate(found["true"]),
        prediction=np.concatenate(found["prediction"]),
        parts={name: np.concatenate(value) for name, value in part_rows.items()},
        graph=placed[:, 0],
        pos=pos,
        cells=cells,
        references={name: np.concatenate(value) for name, value in reference_rows.items()},
    )


# ------------------------------------------------------------------------------- the whole


@dataclass
class GeneScales:
    """The curves of every map over one set of neighbours, the null of the expression's, and the per-gene table.

    Attributes
    ----------
    genes
        The gene names, the columns of every map.
    curves
        ``{map: MoranCurves}``: :data:`TRUTH` (the expression) and every other map handed over (a model's parts).
    null
        The permutation null of the expression's curves.
    table
        Per gene: :func:`length_scales` of the expression, and whatever was joined (shares, cosines).
    """

    genes: list[str]
    curves: dict[str, MoranCurves]
    null: MoranNull
    table: pd.DataFrame

    @property
    def ks(self) -> np.ndarray:
        return self.curves[TRUTH].ks

    @property
    def reach_um(self) -> np.ndarray:
        return self.curves[TRUTH].reach_um


def gene_scales(
    maps: dict[str, np.ndarray],
    graph,
    pos,
    genes,
    *,
    ks=DEFAULT_KS,
    n_centres: int | None = 2000,
    seed: int = 0,
    n_permutations: int = 10,
    unit_um: float = 1.0,
    z: float = 3.0,
    columns: pd.DataFrame | None = None,
    dtype=np.float32,
) -> GeneScales:
    """Moran's I curves of every map, the expression's length scales, and ``columns`` joined per gene.

    Parameters
    ----------
    maps
        ``{name: [N, G]}`` over the same cells; :data:`TRUTH` is the expression (required), the others e.g. a model's
        parts.
    graph, pos
        ``[N]`` each cell's graph and ``[N, D]`` its position (in units of ``unit_um`` µm).
    genes
        The ``G`` gene names.
    ks
        Neighbourhood sizes, increasing.
    n_centres, seed
        Centres sampled per graph (:func:`sample_neighbours`), the same for every map, and the seed of the draw and the
        null.
    n_permutations
        Shuffles of the null.
    z
        Null SDs a first shell must clear to count as variable.
    columns
        Per-gene columns to join (rows in the order of ``genes``), e.g. :func:`part_shares`.
    dtype
        Of the products (:func:`moran_by_rank`).
    """
    if TRUTH not in maps:
        raise ValueError(f"maps needs the expression under {TRUTH!r}, got {sorted(maps)}.")
    genes = list(genes)
    ks = [int(k) for k in ks]
    neighbours = sample_neighbours(pos, graph, k_max=max(ks), n_centres=n_centres, seed=seed)
    curves = {
        name: moran_curves(values, graph, neighbours, ks, unit_um=unit_um, dtype=dtype) for name, values in maps.items()
    }
    null = permutation_null(
        maps[TRUTH], graph, neighbours, ks, n_permutations=n_permutations, seed=seed + 1, dtype=dtype
    )
    table = length_scales(curves[TRUTH], null, genes, z=z)
    if columns is not None:
        if len(columns) != len(genes):
            raise ValueError(f"columns has {len(columns)} rows, the genes are {len(genes)}.")
        table = table.join(columns.set_axis(table.index, axis=0))
    return GeneScales(genes, curves, null, table)


def _rank_ready(values: pd.Series) -> pd.Series:
    """``inf`` (a scale beyond the sizes) above every finite value, so a rank correlation keeps it as the longest."""
    finite = values[np.isfinite(values)]
    top = float(finite.max()) if len(finite) else 1.0
    return values.replace(np.inf, 2.0 * abs(top) + 1.0)


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    use = a.notna() & b.notna()
    if use.sum() < 3 or a[use].nunique() < 2 or b[use].nunique() < 2:
        return float("nan"), int(use.sum())
    return float(spearmanr(_rank_ready(a[use]), _rank_ready(b[use]))[0]), int(use.sum())


def scale_tables(
    result: GeneScales,
    *,
    global_part: str = "global",
    planted_um: pd.Series | None = None,
    design: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """The tables of a run's sections.

    * ``gene_scales`` -- the per-gene table (``planted_um`` and ``design`` joined when given), the genes with the
      largest global share first;
    * ``gene_scales_summary`` -- one row: genes, variable genes, and over the variable genes the Spearman correlation of
      ``ell_um`` with the global part's share (``rho_ell_share``: positive = the longer a gene's scale, the more of it
      the global part carries) and the share of them it carries most of (``global_majority``); with ``planted_um``, the
      instrument against the truth (``rho_ell_planted``: the measured scale against the planted one) and the model
      against it (``rho_share_planted``);
    * ``gene_scales_by_design`` -- with ``design`` (a class per gene, e.g. local / global / none from a synthetic
      dataset's design): per class the genes, how many are variable, the median measured scale, the median global share
      and the share of genes the global part carries most of.

    ``planted_um`` and ``design`` are indexed by gene; genes they lack are NaN there.
    """
    table = result.table.copy()
    share = f"share_{global_part}"
    if planted_um is not None:
        table["planted_um"] = pd.to_numeric(planted_um.reindex(table.index), errors="coerce")
    if design is not None:
        table["design"] = design.reindex(table.index).astype(object)
    variable = table["pattern"] == VARIABLE
    summary = {
        "n_genes": len(table),
        "n_variable": int(variable.sum()),
        "n_anti": int((table["pattern"] == ANTI).sum()),
    }
    if share in table:
        rho, n = _spearman(table.loc[variable, "ell_um"], table.loc[variable, share])
        summary.update(
            {
                "rho_ell_share": rho,
                "n_rho": n,
                "global_majority": float((table.loc[variable, share] > 0.5).mean()) if variable.any() else float("nan"),
                "median_share": float(table.loc[variable, share].median()) if variable.any() else float("nan"),
            }
        )
    if "planted_um" in table:
        rho, n = _spearman(table.loc[variable, "ell_um"], table.loc[variable, "planted_um"])
        summary.update({"rho_ell_planted": rho, "n_planted": n})
        if share in table:
            summary["rho_share_planted"] = _spearman(table.loc[variable, share], table.loc[variable, "planted_um"])[0]
    out = {}
    order = table.sort_values(share, ascending=False, na_position="last") if share in table else table
    out["gene_scales"] = order
    out["gene_scales_summary"] = pd.DataFrame([summary])
    if "design" in table:
        rows = []
        for label, part in table.groupby("design", sort=False, dropna=True):
            var = part["pattern"] == VARIABLE
            row = {
                "design": label,
                "n_genes": len(part),
                "n_variable": int(var.sum()),
                "median_ell_um": float(part.loc[var, "ell_um"].median()) if var.any() else float("nan"),
            }
            if share in part:
                row["median_share"] = float(part[share].median())
                row["global_majority"] = float((part[share] > 0.5).mean())
            rows.append(row)
        out["gene_scales_by_design"] = pd.DataFrame(rows).set_index("design")
    return out


# ------------------------------------------------------------------------------- figures

#: The figures' colours: the dataviz reference palette on its light surface -- categorical slots 1 and 2 for the two
#: parts (validated all-pairs, CVD Delta E 24.7), the secondary ink for the expression, chrome for the rest.
COLOURS = {
    "local": "#2a78d6",
    "global": "#eb6834",
    TRUTH: "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "surface": "#fcfcfb",
}
#: The diverging map of the maps: blue and red poles, a neutral grey midpoint (the reference diverging pair).
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
#: A gene's class in :func:`assignment` when its expression is not spatially variable.
NOT_VARIABLE = "not variable"


def assignment(result: GeneScales, *, global_part: str = "global", classes: pd.Series | None = None) -> pd.Series:
    """Each gene's class for the colours: ``classes`` (indexed by gene) when given, else the part that carries most of
    its prediction -- ``global`` where its global share is above 0.5, else ``local``. A gene whose expression is not
    spatially variable is :data:`NOT_VARIABLE` either way, as is one without a class.
    """
    table = result.table
    if classes is not None:
        found = classes.reindex(table.index).astype(object)
    elif f"share_{global_part}" in table:
        share = table[f"share_{global_part}"]
        found = pd.Series(np.where(share > 0.5, "global", "local"), index=table.index, dtype=object).where(
            share.notna()
        )
    else:
        raise ValueError(f"no share_{global_part} column to colour by, and no classes given.")
    found = found.where(table["pattern"] == VARIABLE)
    return found.fillna(NOT_VARIABLE)


def pick_genes(result: GeneScales, n: int = 8, *, global_part: str = "global") -> list[str]:
    """``n`` variable genes for :func:`plot_part_curves`: the ones the global part carries most of, then the ones it
    carries least of (by the first shell's Moran's I without shares).
    """
    table = result.table[result.table["pattern"] == VARIABLE]
    column = f"share_{global_part}"
    if column not in table:
        return list(table["i_first"].sort_values(ascending=False).index[:n])
    ranked = table[column].dropna().sort_values()
    first = list(ranked.index[::-1][: n - n // 2])
    return list(dict.fromkeys(first + list(ranked.index[: n // 2])))


def _style(ax) -> None:
    """Recessive chrome: the surface, hairline axes and grid, muted ticks."""
    ax.set_facecolor(COLOURS["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOURS["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=COLOURS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOURS["muted"], labelcolor=COLOURS["ink_secondary"], labelsize=8)


def _k_axis(ax, curves: MoranCurves, *, top: bool = True) -> None:
    """Log-2 x axis with a tick at every ``k``; on top the median distance to the ``k``-th neighbour in µm."""
    from matplotlib.ticker import NullLocator

    ax.set_xscale("log", base=2)
    ax.set_xticks(curves.ks)
    ax.set_xticklabels([str(int(k)) for k in curves.ks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(curves.ks[0] / 1.15, curves.ks[-1] * 1.15)
    if top:
        upper = ax.secondary_xaxis("top")
        upper.set_xticks(curves.ks)
        upper.set_xticklabels([f"{r:.0f}" if np.isfinite(r) else "" for r in curves.reach_um])
        upper.xaxis.set_minor_locator(NullLocator())
        upper.tick_params(colors=COLOURS["muted"], labelcolor=COLOURS["ink_secondary"], labelsize=7)
        upper.spines["top"].set_color(COLOURS["axis"])
        upper.set_xlabel("median distance to the k-th neighbour (µm)", color=COLOURS["ink_secondary"], fontsize=8)


def _null_band(ax, result: GeneScales, form: str, columns=None) -> None:
    """The permutation null, ±2 SD around its mean: per gene for one column, else the median over the genes."""
    mean, sd = result.null.mean[form], result.null.sd[form]
    if columns is not None:
        mean, sd = mean[:, columns], sd[:, columns]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre, width = np.nanmedian(mean, axis=1), 2.0 * np.nanmedian(sd, axis=1)
    ax.fill_between(result.ks, centre - width, centre + width, color=COLOURS["muted"], alpha=0.18, linewidth=0)
    ax.axhline(0.0, color=COLOURS["axis"], linewidth=0.8)


def plot_scale_curves(
    result: GeneScales,
    *,
    global_part: str = "global",
    classes: pd.Series | None = None,
    form: str = "shell",
    title: str | None = None,
):
    """Plot A: every gene's Moran's I curve of its expression, coloured by the part of the model that carries it.

    One thin line per gene and a bold median per class; the classes are :func:`assignment`'s (the model's split, or the
    ``classes`` given, e.g. a synthetic dataset's design for a check without a model). Genes whose expression is not
    spatially variable stay grey. A model that separates the scales colours the curves that stay up ``global`` and the
    ones that drop ``local``. Returns the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.lines import Line2D

    curves = result.curves[TRUTH]
    values = curves.form(form)
    groups = assignment(result, global_part=global_part, classes=classes).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    fig.patch.set_facecolor(COLOURS["surface"])
    _style(ax)
    _null_band(ax, result, form)
    handles = []
    grey = np.flatnonzero(~np.isin(groups, ("local", "global")))
    if len(grey):
        ax.plot(curves.ks, values[:, grey], color=COLOURS["muted"], linewidth=0.5, alpha=0.3)
        what = "not spatially variable" if classes is None else "not variable, or no local / global class"
        handles.append(Line2D([], [], color=COLOURS["muted"], linewidth=1.2, label=f"{what}: {_count(len(grey))}"))
    for label in ("local", "global"):
        columns = np.flatnonzero(groups == label)
        if not len(columns):
            continue
        colour = COLOURS[label]
        ax.plot(curves.ks, values[:, columns], color=colour, linewidth=0.7, alpha=0.35)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values[:, columns], axis=1)
        ax.plot(
            curves.ks,
            median,
            color=colour,
            linewidth=2.4,
            path_effects=[patheffects.withStroke(linewidth=4.4, foreground=COLOURS["surface"])],
        )
        name = f"{label} part carries most" if classes is None else f"{label} class"
        handles.append(
            Line2D([], [], color=colour, linewidth=2.4, label=f"{name}: {_count(len(columns))} (bold: median)")
        )
    handles.append(Line2D([], [], color=COLOURS["muted"], alpha=0.35, linewidth=6, label="permutation null, ±2 SD"))
    _k_axis(ax, curves)
    what = "neighbours ranked k/2+1 … k" if form == "shell" else "neighbours 1 … k"
    ax.set_xlabel("neighbours k", color=COLOURS["ink_secondary"], fontsize=9)
    ax.set_ylabel(f"Moran's I ({what})", color=COLOURS["ink_secondary"], fontsize=9)
    legend = ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8
    )
    for text in legend.get_texts():
        text.set_color(COLOURS["ink"])
    if title:
        ax.set_title(title, color=COLOURS["ink"], fontsize=10, loc="left", pad=28)
    fig.tight_layout()
    return fig


def _count(n: int) -> str:
    return f"{n} gene" if n == 1 else f"{n} genes"


def plot_part_curves(
    result: GeneScales,
    genes,
    *,
    parts=("local", "global"),
    global_part: str = "global",
    form: str = "shell",
    ncols: int = 4,
    title: str | None = None,
):
    """Plot B: per gene, the Moran's I curve of its expression and of each part of the prediction, beside the null.

    A separated model has a local part that drops with the neighbourhood and a global part that stays up. Each panel
    names the gene's global share and measured scale (and the planted one, when the table has ``planted_um``): a part's
    curve is of its z-scored map, so its size is in the share, not in the curve. Returns the figure.
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    genes = [g for g in genes if g in result.genes]
    if not genes:
        raise ValueError("none of the genes is among the result's.")
    ncols = max(1, min(int(ncols), len(genes)))
    nrows = math.ceil(len(genes) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows + 0.9), sharex=True, sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(COLOURS["surface"])
    table = result.table
    for k, gene in enumerate(genes):
        ax = axes[k // ncols][k % ncols]
        _style(ax)
        j = result.genes.index(gene)
        _null_band(ax, result, form, columns=[j])
        ax.plot(result.ks, result.curves[TRUTH].form(form)[:, j], color=COLOURS[TRUTH], linewidth=2)
        for part in parts:
            if part in result.curves:
                ax.plot(
                    result.ks,
                    result.curves[part].form(form)[:, j],
                    color=COLOURS.get(part, COLOURS["muted"]),
                    linewidth=2,
                )
        _k_axis(ax, result.curves[TRUTH], top=False)
        notes = []
        if f"share_{global_part}" in table:
            notes.append(f"global share {table.loc[gene, f'share_{global_part}']:.2f}")
        ell = table.loc[gene, "ell_um"]
        notes.append(f"ℓ {ell:.0f} µm" if np.isfinite(ell) else ("ℓ beyond the sizes" if ell == np.inf else "no scale"))
        if "planted_um" in table and np.isfinite(table.loc[gene, "planted_um"]):
            notes.append(f"planted {table.loc[gene, 'planted_um']:.0f}")
        ax.set_title(f"{gene}\n{' · '.join(notes)}", color=COLOURS["ink"], fontsize=8, loc="left")
    for k in range(len(genes), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("neighbours k", color=COLOURS["ink_secondary"], fontsize=8)
    for row in axes:
        row[0].set_ylabel("Moran's I", color=COLOURS["ink_secondary"], fontsize=8)
    handles = [Line2D([], [], color=COLOURS[TRUTH], linewidth=2, label="expression")]
    handles += [
        Line2D([], [], color=COLOURS.get(part, COLOURS["muted"]), linewidth=2, label=f"{part} part")
        for part in parts
        if part in result.curves
    ]
    handles.append(Line2D([], [], color=COLOURS["muted"], alpha=0.35, linewidth=6, label="null, ±2 SD"))
    if title:
        fig.suptitle(title, color=COLOURS["ink"], fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(COLOURS["ink"])
    return fig


def plot_part_maps(
    fold: FoldParts,
    genes,
    all_genes,
    *,
    graph: int | None = None,
    parts=("local", "global"),
    shares: pd.DataFrame | None = None,
    global_part: str = "global",
    title: str | None = None,
):
    """The maps behind Plot B on one slide: per gene (rows) the expression, the prediction and each part (columns).

    Each map is z-scored within the slide and clipped at ±2.5 SD, so the parts show their pattern on one scale; their
    size is the share in the row label (``shares``: :func:`part_shares` or the result's table). ``graph`` picks the
    slide (default: the first in ``fold``). Returns the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    all_genes = list(all_genes)
    genes = [g for g in genes if g in all_genes]
    if not genes:
        raise ValueError("none of the genes is among all_genes.")
    graph = int(fold.graph[0]) if graph is None else int(graph)
    rows = fold.graph == graph
    if not rows.any():
        raise ValueError(f"no cell of graph {graph} among the predictions.")
    xy = fold.pos[rows][:, :2]
    columns = [("expression", fold.true), ("prediction", fold.prediction)]
    columns += [(f"{part} part", fold.parts[part]) for part in parts if part in fold.parts]
    cmap = LinearSegmentedColormap.from_list("gene_scales_diverging", DIVERGING)
    side = 2.2
    fig, axes = plt.subplots(
        len(genes), len(columns), figsize=(side * len(columns) + 0.8, side * len(genes) + 0.5), squeeze=False
    )
    fig.patch.set_facecolor(COLOURS["surface"])
    span = np.ptp(xy, axis=0).max() if len(xy) > 1 else 1.0
    spacing = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]) if len(xy) > 1 else 1.0
    size = (72.0 * side * 0.8 * spacing / max(span, 1e-12)) ** 2  # one cell's side in points, squared
    image = None
    for r, gene in enumerate(genes):
        j = all_genes.index(gene)
        for c, (name, values) in enumerate(columns):
            ax = axes[r][c]
            v = values[rows, j].astype(np.float64)
            sd = v.std()
            v = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
            image = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=np.clip(v, -2.5, 2.5),
                cmap=cmap,
                vmin=-2.5,
                vmax=2.5,
                s=size,
                marker="s",
                linewidths=0,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor(COLOURS["surface"])
            if r == 0:
                ax.set_title(name, color=COLOURS["ink"], fontsize=9)
            if c == 0:
                label = gene
                if shares is not None and f"share_{global_part}" in shares and gene in shares.index:
                    label += f"\nglobal share {shares.loc[gene, f'share_{global_part}']:.2f}"
                ax.set_ylabel(label, color=COLOURS["ink"], fontsize=8)
    bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.02)
    bar.set_label("z-score within the slide", color=COLOURS["ink_secondary"], fontsize=8)
    bar.ax.tick_params(colors=COLOURS["muted"], labelcolor=COLOURS["ink_secondary"], labelsize=7)
    bar.outline.set_visible(False)
    if title:
        fig.suptitle(title, color=COLOURS["ink"], fontsize=10, x=0.01, ha="left")
    return fig
