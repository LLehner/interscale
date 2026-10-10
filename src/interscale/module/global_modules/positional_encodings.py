"""Positional encodings (PEs) for the global (transformer) component.

Selected by ``model.global_component.parameters.pe``; the staged design is in
``.claude/PE_plan.md``. Three kinds: *node* encodings (``pe.node``), summed onto each cell's token
before ``pad_batch`` so that subsampling, node masking and padding carry them for free; attention
*biases* (``pe.bias``), added to the attention logits of every pair of tokens and merged into the
attention mask where it is built; and the *rotary* encoding (``pe.rotary``), which turns every
query and key by its cell's position inside each attention layer.

Three properties every encoding keeps:

* **Off is today's model.** With ``pe.node: []`` nothing is built -- no parameters, no state-dict
  keys, no RNG draws -- so a PE-free run is the run from before PEs existed.
* **On changes no other initial weight.** The encoders are built inside a forked RNG, so a PE run
  and a PE-free run at the same seed share every other initial weight and the batch order.
  Node encodings and biases also share the first forward pass: their output (a projection, or a
  bias table) starts at zero, so whatever differs later is what the PE was used for. The rotary
  encoding cannot: a rotation has no zero, and it changes the attention from the first step.
* **Augmentation has its own generator.** ``rotate_train`` and LapPE's sign flips draw from a
  generator seeded from ``optim.seed``, never from the global torch RNG, which also decides the
  training batch order. It belongs to the :class:`CoordinateFrame` the coordinate encodings share,
  so that the node encodings and the rotary one see a graph turned by the same angle.

Inputs that are expensive to compute -- Laplacian eigenvectors -- are precomputed once per graph by
:func:`attach_positional_inputs` when the graphs are built, and ride on the ``Data`` object.

Adding an encoding is one :func:`register_pe` entry plus one config sub-block in
``get_global_component_cfg``; nothing in the transformer changes.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import scatter

from interscale.tl.padding import pad_like
from interscale.tl.positional import laplacian_pe, laplacian_spectrum, point_set_diameter, random_walk_pe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PESpec:
    """What the rest of the code needs to know about one encoding.

    ``kind`` is ``"node"`` (listed under ``pe.node``), ``"bias"`` (under ``pe.bias``) or
    ``"rotary"`` (under ``pe.rotary``). ``requires`` names the ``Data`` attribute the encoder reads
    (``"pos"`` for coordinates). ``build(pe_cfg, *dims)`` returns the encoder -- ``dims`` is
    ``n_embed`` for a node encoding (its output width), ``n_heads`` for a bias (one value per head)
    and ``n_heads, head_dim`` for a rotary encoding (the width it rotates). ``validate(pe_cfg)``
    raises on settings that would fail mid-run, and runs at config load. ``precompute(data, pe_cfg)``,
    when set, returns the ``requires`` attribute for one graph -- or a dict of attributes by name,
    for an encoding that needs several -- and :func:`attach_positional_inputs` stores it.
    """

    kind: str
    requires: str
    build: Callable
    validate: Callable
    precompute: Callable | None = None


PE_REGISTRY: dict[str, PESpec] = {}


def register_pe(name: str, *, kind: str, requires: str, validate: Callable, precompute: Callable | None = None):
    """Register ``build(pe_cfg, width) -> nn.Module`` as the encoding ``name``."""

    def decorator(build):
        if name in PE_REGISTRY:
            raise ValueError(f"positional encoding '{name}' is already registered")
        PE_REGISTRY[name] = PESpec(kind=kind, requires=requires, build=build, validate=validate, precompute=precompute)
        return build

    return decorator


def _zero_(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class NaivePE(nn.Module):
    """SpaFormer's naive PE: an MLP of the cell's centred coordinates, in units of ``length_scale``.

    A fixed physical scale rather than SpaFormer's per-field-of-view min-max normalisation: the
    graphs here differ in size, and min-max would encode the same distance differently in each.
    """

    def __init__(self, n_embed: int, hidden_dim: int, length_scale: float):
        super().__init__()
        self.length_scale = float(length_scale)
        self.mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), _zero_(nn.Linear(hidden_dim, n_embed)))

    def forward(self, pos_um: torch.Tensor) -> torch.Tensor:
        return self.mlp(pos_um / self.length_scale)


class SinusoidalPE(nn.Module):
    """2D sinusoidal features of the cell's coordinates, projected to the token width.

    Per axis, sin and cos at ``dim / 4`` wavelengths spaced geometrically from ``min_wavelength``
    to ``max_wavelength`` (µm); features are ordered ``[x: sin, cos | y: sin, cos]``. DETR's
    ``10000 ** (4i / d)`` schedule assumes integer positions; for continuous coordinates the same
    family is better stated by its wavelength range, whose ends have a physical meaning -- about a
    cell diameter, and about the extent of a graph.
    """

    def __init__(self, n_embed: int, dim: int, min_wavelength: float, max_wavelength: float):
        super().__init__()
        wavelengths = torch.logspace(math.log10(min_wavelength), math.log10(max_wavelength), dim // 4)
        # Persistent, so a reloaded model keeps the frequencies its projection was trained against.
        self.register_buffer("omega", 2 * math.pi / wavelengths)
        self.proj = _zero_(nn.Linear(dim, n_embed))

    def features(self, pos_um: torch.Tensor) -> torch.Tensor:
        """``[N, dim]`` sin/cos features of ``[N, 2]`` coordinates in µm."""
        angles = pos_um.unsqueeze(-1) * self.omega  # [N, 2, dim / 4]
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)

    def forward(self, pos_um: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(pos_um))


class LapPE(nn.Module):
    """Laplacian eigenvectors of the neighbour graph (precomputed), projected to the token width.

    An eigenvector is only defined up to sign, so with ``sign_flip`` each of the ``k`` columns is
    multiplied by a random sign per graph in training -- the standard way to make the projection
    not depend on an arbitrary choice. Evaluation sees the canonical signs ``laplacian_pe`` fixes.
    """

    def __init__(self, n_embed: int, k: int, sign_flip: bool):
        super().__init__()
        self.k = int(k)
        self.sign_flip = bool(sign_flip)
        self.proj = _zero_(nn.Linear(self.k, n_embed))

    def _check(self, eigvecs: torch.Tensor) -> torch.Tensor:
        if eigvecs.shape[1] != self.k:
            raise ValueError(
                f"lap_pe has {eigvecs.shape[1]} eigenvectors per cell but pe.lap.k is {self.k}; the graphs "
                "were built with a different config. Rebuild them (attach_positional_inputs)."
            )
        return eigvecs

    def augment(self, eigvecs: torch.Tensor, batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Flip each eigenvector's sign at random, independently per graph. Training only."""
        eigvecs = self._check(eigvecs)
        if not self.sign_flip:
            return eigvecs
        n_graphs = int(batch.max().item()) + 1
        signs = torch.randint(0, 2, (n_graphs, self.k), generator=generator).to(eigvecs) * 2 - 1
        return eigvecs * signs[batch]

    def forward(self, eigvecs: torch.Tensor) -> torch.Tensor:
        return self.proj(self._check(eigvecs))


class RWPE(nn.Module):
    """Random-walk return probabilities (precomputed), normalised, projected to the token width.

    The probabilities are small and fall with the number of steps, so a BatchNorm over the batch's
    cells puts the columns on one scale before the projection, as GraphGPS does for its RWSE. Its
    running statistics are what evaluation uses; they only move in ``train()`` mode, and the online
    probe runs in ``eval()``.

    ``eps`` is 1e-8, not BatchNorm's 1e-5: the spread of a late step's return probability across
    cells is ~5e-3 on a 14-neighbour graph and smaller on denser ones, so a variance of ~1e-5 is
    typical and the default would divide by ~sqrt(2x) of it -- shrinking exactly the columns the
    normalisation is for (to 84% at step 16 there, and further as density rises). An exactly
    constant column, such as step 1, still comes out at zero.
    """

    def __init__(self, n_embed: int, steps: int):
        super().__init__()
        self.steps = int(steps)
        self.norm = nn.BatchNorm1d(self.steps, eps=1e-8)
        self.proj = _zero_(nn.Linear(self.steps, n_embed))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.shape[1] != self.steps:
            raise ValueError(
                f"rw_pe has {probs.shape[1]} steps per cell but pe.rw.steps is {self.steps}; the graphs "
                "were built with a different config. Rebuild them (attach_positional_inputs)."
            )
        return self.proj(self.norm(probs))


def _knot_lookup(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``table`` ``[heads, K]`` read at knot positions ``t`` (in ``[0, K - 1]``), linear in between: ``[*t.shape, heads]``.

    Each value touches only its two neighbouring knots, so this is two embedding lookups rather
    than an expansion into ``K`` kernel features.
    """
    last = table.shape[1] - 1
    lower = t.floor().clamp(max=last - 1)
    frac = (t - lower).unsqueeze(-1)
    values = table.t()  # [K, heads]
    lo = nn.functional.embedding(lower.long(), values)
    hi = nn.functional.embedding(lower.long() + 1, values)
    return lo + frac * (hi - lo)


class DistanceBias(nn.Module):
    """A learned function of the distance between two cells, one per head, added to their logit.

    ``b_h(d)`` is piecewise linear between ``num_kernels`` knots spaced evenly over
    ``[0, max_dist]`` (µm), and constant beyond. That is the plan's "K kernels -> Linear(K ->
    heads)" with triangular kernels instead of Gaussians: each distance touches only its two
    neighbouring knots, so the bias is two lookups into a ``[heads, K]`` table rather than a
    ``[B, S, S, K]`` kernel expansion -- about 1.6 GB of activations per batch of four 2500-spot
    graphs at K = 16. The table *is* the learned profile; :meth:`profile` reads it out.

    The table starts at zero, so switching the bias on changes nothing at step 0. One table serves
    every layer, as in Graphormer. With the long-range mask on, the pairs inside the mask never
    reach the softmax, so the profile below the mask's reach is never trained.
    """

    def __init__(self, n_heads: int, num_kernels: int, max_dist: float):
        super().__init__()
        self.num_kernels = int(num_kernels)
        # A buffer, so the checkpoint carries the range the table was trained on: with `max_dist`
        # derived from the data, a model loaded onto other slides would otherwise re-derive a
        # different range and read every learned knot at the wrong distance.
        self.register_buffer("max_dist", torch.tensor(float(max_dist)))
        self.table = nn.Parameter(torch.zeros(int(n_heads), self.num_kernels))

    def extra_repr(self) -> str:
        return f"num_kernels={self.num_kernels}, max_dist={float(self.max_dist):.1f} um"

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """``[B, S, S]`` distances in µm to ``[B, heads, S, S]`` biases."""
        last = self.num_kernels - 1
        t = (dist * (last / self.max_dist)).clamp(max=last)
        return _knot_lookup(self.table, t).permute(0, 3, 1, 2)

    def profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(knots [K] in µm, values [heads, K])`` -- each head's bias at each knot."""
        return torch.linspace(0.0, float(self.max_dist), self.num_kernels), self.table.detach().cpu().clone()


class LinearDistanceBias(nn.Module):
    """One learned slope per head: ``b_h(d) = w_h * d``, with ``d`` in millimetres.

    The ALiBi form. A negative slope makes a head's attention fall off with distance, a positive one
    favours far cells; it cannot single out a particular distance, which is what the ``profile``
    kind is for. No range to set, and ``heads`` parameters in total. Distances are taken in mm only
    so that a slope of order 1 matters across a slide; Adam's steps do not depend on that unit.

    The slopes start at zero, so switching the bias on changes nothing at step 0.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.slope = nn.Parameter(torch.zeros(int(n_heads)))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """``[B, S, S]`` distances in µm to ``[B, heads, S, S]`` biases."""
        return (dist / 1000.0).unsqueeze(1) * self.slope.view(1, -1, 1, 1)

    def profile(self, max_dist: float = 1000.0) -> tuple[torch.Tensor, torch.Tensor]:
        """``(knots [2] in µm, values [heads, 2])`` -- the line at 0 and ``max_dist``, as for ``profile``."""
        knots = torch.tensor([0.0, float(max_dist)])
        return knots, self.slope.detach().cpu()[:, None] * (knots / 1000.0)


class SpectralBias(nn.Module):
    """A learned filter of the neighbour graph's spectrum per head -- a diffusion-like kernel as bias.

    ``b_h(j, l) = sum_i h_h(lambda_i) * N * v_i[j] * v_i[l]`` over the ``k`` lowest non-trivial
    eigenpairs ``(lambda_i, v_i)`` of the symmetric normalised Laplacian LapPE uses, for a graph of
    ``N`` cells. With ``h(lambda) = exp(-t * lambda)`` that is the heat kernel truncated to ``k``
    modes, the matrix diffusion distances are made of. Here each head learns its own ``h_h``,
    piecewise linear in ``log(lambda)`` between ``num_knots`` knots spaced evenly over
    ``[min_eigval, max_eigval]`` and flat beyond: low-pass (smooth, long-range), band-pass, or
    anything between.

    It is the transformer-native form of what BasisNet is shown to approximate (Lim et al. 2022,
    Prop. 4). ``h`` depends on the eigenvalue alone, so the bias sees each eigenspace only through
    its projector ``V V^T``, which no choice of signs or of basis within the eigenspace changes --
    provided each eigenspace is kept whole, which ``tl.laplacian_spectrum`` ensures at the cut.

    Distances here run along the tissue: two cells on either side of a hole are far apart, unlike
    for the ``distance`` bias. The lowest ``k`` modes only carry smooth structure -- on a slide of
    side ``L`` the shortest wavelength is about ``L * sqrt(pi / k)`` -- which is the long-range scale
    the global module is for.

    Scaled for size: eigenvectors are unit-norm over their graph, so ``N * v_i[j] * v_i[l]`` is
    ``O(1)`` at any ``N`` (its mean over ``j = l`` is exactly 1) and each mode adds ``O(h_h)``. Summed,
    not averaged over the modes: a filter that keeps only the few slowest modes -- the long-range
    case -- would otherwise be shrunk by ``k`` and need a table ``k`` times larger to matter. The
    table starts at zero, so switching the bias on changes nothing at step 0. One table serves every
    layer, as for ``distance``.
    """

    def __init__(self, n_heads: int, num_knots: int, min_eigval: float, max_eigval: float):
        super().__init__()
        if num_knots < 2:
            raise ValueError(f"the spectral filter needs >= 2 knots (the two ends), got {num_knots}.")
        if not 0 < min_eigval < max_eigval:
            raise ValueError(f"the spectral filter needs 0 < min_eigval < max_eigval, got {min_eigval}, {max_eigval}.")
        self.num_knots = int(num_knots)
        # A buffer, so a checkpoint carries the range its table was trained on.
        self.register_buffer("log_eigval_range", torch.tensor([math.log(min_eigval), math.log(max_eigval)]))
        self.table = nn.Parameter(torch.zeros(int(n_heads), self.num_knots))

    def extra_repr(self) -> str:
        lo, hi = self.log_eigval_range.exp().tolist()
        return f"num_knots={self.num_knots}, eigenvalues={lo:.1e}-{hi:.1e}"

    def filter(self, eigvals: torch.Tensor) -> torch.Tensor:
        """``[..., k]`` eigenvalues to every head's weight ``h_h(lambda)``, ``[..., k, heads]``."""
        lo, hi = self.log_eigval_range
        last = self.num_knots - 1
        t = (torch.maximum(eigvals, lo.exp()).log() - lo) * (last / (hi - lo))
        return _knot_lookup(self.table, t.clamp(min=0.0, max=last))

    def forward(self, eigvecs: torch.Tensor, eigvals: torch.Tensor) -> torch.Tensor:
        """``[B, S, k]`` eigenvectors times ``sqrt(N)``, and ``[B, k]`` eigenvalues, to ``[B, heads, S, S]``.

        Columns that are zero (padding tokens, or eigenpairs a small graph does not have) add
        nothing, whatever weight their eigenvalue gets.
        """
        weights = self.filter(eigvals)  # [B, k, heads]
        weighted = torch.einsum("bsk,bkh->bhsk", eigvecs, weights)
        return weighted @ eigvecs.transpose(-1, -2).unsqueeze(1)

    def profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(knots [K] as eigenvalues, values [heads, K])`` -- each head's filter at each knot."""
        lo, hi = self.log_eigval_range.tolist()
        return torch.linspace(lo, hi, self.num_knots).exp(), self.table.detach().cpu().clone()


ROPE_KINDS = ("axial", "mixed")


class RotaryPE(nn.Module):
    """2D rotary position embedding: each head's queries and keys turned by the cell's coordinates.

    RoPE (Su et al. 2021) splits a head's ``head_dim`` into pairs and rotates pair ``i`` of the
    query and the key of a cell at ``p`` by the angle ``omega_i . p``. Their product then depends
    on ``omega_i . (p_n - p_m)`` only: attention sees the offset between two cells, not where they
    are -- and, unlike the distance bias, through a factor the query and key contents set, so a head
    can look for one kind of cell at one offset. In 2D every ``omega_i`` is a vector (Heo et al.,
    ECCV 2024). ``axial`` puts half of each head's pairs along x and half along y. ``mixed`` starts
    head ``h`` from that frame turned by ``h * 90 / n_heads`` degrees, which spreads the heads'
    axes evenly over the directions, and learns every pair's frequency vector.

    The ``n_heads * head_dim / 4`` wavelengths are spaced geometrically over
    ``[min_wavelength, max_wavelength]`` (µm) and dealt out across the heads, ALiBi-style: head ``h``
    takes every ``n_heads``-th, starting at the ``h``-th. InterScale's heads are narrow -- 4 dims at
    the default ``n_embed`` 16 and 4 heads, one wavelength per axis -- so each head gets its own
    scale instead of every head the same one; with wide heads each still spans the whole range.

    Between two cells far more than a wavelength apart, that pair turns by an effectively random
    angle and carries no usable signal; a wavelength far longer than their distance barely turns
    and leaves a content-only channel. So ``min_wavelength`` should be about the shortest distance
    the transformer attends over, and ``max_wavelength`` about twice the longest -- the default that
    :func:`resolve_rope_range` derives.

    ``mixed`` learns each frequency as ``magnitude * direction`` with only the unit-scale
    ``direction`` trained, so an optimiser step changes every frequency by about the same *relative*
    amount: in rad/µm they span three decades, and Adam's steps are absolute. Angles reach hundreds
    of radians across a slide; they are formed in float64, so the only rounding left in a pair's
    angle difference is that of the float32 coordinates themselves (~1e-3 µm at 5 mm).
    """

    def __init__(self, n_heads: int, head_dim: int, kind: str, min_wavelength: float, max_wavelength: float):
        super().__init__()
        if head_dim % 4:
            raise ValueError(
                "2D RoPE rotates pairs, one per axis and wavelength, so the head width n_embed / n_heads "
                f"must be a multiple of 4; got {head_dim}."
            )
        if kind not in ROPE_KINDS:
            raise ValueError(f"RoPE kind must be one of {ROPE_KINDS}, got {kind!r}.")
        if not 0 < min_wavelength <= max_wavelength:
            raise ValueError(
                f"RoPE needs 0 < min_wavelength <= max_wavelength, got {min_wavelength} and {max_wavelength} um."
            )
        self.n_heads, self.head_dim, self.kind = int(n_heads), int(head_dim), kind
        per_axis = self.head_dim // 4
        n_waves = self.n_heads * per_axis
        wavelengths = torch.logspace(
            math.log10(min_wavelength), math.log10(max_wavelength), n_waves, dtype=torch.float64
        )
        magnitude = (2 * math.pi / wavelengths).view(per_axis, self.n_heads).t()  # [heads, per_axis]

        frame = torch.arange(self.n_heads, dtype=torch.float64) * (math.pi / 2 / self.n_heads)
        if kind == "axial":
            frame = torch.zeros_like(frame)
        axis = frame[:, None].expand(-1, per_axis)
        angle = torch.cat([axis, axis + math.pi / 2], dim=1)  # [heads, pairs]: the frame's two axes
        direction = torch.stack([angle.cos(), angle.sin()], dim=-1)
        direction[direction.abs() < 1e-12] = 0.0  # cos(pi/2) is 6e-17: keep the axial axes exact

        # Buffers, so a reloaded model keeps the frequencies its projections were trained against
        # even where it derives another max_wavelength from other data.
        self.register_buffer("magnitude", torch.cat([magnitude, magnitude], dim=1).float())  # rad/µm
        if kind == "mixed":
            self.direction = nn.Parameter(direction.float())
        else:
            self.register_buffer("direction", direction.float())  # [heads, pairs, 2]

    def extra_repr(self) -> str:
        waves = self.wavelengths()
        return (
            f"kind={self.kind}, n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"wavelengths={waves.min():.0f}-{waves.max():.0f} um"
        )

    def frequencies(self) -> torch.Tensor:
        """``[heads, head_dim / 2, 2]``: each rotation pair's frequency vector, in rad/µm."""
        return self.magnitude.unsqueeze(-1) * self.direction

    def wavelengths(self) -> torch.Tensor:
        """``[heads, head_dim / 2]``: each pair's wavelength in µm (``mixed`` learns them)."""
        return (2 * math.pi / self.frequencies().detach().norm(dim=-1)).cpu()

    def forward(self, pos_um: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, S, 2]`` positions in µm to ``(cos, sin)``, each ``[B * heads, S, head_dim]``.

        Graph-major over heads -- row ``b * heads + h`` -- which is how the attention splits its
        batch into heads, and in the layout :func:`~.transformer_utils.apply_rotary` reads.
        """
        angles = torch.einsum("bsc,hpc->bhsp", pos_um.double(), self.frequencies().double())
        n_graphs, n_heads, seq_len, n_pairs = angles.shape
        shape = (n_graphs * n_heads, seq_len, 2 * n_pairs)
        cos = angles.cos().to(dtype).repeat(1, 1, 1, 2).reshape(shape)
        sin = angles.sin().to(dtype).repeat(1, 1, 1, 2).reshape(shape)
        return cos, sin


def _positive(value, name):
    if not value > 0:
        raise ValueError(f"model.global_component.parameters.pe.{name} must be > 0, got {value}.")


def _validate_naive(pe):
    _positive(pe.naive.hidden_dim, "naive.hidden_dim")
    _positive(pe.naive.length_scale, "naive.length_scale")


def _validate_sinusoidal(pe):
    s = pe.sinusoidal
    if s.dim <= 0 or s.dim % 4:
        raise ValueError(
            f"pe.sinusoidal.dim must be a positive multiple of 4 (sin and cos for each of two axes), got {s.dim}."
        )
    _positive(s.min_wavelength, "sinusoidal.min_wavelength")
    if not s.max_wavelength >= s.min_wavelength:
        raise ValueError(
            f"pe.sinusoidal.max_wavelength ({s.max_wavelength}) must be >= min_wavelength ({s.min_wavelength})."
        )


def _validate_lap(pe):
    _positive(pe.lap.k, "lap.k")


def _validate_rw(pe):
    _positive(pe.rw.steps, "rw.steps")


DISTANCE_KINDS = ("profile", "linear")


def _validate_distance(pe):
    if pe.distance.kind not in DISTANCE_KINDS:
        raise ValueError(f"pe.distance.kind must be one of {DISTANCE_KINDS}, got {pe.distance.kind!r}.")
    if pe.distance.kind == "linear":
        return  # num_kernels and max_dist only describe the profile
    if pe.distance.num_kernels < 2:
        raise ValueError(f"pe.distance.num_kernels must be >= 2 (the two ends), got {pe.distance.num_kernels}.")
    if not pe.distance.max_dist >= 0:
        raise ValueError(f"pe.distance.max_dist must be >= 0 (0 derives it from the data), got {pe.distance.max_dist}.")


def _validate_rope(pe):
    r = pe.rope
    if r.kind not in ROPE_KINDS:
        raise ValueError(f"pe.rope.kind must be one of {ROPE_KINDS}, got {r.kind!r}.")
    _positive(r.min_wavelength, "rope.min_wavelength")
    if not r.max_wavelength >= 0:
        raise ValueError(f"pe.rope.max_wavelength must be >= 0 (0 derives it from the data), got {r.max_wavelength}.")
    if r.max_wavelength > 0 and r.max_wavelength < r.min_wavelength:
        raise ValueError(
            f"pe.rope.max_wavelength ({r.max_wavelength}) must be >= min_wavelength ({r.min_wavelength}), or 0."
        )


def _validate_spectral(pe):
    s = pe.spectral
    _positive(s.k, "spectral.k")
    if s.num_knots < 2:
        raise ValueError(f"pe.spectral.num_knots must be >= 2 (the two ends), got {s.num_knots}.")
    _positive(s.min_eigval, "spectral.min_eigval")
    if not s.max_eigval > s.min_eigval:
        raise ValueError(f"pe.spectral.max_eigval ({s.max_eigval}) must be > min_eigval ({s.min_eigval}).")


def _spectral_inputs(data, pe) -> dict:
    """The spectral bias's precomputed inputs for one graph: two attributes, per cell and per graph."""
    vecs, vals = laplacian_spectrum(data.edge_index, data.num_nodes, int(pe.spectral.k))
    return {
        # Times sqrt(N), so that products of two entries are O(1) whatever the graph's size.
        "spectral_pe": vecs * math.sqrt(max(int(data.num_nodes), 1)),
        # [1, k]: PyG stacks one row per graph into [B, k].
        "spectral_eigval": vals.unsqueeze(0),
    }


@register_pe("naive", kind="node", requires="pos", validate=_validate_naive)
def _build_naive(pe, n_embed):
    return NaivePE(n_embed, int(pe.naive.hidden_dim), float(pe.naive.length_scale))


@register_pe("sinusoidal", kind="node", requires="pos", validate=_validate_sinusoidal)
def _build_sinusoidal(pe, n_embed):
    s = pe.sinusoidal
    return SinusoidalPE(n_embed, int(s.dim), float(s.min_wavelength), float(s.max_wavelength))


@register_pe(
    "lap",
    kind="node",
    requires="lap_pe",
    validate=_validate_lap,
    precompute=lambda data, pe: laplacian_pe(data.edge_index, data.num_nodes, int(pe.lap.k)),
)
def _build_lap(pe, n_embed):
    return LapPE(n_embed, int(pe.lap.k), bool(pe.lap.sign_flip))


@register_pe(
    "rw",
    kind="node",
    requires="rw_pe",
    validate=_validate_rw,
    precompute=lambda data, pe: random_walk_pe(data.edge_index, data.num_nodes, int(pe.rw.steps)),
)
def _build_rw(pe, n_embed):
    return RWPE(n_embed, int(pe.rw.steps))


@register_pe("distance", kind="bias", requires="pos", validate=_validate_distance)
def _build_distance(pe, n_heads):
    if pe.distance.kind == "linear":
        return LinearDistanceBias(n_heads)
    if not pe.distance.max_dist > 0:
        raise ValueError(
            "pe.distance.max_dist is 0, which means 'derive it from the data', but nothing derived it. "
            "A model built from AnnData does so in its constructor (resolve_distance_range); a module "
            "built directly from the config needs max_dist set, or resolve_distance_range(cfg, adata) first."
        )
    return DistanceBias(n_heads, int(pe.distance.num_kernels), float(pe.distance.max_dist))


@register_pe("spectral", kind="bias", requires="spectral_pe", validate=_validate_spectral, precompute=_spectral_inputs)
def _build_spectral(pe, n_heads):
    s = pe.spectral
    return SpectralBias(n_heads, int(s.num_knots), float(s.min_eigval), float(s.max_eigval))


@register_pe("rope", kind="rotary", requires="pos", validate=_validate_rope)
def _build_rope(pe, n_heads, head_dim):
    r = pe.rope
    if not r.max_wavelength > 0:
        raise ValueError(
            "pe.rope.max_wavelength is 0, which means 'derive it from the data', but nothing derived it. "
            "A model built from AnnData does so in its constructor (resolve_rope_range); a module built "
            "directly from the config needs max_wavelength set, or resolve_rope_range(cfg, adata) first."
        )
    return RotaryPE(n_heads, head_dim, r.kind, float(r.min_wavelength), float(r.max_wavelength))


class CoordinateFrame:
    """Cell coordinates as the coordinate encodings read them, and the generator they augment with.

    Converts to µm (``spatial_unit_um``), subtracts each graph's centroid (``center_coords``;
    absolute slide offsets are scanner artefacts), and in training optionally rotates each graph by
    a random angle (``rotate_train``; tissue has no canonical orientation). One per model, shared
    by the node encodings and the rotary encoding, so that within a forward pass both see a graph
    turned by the same angle. Its generator, seeded from ``optim.seed``, also draws LapPE's sign
    flips. Not a module: it holds nothing to train or save, so callers pass ``training``.
    """

    def __init__(self, *, spatial_unit_um: float, center_coords: bool, rotate_train: bool, seed: int):
        self.spatial_unit_um = float(spatial_unit_um)
        self.center_coords = bool(center_coords)
        self.rotate_train = bool(rotate_train)
        self.generator = torch.Generator().manual_seed(int(seed))

    def coordinates(self, batched_data, dtype: torch.dtype, device: torch.device, *, training: bool, users):
        """``[N, 2]`` coordinates in µm, centred and (in training, if enabled) rotated per graph.

        ``users`` names the encodings asking, for the error raised when there are no coordinates.
        """
        pos = getattr(batched_data, "pos", None)
        if pos is None:
            raise ValueError(
                f"positional encodings {list(users)} need cell coordinates, but the graphs carry no "
                "`pos`. Set dataset.spatial_key (e.g. 'spatial') so they are attached."
            )
        pos = pos.to(device=device, dtype=dtype) * self.spatial_unit_um
        batch = batched_data.batch.to(device=device, dtype=torch.long)

        if self.center_coords:
            pos = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
        if self.rotate_train and training:
            n_graphs = int(batch.max().item()) + 1
            theta = (torch.rand(n_graphs, generator=self.generator) * (2 * math.pi)).to(device=device, dtype=dtype)
            cos, sin = theta.cos()[batch], theta.sin()[batch]
            pos = torch.stack([cos * pos[:, 0] - sin * pos[:, 1], sin * pos[:, 0] + cos * pos[:, 1]], dim=1)
        return pos


class NodePositionalEncoding(nn.Module):
    """The enabled node encodings, summed into one ``[N, n_embed]`` term per cell.

    Prepares each encoder's input once; coordinates come from the model's :class:`CoordinateFrame`.
    """

    def __init__(self, encoders: dict[str, nn.Module], *, frame: CoordinateFrame):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.requires = {name: PE_REGISTRY[name].requires for name in encoders}
        self.frame = frame

    @property
    def generator(self) -> torch.Generator:
        """The PE generator, the frame's: rotations and LapPE's sign flips draw from it."""
        return self.frame.generator

    def coordinates(self, batched_data, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``[N, 2]`` coordinates in µm, centred and (in training, if enabled) rotated per graph."""
        return self.frame.coordinates(batched_data, dtype, device, training=self.training, users=sorted(self.encoders))

    def forward(self, batched_data, dtype: torch.dtype, device: torch.device, inputs: dict | None = None):
        """The summed encoding, ``[N, n_embed]``.

        ``inputs``, when given, is filled with each prepared input under its ``Data`` key -- the
        transformer passes the coordinates found there on to the rotary encoding, so that both use
        the same rotation of each graph.
        """
        inputs = {} if inputs is None else inputs
        total = 0
        for name, encoder in self.encoders.items():
            key = self.requires[name]
            if key not in inputs:
                if key == "pos":
                    inputs[key] = self.coordinates(batched_data, dtype, device)
                else:
                    value = getattr(batched_data, key, None)
                    if value is None:
                        raise ValueError(
                            f"positional encoding '{name}' reads `{key}` off the graphs, but they carry "
                            "none. It is precomputed by attach_positional_inputs when the graphs are "
                            "built (prepare_geome_dataset, get_model_output)."
                        )
                    inputs[key] = value.to(device=device, dtype=dtype)
            x = inputs[key]
            if self.training and hasattr(encoder, "augment"):
                x = encoder.augment(x, batched_data.batch.to(device=device, dtype=torch.long), self.generator)
            total = total + encoder(x)
        return total


class AttentionBias(nn.Module):
    """The enabled attention biases, laid out as a float attention mask ``[B * heads, S+1, S+1]``.

    Per-cell inputs -- coordinates, eigenvectors -- are placed with
    :func:`~interscale.tl.padding.pad_like` from the ``index_nodes`` that ``pad_batch`` returned --
    never padded a second time, since ``pad_batch`` subsamples with its own randomness -- so bias
    entry ``(i, j)`` belongs to tokens ``i`` and ``j`` of the sequence. The layout is the boolean
    mask's: left-padded, CLS last, graph-major over heads. The CLS row and column and every padding
    row and column carry zero, so merging the boolean mask in as ``-inf`` can never empty a row:
    CLS stays attendable for every query.
    """

    def __init__(self, encoders: dict[str, nn.Module], *, n_heads: int, spatial_unit_um: float):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.requires = {name: PE_REGISTRY[name].requires for name in encoders}
        self.n_heads = int(n_heads)
        self.spatial_unit_um = float(spatial_unit_um)

    def _inputs(self, key: str, batched_data, batch, index_nodes, seq_len: int, dtype, device) -> tuple:
        """What the biases reading ``key`` take, padded like the tokens."""
        if key == "pos":
            pos = getattr(batched_data, "pos", None)
            if pos is None:
                users = sorted(name for name, need in self.requires.items() if need == "pos")
                raise ValueError(
                    f"attention biases {users} need cell coordinates, but the graphs carry no "
                    "`pos`. Set dataset.spatial_key (e.g. 'spatial') so they are attached."
                )
            padded = pad_like(pos.to(device=device, dtype=dtype) * self.spatial_unit_um, batch, index_nodes, seq_len)
            # The exact path: the matrix-product shortcut loses ~µm on coordinates of a few mm.
            return (torch.cdist(padded, padded, compute_mode="donot_use_mm_for_euclid_dist"),)
        if key == "spectral_pe":
            vecs, vals = getattr(batched_data, key, None), getattr(batched_data, "spectral_eigval", None)
            if vecs is None or vals is None:
                raise ValueError(
                    "the spectral bias reads `spectral_pe` and `spectral_eigval` off the graphs, but they "
                    "carry none. They are precomputed by attach_positional_inputs when the graphs are "
                    "built (prepare_geome_dataset, get_model_output)."
                )
            vecs = pad_like(vecs.to(device=device, dtype=dtype), batch, index_nodes, seq_len)  # [B, S, k]
            return vecs, vals.to(device=device, dtype=dtype).reshape(len(index_nodes), -1)  # [B, k]
        raise ValueError(f"no attention bias input is prepared from `{key}`.")

    def forward(self, batched_data, index_nodes, src_padding_mask, dtype, device) -> torch.Tensor:
        batch = batched_data.batch.to(device=device, dtype=torch.long)
        n_graphs, seq_len = src_padding_mask.shape

        inputs, bias = {}, 0
        for name, encoder in self.encoders.items():
            key = self.requires[name]
            if key not in inputs:
                inputs[key] = self._inputs(key, batched_data, batch, index_nodes, seq_len, dtype, device)
            bias = bias + encoder(*inputs[key])  # [B, heads, S, S]

        pad = src_padding_mask.to(device=device, dtype=torch.bool)
        bias = bias.masked_fill(pad[:, None, :, None] | pad[:, None, None, :], 0.0)
        full = bias.new_zeros(n_graphs, self.n_heads, seq_len + 1, seq_len + 1)
        full[:, :, :seq_len, :seq_len] = bias
        return full.reshape(n_graphs * self.n_heads, seq_len + 1, seq_len + 1)


class RotaryEncoding(nn.Module):
    """The enabled rotary encoding, as the rotation every attention layer applies to queries and keys.

    Positions come from the :class:`CoordinateFrame` -- in a pass where node encodings already read
    coordinates, the very same ones -- and are placed with :func:`~interscale.tl.padding.pad_like`
    from the ``index_nodes`` that ``pad_batch`` returned, never padded a second time. The layout is
    the tokens': left-padded, CLS last, graph-major over heads. CLS and the padding sit at the
    origin, where the rotation is the identity. That makes the origin matter for CLS alone -- with
    ``center_coords`` it is the graph's centroid -- while every cell-to-cell logit depends on the
    offset between the two cells only.
    """

    def __init__(self, encoders: dict[str, nn.Module], *, frame: CoordinateFrame):
        super().__init__()
        if len(encoders) != 1:
            raise ValueError(f"exactly one rotary encoding can turn the queries and keys, got {sorted(encoders)}.")
        self.encoders = nn.ModuleDict(encoders)
        self.frame = frame

    def forward(self, batched_data, index_nodes, seq_len: int, dtype, device, pos=None):
        """``(cos, sin)``, each ``[B * heads, seq_len + 1, head_dim]``, for ``pad_batch``'s ``index_nodes``.

        ``pos`` takes the ``[N, 2]`` coordinates the node encodings prepared in this pass, if any;
        otherwise the frame prepares them here.
        """
        if pos is None:
            pos = self.frame.coordinates(
                batched_data, dtype, device, training=self.training, users=sorted(self.encoders)
            )
        batch = batched_data.batch.to(device=device, dtype=torch.long)
        padded = pad_like(pos, batch, index_nodes, seq_len)  # [B, S, 2], zero at the padding
        padded = torch.cat([padded, padded.new_zeros(padded.shape[0], 1, 2)], dim=1)  # CLS last, at the origin
        (encoder,) = self.encoders.values()
        return encoder(padded, dtype)


def _enabled(cfg, kind: str = "node") -> tuple[list[str], object]:
    """The encodings ``pe.node`` (``kind="node"``) or ``pe.bias`` (``"bias"``) lists, and ``pe``."""
    params = cfg.model.global_component.get("parameters", None)
    pe = params.get("pe", None) if params is not None else None
    return (list(pe.get(kind, [])) if pe is not None else []), pe


def attach_positional_inputs(datas, cfg) -> None:
    """Store what the enabled encodings precompute (e.g. ``lap_pe``) on each graph, in place.

    Called wherever graphs are built -- ``prepare_geome_dataset`` and the models'
    ``get_model_output`` -- so training and evaluation see the same inputs. A no-op when no enabled
    encoding precomputes anything.
    """
    names, pe = _enabled(cfg, "node")
    names += _enabled(cfg, "bias")[0] + _enabled(cfg, "rotary")[0]
    for name in names:
        spec = PE_REGISTRY[name]
        if spec.precompute is None:
            continue
        for data in datas:
            value = spec.precompute(data, pe)
            # One attribute, or several by name for an encoding that needs more than one (the
            # spectral bias: eigenvectors per cell, eigenvalues per graph).
            for key, item in value.items() if isinstance(value, dict) else [(spec.requires, value)]:
                setattr(data, key, item)


def report_spectral_range(datas, cfg) -> tuple[float, float] | None:
    """Log the eigenvalues the spectral bias got against its filter's range, and warn where they fall outside.

    The filter is flat beyond ``[min_eigval, max_eigval]``, so eigenvalues out there all get one
    weight -- harmless at a small margin, but a range that misses the data leaves the bias almost
    unable to tell its modes apart. Also says how many graphs keep fewer than ``k`` eigenpairs (a
    small graph, or a tied eigenspace dropped at the cut). Returns the ``(min, max)`` eigenvalue
    found, or None when the bias is off or there is none.
    """
    names, pe = _enabled(cfg, "bias")
    if "spectral" not in names:
        return None
    rows = [d.spectral_eigval.reshape(-1) for d in datas if getattr(d, "spectral_eigval", None) is not None]
    if not rows:
        return None
    vals = torch.cat(rows)
    vals = vals[vals > 0]
    if not len(vals):
        return None
    short = sum(int((row > 0).sum()) < int(pe.spectral.k) for row in rows)
    # A cell whose row is zero in every kept mode sits in a piece of the graph that none of the k
    # lowest modes reaches; its bias to every other cell is 0. Entries are O(1) after the sqrt(N).
    vecs = [d.spectral_pe for d in datas if getattr(d, "spectral_pe", None) is not None]
    empty = sum(int((v.abs().amax(dim=1) < 1e-4).sum()) for v in vecs) / max(sum(len(v) for v in vecs), 1)
    lo, hi = float(vals.min()), float(vals.max())
    logger.info(
        "Spectral bias: eigenvalues %.2e-%.2e over %d graphs; the filter's knots span %.1e-%.1e. "
        "%d graph(s) keep fewer than k = %d eigenpairs; %.1f%% of cells are in no kept mode.",
        lo,
        hi,
        len(rows),
        pe.spectral.min_eigval,
        pe.spectral.max_eigval,
        short,
        int(pe.spectral.k),
        100 * empty,
    )
    if empty > 0.1:
        logger.warning(
            "Spectral bias: %.0f%% of cells are in no kept mode -- they sit in pieces of the graph that "
            "none of the k = %d lowest modes reaches, so their bias to every cell is 0. The graph is "
            "fragmented (radius below the cells' spacing?); a larger k reaches more pieces.",
            100 * empty,
            int(pe.spectral.k),
        )
    if lo < pe.spectral.min_eigval or hi > pe.spectral.max_eigval:
        logger.warning(
            "Spectral bias: eigenvalues %.2e-%.2e reach beyond the filter's knots (%.1e-%.1e), where it is "
            "flat. Widen pe.spectral.min_eigval / max_eigval to cover them.",
            lo,
            hi,
            pe.spectral.min_eigval,
            pe.spectral.max_eigval,
        )
    return lo, hi


def resolve_distance_range(cfg, adata):
    """``cfg`` with a derived ``pe.distance.max_dist`` when it is 0, else ``cfg`` itself.

    0 means "the largest slide diameter in the data": the farthest pair of cells within any one
    ``dataset.sample_key`` group of ``adata``, over every split, in µm. That is the longest
    distance the bias is ever asked about, so the profile's knots span exactly the distances
    that occur. Returns a clone -- the caller's config is not modified. Called by every model
    constructor; a no-op unless the ``profile`` distance bias is enabled with ``max_dist`` 0.
    """
    names, pe = _enabled(cfg, "bias")
    if "distance" not in names or pe.distance.kind != "profile" or pe.distance.max_dist > 0:
        return cfg
    max_dist = max_slide_diameter_um(adata, cfg)
    if not max_dist > 0:
        raise ValueError(
            f"pe.distance.max_dist 0 asks for the largest slide diameter, but every slide in "
            f"obsm[{cfg.dataset.spatial_key!r}] has its cells at one point. Set max_dist explicitly."
        )
    cfg = cfg.clone()
    cfg.defrost()
    cfg.model.global_component.parameters.pe.distance.max_dist = max_dist
    cfg.freeze()
    logger.info("pe.distance.max_dist derived from the data: %.1f um (largest slide diameter).", max_dist)
    return cfg


def resolve_rope_range(cfg, adata):
    """``cfg`` with a derived ``pe.rope.max_wavelength`` when it is 0, else ``cfg`` itself.

    0 means twice the largest slide diameter (:func:`max_slide_diameter_um`). The longest wave then
    turns by at most half a cycle between two cells of one slide, so its phase alone tells apart
    every offset along its direction -- the coarsest scale reaches across the whole slide without
    wrapping around. Returns a clone; the caller's config is not modified. Called by every model
    constructor; a no-op unless RoPE is enabled with ``max_wavelength`` 0.
    """
    names, pe = _enabled(cfg, "rotary")
    if "rope" not in names or pe.rope.max_wavelength > 0:
        return cfg
    max_wavelength = 2.0 * max_slide_diameter_um(adata, cfg)
    if not max_wavelength >= pe.rope.min_wavelength:
        raise ValueError(
            f"pe.rope.max_wavelength 0 asks for twice the largest slide diameter, {max_wavelength:.1f} um, "
            f"which is below pe.rope.min_wavelength ({pe.rope.min_wavelength}). Set both explicitly."
        )
    cfg = cfg.clone()
    cfg.defrost()
    cfg.model.global_component.parameters.pe.rope.max_wavelength = max_wavelength
    cfg.freeze()
    logger.info(
        "pe.rope.max_wavelength derived from the data: %.1f um (twice the largest slide diameter).", max_wavelength
    )
    return cfg


def max_slide_diameter_um(adata, cfg) -> float:
    """The farthest pair of cells within one ``dataset.sample_key`` group, in µm, over all groups."""
    key = cfg.dataset.spatial_key
    if key is None or key not in adata.obsm:
        raise ValueError(f"deriving a length from the data needs coordinates, but obsm[{key!r}] does not exist.")
    coords = np.asarray(adata.obsm[key], dtype=np.float64)
    groups = adata.obs.groupby(list(cfg.dataset.sample_key), observed=True).indices
    diameter = max(point_set_diameter(coords[rows]) for rows in groups.values())
    return diameter * float(cfg.dataset.spatial_unit_um)


#: LapPE entries scale like 1/sqrt(N); above this ratio of largest to smallest graph the same
#: position is encoded at least 2x larger in the small graphs than in the large ones.
LAP_SIZE_RATIO_WARNING = 4.0


def warn_if_lap_scales_differ(datas, cfg) -> float | None:
    """Warn when LapPE is on and the graphs' sizes differ by more than :data:`LAP_SIZE_RATIO_WARNING`.

    Each eigenvector is unit-norm over its graph, so its entries scale like ``1/sqrt(N)``: across
    graphs of very different size (whole slides vs windows, or a cell-level split that leaves val
    graphs a fraction of the train ones) one encoder sees the same structure at different
    magnitudes. Returns the size ratio, or None when LapPE is off or there is nothing to compare.
    """
    if "lap" not in _enabled(cfg, "node")[0]:
        return None
    sizes = [int(d.num_nodes) for d in datas if d.num_nodes]
    if len(sizes) < 2:
        return None
    ratio = max(sizes) / min(sizes)
    if ratio > LAP_SIZE_RATIO_WARNING:
        logger.warning(
            "LapPE: graph sizes range %d-%d cells (x%.1f), so eigenvector entries differ ~x%.1f in "
            "scale between the largest and smallest graph (they go like 1/sqrt(N)). Scaling them by "
            "sqrt(N) is the candidate fix; see .claude/PE_plan.md.",
            min(sizes),
            max(sizes),
            ratio,
            math.sqrt(ratio),
        )
    return ratio


def build_coordinate_frame(cfg) -> CoordinateFrame:
    """The coordinate frame, and PE generator, that the node and rotary encodings of a model share."""
    pe = _enabled(cfg)[1]
    return CoordinateFrame(
        spatial_unit_um=cfg.dataset.spatial_unit_um,
        center_coords=pe.center_coords,
        rotate_train=pe.rotate_train,
        seed=cfg.optim.seed,
    )


def build_positional_encodings(cfg, *, n_embed: int | None, n_heads: int) -> dict:
    """Every positional module ``cfg`` enables, as the transformer's keyword arguments (None when off).

    The node and the rotary encoding get one :class:`CoordinateFrame`, so in a forward pass they see
    each graph rotated by the same angle, and draw from a single PE generator.
    """
    needs_frame = _enabled(cfg, "node")[0] or _enabled(cfg, "rotary")[0]
    frame = build_coordinate_frame(cfg) if needs_frame else None
    return {
        "positional_encoding": build_node_positional_encoding(cfg, n_embed, frame=frame),
        "attention_bias": build_attention_bias(cfg, n_heads),
        "rotary_encoding": build_rotary_encoding(cfg, n_heads, n_embed, frame=frame),
    }


def build_node_positional_encoding(
    cfg, n_embed: int | None, frame: CoordinateFrame | None = None
) -> NodePositionalEncoding | None:
    """Build the node encodings ``cfg`` enables, or None when ``pe.node`` is empty.

    The encoders are built inside a forked RNG, so enabling one does not shift the initialisation
    of any weight built after it. ``frame`` is the model's shared one; without it, a new one.
    """
    names, pe = _enabled(cfg, "node")
    if not names:
        return None
    if n_embed is None:
        raise ValueError("building positional encodings needs n_embed, the token width they are added to.")

    with torch.random.fork_rng(devices=[]):
        encoders = {name: PE_REGISTRY[name].build(pe, n_embed) for name in names}
    return NodePositionalEncoding(encoders, frame=frame if frame is not None else build_coordinate_frame(cfg))


def build_rotary_encoding(
    cfg, n_heads: int, n_embed: int | None, frame: CoordinateFrame | None = None
) -> RotaryEncoding | None:
    """Build the rotary encoding ``cfg`` enables, or None when ``pe.rotary`` is empty.

    Built inside a forked RNG, like the others. ``frame`` is the model's shared one; without it, a
    new one -- whose rotations then differ from the node encodings' in training.
    """
    names, pe = _enabled(cfg, "rotary")
    if not names:
        return None
    if n_embed is None or n_embed % n_heads:
        raise ValueError(f"the rotary encoding needs n_embed divisible by n_heads ({n_heads}), got {n_embed}.")

    with torch.random.fork_rng(devices=[]):
        encoders = {name: PE_REGISTRY[name].build(pe, int(n_heads), int(n_embed) // int(n_heads)) for name in names}
    return RotaryEncoding(encoders, frame=frame if frame is not None else build_coordinate_frame(cfg))


def build_attention_bias(cfg, n_heads: int) -> AttentionBias | None:
    """Build the attention biases ``cfg`` enables, or None when ``pe.bias`` is empty.

    Built inside a forked RNG, like the node encodings, so enabling one shifts no other weight.
    """
    names, pe = _enabled(cfg, "bias")
    if not names:
        return None
    with torch.random.fork_rng(devices=[]):
        encoders = {name: PE_REGISTRY[name].build(pe, int(n_heads)) for name in names}
    return AttentionBias(encoders, n_heads=n_heads, spatial_unit_um=cfg.dataset.spatial_unit_um)
