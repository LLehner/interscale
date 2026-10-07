"""Positional encodings (PEs) for the global (transformer) component.

Selected by ``model.global_component.parameters.pe``; the staged design is in
``.claude/PE_plan.md``. Two kinds: *node* encodings (``pe.node``), summed onto each cell's token
before ``pad_batch`` so that subsampling, node masking and padding carry them for free, and
attention *biases* (``pe.bias``), added to the attention logits of every pair of tokens and merged
into the attention mask where it is built.

Three properties every encoding keeps:

* **Off is today's model.** With ``pe.node: []`` nothing is built -- no parameters, no state-dict
  keys, no RNG draws -- so a PE-free run is the run from before PEs existed.
* **On changes nothing at step 0.** Every encoder's output (a projection, or a bias table) starts
  at zero, and the encoders are built inside a forked RNG. A PE run and a PE-free run at the same seed therefore
  share every other initial weight and the first forward pass; whatever differs later is what the
  PE was used for.
* **Augmentation has its own generator.** ``rotate_train`` and LapPE's sign flips draw from a
  generator seeded from ``optim.seed``, never from the global torch RNG, which also decides the
  training batch order.

Inputs that are expensive to compute -- Laplacian eigenvectors -- are precomputed once per graph by
:func:`attach_positional_inputs` when the graphs are built, and ride on the ``Data`` object.

Adding an encoding is one :func:`register_pe` entry plus one config sub-block in
``get_global_component_cfg``; nothing in the transformer changes.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

from interscale.tl.padding import pad_like
from interscale.tl.positional import laplacian_pe, random_walk_pe


@dataclass(frozen=True)
class PESpec:
    """What the rest of the code needs to know about one encoding.

    ``kind`` is ``"node"`` (listed under ``pe.node``) or ``"bias"`` (under ``pe.bias``).
    ``requires`` names the ``Data`` attribute the encoder reads (``"pos"`` for coordinates).
    ``build(pe_cfg, width)`` returns the encoder -- ``width`` is ``n_embed`` for a node encoding
    (its output width) and ``n_heads`` for a bias (one value per head). ``validate(pe_cfg)`` raises
    on settings that would fail mid-run, and runs at config load. ``precompute(data, pe_cfg)``, when set, returns
    the ``requires`` attribute for one graph; :func:`attach_positional_inputs` stores it.
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
        self.max_dist = float(max_dist)
        self.table = nn.Parameter(torch.zeros(int(n_heads), self.num_kernels))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """``[B, S, S]`` distances in µm to ``[B, heads, S, S]`` biases."""
        last = self.num_kernels - 1
        t = (dist * (last / self.max_dist)).clamp(max=last)
        lower = t.floor().clamp(max=last - 1)
        frac = (t - lower).unsqueeze(-1)
        values = self.table.t()  # [K, heads]
        lo = nn.functional.embedding(lower.long(), values)
        hi = nn.functional.embedding(lower.long() + 1, values)
        return (lo + frac * (hi - lo)).permute(0, 3, 1, 2)

    def profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(knots [K] in µm, values [heads, K])`` -- each head's bias at each knot."""
        return torch.linspace(0.0, self.max_dist, self.num_kernels), self.table.detach().cpu().clone()


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
    _positive(pe.distance.max_dist, "distance.max_dist")


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
    return DistanceBias(n_heads, int(pe.distance.num_kernels), float(pe.distance.max_dist))


class NodePositionalEncoding(nn.Module):
    """The enabled node encodings, summed into one ``[N, n_embed]`` term per cell.

    Prepares each encoder's input once. For coordinates that means: convert to µm
    (``spatial_unit_um``), subtract each graph's centroid (``center_coords``; absolute slide
    offsets are scanner artefacts), and in training optionally rotate each graph by a random angle
    (``rotate_train``; tissue has no canonical orientation).
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        *,
        spatial_unit_um: float,
        center_coords: bool,
        rotate_train: bool,
        seed: int,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.requires = {name: PE_REGISTRY[name].requires for name in encoders}
        self.spatial_unit_um = float(spatial_unit_um)
        self.center_coords = bool(center_coords)
        self.rotate_train = bool(rotate_train)
        self.generator = torch.Generator().manual_seed(int(seed))

    def coordinates(self, batched_data, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``[N, 2]`` coordinates in µm, centred and (in training, if enabled) rotated per graph."""
        pos = getattr(batched_data, "pos", None)
        if pos is None:
            raise ValueError(
                f"positional encodings {sorted(self.encoders)} need cell coordinates, but the graphs "
                "carry no `pos`. Set dataset.spatial_key (e.g. 'spatial') so they are attached."
            )
        pos = pos.to(device=device, dtype=dtype) * self.spatial_unit_um
        batch = batched_data.batch.to(device=device, dtype=torch.long)

        if self.center_coords:
            pos = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
        if self.rotate_train and self.training:
            n_graphs = int(batch.max().item()) + 1
            theta = (torch.rand(n_graphs, generator=self.generator) * (2 * math.pi)).to(device=device, dtype=dtype)
            cos, sin = theta.cos()[batch], theta.sin()[batch]
            pos = torch.stack([cos * pos[:, 0] - sin * pos[:, 1], sin * pos[:, 0] + cos * pos[:, 1]], dim=1)
        return pos

    def forward(self, batched_data, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        inputs = {}
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

    Positions are placed with :func:`~interscale.tl.padding.pad_like` from the ``index_nodes`` that
    ``pad_batch`` returned -- never padded a second time, since ``pad_batch`` subsamples with its
    own randomness -- so bias entry ``(i, j)`` belongs to tokens ``i`` and ``j`` of the sequence.
    The layout is the boolean mask's: left-padded, CLS last, graph-major over heads. The CLS row and
    column and every padding row and column carry zero, so merging the boolean mask in as ``-inf``
    can never empty a row: CLS stays attendable for every query.
    """

    def __init__(self, encoders: dict[str, nn.Module], *, n_heads: int, spatial_unit_um: float):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.n_heads = int(n_heads)
        self.spatial_unit_um = float(spatial_unit_um)

    def forward(self, batched_data, index_nodes, src_padding_mask, dtype, device) -> torch.Tensor:
        pos = getattr(batched_data, "pos", None)
        if pos is None:
            raise ValueError(
                f"attention biases {sorted(self.encoders)} need cell coordinates, but the graphs carry no "
                "`pos`. Set dataset.spatial_key (e.g. 'spatial') so they are attached."
            )
        pos = pos.to(device=device, dtype=dtype) * self.spatial_unit_um
        batch = batched_data.batch.to(device=device, dtype=torch.long)
        n_graphs, seq_len = src_padding_mask.shape

        padded = pad_like(pos, batch, index_nodes, seq_len)  # [B, S, 2]
        # The exact path: the matrix-product shortcut loses ~µm on coordinates of a few mm.
        dist = torch.cdist(padded, padded, compute_mode="donot_use_mm_for_euclid_dist")
        bias = sum(encoder(dist) for encoder in self.encoders.values())  # [B, heads, S, S]

        pad = src_padding_mask.to(device=device, dtype=torch.bool)
        bias = bias.masked_fill(pad[:, None, :, None] | pad[:, None, None, :], 0.0)
        full = bias.new_zeros(n_graphs, self.n_heads, seq_len + 1, seq_len + 1)
        full[:, :, :seq_len, :seq_len] = bias
        return full.reshape(n_graphs * self.n_heads, seq_len + 1, seq_len + 1)


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
    names += _enabled(cfg, "bias")[0]
    for name in names:
        spec = PE_REGISTRY[name]
        if spec.precompute is None:
            continue
        for data in datas:
            setattr(data, spec.requires, spec.precompute(data, pe))


def build_node_positional_encoding(cfg, n_embed: int | None) -> NodePositionalEncoding | None:
    """Build the node encodings ``cfg`` enables, or None when ``pe.node`` is empty.

    The encoders are built inside a forked RNG, so enabling one does not shift the initialisation
    of any weight built after it.
    """
    names, pe = _enabled(cfg, "node")
    if not names:
        return None
    if n_embed is None:
        raise ValueError("building positional encodings needs n_embed, the token width they are added to.")

    with torch.random.fork_rng(devices=[]):
        encoders = {name: PE_REGISTRY[name].build(pe, n_embed) for name in names}
    return NodePositionalEncoding(
        encoders,
        spatial_unit_um=cfg.dataset.spatial_unit_um,
        center_coords=pe.center_coords,
        rotate_train=pe.rotate_train,
        seed=cfg.optim.seed,
    )


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
