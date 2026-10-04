"""Positional encodings (PEs) for the global (transformer) component.

Selected by ``model.global_component.parameters.pe``; the staged design is in
``.claude/PE_plan.md``. The plan has two kinds: *node* encodings, summed onto each cell's token
before ``pad_batch`` so that subsampling, node masking and padding carry them for free, and
attention *biases* (not built yet). This module holds the node kind and the registry both will use.

Three properties every node encoding keeps:

* **Off is today's model.** With ``pe.node: []`` nothing is built -- no parameters, no state-dict
  keys, no RNG draws -- so a PE-free run is the run from before PEs existed.
* **On changes nothing at step 0.** Every encoder's output projection starts at zero, and the
  encoders are built inside a forked RNG. A PE run and a PE-free run at the same seed therefore
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

from interscale.tl.positional import laplacian_pe, random_walk_pe


@dataclass(frozen=True)
class PESpec:
    """What the rest of the code needs to know about one encoding.

    ``requires`` names the ``Data`` attribute the encoder reads (``"pos"`` for coordinates).
    ``build(pe_cfg, n_embed)`` returns the encoder; ``validate(pe_cfg)`` raises on settings that
    would fail mid-run, and runs at config load. ``precompute(data, pe_cfg)``, when set, returns
    the ``requires`` attribute for one graph; :func:`attach_positional_inputs` stores it.
    """

    kind: str
    requires: str
    build: Callable
    validate: Callable
    precompute: Callable | None = None


PE_REGISTRY: dict[str, PESpec] = {}


def register_pe(name: str, *, kind: str, requires: str, validate: Callable, precompute: Callable | None = None):
    """Register ``build(pe_cfg, n_embed) -> nn.Module`` as the encoding ``name``."""

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


def _enabled(cfg) -> tuple[list[str], object]:
    params = cfg.model.global_component.get("parameters", None)
    pe = params.get("pe", None) if params is not None else None
    return (list(pe.node) if pe is not None else []), pe


def attach_positional_inputs(datas, cfg) -> None:
    """Store what the enabled encodings precompute (e.g. ``lap_pe``) on each graph, in place.

    Called wherever graphs are built -- ``prepare_geome_dataset`` and the models'
    ``get_model_output`` -- so training and evaluation see the same inputs. A no-op when no enabled
    encoding precomputes anything.
    """
    names, pe = _enabled(cfg)
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
    names, pe = _enabled(cfg)
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
