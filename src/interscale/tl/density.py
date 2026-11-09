"""Relative cell density per section, as an ordered categorical that reaches the graphs (``dataset.extra_obs_keys``).

Used by the sparse-key penalty (``optim.aux_loss_weights.sparse_keys``): attention mass on keys in sparse tissue -- the tissue's
edges, holes, thinly populated stroma, the seams of fields of view -- is penalised in proportion to how sparse the key is.

Density is counted on the whole section, before any crop or window, so the edge of a crop does not look like the edge of the
tissue. Each section gets its own radius, ``radius_factor`` times its median nearest-neighbour distance: at the section's typical
density a disc of that radius holds about ``pi * radius_factor**2 / 4`` cells (~20 at the default 5), whatever the platform's cell
size. A cell's relative density is its count of other cells within the radius over the section's median count.

The obs column holds ``n_bins + 1`` ordered categories, ``d00 .. d{n_bins-1}`` for ``[c / n_bins, (c + 1) / n_bins)`` of the
relative density and a last one for 1 and above. Category order is code order: ``pd.get_dummies`` keeps it, so the one-hot the
graphs carry decodes with :func:`bin_values`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_density(
    coords_um: np.ndarray, sections, radius_factor: float = 5.0
) -> tuple[np.ndarray, dict[str, float]]:
    """``(relative density [n], radius per section in µm)``: each cell's count of other cells within its section's radius, over the
    section's median count. A section with a single cell gets 1."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords_um, dtype=np.float64)[:, :2]
    sections = np.asarray(pd.Series(sections).astype(str))
    out = np.ones(len(coords), dtype=np.float64)
    radii: dict[str, float] = {}
    for section in pd.unique(sections):
        rows = np.flatnonzero(sections == section)
        if len(rows) < 2:
            radii[section] = float("nan")
            continue
        tree = cKDTree(coords[rows])
        nearest = tree.query(coords[rows], k=2)[0][:, 1]
        radius = float(radius_factor) * float(np.median(nearest))
        counts = tree.query_ball_point(coords[rows], r=radius, return_length=True).astype(np.float64) - 1.0
        median = float(np.median(counts))
        out[rows] = counts / median if median > 0 else 1.0
        radii[section] = radius
    return out, radii


def density_bins(relative: np.ndarray, n_bins: int = 20) -> pd.Categorical:
    """The ordered categorical of ``relative`` (see the module docstring)."""
    names = [f"d{c:02d}" for c in range(n_bins)] + [f"d{n_bins:02d}+"]
    codes = np.clip(np.floor(np.asarray(relative, dtype=np.float64) * n_bins), 0, n_bins).astype(np.int64)
    return pd.Categorical.from_codes(codes, categories=names, ordered=True)


def bin_values(n_bins: int = 20) -> np.ndarray:
    """``[n_bins + 1]`` the relative density each code stands for: a bin's centre, 1 for the last (1 and above)."""
    return np.append((np.arange(n_bins) + 0.5) / n_bins, 1.0)


def sparsity_weights(n_bins: int = 20, threshold: float = 0.6) -> np.ndarray:
    """``[n_bins + 1]`` the penalty weight of a key per code: ``clamp(1 - density / threshold, 0, 1)`` -- 0 from ``threshold`` of
    the section's median density up, rising linearly to 1 for an isolated cell."""
    if not threshold > 0:
        raise ValueError(f"the sparse-key threshold must be positive, got {threshold}")
    return np.clip(1.0 - bin_values(n_bins) / float(threshold), 0.0, 1.0)
