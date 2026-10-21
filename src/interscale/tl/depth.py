"""How deep a hidden cell lies, and the loss regimes that depth defines (``.claude/decoder_plan.md``, D2.2).

A cell hidden whole is *near* when a visible cell lies within the local component's reach, so its
neighbourhood can fill it in, and *deep* beyond that, where only the global component can reach.
``optim.regime_loss`` gives each head of the dual decoder the cells of its regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: What ``optim.regime_loss.{local,global}_cells`` may name: the hidden cells within ``depth_um`` of a
#: visible cell, those beyond it, or both.
REGIME_CELLS = ("near", "deep", "all")
#: ``optim.regime_loss.kind``: no split (every run before D2.2), or a split by depth.
REGIME_KINDS = ("none", "depth")

#: Hidden cells whose distances to the visible cells are formed at once: bounds a ``[chunk, visible]`` matrix.
DEPTH_CHUNK = 4096


def hidden_depth(pos: torch.Tensor, hidden_cell: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    """``[N]`` distance from each hidden cell to the nearest non-hidden cell of the same graph.

    In the units of ``pos``; 0 for a cell that is not hidden, ``inf`` in a graph hidden completely.
    """
    depth = torch.zeros(len(pos), dtype=torch.float64, device=pos.device)
    for g in torch.unique(graph):
        rows = torch.where(graph == g)[0]
        hid = hidden_cell[rows]
        if not hid.any():
            continue
        p = pos[rows].double()
        if hid.all():
            depth[rows] = float("inf")
            continue
        visible = p[~hid]
        hidden_rows = rows[hid]
        hidden_pos = p[hid]
        for start in range(0, len(hidden_rows), DEPTH_CHUNK):
            chunk = slice(start, start + DEPTH_CHUNK)
            depth[hidden_rows[chunk]] = torch.cdist(hidden_pos[chunk], visible).min(dim=1).values
    return depth


def regime_rows(depth: torch.Tensor, threshold: float, cells: str) -> torch.Tensor:
    """``[N]`` bool: the cells of ``depth`` that ``cells`` (:data:`REGIME_CELLS`) selects.

    ``threshold`` is in the units of ``depth``.

    ``near`` is a depth up to the threshold (a cell that is not hidden has depth 0, so it is near), ``deep``
    beyond it (a graph hidden completely, depth ``inf``, is deep).
    """
    if cells == "near":
        return depth <= threshold
    if cells == "deep":
        return depth > threshold
    if cells == "all":
        return torch.ones_like(depth, dtype=torch.bool)
    raise ValueError(f"cells must be one of {REGIME_CELLS}, got {cells!r}.")


def regime_loss_on(cfg) -> bool:
    """Whether ``cfg`` splits the dual decoder's loss by depth (``optim.regime_loss.kind`` other than ``none``)."""
    regime = cfg.optim.get("regime_loss", None)
    return regime is not None and regime.get("kind", "none") != "none"


@dataclass(frozen=True)
class RegimeLoss:
    """Which hidden cells each head of a dual decoder is trained on (``optim.regime_loss``).

    Attributes
    ----------
    threshold
        The depth that separates near from deep, in the units of the graphs' ``pos``.
    local_cells, global_cells
        The regime of each head's loss (:data:`REGIME_CELLS`).
    """

    threshold: float
    local_cells: str = "near"
    global_cells: str = "deep"

    @classmethod
    def from_config(cls, cfg) -> RegimeLoss | None:
        """The regimes ``cfg`` asks for, or None (the loss on every hidden cell, as before D2.2)."""
        if not regime_loss_on(cfg):
            return None
        regime = cfg.optim.regime_loss
        return cls(
            threshold=float(regime.depth_um) / float(cfg.dataset.spatial_unit_um),
            local_cells=regime.local_cells,
            global_cells=regime.global_cells,
        )

    def loss_mask(self, entry_mask, hidden_cell, pos, graph, node_idx) -> torch.Tensor:
        """``[2K, F]`` bool: ``entry_mask`` ``[K, F]`` stacked twice, each half cut to its head's regime.

        The local half keeps the local regime's cells, the global half the global regime's.

        ``hidden_cell`` ``[N]`` marks the cells hidden whole, ``pos`` ``[N, 2]`` and ``graph`` ``[N]`` are the
        batch's, and ``node_idx`` ``[K]`` the batch-global index of each of the ``K`` kept cells, in the order of
        ``entry_mask``. A hidden cell's depth is its distance to the nearest visible cell of its own graph, over
        every cell of the graph (also those ``pad_batch`` left out of the transformer, which the local
        component saw).
        """
        depth = hidden_depth(pos, hidden_cell, graph)[node_idx]
        local = regime_rows(depth, self.threshold, self.local_cells)
        glob = regime_rows(depth, self.threshold, self.global_cells)
        return torch.cat([entry_mask & local[:, None], entry_mask & glob[:, None]], dim=0)
