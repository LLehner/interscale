"""Training on crops of whole sections (``dataset.crop``; roadmap Tier 3, K18 and decision 12).

On CosMx one optimiser step is one section and a fold trains on two of them: 2 steps per epoch, against 24 on synth_spot, and the
whole-section runs stayed undertrained. With ``dataset.crop.size_um`` > 0 every training graph hands the trainer
``crop.per_graph`` random crops per epoch instead -- squares (or discs) of ``size_um`` around a cell drawn uniformly, cut out with
everything the graph carries per cell (expression, masks, positions, precomputed encodings, annotations) and the edges among the
kept cells. Validation and test stay whole sections, which is what the model is read on (decision 12).

Invariants:

* **The masks are the section's.** A crop slices the training graph's current ``mask`` / ``gene_mask``, drawn on the whole section
  (discs included) and redrawn every epoch as before; a crop draws no mask of its own.
* **Positions alone choose a crop.** The centre is a cell drawn uniformly, the window a fixed size in µm; nothing hidden decides it.
* **Its own randomness.** In the main process a private generator seeded from ``optim.seed``; in a loader worker one seeded from the
  worker's seed (redrawn by the loader every epoch) and the item, so an epoch's crops are new and a run is reproducible either way.
  Off (``size_um`` 0), nothing is drawn and the training graphs are handed over as before.
* **What changes between crops and sections** (read before trusting a run trained on crops): per-sequence quantities are taken over the
  crop -- ``attention.center_qk`` centres on the crop's mean token, ``pe.center_coords`` on its centroid, CLS reads the crop, pooled
  grids start at its corner, a hierarchy is built per crop -- and at inference over the section. A precomputed encoding (LapPE, RWPE,
  spectral, the density radius) keeps the section's values.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

#: The shapes of a crop (``dataset.crop.shape``): a square of side ``size``, or a disc of diameter ``size``.
CROP_SHAPES = ("square", "disc")


def crop_window(pos: torch.Tensor, centre: torch.Tensor, size: float, shape: str) -> torch.Tensor:
    """``[N]`` bool: the cells of ``pos`` ``[N, D]`` inside the window of ``size`` (side or diameter, ``pos`` units) at ``centre``."""
    offset = pos[:, :2] - centre[:2]
    if shape == "square":
        return (offset.abs() <= size / 2).all(dim=1)
    if shape == "disc":
        return offset.norm(dim=1) <= size / 2
    raise ValueError(f"crop shape must be one of {CROP_SHAPES}, got {shape!r}.")


def crop_graph(
    data: Data, size: float, *, shape: str = "square", generator: torch.Generator, min_cells: int = 2, tries: int = 20
) -> Data:
    """A random crop of ``data``: the subgraph of the cells in a window of ``size`` (``pos`` units) around a cell drawn uniformly.

    Every per-cell attribute is sliced and the edges among the kept cells are kept, renumbered (``Data.subgraph``); per-graph ones
    (e.g. ``spectral_eigval``) stay. A window with fewer than ``min_cells`` cells is drawn again, up to ``tries`` times, then the
    largest seen is used.
    """
    pos = getattr(data, "pos", None)
    if pos is None:
        raise ValueError("a crop is cut by position, but the graph carries no `pos`. Set dataset.spatial_key.")
    n = int(data.num_nodes)
    best = None
    for _ in range(int(tries)):
        centre = pos[int(torch.randint(n, (1,), generator=generator))]
        inside = crop_window(pos, centre, float(size), shape)
        if best is None or int(inside.sum()) > int(best.sum()):
            best = inside
        if int(inside.sum()) >= int(min_cells):
            break
    return data.subgraph(best.nonzero().squeeze(1))


class CropDataset(torch.utils.data.Dataset):
    """``per_graph`` random crops of each of ``graphs`` (:func:`crop_graph`), new ones every time an item is read.

    Item ``i`` is a crop of graph ``i // per_graph``. The graphs are held, not copied: their masks, redrawn in place every epoch,
    are the ones a crop slices.
    """

    def __init__(self, graphs, *, size: float, shape: str, per_graph: int, min_cells: int, seed: int):
        if not size > 0 or int(per_graph) < 1 or shape not in CROP_SHAPES:
            raise ValueError(
                f"crops need size > 0, per_graph >= 1 and a shape in {CROP_SHAPES}; got {size}, {per_graph}, {shape!r}."
            )
        self.graphs = graphs
        self.size, self.shape, self.per_graph, self.min_cells = float(size), shape, int(per_graph), int(min_cells)
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)

    def __len__(self) -> int:
        return len(self.graphs) * self.per_graph

    def _generator(self, index: int) -> torch.Generator:
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            return self.generator  # the main process: one stream, never restarted
        # A worker holds a copy made when the epoch's iterator started; its seed is redrawn by the loader every epoch.
        return torch.Generator().manual_seed((int(worker.seed) * 1_000_003 + int(index)) % (2**63 - 1))

    def __getitem__(self, index: int) -> Data:
        graph = self.graphs[int(index) // self.per_graph]
        return crop_graph(
            graph, self.size, shape=self.shape, generator=self._generator(int(index)), min_cells=self.min_cells
        )
