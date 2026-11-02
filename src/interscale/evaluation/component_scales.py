"""V3: a per-gene split of one prediction by the spatial scale of its components (``.claude/yardsticks_plan.md``).

A global-only model has one head and no parts, so it cannot be split the way the additive decoder is. Its prediction can: the
SVD of the prediction over the cells (the rotation-free directions of ``dimensions.prediction_readout``) gives components, each
with a map over the cells and a loading per gene. A component whose map varies on a scale wider than the local component's
reach is **global**, one within it **local**, one without spatial structure **unstructured** (NSFH's spatial importance score
and MEFISTO's per-factor smoothness, read off a trained model). A gene's prediction variance splits over the components, so
each gene gets a global, a local and an unstructured share, and a loading-weighted length scale.

:func:`component_scales` needs every cell of the sections predicted once (a fold pass: ``gene_scales.collect_fold_parts``,
``dimensions.collect_dimensions(folds=...)``), or the maps are patches. It reads any ``[N, G]`` prediction: a model head, a
reference (``ref_nearest_<k>``), the truth (the data's own split, the comparison). The component maps' scales come from
``gene_scales`` (Moran's I curves against a permutation null, the half point after the peak), with the components in place of
genes. Dataset-agnostic: the reach, the neighbourhood sizes and the gene names come from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from interscale.evaluation.dimensions import _centre_rows, _signed
from interscale.evaluation.gene_scales import (
    DEFAULT_KS,
    VARIABLE,
    _reach_um,
    hop_neighbourhoods,
    length_scales,
    moran_curves,
    permutation_null,
    sample_neighbours,
)

#: A component's class.
GLOBAL, LOCAL, UNSTRUCTURED = "global", "local", "unstructured"
CLASSES = (GLOBAL, LOCAL, UNSTRUCTURED)


@dataclass
class FoldPredictions:
    """Every head's prediction at the kept rows: ``heads`` ``{name: [N, G]}``, ``true``, ``hidden`` ``[N, G]``, ``graph`` ``[N]``
    (the graph's index in the list), ``pos`` ``[N, 2]`` (the graphs' units) or None.
    """

    heads: dict
    true: np.ndarray
    hidden: np.ndarray
    graph: np.ndarray
    pos: np.ndarray | None


def _prediction_pass(module, data_list, *, batch_size, device):
    import torch
    from torch_geometric.loader import DataLoader

    from interscale.evaluation.masked_probing import _model_heads

    chunks = {k: [] for k in ("true", "hidden", "graph", "pos")}
    heads: dict = {}
    loader = DataLoader(dataset=data_list, shuffle=False, batch_size=batch_size, num_workers=0)
    start = 0
    for batch in loader:
        batch = batch.to(device)
        out = module._common_step(batch, "regression", "node")
        found, y_true, entry_mask = _model_heads(module, out)
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(batch.num_nodes, device=batch.x.device)
        for name, value in found.items():
            heads.setdefault(name, []).append(value.detach().double().cpu().numpy())
        chunks["true"].append(y_true.detach().double().cpu().numpy())
        chunks["hidden"].append(entry_mask.bool().cpu().numpy())
        chunks["graph"].append((batch.batch[idx] + start).cpu().numpy())
        pos = getattr(batch, "pos", None)
        chunks["pos"].append(None if pos is None else pos[idx, :2].detach().double().cpu().numpy())
        start += int(batch.num_graphs)
    return heads, chunks


def collect_fold_predictions(module, data_list, *, batch_size: int, device, folds: int | None = None, seed: int = 0):
    """:class:`FoldPredictions` of every head over ``data_list``, read in eval mode.

    With ``folds`` (>= 2) every cell is hidden whole once (``gene_scales.cell_folds``) and each fold's hidden cells are kept, the
    graphs' own masks put back: every cell predicted from its context once (cell masking). Without, the graphs' own masks and
    the rows with a hidden entry (programme masking: every cell, its programme genes hidden). Any decoder, parts summed as the
    model reports them. No global generator is touched.
    """
    import torch

    from interscale.evaluation.gene_scales import _hide_fold, _masks_kept, cell_folds
    from interscale.evaluation.online_probes import _rng_untouched

    was_training = module.training
    module.eval()
    passes = []
    try:
        with _rng_untouched(device), torch.no_grad():
            if folds is None:
                passes.append((_prediction_pass(module, data_list, batch_size=batch_size, device=device), None))
            else:
                assigned = cell_folds(data_list, int(folds), seed)
                with _masks_kept(data_list):
                    for fold in range(int(folds)):
                        _hide_fold(data_list, assigned, fold)
                        passes.append((_prediction_pass(module, data_list, batch_size=batch_size, device=device), fold))
    finally:
        module.train(was_training)
    rows = {k: [] for k in ("true", "hidden", "graph", "pos")}
    kept_heads: dict = {}
    for (heads, chunks), fold in passes:
        for i in range(len(chunks["true"])):
            keep = chunks["hidden"][i].all(axis=1) if fold is not None else chunks["hidden"][i].any(axis=1)
            for key in rows:
                value = chunks[key][i]
                rows[key].append(None if value is None else value[keep])
            for name, values in heads.items():
                kept_heads.setdefault(name, []).append(values[i][keep])

    def cat(parts):
        return None if any(p is None for p in parts) else np.concatenate(parts)

    return FoldPredictions(
        heads={name: np.concatenate(v) for name, v in kept_heads.items()},
        true=cat(rows["true"]),
        hidden=cat(rows["hidden"]),
        graph=cat(rows["graph"]),
        pos=cat(rows["pos"]),
    )


def reach_from_graph(adjacency, graph, pos, *, hops: int = 2, unit_um: float = 1.0) -> float:
    """The local component's reach in µm: the median over cells of the distance to the farthest cell within ``hops`` hops."""
    within = hop_neighbourhoods(adjacency, np.asarray(graph), int(hops))[-1]
    return _reach_um(within, np.asarray(pos, dtype=np.float64), float(unit_um))


def prediction_components(pred, graph, *, n_components: int = 20):
    """``(scores [N, K], loadings [G, K], singular_values [K], gene_variance [G])`` of a prediction centred within each graph.

    ``pred = C S U^T`` over ``sqrt(n - 1)``: ``scores`` = ``C S`` (each component's map, in the units of the prediction),
    ``loadings`` = ``U`` (sign-fixed: each column's largest gene positive); ``gene_variance`` each gene's variance, the sum of
    ``S_k² U_gk²`` over all components.
    """
    centred = _centre_rows(np.asarray(pred, dtype=np.float64), np.asarray(graph))
    n = len(centred)
    scale = np.sqrt(max(n - 1, 1))
    u_cells, s, vt = np.linalg.svd(centred / scale, full_matrices=False)
    k = min(int(n_components), len(s))
    loadings = _signed(vt[:k].T)
    sign = np.sign(np.sum(loadings * vt[:k].T, axis=0))  # the flips _signed made, applied to the maps too
    scores = u_cells[:, :k] * s[:k] * scale * sign
    gene_variance = (centred**2).sum(axis=0) / max(n - 1, 1)
    return scores, loadings, s[:k], gene_variance


def _weighted_median_scale(weight: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Per gene the loading-weighted median of the components' length scales (``inf`` allowed: a component that never halves
    within the sizes); NaN for a gene with no weight on them. A median, not a mean: one wide component's ``inf`` would
    otherwise be dropped or swamp the rest.
    """
    out = np.full(weight.shape[0], np.nan)
    if weight.shape[1] == 0:
        return out
    order = np.argsort(ell, kind="stable")
    w, e = weight[:, order], ell[order]
    total = w.sum(axis=1)
    cum = np.cumsum(w, axis=1)
    for g in np.flatnonzero(total > 0):
        out[g] = e[int(np.searchsorted(cum[g], 0.5 * total[g]))]
    return out


@dataclass
class ComponentScales:
    """``components``: per component its singular value, share of the prediction's variance, Moran's I pattern and length
    scale (``ell_um``) and class; ``genes``: per gene ``share_global``, ``share_local``, ``share_unstructured``, ``share_rest``
    (variance beyond the components read), ``ell_um`` (loading-weighted median of the structured components' scales, ``inf``
    for a gene carried by components wider than the sizes read)
    and ``variance``; ``settings``.
    """

    components: pd.DataFrame
    genes: pd.DataFrame
    settings: dict


def component_scales(
    pred,
    graph,
    pos,
    genes,
    *,
    reach_um: float,
    n_components: int = 20,
    ks=DEFAULT_KS,
    n_centres: int | None = 2000,
    n_permutations: int = 10,
    z: float = 3.0,
    unit_um: float = 1.0,
    seed: int = 0,
) -> ComponentScales:
    """Split each gene of ``pred`` ``[N, G]`` (every cell of the graphs once) by the scale of the prediction's components.

    Parameters
    ----------
    pred
        The prediction (or the truth) at every cell.
    graph, pos
        ``[N]`` each row's graph, ``[N, 2]`` its position in units of ``unit_um`` µm.
    genes
        The ``G`` gene names.
    reach_um
        The local component's reach in µm (:func:`reach_from_graph`): a component wider than it is global.
    n_components
        Components read (the rest of each gene's variance is ``share_rest``).
    ks, n_centres, n_permutations, z, seed
        The Moran's I curves of the component maps (``gene_scales``).
    """
    genes = [str(g) for g in genes]
    pred = np.asarray(pred, dtype=np.float64)
    graph = np.asarray(graph)
    if pred.shape != (len(graph), len(genes)):
        raise ValueError(f"the prediction is {pred.shape}, the rows {len(graph)} and the genes {len(genes)}.")
    if not np.isfinite(pred).all():
        raise ValueError("the prediction must be finite at every row (one fold pass over every cell).")
    scores, loadings, s, gene_variance = prediction_components(pred, graph, n_components=n_components)
    names = [f"c{k}" for k in range(len(s))]
    k_max = int(max(ks))
    neighbours = sample_neighbours(pos, graph, k_max=k_max, n_centres=n_centres, seed=seed)
    curves = moran_curves(scores, graph, neighbours, ks, unit_um=unit_um)
    null = permutation_null(scores, graph, neighbours, ks, n_permutations=n_permutations, seed=seed + 1)
    scales = length_scales(curves, null, names, z=z)
    variable = (scales["pattern"] == VARIABLE).to_numpy()
    ell = scales["ell_um"].to_numpy(dtype=np.float64)
    wide = variable & (ell > float(reach_um))  # inf (never halves within the sizes) counts as wide
    classes = np.where(wide, GLOBAL, np.where(variable, LOCAL, UNSTRUCTURED))
    total = float((s**2).sum()) if len(s) else 0.0
    components = scales.copy()
    components.insert(0, "singular_value", s)
    components.insert(1, "share", s**2 / total if total > 0 else np.nan)
    components["class"] = classes

    weight = loadings**2 * s**2  # [G, K]: gene g's variance carried by component k
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {}
        for c in CLASSES:
            out[f"share_{c}"] = np.where(gene_variance > 0, weight[:, classes == c].sum(axis=1) / gene_variance, np.nan)
        out["share_rest"] = np.where(gene_variance > 0, 1.0 - weight.sum(axis=1) / gene_variance, np.nan)
        out["ell_um"] = _weighted_median_scale(weight[:, variable], ell[variable])
        out["variance"] = gene_variance
    gene_table = pd.DataFrame(out, index=pd.Index(genes, name="gene"))
    settings = {
        "reach_um": float(reach_um),
        "n_components": int(len(s)),
        "ks": [int(k) for k in ks],
        "n_centres": None if n_centres is None else int(n_centres),
        "n_permutations": int(n_permutations),
        "z": float(z),
        "seed": int(seed),
        "n_rows": int(len(graph)),
        "n_graphs": int(len(pd.unique(graph))),
    }
    return ComponentScales(components=components, genes=gene_table, settings=settings)
