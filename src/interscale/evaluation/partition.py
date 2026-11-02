"""V1: unique and shared R² per gene from frozen features (``.claude/yardsticks_plan.md``).

What does a model's global embedding know about each gene beyond what the cell and its neighbourhood show -- and, for a dual
model, beyond its local embedding? ``masked_probing.attribution_shares`` splits a prediction's shared variance by how training
went; variation partitioning (Peres-Neto et al. 2006) does not: per gene, the out-of-sample R² of a ridge from each group of
feature blocks alone and from all together, and the unique and shared fractions from them.

**Features** (:func:`collect_feature_blocks`): on a masked pass (with ``folds`` every cell hidden once, the graphs' own masks
put back), per kept row the truth, the hidden entries, graph, position and the blocks

* ``own`` -- the cell's input with its hidden entries at 0 (a cell hidden whole has an all-0 row: nothing of its own);
* ``near`` -- per gene the mean of the *visible* entries over the cells within ``hops`` hops on the batch's graph (the local
  component's reach), the graph's visible mean where there is none: what a neighbourhood smoother of the input holds;
* ``local`` -- the local embedding (combined, dual, local models), gathered by ``padded_node_idx``;
* ``global`` -- the global tokens (``gather_tokens``; under the context query its read-out).

``own`` and ``near`` need no model: together they are the local information any model could have.

**Fit** (:func:`partition_r2`): targets are the genes hidden in every kept row (cell folds: every gene; programme masking: the
programme); target and features centred within each graph (a within-section R², as ``masked_probing.within_section``); for
every subset of the named groups a ridge (an alpha per gene by generalised cross-validation, features standardised on the
training rows) fitted out of fold -- folds by graph when there are at least as many graphs as folds, else spatial blocks
(:func:`spatial_folds`). Two groups give ``unique_<A>``, ``unique_<B>`` and ``shared``; more groups their Shapley shares.
Everything after the collector is a function of arrays; nothing names a dataset or a gene.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation.gene_scales import _hide_fold, _masks_kept, cell_folds, hop_neighbourhoods
from interscale.evaluation.online_probes import _rng_untouched
from interscale.evaluation.residuals import shapley_from_subsets

#: The blocks :func:`collect_feature_blocks` can return.
BLOCKS = ("own", "near", "local", "global")
#: The ridge penalties tried per gene (the features are standardised).
DEFAULT_ALPHAS = tuple(np.logspace(-2, 5, 15))
#: The most groups :func:`partition_r2` takes (``2^B - 1`` fits per fold).
MAX_GROUPS = 4


# ------------------------------------------------------------------------------- features


@dataclass
class FeatureBlocks:
    """One pass's frozen features, by kept row.

    ``true`` ``[N, G]``; ``hidden`` ``[N, G]`` the entries the loss scored; ``graph`` ``[N]`` each row's graph in the list;
    ``pos`` ``[N, 2]`` (the graphs' units) or None; ``blocks`` ``{name: [N, P]}``.
    """

    true: np.ndarray
    hidden: np.ndarray
    graph: np.ndarray
    pos: np.ndarray | None
    blocks: dict = field(default_factory=dict)


def visible_neighbour_mean(x: np.ndarray, visible: np.ndarray, reach, graph: np.ndarray) -> np.ndarray:
    """``[N, G]`` per entry the mean of the visible values over each cell's neighbours in ``reach`` (0/1, never itself).

    Where no neighbour shows a gene, the mean of the gene's visible entries in the cell's graph; where the graph shows none, 0.
    """
    import scipy.sparse as sp

    x = np.asarray(x, dtype=np.float64)
    vis = np.asarray(visible, dtype=np.float64)
    reach = sp.csr_matrix(reach, dtype=np.float64)
    num = np.asarray(reach @ (x * vis))
    den = np.asarray(reach @ vis)
    out = np.zeros_like(x)
    has = den > 0
    out[has] = num[has] / den[has]
    graph = np.asarray(graph)
    for g in pd.unique(graph):
        rows = graph == g
        count = vis[rows].sum(axis=0)
        mean = np.where(count > 0, (x[rows] * vis[rows]).sum(axis=0) / np.maximum(count, 1), 0.0)
        fill = rows[:, None] & ~has
        out[fill] = np.broadcast_to(mean, out.shape)[fill]
    return out


def _batch_reach(batch, hops: int):
    import scipy.sparse as sp

    n = int(batch.num_nodes)
    edge = batch.edge_index.detach().cpu().numpy()
    graph = batch.batch.detach().cpu().numpy() if getattr(batch, "batch", None) is not None else np.zeros(n, int)
    adjacency = sp.csr_matrix((np.ones(edge.shape[1]), (edge[0], edge[1])), shape=(n, n))
    return hop_neighbourhoods(adjacency, graph, int(hops))[-1], graph


def _one_pass(module, data_list, *, batch_size, device, hops, blocks, offset_graph=0):
    from interscale.evaluation.masked_probing import _model_heads

    chunks = {k: [] for k in ("true", "hidden", "graph", "pos")}
    feats = {b: [] for b in blocks}
    loader = DataLoader(dataset=data_list, shuffle=False, batch_size=batch_size, num_workers=0)
    start = 0
    for batch in loader:
        batch = batch.to(device)
        out = module._common_step(batch, "regression", "node")
        _, y_true, entry_mask = _model_heads(module, out)
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(batch.num_nodes, device=batch.x.device)
        visible = module.visible_entries(batch)
        if not torch.equal(~visible[idx], entry_mask.bool()):
            raise RuntimeError("the visible entries of the batch and the entries the step scored disagree.")
        x = batch.x.double()
        idx_np = idx.cpu().numpy()
        if "own" in blocks:
            feats["own"].append(torch.where(visible, x, torch.zeros_like(x))[idx].cpu().numpy())
        if "near" in blocks:
            reach, graph = _batch_reach(batch, hops)
            near = visible_neighbour_mean(x.cpu().numpy(), visible.cpu().numpy(), reach, graph)
            feats["near"].append(near[idx_np])
        if "local" in blocks:
            local = out.view.local_embedding
            if local is None:
                raise ValueError("this module has no local embedding; drop 'local' from the blocks.")
            feats["local"].append(local[idx].detach().double().cpu().numpy())
        if "global" in blocks:
            if out.view.global_embedding is None:
                raise ValueError("this module has no global embedding; drop 'global' from the blocks.")
            tokens = out.view.tokens()
            if len(tokens) != len(idx):
                raise RuntimeError("the global tokens and the scored rows differ in number.")
            feats["global"].append(tokens.detach().double().cpu().numpy())
        chunks["true"].append(y_true.detach().double().cpu().numpy())
        chunks["hidden"].append(entry_mask.bool().cpu().numpy())
        chunks["graph"].append((batch.batch[idx] + start + offset_graph).cpu().numpy())
        pos = getattr(batch, "pos", None)
        chunks["pos"].append(None if pos is None else pos[idx, :2].detach().double().cpu().numpy())
        start += int(batch.num_graphs)
    return chunks, feats


def collect_feature_blocks(
    module, data_list, *, batch_size: int, device, blocks=BLOCKS, hops: int = 2, folds: int | None = None, seed: int = 0
) -> FeatureBlocks:
    """:class:`FeatureBlocks` of ``module`` over ``data_list``, read on masked passes in eval mode.

    Without ``folds`` the graphs keep the masks they carry (the evaluation's discs, programme masking) and every scored row is
    kept. With ``folds`` (>= 2) every cell is hidden whole once (:func:`gene_scales.cell_folds`) and only each fold's hidden
    cells are kept; the graphs' own masks are put back. ``hops`` sets the ``near`` block's reach. Draws nothing from the global
    generators.
    """
    blocks = tuple(blocks)
    unknown = set(blocks) - set(BLOCKS)
    if unknown or not blocks:
        raise ValueError(f"blocks are some of {BLOCKS}, got {blocks}.")
    was_training = module.training
    module.eval()
    passes = []
    try:
        with _rng_untouched(device), torch.no_grad():
            kwargs = dict(batch_size=batch_size, device=device, hops=hops, blocks=blocks)
            if folds is None:
                passes.append((_one_pass(module, data_list, **kwargs), None))
            else:
                assigned = cell_folds(data_list, int(folds), seed)
                with _masks_kept(data_list):
                    for fold in range(int(folds)):
                        _hide_fold(data_list, assigned, fold)
                        passes.append((_one_pass(module, data_list, **kwargs), fold))
    finally:
        module.train(was_training)

    rows = {k: [] for k in ("true", "hidden", "graph", "pos")}
    feats = {b: [] for b in blocks}
    for (chunks, fchunks), fold in passes:
        for i in range(len(chunks["true"])):
            keep = chunks["hidden"][i].all(axis=1) if fold is not None else chunks["hidden"][i].any(axis=1)
            for key in rows:
                value = chunks[key][i]
                rows[key].append(None if value is None else value[keep])
            for b in blocks:
                feats[b].append(fchunks[b][i][keep])

    def cat(parts):
        return None if any(p is None for p in parts) else np.concatenate(parts)

    return FeatureBlocks(
        true=cat(rows["true"]),
        hidden=cat(rows["hidden"]),
        graph=cat(rows["graph"]),
        pos=cat(rows["pos"]),
        blocks={b: cat(feats[b]) for b in blocks},
    )


# ------------------------------------------------------------------------------- folds


def spatial_folds(graph, pos=None, n_folds: int = 5, *, block: float | None = None, seed: int = 0) -> np.ndarray:
    """``[N]`` a cross-fitting fold per row.

    With ``block`` None, whole graphs (needs at least ``n_folds`` graphs): no fold sees a neighbour of another fold's cells.
    With ``block`` (in the units of ``pos``), square blocks of that side within each graph, dealt to the folds at random: the
    choice when there are few graphs; a neighbour across a block edge still leaks a little.
    """
    graph = np.asarray(graph)
    rng = np.random.default_rng(seed)
    if int(n_folds) < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    if block is None:
        graphs = pd.unique(graph)
        if len(graphs) < int(n_folds):
            raise ValueError(f"{len(graphs)} graphs for {n_folds} folds by graph; pass `block` for spatial blocks.")
        order = rng.permutation(len(graphs))
        fold_of = {graphs[g]: i % int(n_folds) for i, g in enumerate(order)}
        return np.array([fold_of[g] for g in graph])
    if pos is None:
        raise ValueError("spatial blocks need positions.")
    pos = np.asarray(pos, dtype=np.float64)
    cell = np.floor(pos[:, :2] / float(block)).astype(np.int64)
    keys = pd.Series(list(zip(graph.tolist(), cell[:, 0].tolist(), cell[:, 1].tolist(), strict=True)))
    codes, uniques = pd.factorize(keys)
    order = rng.permutation(len(uniques))
    assign = np.empty(len(uniques), dtype=np.int64)
    assign[order] = np.arange(len(uniques)) % int(n_folds)
    return assign[codes]


# ------------------------------------------------------------------------------- the partition


def _centre(x: np.ndarray, graph: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float64)
    for g in pd.unique(graph):
        rows = graph == g
        out[rows] -= out[rows].mean(axis=0)
    return out


def _oof_ridge(x: np.ndarray, y: np.ndarray, folds: np.ndarray, alphas) -> np.ndarray:
    """Out-of-fold predictions of ``y`` from ``x`` (ridge, an alpha per target, standardised on each training part)."""
    from sklearn.linear_model import RidgeCV

    pred = np.zeros_like(y)
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        if not train.any() or not test.any():
            continue
        mean, sd = x[train].mean(axis=0), x[train].std(axis=0)
        keep = sd > 1e-12
        y_mean = y[train].mean(axis=0)
        if not keep.any():
            pred[test] = y_mean[None, :]  # nothing varies: the training mean, as the ridge's intercept would give
            continue
        xs_train = (x[train][:, keep] - mean[keep]) / sd[keep]
        xs_test = (x[test][:, keep] - mean[keep]) / sd[keep]
        model = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64), alpha_per_target=True, fit_intercept=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(xs_train, y[train] - y_mean)
        pred[test] = np.asarray(model.predict(xs_test)).reshape(-1, y.shape[1]) + y_mean
    return pred


@dataclass
class Partition:
    """Per gene the R² of every subset of the groups and the partition.

    ``table``: ``r2_<subset>`` for every non-empty subset (groups joined by ``+``), and for two groups ``unique_<A>``,
    ``unique_<B>``, ``shared``; for any number ``share_<group>`` (Shapley) and ``total`` (all groups). ``settings``: the groups,
    folds, rows.
    """

    table: pd.DataFrame
    settings: dict


def partition_r2(
    features: FeatureBlocks,
    genes,
    *,
    groups: dict | None = None,
    folds: np.ndarray,
    alphas=DEFAULT_ALPHAS,
    within_graph: bool = True,
) -> Partition:
    """Variation partitioning of each hidden gene among groups of feature blocks.

    Parameters
    ----------
    features
        From :func:`collect_feature_blocks`.
    genes
        The ``G`` gene names.
    groups
        ``{group: [block, ...]}``; default each block its own group. At most :data:`MAX_GROUPS`.
    folds
        ``[N]`` the cross-fitting fold of each row (:func:`spatial_folds`).
    alphas
        Ridge penalties tried per gene.
    within_graph
        Centre target and features within each graph (the within-section R²).
    """
    genes = [str(g) for g in genes]
    true, hidden, graph = features.true, features.hidden, np.asarray(features.graph)
    if true.shape != (len(graph), len(genes)) or hidden.shape != true.shape:
        raise ValueError(f"the truth is {true.shape}, the rows {len(graph)} and the genes {len(genes)}.")
    groups = {b: [b] for b in features.blocks} if groups is None else {str(k): list(v) for k, v in groups.items()}
    if not 1 <= len(groups) <= MAX_GROUPS:
        raise ValueError(f"partition_r2 takes 1 to {MAX_GROUPS} groups, got {len(groups)}.")
    for name, members in groups.items():
        missing = [m for m in members if m not in features.blocks]
        if missing or not members:
            raise ValueError(
                f"group {name!r} names blocks {missing or members} the features do not hold ({list(features.blocks)})."
            )
    folds = np.asarray(folds)
    if len(folds) != len(graph):
        raise ValueError(f"{len(folds)} folds for {len(graph)} rows.")
    scored = hidden.all(axis=0)
    if not scored.any():
        raise ValueError(
            "no gene is hidden in every row: the partition needs cell folds (folds=...) or a gene set hidden in every cell."
        )
    y = true[:, scored]
    xs = {b: np.asarray(v, dtype=np.float64) for b, v in features.blocks.items()}
    if within_graph:
        y = _centre(y, graph)
        xs = {b: _centre(v, graph) for b, v in xs.items()}
    else:
        y = y - y.mean(axis=0)
    ss_tot = (y**2).sum(axis=0)
    names = list(groups)
    r2 = {(): np.zeros(y.shape[1])}
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(range(len(names)), size):
            blocks = [b for i in subset for b in groups[names[i]]]
            x = np.hstack([xs[b] for b in dict.fromkeys(blocks)])
            pred = _oof_ridge(x, y, folds, alphas)
            with np.errstate(invalid="ignore", divide="ignore"):
                r2[subset] = np.where(
                    ss_tot > 0, 1.0 - ((y - pred) ** 2).sum(axis=0) / np.where(ss_tot > 0, ss_tot, 1.0), np.nan
                )
    table = {}
    for subset, value in r2.items():
        if subset:
            table["r2_" + "+".join(names[i] for i in subset)] = value
    everything = tuple(range(len(names)))
    if len(names) == 2:
        a, b = names
        table[f"unique_{a}"] = r2[(0, 1)] - r2[(1,)]
        table[f"unique_{b}"] = r2[(0, 1)] - r2[(0,)]
        table["shared"] = r2[(0,)] + r2[(1,)] - r2[(0, 1)]
    for name, share in shapley_from_subsets(r2, names).items():
        table[f"share_{name}"] = share
    table["total"] = r2[everything]
    index = pd.Index([g for g, s in zip(genes, scored, strict=True) if s], name="gene")
    settings = {
        "groups": groups,
        "n_rows": int(len(graph)),
        "n_graphs": int(len(pd.unique(graph))),
        "n_folds": int(len(np.unique(folds))),
        "within_graph": bool(within_graph),
        "n_genes_scored": int(scored.sum()),
    }
    return Partition(table=pd.DataFrame(table, index=index), settings=settings)
