"""Effective receptive fields: which cells a hidden cell's prediction depends on, by gradient.

This replaces the relevance rollout (``tl.self_attn_relevance``) for node-level models. That
rollout backpropagates the sum of the transformer output after its affine LayerNorm, which has no
gradient while the norm's scale is uniform (at initialisation) and an arbitrary one after: two
random scales on otherwise equal weights give relevance maps correlating at 0.14.

Here the target is one hidden cell's own prediction, projected on a random direction. For ``w``
drawn from a standard normal, ``E_w ||d(w . f_i) / d u_j||^2 = ||J_ij||_F^2``: the squared Frobenius
norm of the Jacobian of cell ``i``'s prediction ``f_i`` with respect to cell ``j``'s input ``u_j``
(Hutchinson's estimator; ``n_projections`` directions are averaged). That is the influence of cell
``j`` on the prediction of cell ``i``, one backward pass per query and direction, for any model:

* the inputs ``u_j`` are ``batch.x`` for a model with a GCN, and the per-cell tokens of
  ``gex_embedding`` for one without (a learned map of the expression, or PCA/NMF/precomputed vectors,
  which are cut off from ``batch.x``). Either way row ``j`` holds cell ``j``'s own data alone.
* the graphs keep the masks they carry (e.g. discs from ``masked_probing.draw_masks``), as in
  training. Queries are wholly hidden cells, sampled per depth bin (distance to the nearest visible
  cell), and influence is read over the visible cells: a hidden cell's input is a constant.
* a GCN's head is blind beyond its hops by construction, which makes a local model the check of the
  instrument: its deep queries must read no influence at all.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation._spatial_weights import (
    add_finite,
    bin_labels,
    distance_profile,
    graph_edges,
    hidden_cells,
    matched_ranks,
    pair_exclusions,
    query_groups,
    row_statistics,
)
from interscale.evaluation.online_probes import _rng_untouched, hidden_depth

if TYPE_CHECKING:
    from collections.abc import Sequence


@contextmanager
def _differentiable_inputs(module, batch):
    """Make the per-cell inputs of the next pass differentiable; yield a dict that will hold them.

    A module without a local component reads its tokens from ``gex_embedding``; they are captured
    there (as leaves, since the parameters are frozen during the pass). Any other module reads
    ``batch.x``, which becomes a leaf that requires grad.
    """
    captured = {}
    if hasattr(module, "gex_embedding") and not hasattr(module, "local_module"):
        original = module.gex_embedding

        def capture(b, hidden=None):
            tokens = original(b, hidden=hidden).detach().requires_grad_(True)
            captured["inputs"] = tokens
            return tokens

        module.gex_embedding = capture
        try:
            yield captured
        finally:
            del module.gex_embedding
    else:
        batch.x = batch.x.detach().clone().requires_grad_(True)
        captured["inputs"] = batch.x
        yield captured


@contextmanager
def _frozen_parameters(module):
    """No parameter gradients during the pass; each parameter's own ``requires_grad`` restored after."""
    saved = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def _sample(candidates: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Up to ``n`` of ``candidates``, drawn without replacement, in ascending order."""
    if len(candidates) <= n:
        return np.sort(candidates)
    return np.sort(rng.choice(candidates, size=n, replace=False))


def receptive_field(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    spatial_unit_um: float,
    depth_bins_um: Sequence[float],
    radii_um: Sequence[float] = (),
    distance_bins_um: Sequence[float] = (),
    queries_per_bin: int = 16,
    n_projections: int = 2,
    edges: pd.DataFrame | None = None,
    edge_queries: int = 16,
    tolerance_um: float = 10.0,
    seed: int = 0,
    max_graphs: int = 0,
) -> dict[str, pd.DataFrame]:
    """The influence of every visible cell on sampled hidden cells' predictions, per decoder head.

    Parameters
    ----------
    module
        The module of any model: a dual-decoder ``CombinedModel`` gives a ``local`` and a ``global``
        head, the others one (``masked_probing._model_heads``).
    data_list
        PyG ``Data`` objects carrying ``pos`` and the masks to read the model under.
    prediction_task, prediction_level
        Must be ``"regression"`` and ``"node"``.
    device
        Device to move each graph to.
    spatial_unit_um
        µm per unit of ``pos``.
    depth_bins_um
        Depth bins (µm to the nearest visible cell) the queries are sampled from and grouped by.
    radii_um
        Radii for the share of influence beyond them. For a query at depth ``d`` every visible cell
        lies at least ``d`` away, so a radius below the depth reads 1 whenever there is influence.
    distance_bins_um
        Bins for the influence profile by distance.
    queries_per_bin
        Hidden cells sampled per depth bin and graph.
    n_projections
        Random directions averaged per query (Hutchinson); each costs one backward pass.
    edges
        Planted ``sender``, ``receiver``, ``range_class`` pairs by position in the AnnData the graphs
        were built from. Up to ``edge_queries`` wholly hidden receivers with a visible sender are
        sampled per range class and graph, beside the depth sample.
    tolerance_um
        How far a comparison cell's distance to the receiver may differ from the sender's.
    seed
        Seeds the samples and the directions; no global generator is touched.
    max_graphs
        How many graphs to read; 0 or less reads all of them.

    Returns
    -------
    dict
        ``"by_query"``: ``head, query, metric, value, n_queries`` for the depth sample -- ``influence``,
        the summed influence of the visible cells (exactly 0 when none reaches the prediction), and
        over its shares ``span_um``, ``span_rel``, ``effective_rel``, ``beyond_<r>um`` against uniform
        weights on the visible cells (:func:`~interscale.evaluation._spatial_weights.row_statistics`).
        ``"profile"`` (with ``distance_bins_um``): ``head, query, distance_um, lift, n_queries``, the
        influence per distance bin over uniform's. ``"edges"`` (with ``edges``): ``head, range_class,
        metric, value, n_pairs`` -- ``matched_rank``, the sender's rank among the visible cells about
        as far from the receiver: 0.5 is influence blind to the edge, 1 the top.
    """
    from interscale.evaluation.masked_probing import _model_heads

    if "regression" not in prediction_task or prediction_level != "node":
        raise ValueError("The receptive field is read off reconstructed expression; it needs regression at node level.")
    graphs = list(data_list)
    if max_graphs > 0:
        graphs = graphs[:max_graphs]
    groups = bin_labels(depth_bins_um)
    unit = float(spatial_unit_um)
    rng = np.random.default_rng(seed)
    directions = torch.Generator().manual_seed(int(seed))
    sums: dict[tuple, list] = {}
    profile: dict[tuple, np.ndarray] = {}
    edge_sums: dict[tuple, np.ndarray] = {}
    n_bins = max(len(distance_bins_um) - 1, 0)

    was_training = module.training
    module.eval()
    loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
    try:
        with _rng_untouched(device), _frozen_parameters(module):
            for batch in loader:
                batch = batch.to(device)
                if getattr(batch, "pos", None) is None:
                    raise ValueError("receptive_field reads distances, but a graph carries no `pos`.")
                hidden = hidden_cells(module, batch)
                depth = hidden_depth(batch.pos, hidden, batch.batch) * unit
                group = query_groups(depth.cpu().numpy(), hidden.cpu().numpy(), depth_bins_um)
                chosen = [_sample(np.flatnonzero(group == g), queries_per_bin, rng) for g in groups]
                depth_queries = np.concatenate(chosen) if chosen else np.empty(0, np.int64)
                pairs = pd.DataFrame(columns=["sender", "receiver", "range_class"])
                if edges is not None:
                    pairs = graph_edges(edges, batch.obs_names)
                    hid = hidden.cpu().numpy()
                    pairs = pairs[hid[pairs["receiver"].to_numpy()] & ~hid[pairs["sender"].to_numpy()]]
                    picked = [
                        _sample(np.unique(sub["receiver"].to_numpy()), edge_queries, rng)
                        for _, sub in pairs.groupby("range_class")
                    ]
                    receivers = np.concatenate(picked) if picked else np.empty(0, np.int64)
                    pairs = pairs[np.isin(pairs["receiver"].to_numpy(), receivers)].reset_index(drop=True)
                queries = np.union1d(depth_queries, pairs["receiver"].to_numpy().astype(np.int64))
                if not len(queries):
                    continue

                influence = _influence(module, batch, queries, prediction_task, n_projections, directions, _model_heads)
                where = batch.pos.double() * unit
                dist = torch.cdist(where[queries], where, compute_mode="donot_use_mm_for_euclid_dist")
                keep = (~hidden)[None, :].expand(len(queries), -1)
                in_depth_sample = np.isin(queries, depth_queries)
                query_group = group[queries]
                row_of = {int(q): k for k, q in enumerate(queries)}
                for head, weights in influence.items():
                    if weights is None:
                        continue
                    stats = {
                        "influence": torch.where(keep, weights, torch.zeros_like(weights)).sum(1),
                        **row_statistics(weights, dist, keep, radii_um),
                    }
                    for metric, values in stats.items():
                        values = values.cpu().numpy()
                        for g in groups:
                            add_finite(sums, (head, g, metric), values, in_depth_sample & (query_group == g))
                    if n_bins:
                        weight_in, uniform_in = distance_profile(weights, dist, keep, distance_bins_um)
                        defined = torch.isfinite(weight_in).all(1).cpu().numpy()
                        weight_in, uniform_in = weight_in.cpu().numpy(), uniform_in.cpu().numpy()
                        for g in groups:
                            pick = in_depth_sample & (query_group == g) & defined
                            if pick.any():
                                acc = profile.setdefault((head, g), np.zeros((3, n_bins)))
                                acc += [weight_in[pick].sum(0), uniform_in[pick].sum(0), np.full(n_bins, pick.sum())]
                    if len(pairs):
                        rows = torch.as_tensor([row_of[int(r)] for r in pairs["receiver"]], device=weights.device)
                        senders = torch.as_tensor(pairs["sender"].to_numpy().astype(np.int64), device=weights.device)
                        ranks = (
                            matched_ranks(
                                weights,
                                dist,
                                keep,
                                rows,
                                senders,
                                pair_exclusions(pairs, batch.num_nodes, weights.device),
                                tolerance_um,
                            )
                            .cpu()
                            .numpy()
                        )
                        classes = pairs["range_class"].astype(str).to_numpy()
                        for range_class in [*np.unique(classes), "any"]:
                            pick = np.ones(len(classes), bool) if range_class == "any" else classes == range_class
                            ranked = pick & np.isfinite(ranks)
                            acc = edge_sums.setdefault((head, str(range_class)), np.zeros(2))
                            acc += [ranks[ranked].sum(), ranked.sum()]
    finally:
        module.train(was_training)

    order = {g: k for k, g in enumerate(groups)}
    rows = [
        {"head": head, "query": g, "metric": metric, "value": total / count, "n_queries": count}
        for (head, g, metric), (total, count) in sorted(
            sums.items(), key=lambda kv: (kv[0][0], order[kv[0][1]], kv[0][2])
        )
    ]
    out = {"by_query": pd.DataFrame(rows, columns=["head", "query", "metric", "value", "n_queries"])}
    if n_bins:
        labels = bin_labels(distance_bins_um)
        rows = []
        for (head, g), (weight, uniform, count) in sorted(profile.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
            for b, label in enumerate(labels):
                lift = weight[b] / uniform[b] if uniform[b] > 0 else np.nan
                rows.append({"head": head, "query": g, "distance_um": label, "lift": lift, "n_queries": int(count[b])})
        out["profile"] = pd.DataFrame(rows, columns=["head", "query", "distance_um", "lift", "n_queries"])
    if edges is not None:
        rows = [
            {
                "head": head,
                "range_class": rc,
                "metric": "matched_rank",
                "value": s / n if n else np.nan,
                "n_pairs": int(n),
            }
            for (head, rc), (s, n) in sorted(edge_sums.items())
        ]
        out["edges"] = pd.DataFrame(rows, columns=["head", "range_class", "metric", "value", "n_pairs"])
    return out


def _influence(module, batch, queries, prediction_task, n_projections, directions, model_heads) -> dict:
    """``{head: [Q, N]}``: each query's estimated ``||J_ij||_F^2`` for every cell ``j``, in batch order.

    One forward pass, then ``n_projections`` backward passes per query and head. A query a head
    did not predict (``pad_batch`` left it out of a long graph) gets a NaN row.
    """
    with torch.enable_grad(), _differentiable_inputs(module, batch) as captured:
        out = module._common_step(batch, prediction_task, "node")
        heads, _, _ = model_heads(module, out)
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(batch.num_nodes, device=batch.x.device)
        inputs = captured["inputs"]
        position = torch.full((batch.num_nodes,), -1, dtype=torch.long)
        position[idx.cpu()] = torch.arange(len(idx))
        result = {}
        for name, pred in heads.items():
            if not pred.requires_grad:
                warnings.warn(
                    f"receptive_field: the {name} head does not depend on the inputs.", RuntimeWarning, stacklevel=3
                )
                result[name] = torch.zeros(len(queries), batch.num_nodes, dtype=torch.float64, device=inputs.device)
                continue
            rows = []
            for q in queries:
                r = int(position[int(q)])
                if r < 0:
                    rows.append(torch.full((batch.num_nodes,), float("nan"), dtype=torch.float64, device=inputs.device))
                    continue
                acc = torch.zeros(batch.num_nodes, dtype=torch.float64, device=inputs.device)
                for _ in range(n_projections):
                    w = torch.randn(pred.shape[1], generator=directions).to(device=pred.device, dtype=pred.dtype)
                    (grad,) = torch.autograd.grad((pred[r] * w).sum(), inputs, retain_graph=True)
                    acc += grad.double().pow(2).sum(1)
                rows.append(acc / n_projections)
            result[name] = torch.stack(rows)
        del out, heads
    return result


def _program_influence(module, batch, queries, gene_masks, prediction_task, n_projections, directions, model_heads):
    """``{head: [Q, N]}``: each query's estimated influence of every cell, on the query's genes ``gene_masks`` only.

    As :func:`_influence`, with the random direction of query ``q`` zero outside ``gene_masks[q]``
    -- the response genes of the programme it is read for -- and, when the inputs are the
    expression itself (``[N, G]``), a cell counted through its visible entries only: a hidden
    entry is a constant the model was handed, not information.
    """
    hidden = getattr(batch, "gene_mask", None)
    with torch.enable_grad(), _differentiable_inputs(module, batch) as captured:
        out = module._common_step(batch, prediction_task, "node")
        heads, _, _ = model_heads(module, out)
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(batch.num_nodes, device=batch.x.device)
        inputs = captured["inputs"]
        visible = None
        if hidden is not None and inputs.shape == hidden.shape:
            visible = (~hidden.bool()).double()
        position = torch.full((batch.num_nodes,), -1, dtype=torch.long)
        position[idx.cpu()] = torch.arange(len(idx))
        result = {}
        for name, pred in heads.items():
            rows = []
            for q, genes in zip(queries, gene_masks, strict=True):
                r = int(position[int(q)])
                if r < 0 or not pred.requires_grad:
                    rows.append(torch.full((batch.num_nodes,), float("nan"), dtype=torch.float64, device=inputs.device))
                    continue
                acc = torch.zeros(batch.num_nodes, dtype=torch.float64, device=inputs.device)
                for _ in range(n_projections):
                    w = torch.randn(pred.shape[1], generator=directions).to(device=pred.device, dtype=pred.dtype)
                    w = w * torch.as_tensor(genes, device=pred.device, dtype=pred.dtype)
                    (grad,) = torch.autograd.grad((pred[r] * w).sum(), inputs, retain_graph=True)
                    squared = grad.double().pow(2)
                    acc += (squared * visible).sum(1) if visible is not None else squared.sum(1)
                rows.append(acc / n_projections)
            result[name] = torch.stack(rows) if rows else None
        del out, heads
    return result


def response_influence(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    spatial_unit_um: float,
    edges: pd.DataFrame,
    edge_genes,
    queries_per_program: int = 16,
    n_projections: int = 2,
    tolerance_um: float = 10.0,
    distance_bins_um: Sequence[float] = (),
    seed: int = 0,
    max_graphs: int = 0,
) -> dict[str, pd.DataFrame]:
    """Does a receiver's predicted response depend on its planted sender? Read under a gene-set mask.

    The graphs carry a ``gene_mask`` hiding a programme's response genes -- in every cell, say
    (``masked_probing.draw_masks(kind="gene_set")``) -- so a response can only come from what
    drives it. For each planted programme (``edges['program']``), up to ``queries_per_program``
    receivers whose response genes ``edge_genes(program)`` are all hidden are sampled per graph;
    the influence of every cell on the receiver's predicted response genes is estimated by gradient
    (:func:`_program_influence`), and the planted sender is ranked among the cells about as far
    from the receiver (:func:`~interscale.evaluation._spatial_weights.matched_ranks`): 0.5 is a
    prediction blind to the sender but not to distance, 1 one that reads the sender above every
    cell as far away. A model that predicts responses from their drivers ranks senders high; a
    smoother of positions, or of the cell's own genes, does not.

    Parameters
    ----------
    edges
        Planted ``sender``, ``receiver``, ``range_class``, ``program`` pairs by position in the
        AnnData the graphs were built from.
    edge_genes
        ``program -> [columns]``: the response genes a programme's receivers express.
    distance_bins_um
        Bins for the influence profile of the sampled receivers by distance (lift over uniform).
    seed
        Seeds the samples and the directions; no global generator is touched.

    Returns
    -------
    dict
        ``"edges"``: ``head, range_class, program, metric, value, n_pairs`` -- ``matched_rank`` per
        programme, and per range class over its programmes (``program`` ``"any"``). ``"profile"``
        (with ``distance_bins_um``): ``head, range_class, distance_um, lift, n_queries``.
    """
    from interscale.evaluation.masked_probing import _model_heads

    if "regression" not in prediction_task or prediction_level != "node":
        raise ValueError("response_influence reads reconstructed expression; it needs regression at node level.")
    graphs = list(data_list)
    if max_graphs > 0:
        graphs = graphs[:max_graphs]
    unit = float(spatial_unit_um)
    rng = np.random.default_rng(seed)
    directions = torch.Generator().manual_seed(int(seed))
    sums: dict[tuple, np.ndarray] = {}
    profile: dict[tuple, np.ndarray] = {}
    n_bins = max(len(distance_bins_um) - 1, 0)

    was_training = module.training
    module.eval()
    loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
    try:
        with _rng_untouched(device), _frozen_parameters(module):
            for batch in loader:
                batch = batch.to(device)
                if getattr(batch, "pos", None) is None or getattr(batch, "gene_mask", None) is None:
                    raise ValueError("response_influence needs graphs with `pos` and a `gene_mask`.")
                hidden = batch.gene_mask.bool()
                pairs = graph_edges(edges, batch.obs_names)
                if not len(pairs):
                    continue
                # One query per (receiver, programme): the receiver's prediction read on that programme's genes.
                queries, masks, query_class, query_of, picked = [], [], [], {}, []
                for program, sub in pairs.groupby("program"):
                    columns = list(edge_genes(str(program)))
                    if not columns:
                        continue
                    covered = hidden[:, columns].all(dim=1).cpu().numpy()
                    receivers = np.unique(sub["receiver"].to_numpy())
                    receivers = _sample(receivers[covered[receivers]], queries_per_program, rng)
                    genes = np.zeros(batch.x.shape[1], dtype=bool)
                    genes[columns] = True
                    for receiver in receivers:
                        query_of[(int(receiver), str(program))] = len(queries)
                        queries.append(int(receiver))
                        masks.append(genes)
                        query_class.append(str(sub["range_class"].iloc[0]))
                    picked.append(sub[np.isin(sub["receiver"].to_numpy(), receivers)])
                if not queries:
                    continue
                influence = _program_influence(
                    module, batch, queries, masks, prediction_task, n_projections, directions, _model_heads
                )
                where = batch.pos.double() * unit
                q = torch.as_tensor(queries, device=where.device)
                dist = torch.cdist(where[q], where, compute_mode="donot_use_mm_for_euclid_dist")
                keep = torch.ones_like(dist, dtype=torch.bool)
                keep[torch.arange(len(q), device=q.device), q] = False  # a cell is not its own comparison
                pairs = pd.concat(picked, ignore_index=True)
                receivers = pairs["receiver"].to_numpy().astype(np.int64)
                program_names = pairs["program"].astype(str).to_numpy()
                classes = pairs["range_class"].astype(str).to_numpy()
                rows = torch.as_tensor(
                    [query_of[(int(r), p)] for r, p in zip(receivers, program_names, strict=True)], device=q.device
                )
                senders = torch.as_tensor(pairs["sender"].to_numpy().astype(np.int64), device=q.device)
                # A receiver's other senders in the same programme are no comparison for any of them.
                exclude = torch.zeros(len(pairs), batch.num_nodes, dtype=torch.bool, device=q.device)
                same = (
                    pairs.groupby(["receiver", "program"])["sender"].apply(lambda v: np.unique(v.to_numpy())).to_dict()
                )
                for k, key in enumerate(zip(receivers, program_names, strict=True)):
                    exclude[k, torch.as_tensor(same[(int(key[0]), key[1])], device=q.device)] = True
                query_class = np.asarray(query_class)
                for head, weights in influence.items():
                    if weights is None:
                        continue
                    ranks = matched_ranks(weights, dist, keep, rows, senders, exclude, tolerance_um).cpu().numpy()
                    groups = [
                        (c, p, (classes == c) & (program_names == p))
                        for c, p in set(zip(classes, program_names, strict=True))
                    ]
                    groups += [(c, "any", classes == c) for c in np.unique(classes)]
                    for key_class, key_program, pick in groups:
                        ranked = pick & np.isfinite(ranks)
                        acc = sums.setdefault((head, str(key_class), str(key_program)), np.zeros(2))
                        acc += [ranks[ranked].sum(), ranked.sum()]
                    if n_bins:
                        weight_in, uniform_in = distance_profile(weights, dist, keep, distance_bins_um)
                        defined = torch.isfinite(weight_in).all(1).cpu().numpy()
                        weight_in, uniform_in = weight_in.cpu().numpy(), uniform_in.cpu().numpy()
                        for c in np.unique(query_class):
                            pick = (query_class == c) & defined
                            if pick.any():
                                acc = profile.setdefault((head, str(c)), np.zeros((3, n_bins)))
                                acc += [weight_in[pick].sum(0), uniform_in[pick].sum(0), np.full(n_bins, pick.sum())]
    finally:
        module.train(was_training)

    rows = [
        {
            "head": head,
            "range_class": rc,
            "program": prog,
            "metric": "matched_rank",
            "value": s / n if n else np.nan,
            "n_pairs": int(n),
        }
        for (head, rc, prog), (s, n) in sorted(sums.items())
    ]
    out = {"edges": pd.DataFrame(rows, columns=["head", "range_class", "program", "metric", "value", "n_pairs"])}
    if n_bins:
        labels = bin_labels(distance_bins_um)
        rows = []
        for (head, rc), (weight, uniform, count) in sorted(profile.items()):
            for b, label in enumerate(labels):
                lift = weight[b] / uniform[b] if uniform[b] > 0 else np.nan
                rows.append(
                    {"head": head, "range_class": rc, "distance_um": label, "lift": lift, "n_queries": int(count[b])}
                )
        out["profile"] = pd.DataFrame(rows, columns=["head", "range_class", "distance_um", "lift", "n_queries"])
    return out
