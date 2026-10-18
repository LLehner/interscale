"""Where the cells of a patch draw their predictions from, and how much of that depends on what is there.

A coarse grid of square patches lies over each slide. For the cells of a source patch that the model
is asked about (those with a hidden entry), two measures say where their predictions come from, as a
weight over the input cells:

* ``influence`` -- gradient flow: the squared Jacobian norm of the cells' predicted hidden entries
  with respect to each input cell, counted through that cell's visible entries only (a hidden entry
  is a constant the model was handed). Random probes drawn independently for every query cell and
  hidden entry give ``E ||sum_i J_i^T w_i||^2 = sum_i ||J_i||^2`` (Hutchinson), so one backward pass
  per probe yields a whole patch's summed influence. Covers the whole model, GCN included.
* ``rollout`` -- attention flow: every layer's head-mean attention with the residual path,
  ``(A + I) / 2``, composed over the layers (Abnar & Zuidema 2020): how much of a query's final token
  traces back to each input token. One layer's raw attention is not the flow -- the values, the
  residual stream and the next layer all change what reaches the output. Covers the transformer only
  (behind a GCN its inputs are the GCN's tokens). Mass on CLS and on wholly hidden cells (a mask fill
  carries no information) is left out and each query's row renormalised.

Per source patch, the share it draws from each target patch; per label pair (query label x key
label), the same. Each is read on the slide as it is and on ``permutations`` copies with the visible
cells' expression permuted among them: positions, masks, densities and positional encodings stay,
the arrangement of what is expressed where goes. Everything that is geometry -- distance, masking,
the slide's edge, the encodings -- survives the permutation, so observed against the permutations is
what depends on the tissue's arrangement: log2 ratio, and z against the permutations' spread. The
same comparison of each permutation against the others gives the floor a measurement noise (the
influence's random probes) alone would reach.

Consumes no global RNG and leaves the module as it was, like the other masked-pass readouts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation.online_probes import _rng_untouched
from interscale.tl.geome_utils import label_codes

#: The measures :func:`patch_flow` reads.
KINDS = ("influence", "rollout")


def patch_index(pos_um: np.ndarray, patch_um: float) -> tuple[np.ndarray, np.ndarray]:
    """``(patch [N], centres [P, 2])``: each cell's square patch of side ``patch_um`` from the graph's lower corner.

    Patches are numbered row-major over the bounding box; every patch of the box gets a centre,
    empty ones included, so a map can be drawn on the full grid.
    """
    low = pos_um.min(0)
    cell = np.floor((pos_um - low) / float(patch_um)).astype(np.int64)
    nx, ny = cell.max(0) + 1
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centres = low + (np.stack([gx.ravel(), gy.ravel()], 1) + 0.5) * float(patch_um)
    return cell[:, 0] * ny + cell[:, 1], centres


def _scored(batch, mask_strategy: str) -> torch.Tensor:
    """``[N, G]``: the entries the model is asked about -- ``gene_mask`` under gene masking, else whole hidden rows."""
    gene_mask = getattr(batch, "gene_mask", None) if mask_strategy == "gene" else None
    if gene_mask is not None:
        return gene_mask.bool()
    return batch.mask.bool()[:, None].expand_as(batch.x)


def _permuted(batch, scored: torch.Tensor, generator: torch.Generator, label_key: str | None):
    """``batch`` with the rows of every cell that has a visible entry permuted among those cells.

    Expression, precomputed embeddings and the label move together, as one cell's profile; the masks
    and positions stay where they were. Returns the copy and the permutation (``perm[k]`` is the cell
    whose profile position ``k`` now holds).
    """
    copy = batch.clone()
    open_cells = (~scored.all(dim=1)).nonzero().squeeze(1)
    order = torch.randperm(len(open_cells), generator=generator).to(open_cells.device)
    perm = torch.arange(batch.num_nodes, device=open_cells.device)
    perm[open_cells] = open_cells[order]
    copy.x = batch.x[perm]
    for name in ("embeddings", label_key):
        value = getattr(batch, name, None) if name else None
        if isinstance(value, torch.Tensor) and value.shape[0] == batch.num_nodes:
            setattr(copy, name, value[perm])
    return copy, perm


def _rollout(module, batch, prediction_task: str, prediction_level: str) -> torch.Tensor | None:
    """``[N, N]`` in batch order: each cell's final token traced back to the input cells' tokens, CLS dropped.

    None when the module has no transformer or the backend forms no maps.
    """
    transformer = getattr(module, "global_module", module)
    if not hasattr(transformer, "recording_attention"):
        return None
    with torch.no_grad(), transformer.recording_attention() as record:
        module._common_step(batch, prediction_task, prediction_level)
    entry = record[0]
    maps = entry.get("maps")
    if maps is None:
        return None
    dev = maps.device
    padding = entry["src_padding_mask"][0].bool().to(dev)
    spots = (~padding).nonzero().squeeze(1)
    nodes = torch.as_tensor(entry["index_nodes"][0], dtype=torch.long, device=dev)
    length = maps.shape[-1]
    eye = torch.eye(length, dtype=torch.float64, device=dev)
    flow = eye
    for layer in range(maps.shape[0]):
        step = 0.5 * maps[layer].mean(0).double() + 0.5 * eye
        flow = step @ flow
    full = torch.zeros(batch.num_nodes, batch.num_nodes, dtype=torch.float64, device=dev)
    full[nodes[:, None], nodes[None, :]] = flow[spots][:, spots]
    return full


def _model_head(module, out, head: str | None):
    from interscale.evaluation.masked_probing import _model_heads

    heads, _, _ = _model_heads(module, out)
    if head is not None and head in heads:
        return heads[head]
    return heads["global"] if "global" in heads else next(iter(heads.values()))


def _influence(module, batch, groups, scored, prediction_task, n_projections, generator, head) -> torch.Tensor:
    """``[len(groups), N]``: per group of query cells, the summed influence of every *other* cell on their predicted hidden entries.

    Influence goes through a cell's visible entries when the inputs are the expression; with token
    inputs (a GlobalModel's ``gex_embedding``) a wholly hidden cell's token is the fill and reads 0.

    A query's influence on itself -- its own visible genes -- is left out: the flow is about context.
    One probe sums over the group's queries, so it cannot tell a cell's influence on itself from its
    influence on a group-mate. When any query has a visible entry, each probe therefore takes a random
    half ``H`` of the group as queries and reads only the cells outside ``H``; a key inside the group
    is outside ``H`` less often than one outside it, by ``(n - h) / (n - 1)``, and is scaled back up
    by that factor, so the expectation is ``h / n`` times ``sum_{i != j} ||J_ij||^2`` for every key.
    When every query is wholly hidden its own input is a constant anyway, and all queries probe at once.
    """
    from interscale.evaluation.receptive_field import _differentiable_inputs, _frozen_parameters

    n = batch.num_nodes
    out_rows = torch.zeros(len(groups), n, dtype=torch.float64, device=batch.x.device)
    with _frozen_parameters(module), torch.enable_grad(), _differentiable_inputs(module, batch) as captured:
        out = module._common_step(batch, prediction_task, "node")
        pred = _model_head(module, out, head)
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(n, device=batch.x.device)
        inputs = captured["inputs"]
        if not pred.requires_grad:
            return out_rows
        position = torch.full((n,), -1, dtype=torch.long, device=pred.device)
        position[idx] = torch.arange(len(idx), device=pred.device)
        if inputs.shape == scored.shape:
            through = (~scored).double()
        else:
            through = (~scored.all(dim=1)).double()[:, None].expand(n, inputs.shape[1])
        own_input = through.sum(1) > 0  # cells whose own input carries something
        for g, cells in enumerate(groups):
            cells = torch.as_tensor(cells, dtype=torch.long, device=pred.device)
            rows = position[cells]
            cells, rows = cells[rows >= 0], rows[rows >= 0]
            n_group = len(rows)
            if not n_group:
                continue
            split = bool(own_input[cells].any())
            for _ in range(n_projections):
                if split:
                    order = torch.randperm(n_group, generator=generator).to(pred.device)
                    half = order[: (n_group + 1) // 2]
                else:
                    half = torch.arange(n_group, device=pred.device)
                probe_cells, probe_rows = cells[half], rows[half]
                weight = scored[probe_cells].to(pred.dtype)
                w = torch.randn(weight.shape, generator=generator).to(device=pred.device, dtype=pred.dtype) * weight
                (grad,) = torch.autograd.grad((pred[probe_rows] * w).sum(), inputs, retain_graph=True)
                contribution = (grad.double().pow(2) * through).sum(1)
                if split:
                    contribution[probe_cells] = 0.0
                    if n_group > 1:
                        inside = cells[~torch.isin(cells, probe_cells)]
                        contribution[inside] *= (n_group - 1) / (n_group - len(half))
                out_rows[g] += contribution
        del out, pred
    return out_rows / max(1, n_projections)


class _Sums:
    """Raw flows summed over draws: ``(graph, kind, variant) -> [sources, targets]`` and ``(kind, variant) -> [labels, labels]``."""

    def __init__(self):
        self.patches: dict[tuple, np.ndarray] = {}
        self.queries: dict[tuple, np.ndarray] = {}
        self.labels: dict[tuple, np.ndarray] = {}
        self.label_queries: dict[tuple, np.ndarray] = {}

    @staticmethod
    def _add(store, key, value):
        if key in store:
            store[key] += value
        else:
            store[key] = value.copy()


def _onehot(codes: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((len(codes), width))
    out[np.arange(len(codes)), codes] = 1.0
    return out


def patch_flow(
    module,
    draws: Sequence[Sequence],
    *,
    prediction_task: str,
    prediction_level: str,
    mask_strategy: str,
    device,
    spatial_unit_um: float,
    patch_um: float,
    kinds: Sequence[str] = KINDS,
    label_key: str | None = None,
    permutations: int = 4,
    n_projections: int = 2,
    head: str | None = None,
    min_queries: int = 3,
    seed: int = 0,
    max_graphs: int = 0,
) -> dict[str, pd.DataFrame]:
    """Where each source patch's cells draw their predictions from, against the expression-permuted slide.

    Parameters
    ----------
    module
        Any model's module; ``rollout`` needs a transformer, ``influence`` works for every model.
    draws
        One or more lists of the same graphs, each list carrying its own masks (e.g. disc draws with
        different seeds, so that most patches hold hidden cells in some draw). Graph ``g`` must be the
        same slide in every draw; flows add up over the draws.
    mask_strategy
        ``cfg.dataset.mask_strategy``: which attribute says what is hidden.
    spatial_unit_um
        µm per unit of ``pos``.
    patch_um
        Side of the square patches, µm.
    kinds
        From :data:`KINDS`.
    label_key
        An annotation on the graphs (one-hot) for the flow between labels, or None.
    permutations
        Copies of each slide with the visible cells' expression permuted, for the null (>= 2 for its
        spread and floor).
    n_projections
        Random probes per patch for ``influence``.
    head
        The decoder head whose predictions ``influence`` reads (``global`` by default where there is one).
    min_queries
        Source patches with fewer query cells over every draw are left out of the summary.
    seed
        Seeds the permutations and the probes; no global generator is touched.

    Returns
    -------
    dict
        ``"patches"``: ``graph, kind, source, target, observed, null_mean, null_std, n_queries`` --
        shares of each source patch's flow (summing to 1 over the targets). ``"centres"``:
        ``graph, patch, x_um, y_um, n_cells``. ``"summary"``: ``kind, metric, value`` (see
        :func:`flow_summary`). ``"labels"`` (with ``label_key``): ``kind, query_label, key_label,
        observed, null_mean, null_std``, shares of each query label's flow.
    """
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"kinds must come from {KINDS}, got {unknown}.")
    draws = [list(d)[:max_graphs] if max_graphs > 0 else list(d) for d in draws]
    if not draws or any(len(d) != len(draws[0]) for d in draws):
        raise ValueError("every draw must hold the same graphs.")
    generator = torch.Generator().manual_seed(int(seed))
    sums = _Sums()
    centres_rows: list[dict] = []
    n_labels = None

    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device):
            for d, graphs in enumerate(draws):
                loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
                for g, batch in enumerate(loader):
                    batch = batch.to(device)
                    if getattr(batch, "pos", None) is None:
                        raise ValueError("patch_flow reads positions, but a graph carries no `pos`.")
                    where = batch.pos.double().cpu().numpy() * float(spatial_unit_um)
                    patch, centres = patch_index(where, patch_um)
                    if d == 0:
                        counts = np.bincount(patch, minlength=len(centres))
                        centres_rows += [
                            {"graph": g, "patch": p, "x_um": c[0], "y_um": c[1], "n_cells": int(counts[p])}
                            for p, c in enumerate(centres)
                        ]
                    to_patch = _onehot(patch, len(centres))
                    scored = _scored(batch, mask_strategy)
                    queries = scored.any(dim=1).cpu().numpy()
                    sources = np.unique(patch[queries])
                    variants = [("observed", batch, None)]
                    for k in range(permutations):
                        copy, perm = _permuted(batch, scored, generator, label_key)
                        variants.append((f"null{k}", copy, perm))
                    for variant, data, _ in variants:
                        labels = None
                        if label_key is not None:
                            labels = label_codes(data, label_key).cpu().numpy()
                            n_labels = int(getattr(data, label_key).shape[1])
                        for kind in kinds:
                            if kind == "influence":
                                groups = [np.flatnonzero(queries & (patch == s)) for s in sources]
                                flow = (
                                    _influence(
                                        module, data, groups, scored, prediction_task, n_projections, generator, head
                                    )
                                    .cpu()
                                    .numpy()
                                )
                                per_source = flow @ to_patch
                                label_flow = None
                                if labels is not None:
                                    label_groups = [np.flatnonzero(queries & (labels == a)) for a in range(n_labels)]
                                    raw = (
                                        _influence(
                                            module,
                                            data,
                                            label_groups,
                                            scored,
                                            prediction_task,
                                            n_projections,
                                            generator,
                                            head,
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    label_flow = raw @ _onehot(labels, n_labels)
                            else:
                                rows = _rollout(module, data, prediction_task, prediction_level)
                                if rows is None:
                                    continue
                                informative = (~scored.all(dim=1)).double()
                                rows = rows * informative[None, :]
                                rows.fill_diagonal_(0.0)  # a query's own input: the flow is about context
                                rows = (rows / rows.sum(1, keepdim=True).clamp_min(1e-300)).cpu().numpy()
                                rows[~queries] = 0.0
                                per_source = np.stack([rows[queries & (patch == s)].sum(0) for s in sources]) @ to_patch
                                label_flow = None
                                if labels is not None:
                                    # Rows of non-query cells are 0: query label x key label, summed.
                                    onehot = _onehot(labels, n_labels)
                                    label_flow = onehot.T @ (rows @ onehot)
                            full = np.zeros((len(centres), len(centres)))
                            full[sources] = per_source
                            sums._add(sums.patches, (g, kind, variant), full)
                            n_queries = np.bincount(patch[queries], minlength=len(centres)).astype(float)
                            sums._add(sums.queries, (g, kind, variant), n_queries)
                            if label_flow is not None:
                                sums._add(sums.labels, (kind, variant), label_flow)
                                sums._add(
                                    sums.label_queries,
                                    (kind, variant),
                                    np.bincount(labels[queries], minlength=n_labels).astype(float),
                                )
    finally:
        module.train(was_training)
    return _tables(sums, centres_rows, permutations, min_queries, label_key is not None)


def _shares(raw: np.ndarray) -> np.ndarray:
    total = raw.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, raw / np.where(total > 0, total, 1.0), np.nan)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    keep = p > 0
    return float((p[keep] * np.log(p[keep] / np.maximum(q[keep], 1e-12))).sum())


def _tables(
    sums: _Sums, centres_rows, permutations: int, min_queries: int, with_labels: bool
) -> dict[str, pd.DataFrame]:
    centres = pd.DataFrame(centres_rows)
    patch_rows, summary_rows = [], []
    keys = sorted({(g, kind) for g, kind, _ in sums.patches})
    for g, kind in keys:
        observed = _shares(sums.patches[(g, kind, "observed")])
        nulls = [_shares(sums.patches[(g, kind, f"null{k}")]) for k in range(permutations)]
        n_queries = sums.queries[(g, kind, "observed")]
        null_mean = np.nanmean(np.stack(nulls), axis=0) if nulls else np.full_like(observed, np.nan)
        null_std = np.nanstd(np.stack(nulls), axis=0, ddof=1) if len(nulls) > 1 else np.full_like(observed, np.nan)
        sources, targets = np.nonzero(np.nan_to_num(observed) + np.nan_to_num(null_mean))
        for s, t in zip(sources, targets, strict=True):
            patch_rows.append(
                {
                    "graph": g,
                    "kind": kind,
                    "source": int(s),
                    "target": int(t),
                    "observed": observed[s, t],
                    "null_mean": null_mean[s, t],
                    "null_std": null_std[s, t],
                    "n_queries": int(n_queries[s]),
                }
            )
        xy = centres[centres["graph"] == g].sort_values("patch")[["x_um", "y_um"]].to_numpy()
        summary_rows.append({"graph": g, "kind": kind, **flow_summary(observed, nulls, n_queries, xy, min_queries)})
    summary = pd.DataFrame(summary_rows)
    columns = ["graph", "kind", "source", "target", "observed", "null_mean", "null_std", "n_queries"]
    out = {"patches": pd.DataFrame(patch_rows, columns=columns), "centres": centres}
    if len(summary):
        # Over the graphs, each weighted by its source queries.
        weight = summary.pop("n_sources_queries")
        metrics = [c for c in summary.columns if c not in ("graph", "kind")]
        rows = []
        for kind, part in summary.assign(weight=weight).groupby("kind"):
            for metric in metrics:
                values, w = part[metric].to_numpy(float), part["weight"].to_numpy(float)
                keep = np.isfinite(values) & (w > 0)
                value = float((values[keep] * w[keep]).sum() / w[keep].sum()) if keep.any() else np.nan
                rows.append({"kind": kind, "metric": metric, "value": value})
        out["summary"] = pd.DataFrame(rows, columns=["kind", "metric", "value"])
    if with_labels and sums.labels:
        label_rows = []
        for kind in sorted({kind for kind, _ in sums.labels}):
            observed = _shares(sums.labels[(kind, "observed")])
            nulls = [
                _shares(sums.labels[(kind, f"null{k}")])
                for k in range(permutations)
                if (kind, f"null{k}") in sums.labels
            ]
            null_mean = np.nanmean(np.stack(nulls), axis=0) if nulls else np.full_like(observed, np.nan)
            null_std = np.nanstd(np.stack(nulls), axis=0, ddof=1) if len(nulls) > 1 else np.full_like(observed, np.nan)
            queries = sums.label_queries[(kind, "observed")]
            for a in range(observed.shape[0]):
                if queries[a] == 0:
                    continue
                for b in range(observed.shape[1]):
                    label_rows.append(
                        {
                            "kind": kind,
                            "query_label": a,
                            "key_label": b,
                            "observed": observed[a, b],
                            "null_mean": null_mean[a, b],
                            "null_std": null_std[a, b],
                        }
                    )
        out["labels"] = pd.DataFrame(label_rows)
    return out


def _centre_of_mass(shares: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.nan_to_num(shares) @ xy


def flow_summary(
    observed: np.ndarray, nulls: list[np.ndarray], n_queries: np.ndarray, xy: np.ndarray, min_queries: int
) -> dict:
    """One graph's flow in numbers, over its source patches (each weighted by its query cells).

    * ``reach_um`` / ``reach_null_um`` -- the share-weighted distance from a source patch's centre to
      where it draws from, observed and permuted;
    * ``self_share`` / ``self_share_null`` -- the share drawn from the source patch itself;
    * ``kl_null`` -- KL of the observed target distribution from the permutations' mean: how much the
      tissue's arrangement changes where a patch draws from (0: geometry alone), against
      ``kl_null_floor``, each permutation against the mean of the others (the noise level);
    * ``shift_um`` -- the distance between the observed and the permuted centre of mass, against
      ``shift_floor_um``;
    * ``above_null`` -- the share of the observed flow on targets more than 2 null standard
      deviations above the permutations' mean.
    """
    sources = np.flatnonzero((n_queries >= min_queries) & np.isfinite(observed).all(1))
    out = {"n_sources_queries": float(n_queries[sources].sum()), "n_sources": float(len(sources))}
    if not len(sources) or not nulls:
        return out
    w = n_queries[sources] / n_queries[sources].sum()
    stack = np.stack(nulls)
    null_mean = np.nanmean(stack, axis=0)
    distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))

    def mean(values):
        return (
            float(np.nansum(np.asarray(values) * w) / w[np.isfinite(values)].sum())
            if np.isfinite(values).any()
            else np.nan
        )

    out["reach_um"] = mean([np.nansum(observed[s] * distance[s]) for s in sources])
    out["reach_null_um"] = mean([np.nansum(null_mean[s] * distance[s]) for s in sources])
    out["self_share"] = mean([observed[s, s] for s in sources])
    out["self_share_null"] = mean([null_mean[s, s] for s in sources])
    out["kl_null"] = mean([_kl(observed[s], null_mean[s]) for s in sources])
    com_obs, com_null = _centre_of_mass(observed, xy), _centre_of_mass(null_mean, xy)
    out["shift_um"] = mean([np.linalg.norm(com_obs[s] - com_null[s]) for s in sources])
    if len(nulls) > 1:
        floors, shifts = [], []
        for k in range(len(nulls)):
            others = np.nanmean(np.delete(stack, k, axis=0), axis=0)
            floors.append([_kl(nulls[k][s], others[s]) for s in sources])
            com_k, com_o = _centre_of_mass(nulls[k], xy), _centre_of_mass(others, xy)
            shifts.append([np.linalg.norm(com_k[s] - com_o[s]) for s in sources])
        out["kl_null_floor"] = mean(np.mean(floors, axis=0))
        out["shift_floor_um"] = mean(np.mean(shifts, axis=0))
        null_std = np.nanstd(stack, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (observed - null_mean) / null_std
        out["above_null"] = mean([np.nansum(np.where(z[s] > 2, observed[s], 0.0)) for s in sources])
    return out


# --------------------------------------------------------------------------- figures

#: Chart tokens of the validated reference palette (dataviz skill): surface, ink, grid; the blue ramp for
#: shares, blue <-> red with a neutral grey midpoint for ratios.
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#9ec5f4", "#f0efec", "#f2a3a2", "#d03b3b", "#6e1817"]


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap

    return (
        LinearSegmentedColormap.from_list("flow_share", SEQUENTIAL),
        LinearSegmentedColormap.from_list("flow_ratio", DIVERGING),
    )


def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_aspect("equal")


def regular_sources(patches: pd.DataFrame, centres: pd.DataFrame, graph: int, kind: str, n: int = 6) -> list[int]:
    """``n`` source patches spread evenly over the tissue: the nearest source to each point of a regular lattice."""
    sources = np.unique(patches[(patches["graph"] == graph) & (patches["kind"] == kind)]["source"])
    if not len(sources):
        return []
    grid = centres[centres["graph"] == graph].set_index("patch")
    xy = grid.loc[sources, ["x_um", "y_um"]].to_numpy()
    side = int(np.ceil(np.sqrt(n)))
    lo, hi = xy.min(0), xy.max(0)
    fx, fy = np.meshgrid(np.linspace(0.15, 0.85, side), np.linspace(0.15, 0.85, side))
    chosen = []
    for point in lo + np.stack([fx.ravel(), fy.ravel()], 1) * (hi - lo):
        order = np.argsort(np.linalg.norm(xy - point, axis=1))
        pick = next((int(sources[o]) for o in order if int(sources[o]) not in chosen), None)
        if pick is not None:
            chosen.append(pick)
    return chosen[:n]


def _tissue(ax, grid: pd.DataFrame, patch_um: float, values=None, cmap=None, norm=None):
    from matplotlib.patches import Rectangle

    for patch, row in grid.iterrows():
        if row["n_cells"] == 0:
            continue
        value = None if values is None else values.get(patch, np.nan)
        colour = "#f0efec" if value is None or not np.isfinite(value) else cmap(norm(value))
        ax.add_patch(
            Rectangle((row["x_um"] - patch_um / 2, row["y_um"] - patch_um / 2), patch_um, patch_um, color=colour, lw=0)
        )
    ax.set_xlim(grid["x_um"].min() - patch_um, grid["x_um"].max() + patch_um)
    ax.set_ylim(grid["y_um"].min() - patch_um, grid["y_um"].max() + patch_um)


def plot_flow_field(
    patches: pd.DataFrame, centres: pd.DataFrame, *, graph: int, kind: str, patch_um: float, title: str = ""
):
    """Arrows from each source patch: to the centre of mass of what it draws from, and of what it draws beyond the null.

    Left: the observed flow's centre of mass -- for a model that draws from the whole slide every
    arrow points near the slide's middle, for a local one they stay short. Right: the centre of mass
    of the excess over the expression-permuted slide (observed minus permuted, positive part), opaque
    in proportion to the excess share -- where the arrangement of the tissue pulls a patch.
    """
    import matplotlib.pyplot as plt

    grid = centres[centres["graph"] == graph].set_index("patch")
    part = patches[(patches["graph"] == graph) & (patches["kind"] == kind)]
    xy = grid[["x_um", "y_um"]]
    arrows = []
    for source, sub in part.groupby("source"):
        target_xy = xy.loc[sub["target"]].to_numpy()
        obs = np.nan_to_num(sub["observed"].to_numpy())
        excess = np.clip(obs - np.nan_to_num(sub["null_mean"].to_numpy()), 0.0, None)
        start = xy.loc[source].to_numpy()
        to_obs = obs @ target_xy / obs.sum() if obs.sum() > 0 else start
        to_excess = excess @ target_xy / excess.sum() if excess.sum() > 0 else start
        arrows.append((start, to_obs, to_excess, float(excess.sum())))
    largest = max((a[3] for a in arrows), default=0.0) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), facecolor=SURFACE)
    panels = ((1, "centre of mass of what the patch draws from"), (2, "... of what it draws beyond the permuted slide"))
    for ax, (which, name) in zip(axes, panels, strict=True):
        _tissue(ax, grid, patch_um)
        for start, to_obs, to_excess, excess in arrows:
            end = to_obs if which == 1 else to_excess
            alpha = 1.0 if which == 1 else 0.15 + 0.85 * excess / largest
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": INK_SECONDARY,
                    "lw": 0.9,
                    "alpha": alpha,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
            ax.plot(*start, "o", ms=2.5, color=INK_MUTED)
        _style(ax, name)
    fig.suptitle(title or f"{kind}: where each patch draws from", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def plot_source_maps(
    patches: pd.DataFrame,
    centres: pd.DataFrame,
    *,
    graph: int,
    kind: str,
    patch_um: float,
    sources: list[int],
    title: str = "",
):
    """Per source patch (outlined): the share drawn from each patch (top) and log2 observed / permuted (bottom)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    share_cmap, ratio_cmap = _cmaps()
    grid = centres[centres["graph"] == graph].set_index("patch")
    part = patches[(patches["graph"] == graph) & (patches["kind"] == kind)]
    fig, axes = plt.subplots(2, len(sources), figsize=(2.7 * len(sources) + 1.2, 5.6), squeeze=False, facecolor=SURFACE)
    positive = part["observed"][part["observed"] > 0]
    share_norm = LogNorm(
        vmin=max(float(positive.quantile(0.05)) if len(positive) else 1e-4, 1e-6),
        vmax=float(positive.max()) if len(positive) else 1.0,
    )
    ratio_norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    for column, source in enumerate(sources):
        sub = part[part["source"] == source].set_index("target")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2(sub["observed"] / sub["null_mean"]).replace([np.inf, -np.inf], np.nan).clip(-2, 2)
        for row, (values, cmap, norm) in enumerate(
            ((sub["observed"], share_cmap, share_norm), (ratio, ratio_cmap, ratio_norm))
        ):
            ax = axes[row, column]
            _tissue(ax, grid, patch_um, values.to_dict(), cmap, norm)
            x, y = grid.loc[source, ["x_um", "y_um"]]
            ax.add_patch(
                Rectangle((x - patch_um / 2, y - patch_um / 2), patch_um, patch_um, fill=False, ec=INK, lw=1.6)
            )
            _style(ax, f"source {source}" if row == 0 else "")
    for row, (cmap, norm, label) in enumerate(
        ((share_cmap, share_norm, "share drawn from the patch"), (ratio_cmap, ratio_norm, "log2 observed / permuted"))
    ):
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[row].tolist(), shrink=0.85)
        bar.set_label(label, color=INK_SECONDARY, fontsize=8)
        bar.ax.tick_params(colors=INK_MUTED, labelsize=7, length=0)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return fig


def plot_label_flow(labels: pd.DataFrame, *, kind: str, names: list[str] | None = None, title: str = ""):
    """Query label x key label: the share drawn (observed, permuted) and log2 observed / permuted, starred by z."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    share_cmap, ratio_cmap = _cmaps()
    part = labels[labels["kind"] == kind]
    observed = part.pivot(index="query_label", columns="key_label", values="observed")
    null = part.pivot(index="query_label", columns="key_label", values="null_mean")
    spread = part.pivot(index="query_label", columns="key_label", values="null_std")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log2(observed / null)
        z = (observed - null) / spread
    if names is not None:
        for table in (observed, null, ratio, z):
            table.index = [names[i] for i in table.index]
            table.columns = [names[i] for i in table.columns]
    vmax = float(np.nanmax([observed.to_numpy(), null.to_numpy()]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor=SURFACE)
    panels = [
        (observed, "observed share", share_cmap, plt.Normalize(0, vmax)),
        (null, "permuted expression", share_cmap, plt.Normalize(0, vmax)),
        (
            ratio.clip(-2, 2),
            "log2 observed / permuted (* |z|>1.96, ** >2.58, *** >3.29; read against the untrained twin)",
            ratio_cmap,
            TwoSlopeNorm(vcenter=0.0, vmin=-2.0, vmax=2.0),
        ),
    ]
    for i, (ax, (table, name, cmap, norm)) in enumerate(zip(axes, panels, strict=True)):
        image = ax.imshow(table.to_numpy(float), cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=60, ha="right", fontsize=7, color=INK_SECONDARY)
        ax.set_yticks(
            range(table.shape[0]), table.index if i == 0 else [""] * table.shape[0], fontsize=7, color=INK_SECONDARY
        )
        ax.set_xlabel("key label (drawn from)", fontsize=8, color=INK_SECONDARY)
        if i == 0:
            ax.set_ylabel("query label (drawing)", fontsize=8, color=INK_SECONDARY)
        ax.set_title(name, fontsize=9, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 2:
            stars = np.where(
                np.abs(z) > 3.29, "***", np.where(np.abs(z) > 2.58, "**", np.where(np.abs(z) > 1.96, "*", ""))
            )
            for (r, c), mark in np.ndenumerate(stars):
                if mark:
                    ax.text(c, r, mark, ha="center", va="center", fontsize=6, color=INK)
        bar = fig.colorbar(image, ax=ax, shrink=0.7)
        bar.ax.tick_params(colors=INK_MUTED, labelsize=7, length=0)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    return fig
