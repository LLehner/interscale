"""Where the attention of a masked pass flows on a slide: patch to patch, and cluster to cluster.

A coarse grid of square patches is laid over each slide. For every layer's head-mean attention map
-- and, optionally, for the receptive field of hidden cells' predictions (the gradient influence of
each visible cell, ``receptive_field``) -- the weight each query puts on each key is summed over the
queries of a source patch and the keys of a target patch. Two expectations are summed the same way:

* ``uniform`` -- the same mass spread evenly over the keys each query can attend to: a patch that
  only holds more cells receives more;
* ``distance`` -- the mass a distance kernel would put there: per graph and layer, the map's own
  average weight per key at each distance (``distance_bins_um``), spread over the query's open keys.
  It is what is left once "attends to what is near" is accounted for.

``observed / distance`` above 1 marks a target the queries of a patch look at more than its
distance explains -- the interesting part of a flow map, since attention near uniform or decaying
with distance is already known from the distance profiles. Per source patch the
Kullback-Leibler divergence of the observed target distribution from each expectation sums it up
(0: explained by it).

Queries are read in two groups, ``hidden`` (wholly hidden cells, which need context) and ``visible``.
With ``mask_seeds`` the discs are drawn several times so that most patches serve as hidden sources
in some draw; masks on the graphs are left as they were. The same is read by label (e.g. Leiden
clusters, an evaluation-only annotation the model never sees): query label x key label.

Consumes no global RNG and leaves the module as it was, like the other masked-pass readouts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation._spatial_weights import hidden_cells
from interscale.evaluation.online_probes import _rng_untouched
from interscale.tl.geome_utils import label_codes

#: The query groups.
GROUPS = ("hidden", "visible")
#: The row name of the receptive field among the layers.
INFLUENCE = "influence"


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


def _expected_by_distance(weights: torch.Tensor, open_keys: torch.Tensor, bins: torch.Tensor, n_bins: int):
    """``[Q, K]``: each query's open keys weighted by the map's mean weight per key at their distance bin."""
    flat_bins = bins[open_keys]
    total = torch.zeros(n_bins, dtype=weights.dtype, device=weights.device).index_add_(0, flat_bins, weights[open_keys])
    count = torch.bincount(flat_bins, minlength=n_bins).to(weights.dtype)
    per_key = total / count.clamp_min(1.0)
    expected = torch.where(open_keys, per_key[bins], torch.zeros_like(weights))
    return expected / expected.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def _normalised(weights: torch.Tensor, open_keys: torch.Tensor) -> torch.Tensor:
    weights = torch.where(open_keys, weights, torch.zeros_like(weights))
    return weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class _Flows:
    """Running ``observed``, ``uniform``, ``distance`` sums per (graph, layer, group) over (source, target)."""

    def __init__(self):
        self.patches: dict[tuple, np.ndarray] = {}
        self.labels: dict[tuple, np.ndarray] = {}

    @staticmethod
    def _add(store, key, source_onehot, target_onehot, parts, n_queries):
        sums = [source_onehot.T @ part @ target_onehot for part in parts]
        block = torch.stack(sums).cpu().numpy()  # [3, S, T]
        counts = source_onehot.sum(0).cpu().numpy()  # queries per source
        if key in store:
            store[key][0] += block
            store[key][1] += counts
        else:
            store[key] = [block, counts]

    def add(self, graph, layer, group, rows, parts, patch_onehot, label_onehot):
        """``parts``: observed, uniform, distance ``[Q, K]`` for the query ``rows`` over every cell as key."""
        if not len(rows):
            return
        self._add(self.patches, (graph, layer, group), patch_onehot[rows], patch_onehot, parts, len(rows))
        if label_onehot is not None:
            self._add(self.labels, (layer, group), label_onehot[rows], label_onehot, parts, len(rows))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    keep = p > 0
    return float((p[keep] * np.log(p[keep] / np.maximum(q[keep], 1e-300))).sum())


def patch_flow(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    spatial_unit_um: float,
    patch_um: float,
    distance_bins_um: Sequence[float],
    label_key: str | None = None,
    mask_seeds: Sequence[int] | None = None,
    block_radius_um: float | None = None,
    block_fraction: float | None = None,
    influence_queries_per_patch: int = 0,
    n_projections: int = 1,
    min_queries: int = 3,
    seed: int = 0,
    max_graphs: int = 0,
) -> dict[str, pd.DataFrame]:
    """Attention (and influence) flow between patches of each slide, and between labels, on masked passes.

    Parameters
    ----------
    module
        A module with a transformer (``recording_attention``); a model without one gives only
        influence rows.
    data_list
        PyG ``Data`` objects carrying ``pos`` and, without ``mask_seeds``, the masks to read under.
    prediction_task, prediction_level
        Passed to ``module._common_step``.
    device
        Device to move each graph to.
    spatial_unit_um
        µm per unit of ``pos``.
    patch_um
        Side of the square patches, µm.
    distance_bins_um
        Distance bins of the distance kernel the ``distance`` expectation is built from; distances
        beyond the last edge share the last bin.
    label_key
        An annotation on the graphs (one-hot, as ``extra_obs_keys`` attaches it) for the label flow.
    mask_seeds
        Draw discs (``masked_probing.draw_masks`` with ``block_radius_um``, ``block_fraction``) once
        per seed on copies of the graphs and add every draw up; None reads the masks the graphs carry.
    influence_queries_per_patch
        Hidden cells sampled per patch and draw whose receptive field is read as one more row,
        ``"influence"``, over the visible cells; 0 skips it (each costs ``n_projections`` backward
        passes per head).
    min_queries
        A source patch with fewer queries (over every draw) is left out of the summary.
    seed
        Seeds the influence samples and directions.

    Returns
    -------
    dict
        ``"summary"``: ``layer, group, metric, value, n_sources`` -- over source patches, weighted by
        their queries: ``kl_uniform``, ``kl_distance`` (the observed target distribution against each
        expectation), ``self_mass`` (the share kept within the source patch) with ``self_uniform`` and
        ``self_distance``, and ``beyond_distance`` (the share of mass on target patches that get more
        than twice their distance expectation). ``"patches"``: ``graph, layer, group, source, target,
        observed, uniform, distance, n_queries`` -- every non-empty pair, for maps. ``"centres"``:
        ``graph, patch, x_um, y_um``. ``"labels"`` (with ``label_key``): ``layer, group,
        query_label, key_label, mass, lift_uniform, lift_distance`` -- mass is the share of the
        query label's attention on the key label.
    """
    from interscale.evaluation.masked_probing import _model_heads, draw_masks
    from interscale.evaluation.receptive_field import _frozen_parameters, _influence

    graphs = list(data_list)
    if max_graphs > 0:
        graphs = graphs[:max_graphs]
    if mask_seeds is not None:
        if block_radius_um is None or block_fraction is None:
            raise ValueError("mask_seeds draws discs; pass block_radius_um and block_fraction.")
        draws = []
        for s in mask_seeds:
            copies = [g.clone() for g in graphs]
            draw_masks(
                copies, "blocks", float(block_fraction), seed=int(s), radius=float(block_radius_um) / spatial_unit_um
            )
            draws.append(copies)
    else:
        draws = [graphs]
    edges = torch.as_tensor([float(b) for b in distance_bins_um], dtype=torch.float64)
    n_bins = max(len(edges) - 1, 1)
    transformer = getattr(module, "global_module", module)
    has_attention = hasattr(transformer, "recording_attention")
    rng = np.random.default_rng(seed)
    directions = torch.Generator().manual_seed(int(seed))
    flows = _Flows()
    centres_rows = []

    was_training = module.training
    module.eval()
    try:
        with _rng_untouched(device):
            for draw in draws:
                loader = DataLoader(dataset=draw, shuffle=False, batch_size=1, num_workers=0)
                for g, batch in enumerate(loader):
                    batch = batch.to(device)
                    if getattr(batch, "pos", None) is None:
                        raise ValueError("patch_flow reads positions, but a graph carries no `pos`.")
                    dev = batch.pos.device
                    where = batch.pos.double() * float(spatial_unit_um)
                    patch, centres = patch_index(where.cpu().numpy(), patch_um)
                    if draw is draws[0]:
                        centres_rows += [
                            {"graph": g, "patch": p, "x_um": float(c[0]), "y_um": float(c[1])}
                            for p, c in enumerate(centres)
                        ]
                    patch_onehot = torch.zeros(batch.num_nodes, len(centres), dtype=torch.float64, device=dev)
                    patch_onehot[torch.arange(batch.num_nodes, device=dev), torch.as_tensor(patch, device=dev)] = 1.0
                    label_onehot = None
                    if label_key is not None:
                        codes = label_codes(batch, label_key)
                        n_labels = int(getattr(batch, label_key).shape[1])
                        label_onehot = torch.nn.functional.one_hot(codes, n_labels).double()
                    hidden = hidden_cells(module, batch)
                    dist = torch.cdist(where, where, compute_mode="donot_use_mm_for_euclid_dist")
                    bins = (torch.bucketize(dist, edges.to(dev), right=True) - 1).clamp(0, n_bins - 1)

                    if has_attention:
                        with transformer.recording_attention(reference=True) as record:
                            module._common_step(batch, prediction_task, prediction_level)
                        entry = record[0]
                        maps = entry.get("maps")
                        if maps is not None:
                            reference = entry.get("reference")
                            padding = entry["src_padding_mask"][0].bool().to(dev)
                            spots = (~padding).nonzero().squeeze(1)
                            nodes = torch.as_tensor(entry["index_nodes"][0], dtype=torch.long, device=dev)
                            allowed = entry.get("open")
                            for layer in range(maps.shape[0]):
                                rows = maps[layer].mean(0)[spots][:, spots].double()
                                exact = (
                                    rows if reference is None else reference[layer].mean(0)[spots][:, spots].double()
                                )
                                # Open keys from the mask: a sparse backend's zeros are not blocked keys.
                                allowed_keys = allowed[0][spots][:, spots].to(dev) if allowed is not None else exact > 0
                                # Into batch order: token k is cell nodes[k]; cells pad_batch left out stay empty.
                                full = torch.zeros(batch.num_nodes, batch.num_nodes, dtype=torch.float64, device=dev)
                                full[nodes[:, None], nodes[None, :]] = rows
                                open_keys = torch.zeros_like(full, dtype=torch.bool)
                                open_keys[nodes[:, None], nodes[None, :]] = allowed_keys
                                _add_rows(
                                    flows,
                                    g,
                                    f"l{layer}",
                                    full,
                                    open_keys,
                                    bins,
                                    n_bins,
                                    hidden,
                                    nodes,
                                    patch_onehot,
                                    label_onehot,
                                )

                    if influence_queries_per_patch > 0:
                        candidates = [np.flatnonzero((patch == p) & hidden.cpu().numpy()) for p in range(len(centres))]
                        picked = [
                            rng.choice(c, min(len(c), influence_queries_per_patch), replace=False)
                            for c in candidates
                            if len(c)
                        ]
                        queries = np.sort(np.concatenate(picked)) if picked else np.empty(0, np.int64)
                        if len(queries):
                            with _frozen_parameters(module):
                                heads = _influence(
                                    module, batch, queries, prediction_task, n_projections, directions, _model_heads
                                )
                            visible_keys = (~hidden)[None, :].expand(len(queries), -1)
                            for name, weights in heads.items():
                                if weights is None:
                                    continue
                                full = torch.zeros(batch.num_nodes, batch.num_nodes, dtype=torch.float64, device=dev)
                                full[torch.as_tensor(queries, device=dev)] = torch.nan_to_num(weights.double())
                                open_keys = torch.zeros_like(full, dtype=torch.bool)
                                open_keys[torch.as_tensor(queries, device=dev)] = visible_keys
                                layer = INFLUENCE if len(heads) == 1 else f"{INFLUENCE}_{name}"
                                rows_of = torch.as_tensor(queries, device=dev)
                                _add_rows(
                                    flows,
                                    g,
                                    layer,
                                    full,
                                    open_keys,
                                    bins,
                                    n_bins,
                                    hidden,
                                    rows_of,
                                    patch_onehot,
                                    label_onehot,
                                    groups=("hidden",),
                                )
    finally:
        module.train(was_training)
    return _tables(flows, centres_rows, min_queries)


def _add_rows(
    flows, graph, layer, full, open_keys, bins, n_bins, hidden, query_cells, patch_onehot, label_onehot, groups=GROUPS
):
    """Add the rows of ``query_cells`` of one ``[N, N]`` weight matrix (batch order) to the flows, by group."""
    q = query_cells
    weights, keys = full[q], open_keys[q]
    observed = _normalised(weights, keys)
    has_key = keys.any(1)
    uniform = keys.double() / keys.sum(1, keepdim=True).clamp_min(1)
    distance = _expected_by_distance(observed, keys, bins[q], n_bins)
    hid = hidden[q]
    for group in groups:
        pick = has_key & (hid if group == "hidden" else ~hid)
        idx = pick.nonzero().squeeze(1)
        if len(idx):
            parts = [observed[idx], uniform[idx], distance[idx]]
            flows.add(graph, layer, group, q[idx], parts, patch_onehot, label_onehot)


def _tables(flows: _Flows, centres_rows, min_queries) -> dict[str, pd.DataFrame]:
    patch_rows, summary = [], {}
    for (graph, layer, group), (block, counts) in flows.patches.items():
        observed, uniform, distance = block
        sources, targets = np.nonzero(observed + uniform)
        for s, t in zip(sources, targets, strict=True):
            patch_rows.append(
                {
                    "graph": graph,
                    "layer": layer,
                    "group": group,
                    "source": int(s),
                    "target": int(t),
                    "observed": observed[s, t],
                    "uniform": uniform[s, t],
                    "distance": distance[s, t],
                    "n_queries": int(counts[s]),
                }
            )
        acc = summary.setdefault((layer, group), {"weight": 0.0, "n": 0, **dict.fromkeys(_SUMMARY, 0.0)})
        for s in np.flatnonzero(counts >= min_queries):
            o, u, d = observed[s], uniform[s], distance[s]
            if o.sum() <= 0:
                continue
            w = float(counts[s])
            share = o / o.sum()
            values = {
                "kl_uniform": _kl(o, u),
                "kl_distance": _kl(o, d),
                "self_mass": share[s],
                "self_uniform": u[s] / u.sum(),
                "self_distance": d[s] / d.sum(),
                "beyond_distance": float(share[o > 2.0 * d].sum()),
            }
            for metric, value in values.items():
                acc[metric] += w * value
            acc["weight"] += w
            acc["n"] += 1
    rows = [
        {"layer": layer, "group": group, "metric": metric, "value": acc[metric] / acc["weight"], "n_sources": acc["n"]}
        for (layer, group), acc in sorted(summary.items())
        if acc["weight"] > 0
        for metric in _SUMMARY
    ]
    out = {
        "summary": pd.DataFrame(rows, columns=["layer", "group", "metric", "value", "n_sources"]),
        "patches": pd.DataFrame(patch_rows),
        "centres": pd.DataFrame(centres_rows),
    }
    label_rows = []
    for (layer, group), (block, _) in sorted(flows.labels.items()):
        observed, uniform, distance = block
        mass = observed / np.maximum(observed.sum(1, keepdims=True), 1e-300)
        with np.errstate(invalid="ignore", divide="ignore"):
            lift_u = observed / uniform
            lift_d = observed / distance
        for a, b in zip(*np.nonzero(observed + uniform), strict=True):
            label_rows.append(
                {
                    "layer": layer,
                    "group": group,
                    "query_label": int(a),
                    "key_label": int(b),
                    "mass": mass[a, b],
                    "lift_uniform": lift_u[a, b],
                    "lift_distance": lift_d[a, b],
                }
            )
    if flows.labels:
        out["labels"] = pd.DataFrame(label_rows)
    return out


_SUMMARY = ("kl_uniform", "kl_distance", "self_mass", "self_uniform", "self_distance", "beyond_distance")


def interesting_sources(patches: pd.DataFrame, graph: int, layer: str, group: str = "hidden", n: int = 4) -> list[int]:
    """The ``n`` source patches of one graph whose flow departs most from the distance expectation (KL)."""
    table = patches[(patches["graph"] == graph) & (patches["layer"] == layer) & (patches["group"] == group)]
    scores = {}
    for source, part in table.groupby("source"):
        if part["observed"].sum() > 0:
            scores[source] = _kl(part["observed"].to_numpy(), part["distance"].to_numpy())
    return [int(s) for s, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:n]]


def plot_patch_flow(
    patches: pd.DataFrame,
    centres: pd.DataFrame,
    *,
    graph: int,
    layer: str,
    sources: list[int],
    patch_um: float,
    group: str = "hidden",
    title: str = "",
):
    """One panel per source patch: the grid coloured by log2(observed / distance expectation), the source outlined.

    Returns the matplotlib figure. Blue: less than distance explains; red: more; grey: no mass.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    grid = centres[centres["graph"] == graph].set_index("patch")
    table = patches[(patches["graph"] == graph) & (patches["layer"] == layer) & (patches["group"] == group)]
    fig, axes = plt.subplots(1, len(sources), figsize=(3.2 * len(sources), 3.4), squeeze=False)
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    image = None
    for ax, source in zip(axes[0], sources, strict=True):
        part = table[table["source"] == source].set_index("target")
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.log2(part["observed"] / part["distance"]).replace([np.inf, -np.inf], np.nan)
        for patch, (x, y) in grid[["x_um", "y_um"]].iterrows():
            value = lift.get(patch, np.nan)
            colour = (0.85, 0.85, 0.85) if not np.isfinite(value) else plt.cm.RdBu_r(norm(np.clip(value, -3, 3)))
            ax.add_patch(Rectangle((x - patch_um / 2, y - patch_um / 2), patch_um, patch_um, color=colour, lw=0))
        x, y = grid.loc[source, ["x_um", "y_um"]]
        ax.add_patch(Rectangle((x - patch_um / 2, y - patch_um / 2), patch_um, patch_um, fill=False, ec="black", lw=2))
        ax.set_xlim(grid["x_um"].min() - patch_um, grid["x_um"].max() + patch_um)
        ax.set_ylim(grid["y_um"].min() - patch_um, grid["y_um"].max() + patch_um)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"source {source}", fontsize=9)
        image = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="log2 observed / distance kernel")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
