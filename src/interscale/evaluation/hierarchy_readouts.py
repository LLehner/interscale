"""What a hierarchical model's attention says, on a masked pass (``.claude/hierarchical_attention_plan.md``, T8 and T10).

Reads the export of :meth:`TransformerNodeEncoderHook.recording_hierarchy`: per layer and graph, each level's attention (dense,
or the routed rows) over the cluster hierarchy. Everything here is computed from the cell level's rows, a query's weights over
the cells it can read (CLS left out of every position-based quantity), so dense and routed models are read alike.

* **Fields** (T8): ``read_vec_i = sum_j A_ij (x_j - x_i)`` -- it points to where cell ``i`` reads from (the sources; the signal
  flows the other way) -- split exactly into a coarse part ``sum_j A_ij (c_p(j) - x_i)`` (which cluster) and a refinement
  ``sum_j A_ij (x_j - c_p(j))`` (where inside it), ``c`` the parent cluster's centroid; the row's entropy, its mean distance and
  its top targets. :func:`export_fields` writes them per slide (npz), :func:`plot_fields` draws them.
* **Planted pairs** (T10, with ``edges``): per range class, the attention mass on the true senders over chance
  (``|senders in C_i| / |C_i|``), the pair AUPRC with the attention weight as the score, the angle between ``read_vec`` and the
  true source direction (weighted by ``|read_vec|``), and -- routed -- the routing recall at each level: the share of
  sender-receiver pairs whose sender's cluster at that level is among the receiver's keys.
* **Health:** the coarse attention's entropy, the share of regions in no query's top-k (routing collapse), keys per query,
  the learned ``lambda`` and ``mu``, the hierarchy's cluster sizes and build time; the distance profile of the attention against
  uniform attention over the same keys.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from interscale.evaluation._spatial_weights import graph_edges
from interscale.evaluation.online_probes import _rng_untouched
from interscale.module.global_modules.hierarchical_attention import SparseRows


def cell_rows(entry: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``(keys [n, K], weights [H, n, K])``: each cell's attention at the cell level; key ``n`` is CLS, -1 padding."""
    hier = entry["hierarchy"]
    n = hier.counts[-1]
    attention = entry["attention"][-1]
    if isinstance(attention, SparseRows):
        return attention.rows(torch.arange(n))
    weights = attention[:, :n].float()
    return torch.arange(n + 1).expand(n, -1), weights


def read_vectors(keys: torch.Tensor, weights: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(read_vec [H, n, 2], cell_mass [H, n])`` from rows over cell keys (``keys`` in ``[0, n)``; CLS and padding left out)."""
    n = len(pos)
    cells = (keys >= 0) & (keys < n)
    w = torch.where(cells, weights, torch.zeros_like(weights))
    offsets = pos[keys.clamp(0, n - 1)] - pos[:, None, :]  # [n, K, 2]
    return torch.einsum("hnk,nkc->hnc", w, offsets), w.sum(-1)


def decompose(keys, weights, pos, parent, centroid) -> tuple[torch.Tensor, torch.Tensor]:
    """``(coarse, refine)``, each ``[H, n, 2]``: ``read_vec`` split by the keys' parent clusters (they sum to it exactly).

    ``parent`` ``[n]`` each cell's cluster one level up, ``centroid`` ``[n_par, 2]`` the clusters' centroids.
    """
    n = len(pos)
    cells = (keys >= 0) & (keys < n)
    w = torch.where(cells, weights, torch.zeros_like(weights))
    k = keys.clamp(0, n - 1)
    centre = centroid[parent[k]]  # [n, K, 2]
    coarse = torch.einsum("hnk,nkc->hnc", w, centre - pos[:, None, :])
    refine = torch.einsum("hnk,nkc->hnc", w, pos[k] - centre)
    return coarse, refine


def row_statistics(keys, weights, pos) -> dict[str, torch.Tensor]:
    """Per head and cell: the entropy of the row over its cell keys (renormalised) and its mean distance (µm)."""
    n = len(pos)
    cells = (keys >= 0) & (keys < n)
    w = torch.where(cells, weights, torch.zeros_like(weights))
    share = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(share * share.clamp_min(1e-30).log()).sum(-1)
    dist = (pos[keys.clamp(0, n - 1)] - pos[:, None, :]).norm(dim=-1)  # [n, K]
    return {"entropy": entropy, "mean_distance_um": (share * dist).sum(-1), "n_keys": cells.sum(-1)}


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, scores))


def planted_metrics(entry: dict, pairs: pd.DataFrame) -> list[dict]:
    """Per range class: sender lift, pair AUPRC, arrow angle and routing recall, for ``pairs`` in sequence positions.

    ``pairs``: ``sender``, ``receiver`` (positions in the sequence), ``range_class`` and optionally ``weight``.
    """
    hier, pos = entry["hierarchy"], entry["pos_um"].float()
    n = hier.counts[-1]
    keys, weights = cell_rows(entry)
    H = weights.shape[0]
    read, _ = read_vectors(keys, weights, pos)
    rows = []
    for range_class, group in pairs.groupby("range_class", observed=True):
        receivers = np.unique(group["receiver"].to_numpy())
        senders = {int(r): group.loc[group["receiver"] == r, "sender"].to_numpy() for r in receivers}
        weight_of = {
            int(r): (
                group.loc[group["receiver"] == r, "weight"].to_numpy()
                if "weight" in group
                else np.ones(len(senders[int(r)]))
            )
            for r in receivers
        }
        lift = np.full((H + 1, len(receivers)), np.nan)
        angle, magnitude = np.full((H + 1, len(receivers)), np.nan), np.zeros((H + 1, len(receivers)))
        ap_scores, ap_labels, covered = [[] for _ in range(H + 1)], [], 0
        for k, r in enumerate(receivers):
            r = int(r)
            row_keys = keys[r]
            open_cells = (row_keys >= 0) & (row_keys < n) & (row_keys != r)
            ids = row_keys[open_cells].numpy()
            is_sender = np.isin(ids, senders[r])
            w_rows = weights[:, r][:, open_cells].numpy()  # [H, |C|]
            w_all = np.concatenate([w_rows, w_rows.mean(0, keepdims=True)], 0)  # heads, then their mean
            if is_sender.any():
                covered += 1
                mass = w_all[:, is_sender].sum(1) / np.maximum(w_all.sum(1), 1e-12)
                lift[:, k] = mass / (is_sender.sum() / len(ids))
                for h in range(H + 1):
                    ap_scores[h].append(w_all[h])
                ap_labels.append(is_sender)
            truth = (pos[senders[r]] - pos[r]).numpy() * weight_of[r][:, None]
            truth = truth.sum(0)
            if np.linalg.norm(truth) > 0:
                vecs = torch.cat([read[:, r], read[:, r].mean(0, keepdim=True)], 0).numpy()  # [H + 1, 2]
                norms = np.linalg.norm(vecs, axis=1)
                cos = (vecs @ truth) / np.maximum(norms * np.linalg.norm(truth), 1e-12)
                angle[:, k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                magnitude[:, k] = norms
        labels = np.concatenate(ap_labels) if ap_labels else np.zeros(0, dtype=bool)
        heads = [f"h{h}" for h in range(H)] + ["mean"]
        for h, head in enumerate(heads):
            ap = _average_precision(np.concatenate(ap_scores[h]), labels) if ap_labels else float("nan")
            m = magnitude[h]
            ok = ~np.isnan(angle[h]) & (m > 0)
            rows += [
                {
                    "range_class": range_class,
                    "head": head,
                    "metric": "sender_lift",
                    "value": float(np.nanmean(lift[h])),
                },
                {"range_class": range_class, "head": head, "metric": "pair_auprc", "value": ap},
                {
                    "range_class": range_class,
                    "head": head,
                    "metric": "pair_auprc_chance",
                    "value": float(labels.mean()) if len(labels) else float("nan"),
                },
                {
                    "range_class": range_class,
                    "head": head,
                    "metric": "arrow_angle_deg",
                    "value": float(np.average(angle[h][ok], weights=m[ok])) if ok.any() else float("nan"),
                },
            ]
        rows.append(
            {
                "range_class": range_class,
                "head": "all",
                "metric": "sender_in_keys",
                "value": covered / max(len(receivers), 1),
            }
        )
        rows += routing_recall(entry, group, range_class)
        for row in rows:
            row.setdefault("n_receivers", len(receivers))
    return rows


def routing_recall(entry: dict, pairs: pd.DataFrame, range_class) -> list[dict]:
    """Routed levels only: per level, the share of pairs whose sender's cluster at that level is among the receiver's keys."""
    hier = entry["hierarchy"]
    L = hier.n_levels
    out = []
    sender = torch.as_tensor(pairs["sender"].to_numpy(), dtype=torch.long)
    receiver = torch.as_tensor(pairs["receiver"].to_numpy(), dtype=torch.long)
    for level in range(1, L + 1):
        attention = entry["attention"][level]
        if not isinstance(attention, SparseRows):
            continue
        if level < L:
            s, r = hier.assign[level].cpu()[sender], hier.assign[level].cpu()[receiver]
        else:
            s, r = sender, receiver
        keys, _ = attention.rows(r)
        hit = (keys == s[:, None]).any(-1).float().mean()
        out.append(
            {"range_class": range_class, "head": "all", "metric": f"routing_recall_l{level}", "value": float(hit)}
        )
    return out


def health(entry: dict) -> list[dict]:
    """Coarse entropy, routing collapse, keys per query per level, lambda and mu, of one layer's export."""
    hier = entry["hierarchy"]
    rows = []
    L = hier.n_levels
    if L > 0:
        a0 = entry["attention"][0].float()  # [H, n0 + 1, n0 + 1]
        n0 = hier.counts[0]
        regions = a0[:, :n0]
        entropy = -(regions * regions.clamp_min(1e-30).log()).sum(-1) / math.log(n0 + 1)
        rows.append({"level": 0, "head": "mean", "metric": "coarse_entropy", "value": float(entropy.mean())})
        k = min(int(entry["topk"][0]), n0)
        top = regions[:, :, :n0].amax(0).topk(k, dim=-1).indices
        chosen = torch.zeros(n0, dtype=torch.bool)
        chosen[top.reshape(-1)] = True
        rows.append(
            {"level": 0, "head": "all", "metric": "never_selected", "value": float(1.0 - chosen.float().mean())}
        )
    for level in range(1, L + 1):
        attention = entry["attention"][level]
        if isinstance(attention, SparseRows):
            n = hier.counts[level]
            keys, _ = attention.rows(torch.arange(n))
            rows.append(
                {
                    "level": level,
                    "head": "all",
                    "metric": "keys_per_query",
                    "value": float(((keys >= 0) & (keys < n)).sum(-1).float().mean()),
                }
            )
    for name in ("lambda", "mu"):
        value = entry.get(name)
        if value is None:
            continue
        for level in range(value.shape[0]):
            for head in range(value.shape[1]):
                rows.append(
                    {"level": level + 1, "head": f"h{head}", "metric": name, "value": float(value[level, head])}
                )
    return rows


def distance_profile(entry: dict, bins_um: Sequence[float], max_queries: int = 512, seed: int = 0) -> list[dict]:
    """The head-mean attention's weight per distance bin over what uniform attention over the same keys puts there."""
    pos = entry["pos_um"].float()
    n = len(pos)
    keys, weights = cell_rows(entry)
    rng = np.random.default_rng(seed)
    queries = torch.as_tensor(np.sort(rng.choice(n, size=min(max_queries, n), replace=False)))
    k = keys[queries]
    w = weights[:, queries].mean(0)
    cells = (k >= 0) & (k < n) & (k != queries[:, None])
    dist = (pos[k.clamp(0, n - 1)] - pos[queries][:, None]).norm(dim=-1)
    w = torch.where(cells, w, torch.zeros_like(w))
    share = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    uniform = cells.float() / cells.sum(-1, keepdim=True).clamp_min(1).float()
    rows = []
    for low, high in zip(bins_um[:-1], bins_um[1:], strict=True):
        inside = (dist > low) & (dist <= high)
        got, base = float((share * inside).sum(-1).mean()), float((uniform * inside).sum(-1).mean())
        rows.append(
            {
                "distance_um": f"{low:g}-{high:g}",
                "share": got,
                "uniform": base,
                "lift": got / base if base > 0 else float("nan"),
            }
        )
    return rows


def fields(entry: dict, top: int = 5) -> dict[str, np.ndarray]:
    """The T8 vector fields of one layer and graph, head-mean and per head, as arrays (what :func:`export_fields` saves)."""
    hier, pos = entry["hierarchy"], entry["pos_um"].float()
    keys, weights = cell_rows(entry)
    n = len(pos)
    read, mass = read_vectors(keys, weights, pos)
    stats = row_statistics(keys, weights, pos)
    out = {
        "pos_um": pos.numpy(),
        "visible": entry["visible"].numpy(),
        "read_vec": read.mean(0).numpy(),
        "read_vec_heads": read.numpy(),
        "cell_mass": mass.mean(0).numpy(),
        "entropy": stats["entropy"].mean(0).numpy(),
        "mean_distance_um": stats["mean_distance_um"].mean(0).numpy(),
    }
    if hier.n_levels > 0:
        parent, centroid = hier.up[-1].cpu(), hier.centroid[-2].cpu().float()
        coarse, refine = decompose(keys, weights, pos, parent, centroid)
        out["coarse_vec"], out["refine_vec"] = coarse.mean(0).numpy(), refine.mean(0).numpy()
        for level in range(hier.n_levels):
            out[f"assign_l{level}"] = hier.assign[level].cpu().numpy()
    cells = (keys >= 0) & (keys < n)
    w = torch.where(cells, weights.mean(0), torch.zeros_like(weights[0]))
    best = w.topk(min(top, w.shape[1]), dim=-1)
    out["top_targets"] = torch.where(
        best.values > 0, keys.gather(-1, best.indices), torch.full_like(best.indices, -1)
    ).numpy()
    out["top_weights"] = best.values.numpy()
    return out


def hierarchy_readouts(
    module,
    data_list,
    *,
    prediction_task: str,
    prediction_level: str,
    device,
    edges: pd.DataFrame | None = None,
    distance_bins_um: Sequence[float] = (),
    layers: Sequence[int] | None = None,
    max_graphs: int = 0,
    export_dir: str | Path | None = None,
    export_graphs: int = 0,
    graph_names: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Every readout above, over the graphs of ``data_list`` under the masks they carry (the module corrupts its input as in
    training). ``layers``: which layers' planted metrics and profiles to report (default: all). ``export_dir``: write the
    last-layer fields of the first ``export_graphs`` graphs (0: every graph) there (``<name>.npz``, with the planted pairs in
    sequence positions when there are any). Consumes no RNG and leaves the module's mode as it was.

    Returns ``hier_planted`` (layer, range_class, head, metric, value, n_receivers), ``hier_health``, ``hier_profile`` and
    ``hier_stats`` (per graph: the hierarchy's sizes and build time), each only when it has rows.
    """
    transformer = getattr(module, "global_module", module)
    hierarchical = getattr(transformer, "hierarchy", None) is not None
    if not hierarchical and (
        not hasattr(transformer, "recording_attention")
        or getattr(transformer, "pooling", None) is not None
        or getattr(transformer, "context_only", "none") != "none"
    ):
        return {}  # no transformer, or one whose maps are not over cells: nothing to read alike
    graphs = list(data_list)[:max_graphs] if max_graphs else list(data_list)
    planted, healthy, profile, stats = [], [], [], []
    was_training = module.training
    module.eval()
    loader = DataLoader(dataset=graphs, shuffle=False, batch_size=1, num_workers=0)
    if export_dir is not None:
        Path(export_dir).mkdir(parents=True, exist_ok=True)
    try:
        with _rng_untouched(device):
            for g, batch in enumerate(loader):
                batch = batch.to(device)
                if hierarchical:
                    with torch.no_grad(), transformer.recording_hierarchy() as record:
                        module._common_step(batch, prediction_task, prediction_level)
                else:
                    record = _dense_record(module, transformer, batch, prediction_task, prediction_level)
                name = graph_names[g] if graph_names is not None else f"graph{g}"
                n_layers = len(record)
                chosen = {(layer % n_layers) for layer in layers} if layers is not None else set(range(n_layers))
                pairs = None
                if edges is not None:
                    found = graph_edges(edges, batch.obs_names)
                    nodes = record[0]["nodes"] if record and record[0]["nodes"] is not None else None
                    if nodes is not None and len(found):
                        place = pd.Series(np.arange(len(nodes)), index=np.asarray(nodes))
                        keep = np.isin(found["sender"], nodes) & np.isin(found["receiver"], nodes)
                        found = found[keep].copy()
                        found["sender"] = place.loc[found["sender"]].to_numpy()
                        found["receiver"] = place.loc[found["receiver"]].to_numpy()
                    pairs = found if len(found) else None
                for entry in record:
                    layer = int(entry["layer"])
                    if layer == 0:
                        stats.append({"graph": name, **entry["hierarchy"].stats()})
                    healthy += [{"graph": name, "layer": layer, **row} for row in health(entry)]
                    if layer not in chosen:
                        continue
                    if pairs is not None:
                        planted += [{"graph": name, "layer": layer, **row} for row in planted_metrics(entry, pairs)]
                    if len(distance_bins_um) > 1:
                        profile += [
                            {"graph": name, "layer": layer, **row} for row in distance_profile(entry, distance_bins_um)
                        ]
                if export_dir is not None and record and (not export_graphs or g < export_graphs):
                    arrays = fields(record[-1])
                    if pairs is not None:
                        arrays |= {
                            "planted_sender": pairs["sender"].to_numpy(),
                            "planted_receiver": pairs["receiver"].to_numpy(),
                            "planted_class": np.asarray(pairs["range_class"].astype(str).tolist(), dtype=str),
                        }
                    np.savez_compressed(Path(export_dir) / f"{name}.npz", **arrays)
    finally:
        module.train(was_training)

    tables = {}
    if planted:
        frame = pd.DataFrame(planted)
        tables["hier_planted"] = (
            frame.groupby(["layer", "range_class", "head", "metric"], observed=True)
            .agg(value=("value", "mean"), n_receivers=("n_receivers", "sum"))
            .reset_index()
        )
    if healthy:
        frame = pd.DataFrame(healthy)
        tables["hier_health"] = (
            frame.groupby(["layer", "level", "head", "metric"]).agg(value=("value", "mean")).reset_index()
        )
    if profile:
        frame = pd.DataFrame(profile)
        tables["hier_profile"] = (
            frame.groupby(["layer", "distance_um"], sort=False)
            .agg(share=("share", "mean"), uniform=("uniform", "mean"), lift=("lift", "mean"))
            .reset_index()
        )
    if stats:
        tables["hier_stats"] = pd.DataFrame(stats)
    return tables


def _dense_record(module, transformer, batch, prediction_task, prediction_level) -> list[dict]:
    """A dense transformer's maps of one graph in the hierarchy's export format (no coarse level), so both read alike."""
    from interscale.module.global_modules.hierarchy import flat_hierarchy

    with torch.no_grad(), transformer.recording_attention() as recorded:
        module._common_step(batch, prediction_task, prediction_level)
    entry = recorded[0]
    if entry.get("maps") is None:
        return []
    padding = entry["src_padding_mask"][0].bool()
    cells = (~padding).nonzero().squeeze(1).cpu()
    columns = torch.cat([cells, torch.tensor([padding.shape[0]])])  # the cells, then CLS
    nodes = list(entry["index_nodes"][0])
    unit = (
        float(getattr(transformer.attention_bias, "spatial_unit_um", 1.0))
        if transformer.attention_bias is not None
        else 1.0
    )
    pos = batch.pos[:, :2].float().cpu()[torch.as_tensor(nodes, dtype=torch.long)] * unit
    hidden = transformer.wholly_hidden(module._common_step_masking(batch)[0]).cpu()[
        torch.as_tensor(nodes, dtype=torch.long)
    ]
    out = []
    for layer, maps in enumerate(entry["maps"].cpu()):
        rows = maps[:, cells][:, :, columns]  # [H, n, n + 1]
        out.append(
            {
                "layer": layer,
                "graph": 0,
                "hierarchy": flat_hierarchy(pos.numpy()),
                "pos_um": pos,
                "visible": (~hidden).float(),
                "nodes": nodes,
                "attention": [rows],
                "topk": [],
                "candidates": "dense",
            }
        )
    return out


def plot_fields(
    npz_path,
    out_path,
    *,
    senders: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
    max_arrows: int = 1500,
    seed: int = 0,
) -> None:
    """Quiver plots of one slide's fields: ``read_vec`` coloured by entropy, its coarse and refinement parts; planted senders
    as black crosses -- ``senders`` (positions ``[k, 2]``, µm), else the npz's planted pairs of the range ``classes`` (None:
    all). Arrows point to where a cell reads from (the sources); the signal flows the other way.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    pos = data["pos_um"]
    if senders is None and "planted_sender" in data:
        keep = np.ones(len(data["planted_sender"]), dtype=bool)
        if classes is not None:
            keep = np.isin(data["planted_class"], list(classes))
        senders = pos[np.unique(data["planted_sender"][keep])]
    rng = np.random.default_rng(seed)
    show = np.sort(rng.choice(len(pos), size=min(max_arrows, len(pos)), replace=False))
    panels = [("read_vec", "where each cell reads from (head mean)")]
    if "coarse_vec" in data:
        panels += [("coarse_vec", "coarse part: which cluster"), ("refine_vec", "refinement: where inside it")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels, strict=True):
        vec = data[key]
        quiver = ax.quiver(
            pos[show, 0],
            pos[show, 1],
            vec[show, 0],
            vec[show, 1],
            data["entropy"][show],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="viridis",
            width=0.003,
        )
        if senders is not None and len(senders):
            ax.scatter(
                senders[:, 0], senders[:, 1], marker="x", c="black", s=12, linewidths=0.8, label="planted senders"
            )
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.colorbar(quiver, ax=axes[0].tolist(), shrink=0.8, label="row entropy (nats)")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
