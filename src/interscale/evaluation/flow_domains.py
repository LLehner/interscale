"""Attention -> net flow -> vector fields -> flow domains -> source / sink roles, with planted truth to validate it.

The pipeline of a thesis chapter on host-pathogen Visium (roadmap.md, "Vector-field context", 2026-11-12), which turns
InterScale's attention into continuous fields and segments the tissue by them, rebuilt here as functions of arrays so it runs
on any dataset and, on synth_spot, against planted senders and receivers. Its steps and the chapter's equations:

1. **Relevance** (Eq. 4.3): ``A = ReLU(mean_{l,h} grad_Attn(l,h) L ⊙ Attn(l,h))`` with ``L`` the masked reconstruction loss
   (:func:`relevance_matrix`); or the last layer's head mean (what :mod:`flow_maps` reads). Never the gradient of a summed
   output after a LayerNorm (notes.md: degenerate).
2. **Net flow** ``F_ij = A_ij - A_ji`` (:func:`~interscale.evaluation.flow_maps.net_flow_pairs`; ``F > 0``: cell ``i`` reads
   ``j`` more, flow ``j -> i``), normalised by its largest magnitude per graph (Eq. 4.6).
3. **Cell vectors**, two constructions:
   - ``sender`` (Eq. 4.8): ``v_j = sum_i F+_ji unit(p_i - p_j) exp(-|p_i - p_j|^2 / 2 d_max^2)``, anchored at the sender ``j``
     and pointing at its receivers (:func:`sender_vectors`). The chapter's own limitation: a sender's contribution has positive
     divergence upstream and negative downstream wherever the receivers are, and the kernel weights distance while the net
     balance does not.
   - ``midpoint`` (ours, :mod:`flow_maps`): each pair's arrow deposited at its midpoint, so the exact per-cell divergence is
     ``out - in`` (:func:`~interscale.evaluation.flow_maps.cell_strengths`), summing to 0 (:func:`midpoint_vectors` returns
     the per-pair arrows at midpoints; :func:`exact_divergence` the per-cell value).
4. **Gridded fields** per channel (a label; one channel without labels; Eq. 4.9-4.10): Gaussian deposit of the vectors on an
   ``R x R`` grid over the bounding box, Gaussian smoothing, magnitude ``|V_c|`` and divergence ``div V_c`` by central
   differences (:func:`grid_fields`).
5. **Domains** (Eq. 4.11-4.12): k-means on the standardised ``[div_c, |V_c|]`` of every channel per grid point
   (:func:`flow_domains`); cells take their nearest grid point's domain (:func:`cells_to_domains`).
6. **Roles** (Eq. 4.13): per channel and domain, source / sink / neutral by the mean divergence against ``mu +- kappa sigma``
   of the channel's grid distribution (:func:`domain_roles`). **Net balance** ``B(m)`` (Eq. 4.14) of the discrete flow
   between domains, Spearman with the domain divergence (:func:`net_balance`).
7. **Validation** on planted truth (:func:`planted_scores`): AUROC of a cell's divergence for planted senders against the
   rest (sources should diverge) and of its negative for receivers (sinks should converge); the share of planted senders in a
   source domain and of receivers in a sink domain, per programme; the same for the exact ``out - in``. Compared with an
   untrained twin and across seeds (:func:`partition_agreement`: ARI of two domain partitions on the same grid).

:func:`run_pipeline` strings 2-6 together for one graph from ``A``, positions and labels; :func:`pipeline_tables` scores it.
Everything here consumes no RNG beyond the k-means seed it is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from interscale.evaluation.flow_maps import cell_strengths, net_flow_pairs

ROLES = ("source", "sink", "neutral")


# ----------------------------------------------------------------------------------------------- 1. relevance


def relevance_matrix(maps, grads) -> np.ndarray:
    """Eq. 4.3: ``ReLU(mean over layers and heads of grad ⊙ attn)`` over the cell keys.

    ``maps`` and ``grads``: ``[L, H, n(+1), n(+1)]`` (the same shape; a trailing CLS row / column is dropped), as the encoder's
    ``get_attn_output_weights`` / ``get_attn_gradients`` give them per layer for one graph. Returns ``[n, n]`` with a zero diagonal.
    """
    m = np.asarray(maps, dtype=np.float64)
    g = np.asarray(grads, dtype=np.float64)
    if m.shape != g.shape:
        raise ValueError(f"maps {m.shape} and gradients {g.shape} differ in shape.")
    rel = np.maximum((m * g).mean(axis=(0, 1)), 0.0)
    np.fill_diagonal(rel, 0.0)
    return rel


def head_mean_matrix(maps) -> np.ndarray:
    """The last layer's head mean over the cell keys: ``[n, n]`` from ``[L, H, n, n]`` (or ``[H, n, n]``), zero diagonal."""
    m = np.asarray(maps, dtype=np.float64)
    last = m[-1] if m.ndim == 4 else m
    a = last.mean(axis=0)
    np.fill_diagonal(a, 0.0)
    return a


def attention_with_gradients(module, batch, prediction_task: str, prediction_level: str):
    """One masked pass of ``module`` on ``batch`` (one graph) with the maps hooked: ``(maps, grads, nodes, hidden)``.

    ``maps`` / ``grads`` ``[L, H, n, n]`` over the kept cells (CLS dropped), the gradient of the masked squared error of the
    pass (the reconstruction loss over the hidden entries) with respect to each layer's map; ``nodes`` the kept cells' node
    indices; ``hidden`` ``[n]`` bool, the cells hidden whole. The module is left in the mode it was in; its parameters' gradients
    are cleared afterwards.
    """
    import torch

    transformer = getattr(module, "global_module", module)
    was_training = module.training
    module.eval()
    try:
        with transformer.hooking_attention():
            out = module._common_step(batch, prediction_task, prediction_level)
            y_pred, y_true = out.y_pred, out.y_true
            entry = out.entry_mask if out.entry_mask is not None else torch.ones_like(y_true, dtype=torch.bool)
            loss = ((y_pred - y_true)[entry] ** 2).mean()
            module.zero_grad(set_to_none=True)
            loss.backward()
        view = out.view
        padding = view.src_padding_mask[0].bool()
        cells = (~padding[:-1]).nonzero().squeeze(1).cpu()
        maps, grads = [], []
        for layer in transformer.transformer_encoder.layers:
            w, g = layer.get_attn_output_weights(), layer.get_attn_gradients()
            if w is None or g is None:
                raise ValueError("the pass kept no map or no gradient: the transformer must form explicit maps (dense attention).")
            w, g = w.detach().cpu(), g.detach().cpu()
            maps.append(w[:, cells][:, :, cells].numpy())  # [H, n, n]: one graph, heads first
            grads.append(g[:, cells][:, :, cells].numpy())
        nodes = np.asarray(view.padded_node_idx.cpu())
        masked, _, _ = module._common_step_masking(batch)
        hidden = transformer.wholly_hidden(masked).cpu().numpy()[nodes]
        module.zero_grad(set_to_none=True)
        return np.stack(maps), np.stack(grads), nodes, hidden
    finally:
        module.train(was_training)


# ----------------------------------------------------------------------------------------------- 2-3. net flow and vectors


def normalised_net_flow(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(source, receiver, flow)`` of every pair with a positive net flow, divided by the largest magnitude (Eq. 4.6; 0 if none)."""
    source, receiver, flow, _ = net_flow_pairs(csr_matrix(np.asarray(A, dtype=np.float32)), mutual=False)
    scale = float(flow.max()) if len(flow) else 0.0
    return source, receiver, (flow / scale if scale > 0 else flow)


def sender_vectors(pos, source, receiver, flow, d_max: float, eps: float = 1e-6) -> np.ndarray:
    """Eq. 4.8: ``[n, 2]``, each sender's vector towards the cells that receive from it, net flow and distance weighted."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    v = np.zeros((len(pos), 2), dtype=np.float64)
    if not len(source):
        return v
    delta = pos[receiver] - pos[source]
    dist = np.hypot(delta[:, 0], delta[:, 1])
    w = np.asarray(flow, dtype=np.float64) * np.exp(-(dist**2) / (2.0 * d_max**2)) / (dist + eps)
    np.add.at(v, source, delta * w[:, None])
    return v


def midpoint_vectors(pos, source, receiver, flow) -> tuple[np.ndarray, np.ndarray]:
    """Ours: ``(points [P, 2], vectors [P, 2])``, each positive pair's unit arrow times its net flow, at the pair's midpoint."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    delta = pos[receiver] - pos[source]
    dist = np.hypot(delta[:, 0], delta[:, 1])
    ok = dist > 0
    unit = np.zeros_like(delta)
    unit[ok] = delta[ok] / dist[ok, None]
    return (pos[source] + pos[receiver]) / 2.0, unit * np.asarray(flow, dtype=np.float64)[:, None]


def exact_divergence(n: int, source, receiver, flow) -> np.ndarray:
    """``out - in`` per cell: the exact divergence of the net flow (sums to 0)."""
    out, into = cell_strengths(n, source, receiver, flow)
    return (out - into).astype(np.float64)


# ----------------------------------------------------------------------------------------------- 4. gridded fields


def make_grid(pos, resolution: int) -> dict:
    """An ``R x R`` grid over the bounding box of ``pos``: ``origin``, ``step`` (um, isotropic), ``x`` / ``y`` coordinates."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    low, high = pos.min(0), pos.max(0)
    span = max(float((high - low).max()), 1e-9)
    step = span / max(int(resolution) - 1, 1)
    xs = low[0] + step * np.arange(resolution)
    ys = low[1] + step * np.arange(resolution)
    return {"origin": low, "step": step, "x": xs, "y": ys, "resolution": int(resolution)}


def deposit(grid: dict, points, values, sigma_um: float) -> np.ndarray:
    """Eq. 4.9-4.10: ``[R, R, D]``, Gaussian deposit of ``values`` ``[P, D]`` at ``points`` ``[P, 2]`` on the grid (bin, then blur)."""
    R, step = grid["resolution"], grid["step"]
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    idx = np.clip(np.rint((np.asarray(points, dtype=np.float64)[:, :2] - grid["origin"]) / step), 0, R - 1).astype(np.int64)
    flat = idx[:, 0] * R + idx[:, 1]
    out = np.zeros((R * R, values.shape[1]), dtype=np.float64)
    for d in range(values.shape[1]):
        out[:, d] = np.bincount(flat, weights=values[:, d], minlength=R * R)
    out = out.reshape(R, R, -1)
    s = sigma_um / step
    if s > 0:
        out = ndimage.gaussian_filter(out, sigma=(s, s, 0), mode="constant")
    return out


def grid_fields(
    grid: dict, points, vectors, channel, n_channels: int, *, sigma_grid_um: float, sigma_smooth_um: float
) -> dict[str, np.ndarray]:
    """Per channel the gridded vector field, its magnitude and divergence: ``V [C, R, R, 2]``, ``mag [C, R, R]``, ``div [C, R, R]``.

    ``channel`` ``[P]`` the channel of each vector's source (a label code; 0 everywhere for one channel). Divergence by central
    differences on the smoothed field, in grid-index units (as the chapter: its scale depends on the resolution).
    """
    R, step = grid["resolution"], grid["step"]
    channel = np.asarray(channel, dtype=np.int64)
    V = np.zeros((n_channels, R, R, 2), dtype=np.float64)
    for c in range(n_channels):
        keep = channel == c
        if keep.any():
            V[c] = deposit(grid, np.asarray(points)[keep], np.asarray(vectors)[keep], sigma_grid_um)
    s = sigma_smooth_um / step
    if s > 0:
        V = ndimage.gaussian_filter(V, sigma=(0, s, s, 0), mode="constant")
    mag = np.sqrt((V**2).sum(-1))
    div = np.stack([np.gradient(V[c, :, :, 0], axis=0) + np.gradient(V[c, :, :, 1], axis=1) for c in range(n_channels)])
    return {"V": V, "mag": mag, "div": div}


def grid_values(grid: dict, field: np.ndarray, pos) -> np.ndarray:
    """``field`` ``[R, R]`` read at each cell's nearest grid point: ``[n]``."""
    R, step = grid["resolution"], grid["step"]
    idx = np.clip(np.rint((np.asarray(pos, dtype=np.float64)[:, :2] - grid["origin"]) / step), 0, R - 1).astype(np.int64)
    return np.asarray(field)[idx[:, 0], idx[:, 1]]


# ----------------------------------------------------------------------------------------------- 5-6. domains, roles, balance


def flow_domains(fields: dict, k: int, seed: int = 0, *, occupied: np.ndarray | None = None) -> np.ndarray:
    """Eq. 4.11-4.12: k-means on the standardised ``[div_c, mag_c]`` of every channel per grid point: ``[R, R]`` labels (-1 off tissue).

    ``occupied`` ``[R, R]`` bool restricts the clustering to grid points with tissue (the chapter clusters every grid point of the
    bounding box; on a slide with empty slide around it the empty points would form their own domain).
    """
    from sklearn.cluster import KMeans

    div, mag = fields["div"], fields["mag"]
    C, R = div.shape[0], div.shape[1]
    feats = np.stack([div.reshape(C, -1), mag.reshape(C, -1)], axis=1).reshape(2 * C, -1).T  # [R*R, 2C]
    keep = np.ones(R * R, dtype=bool) if occupied is None else np.asarray(occupied, dtype=bool).reshape(-1)
    x = feats[keep]
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    labels = np.full(R * R, -1, dtype=np.int64)
    k = int(min(k, max(len(x), 1)))
    if len(x):
        labels[keep] = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit_predict(x)
    return labels.reshape(R, R)


def occupied_grid(grid: dict, pos, radius_um: float) -> np.ndarray:
    """``[R, R]`` bool: grid points within ``radius_um`` of a cell (the tissue)."""
    from scipy.spatial import cKDTree

    R = grid["resolution"]
    gx, gy = np.meshgrid(grid["x"], grid["y"], indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d, _ = cKDTree(np.asarray(pos, dtype=np.float64)[:, :2]).query(points)
    return (d <= radius_um).reshape(R, R)


def cells_to_domains(grid: dict, domains: np.ndarray, pos) -> np.ndarray:
    """Each cell's domain: its nearest grid point's label (Eq. 4.13, step 4)."""
    return grid_values(grid, domains, pos).astype(np.int64)


def domain_roles(fields: dict, domains: np.ndarray, kappa: float) -> pd.DataFrame:
    """Eq. 4.13: per channel and domain the mean divergence and its role against ``mu +- kappa sigma`` of the channel's grid values."""
    rows = []
    div = fields["div"]
    on = domains >= 0
    for c in range(div.shape[0]):
        values = div[c][on]
        mu, sd = float(values.mean()), float(values.std())
        for m in np.unique(domains[on]):
            mean = float(div[c][domains == m].mean())
            role = "source" if mean > mu + kappa * sd else "sink" if mean < mu - kappa * sd else "neutral"
            rows.append({"channel": c, "domain": int(m), "mean_divergence": mean, "role": role, "n_grid": int((domains == m).sum())})
    return pd.DataFrame(rows)


def net_balance(domain_of_cell: np.ndarray, source, receiver, flow) -> pd.DataFrame:
    """Eq. 4.14 on the cells' domains: ``B(m)`` = flow received from other domains minus flow sent to them (positive: a net receiver).

    The flow is normalised by its largest between-domain magnitude as the chapter normalises its domain matrix; ``B`` sums to 0.
    """
    d = np.asarray(domain_of_cell, dtype=np.int64)
    s, r, f = d[source], d[receiver], np.asarray(flow, dtype=np.float64)
    labels = np.unique(d[d >= 0])
    index = {m: i for i, m in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    keep = (s >= 0) & (r >= 0) & (s != r)
    np.add.at(M, (np.vectorize(index.get)(s[keep]), np.vectorize(index.get)(r[keep])), f[keep]) if keep.any() else None
    net = M - M.T  # net[a, b] > 0: a sends more to b than it receives
    scale = float(np.abs(net).max()) if net.size else 0.0
    if scale > 0:
        net = net / scale
    balance = net.sum(0) - net.sum(1)  # received minus sent
    return pd.DataFrame({"domain": labels, "net_balance": balance})


# ----------------------------------------------------------------------------------------------- the pipeline


def run_pipeline(
    A: np.ndarray,
    pos,
    *,
    labels=None,
    n_channels: int | None = None,
    construction: str = "sender",
    d_max_um: float,
    resolution: int = 64,
    sigma_grid_um: float,
    sigma_smooth_um: float,
    k: int = 5,
    kappa: float = 1.0,
    seed: int = 0,
    tissue_radius_um: float | None = None,
) -> dict:
    """Steps 2-6 on one graph. ``A`` ``[n, n]`` the attention or relevance matrix (cell ``i`` reads ``j``), ``pos`` in um,
    ``labels`` ``[n]`` channel codes (None: one channel), ``construction`` ``sender`` (Eq. 4.8) or ``midpoint`` (ours).

    Returns the grid, the fields, the domain grid, each cell's domain, divergence at the cell (from the field, summed over
    channels) and exact ``out - in``, the roles and the net balance with its Spearman against the domains' divergence.
    """
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    n = len(pos)
    channel = np.zeros(n, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
    C = int(channel.max()) + 1 if n_channels is None else int(n_channels)
    source, receiver, flow = normalised_net_flow(A)
    if construction == "sender":
        vectors = sender_vectors(pos, source, receiver, flow, d_max_um)
        points, vec_channel = pos, channel
    elif construction == "midpoint":
        points, vectors = midpoint_vectors(pos, source, receiver, flow)
        vec_channel = channel[source]
    else:
        raise ValueError(f"construction must be 'sender' or 'midpoint', got {construction!r}.")
    grid = make_grid(pos, resolution)
    fields = grid_fields(grid, points, vectors, vec_channel, C, sigma_grid_um=sigma_grid_um, sigma_smooth_um=sigma_smooth_um)
    occupied = None if tissue_radius_um is None else occupied_grid(grid, pos, tissue_radius_um)
    domains = flow_domains(fields, k, seed, occupied=occupied)
    cell_domain = cells_to_domains(grid, domains, pos)
    roles = domain_roles(fields, domains, kappa)
    balance = net_balance(cell_domain, source, receiver, flow)
    total = roles.groupby("domain")["mean_divergence"].mean().rename("total_divergence")
    balance = balance.merge(total, left_on="domain", right_index=True, how="left")
    rho = float(spearmanr(balance["net_balance"], balance["total_divergence"]).correlation) if len(balance) > 2 else float("nan")
    cell_div = grid_values(grid, fields["div"].sum(0), pos)
    return {
        "grid": grid,
        "fields": fields,
        "domains": domains,
        "cell_domain": cell_domain,
        "cell_divergence": cell_div,
        "cell_exact": exact_divergence(n, source, receiver, flow),
        "roles": roles,
        "balance": balance,
        "balance_rho": rho,
        "construction": construction,
        "n_pairs": int(len(flow)),
    }


# ----------------------------------------------------------------------------------------------- 7. validation


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = np.asarray(score)[positive], np.asarray(score)[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def planted_scores(result: dict, senders: dict[str, np.ndarray], receivers: dict[str, np.ndarray], channel=None) -> pd.DataFrame:
    """How the pipeline's output matches planted truth, per programme: one row per (programme, role, measure).

    ``senders`` / ``receivers``: ``program -> [n] bool`` (the planted sender / receiver cells). Measures: ``auroc_field`` (the
    field divergence at the cell ranks senders above the rest; for receivers its negative), ``auroc_exact`` (the same with
    ``out - in``), ``in_role_domain`` (the share of the cells whose domain has the matching role -- source for senders, sink for
    receivers -- in the cell's channel), ``n`` cells.
    """
    roles = result["roles"]
    role_of = {(int(r.channel), int(r.domain)): r.role for r in roles.itertuples()}
    cell_domain = result["cell_domain"]
    n = len(cell_domain)
    channel = np.zeros(n, dtype=np.int64) if channel is None else np.asarray(channel, dtype=np.int64)
    rows = []
    for role, flags, sign, want in (("sender", senders, 1.0, "source"), ("receiver", receivers, -1.0, "sink")):
        for program, flag in flags.items():
            flag = np.asarray(flag, dtype=bool)
            if not flag.any():
                continue
            in_role = np.mean([role_of.get((int(channel[i]), int(cell_domain[i]))) == want for i in np.flatnonzero(flag)])
            rows.append({"program": program, "role": role, "n": int(flag.sum()),
                         "auroc_field": _auroc(sign * result["cell_divergence"], flag),
                         "auroc_exact": _auroc(sign * result["cell_exact"], flag),
                         "in_role_domain": float(in_role)})
    return pd.DataFrame(rows)


def partition_agreement(domains_a: np.ndarray, domains_b: np.ndarray) -> float:
    """ARI of two domain partitions of the same grid (points off tissue in either are left out)."""
    from sklearn.metrics import adjusted_rand_score

    a, b = np.asarray(domains_a).ravel(), np.asarray(domains_b).ravel()
    keep = (a >= 0) & (b >= 0)
    return float(adjusted_rand_score(a[keep], b[keep])) if keep.any() else float("nan")


def planted_flags(pairs: pd.DataFrame, n: int, by: str = "program") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """From a graph's planted pairs (``sender``, ``receiver`` node indices, a ``by`` column): ``(senders, receivers)`` flags per group, plus ``all``."""
    senders, receivers = {}, {}
    groups = pairs[by].astype(str).to_numpy() if by in pairs else np.array(["all"] * len(pairs))
    for g in np.unique(groups):
        rows = pairs[groups == g]
        s, r = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        s[rows["sender"].to_numpy().astype(int)] = True
        r[rows["receiver"].to_numpy().astype(int)] = True
        senders[g], receivers[g] = s, r
    s_all, r_all = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    s_all[pairs["sender"].to_numpy().astype(int)] = True
    r_all[pairs["receiver"].to_numpy().astype(int)] = True
    senders["all"], receivers["all"] = s_all, r_all
    return senders, receivers


def pipeline_tables(result: dict, pairs: pd.DataFrame | None, channel=None) -> dict[str, pd.DataFrame]:
    """The report tables of one run: ``roles``, ``balance`` (with the Spearman), and with ``pairs`` the planted ``scores``."""
    tables = {"roles": result["roles"].copy(), "balance": result["balance"].assign(balance_rho=result["balance_rho"])}
    if pairs is not None and len(pairs):
        senders, receivers = planted_flags(pairs, len(result["cell_domain"]))
        tables["scores"] = planted_scores(result, senders, receivers, channel)
    return tables
