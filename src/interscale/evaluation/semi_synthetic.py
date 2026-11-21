"""Planted interactions on real cells (``.claude/ccc_model_plan.md``, E2): AMICI's semi-synthetic benchmark, as a count spike-in.

Everything of the real data stays -- coordinates, cell types, every cell's own counts with their noise, FOV offsets and spillover --
and a few response genes of a receiver type get extra counts as a function of the distance to the nearest cell (or the cells) of
a sender type. So the truth is exact (which receivers, which senders, which genes, which range) while geometry, composition,
segmentation and noise are the data's own: it answers "would a model have found a 200-500 um interaction on this tissue if one
were there?", which a lattice (synth_spot) cannot.

* :class:`PlantedPair` -- sender type, receiver type, kernel (``step``: contact within ``length_um``; ``exp``: ``exp(-d / l)`` of the
  nearest sender; ``field``: ``1 - exp(-ln 2 * S_i / median S)``, ``S_i = sum_j exp(-d_ij / l)`` over every sender within
  ``3 l``, scaled so the median reached receiver sits at 0.5: a morphogen that follows the senders' density -- unscaled, 20 senders
  in reach saturate it at 1 for every receiver, a constant the within-type target removes),
  the fold change at the kernel's maximum, the number of response genes. ``spike: False`` makes it a control: the genes are
  chosen and the truth written, no count is added (false positives).
* :func:`choose_genes` -- response genes expressed in the receiver type (detected in >= 20 % of its cells), not higher in the
  sender type (so spillover from a sender cannot imitate the response), with little within-type spatial autocorrelation (the
  lowest third by Moran's I among the receiver type's own 8 nearest neighbours), drawn among those; never two pairs' genes.
* :func:`spike_in` -- ``x_ig += Poisson((fold - 1) f_i mu_Rg s_i)``: ``f_i`` the kernel, ``mu_Rg`` the receiver type's mean count,
  ``s_i`` the cell's library size over the type's mean. Writes the truth in synth_spot's format (``uns['synthetic']
  ['interaction_edges']``: sender, receiver -- row positions --, program, range_class, slide, weight; ``var['interaction']``,
  ``var['role']``; ``obs['spike_<program>']`` the kernel), so the planted-pair readouts read it unchanged.
* :func:`signal_check` -- model-free: per programme, the Spearman of the receivers' summed within-type residual of the response genes
  with their kernel, against the same for the control genes (is the spike there at all at this effect size?).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

KERNELS = ("step", "exp", "field")


@dataclass
class PlantedPair:
    sender: str
    receiver: str
    kernel: str
    length_um: float
    fold: float = 2.0
    n_genes: int = 8
    spike: bool = True

    @property
    def name(self) -> str:
        tag = "" if self.spike else "null:"
        return f"{tag}{self.sender}>{self.receiver}@{self.kernel}{self.length_um:g}x{self.fold:g}"

    @property
    def range_class(self) -> str:
        if self.kernel == "field":
            return "field"
        reach = self.length_um if self.kernel == "step" else self.length_um * np.log(10.0)
        return "contact" if reach <= 25 else "short" if reach <= 100 else "mid" if reach <= 300 else "long"


def _trees(pos, rows):
    from scipy.spatial import cKDTree

    return cKDTree(pos[rows]) if len(rows) else None


def kernel_values(pos_um, graph, codes, pair: PlantedPair, sender_code: int, receiver_code: int, *, max_senders: int = 20):
    """``(f [N], edges DataFrame)``: the kernel at every receiver-type cell (0 elsewhere), and per receiver the senders it reads
    (``step``: within ``l``; ``exp``: where the kernel is >= 0.1; ``field``: the ``max_senders`` nearest within ``3 l``), with their
    kernel weight, as row positions. The field's scale (the median summed weight) is taken over every graph at once."""
    if pair.kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {pair.kernel!r}.")
    f = np.zeros(len(pos_um))
    senders_out, receivers_out, weights_out = [], [], []
    length = float(pair.length_um)
    for g in pd.unique(graph):
        rows = np.flatnonzero(graph == g)
        send = rows[codes[rows] == sender_code]
        recv = rows[codes[rows] == receiver_code]
        tree = _trees(pos_um, send)
        if tree is None or not len(recv):
            continue
        if pair.kernel in ("step", "exp"):
            reach = length if pair.kernel == "step" else length * np.log(10.0)
            d, _ = tree.query(pos_um[recv], k=1)
            f[recv] = (d <= length).astype(float) if pair.kernel == "step" else np.exp(-d / length)
            near = tree.query_ball_point(pos_um[recv], r=reach)
            for i, js in zip(recv, near):
                for j in js:
                    dist = float(np.linalg.norm(pos_um[i] - pos_um[send[j]]))
                    senders_out.append(send[j])
                    receivers_out.append(i)
                    weights_out.append(1.0 if pair.kernel == "step" else float(np.exp(-dist / length)))
        else:
            k = min(max_senders, len(send))
            d, j = tree.query(pos_um[recv], k=k, distance_upper_bound=3 * length)
            d, j = np.atleast_2d(d), np.atleast_2d(j)
            if k == 1:
                d, j = d.T, j.T
            w = np.where(np.isfinite(d), np.exp(-d / length), 0.0)
            # the summed weight S over every sender within 3 l (the nearest max_senders alone would make S nearly constant where
            # senders are dense); scaled below, once every graph is in. The truth lists the nearest max_senders.
            for i, js in zip(recv, tree.query_ball_point(pos_um[recv], r=3 * length)):
                if js:
                    f[i] = float(np.exp(-np.linalg.norm(pos_um[send[js]] - pos_um[i], axis=1) / length).sum())
            for r, (ws, js) in enumerate(zip(w, j)):
                for weight, jj in zip(ws, js):
                    if weight > 0 and jj < len(send):
                        senders_out.append(send[jj])
                        receivers_out.append(recv[r])
                        weights_out.append(float(weight))
    if pair.kernel == "field" and (f > 0).any():
        f = 1.0 - np.exp(-np.log(2.0) * f / np.median(f[f > 0]))
    edges = pd.DataFrame({"sender": np.asarray(senders_out, dtype=np.int64), "receiver": np.asarray(receivers_out, dtype=np.int64),
                          "weight": np.asarray(weights_out)})
    return f, edges


def _within_type_moran(values: np.ndarray, pos_um: np.ndarray, graph: np.ndarray, rows: np.ndarray, k: int = 8) -> np.ndarray:
    """Per gene, Moran's I among the given cells, each against its ``k`` nearest of them in its own graph."""
    from scipy.spatial import cKDTree

    num, den = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    for g in pd.unique(graph[rows]):
        cells = rows[graph[rows] == g]
        if len(cells) <= k:
            continue
        z = values[cells] - values[cells].mean(0)
        _, nb = cKDTree(pos_um[cells]).query(pos_um[cells], k=k + 1)
        lag = z[nb[:, 1:]].mean(1)
        num += (z * lag).sum(0)
        den += (z * z).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def choose_genes(
    log_x: np.ndarray,
    counts: np.ndarray,
    codes: np.ndarray,
    pos_um: np.ndarray,
    graph: np.ndarray,
    pair: PlantedPair,
    sender_code: int,
    receiver_code: int,
    *,
    taken: set[int],
    rng: np.random.Generator,
    min_detection: float = 0.2,
    max_cells: int = 20000,
) -> list[int]:
    """``pair.n_genes`` response genes for the pair (see the module docstring); ``taken`` genes are never chosen again."""
    recv = np.flatnonzero(codes == receiver_code)
    send = np.flatnonzero(codes == sender_code)
    if len(recv) > max_cells:
        recv = np.sort(rng.choice(recv, size=max_cells, replace=False))
    detection = (counts[recv] > 0).mean(0)
    mean_r, mean_s = log_x[recv].mean(0), (log_x[send].mean(0) if len(send) else np.zeros(log_x.shape[1]))
    ok = (detection >= min_detection) & (mean_s <= mean_r)
    ok[list(taken)] = False
    candidates = np.flatnonzero(ok)
    if len(candidates) < pair.n_genes:
        raise ValueError(f"{pair.name}: only {len(candidates)} genes qualify (expressed in {pair.receiver}, not higher in {pair.sender}).")
    moran = _within_type_moran(log_x[:, candidates], pos_um, graph, recv)
    order = candidates[np.argsort(np.nan_to_num(np.abs(moran), nan=1.0))]
    pool = order[: max(pair.n_genes, len(order) // 3)]
    return sorted(rng.choice(pool, size=pair.n_genes, replace=False).tolist())


def spike_in(
    adata,
    pairs: list[PlantedPair],
    *,
    label_key: str,
    graph_key: str,
    counts_layer: str = "raw",
    log_layer: str = "log1p_norm",
    seed: int = 0,
    exclude_genes=(),
) -> pd.DataFrame:
    """Add the planted counts to ``adata.layers[counts_layer]`` (and ``X`` when it holds the counts), in place, and write the truth.

    Returns the programme table (one row per pair: genes, receivers, mean kernel, counts added). The log layer is *not* recomputed:
    the caller normalises as its preprocessing does (``make_inputs.py``: ``normalize_total`` to 1e4, ``log1p``). ``exclude_genes``
    (names) are never chosen as response genes: the negative probes, which the runs drop.
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    labels = adata.obs[label_key].astype(str).to_numpy()
    names = sorted(pd.unique(labels))
    codes = np.searchsorted(names, labels)
    graph = adata.obs[graph_key].astype(str).to_numpy()
    pos = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    counts = adata.layers[counts_layer]
    dense = np.asarray(counts.todense()) if sp.issparse(counts) else np.asarray(counts).copy()
    log_x = adata.layers[log_layer]
    log_x = np.asarray(log_x.todense()) if sp.issparse(log_x) else np.asarray(log_x)
    library = dense.sum(1)
    interaction = np.full(adata.n_vars, "", dtype=object)
    role = np.full(adata.n_vars, "background", dtype=object)
    # never a response gene: e.g. the negative probes, which the runs drop
    taken: set[int] = {int(adata.var_names.get_loc(g)) for g in exclude_genes}
    edges_all, rows = [], []
    for pair in pairs:
        if pair.sender not in names or pair.receiver not in names:
            raise ValueError(f"{pair.name}: label not in {label_key} ({names}).")
        s, r = names.index(pair.sender), names.index(pair.receiver)
        genes = choose_genes(log_x, dense, codes, pos, graph, pair, s, r, taken=taken, rng=rng)
        taken |= set(genes)
        f, edges = kernel_values(pos, graph, codes, pair, s, r)
        recv = np.flatnonzero(codes == r)
        mu = dense[recv][:, genes].mean(0)
        size = library[recv] / max(library[recv].mean(), 1e-9)
        added = 0
        if pair.spike:
            lam = (pair.fold - 1.0) * f[recv][:, None] * mu[None, :] * size[:, None]
            extra = rng.poisson(lam)
            dense[np.ix_(recv, genes)] += extra
            added = int(extra.sum())
        interaction[genes] = pair.name
        role[genes] = "response" if pair.spike else "control"
        edges["program"] = pair.name
        edges["range_class"] = pair.range_class
        edges["slide"] = graph[edges["receiver"].to_numpy()] if len(edges) else []
        edges_all.append(edges)
        adata.obs[f"spike_{pair.name}"] = f
        rows.append({**asdict(pair), "program": pair.name, "range_class": pair.range_class,
                     "genes": ", ".join(adata.var_names[genes]), "receivers_hit": int((f[recv] > 0.1).sum()),
                     "receivers": len(recv), "mean_kernel": float(f[recv].mean()), "counts_added": added})
    adata.layers[counts_layer] = sp.csr_matrix(dense) if sp.issparse(counts) else dense
    if counts_layer != "X" and adata.X is not None and adata.X.shape == dense.shape:
        adata.X = adata.layers[counts_layer].copy()
    adata.var["interaction"] = interaction
    adata.var["role"] = role
    edges = pd.concat(edges_all, ignore_index=True) if edges_all else pd.DataFrame()
    for column in ("program", "range_class", "slide"):
        if column in edges:
            edges[column] = pd.Categorical(edges[column].astype(str))
    programmes = pd.DataFrame(rows)
    adata.uns["synthetic"] = {"interaction_edges": edges, "programmes": programmes.astype({"genes": str})}
    adata.uns.setdefault("preprocessing", {})["n_spots"] = int(adata.n_obs)
    return programmes


def signal_check(adata, programmes: pd.DataFrame, *, label_key: str, graph_key: str, log_layer: str = "log1p_norm", rows=None) -> pd.DataFrame:
    """Model-free: per programme, Spearman over its receivers (``rows``: e.g. the test sections) of the summed within-type residual
    of its genes with its kernel. A spiked programme should rise with its effect size; a control should sit at 0."""
    import scipy.sparse as sp
    from scipy.stats import spearmanr

    x = adata.layers[log_layer]
    x = np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)
    labels = adata.obs[label_key].astype(str).to_numpy()
    graph = adata.obs[graph_key].astype(str).to_numpy()
    keep = np.ones(adata.n_obs, dtype=bool) if rows is None else np.asarray(rows)
    out = []
    for _, prog in programmes.iterrows():
        genes = [adata.var_names.get_loc(g.strip()) for g in str(prog["genes"]).split(",") if g.strip()]
        recv = np.flatnonzero((labels == prog["receiver"]) & keep)
        if len(recv) < 10:
            continue
        resid = x[recv][:, genes].sum(1)
        for g in pd.unique(graph[recv]):  # within section x type
            sel = graph[recv] == g
            resid[sel] -= resid[sel].mean()
        f = adata.obs[f"spike_{prog['program']}"].to_numpy()[recv]
        rho = spearmanr(resid, f).statistic if np.std(f) > 0 else np.nan
        out.append({"program": prog["program"], "spike": bool(prog["spike"]), "receivers": len(recv), "spearman_resid_kernel": float(rho)})
    return pd.DataFrame(out)
