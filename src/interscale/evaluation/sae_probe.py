"""SAFFRON's way of reading an embedding, for InterScale's embeddings (``.claude/ccc_model_plan.md``, E1).

Handa et al. (bioRxiv 2026) decompose a dense embedding with a Matryoshka BatchTopK sparse autoencoder and ask whether any single
sparse feature tracks a known spatial variable -- a 1-D gradient (an isodepth, a position-only coordinate) or a local
microenvironment (density of a cell type, distance to a structure) -- within each spatial domain, against a permutation null and
naive baselines (PCA, an SAE on the expression). Here the same, on any embedding of any dataset (variables from a spec, never
hard-coded):

* :class:`MatryoshkaSAE` / :func:`train_sae` -- ReLU encoder, BatchTopK sparsity (a batch keeps its ``K x batch`` largest
  activations; inference keeps activations above a threshold learned in training), the loss the mean over prefixes ``M/4, M/2,
  M`` of each prefix's reconstruction error (early features must rebuild the embedding alone: coarse; later ones refine), dead
  features resampled every 10 epochs, input noise annealed to 0 (SAFFRON App. B). Inputs are standardised on the training rows.
* :func:`spatial_variables` -- the variables from a list of specs (kinds: ``obs``, ``layer``, ``distance_to_label`` with
  ``signed``, ``density_of_label``, ``density_near_label``), per graph, from positions and labels only unless the spec says otherwise.
* :func:`best_features` -- per variable: max |Spearman| over features within each domain, averaged over domains (SAFFRON's
  score, favourable to the embedding by design), its 95th percentile under within-domain permutations of the variable, the
  feature that wins most domains, and the best |Spearman| within each prefix block (early / middle / late features).
* :func:`omp_curve` -- how few features rebuild the variable: cross-validated R^2 of orthogonal matching pursuit with 1..k
  features (and of a ridge on all of them).
* :func:`feature_scales` -- each feature's activation map's length scale (Moran's I decay, :mod:`gene_scales`) against its index:
  if the nesting orders features by scale (wide first), the Spearman of index and scale is negative.
* :func:`feature_genes` -- each feature's genes through a linear decoder of the embedding (the model's own).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


# --------------------------------------------------------------------------- the sparse autoencoder


class MatryoshkaSAE(nn.Module):
    """``h = TopK_batch(ReLU(W_enc (z - b_dec) + b_enc))``, ``z_hat^(r) = W_dec[:, :M_r] h[:M_r] + b_dec`` for each prefix ``M_r``."""

    def __init__(self, dim: int, n_features: int, k: int, prefixes: tuple[int, ...] | None = None, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(int(seed))
        self.dim, self.n_features, self.k = int(dim), int(n_features), int(k)
        self.prefixes = tuple(prefixes or (n_features // 4, n_features // 2, n_features))
        if self.prefixes[-1] != self.n_features or list(self.prefixes) != sorted(self.prefixes):
            raise ValueError(f"prefixes must rise to n_features ({n_features}), got {self.prefixes}.")
        w = torch.randn(self.dim, self.n_features, generator=gen)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(w)
        self.W_enc = nn.Parameter(w.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.b_dec = nn.Parameter(torch.zeros(self.dim))
        self.register_buffer("threshold", torch.zeros(()))

    def pre(self, z: torch.Tensor) -> torch.Tensor:
        return torch.relu((z - self.b_dec) @ self.W_enc.T + self.b_enc)

    def encode(self, z: torch.Tensor, batch_topk: bool) -> torch.Tensor:
        acts = self.pre(z)
        if batch_topk:
            flat = acts.flatten()
            keep = min(flat.numel(), self.k * z.shape[0])
            values, idx = flat.topk(keep)
            out = torch.zeros_like(flat).scatter(0, idx, values).view_as(acts)
            if self.training and keep:
                smallest = values[-1].detach()
                self.threshold.mul_(0.99).add_(0.01 * smallest)
            return out
        return acts * (acts > self.threshold)

    def decode(self, h: torch.Tensor, prefix: int | None = None) -> torch.Tensor:
        if prefix is None:
            return h @ self.W_dec.T + self.b_dec
        return h[:, :prefix] @ self.W_dec[:, :prefix].T + self.b_dec

    def loss(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(z, batch_topk=True)
        losses = [((self.decode(h, p) - z) ** 2).sum(1).mean() for p in self.prefixes]
        return torch.stack(losses).mean(), h


@dataclass
class FittedSAE:
    """A trained :class:`MatryoshkaSAE` and the standardisation it was trained under."""

    sae: MatryoshkaSAE
    mean: np.ndarray
    scale: np.ndarray
    r2: float
    dead: float

    @torch.no_grad()
    def features(self, z: np.ndarray, batch: int = 8192) -> np.ndarray:
        self.sae.eval()
        device = self.sae.W_enc.device
        x = (np.asarray(z, dtype=np.float32) - self.mean) / self.scale
        out = [self.sae.encode(torch.as_tensor(x[i : i + batch], device=device), batch_topk=False).cpu().numpy() for i in range(0, len(x), batch)]
        return np.concatenate(out) if out else np.zeros((0, self.sae.n_features), dtype=np.float32)

    def directions(self) -> np.ndarray:
        """``[D, M]``: each feature's decoder direction in the embedding's own units (the standardisation undone)."""
        return self.sae.W_dec.detach().cpu().numpy() * self.scale[:, None]


def train_sae(
    z: np.ndarray,
    *,
    expansion: int = 4,
    k_fraction: float = 0.0625,
    n_features: int | None = None,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    noise: float = 0.02,
    resample_every: int = 10,
    seed: int = 0,
    device: str | None = None,
) -> FittedSAE:
    """Train a Matryoshka BatchTopK SAE on the rows ``z`` ``[N, D]`` (SAFFRON's settings: ``M = expansion * D``, ``K / M`` about 6 %,
    prefixes ``M/4, M/2, M``, Adam 1e-3, batch 512, dead features resampled every 10 epochs, noise annealed from ``noise`` to 0).

    ``n_features`` fixes ``M`` instead (SAFFRON's matched-dictionary control: the chance ceiling of a max over features grows with
    ``M``, so embeddings are compared at one ``M``)."""
    z = np.asarray(z, dtype=np.float32)
    mean, scale = z.mean(0), z.std(0) + 1e-6
    x = torch.as_tensor((z - mean) / scale)
    dim = z.shape[1]
    m = int(n_features or expansion * dim)
    k = max(1, int(round(k_fraction * m)))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = MatryoshkaSAE(dim, m, k, seed=seed).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    fired = torch.zeros(m, dtype=torch.bool, device=device)
    for epoch in range(int(epochs)):
        sae.train()
        sigma = noise * max(0.0, 1.0 - epoch / max(1, epochs - 1))
        order = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), batch_size):
            batch = x[order[i : i + batch_size]].to(device)
            if sigma > 0:
                batch = batch + sigma * torch.randn(batch.shape, generator=gen).to(device)
            loss, h = sae.loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.W_dec.data /= sae.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
            fired |= (h > 0).any(0)
        if resample_every and (epoch + 1) % resample_every == 0 and epoch + 1 < epochs:
            dead = (~fired).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    sample = x[torch.randint(len(x), (len(dead),), generator=gen)].to(device)
                    residual = sample - sae.decode(sae.encode(sample, batch_topk=False))
                    directions = residual / residual.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    sae.W_dec.data[:, dead] = directions.T
                    sae.W_enc.data[dead] = directions * 0.2
                    sae.b_enc.data[dead] = 0.0
            fired.zero_()
    sae.eval()
    with torch.no_grad():
        full = x.to(device)
        h = torch.cat([sae.encode(full[i : i + 8192], batch_topk=False) for i in range(0, len(full), 8192)])
        rec = torch.cat([sae.decode(h[i : i + 8192]) for i in range(0, len(h), 8192)])
        r2 = float(1.0 - ((rec - full) ** 2).sum() / ((full - full.mean(0)) ** 2).sum())
        dead_share = float(((h > 0).sum(0) == 0).float().mean())
    return FittedSAE(sae, mean, scale, r2, dead_share)


# --------------------------------------------------------------------------- known spatial variables


def _knn_tree(points: np.ndarray):
    from scipy.spatial import cKDTree

    return cKDTree(points) if len(points) else None


def spatial_variables(
    pos_um: np.ndarray,
    graph: np.ndarray,
    specs: list[dict],
    *,
    obs: pd.DataFrame | None = None,
    layers: dict[str, np.ndarray] | None = None,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """``[N, variables]``: each spec's variable for every cell, per graph (neighbours never across graphs).

    Spec keys: ``name``, ``kind`` and the kind's own --

    * ``obs``: ``column`` (numeric obs column, e.g. synth_spot's ``dist_M7``, a FOV index);
    * ``layer``: ``layer``, ``gene`` (e.g. synth_spot's noise-free ``expected`` response: the planted field itself);
    * ``distance_to_label``: ``column``, ``values`` -- distance to the nearest cell with one of the labels (um); ``signed``: a cell
      with the label gets minus its distance to the nearest cell without it (a boundary coordinate: an islet's inside negative);
    * ``density_of_label``: ``column``, ``values``, ``radius_um`` -- share of the cells within the radius that carry the label;
    * ``density_near_label``: as ``density_of_label``, but NaN for cells farther than ``near_um`` from a cell with the labels
      ``near_values`` (e.g. immune density around islets only: insulitis).
    """
    n = len(pos_um)
    out = {}
    groups = [np.flatnonzero(graph == g) for g in pd.unique(graph)]
    for spec in specs:
        name, kind = str(spec["name"]), str(spec["kind"])
        values = np.full(n, np.nan, dtype=np.float64)
        if kind == "obs":
            # Through strings: a categorical column (sweeps make context labels categorical) or a boolean one reads as numbers too.
            text = obs[spec["column"]].astype(str)
            numbers = pd.to_numeric(text, errors="coerce")
            if numbers.isna().all():
                numbers = text.map({"True": 1.0, "False": 0.0})
            values = numbers.to_numpy(dtype=np.float64)
        elif kind == "layer":
            layer = layers[spec["layer"]]
            column = list(var_names).index(spec["gene"])
            values = np.asarray(layer[:, column]).ravel().astype(np.float64)
        elif kind in ("distance_to_label", "density_of_label", "density_near_label"):
            labels = obs[spec["column"]].astype(str).to_numpy()
            member = np.isin(labels, [str(v) for v in spec["values"]])
            near_member = None
            if kind == "density_near_label":
                near_member = np.isin(labels, [str(v) for v in spec["near_values"]])
            for rows in groups:
                p = pos_um[rows]
                inside = member[rows]
                if kind == "distance_to_label":
                    tree = _knn_tree(p[inside])
                    if tree is None:
                        continue
                    d, _ = tree.query(p, k=1)
                    if spec.get("signed", False):
                        outside = _knn_tree(p[~inside])
                        if outside is not None:
                            d_out, _ = outside.query(p[inside], k=1)
                            d = d.astype(np.float64)
                            d[inside] = -d_out
                    values[rows] = d
                else:
                    tree_all = _knn_tree(p)
                    radius = float(spec["radius_um"])
                    counts = np.array([len(c) for c in tree_all.query_ball_point(p, r=radius)], dtype=np.float64)
                    tree_in = _knn_tree(p[inside])
                    hits = (
                        np.array([len(c) for c in tree_in.query_ball_point(p, r=radius)], dtype=np.float64)
                        if tree_in is not None
                        else np.zeros(len(rows))
                    )
                    dens = hits / np.maximum(counts, 1.0)
                    if near_member is not None:
                        near_tree = _knn_tree(p[near_member[rows]])
                        if near_tree is None:
                            dens[:] = np.nan
                        else:
                            d, _ = near_tree.query(p, k=1)
                            dens[d > float(spec["near_um"])] = np.nan
                    values[rows] = dens
        else:
            raise ValueError(f"unknown spatial variable kind {kind!r} (spec {spec}).")
        out[name] = values
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- scores


def _rank_columns(x: np.ndarray) -> np.ndarray:
    """Column-wise ranks (average ties), centred and unit norm: a Spearman correlation is then a dot product."""
    from scipy.stats import rankdata

    r = rankdata(x, axis=0, method="average").astype(np.float64)
    r -= r.mean(0, keepdims=True)
    norm = np.linalg.norm(r, axis=0, keepdims=True)
    norm[norm == 0] = np.inf
    return r / norm


def best_features(
    features: np.ndarray,
    variables: pd.DataFrame,
    domains: np.ndarray,
    *,
    prefixes: tuple[int, ...],
    n_permutations: int = 200,
    max_cells: int = 4000,
    min_cells: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Per variable: SAFFRON's score and its null, on ``features`` ``[N, M]`` (prefix blocks from ``prefixes``).

    Within each domain (cells with a finite variable, at most ``max_cells`` drawn), the max over features of |Spearman|; the score is
    its mean over domains. ``null95``: the 95th percentile of the same mean with the variable permuted within each domain
    (``n_permutations`` draws). ``feature``: the feature that is the best in most domains; ``block_<b>``: the score within prefix
    block ``b`` (0 the first ``M/4``: the coarse features). Constant features are left out (a feature that never fires has no rank).
    """
    rng = np.random.default_rng(seed)
    blocks = [(0, prefixes[0])] + [(prefixes[i], prefixes[i + 1]) for i in range(len(prefixes) - 1)]
    rows = []
    for name in variables.columns:
        tau = variables[name].to_numpy(dtype=np.float64)
        per_domain, null_draws, winners, block_scores = [], [], [], [[] for _ in blocks]
        for d in pd.unique(domains):
            cells = np.flatnonzero((domains == d) & np.isfinite(tau))
            if len(cells) < min_cells or np.nanstd(tau[cells]) == 0:
                continue
            if len(cells) > max_cells:
                cells = rng.choice(cells, size=max_cells, replace=False)
            f = features[cells]
            active = f.std(0) > 0
            if not active.any():
                continue
            ranks = _rank_columns(f[:, active])
            t = _rank_columns(tau[cells][:, None])[:, 0]
            rho = np.abs(t @ ranks)
            per_domain.append(rho.max())
            winners.append(np.flatnonzero(active)[int(rho.argmax())])
            index = np.flatnonzero(active)
            for b, (lo, hi) in enumerate(blocks):
                inside = (index >= lo) & (index < hi)
                block_scores[b].append(rho[inside].max() if inside.any() else np.nan)
            perm = np.stack([rng.permutation(t) for _ in range(int(n_permutations))])  # [P, n]
            null_draws.append(np.abs(perm @ ranks).max(1))
        if not per_domain:
            continue
        null = np.mean(np.stack(null_draws), axis=0)
        values, counts = np.unique(winners, return_counts=True)
        row = {
            "variable": name,
            "score": float(np.mean(per_domain)),
            "null95": float(np.quantile(null, 0.95)),
            "domains": len(per_domain),
            "feature": int(values[counts.argmax()]),
        }
        for b in range(len(blocks)):
            row[f"block_{b}"] = float(np.nanmean(block_scores[b])) if block_scores[b] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def omp_curve(features: np.ndarray, tau: np.ndarray, groups: np.ndarray, *, max_features: int = 15, folds: int = 5) -> dict:
    """Cross-validated R^2 (folds by ``groups``: graphs or spatial blocks) of OMP with 1..``max_features`` features, and of a ridge
    on all of them: ``{"omp": [R^2 per k], "ridge": R^2}`` (SAFFRON Fig. 3b)."""
    from sklearn.linear_model import OrthogonalMatchingPursuit, RidgeCV
    from sklearn.model_selection import GroupKFold

    keep = np.isfinite(tau)
    x, y, g = features[keep], tau[keep], np.asarray(groups)[keep]
    active = x.std(0) > 0
    x = x[:, active]
    n_groups = len(np.unique(g))
    if n_groups < 2 or x.shape[1] == 0:
        return {"omp": [np.nan] * max_features, "ridge": np.nan}
    splitter = GroupKFold(n_splits=min(folds, n_groups))
    ks = list(range(1, min(max_features, x.shape[1]) + 1))
    pred = {k: np.zeros_like(y) for k in ks}
    ridge = np.zeros_like(y)
    for train, test in splitter.split(x, y, g):
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
        xt, xs = (x[train] - mu) / sd, (x[test] - mu) / sd
        for k in ks:
            model = OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(xt, y[train])
            pred[k][test] = model.predict(xs)
        ridge[test] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xt, y[train]).predict(xs)
    total = ((y - y.mean()) ** 2).sum()
    curve = [float(1 - ((pred[k] - y) ** 2).sum() / total) for k in ks] + [np.nan] * (max_features - len(ks))
    return {"omp": curve, "ridge": float(1 - ((ridge - y) ** 2).sum() / total)}


def feature_scales(features: np.ndarray, pos_um: np.ndarray, graph: np.ndarray, *, k_max: int = 256, n_centres: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per feature: the length scale of its activation map (``gene_scales.length_scales``: where its Moran's I shell curve halves,
    against a permutation null) and its index; features that never fire are left out."""
    from interscale.evaluation.gene_scales import length_scales, moran_curves, permutation_null, sample_neighbours

    active = np.flatnonzero(features.std(0) > 0)
    if not len(active):
        return pd.DataFrame()
    nb = sample_neighbours(pos_um, graph, k_max=k_max, n_centres=n_centres, seed=seed)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= k_max]
    values = features[:, active].astype(np.float64)
    curves = moran_curves(values, graph, nb, ks)
    null = permutation_null(values, graph, nb, ks, n_permutations=4, seed=seed)
    table = length_scales(curves, null, [f"f{i}" for i in active])
    table["feature"] = active
    return table


def index_scale_test(scales: pd.DataFrame, prefixes: tuple[int, ...]) -> dict:
    """Do early features vary on wider scales? Spearman of feature index and ``log ell_um`` over the spatially variable features
    (``inf`` -- wider than the largest neighbourhood -- ranked last), and the median ``ell_um`` per prefix block."""
    from scipy.stats import spearmanr

    if scales.empty or "ell_um" not in scales:
        return {"spearman": np.nan, "features": 0}
    variable = scales[np.isfinite(scales["ell_um"]) | np.isinf(scales["ell_um"])].dropna(subset=["ell_um"])
    out = {"features": int(len(variable))}
    if len(variable) >= 5:
        ell = variable["ell_um"].replace(np.inf, variable["ell_um"][np.isfinite(variable["ell_um"])].max() * 10 if np.isfinite(variable["ell_um"]).any() else 1e9)
        out["spearman"] = float(spearmanr(variable["feature"], np.log(ell)).statistic)
    else:
        out["spearman"] = np.nan
    blocks = [(0, prefixes[0])] + [(prefixes[i], prefixes[i + 1]) for i in range(len(prefixes) - 1)]
    for b, (lo, hi) in enumerate(blocks):
        part = variable[(variable["feature"] >= lo) & (variable["feature"] < hi)]["ell_um"]
        out[f"median_ell_block_{b}"] = float(np.median(part)) if len(part) else np.nan
    return out


def feature_genes(fitted: FittedSAE, decoder_weight: np.ndarray, gene_names: list[str], features: list[int], top: int = 8) -> pd.DataFrame:
    """Each feature's genes through a linear decoder ``[G, D]`` of the embedding: the decoder applied to the feature's direction;
    the ``top`` genes by loading (signed)."""
    directions = fitted.directions()  # [D, M]
    rows = []
    for f in features:
        loading = decoder_weight @ directions[:, f]
        order = np.argsort(-np.abs(loading))[:top]
        rows.append({"feature": int(f), "genes": ", ".join(f"{gene_names[g]}{'+' if loading[g] > 0 else '-'}" for g in order)})
    return pd.DataFrame(rows)


def probe_embedding(
    name: str,
    z_train: np.ndarray,
    z_test: np.ndarray,
    variables: pd.DataFrame,
    domains: np.ndarray,
    groups: np.ndarray,
    pos_um: np.ndarray,
    graph: np.ndarray,
    *,
    n_features: int = 256,
    k_fraction: float = 0.0625,
    epochs: int = 100,
    seed: int = 0,
    decoder_weight: np.ndarray | None = None,
    gene_names: list[str] | None = None,
    n_permutations: int = 200,
    scales: bool = True,
) -> dict[str, pd.DataFrame]:
    """E1 for one embedding: the SAE on the training rows, then on the test rows (``variables``, ``domains``, ``groups``, ``pos_um``,
    ``graph`` all in test-row order): ``sae_best`` (per variable: score, null95, feature, blocks, OMP R^2 at 1 / 2 / 5 / 15 features,
    ridge), ``sae_fit`` (reconstruction R^2, dead share), ``sae_scales`` (index-scale test), ``sae_genes`` (the winning features'
    genes, when a decoder is given)."""
    fitted = train_sae(z_train, n_features=n_features, k_fraction=k_fraction, epochs=epochs, seed=seed)
    features = fitted.features(z_test)
    prefixes = fitted.sae.prefixes
    best = best_features(features, variables, domains, prefixes=prefixes, n_permutations=n_permutations, seed=seed)
    for i, row in best.iterrows():
        curve = omp_curve(features, variables[row["variable"]].to_numpy(dtype=np.float64), groups)
        for k in (1, 2, 5, 15):
            best.loc[i, f"omp_r2_{k}"] = curve["omp"][k - 1] if k - 1 < len(curve["omp"]) else np.nan
        best.loc[i, "ridge_r2"] = curve["ridge"]
    best.insert(0, "embedding", name)
    out = {
        "sae_best": best,
        "sae_fit": pd.DataFrame([{"embedding": name, "n_features": fitted.sae.n_features, "k": fitted.sae.k, "r2": fitted.r2, "dead": fitted.dead}]),
    }
    if scales:
        table = feature_scales(features, pos_um, graph, seed=seed)
        test = index_scale_test(table, prefixes)
        out["sae_scales"] = pd.DataFrame([{"embedding": name, **test}])
    if decoder_weight is not None and gene_names is not None and len(best):
        genes = feature_genes(fitted, decoder_weight, gene_names, sorted(set(best["feature"].astype(int))))
        genes.insert(0, "embedding", name)
        out["sae_genes"] = genes
    out["_features"] = features
    return out
