"""Downstream task 2: how much of the attention is *not* explained by trivial structure.

Regress the attention weight of a cell pair on distance, cell-type pair identity, region/niche,
total counts and the remaining QC covariates, then report how much variance is left. Attention
that is fully predicted by a distance kernel plus cell-type composition tells you the global
module has learned a spatial prior, not interactions. The residual is the part worth interpreting
biologically -- and on the synthetic dataset it can be scored directly, because the pairs that
truly influence each other are recorded in ``uns['synthetic']['interaction_edges']``.

The covariates are added in blocks and the fit is repeated cumulatively, so the output is an
incremental R^2 per block rather than one opaque number: distance first (it is the strongest and
least interesting predictor), then cell-type pair, niche, QC, and finally slide/condition.

Two mechanics worth knowing:

* ``adata.obsm[f'{prefix}_attn_matrix']`` is ``[n_cells, max_seq_len]`` where column ``j`` is the
  *token position inside that cell's own graph*. With one graph per slide and no truncation, token
  order is the order the cells appear in ``adata``, which is what :func:`attention_pairs`
  reconstructs. It refuses to guess when a graph was longer than ``max_seq_len``, because
  :func:`interscale.tl.padding.pad_batch` then keeps a random subset of tokens and the mapping is
  gone. If ``uns[f'{prefix}_attn_index']`` exists it is used instead of the order convention.
* Relevance is softmax-normalised per graph, so raw values are not comparable between graphs of
  different size. ``normalize='graph_z'`` (the default) centres and scales each graph's block,
  which fixes that without touching the structure *within* a graph. ``'row_rank'`` and ``'row_z'``
  are more aggressive: they normalise each row, which by construction removes any effect that is
  constant within a row -- i.e. every purely receiver-side effect, including a receiver cell type
  that simply attracts more attention overall. Use them when attention sinks dominate, but read
  the cell-type block as "sender-side and pairwise only" afterwards.

Run as a script::

    python src/interscale/evaluation/downstream_regression.py \
        --h5ad results/synth_data_0_model_output.h5ad --sample-key slide
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

NORMALIZATIONS = ("none", "row_rank", "row_z", "graph_z")

ESTIMATORS = {
    "ridge": lambda seed: Ridge(alpha=1.0, random_state=seed),
    "gbm": lambda seed: HistGradientBoostingRegressor(random_state=seed),
}


def _graph_members(adata: AnnData, sample_key: str, prefix: str, n_tokens: int) -> dict[str, np.ndarray]:
    """Positional indices of the cells of each graph, in the order the transformer saw them.

    Raises
    ------
    ValueError
        If a graph has more cells than the attention matrix has columns. ``pad_batch`` then kept a
        random subset of tokens, so no token-to-cell mapping can be recovered after the fact --
        either lower the cells per graph or raise ``max_seq_len`` and re-run inference.
    """
    index_key = f"{prefix}_attn_index"
    if index_key in adata.uns:
        # Explicit mapping written at inference time; always preferred over the order convention.
        stored = adata.uns[index_key]
        name_to_pos = {n: i for i, n in enumerate(adata.obs_names)}
        return {str(k): np.array([name_to_pos[str(n)] for n in v]) for k, v in stored.items()}

    members = {}
    codes = adata.obs[sample_key].astype(str).to_numpy()
    for g in pd.unique(codes):
        idx = np.flatnonzero(codes == g)
        if len(idx) > n_tokens:
            raise ValueError(
                f"graph '{g}' has {len(idx)} cells but the attention matrix has {n_tokens} columns: "
                "pad_batch subsampled tokens at random, so the token-to-cell mapping is lost. "
                "Raise max_seq_len (or lower the cells per graph) and re-run get_model_output."
            )
        members[g] = idx
    return members


def _normalize_rows(block: np.ndarray, how: str) -> np.ndarray:
    """Put one graph's attention block on a comparable scale."""
    if how == "none":
        return block
    if how == "row_rank":
        order = block.argsort(axis=1).argsort(axis=1).astype(np.float64)
        return order / max(block.shape[1] - 1, 1)
    if how == "row_z":
        mu = block.mean(1, keepdims=True)
        sd = block.std(1, keepdims=True)
        return (block - mu) / np.where(sd > 0, sd, 1.0)
    if how == "graph_z":
        sd = block.std()
        return (block - block.mean()) / (sd if sd > 0 else 1.0)
    raise ValueError(f"normalize must be one of {NORMALIZATIONS}")


def attention_pairs(
    adata: AnnData,
    *,
    prefix: str = "combined",
    sample_key: str = "slide",
    obs_features: tuple[str, ...] = ("cell_type", "niche", "total_counts", "n_genes_by_counts"),
    normalize: str = "graph_z",
    max_pairs: int = 200_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Turn the per-cell attention matrix into a long table of cell pairs with covariates.

    Pairs are sampled uniformly at random, which keeps the rate of truly-interacting pairs intact
    and therefore leaves both the regression and the ROC scoring unbiased.

    Parameters
    ----------
    adata
        Object written by ``CombinedModel.get_model_output``.
    prefix
        Prefix that output was written under.
    sample_key
        ``.obs`` column defining one graph / transformer sequence (``slide`` here, a window key in
        datasets that use sliding windows).
    obs_features
        ``.obs`` columns to attach. Each becomes ``<col>_i`` (query cell) and ``<col>_j`` (key
        cell); categorical columns additionally get ``pair_<col>`` (``"A->B"`` identity) and
        ``same_<col>``. Anything present in ``.obs`` can be named here.
    normalize
        One of :data:`NORMALIZATIONS`; see the module docstring.
    max_pairs
        Total number of pairs to sample across all graphs.
    random_state
        Seed for the pair sampling.

    Returns
    -------
    pandas.DataFrame
        One row per sampled ordered pair, with ``attn`` the (normalised) attention of query ``i``
        on key ``j``, ``dist`` their euclidean distance, and the requested covariates.
    """
    key = f"{prefix}_attn_matrix"
    if key not in adata.obsm:
        raise KeyError(f"'{key}' not in adata.obsm -- run CombinedModel.get_model_output first")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}")

    attn = np.asarray(adata.obsm[key], dtype=np.float64)
    members = _graph_members(adata, sample_key, prefix, attn.shape[1])
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    rng = np.random.default_rng(random_state)

    total = sum(len(v) * (len(v) - 1) for v in members.values())
    frames = []
    for graph, idx in members.items():
        n = len(idx)
        if n < 2:
            continue
        block = _normalize_rows(attn[np.ix_(idx, np.arange(n))], normalize)

        share = int(round(max_pairs * n * (n - 1) / total)) if total else 0
        share = max(min(share, n * (n - 1)), 1)
        # Oversample then drop the diagonal: cheaper than enumerating n^2 pairs for large graphs.
        ii = rng.integers(0, n, size=int(share * 1.3) + 16)
        jj = rng.integers(0, n, size=len(ii))
        keep = ii != jj
        ii, jj = ii[keep][:share], jj[keep][:share]

        value = block[ii, jj]
        good = np.isfinite(value)
        ii, jj, value = ii[good], jj[good], value[good]

        gi, gj = idx[ii], idx[jj]
        frames.append(
            pd.DataFrame(
                {
                    "graph": graph,
                    "i": gi,
                    "j": gj,
                    "attn": value,
                    "dist": np.linalg.norm(coords[gi] - coords[gj], axis=1),
                }
            )
        )

    pairs = pd.concat(frames, ignore_index=True)

    for col in obs_features:
        if col not in adata.obs:
            raise KeyError(f"'{col}' not in adata.obs")
        values = adata.obs[col]
        vi = values.to_numpy()[pairs["i"].to_numpy()]
        vj = values.to_numpy()[pairs["j"].to_numpy()]
        pairs[f"{col}_i"] = vi
        pairs[f"{col}_j"] = vj
        if not pd.api.types.is_numeric_dtype(values):
            # The directed pair identity is the covariate the question is about: "is this pair's
            # attention explained by what the two cells are".
            pairs[f"pair_{col}"] = pd.Categorical([f"{b}->{a}" for a, b in zip(vi, vj, strict=True)])
            pairs[f"same_{col}"] = (vi == vj).astype(float)
    return pairs


def default_blocks(pairs: pd.DataFrame, batch_features: tuple[str, ...] = ("condition", "graph")) -> dict[str, list]:
    """The covariate blocks, ordered least to most interesting, from what the table actually has.

    Distance goes first deliberately: it is the strongest predictor and the least informative one,
    so everything after it is measured as "beyond a distance kernel".
    """
    blocks: dict[str, list] = {"distance": ["dist"]}
    if "pair_cell_type" in pairs:
        blocks["celltype_pair"] = ["pair_cell_type"]
    niche = [c for c in ("pair_niche", "same_niche") if c in pairs]
    if niche:
        blocks["niche"] = niche
    qc = [
        c
        for c in pairs.columns
        if c.startswith(("total_counts_", "n_genes_by_counts_", "lib_factor_", "pct_counts_"))
    ]
    if qc:
        blocks["qc"] = qc
    batch = [c for c in batch_features if c in pairs]
    if batch:
        blocks["batch"] = batch
    return blocks


def _design(columns: list[str], pairs: pd.DataFrame, n_knots: int) -> ColumnTransformer:
    """Spline basis for distance, standardisation for other numerics, one-hot for categoricals."""
    transformers = []
    for col in columns:
        if col == "dist":
            # A linear distance term underfits badly and dumps its own misspecification into the
            # residual, which is exactly the quantity being interpreted -- hence splines.
            transformers.append((col, SplineTransformer(n_knots=n_knots, degree=3), [col]))
        elif pd.api.types.is_numeric_dtype(pairs[col]):
            transformers.append((col, StandardScaler(), [col]))
        else:
            transformers.append((col, OneHotEncoder(handle_unknown="ignore", sparse_output=False), [col]))
    return ColumnTransformer(transformers, remainder="drop")


@dataclass
class AttentionRegressionResult:
    """Output of :func:`regress_attention`."""

    variance: pd.DataFrame
    pairs: pd.DataFrame
    nonlinear_r2: float

    def __repr__(self) -> str:
        return (
            f"AttentionRegressionResult(n_pairs={len(self.pairs)}, "
            f"full_r2={self.variance['cumulative_r2'].iloc[-1]:.3f}, "
            f"residual={self.variance['residual_variance'].iloc[-1]:.3f}, "
            f"nonlinear_r2={self.nonlinear_r2:.3f})"
        )


def regress_attention(
    pairs: pd.DataFrame,
    *,
    blocks: dict[str, list] | None = None,
    estimator: str = "ridge",
    n_knots: int = 8,
    group_key: str = "graph",
    cv_folds: int = 5,
    nonlinear_reference: bool = True,
    random_state: int = 0,
) -> AttentionRegressionResult:
    """Cumulative variance decomposition of attention over covariate blocks.

    Every fit is scored out-of-fold with the folds grouped by ``group_key``, so a block cannot
    earn R^2 by memorising a graph.

    Parameters
    ----------
    pairs
        Output of :func:`attention_pairs`.
    blocks
        Ordered ``{block_name: [columns]}``. Defaults to :func:`default_blocks`; pass your own to
        add or reorder covariates -- any column of ``pairs`` is allowed.
    estimator
        ``"ridge"`` (the interpretable default) or ``"gbm"``.
    n_knots
        Knots of the distance spline basis.
    group_key
        Column whose levels must not be split across CV folds.
    cv_folds
        Number of CV folds.
    nonlinear_reference
        Also fit a gradient-boosted model on the full covariate set. The gap to the linear model
        says how much of the "residual" is just functional-form misspecification rather than
        biology.
    random_state
        Seed.

    Returns
    -------
    AttentionRegressionResult
        ``variance`` with one row per block, ``pairs`` with the full model's out-of-fold
        prediction and residual attached, and the nonlinear reference R^2.
    """
    blocks = blocks or default_blocks(pairs)
    y = pairs["attn"].to_numpy()
    groups = pairs[group_key].astype(str).to_numpy()
    n_splits = min(cv_folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError(f"need at least 2 levels of '{group_key}' for grouped CV")
    cv = GroupKFold(n_splits=n_splits)

    rows, used, prev_r2, oof_full = [], [], 0.0, None
    for name, cols in blocks.items():
        used = used + list(cols)
        model = Pipeline([("design", _design(used, pairs, n_knots)), ("reg", ESTIMATORS[estimator](random_state))])
        oof = cross_val_predict(model, pairs[used], y, cv=cv, groups=groups)
        r2 = r2_score(y, oof)
        rows.append(
            {
                "block": name,
                "added_columns": ", ".join(cols),
                "cumulative_r2": r2,
                "delta_r2": r2 - prev_r2,
                "residual_variance": 1.0 - r2,
            }
        )
        prev_r2, oof_full = r2, oof

    out = pairs.copy()
    out["pred"] = oof_full
    out["resid"] = y - oof_full

    nonlinear_r2 = float("nan")
    if nonlinear_reference:
        model = Pipeline([("design", _design(used, pairs, n_knots)), ("reg", ESTIMATORS["gbm"](random_state))])
        nonlinear_r2 = r2_score(y, cross_val_predict(model, pairs[used], y, cv=cv, groups=groups))

    return AttentionRegressionResult(pd.DataFrame(rows), out, nonlinear_r2)


def residual_flow(pairs: pd.DataFrame, by: str = "cell_type", value: str = "resid") -> pd.DataFrame:
    """Mean residual attention per sender -> receiver cell-type pair.

    Rows are the key (sending) cell type, columns the query (receiving) one, so the matrix reads
    the same way as the net-flow maps in :mod:`interscale.evaluation.net_streams`.
    """
    df = pairs.groupby([f"{by}_j", f"{by}_i"], observed=True)[value].mean().unstack()
    df.index.name = f"sender_{by}"
    df.columns.name = f"receiver_{by}"
    return df


def score_against_truth(
    adata: AnnData,
    pairs: pd.DataFrame,
    *,
    score_columns: tuple[str, ...] = ("attn", "resid"),
) -> pd.DataFrame:
    """ROC AUC of attention, and of its residual, against the simulated interaction edges.

    Reported per range class, each against the same set of non-interacting pairs.

    Read ``resid`` against chance (0.5), not against ``attn``. True interactions are themselves
    distance-dependent, so regressing distance out necessarily removes part of the signal and the
    residual AUC is normally the lower of the two; what it answers is whether anything survives
    that a distance kernel plus cell-type identity could not have produced. A residual AUC at 0.5
    means the attention was the spatial prior and nothing more.
    """
    if "synthetic" not in adata.uns or "interaction_edges" not in adata.uns["synthetic"]:
        raise KeyError("no ground truth in adata.uns['synthetic']; this scoring only applies to the simulated data")

    edges = pd.DataFrame(adata.uns["synthetic"]["interaction_edges"])
    n = adata.n_obs
    truth: dict[str, set] = {}
    for rc, sub in edges.groupby("range_class", observed=True):
        truth[str(rc)] = set((sub["sender"].to_numpy().astype(np.int64) * n + sub["receiver"].to_numpy()).tolist())

    # `i` is the query (receiving) cell and `j` the key (sending) one, so a true sender->receiver
    # edge is the key `j * n + i`.
    directed = pairs["j"].to_numpy().astype(np.int64) * n + pairs["i"].to_numpy()
    any_edge = np.zeros(len(pairs), dtype=bool)
    for keys in truth.values():
        any_edge |= np.isin(directed, list(keys))

    rows = []
    for rc, keys in list(truth.items()) + [("any", set().union(*truth.values()))]:
        positive = np.isin(directed, list(keys))
        # Non-edges are pairs that are not an interaction of *any* range class, so the negative set
        # is identical across rows and the AUCs are comparable.
        mask = positive | ~any_edge
        if positive.sum() < 10 or (~any_edge).sum() < 10:
            continue
        for col in score_columns:
            if col not in pairs:
                continue
            rows.append(
                {
                    "range_class": rc,
                    "score": col,
                    "n_positive": int(positive.sum()),
                    "n_negative": int((~any_edge).sum()),
                    "roc_auc": roc_auc_score(positive[mask], pairs[col].to_numpy()[mask]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the attention regression from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", type=Path, required=True, help="output of CombinedModel.get_model_output")
    parser.add_argument("--prefix", type=str, default="combined")
    parser.add_argument("--sample-key", type=str, default="slide")
    parser.add_argument("--obs-features", type=str, nargs="*",
                        default=["cell_type", "niche", "total_counts", "n_genes_by_counts"])
    parser.add_argument("--normalize", choices=NORMALIZATIONS, default="graph_z")
    parser.add_argument("--max-pairs", type=int, default=200_000)
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS), default="ridge")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write the variance table as CSV")
    args = parser.parse_args()

    import anndata as ad

    adata = ad.read_h5ad(args.h5ad)
    pairs = attention_pairs(
        adata,
        prefix=args.prefix,
        sample_key=args.sample_key,
        obs_features=tuple(args.obs_features),
        normalize=args.normalize,
        max_pairs=args.max_pairs,
        random_state=args.seed,
    )
    result = regress_attention(
        pairs, estimator=args.estimator, cv_folds=args.cv_folds, random_state=args.seed
    )
    adata.uns["downstream_attention_regression"] = result.variance

    print(f"pairs: {len(pairs)}  normalize: {args.normalize}")
    print("\n== variance explained (out-of-fold, folds grouped by graph) ==")
    print(result.variance.round(4).to_string(index=False))
    print(f"\nnonlinear reference (gbm, full covariate set): R2 = {result.nonlinear_r2:.4f}")
    print("\n== mean residual attention, sender -> receiver ==")
    print(residual_flow(result.pairs).round(4).to_string())
    try:
        print("\n== ranking of true interaction pairs ==")
        print(score_against_truth(adata, result.pairs).round(4).to_string(index=False))
    except KeyError as err:
        print(f"(skipped: {err})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.variance.to_csv(args.out, index=False)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
