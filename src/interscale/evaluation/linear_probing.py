"""Linear probes: what is readable from the embeddings, by a model simple enough to prove it.

The question this answers is not "can we classify" but "which representation carries the label, and
does it beat the raw expression it was built from". Both are answered by running the same probe
over several feature sets at once -- local embedding, global embedding, CLS token, their
concatenation, raw expression, and a shuffled-label null -- and reporting them in one table. A
feature set is any ``.obsm`` key, or several joined by ``+``, so probing an ``X_pca``, ``X_scVI``
or precomputed CellCharter representation alongside them needs no change to this module.

Two rules keep the numbers honest:

* the probe reuses ``adata.obs[split_key]``, the split the encoder itself was trained under, so it
  never scores cells the encoder has seen;
* slide- or condition-level targets are evaluated with a donor-grouped CV, because with ~24 slides
  from 12 donors a random split would mostly measure donor identity.

Models come from scikit-learn; nothing here is hand-rolled. Feature sets are plain dicts naming
``.obsm`` keys and ``.obs`` columns, so any annotation in the object can be probed or used as a
covariate without touching this file.

Run as a script::

    python src/interscale/evaluation/linear_probing.py \
        --h5ad results/synth_data_0_model_output.h5ad --target cell_type --level node \
        --features X_pca X_scVI combined_local_emb+combined_global_emb
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ESTIMATORS = {
    "logreg": lambda seed: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    "rf": lambda seed: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
    "gbm": lambda seed: HistGradientBoostingClassifier(random_state=seed),
    "mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed),
}


def build_features(
    adata: AnnData,
    *,
    obsm_keys: tuple[str, ...] = (),
    obs_keys: tuple[str, ...] = (),
    expression: bool = False,
    layer: str | None = None,
    n_pca: int = 30,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Assemble a cell-level feature matrix from ``.obsm`` keys, ``.obs`` columns and expression.

    Categorical ``.obs`` columns are one-hot encoded, which also makes them aggregate sensibly:
    mean-pooling a one-hot column over a slide gives that slide's composition.

    Parameters
    ----------
    adata
        Annotated data object, normally the output of ``CombinedModel.get_model_output``.
    obsm_keys
        Keys of ``adata.obsm`` to use, e.g. ``("combined_global_emb",)``.
    obs_keys
        Columns of ``adata.obs`` to use; numeric columns are taken as-is, categorical ones are
        one-hot encoded.
    expression
        Include the expression matrix (``layer`` or ``.X``), reduced to ``n_pca`` components.
    layer
        Layer to read expression from. ``None`` means ``.X``.
    n_pca
        Number of principal components; ``0`` uses the full matrix.
    random_state
        Seed for the PCA.

    Returns
    -------
    tuple
        The ``(n_cells, n_features)`` matrix and the matching feature names.
    """
    blocks, names = [], []

    for key in obsm_keys:
        if key not in adata.obsm:
            raise KeyError(f"'{key}' not in adata.obsm. Available: {list(adata.obsm)}")
        block = np.asarray(adata.obsm[key], dtype=np.float64)
        block = np.nan_to_num(block)
        blocks.append(block)
        names += [f"{key}[{i}]" for i in range(block.shape[1])]

    for key in obs_keys:
        if key not in adata.obs:
            raise KeyError(f"'{key}' not in adata.obs")
        col = adata.obs[key]
        if pd.api.types.is_numeric_dtype(col):
            blocks.append(np.nan_to_num(col.to_numpy(dtype=np.float64))[:, None])
            names.append(key)
        else:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            block = enc.fit_transform(col.astype(str).to_numpy()[:, None])
            blocks.append(block)
            names += [f"{key}={c}" for c in enc.categories_[0]]

    if expression:
        X = adata.layers[layer] if layer is not None else adata.X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        if n_pca and n_pca < X.shape[1]:
            X = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)
            names += [f"PC{i}" for i in range(X.shape[1])]
        else:
            names += list(adata.var_names)
        blocks.append(X.astype(np.float64))

    if not blocks:
        raise ValueError("feature set is empty: give at least one of obsm_keys, obs_keys, expression")
    return np.concatenate(blocks, axis=1), names


def feature_sets_from_spec(adata: AnnData, specs: Sequence[str]) -> dict[str, dict]:
    """Build feature sets from ``"a+b"`` strings, so any combination can be named in one place.

    Each token is resolved in order against ``.obsm`` (an embedding), ``.obs`` (a covariate, one-hot
    encoded when categorical), or the literals ``"expression"``/``"X"`` (the expression matrix,
    PCA-reduced). Tokens joined by ``+`` are concatenated into a single feature matrix.

    Examples
    --------
    ``["combined_local_emb+combined_global_emb", "X_scVI", "X_pca+cell_type"]`` gives three feature
    sets: the two InterScale embeddings together, an scVI embedding on its own, and a PCA of the
    expression with the cell-type annotation appended.
    """
    sets: dict[str, dict] = {}
    for spec in specs:
        obsm_keys, obs_keys, expression = [], [], False
        for token in (t.strip() for t in spec.split("+")):
            if not token:
                continue
            if token in ("expression", "X"):
                expression = True
            elif token in adata.obsm:
                obsm_keys.append(token)
            elif token in adata.obs:
                obs_keys.append(token)
            else:
                raise KeyError(
                    f"'{token}' is in neither adata.obsm nor adata.obs. "
                    f"Keys in .obsm: {sorted(adata.obsm)}"
                )
        sets[spec] = {"obsm": tuple(obsm_keys), "obs": tuple(obs_keys), "expression": expression}
    return sets


def default_feature_sets(adata: AnnData, prefix: str = "combined") -> dict[str, dict]:
    """The InterScale feature sets, skipping the ones this object does not carry.

    The point of keeping local, global and CLS apart is attribution: a label readable from the
    global embedding but not the local one is a tissue-scale property, and vice versa.

    This is only the zero-argument default. To probe anything else -- ``X_pca``, ``X_scVI``, a
    precomputed CellCharter representation, or any combination -- name the keys through
    :func:`feature_sets_from_spec`.
    """
    sets: dict[str, dict] = {}
    if f"{prefix}_local_emb" in adata.obsm:
        sets["local_emb"] = {"obsm": (f"{prefix}_local_emb",)}
    if f"{prefix}_global_emb" in adata.obsm:
        sets["global_emb"] = {"obsm": (f"{prefix}_global_emb",)}
    if len(sets) == 2:
        sets["local+global"] = {"obsm": (f"{prefix}_local_emb", f"{prefix}_global_emb")}
    cls_cols = [c for c in (f"{prefix}_cls_horizontal", f"{prefix}_cls_vertical") if c in adata.obs]
    if cls_cols:
        sets["cls"] = {"obs": tuple(cls_cols)}

    return sets


def _aggregate(X: np.ndarray, y: pd.Series, groups: np.ndarray, split: np.ndarray, unit: np.ndarray):
    """Mean-pool cell-level features to one row per ``unit`` (slide, sample, ...)."""
    df = pd.DataFrame(X)
    df["_unit"] = unit
    pooled = df.groupby("_unit", observed=True).mean()
    idx = pooled.index

    meta = pd.DataFrame({"y": np.asarray(y), "g": groups, "s": split, "_unit": unit})
    first = meta.groupby("_unit", observed=True).agg(lambda v: v.iloc[0])
    nunique = meta.groupby("_unit", observed=True)["y"].nunique()
    if (nunique > 1).any():
        bad = nunique[nunique > 1].index.tolist()[:5]
        raise ValueError(f"target is not constant within the aggregation unit (e.g. {bad}); use level='node'")

    first = first.loc[idx]
    return pooled.to_numpy(), first["y"].to_numpy(), first["g"].to_numpy(), first["s"].to_numpy()


def _grouped_splitter(y: np.ndarray, groups: np.ndarray, n_folds: int, random_state: int):
    """A donor-grouped CV splitter that also keeps every fold class-balanced where it can.

    Plain :class:`~sklearn.model_selection.GroupKFold` over donors happily produces a fold whose
    test set is entirely one condition, which makes balanced accuracy meaningless. Stratifying on
    top of the grouping avoids that; it is only possible while each class has at least ``n_splits``
    distinct groups, so the fold count is capped accordingly.
    """
    classes, counts = np.unique(y, return_counts=True)
    groups_per_class = [len(np.unique(groups[y == c])) for c in classes]
    n_splits = min(n_folds, len(np.unique(groups)), min(groups_per_class))
    if n_splits < 2:
        raise ValueError(
            f"cannot build a grouped CV: the rarest class spans {min(groups_per_class)} group(s). "
            "Use level='node', or a group_key with more levels."
        )
    if min(counts) >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return GroupKFold(n_splits=n_splits)


def _scores(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    """Balanced accuracy, macro F1 and the probability-based metrics, guarded for degenerate cases."""
    with warnings.catch_warnings():
        # A small fold can legitimately miss a class that the model still predicts; recall is then
        # averaged over the classes actually present, which is what we want, so the warnings are noise.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found")
        out = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0),
            "accuracy": float((y_true == y_pred).mean()),
        }
    onehot = np.zeros((len(y_true), len(classes)))
    for k, c in enumerate(classes):
        onehot[y_true == c, k] = 1
    seen = onehot.sum(0) > 0
    if seen.sum() >= 2:
        out["average_precision_macro"] = average_precision_score(onehot[:, seen], proba[:, seen], average="macro")
        try:
            out["roc_auc_macro"] = roc_auc_score(onehot[:, seen], proba[:, seen], average="macro")
        except ValueError:
            pass
    return out


def _fit_score(estimator, X_tr, y_tr, X_te, y_te, classes) -> dict[str, float]:
    model = Pipeline([("scale", StandardScaler()), ("clf", estimator)])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    # Re-index the columns onto the full class list: a fold may miss a rare class entirely.
    full = np.zeros((len(y_te), len(classes)))
    for k, c in enumerate(model.named_steps["clf"].classes_):
        full[:, list(classes).index(c)] = proba[:, k]
    return _scores(y_te, model.predict(X_te), full, classes)


def classify(
    adata: AnnData,
    target: str,
    *,
    feature_sets: dict[str, dict] | None = None,
    prefix: str = "combined",
    level: str = "node",
    sample_key: str = "slide",
    split_key: str = "split",
    group_key: str = "donor",
    estimator: str = "logreg",
    baselines: tuple[str, ...] = ("expression", "shuffled", "majority"),
    layer: str | None = "log1p_norm",
    n_pca: int = 30,
    cv_folds: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Probe every feature set for ``target`` and return one tidy table of metrics.

    Parameters
    ----------
    adata
        Object carrying the trained model's output.
    target
        Column of ``adata.obs`` to predict -- ``cell_type``, ``condition``, ``niche``, ``slide``, ...
    feature_sets
        ``{name: {"obsm": (...), "obs": (...), "expression": bool}}``. Defaults to
        :func:`default_feature_sets`. Name any other embedding or combination through
        :func:`feature_sets_from_spec` (``["X_scVI", "local_emb+global_emb"]``).
    prefix
        Prefix the model wrote its output under (``get_model_output(prefix=...)``).
    level
        ``"node"`` scores per cell; ``"graph"`` mean-pools features per ``sample_key`` first, which
        is the right level for slide- or condition-scale targets.
    sample_key, split_key, group_key
        ``.obs`` columns naming the aggregation unit, the train/val/test split and the grouping
        variable (donor) that grouped CV must not break across folds. ``group_key=None``, or a
        name absent from ``.obs``, disables grouping so folds are cut over individual cells --
        which is what a target nested inside the grouping variable, such as ``slide`` within
        ``donor``, requires.
    estimator
        One of :data:`ESTIMATORS`.
    baselines
        Any of ``"expression"`` (probe the raw counts the embedding was built from),
        ``"shuffled"`` (labels permuted within train -- the null), ``"majority"``.
    layer, n_pca
        Expression layer and PCA width for the expression baseline.
    cv_folds
        Force donor-grouped CV with this many folds. ``None`` uses the stored split at node level
        and falls back to 5-fold grouped CV at graph level, where a single split is too small.
    random_state
        Seed for estimators and label permutation.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature_set, target, level, estimator, eval, fold, n_train, n_eval, metric, value``.
    """
    if target not in adata.obs:
        raise KeyError(f"target '{target}' not in adata.obs")
    if estimator not in ESTIMATORS:
        raise KeyError(f"unknown estimator '{estimator}'. Available: {sorted(ESTIMATORS)}")

    feature_sets = dict(feature_sets or default_feature_sets(adata, prefix))
    if "expression" in baselines:
        feature_sets["expression"] = {"expression": True}
    if not feature_sets:
        raise ValueError("no feature sets: the object carries no embeddings and no override was given")

    y_all = adata.obs[target].astype(str)
    groups_all = adata.obs[group_key].astype(str).to_numpy() if group_key in adata.obs else np.arange(adata.n_obs)
    split_all = (
        adata.obs[split_key].astype(str).to_numpy()
        if split_key in adata.obs
        else np.array(["train"] * adata.n_obs)
    )
    unit_all = adata.obs[sample_key].astype(str).to_numpy() if sample_key in adata.obs else np.arange(adata.n_obs)

    use_cv = cv_folds is not None or level == "graph"
    n_folds = cv_folds or 5

    rows = []
    for name, spec in feature_sets.items():
        X, _ = build_features(
            adata,
            obsm_keys=tuple(spec.get("obsm", ())),
            obs_keys=tuple(spec.get("obs", ())),
            expression=bool(spec.get("expression", False)),
            layer=layer,
            n_pca=n_pca,
            random_state=random_state,
        )
        y, groups, split, unit = np.asarray(y_all), groups_all, split_all, unit_all
        if level == "graph":
            X, y, groups, split = _aggregate(X, y_all, groups_all, split_all, unit_all)
        classes = np.unique(y)

        if not use_cv:
            # A split built to hold out whole groups can leave the eval folds with classes the
            # training fold never saw -- `slide` under a donor split is the clean example, since
            # no slide appears in two splits. The probe would then score ~0 and look like a
            # finding rather than an impossible question.
            train_classes = set(np.unique(y[split == "train"]))
            eval_classes = set(np.unique(y[np.isin(split, ("val", "test"))]))
            if eval_classes and not (eval_classes & train_classes):
                raise ValueError(
                    f"no class of '{target}' appears in both the train and the eval splits "
                    f"({len(train_classes)} train, {len(eval_classes)} eval, 0 shared), so the "
                    f"stored split cannot score it. Pass cv_folds to cross-validate over the "
                    f"whole object instead, with group_key=None if the target is nested inside "
                    f"the grouping variable."
                )

        variants = {name: y}
        if "shuffled" in baselines:
            variants[f"{name}__shuffled"] = np.random.default_rng(random_state).permutation(y)

        for vname, y_v in variants.items():
            if use_cv:
                splitter = _grouped_splitter(y_v, groups, n_folds, random_state)
                for fold, (tr, te) in enumerate(splitter.split(X, y_v, groups=groups)):
                    m = _fit_score(ESTIMATORS[estimator](random_state), X[tr], y_v[tr], X[te], y_v[te], classes)
                    rows += [
                        {"feature_set": vname, "eval": "cv", "fold": fold, "n_train": len(tr), "n_eval": len(te),
                         "metric": k, "value": v}
                        for k, v in m.items()
                    ]
            else:
                tr = split == "train"
                for ev in ("val", "test"):
                    te = split == ev
                    if te.sum() == 0:
                        continue
                    m = _fit_score(ESTIMATORS[estimator](random_state), X[tr], y_v[tr], X[te], y_v[te], classes)
                    rows += [
                        {"feature_set": vname, "eval": ev, "fold": 0, "n_train": int(tr.sum()),
                         "n_eval": int(te.sum()), "metric": k, "value": v}
                        for k, v in m.items()
                    ]

    if "majority" in baselines:
        y = np.asarray(y_all)
        X = np.zeros((adata.n_obs, 1))
        split, groups = split_all, groups_all
        if level == "graph":
            X, y, groups, split = _aggregate(X, y_all, groups_all, split_all, unit_all)
        classes = np.unique(y)
        tr = split == "train" if not use_cv else np.ones(len(y), bool)
        for ev in (("cv",) if use_cv else ("val", "test")):
            te = np.ones(len(y), bool) if use_cv else split == ev
            if te.sum() == 0:
                continue
            dummy = DummyClassifier(strategy="prior").fit(X[tr], y[tr])
            proba = dummy.predict_proba(X[te])
            rows += [
                {"feature_set": "majority", "eval": ev, "fold": 0, "n_train": int(tr.sum()), "n_eval": int(te.sum()),
                 "metric": k, "value": v}
                for k, v in _scores(y[te], dummy.predict(X[te]), proba, classes).items()
            ]

    out = pd.DataFrame(rows)
    out.insert(1, "target", target)
    out.insert(2, "level", level)
    out.insert(3, "estimator", estimator)
    adata.uns.setdefault("linear_probing", {}).setdefault("classification", {})[f"{target}__{level}"] = out
    return out


def summarize(results: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Collapse folds into mean +/- sd per feature set, for reading at a glance."""
    sub = results[results["metric"] == metric]
    return (
        sub.groupby(["feature_set", "eval"], observed=True)["value"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )


# ---------------------------------------------------------------------------------------
# Continuous probes: can a gene, or a continuous covariate, be read back out?
# ---------------------------------------------------------------------------------------

REGRESSORS = {
    "ridge": lambda seed: Ridge(alpha=1.0, random_state=seed),
    "rf": lambda seed: RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
    "gbm": lambda seed: HistGradientBoostingRegressor(random_state=seed),
}


def targets_of_interest(adata: AnnData, *, n_noise_controls: int = 4, random_state: int = 0) -> list[str]:
    """Genes worth probing one at a time: every annotated programme, plus noise genes as controls.

    The controls are the point. A probe that reconstructs `int_mid` at R^2 0.4 has said nothing
    until you know what it does on a gene with no structure at all -- the embedding reconstructs
    whatever it kept, and on the ``hvg=False`` arm what it kept is mostly noise. Reading the two
    side by side is what separates "found the programme" from "kept the loudest genes".
    """
    if "program" not in adata.var.columns:
        raise KeyError("adata.var has no 'program' column; pass the target genes explicitly")
    programme = [g for g, p in adata.var["program"].items() if p != "noise"]
    noise = [g for g, p in adata.var["program"].items() if p == "noise"]
    rng = np.random.default_rng(random_state)
    controls = list(rng.choice(noise, size=min(n_noise_controls, len(noise)), replace=False)) if noise else []
    return programme + controls


def _program_of(adata: AnnData, target: str) -> str:
    """Label a target by its planted programme, or as a covariate when it is an ``.obs`` column."""
    if target in adata.var_names and "program" in adata.var.columns:
        return str(adata.var["program"][target])
    return "covariate" if target not in adata.var_names else ""


def _resolve_target(adata: AnnData, target: str, layer: str | None) -> np.ndarray:
    """A target is either a gene (from ``var_names``) or a numeric ``.obs`` column."""
    if target in adata.var_names:
        X = adata.layers[layer] if layer is not None else adata.X
        col = X[:, adata.var_names.get_loc(target)]
        return np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
    if target in adata.obs.columns and pd.api.types.is_numeric_dtype(adata.obs[target]):
        return adata.obs[target].to_numpy(dtype=np.float64)
    raise KeyError(f"'{target}' is neither a gene in var_names nor a numeric .obs column")


def probe_continuous(
    adata: AnnData,
    targets: Sequence[str],
    *,
    feature_sets: dict[str, dict] | None = None,
    prefix: str = "combined",
    split_key: str = "split",
    group_key: str | None = "donor",
    estimator: str = "ridge",
    baselines: tuple[str, ...] = ("expression", "shuffled"),
    layer: str | None = "log1p_norm",
    n_pca: int = 30,
    cv_folds: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Predict each target separately from each feature set, and report how well.

    One model per (feature set, target): the question is which genes an embedding retained, and
    fitting them jointly would let an easy gene carry a hard one.

    **The expression baseline drops the target gene from its own features.** Without that it
    reads its own answer and scores 1.0, which is not a baseline but a tautology. Note this does
    not apply to the embeddings: an embedding trained to reconstruct the panel is *supposed* to
    contain the target, and how much of it survived the bottleneck is exactly the measurement.

    Parameters
    ----------
    adata
        Object carrying the model output.
    targets
        Gene names or numeric ``.obs`` columns. :func:`targets_of_interest` builds a sensible
        default: every annotated programme gene plus a few noise genes as controls.
    feature_sets, prefix, split_key, group_key, estimator, baselines, layer, n_pca, cv_folds
        As in :func:`classify`. ``estimator`` indexes :data:`REGRESSORS`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(target, feature_set, eval, fold, metric)``, with ``r2`` and ``pearson_r``.
    """
    if estimator not in REGRESSORS:
        raise KeyError(f"unknown estimator '{estimator}'. Available: {sorted(REGRESSORS)}")

    feature_sets = dict(feature_sets or default_feature_sets(adata, prefix))
    if "expression" in baselines:
        feature_sets["expression"] = {"expression": True}
    if not feature_sets:
        raise ValueError("no feature sets to probe")

    split = adata.obs[split_key].astype(str).to_numpy() if split_key in adata.obs else np.array(["train"] * adata.n_obs)
    groups = adata.obs[group_key].astype(str).to_numpy() if group_key in adata.obs else np.arange(adata.n_obs)
    use_cv = cv_folds is not None
    rng = np.random.default_rng(random_state)

    # Feature matrices do not depend on the target, so build each once rather than per gene.
    built: dict[str, np.ndarray] = {}
    for name, spec in feature_sets.items():
        built[name] = build_features(
            adata,
            obsm_keys=tuple(spec.get("obsm", ())),
            obs_keys=tuple(spec.get("obs", ())),
            expression=bool(spec.get("expression", False)),
            layer=layer,
            n_pca=0 if spec.get("expression") else n_pca,  # keep genes identifiable so the target can be dropped
            random_state=random_state,
        )[0]

    gene_pos = {g: i for i, g in enumerate(adata.var_names)}
    rows = []
    for target in targets:
        y = _resolve_target(adata, target, layer)
        for name, X in built.items():
            if name == "expression" and target in gene_pos:
                X = np.delete(X, gene_pos[target], axis=1)  # never let a gene predict itself

            variants = {name: y}
            if "shuffled" in baselines:
                variants[f"{name}__shuffled"] = rng.permutation(y)

            for vname, y_v in variants.items():
                if use_cv:
                    splitter = GroupKFold(n_splits=min(cv_folds, len(np.unique(groups))))
                    folds = list(splitter.split(X, y_v, groups=groups))
                    evals = [("cv", tr, te) for tr, te in folds]
                else:
                    tr = np.flatnonzero(split == "train")
                    evals = [(ev, tr, np.flatnonzero(split == ev)) for ev in ("val", "test")]

                for fold, (ev, tr, te) in enumerate(evals):
                    if len(te) == 0 or len(tr) == 0:
                        continue
                    model = Pipeline([("scale", StandardScaler()), ("reg", REGRESSORS[estimator](random_state))])
                    model.fit(X[tr], y_v[tr])
                    pred = model.predict(X[te])
                    r = np.corrcoef(pred, y_v[te])[0, 1] if np.std(pred) > 0 and np.std(y_v[te]) > 0 else 0.0
                    rows.append({
                        "target": target,
                        "program": _program_of(adata, target),
                        "feature_set": vname,
                        "eval": ev,
                        "fold": fold,
                        "estimator": estimator,
                        "r2": r2_score(y_v[te], pred),
                        "pearson_r": float(r),
                    })

    out = pd.DataFrame(rows)
    adata.uns.setdefault("linear_probing", {})["continuous"] = out
    return out


def summarize_by_target(results: pd.DataFrame, metric: str = "r2", eval_split: str | None = None) -> pd.DataFrame:
    """Targets as rows, feature sets as columns -- the per-gene summary, at a glance."""
    df = results if eval_split is None else results[results["eval"] == eval_split]
    df = df[~df["feature_set"].str.endswith("__shuffled")]
    table = df.pivot_table(index=["program", "target"], columns="feature_set", values=metric, aggfunc="mean")
    return table.sort_values(by=list(table.columns)[0], ascending=False)


def main() -> None:
    """Run the probe from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", type=Path, required=True, help="output of CombinedModel.get_model_output")
    parser.add_argument("--target", type=str, required=True, help="column of .obs to predict")
    parser.add_argument("--level", choices=["node", "graph"], default="node")
    parser.add_argument("--prefix", type=str, default="combined")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS), default="logreg")
    parser.add_argument("--sample-key", type=str, default="slide")
    parser.add_argument("--group-key", type=str, default="donor")
    parser.add_argument("--split-key", type=str, default="split")
    parser.add_argument("--layer", type=str, default="log1p_norm")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument(
        "--features", type=str, nargs="*", default=None,
        help=(
            "feature sets to probe, one per argument, tokens joined by '+'. A token is an .obsm "
            "key, an .obs column, or 'expression'. Example: --features X_scVI "
            "combined_local_emb+combined_global_emb X_pca+cell_type. "
            "Default: the local/global/CLS sets."
        ),
    )
    parser.add_argument("--metric", type=str, default="balanced_accuracy")
    parser.add_argument("--out", type=Path, default=None, help="write the tidy table as CSV")
    args = parser.parse_args()

    import anndata as ad

    adata = ad.read_h5ad(args.h5ad)
    feature_sets = (
        feature_sets_from_spec(adata, args.features) if args.features else default_feature_sets(adata, args.prefix)
    )
    print(f"feature sets: {list(feature_sets)}")

    results = classify(
        adata,
        args.target,
        feature_sets=feature_sets,
        prefix=args.prefix,
        level=args.level,
        sample_key=args.sample_key,
        split_key=args.split_key,
        group_key=args.group_key,
        estimator=args.estimator,
        layer=args.layer if args.layer != "X" else None,
        cv_folds=args.cv_folds,
    )
    print(json.dumps({"target": args.target, "level": args.level, "metric": args.metric}, indent=None))
    print(summarize(results, args.metric).round(3).to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
