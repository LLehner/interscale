from typing import Literal

import numpy as np
import pandas as pd
import squidpy as sq


def remove_zero_expression_cells(adata):
    zero_expression_cells = np.array(adata.X.sum(axis=1) == 0).flatten()
    print(f"Nr. of zero expression cells: {zero_expression_cells.sum()}")
    if zero_expression_cells.sum() > 0:
        nonzero_cells = np.array(adata.X.sum(axis=1) != 0).flatten()
        adata = adata[nonzero_cells].copy()
    return adata


def drop_genes(adata, cfg):
    """``adata`` without the genes of ``dataset.drop_genes`` (remove and retrain, ``.claude/ranking_plan.md`` R3).

    Selectors as for ``dataset.mask_genes`` (gene names or ``"<var column>=<value>"``; one that matches nothing raises). The genes
    leave the data as it is read, so neither the input nor the target holds them; ``uns['dropped_genes']`` records them. A gene
    the configuration hides as a target (``mask_genes``, ``mask_gene_sets``) is refused: dropping a target changes the task, not
    the input. Empty (the default): ``adata`` unchanged.
    """
    selectors = list(cfg.dataset.get("drop_genes", []) or [])
    if not selectors:
        return adata
    from interscale.tl.masking import resolve_mask_genes

    columns = resolve_mask_genes(selectors, adata.var)
    targets = set(resolve_mask_genes(list(cfg.dataset.get("mask_genes", []) or []), adata.var))
    for one in cfg.dataset.get("mask_gene_sets", []) or []:
        targets |= set(resolve_mask_genes(list(one), adata.var))
    clash = sorted(set(columns) & targets)
    if clash:
        names = [str(adata.var_names[c]) for c in clash]
        raise ValueError(f"dataset.drop_genes would drop target genes of the masking ({names}); drop inputs only.")
    keep = np.ones(adata.n_vars, dtype=bool)
    keep[columns] = False
    dropped = [str(g) for g in adata.var_names[~keep]]
    adata = adata[:, keep].copy()
    adata.uns["dropped_genes"] = dropped
    print(f"dataset.drop_genes: dropped {len(dropped)} genes")
    return adata


PIXEL_TO_UM = 0.138  # Resolve MC1 default; override via cfg if instrument differs


def get_average_local_and_global_size(
    adata,
    cfg,
    *,
    coord_units: Literal["pixels", "micrometer"] = "micrometer",
    pixel_size_um: float = PIXEL_TO_UM,
):
    """Average size of the local and the global window, in cells and in micrometers.

    Parameters
    ----------
    coord_units
        Units of ``adata.obsm['spatial']``. If ``"pixels"``, distances are
        converted to micrometers using ``pixel_size_um``.
    pixel_size_um
        Micrometers per pixel. Ignored when ``coord_units='micrometer'``.
    """
    scale = pixel_size_um if coord_units == "pixels" else 1.0

    ## Local: avg direct-neighbor count * n_layers
    if "spatial_connectivities" not in adata.obsp:
        kwargs = dict(cfg.dataset.spatial_neigbors_kwargs)
        # library_key is only filled in by prepare_geome_dataset, so pick the first sample key here
        kwargs["library_key"] = kwargs.get("library_key") or cfg.dataset.sample_key[0]
        adata.obs[kwargs["library_key"]] = adata.obs[kwargs["library_key"]].astype("category")
        sq.gr.spatial_neighbors(adata, **kwargs)
    n_layers = cfg.model.local_component.parameters.num_layers
    conn = adata.obsp["spatial_connectivities"].tocsr()
    local_cells = conn.getnnz(axis=1).mean() * n_layers
    # reach in micrometers: avg edge length walked over n_layers hops
    local_dist_um = adata.obsp["spatial_distances"].data.mean() * n_layers * scale

    ## Global: avg cells and avg extent per sample (or per sliding window, whichever sample_key defines)
    groups = adata.obs.groupby(list(cfg.dataset.sample_key), observed=True)
    coords = adata.obsm["spatial"]
    global_cells = groups.size().mean()
    global_dist_um = (
        np.mean([np.linalg.norm(coords[idx].max(axis=0) - coords[idx].min(axis=0)) for idx in groups.indices.values()])
        * scale
    )

    return {
        "local_cells": local_cells,
        "local_dist_um": local_dist_um,
        "global_cells": global_cells,
        "global_dist_um": global_dist_um,
    }


def imitation_labels(adata, key: str, *, train, layer: str | None = None, n_clusters: int = 15, n_pcs: int = 30, seed: int = 0):
    """``obs[key]``: clusters that imitate cell types, for a dataset without annotation (``model.decoder.label_offset_key``).

    k-means on the first ``n_pcs`` principal components of ``layer`` (default ``X``), both fitted on the cells of ``train``
    (``[cells]`` bool: the training split, so no held-out cell shapes a cluster), every cell then assigned to its nearest centre.
    An existing column is left alone: real annotation always comes first.
    """
    if key in adata.obs:
        return adata
    import scipy.sparse as sp
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    values = adata.layers[layer] if layer else adata.X
    values = values.toarray() if sp.issparse(values) else np.asarray(values)
    train = np.asarray(train, dtype=bool)
    mean = values[train].mean(0)
    svd = TruncatedSVD(n_components=min(int(n_pcs), values.shape[1] - 1), random_state=seed).fit(values[train] - mean)
    scores = svd.transform(values - mean)
    kmeans = MiniBatchKMeans(n_clusters=int(n_clusters), random_state=seed, n_init=3).fit(scores[train])
    adata.obs[key] = pd.Categorical([f"c{k}" for k in kmeans.predict(scores)])
    return adata
