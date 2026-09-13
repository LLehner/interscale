import numpy as np
import pandas as pd
import torch
from geome import ann2data, iterables, transforms
from yacs.config import CfgNode as CN

#: Optional annotations attached to every PyG ``Data``: attribute name -> (config key, source).
#: Each is attached only when its ``cfg.dataset.*`` entry is set, so the default config produces
#: exactly the graphs it always did. See ``get_dataset_cfg`` for what each is for.
OPTIONAL_FIELDS = {
    "slide": ("slide_key", "obs"),
    "condition": ("condition_key", "obs"),
    "celltype": ("celltype_key", "obs"),
    "pos": ("spatial_key", "obsm"),
}


def optional_fields(cfg: CN) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve the configured optional annotations into geome ``fields`` entries.

    Returns
    -------
    fields : dict
        Entries to merge into the ``fields`` mapping handed to ``Ann2DataBasic``. Empty when no
        optional key is configured.
    preserve : list of str
        The obs columns that must be passed to ``ToCategoryIterator(preserve_categories=...)``.
        This is not optional bookkeeping: the iterator splits the object per sample, and a sample
        that happens to contain no cells of one category would otherwise produce a one-hot matrix
        one column narrower than its neighbours, which fails at collation -- or worse, silently
        shifts what each column means from graph to graph.
    """
    fields: dict[str, list[str]] = {}
    preserve: list[str] = []
    for name, (cfg_key, source) in OPTIONAL_FIELDS.items():
        key = getattr(cfg.dataset, cfg_key, None)
        if key is None:
            continue
        fields[name] = [f"{source}/{key}"]
        if source == "obs":
            preserve.append(key)
    return fields, preserve


def label_codes(data, name: str) -> torch.Tensor:
    """Integer codes ``[N]`` for a one-hot annotation attached by :func:`optional_fields`.

    An obs column reaches the ``Data`` object one-hot encoded, ``[N, n_categories]``. Everything
    that consumes one of these labels only ever asks whether two cells agree -- same slide, same
    cell type -- so codes are the useful form.

    Raises
    ------
    AttributeError
        If the annotation is absent, naming the config key that would attach it. Silently
        returning "all cells are category 0" would make a within-slide negative pool quietly
        become a within-batch one.
    """
    value = getattr(data, name, None)
    if value is None:
        cfg_key = OPTIONAL_FIELDS.get(name, (f"{name}_key", None))[0]
        raise AttributeError(f"Batch carries no {name!r}; set cfg.dataset.{cfg_key} to attach it.")
    return value.argmax(dim=1)


def prepare_a2d_dataset(cfg: CN):
    """ """
    adj_matrix_loc = "adj_matrix"

    category_to_iterate_list = cfg.dataset.sample_key
    prediction_obs = cfg.dataset.prediction_obs
    layer_key = cfg.dataset.layer_key

    for category_to_iterate in category_to_iterate_list:
        cfg.dataset.spatial_neigbors_kwargs.merge_from_list(["library_key", category_to_iterate])
        spatial_neigbors_kwargs = cfg.dataset.spatial_neigbors_kwargs

        one_hot_encode_list = [prediction_obs]

        X_key = f"layers/{layer_key}" if layer_key is not None else "X"
        print(f"Load GEX from .{X_key}")
        obsm_key = None
        if cfg.model.global_component.parameters.type_gex_embedding == "Precomputed":
            obsm_key = cfg.model.global_component.latent_obsm_key
        if "classification" in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "y": [f"obs/{prediction_obs}"],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
                "sample_key": [f"obs/{category_to_iterate}"],
            }

            preprocess = transforms.Compose(
                [transforms.SaveOneHotEncodeLabels(keys=one_hot_encode_list, axis="obs", key_added="one_hot")]
            )
        elif "regression" in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
                "sample_key": [f"obs/{category_to_iterate}"],
            }

            preprocess = None
        if obsm_key is not None:
            fields["embeddings"] = [f"obsm/{obsm_key}"]
        extra_fields, extra_preserve = optional_fields(cfg)
        fields.update(extra_fields)

        transform = transforms.Compose(
            [
                transforms.AddEdgeIndex(
                    edge_index_key="edge_index",
                    func_args=spatial_neigbors_kwargs,
                    spatial_key="spatial",
                    key_added=adj_matrix_loc,
                ),
            ]
        )

        return ann2data.Ann2DataBasic(
            fields=fields,
            adata2iter=iterables.ToCategoryIterator(
                category_to_iterate, axis="obs", preserve_categories=[prediction_obs] + extra_preserve
            ),
            preprocess=preprocess,
            transform=transform,
            save_preprocessed_adata=True,
        )


class _RestoreCategories:
    """Preprocess step re-instating the full object's categories on the given obs columns.

    Three separate things in the pipeline narrow a categorical before the one-hot encoding sees
    it, and each of them is upstream of the mechanism meant to prevent exactly that:

    * subsetting to a split (``adata[adata.obs[split_key] == "val"]``) drops categories the split
      does not contain -- on the *view*, before geome is involved at all;
    * ``transforms.Subset`` drops them again, even with an empty ``key_value`` where it subsets
      nothing;
    * only then does ``ToCategoryIterator(preserve_categories=...)`` run, preserving whatever is
      left rather than what was there originally.

    The consequence is silent: a validation split with no cells of one type yields a one-hot
    annotation one column narrower, with every column past the missing one shifted, and the widths
    only have to agree *within* a split for collation to succeed. This runs last in the preprocess
    chain, with the categories captured from the full object before any of the above.
    """

    def __init__(self, categories: dict[str, pd.Index]):
        self.categories = categories

    def __call__(self, adata):
        for key, cats in self.categories.items():
            if key in adata.obs:
                adata.obs[key] = pd.Categorical(adata.obs[key], categories=cats)
        return adata


def _category_snapshot(adata, keys: list[str]) -> dict[str, pd.Index]:
    """Full-object categories for each categorical column in ``keys``, taken before any subsetting."""
    snapshot = {}
    for key in keys:
        column = adata.obs.get(key)
        if column is not None and isinstance(column.dtype, pd.CategoricalDtype):
            snapshot[key] = column.cat.categories
    return snapshot


def prepare_geome_dataset(adata, cfg: CN):
    """
    Loads, preprocesses and transforms the defined .h5ad data to a list of PyG data according to cfg file.
    """
    assert ("classification" in (cfg.dataset.prediction_task)) or ("regression" in (cfg.dataset.prediction_task))
    if "classification" in cfg.dataset.prediction_task:
        assert str(cfg.dataset.prediction_obs) in adata.obs
    assert isinstance(cfg.dataset.sample_key, list)
    assert len(cfg.dataset.sample_key) >= 0
    assert all(item in adata.obs for item in cfg.dataset.sample_key), "Not all library_keys are in adata.obs_names"

    # Check for duplicate observation names
    adata.obs_names = [
        str(i) for i in range(1, len(adata.obs_names) + 1)
    ]  # ensure that no duplicate observation names are present
    assert len(adata.obs_names) == len(adata.obs_names.unique()), (
        f"Duplicate observation names found. Expected {len(adata.obs_names)} unique names but found {len(adata.obs_names.unique())}"
    )

    # Convert sample_key columns to categorical type to avoid numpy.dtypes.Int64DType error
    for key in cfg.dataset.sample_key:
        if key in adata.obs.columns:
            adata.obs[key] = adata.obs[key].astype("category")

    adj_matrix_loc = "adj_matrix"
    prediction_obs = cfg.dataset.prediction_obs
    category_to_iterate_list = cfg.dataset.sample_key
    layer_key = cfg.dataset.layer_key
    subset_dict = {}

    assert cfg.dataset.split_key in adata.obs.columns, (
        f"split_key '{cfg.dataset.split_key}' not found in adata.obs columns"
    )
    split_key = cfg.dataset.split_key

    # initalize object to save train, val and test PyG datas
    datas_train, datas_val, datas_test = list(), list(), list()

    for category_to_iterate in category_to_iterate_list:
        cfg.dataset.spatial_neigbors_kwargs.merge_from_list(["library_key", category_to_iterate])
        spatial_neigbors_kwargs = cfg.dataset.spatial_neigbors_kwargs

        one_hot_encode_list = [prediction_obs]
        X_key = f"layers/{layer_key}" if layer_key is not None else "X"

        obsm_key = None
        if cfg.model.global_component.parameters.type_gex_embedding == "Precomputed":
            obsm_key = cfg.model.global_component.latent_obsm_key
        if "classification" in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "y": [f"obs/{prediction_obs}"],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
            }

            preprocess = transforms.Compose(
                [
                    transforms.Subset(key_value=subset_dict, axis="obs"),
                    transforms.Categorize(keys=list(subset_dict.keys()) + one_hot_encode_list, axis="obs"),
                    transforms.SaveOneHotEncodeLabels(keys=one_hot_encode_list, axis="obs", key_added="one_hot"),
                ]
            )
        elif "regression" in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
            }

            preprocess = transforms.Compose(
                [
                    transforms.Subset(key_value=subset_dict, axis="obs"),
                ]
            )
        if obsm_key is not None:
            if obsm_key not in adata.obsm:
                raise ValueError(f"Precomputed embeddings key '{obsm_key}' not found in adata.obsm")
            fields["embeddings"] = [f"obsm/{obsm_key}"]
        # The same optional annotations the evaluation path attaches, resolved from the same
        # helper so the two cannot drift apart.
        extra_fields, extra_preserve = optional_fields(cfg)
        fields.update(extra_fields)
        snapshot = _category_snapshot(adata, extra_preserve)
        if snapshot:
            steps = list(preprocess.transforms) if preprocess is not None else []
            preprocess = transforms.Compose([*steps, _RestoreCategories(snapshot)])
        transform = transforms.Compose(
            [
                transforms.AddEdgeIndex(
                    edge_index_key="edge_index",
                    func_args=spatial_neigbors_kwargs,
                    spatial_key="spatial",
                    key_added=adj_matrix_loc,
                ),
            ]
        )

        a2d = ann2data.Ann2DataBasic(
            fields=fields,
            adata2iter=iterables.ToCategoryIterator(
                category_to_iterate, axis="obs", preserve_categories=[prediction_obs] + extra_preserve
            ),
            preprocess=preprocess,
            transform=transform,
            save_preprocessed_adata=True,
        )

        pyg_train, _ = list(a2d(adata[adata.obs[split_key] == "train"]))
        pyg_val, _ = list(a2d(adata[adata.obs[split_key] == "val"]))
        datas_train.extend(pyg_train)
        datas_val.extend(pyg_val)
        if "test" in np.unique(adata.obs[split_key]):
            pyg_test, _ = list(a2d(adata[adata.obs[split_key] == "test"]))
            datas_test.extend(pyg_test)

    if "test" in np.unique(adata.obs[split_key]):
        # datas_test, adata_test = list(a2d(adata[adata.obs[split_key] == "test"]))
        return [datas_train, datas_val, datas_test], _

    return [datas_train, datas_val], _
