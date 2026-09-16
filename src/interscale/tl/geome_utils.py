import warnings

import numpy as np
import pandas as pd
import torch
from geome import ann2data, iterables, transforms
from yacs.config import CfgNode as CN

#: Names the pipeline builds itself. An ``extra_obs_keys`` entry may not shadow one of these:
#: attaching an obs column called ``mask`` would replace the corruption mask with a label and the
#: run would train against its own annotation without failing.
RESERVED_FIELD_NAMES = {"x", "y", "edge_index", "obs_names", "embeddings", "batch", "mask", "gene_mask"}

#: Optional annotations attached to every PyG ``Data``: attribute name -> (config key, source).
#: Each is attached only when its ``cfg.dataset.*`` entry is set, so the default config produces
#: exactly the graphs it always did. See ``get_dataset_cfg`` for what each is for.
#:
#: This dict holds only the roles the CODE reads by name -- negative sampling reads ``slide``,
#: the split check reads ``group``, the flow control reads ``celltype``. Anything a dataset
#: merely carries (niche, region, stage, timepoint) goes through ``dataset.extra_obs_keys``
#: instead, so a new annotation never needs an entry here.
OPTIONAL_FIELDS = {
    "slide": ("slide_key", "obs"),
    "group": ("group_key", "obs"),
    "condition": ("condition_key", "obs"),
    "celltype": ("celltype_key", "obs"),
    "pos": ("spatial_key", "obsm"),
    # Numeric probe targets, stacked into one matrix because geome cannot attach a numeric obs
    # column. See `cfg.dataset.probe_obsm_key`.
    "probe_targets": ("probe_obsm_key", "obsm"),
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

    # Arbitrary further obs columns, attached under their own names. The point of this branch is
    # that a new probe target or stratifier never needs a code change -- OPTIONAL_FIELDS holds
    # only the handful of roles the code reads *by name*.
    for key in getattr(cfg.dataset, "extra_obs_keys", []) or []:
        if key in fields or key in RESERVED_FIELD_NAMES:
            raise ValueError(
                f"dataset.extra_obs_keys entry {key!r} collides with a field the pipeline already "
                f"builds ({sorted(RESERVED_FIELD_NAMES | set(fields))}). Rename the obs column, or "
                "use the dedicated dataset.*_key for that role."
            )
        fields[key] = [f"obs/{key}"]
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


def warn_missing_categories(adata, split_key: str, cfg: CN) -> list[str]:
    """Warn once per (column, split) where a split contains no cells of some category.

    A categorical is narrowed three separate times on the way to its one-hot encoding -- the
    split subset drops unused categories on the view, ``transforms.Subset`` drops them again, and
    ``ToCategoryIterator(preserve_categories=...)`` then preserves only what is left. The two
    consequences are very different, and neither is obvious from where it surfaces:

    * for ``dataset.prediction_obs`` the run **fails**, because ``n_output`` is derived from the
      full object while ``y`` is built per split, so the widths disagree -- but it fails deep in
      the training step with ``"y_true and y_pred must have the same shape"``, which names
      nothing that would lead anyone back to a missing cell type;
    * for the optional annotations (``celltype_key`` and friends) the encoding is now repaired by
      :class:`_RestoreCategories`, so the columns still mean the same thing in every split -- but
      the *data* genuinely has none of that category there, which anything sampling per category
      in that split needs to know.

    Returns
    -------
    list of str
        The messages emitted, so a caller or a test can inspect them.
    """
    _, optional_keys = optional_fields(cfg)
    prediction_obs = cfg.dataset.prediction_obs
    keys = list(dict.fromkeys(([prediction_obs] if prediction_obs else []) + optional_keys))

    messages = []
    for key in keys:
        column = adata.obs.get(key)
        if column is None or not isinstance(column.dtype, pd.CategoricalDtype):
            continue
        full = list(column.cat.categories)
        for split in pd.unique(adata.obs[split_key]):
            present = set(column[adata.obs[split_key] == split].dropna().astype(str))
            missing = [c for c in full if str(c) not in present]
            if not missing:
                continue

            if key == prediction_obs:
                consequence = (
                    f"Its one-hot label will have {len(full) - len(missing)} columns in that split "
                    f"while n_output is {len(full)}, so training will stop with "
                    '"y_true and y_pred must have the same shape". Merge the category, restratify '
                    "the split, or drop those cells."
                )
            else:
                consequence = (
                    "Category ordering is preserved across splits, so the annotation itself stays "
                    "correct; but anything that samples per category in that split, e.g. "
                    "composition-matched negatives, has nothing to draw for it."
                )

            message = (
                f"Split {split!r} contains no cells of {key!r} categor"
                f"{'y' if len(missing) == 1 else 'ies'} {missing!r}. {consequence}"
            )
            messages.append(message)
            warnings.warn(message, UserWarning, stacklevel=2)
    return messages


def check_split_independence(adata, split_key: str, group_key: str | None) -> list[str]:
    """Warn when one group's cells land in more than one split.

    ``group_key`` names the unit of statistical independence -- donor, patient, mouse, processing
    batch. Cells sharing a value are not independent observations, so a split that straddles one
    lets any readout score by recognising the group instead of the biology: the encoder's own
    validation metrics, and every probe fitted on train and scored on val.

    This *reports* rather than enforces, for two reasons. A straddling split is sometimes
    deliberate (a single-donor dataset has no other option), and the right repair -- regroup,
    re-split, drop a group -- depends on the study design, not on anything visible from here.

    Nothing is checked when ``group_key`` is None, which is the default. A dataset that does not
    name its grouping simply gets the previous behaviour; it does not get a false all-clear,
    because no claim is made either way.

    Parameters
    ----------
    adata
        The full object, before any split subsetting.
    split_key
        ``adata.obs`` column holding ``train``/``val``/``test``.
    group_key
        ``adata.obs`` column naming the unit of independence, or None to skip.

    Returns
    -------
    list of str
        The messages emitted, so a caller or a test can inspect them.
    """
    if group_key is None:
        return []
    if group_key not in adata.obs.columns:
        message = (
            f"dataset.group_key is {group_key!r} but that column is not in adata.obs, so the "
            "split cannot be checked for group leakage. Either add the column or unset group_key."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        return [message]

    groups = adata.obs.groupby(group_key, observed=True)[split_key].unique()
    messages = []
    for group, splits in groups.items():
        splits = sorted(str(x) for x in splits)
        if len(splits) < 2:
            continue
        message = (
            f"{group_key} {group!r} has cells in {len(splits)} splits ({', '.join(splits)}). "
            "Cells of one group are not independent, so anything fitted on one split and scored "
            "on another -- validation metrics, and every probe -- can score by recognising the "
            f"group rather than the biology. Split by {group_key!r} instead of by cell."
        )
        messages.append(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    return messages


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
    warn_missing_categories(adata, split_key, cfg)
    check_split_independence(adata, split_key, cfg.dataset.group_key)

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
