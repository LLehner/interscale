"""Optional per-cell annotations on the PyG graphs: off by default, consistent when on.

These fields exist for the contrastive objectives -- which slide a cell is on, what type it is,
where it sits. Two things have to hold: configuring none of them must leave the graphs exactly as
they were, and configuring one must give every graph the same category ordering, because a
one-hot column that means "senderA" in one graph and "stroma" in the next is a silent corruption
that no shape check would catch.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

torch = pytest.importorskip("torch")

from interscale.config import get_cfg_defaults
from interscale.tl.geome_utils import label_codes, optional_fields


@pytest.fixture
def cfg():
    return get_cfg_defaults()


def test_nothing_is_attached_by_default(cfg):
    fields, preserve = optional_fields(cfg)

    assert fields == {}
    assert preserve == []


def test_each_key_resolves_to_its_source(cfg):
    cfg.dataset.slide_key = "slide_id"
    cfg.dataset.celltype_key = "cell_type"
    cfg.dataset.spatial_key = "spatial"

    fields, preserve = optional_fields(cfg)

    assert fields == {
        "slide": ["obs/slide_id"],
        "celltype": ["obs/cell_type"],
        "pos": ["obsm/spatial"],
    }
    # obsm keys are numeric matrices, so only the obs columns need category preservation.
    assert sorted(preserve) == ["cell_type", "slide_id"]


def test_label_codes_reads_a_one_hot_annotation():
    class FakeData:
        celltype = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert torch.equal(label_codes(FakeData(), "celltype"), torch.tensor([1, 0, 2]))


def test_label_codes_refuses_a_missing_annotation_by_name():
    """Returning zeros would turn a within-slide negative pool into a within-batch one, silently."""

    class FakeData:
        pass

    with pytest.raises(AttributeError, match="cfg.dataset.slide_key"):
        label_codes(FakeData(), "slide")


# --------------------------------------------------------------------------- through geome


def _adata(n_per_sample=8, seed=0):
    """Two samples where the SECOND lacks one cell type -- the case preserve_categories exists for."""
    rng = np.random.default_rng(seed)
    n = n_per_sample * 2
    adata = AnnData(X=np.abs(rng.normal(size=(n, 5))).astype(np.float32) + 0.1)
    adata.obsm["spatial"] = rng.uniform(0, 50, size=(n, 2))
    adata.obs["sample"] = pd.Categorical(["s1"] * n_per_sample + ["s2"] * n_per_sample)
    types = ["a", "b", "c"] * n_per_sample
    adata.obs["cell_type"] = pd.Categorical(
        types[:n_per_sample] + ["a" if t == "c" else t for t in types[n_per_sample:n]]
    )
    adata.obs["split"] = pd.Categorical(["train"] * n_per_sample + ["val"] * n_per_sample)
    return adata


def _prepare(cfg, adata):
    from interscale.tl import prepare_geome_dataset

    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.prediction_task = "regression"
    cfg.dataset.spatial_neigbors_kwargs.library_key = "sample"
    cfg.dataset.spatial_neigbors_kwargs.radius = 40
    cfg.model.global_component.parameters = type(cfg.model)()
    cfg.model.global_component.parameters.type_gex_embedding = None
    datas, _ = prepare_geome_dataset(adata, cfg)
    return datas


def test_graphs_are_unchanged_when_no_key_is_set(cfg):
    datas = _prepare(cfg, _adata())
    train = datas[0][0]

    # `pos` is a built-in PyG Data attribute, so it always exists and is None when unset --
    # hasattr is not the question for it.
    for name in ("slide", "condition", "celltype", "pos"):
        assert getattr(train, name, None) is None, f"{name} attached without a configured key"


def test_categories_keep_the_same_width_across_graphs(cfg):
    """The second sample has no 'c' cells; without preserve_categories its one-hot is narrower."""
    cfg.dataset.celltype_key = "cell_type"
    cfg.dataset.spatial_key = "spatial"
    adata = _adata()

    datas = _prepare(cfg, adata)
    graphs = datas[0] + datas[1]

    widths = {g.celltype.shape[1] for g in graphs}
    assert widths == {3}, f"category width differs across graphs: {widths}"
    for g in graphs:
        assert g.pos.shape == (g.num_nodes, 2)
        assert label_codes(g, "celltype").shape == (g.num_nodes,)


def test_annotations_survive_collation(cfg):
    """PyG collates node-level attributes by concatenation -- but only if the widths agree."""
    from torch_geometric.data import Batch

    cfg.dataset.celltype_key = "cell_type"
    cfg.dataset.spatial_key = "spatial"
    datas = _prepare(cfg, _adata())
    graphs = datas[0] + datas[1]

    batch = Batch.from_data_list(graphs)
    n = sum(g.num_nodes for g in graphs)

    assert batch.celltype.shape == (n, 3)
    assert batch.pos.shape == (n, 2)
    # The bridge a contrastive negative pool needs: token -> its graph -> its annotation.
    assert batch.batch.shape == (n,)
    assert label_codes(batch, "celltype").shape == (n,)


# --------------------------------------------------------------------------- arbitrary extras


def test_extra_obs_keys_attach_under_their_own_names(cfg):
    """Adding an annotation must not require editing OPTIONAL_FIELDS."""
    cfg.dataset.extra_obs_keys = ["niche", "region"]

    fields, preserve = optional_fields(cfg)

    assert fields == {"niche": ["obs/niche"], "region": ["obs/region"]}
    assert sorted(preserve) == ["niche", "region"]


def test_extras_and_named_keys_coexist(cfg):
    cfg.dataset.celltype_key = "cell_type"
    cfg.dataset.extra_obs_keys = ["niche"]

    fields, preserve = optional_fields(cfg)

    assert fields["celltype"] == ["obs/cell_type"]
    assert fields["niche"] == ["obs/niche"]
    assert sorted(preserve) == ["cell_type", "niche"]


@pytest.mark.parametrize("reserved", ["x", "mask", "edge_index", "gene_mask"])
def test_an_extra_may_not_shadow_a_field_the_pipeline_builds(cfg, reserved):
    """`mask` is the dangerous one: it would replace the corruption mask with a label."""
    cfg.dataset.extra_obs_keys = [reserved]

    with pytest.raises(ValueError, match="collides"):
        optional_fields(cfg)


def test_an_extra_may_not_shadow_a_named_key_in_use(cfg):
    cfg.dataset.celltype_key = "cell_type"
    cfg.dataset.extra_obs_keys = ["celltype"]

    with pytest.raises(ValueError, match="collides"):
        optional_fields(cfg)


def test_an_extra_survives_the_pipeline_with_its_categories_intact(cfg):
    """Extras get the same category preservation as the named keys, not a weaker path."""
    cfg.dataset.extra_obs_keys = ["cell_type"]
    adata = _adata()

    datas = _prepare(cfg, adata)
    graphs = datas[0] + datas[1]

    widths = {g.cell_type.shape[1] for g in graphs}
    assert widths == {3}, f"category width differs across graphs: {widths}"
    assert label_codes(graphs[0], "cell_type").shape == (graphs[0].num_nodes,)
