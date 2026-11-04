"""The target a label does not explain (``model.decoder.section_offset: label_mean``), the scores within groups, the k-NN form of the
fitted references, and the arms of the CosMx programme run (the user, 2026-11-04: "predicting the effect that is not explained by cell
type / structure alone"; "if cell type annotations exist, use that, if not other clusters that imitate cell types").

Pinned here:

* **the means** -- each gene's mean per label over the given (training) graphs; a label without a training cell gets the overall mean;
  saved with the model, whatever the number of labels;
* **the part** -- the prediction is the decoder's output plus the cell's label mean as a named part; no hidden value reaches it;
* **the loss** -- reads prediction and target minus the label mean, cell by cell: what is left of the target, what the decoder predicts;
* **the switch** -- validator, checkpoint tag; off, the model holds no new state;
* **within_groups** -- a head that only knows the group scores 0, a head that follows the variation inside it scores 1;
* **FittedReferences(k_nearest)** -- on neighbour lists, no hidden value, a saved fit round-trips;
* **the arms** -- channels and a named all-genes mask train through hierarchy and crops, with the coordinates as the only input in the second.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

E, G, LABELS = 8, 6, 3


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _graph(side, seed, *, n_labels=LABELS, hidden_genes=(0, 1)):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * 100.0
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= 101.0).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=edges, pos=pos)
    codes = torch.arange(n) % n_labels
    data.cell_type = torch.nn.functional.one_hot(codes, n_labels).float()
    data.mask = torch.zeros(n, dtype=torch.bool)
    data.gene_mask = torch.zeros(n, G, dtype=torch.bool)
    data.gene_mask[:, list(hidden_genes)] = True
    data.obs_names = torch.arange(n)
    return data


def _cfg(**decoder):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    cfg.model.decoder.section_offset = decoder.get("section", "label_mean")
    cfg.model.decoder.label_offset_key = decoder.get("key", "cell_type")
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 64, 2, 2, 0.0
    params.type_gex_embedding = "Linear"
    cfg.optim.loss = "SCELoss"
    cfg.optim.center_targets = decoder.get("center", True)
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    cfg.dataset.mask_genes = ["g0", "g1"]
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.extra_obs_keys = decoder.get("extra", ["cell_type"])
    return cfg


def _model(seed=0, **decoder):
    from interscale.config import _validate
    from interscale.model.global_model import GlobalModel

    cfg = _cfg(**decoder)
    _validate(cfg)
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obs["cell_type"] = pd.Categorical(["a", "b"] * 20)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(seed)
    return GlobalModel(adata, cfg=cfg)


def _manual_means(graphs):
    x = torch.cat([g.x for g in graphs]).double()
    codes = torch.cat([g.cell_type.argmax(1) for g in graphs])
    return torch.stack([x[codes == k].mean(0) if (codes == k).any() else x.mean(0) for k in range(graphs[0].cell_type.shape[1])])


def _batch(graphs):
    return Batch.from_data_list(graphs)


def test_the_means_are_per_label_over_the_training_graphs_and_an_unseen_label_gets_the_overall_mean():
    net = _model()
    graphs = [_graph(4, 1), _graph(5, 2)]
    net.module.set_label_means(graphs)
    assert torch.allclose(net.module.label_means.double(), _manual_means(graphs), atol=1e-5)
    # A label no cell has: the overall mean.
    lonely = _graph(4, 3, n_labels=LABELS + 1)
    lonely.cell_type[:, LABELS] = 0.0
    lonely.cell_type[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    net.module.set_label_means([lonely])
    assert torch.allclose(net.module.label_means[LABELS].double(), lonely.x.double().mean(0), atol=1e-5)


def test_the_means_travel_with_the_checkpoint():
    net = _model()
    net.module.set_label_means([_graph(4, 1)])
    state = net.module.state_dict()
    other = _model(seed=1)
    assert other.module.label_means.numel() == 0
    other.module.load_state_dict(state, strict=False)
    assert torch.equal(other.module.label_means, net.module.label_means)


def test_the_prediction_is_the_decoder_plus_the_label_mean_and_no_hidden_value_reaches_it():
    net = _model()
    module = _stir(net.module)
    graphs = [_graph(4, 1), _graph(5, 2)]
    module.set_label_means(graphs)
    module.eval()
    out = module._common_step(_batch(graphs), "regression", "node")
    parts = out.parts["global"]
    assert "section" in parts
    assert torch.allclose(out.y_pred, sum(parts.values()))
    assert torch.allclose(out.target_center, parts["section"])
    codes = torch.cat([g.cell_type.argmax(1) for g in graphs])
    assert torch.allclose(parts["section"], module.label_means[codes][out.view.padded_node_idx])

    changed = [g.clone() for g in graphs]
    for g in changed:
        g.x[:, [0, 1]] += 5.0  # the hidden genes, in every cell
    again = module._common_step(_batch(changed), "regression", "node")
    assert torch.allclose(again.y_pred, out.y_pred)


def test_the_loss_reads_what_the_label_leaves():
    net = _model()
    module = _stir(net.module)
    graphs = [_graph(4, 1)]
    module.set_label_means(graphs)
    module.eval()
    out = module._common_step(_batch(graphs), "regression", "node")

    class Plan:
        target_center = torch.full((G,), 123.0)  # never used when a per-cell centre is given

    from interscale.train._trainingplans import TrainingPlan as Real

    y_pred, y_true = Real.centred(Plan, out.y_pred, out.y_true, out.target_center)
    # The prediction left is the decoder's own, the target what the label's mean leaves.
    assert torch.allclose(y_pred, out.y_pred - out.parts["global"]["section"])
    assert torch.allclose(y_true, out.y_true - out.target_center)
    # Without one: the training mean, as before.
    plain_pred, _ = Real.centred(Plan, out.y_pred, out.y_true)
    assert torch.allclose(plain_pred, out.y_pred - 123.0)


def test_without_means_it_refuses_to_predict():
    net = _model()
    with pytest.raises(ValueError, match="no label means"):
        net.module.section_input(_batch([_graph(4, 1)]))


def test_the_switch():
    from interscale.config import _validate

    with pytest.raises(ValueError, match="label_offset_key"):
        _validate(_cfg(key=""))
    with pytest.raises(ValueError, match="does not reach the graphs"):
        _validate(_cfg(extra=[]))
    with pytest.raises(ValueError, match="center_targets"):
        _validate(_cfg(center=False))
    with pytest.raises(ValueError, match="read under section_offset label_mean"):
        _validate(_cfg(section="none"))


def test_the_checkpoint_name_carries_the_label_and_a_long_gene_list_fits_a_file_name():
    from interscale.tl.utils import _decoder_tag

    cfg = _cfg()
    assert "-lab-cell-type" in _decoder_tag(cfg)
    off = _cfg(section="none", key="")
    assert "lab" not in _decoder_tag(off)
    from interscale.tl.utils import _training_tag as tag

    long_list = _cfg(section="none", key="", center=False, extra=[])
    long_list.dataset.mask_genes = [f"GENE{k}" for k in range(500)]
    short = _cfg(section="none", key="", center=False, extra=[])
    assert "gs1-g0-g1" in tag(short)  # a short list keeps its readable name, as every earlier checkpoint
    name = tag(long_list)
    assert len(name) < 100 and "500genes-" in name
    other = _cfg(section="none", key="", center=False, extra=[])
    other.dataset.mask_genes = [f"GENE{k}" for k in range(1, 501)]
    assert tag(other) != name


def test_off_the_model_holds_nothing_new():
    net = _model(section="none", key="", center=False, extra=[])
    assert not hasattr(net.module, "label_means")


def _heads(pred, true, hidden):
    from interscale.evaluation.masked_probing import HeadPredictions

    return HeadPredictions(heads=pred, true=true, hidden=hidden, depth_um=np.zeros(len(true)))


def test_within_groups_scores_a_head_that_only_knows_the_group_zero_and_the_variation_inside_it_one():
    from interscale.evaluation.masked_probing import score_heads, within_groups

    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b"], 50)
    level = np.where(groups[:, None] == "a", 3.0, 9.0) * np.ones((100, G))
    inside = rng.normal(size=(100, G))
    true = level + inside
    hidden = np.ones((100, G), dtype=bool)
    hidden[:, 5] = False
    pred = within_groups(_heads({"group": level.copy(), "follows": true + 7.0, "both": true.copy()}, true, hidden), groups)
    # The group's level is gone from every head and the truth; each head's own constant shift too.
    assert np.allclose(pred.heads["group"][:, :5], 0.0)
    assert np.allclose(pred.heads["follows"][:, :5], pred.true[:, :5])
    # Only hidden entries enter the mean: an unhidden gene's value changes nothing in the hidden ones'.
    hidden_values = true.copy()
    hidden_values[:, 5] += 100.0
    again = within_groups(_heads({"x": hidden_values}, hidden_values, hidden), groups)
    assert np.allclose(again.true[:, :5], pred.true[:, :5])
    # Within a gene too: a visible cell's value does not enter that gene's mean.
    some = hidden.copy()
    some[:10, 0] = False
    moved = true.copy()
    moved[:10, 0] += 1000.0
    first = within_groups(_heads({"x": true}, true, some), groups)
    second = within_groups(_heads({"x": moved}, moved, some), groups)
    assert np.allclose(first.true[10:, 0], second.true[10:, 0])
    scores = score_heads(pred, [f"g{k}" for k in range(G)], center=np.zeros(G), gene_metrics=("cosine",))
    assert scores


def test_the_k_nearest_references_read_no_hidden_value_and_a_saved_fit_round_trips():
    from interscale.evaluation.masked_probing import FittedReferences

    graphs = [_graph(5, 1), _graph(5, 2), _graph(5, 3)]
    fitted = FittedReferences(graphs, spatial_unit_um=1.0, k_nearest=(2, 4), alphas=(1.0,))
    assert set(fitted.models) == {"ref_own", "ref_own+context"}
    batch = _batch([_graph(5, 4)])
    hidden = batch.gene_mask.bool()
    first = fitted(batch, hidden)
    changed = batch.clone()
    changed.x[:, [0, 1]] += 50.0
    second = fitted(changed, hidden)
    for name in first:
        assert torch.allclose(first[name][:, [0, 1]], second[name][:, [0, 1]])
    copy = FittedReferences.from_state(fitted.state())
    assert torch.allclose(copy(batch, hidden)["ref_own+context"], first["ref_own+context"])
    # A fit saved before k_nearest existed still loads, as one without it.
    old = FittedReferences([_graph(5, 1), _graph(5, 2)], spatial_unit_um=1.0, alphas=(1.0,)).state()
    del old["k_nearest"]
    assert FittedReferences.from_state(old).k_nearest == ()


def test_imitation_labels_are_fitted_on_training_cells_and_leave_real_annotation_alone():
    from interscale.tl._preprocessing import imitation_labels

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.1, size=(60, 5)), rng.normal(5, 0.1, size=(60, 5))]).astype(np.float32)
    adata = AnnData(X=x)
    train = np.r_[np.ones(60, bool), np.zeros(60, bool)]
    train[:30] = False
    train[60:90] = True
    imitation_labels(adata, "clusters", train=train, n_clusters=2, n_pcs=2)
    labels = adata.obs["clusters"].astype(str).to_numpy()
    assert len(set(labels[:60])) == 1 and len(set(labels[60:])) == 1 and labels[0] != labels[-1]
    # A held-out cell cannot take a centre: far-away held-out cells leave the two training groups apart.
    far = AnnData(X=np.concatenate([x[train], np.full((60, 5), 1000.0, dtype=np.float32)]))
    imitation_labels(far, "clusters", train=np.r_[np.ones(60, bool), np.zeros(60, bool)], n_clusters=2, n_pcs=2)
    kept = far.obs["clusters"].astype(str).to_numpy()[:60]
    assert len(set(kept)) == 2
    adata.obs["cell_type"] = "kept"
    imitation_labels(adata, "cell_type", train=train)
    assert set(adata.obs["cell_type"]) == {"kept"}


ARMS = {
    "channels_label": (
        "  mask_strategy: gene\n  mask_percentage: 1.0\n  mask_channel_fraction: 0.5\n  extra_obs_keys: ['cell_type']\n",
        "    section_offset: label_mean\n    label_offset_key: cell_type\n    own_cell_skip: True\n    own_cell_skip_input: raw\n",
        "",
    ),
    "channels_space": (
        "  mask_strategy: gene\n  mask_percentage: 1.0\n  mask_channel_fraction: 0.5\n",
        "    own_cell_skip: True\n    own_cell_skip_input: raw\n",
        "      pe:\n        node: ['sinusoidal']\n",
    ),
    "space_alone": (
        "  mask_strategy: gene\n  mask_percentage: 1.0\n  mask_genes: [g0, g1, g2, g3, g4, g5]\n",
        "",
        "      pe:\n        node: ['sinusoidal']\n",
    ),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_each_arm_trains_through_hierarchy_and_crops(tmp_path, arm):
    import scanpy as sc

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, crop_kwargs, gene_set_mask_kwargs
    from interscale.model.global_model import GlobalModel
    from interscale.tl import prepare_geome_dataset

    masking, decoder, pe = ARMS[arm]
    rng = np.random.default_rng(0)
    per = 60
    adata = AnnData(X=np.abs(rng.normal(size=(per * 3, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(per * 3, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2", "s3"], per))
    adata.obs["split"] = pd.Categorical(np.repeat(["train", "val", "test"], per))
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["a", "b", "c"], size=per * 3))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "arm.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "arm.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  global_component:\n    name: self-attn-transformer\n    parameters:\n"
        "      max_seq_len: 256\n      num_layers: 1\n      n_heads: 2\n      long_range_attention: False\n"
        "      type_gex_embedding: Linear\n"
        "      hierarchy:\n        kind: coarse_to_fine\n        cells_per_cluster: [8.0]\n        readout: concat\n"
        f"{pe}"
        f"  decoder:\n    type: linear\n{decoder}  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  center_targets: True\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: armtest\n  prediction_task: regression\n"
        f"  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n{masking}"
        "  spatial_key: spatial\n  batch_size: 1\n  evaluate_test: False\n"
        "  crop:\n    size_um: 400.0\n    shape: square\n    per_graph: 2\n    min_cells: 5\n"
        "  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    data = sc.read_h5ad(cfg.dataset.h5ad_data)
    GlobalModel._setup_anndata(
        adata=data, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    net = GlobalModel(data, cfg=cfg)
    pyg, _ = prepare_geome_dataset(data, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=1, mask_percentage=1.0, mask_strategy="gene", learning_type="node",
        **gene_set_mask_kwargs(cfg, data), **crop_kwargs(cfg),
    )
    before = next(iter(net.module.parameters())).detach().clone()
    net.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    assert not torch.equal(next(iter(net.module.parameters())).detach().cpu(), before.cpu())
    if arm == "channels_label":
        assert net.module.label_means.shape == (3, G)
        # The training graphs' means, never a held-out section's.
        expected = _manual_means(list(dm.train_data)).float()
        assert torch.allclose(net.module.label_means.cpu().float(), expected, atol=1e-5)
        assert not torch.allclose(net.module.label_means.cpu().float(), _manual_means(list(dm.val_data)).float(), atol=1e-3)
