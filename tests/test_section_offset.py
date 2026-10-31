"""The section offset (``model.decoder.section_offset``) and the within-section scores (``masked_probing.within_section``;
roadmap Tier 1, item 1; notes.md 2026-10-31: on CosMx 80-89% of the centred cosine was each section's own offset).

Pinned here:

* **the mean** -- ``tl.masking.section_visible_mean`` is each gene's mean over the visible entries of the cell's graph,
  never a hidden value, 0 for a gene with none;
* **the part** -- a GlobalModel and a LocalModel add ``section = w * mean`` (``w`` per gene, starting at 1) to their prediction
  as a named part, read on the masked batch in training and on the clean one at inference; no hidden value reaches any
  prediction; ``w`` learns; off, the model holds no new state and every other weight starts as before;
* **the switch** -- validator, checkpoint tag, the combined model refuses it, ``predict`` refuses to run without the part;
* **the scores** -- ``collect_head_predictions`` records the same means; ``within_section`` subtracts them from heads and truth,
  so the slide mean is the centre (NaN) and a head that adds nothing to the section mean scores nothing.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

from interscale.tl.masking import section_visible_mean

E, G = 8, 6
SPACING = 100.0


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _lattice(side, seed, hidden=()):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= SPACING * 1.01).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    # A section-specific offset on top of the cell values, so the two graphs differ in their means.
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0 + seed, edge_index=edges, pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    data.mask[list(hidden)] = True
    return data


HIDDEN_A, HIDDEN_B = [0, 1, 2, 7, 14, 20], [3, 4, 9]


def _batch():
    batch = Batch.from_data_list([_lattice(6, seed=1, hidden=HIDDEN_A), _lattice(5, seed=4, hidden=HIDDEN_B)])
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def _cfg(section="visible_mean", local=False, max_seq_len=64, programme=False):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    if local:
        cfg.model.local_component.name = "GCN"
        cfg = get_local_component_cfg(cfg, "GCN")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    cfg.model.decoder.section_offset = section
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = max_seq_len, 2, 2, 0.0
    params.type_gex_embedding = "Linear"
    params.pe.bias = ["distance"]
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = 60.0, 600.0
    cfg.dataset.mask_strategy = "gene" if programme else "node"
    cfg.dataset.mask_percentage = 1.0 if programme else 0.3
    if programme:
        cfg.dataset.mask_genes = ["g0", "g1"]
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    return cfg


def _adata():
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    return adata


def _model(kind="global", section="visible_mean", seed=0, **options):
    from interscale.config import _validate
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel
    from interscale.model.local_model import LocalModel

    cfg = _cfg(section, local=kind != "global", **options)
    if kind == "combined":
        cfg.model.decoder.dual_decoder = True
    _validate(cfg)
    cls = {"global": GlobalModel, "local": LocalModel, "combined": CombinedModel}[kind]
    adata = _adata()
    cls._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(seed)
    return cls(adata, cfg=cfg)


def _manual_mean(batch, hidden_rows):
    """Per cell, each gene's mean over the visible cells of its graph, written out with numpy."""
    x, graph = batch.x.numpy(), batch.batch.numpy()
    visible = np.ones(len(x), bool)
    visible[hidden_rows] = False
    out = np.zeros_like(x)
    for g in np.unique(graph):
        rows = graph == g
        out[rows] = x[rows & visible].mean(0)
    return out


def _hidden_rows():
    n_a = 36
    return HIDDEN_A + [n_a + i for i in HIDDEN_B]


# --------------------------------------------------------------------------- the mean


def test_the_mean_reads_visible_entries_only_and_gives_a_gene_with_none_zero():
    batch = _batch()
    hidden = torch.zeros(batch.num_nodes, G, dtype=torch.bool)
    hidden[_hidden_rows()] = True
    hidden[:, 5] = True  # a gene hidden in every cell (programme masking)
    mean = section_visible_mean(batch.x, ~hidden, batch.batch)
    want = _manual_mean(batch, _hidden_rows())
    assert np.allclose(mean[:, :5].numpy(), want[:, :5], atol=1e-6)
    assert (mean[:, 5] == 0).all()
    moved = batch.x.clone()
    moved[hidden] += torch.rand(int(hidden.sum())) * 50.0
    assert torch.equal(section_visible_mean(moved, ~hidden, batch.batch), mean)
    assert torch.allclose(section_visible_mean(batch.x[:36], ~hidden[:36], None), mean[:36])  # one graph


# --------------------------------------------------------------------------- the part


@pytest.mark.parametrize(
    ("kind", "max_seq_len"), [("global", 64), ("global", 20), ("local", 64)], ids=["global", "global, subsampled", "local"]
)
def test_the_prediction_is_the_decoder_plus_the_weighted_section_mean(kind, max_seq_len):
    model = _model(kind, max_seq_len=max_seq_len)
    module = _stir(model.module).eval()
    assert module.section_offset == "visible_mean"
    batch = _batch()
    with torch.no_grad():
        random.seed(0)
        out = module._common_step(batch, "regression", "node")
    head = "global" if kind == "global" else "local"
    parts = out.parts[head]
    assert "section" in parts and len(parts) == 2
    assert torch.allclose(sum(parts.values()), out.y_pred, atol=1e-6)
    idx = out.view.padded_node_idx
    idx = torch.arange(batch.num_nodes) if idx is None else idx
    want = torch.as_tensor(_manual_mean(batch, _hidden_rows()))[idx] * module.section_offset_weight.detach()
    assert torch.allclose(parts["section"], want, atol=1e-5)


@pytest.mark.parametrize("kind", ["global", "local"])
def test_no_hidden_value_reaches_a_prediction_and_a_visible_one_reaches_every_cell_of_its_section(kind):
    module = _stir(_model(kind).module).eval()
    batch = _batch()

    def predict(b):
        with torch.no_grad():
            random.seed(0)
            out = module._common_step(b, "regression", "node")
        idx = out.view.padded_node_idx
        order = torch.arange(b.num_nodes) if idx is None else idx
        y = torch.empty_like(out.y_pred)
        y[torch.argsort(order)] = out.y_pred[torch.argsort(order)]
        return out.y_pred, order

    before, order = predict(batch)
    moved = batch.clone()
    moved.x = batch.x.clone()
    moved.x[_hidden_rows()] += torch.rand(len(_hidden_rows()), G) * 50.0
    after, _ = predict(moved)
    assert torch.equal(after, before)
    far = batch.clone()
    far.x = batch.x.clone()
    far.x[35, 2] += 30.0  # a visible corner cell of the first section: its mean moves every cell there
    after, _ = predict(far)
    first = (batch.batch[order] == 0) & ~torch.isin(order, torch.tensor([35]))
    changed = (after - before)[:, 2].abs() > 1e-4
    assert changed[first].all()  # also cells far beyond any reach: the section mean
    assert not changed[batch.batch[order] == 1].any()  # the other section's cells are not moved by it


def test_the_weight_starts_at_one_learns_and_off_nothing_changes():
    on, off = _model("global"), _model("global", section="none")
    assert torch.equal(on.module.section_offset_weight.detach(), torch.ones(G))
    extra = set(on.module.state_dict()) - set(off.module.state_dict())
    assert extra == {"section_offset_weight"} and not set(off.module.state_dict()) - set(on.module.state_dict())
    assert all(torch.equal(v, on.module.state_dict()[k]) for k, v in off.module.state_dict().items())
    assert off.module.section_input(_batch()) is None
    module = _stir(on.module)
    random.seed(0)
    out = module._common_step(_batch(), "regression", "node")
    out.y_pred.pow(2).sum().backward()
    assert module.section_offset_weight.grad is not None and module.section_offset_weight.grad.abs().sum() > 0


def test_predict_refuses_to_run_without_the_part_and_inference_reads_the_clean_batch():
    module = _stir(_model("global").module).eval()
    batch = _batch()
    with pytest.raises(ValueError, match="section_offset needs"):
        module.predict(torch.zeros(37, 2, E), torch.zeros(2, 37, dtype=torch.bool), "node", node_idx=torch.arange(72))
    local = _stir(_model("local").module).eval()
    with pytest.raises(ValueError, match="section_offset needs"):
        local.predict(torch.zeros(4, E), "node")
    clean = local.section_input(batch, masked=False)
    everything = torch.ones_like(batch.x, dtype=torch.bool)
    assert torch.allclose(clean, section_visible_mean(batch.x, everything, batch.batch) * local.section_offset_weight)
    masked = local.section_input(batch, masked=True)
    assert not torch.allclose(clean, masked)


@pytest.mark.parametrize("kind", ["global", "local"])
def test_under_programme_masking_a_gene_hidden_everywhere_gets_no_offset_and_no_hidden_value_reaches_it(kind):
    module = _stir(_model(kind, programme=True).module).eval()
    batch = _batch()
    batch.gene_mask = torch.zeros(batch.num_nodes, G, dtype=torch.bool)
    batch.gene_mask[:, [0, 1]] = True
    batch.mask = torch.ones(batch.num_nodes, dtype=torch.bool)

    def run(b):
        with torch.no_grad():
            random.seed(0)
            out = module._common_step(b, "regression", "node")
        return out.parts["global" if kind == "global" else "local"]["section"], out.y_pred

    section, before = run(batch)
    assert (section[:, :2] == 0).all() and (section[:, 2:] != 0).all()
    moved = batch.clone()
    moved.x = batch.x.clone()
    moved.x[:, :2] += torch.rand(batch.num_nodes, 2) * 50.0
    assert torch.equal(run(moved)[1], before)


# --------------------------------------------------------------------------- the switch


def test_the_config_and_the_models_refuse_what_it_cannot_carry():
    from interscale.config import _validate_section_offset
    from interscale.tl.utils import _decoder_tag

    _validate_section_offset(_cfg())
    bad = _cfg("per_gene_mean")
    with pytest.raises(ValueError, match="section_offset must be one of"):
        _validate_section_offset(bad)
    for key, value in (("prediction_task", "classification"), ("prediction_level", "graph")):
        cfg = _cfg()
        cfg.dataset[key] = value
        with pytest.raises(ValueError, match="regression at prediction_level node"):
            _validate_section_offset(cfg)
    assert _decoder_tag(_cfg()) == "dec-linear-sec_" and _decoder_tag(_cfg("none")) == ""
    combined = _model("combined")
    with pytest.raises(ValueError, match="CombinedModel does not carry it"):
        combined._refuse_unsupported_options()


# --------------------------------------------------------------------------- the scores


def test_the_evaluation_records_the_means_and_within_section_takes_them_out():
    from interscale.evaluation.masked_probing import collect_head_predictions, score_heads, within_section

    module = _stir(_model("global", section="none").module)
    graphs = [_lattice(6, seed=1, hidden=HIDDEN_A), _lattice(5, seed=4, hidden=HIDDEN_B)]
    for i, g in enumerate(graphs):
        g.obs_names = torch.arange(g.num_nodes) + 100 * i
    pred = collect_head_predictions(
        module,
        graphs,
        prediction_task="regression",
        prediction_level="node",
        mask_strategy="node",
        batch_size=2,
        device=torch.device("cpu"),
        spatial_unit_um=1.0,
        references={"kernels_um": [150.0]},
    )
    batch = Batch.from_data_list(graphs)
    order = (pred.cells - 100 * (pred.cells >= 100)).astype(int) + 36 * (pred.cells >= 100)
    assert np.allclose(pred.section_mean, _manual_mean(batch, _hidden_rows())[order], atol=1e-5)
    inside = within_section(pred)
    assert np.allclose(inside.true, pred.true - pred.section_mean)
    hidden = pred.hidden.any(1)
    assert np.allclose(inside.heads["ref_slide_mean"][hidden], 0.0, atol=1e-5)  # the slide mean is the centre
    scores = score_heads(inside, [f"g{k}" for k in range(G)], center=np.zeros(G))["cells"]
    slide = scores["ref_slide_mean_cosine_centred"].iloc[0]
    assert np.isnan(slide) or abs(slide) < 0.05  # zero up to float32 rounding: its cosine is noise around 0
    assert np.isfinite(scores["global_cosine_centred"].iloc[0])
    with pytest.raises(ValueError, match="no section means"):
        within_section(type(pred)(heads=pred.heads, true=pred.true, hidden=pred.hidden))


def test_a_model_with_the_offset_shows_it_as_a_part_of_its_head():
    from interscale.evaluation.masked_probing import attribution_shares, collect_head_predictions

    module = _stir(_model("global").module)
    graphs = [_lattice(6, seed=1, hidden=HIDDEN_A), _lattice(5, seed=4, hidden=HIDDEN_B)]
    pred = collect_head_predictions(
        module, graphs, prediction_task="regression", prediction_level="node", mask_strategy="node", batch_size=2,
        device=torch.device("cpu"),
    )
    shares = attribution_shares(pred, [f"g{k}" for k in range(G)])
    assert set(shares["part"]) == {"context", "section"}
    total = shares[shares["group"] == "all"].groupby("head")["cov_share"].sum()
    assert np.allclose(total.to_numpy(), 1.0)


# --------------------------------------------------------------------------- through the trainer and inference


def _yaml(tmp_path, *, local):
    import scanpy as sc  # noqa: F401  (the trainer's reader)

    samples = ["s1", "s2", "s3", "s4"]
    rng = np.random.default_rng(0)
    per = 36
    adata = AnnData(X=np.abs(rng.normal(size=(per * 4, G))).astype(np.float32) + 0.05)
    adata.X += np.repeat(np.arange(4), per)[:, None].astype(np.float32)  # a section offset per sample
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(per * 4, 2)).astype(np.float32)
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, per))
    split = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "section.h5ad"
    adata.write_h5ad(h5ad)
    component = "  local_component:\n    name: GCN\n" if local else ""
    path = tmp_path / "section.yaml"
    path.write_text(
        f"model:\n  n_embed: 8\n{component}  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 64\n      num_layers: 1\n"
        "      n_heads: 2\n      long_range_attention: False\n      type_gex_embedding: Linear\n"
        f"  decoder:\n    type: linear\n    section_offset: visible_mean\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 2\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: sectiontest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 250\n    library_key: sample\n"
    )
    return path


@pytest.mark.parametrize("local", [False, True], ids=["GlobalModel", "LocalModel"])
def test_a_model_trains_names_its_checkpoint_and_predicts_with_the_offset_at_inference(tmp_path, monkeypatch, local):
    import scanpy as sc

    import interscale.model.global_model as global_model
    import interscale.model.local_model as local_model
    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.model.global_model import GlobalModel
    from interscale.model.local_model import LocalModel
    from interscale.tl import prepare_geome_dataset

    cfg = load_config(str(_yaml(tmp_path, local=local)))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    cls = LocalModel if local else GlobalModel
    cls._setup_anndata(
        adata=adata, prediction_task="regression", layer_key="log1p_norm", sample_key_list=["sample"], prediction_obs=None
    )
    net = cls(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    net.train(max_epochs=2, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    assert not torch.equal(net.module.section_offset_weight.detach().cpu(), torch.ones(G))  # the weight trains
    assert any("dec-linear-sec" in p.name for p in (tmp_path / "out").iterdir())

    # Inference decodes outside the step, on the clean graph (a Data without `batch`): the offset is read there too.
    graph = pyg[0]
    while isinstance(graph, (list, tuple)):  # one list of graphs per split
        graph = graph[0]
    monkeypatch.setattr(local_model if local else global_model, "prepare_a2d_dataset", lambda cfg: lambda a: [[graph], None])
    if not local:
        monkeypatch.setattr(global_model, "attach_positional_inputs", lambda pyg, cfg: None)
    saved = {}
    monkeypatch.setattr(net, "save_evaluation_results", lambda adata, prefix, **out: saved.update(out) or adata)
    sub = adata[: graph.num_nodes].copy()
    net.get_model_output(sub)
    found = saved["y_pred_local_df" if local else "y_pred_global_df"].to_numpy(dtype=float)
    assert np.isfinite(found).all() and found.shape == (graph.num_nodes, G)
