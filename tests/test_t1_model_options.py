"""Two model-side options of roadmap.md, T1: a distance prior with a length scale per head, and visible keys only.

``pe.distance.kind: scales`` (``ScaledDistanceBias``): head ``h`` gets the window ``-strength * phi(d / l_h)``, the scales ``l_h``
spaced geometrically over ``[scale_min_um, scale_max_um]`` -- some heads local, some global from step 0, each head's scale known.
Pinned here: the scales, the window in both shapes, the dense and the pair form agree, frozen or learned, and -- through the real
attention with the content logits zeroed -- each head's attention is the softmax of its own window, ordered by scale.

``attention.visible_keys_only``: a cell hidden whole is a query but never a key (K5: the hidden cells otherwise form one block of
identical tokens that the others can attend to for nothing). Pinned here: through the real model and a batch of two graphs of
different sizes (so the left padding matters), hidden keys get exactly zero weight in every layer and head, visible keys and CLS
stay open and every row sums to one; off it they get weight; with nothing hidden it changes nothing; a clean pass is untouched; the
config refuses what it cannot do; checkpoint names carry it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.module.global_modules.positional_encodings import ScaledDistanceBias
from interscale.tl.utils import get_model_filename_prefix

G, E, HEADS, SPACING = 6, 16, 4, 100.0


# --------------------------------------------------------------------------- the scales


def _bias(**kwargs):
    options = {
        "n_heads": HEADS,
        "scale_min_um": 20.0,
        "scale_max_um": 2000.0,
        "shape": "gaussian",
        "strength": 1.0,
        "frozen": False,
    }
    return ScaledDistanceBias(**(options | kwargs))


def test_the_scales_are_spaced_geometrically_across_the_heads():
    scales = _bias().scales()
    assert scales.tolist() == pytest.approx([20.0, 20.0 * 100.0 ** (1 / 3), 20.0 * 100.0 ** (2 / 3), 2000.0], rel=1e-5)
    assert torch.allclose(scales[1:] / scales[:-1], torch.full((3,), 100.0 ** (1 / 3)))
    assert _bias(n_heads=1).scales().tolist() == pytest.approx([200.0])  # one head: the geometric mean of the ends
    assert _bias(n_heads=2).scales().tolist() == pytest.approx([20.0, 2000.0])


@pytest.mark.parametrize("shape, window", [("gaussian", lambda u: 0.5 * u**2), ("exponential", lambda u: u)])
def test_a_head_reads_its_window_of_the_distance_and_the_strength_scales_it(shape, window):
    bias = _bias(shape=shape, strength=2.5)
    dist = torch.tensor([[0.0, 100.0, 1000.0], [100.0, 0.0, 2500.0], [1000.0, 2500.0, 0.0]])[None]  # [1, 3, 3]
    out = bias(dist)  # [1, heads, 3, 3]
    assert out.shape == (1, HEADS, 3, 3)
    for h, scale in enumerate(bias.scales().tolist()):
        assert torch.allclose(out[0, h], -2.5 * window(dist[0] / scale), atol=1e-5)
    assert (out[0, :, range(3), range(3)] == 0).all()  # no penalty at distance 0
    assert (out <= 0).all()


def test_a_prior_is_not_zero_at_step_zero_unlike_the_learned_profile():
    assert (_bias()(torch.tensor([[[0.0, 300.0], [300.0, 0.0]]])) != 0).any()


@pytest.mark.parametrize("shape", ["gaussian", "exponential"])
def test_the_pair_form_is_the_dense_form(shape):
    rng = np.random.default_rng(0)
    pos = torch.tensor(rng.uniform(0, 3000, size=(2, 7, 2)), dtype=torch.float32)  # [B, S, 2], um
    bias = _bias(shape=shape, strength=1.7)
    dense = bias(torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist"))  # [B, heads, S, S]
    graph, query, key = (torch.tensor(v) for v in ([0, 1, 1, 0], [2, 0, 6, 5], [5, 3, 1, 5]))
    pairs = bias.pairs(pos, graph, query, key)
    assert torch.allclose(pairs, dense[graph, :, query, key], atol=1e-4)


def test_the_built_module_honours_frozen_and_the_settings():
    frozen = _module(_cfg(distance={"kind": "scales", "frozen": True, "scale_min_um": 50.0, "strength": 3.0}))
    learned = _module(_cfg(distance={"kind": "scales", "frozen": False}))
    frozen_bias = frozen.attention_bias.encoders["distance"]
    learned_bias = learned.attention_bias.encoders["distance"]
    assert not list(frozen_bias.parameters()) and len(list(learned_bias.parameters())) == 2
    assert frozen_bias.scales()[0].item() == pytest.approx(50.0, rel=1e-5) and (frozen_bias.strength == 3.0).all()
    assert learned_bias.scales()[0].item() == pytest.approx(20.0, rel=1e-5)


def test_frozen_scales_are_buffers_and_learned_ones_are_parameters_with_the_same_state():
    frozen, learned = _bias(frozen=True), _bias(frozen=False)
    assert not list(frozen.parameters()) and {n for n, _ in frozen.named_buffers()} == {"log_scale", "strength"}
    assert {n for n, _ in learned.named_parameters()} == {"log_scale", "strength"}
    assert set(frozen.state_dict()) == set(learned.state_dict())
    learned.load_state_dict(frozen.state_dict())
    out = learned(torch.tensor([[[0.0, 300.0], [300.0, 0.0]]]))
    out.sum().backward()
    assert learned.log_scale.grad.abs().sum() > 0 and learned.strength.grad.abs().sum() > 0


def test_the_profile_reads_the_window_out_per_head():
    knots, values = _bias(shape="exponential").profile(max_dist=1000.0, num=5)
    assert knots.tolist() == [0.0, 250.0, 500.0, 750.0, 1000.0] and values.shape == (HEADS, 5)
    assert torch.allclose(values[0], -knots / 20.0, atol=1e-4)


# --------------------------------------------------------------------------- helpers: a real module and a batch of two graphs


def _cfg(*, distance=None, attention=None, strategy="node", rate=0.3):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.type_gex_embedding = 64, "Linear"
    params.n_heads, params.num_layers, params.dropout_global = HEADS, 2, 0.0
    cfg.model.n_embed = E
    if distance is not None:
        params.pe.bias = ["distance"]
        for key, value in distance.items():
            params.pe.distance[key] = value
    for key, value in (attention or {}).items():
        if isinstance(value, dict):
            for inner, leaf in value.items():
                params.attention[key][inner] = leaf
        else:
            params.attention[key] = value
    cfg.dataset.mask_strategy, cfg.dataset.mask_percentage = strategy, rate
    cfg.dataset.spatial_key, cfg.dataset.spatial_unit_um = "spatial", 1.0
    cfg.dataset.name = "d"
    return cfg


def _module(cfg, seed=0):
    from interscale.module.base import GlobalModule

    torch.manual_seed(seed)
    return GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=E,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=cfg.dataset.mask_percentage,
        mask_strategy=cfg.dataset.mask_strategy,
        block_masking=False,
        type_gex_embedding="Linear",
    ).eval()


def _graph(side, hidden, seed):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * SPACING
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    data = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=torch.zeros((2, 0), dtype=torch.long), pos=pos)
    data.mask = torch.zeros(n, dtype=torch.bool)
    data.mask[list(hidden)] = True
    return data


def _batch(hidden_a=(3, 4, 9, 10, 15, 16), hidden_b=(0, 1, 6, 7)):
    """Graphs of 36 and 25 cells (so the second is left-padded), each with a block of cells hidden whole."""
    graphs = [_graph(6, hidden_a, seed=0), _graph(5, hidden_b, seed=1)]
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch, graphs


def _maps(module, batch):
    with torch.no_grad(), module.recording_attention() as record:
        module._common_step(batch, "regression", "node")
    return record[0]


def _token_columns(entry, graphs):
    """Per graph: the token positions of its cells, and which of them are hidden whole."""
    out = []
    for b, graph in enumerate(graphs):
        positions = np.flatnonzero(~entry["src_padding_mask"][b].bool().numpy())
        nodes = torch.as_tensor(entry["index_nodes"][b], dtype=torch.long)
        out.append((positions, graph.mask[nodes].numpy()))
    return out


# --------------------------------------------------------------------------- the scales, through the real attention


def test_with_zeroed_content_logits_each_head_attends_by_its_own_window_and_wider_scales_look_further():
    cfg = _cfg(
        distance={"kind": "scales", "scale_min_um": 100.0, "scale_max_um": 3000.0, "shape": "gaussian", "frozen": True}
    )
    module = _module(cfg, seed=0)
    for layer in module.transformer_encoder.layers:  # no content: the attention is the prior alone
        layer.self_attn.in_proj_weight.data.zero_()
        layer.self_attn.in_proj_bias.data.zero_()
    batch, graphs = _batch(hidden_a=(), hidden_b=())
    entry = _maps(module, batch)
    maps = entry["maps"][0]  # first layer: [B * heads, S + 1, S + 1]
    scales = module.attention_bias.encoders["distance"].scales()
    pos = graphs[0].pos
    positions, _ = _token_columns(entry, graphs)[0]
    nodes = torch.as_tensor(entry["index_nodes"][0], dtype=torch.long)
    query = 17  # a cell of the first graph
    row_token = positions[(nodes == query).nonzero().item()]
    distance = torch.cdist(pos[query][None], pos[nodes])[0]  # to every kept cell, in token order
    spans = []
    for h in range(HEADS):
        logits = -0.5 * (distance / scales[h]) ** 2
        logits[(nodes == query).nonzero().item()] = float("-inf")  # no attention to itself
        expected = torch.softmax(torch.cat([logits, torch.zeros(1)]), dim=0)  # CLS carries no bias
        got = maps[h, row_token]
        assert torch.allclose(got[torch.as_tensor(positions)], expected[:-1], atol=1e-5), h
        assert got[-1] == pytest.approx(float(expected[-1]), abs=1e-5)
        spans.append(float((expected[:-1] * distance).sum() / expected[:-1].sum()))
    assert spans == sorted(spans) and spans[0] < 0.5 * spans[-1]  # a wider scale looks further


def test_the_scales_bias_needs_no_derived_range_and_the_kind_is_tagged():
    cfg = _cfg(distance={"kind": "scales"})
    assert cfg.model.global_component.parameters.pe.distance.max_dist == 0.0
    _module(cfg)  # builds: a `profile` with max_dist 0 would refuse
    plain = get_model_filename_prefix(_cfg(distance={"kind": "linear"}), False, True)
    names = {
        "scales": get_model_filename_prefix(cfg, False, True),
        "exp": get_model_filename_prefix(_cfg(distance={"kind": "scales", "shape": "exponential"}), False, True),
        "wide": get_model_filename_prefix(_cfg(distance={"kind": "scales", "scale_min_um": 300.0}), False, True),
        "frozen": get_model_filename_prefix(_cfg(distance={"kind": "scales", "frozen": True}), False, True),
        "strong": get_model_filename_prefix(_cfg(distance={"kind": "scales", "strength": 2.0}), False, True),
    }
    assert len(set(names.values()) | {plain}) == 6
    assert "distance-scales-g20-2000_" in names["scales"] and "distance-scales-e20-2000_" in names["exp"]
    assert "-fz" in names["frozen"] and "-s2" in names["strong"]
    assert "pe-distance_" in get_model_filename_prefix(_cfg(distance={"kind": "profile"}), False, True)  # unchanged


@pytest.mark.parametrize(
    "change, message",
    [
        ({"kind": "bogus"}, "kind must be one of"),
        ({"scale_min_um": 0.0}, "scale_min_um"),
        ({"scale_min_um": 3000.0, "scale_max_um": 100.0}, "scale_min_um"),
        ({"shape": "triangle"}, "shape must be one of"),
        ({"strength": 0.0}, "strength must be > 0"),
    ],
)
def test_the_config_refuses_a_bad_distance_prior(change, message):
    from interscale.config import _validate_pe

    cfg = _cfg(distance={"kind": "scales"} | change)
    with pytest.raises(ValueError, match=message):
        _validate_pe(cfg)
    _validate_pe(_cfg(distance={"kind": "scales"}))


# --------------------------------------------------------------------------- visible keys only


def _on(**attention):
    return _cfg(attention={"visible_keys_only": True} | attention)


def test_hidden_keys_get_exactly_zero_weight_and_everything_else_stays_open():
    batch, graphs = _batch()
    entry = _maps(_module(_on()), batch)
    maps = entry["maps"]  # [layers, B * heads, S + 1, S + 1]
    for b, (positions, hidden) in enumerate(_token_columns(entry, graphs)):
        rows = slice(b * HEADS, (b + 1) * HEADS)
        graph_maps = maps[:, rows]
        assert hidden.sum() > 0
        assert (
            graph_maps[..., positions[hidden]] == 0
        ).all()  # no query attends to a hidden cell, in any layer or head
        assert torch.allclose(
            graph_maps[..., :, :].sum(-1), torch.ones(graph_maps.shape[:-1]), atol=1e-5
        )  # rows sum to 1
        for p in positions[hidden]:  # hidden cells still query: their own rows attend to something
            assert (graph_maps[:, :, p, :].sum(-1) > 0.99).all()
        visible = positions[~hidden]
        assert (graph_maps[0][:, visible[0], visible].sum(-1) > 0).all()  # a visible query reaches visible keys
        assert (graph_maps[..., -1] > 0).all()  # CLS stays open for every query


def test_off_the_hidden_cells_are_keys_like_any_other():
    batch, graphs = _batch()
    entry = _maps(_module(_cfg()), batch)
    for b, (positions, hidden) in enumerate(_token_columns(entry, graphs)):
        assert (entry["maps"][:, b * HEADS : (b + 1) * HEADS][..., positions[hidden]] > 0).any()


def test_with_nothing_hidden_it_changes_nothing_and_a_clean_pass_is_untouched():
    batch, graphs = _batch(hidden_a=(), hidden_b=())
    on, off = _module(_on()), _module(_cfg())
    assert torch.equal(_maps(on, batch)["maps"], _maps(off, batch)["maps"])
    hidden_batch, _ = _batch()
    clean = []  # the evaluation's clean pass: no hidden key is blocked
    for module in (on, off):
        embedding = module.gex_embedding(hidden_batch)
        clean.append(module.common_step_local_to_global(hidden_batch, embedding, eval_step=True).attention_mask)
    assert torch.equal(clean[0], clean[1])


def test_it_carries_the_distance_prior_and_the_other_dense_normalisers():
    batch, graphs = _batch()
    for options in (
        {"cfg": _cfg(attention={"visible_keys_only": True}, distance={"kind": "scales"})},
        {"cfg": _cfg(attention={"visible_keys_only": True, "kind": "topk", "topk": {"k": 5}})},
    ):
        entry = _maps(_module(options["cfg"]), batch)
        for b, (positions, hidden) in enumerate(_token_columns(entry, graphs)):
            assert (entry["maps"][:, b * HEADS : (b + 1) * HEADS][..., positions[hidden]] == 0).all()


def test_a_pattern_backend_refuses_it_at_the_forward_pass():
    cfg = _cfg(attention={"visible_keys_only": True, "kind": "performer"})
    batch, _ = _batch()
    with pytest.raises(ValueError, match="dense attention mask"):
        _module(cfg)._common_step(batch, "regression", "node")


def _validate(cfg):
    from interscale.config import _validate_attention

    _validate_attention(cfg)


def test_the_config_takes_dense_backends_with_cells_hidden_whole():
    _validate(_on())
    _validate(_on(kind="entmax"))
    _validate(_on(kind="topk"))
    gene = _on()
    gene.dataset.mask_strategy = "gene"
    gene.dataset.mask_block_fraction, gene.dataset.mask_block_radius_um = 0.3, 500.0
    _validate(gene)


@pytest.mark.parametrize(
    "change, message",
    [
        pytest.param(
            lambda c: c.model.global_component.parameters.attention.__setitem__("kind", "performer"),
            "dense",
            id="performer",
        ),
        pytest.param(
            lambda c: c.model.global_component.parameters.attention.__setitem__("kind", "linear"), "dense", id="linear"
        ),
        pytest.param(
            lambda c: setattr(c.model.global_component.parameters.pooling, "kind", "grid"), "pooled", id="pooling"
        ),
        pytest.param(lambda c: setattr(c.dataset, "mask_genes", ["g0"]), "hides none", id="gene set"),
        pytest.param(lambda c: setattr(c.dataset, "mask_percentage", 0.0), "nothing is masked", id="nothing hidden"),
    ],
)
def test_the_config_refuses_what_it_cannot_do(change, message):
    cfg = _on()
    cfg.dataset.mask_strategy = "node"
    change(cfg)
    if cfg.model.global_component.parameters.pooling.kind == "grid":
        cfg.model.global_component.parameters.pooling.size_um = 200.0
    with pytest.raises(ValueError, match=message):
        _validate(cfg)


def test_the_config_refuses_gene_masking_without_discs():
    cfg = _on()
    cfg.dataset.mask_strategy = "gene"
    with pytest.raises(ValueError, match="only spatial blocks"):
        _validate(cfg)


def test_the_checkpoint_name_carries_it_and_the_default_name_is_unchanged():
    on, off = get_model_filename_prefix(_on(), False, True), get_model_filename_prefix(_cfg(), False, True)
    assert "_attn-dense-vk_" in on and "_attn-" not in off
    assert get_model_filename_prefix(_on(center_qk=True), False, True) != on


def test_off_by_default():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_global_component_cfg(get_cfg_defaults(), "self-attn-transformer")
    assert cfg.model.global_component.parameters.attention.visible_keys_only is False
    assert cfg.model.global_component.parameters.pe.distance.kind == "profile"
    assert _module(_cfg()).visible_keys_only is False


# --------------------------------------------------------------------------- through the trainer


def test_a_dual_decoder_model_trains_on_discs_with_both_options(tmp_path):
    import pandas as pd
    import scanpy as sc
    from anndata import AnnData

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule, block_mask_kwargs
    from interscale.model.combined_model import CombinedModel
    from interscale.tl import prepare_geome_dataset

    side = 11
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    lattice = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(np.float32) * SPACING
    samples = ["s1", "s2", "s3", "s4"]
    rng = np.random.default_rng(0)
    n = len(lattice) * len(samples)
    adata = AnnData(X=np.abs(rng.normal(size=(n, G))).astype(np.float32) + 0.05)
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obsm["spatial"] = np.tile(lattice, (len(samples), 1))
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, len(lattice)))
    split = {"s1": "train", "s2": "train", "s3": "val", "s4": "test"}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / "t1.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / "t1.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  local_component:\n    name: GCN\n  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 200\n      num_layers: 1\n"
        "      n_heads: 2\n      long_range_attention: True\n"
        "      attention:\n        visible_keys_only: True\n"
        "      pe:\n        bias: [distance]\n        distance:\n          kind: scales\n"
        "          scale_min_um: 300.0\n          scale_max_um: 3000.0\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n  save: {tmp_path}/out/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 3\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: t1test\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.0\n  mask_block_radius_um: 400.0\n  mask_block_fraction: 0.3\n"
        "  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 150\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    CombinedModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    net = CombinedModel(adata, cfg=cfg)
    bias = net.module.global_module.attention_bias.encoders["distance"]
    start = bias.log_scale.detach().clone()
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg,
        num_workers=0,
        batch_size=2,
        mask_percentage=0.0,
        mask_strategy="node",
        learning_type="node",
        **block_mask_kwargs(cfg),
    )
    net.train(max_epochs=3, datamodule=dm, early_stopping=False, wandb_use=False)
    assert net.is_trained_
    losses = [row["val_loss"] for row in net.history_.history if "val_loss" in row]
    assert losses and all(np.isfinite(losses))
    assert not torch.equal(bias.log_scale.detach(), start)  # the scales learn
    assert any("distance-scales-g300-3000" in p.name and "-vk" in p.name for p in (tmp_path / "out").iterdir())
