"""Attention diagnostics (attention_plan.md Stage 1).

Pinned here: planted maps read their known values (span in µm, mass beyond a radius, CLS mass,
effective keys), with CLS never a key, the left-padded layout and graph-major heads; the numbers
from a real pass are the maps the step returns; the Performer's reference is the exact softmax it
estimates; and a diagnostic pass draws from no global generator and leaves the module as it was.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.evaluation.attention_diagnostics import (
    MIN_CELL_MASS,
    attention_diagnostics,
    attention_metric_name,
    head_statistics,
    radius_metric,
    summarize,
)

N_EMBED = 16
N_HEADS = 4


def _values(maps, padding, n_heads, pos=None, radii=(), reference=None):
    """head_statistics, pooled per (metric, layer, head) into one array."""
    found = head_statistics(maps, padding, n_heads=n_heads, pos_um=pos, radii_um=radii, reference=reference)
    return {key: np.concatenate(chunks) for key, chunks in found.items()}


def _line(xs):
    """``[1, S, 2]`` coordinates on the x axis."""
    return torch.tensor([[[x, 0.0] for x in xs]], dtype=torch.float64)


# --------------------------------------------------------------------------- planted maps


def test_a_planted_map_reads_its_span_radii_cls_mass_and_effective_keys():
    """Four cells on a line at 0, 100, 300 and 600 µm, no padding, CLS at position 4."""
    maps = torch.zeros(1, 2, 5, 5)
    # Head 0: every cell attends to one other cell, 300 or 500 µm away.
    for query, key in [(0, 2), (1, 3), (2, 0), (3, 1)]:
        maps[0, 0, query, key] = 1.0
    # Head 1: half on CLS, the other half spread evenly over the three other cells.
    for query in range(4):
        maps[0, 1, query, 4] = 0.5
        for key in range(4):
            if key != query:
                maps[0, 1, query, key] = 0.5 / 3
    maps[0, :, 4, 0] = 1.0  # the CLS row: never read
    pos = _line([0.0, 100.0, 300.0, 600.0])

    values = _values(maps, torch.zeros(1, 4, dtype=torch.bool), 2, pos=pos, radii=(400.0,))

    assert np.allclose(values[("span_um", 0, 0)], [300, 500, 300, 500])
    assert np.allclose(values[(radius_metric(400), 0, 0)], [0, 1, 0, 1])
    assert np.allclose(values[("effective_keys", 0, 0)], 1.0)
    assert np.allclose(values[("cls_mass", 0, 0)], 0.0)
    # CLS is the sink, not a key: span and effective keys read the cells only.
    distances = np.abs(np.subtract.outer([0, 100, 300, 600], [0, 100, 300, 600]))
    assert np.allclose(values[("span_um", 0, 1)], distances.sum(1) / 3)
    assert np.allclose(values[("effective_keys", 0, 1)], 3.0)
    assert np.allclose(values[("cls_mass", 0, 1)], 0.5)
    assert ("tv_softmax", 0, 0) not in values


def test_a_query_attending_to_a_cell_300_um_away_reads_300():
    """The plan's own check, through summarize: the logged span, quartiles and radii."""
    maps = torch.zeros(1, 1, 3, 3)
    maps[0, 0, 0, 1] = maps[0, 0, 1, 0] = 1.0
    padding = torch.zeros(1, 2, dtype=torch.bool)
    results = summarize(head_statistics(maps, padding, n_heads=1, pos_um=_line([0, 300]), radii_um=(250, 350)))

    for metric in ("span_um", "span_um_q25", "span_um_q50", "span_um_q75"):
        assert results[attention_metric_name(metric, 0, 0)] == pytest.approx(300.0)
    assert results[attention_metric_name("beyond_250um", 0, 0)] == pytest.approx(1.0)
    assert results[attention_metric_name("beyond_350um", 0, 0)] == pytest.approx(0.0)


def test_padding_is_on_the_left_and_heads_are_graph_major():
    """Two graphs of 2 and 4 cells, so graph 0's cells sit at positions 2 and 3."""
    n_heads, seq_len = 2, 4
    maps = torch.zeros(1, 2 * n_heads, seq_len + 1, seq_len + 1)
    padding = torch.tensor([[True, True, False, False], [False, False, False, False]])
    pos = torch.zeros(2, seq_len, 2, dtype=torch.float64)
    pos[0, 2:, 0] = torch.tensor([0.0, 1000.0])  # graph 0: two cells 1 mm apart
    pos[1, :, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])  # graph 1: 10 µm spacing
    # Row b * heads + h. Graph 0, head 0: the two cells attend to each other.
    maps[0, 0, 2, 3] = maps[0, 0, 3, 2] = 1.0
    # Graph 0, head 1: everything on CLS.
    maps[0, 1, 2:4, seq_len] = 1.0
    # Graph 1, head 0: each cell to a neighbour 10 µm away; head 1: 30 or 20 µm away.
    for query, key in [(0, 1), (1, 0), (2, 3), (3, 2)]:
        maps[0, 2, query, key] = 1.0
    for query, key in [(0, 3), (3, 0), (1, 3), (2, 0)]:
        maps[0, 3, query, key] = 1.0
    # The padded rows of graph 0 hold weight they must never be read for.
    maps[0, 0:2, 0:2, 0] = 1.0

    values = _values(maps, padding, n_heads, pos=pos)

    assert np.allclose(values[("span_um", 0, 0)], [1000, 1000, 10, 10, 10, 10])
    assert np.allclose(values[("span_um", 0, 1)], [30, 20, 20, 30])  # graph 0 is all sink
    assert np.allclose(values[("cls_mass", 0, 1)], [1, 1, 0, 0, 0, 0])


def test_an_all_sink_row_is_skipped_for_the_renormalised_statistics_only():
    maps = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    maps[0, 0, 0, 2] = 1.0  # cell 0: all on CLS
    maps[0, 0, 1, 0], maps[0, 0, 1, 2] = MIN_CELL_MASS / 10, 1.0 - MIN_CELL_MASS / 10  # cell 1: nearly
    padding = torch.zeros(1, 2, dtype=torch.bool)
    values = _values(maps, padding, 1, pos=_line([0, 50]))

    assert values[("span_um", 0, 0)].size == 0 and values[("effective_keys", 0, 0)].size == 0
    assert np.allclose(values[("cls_mass", 0, 0)], [1.0, 1.0 - MIN_CELL_MASS / 10])
    results = summarize(head_statistics(maps, padding, n_heads=1, pos_um=_line([0, 50])))
    assert attention_metric_name("span_um", 0, 0) not in results
    assert results[attention_metric_name("cls_mass", 0, 0)] == pytest.approx(1.0)


def test_the_tv_distance_is_half_the_l1_gap_of_each_row():
    maps = torch.zeros(1, 1, 3, 3)
    maps[0, 0, :2, 0] = 1.0
    reference = torch.zeros(1, 1, 3, 3)
    reference[0, 0, 0, 0] = 1.0
    reference[0, 0, 1, :] = torch.tensor([0.25, 0.25, 0.5])
    values = _values(maps, torch.zeros(1, 2, dtype=torch.bool), 1, reference=reference)

    assert np.allclose(values[("tv_softmax", 0, 0)], [0.0, 0.75])


def test_maps_that_do_not_fit_the_layout_are_rejected():
    with pytest.raises(ValueError, match="do not fit"):
        head_statistics(torch.zeros(1, 3, 5, 5), torch.zeros(2, 4, dtype=torch.bool), n_heads=2)


def test_the_summary_averages_heads_per_layer_and_everything_overall():
    per_cell = {
        ("cls_mass", 0, 0): [np.array([0.0, 1.0])],
        ("cls_mass", 0, 1): [np.array([1.0])],
        ("cls_mass", 1, 0): [np.array([0.0])],
        ("cls_mass", 1, 1): [np.array([0.0])],
    }
    results = summarize(per_cell)

    assert results[attention_metric_name("cls_mass", 0, 0)] == pytest.approx(0.5)
    assert results[attention_metric_name("cls_mass", 0)] == pytest.approx(0.75)
    assert results[attention_metric_name("cls_mass", 1)] == pytest.approx(0.0)
    assert results[attention_metric_name("cls_mass")] == pytest.approx(0.375)


def test_names_chart_a_layers_heads_together_and_carry_no_dot():
    from interscale.evaluation.online_probes import OnlineProbeCallback

    assert radius_metric(300) == "beyond_300um"
    assert radius_metric(37.5) == "beyond_37p5um"
    results = {
        attention_metric_name("span_um", 0, 0): 1.0,
        attention_metric_name("span_um", 0, 1): 2.0,
        attention_metric_name("span_um", 0): 1.5,
        attention_metric_name("span_um"): 1.5,
    }
    charts = OnlineProbeCallback.chart_series(None, None, results)

    assert charts["attn_span_um_l0"] == {"h0": 1.0, "h1": 2.0, "mean": 1.5}
    assert charts["attn_span_um"] == {"mean": 1.5}


# --------------------------------------------------------------------------- a real pass


def _cfg(kind="dense", num_features=64, **attention):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = N_EMBED
    params = cfg.model.global_component.parameters
    params.n_heads = N_HEADS
    params.attention.kind = kind
    params.attention.performer.num_features = num_features
    for key, value in attention.items():
        setattr(params.attention, key, value)
    return cfg


def _module(kind="dense", seed=0, max_seq_len=32, **kwargs):
    from interscale.module.base import GlobalModule

    cfg = _cfg(kind, **kwargs)
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    torch.manual_seed(seed)
    return GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=8,
        n_output=8,
        n_embed=N_EMBED,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )


def _graphs(sizes=(9, 6), seed=0, pos=True):
    rng = np.random.default_rng(seed)
    graphs = []
    for n in sizes:
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        d = Data(x=torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32), edge_index=torch.tensor([src, dst]))
        d.embeddings = torch.tensor(rng.normal(size=(n, N_EMBED)), dtype=torch.float32)
        d.mask = torch.tensor(rng.random(n) < 0.4)
        if pos:
            d.pos = torch.tensor(rng.uniform(0, 50, size=(n, 2)), dtype=torch.float32)
        graphs.append(d)
    return graphs


def _diagnose(module, graphs, **kwargs):
    kwargs.setdefault("spatial_unit_um", 2.0)
    return attention_diagnostics(
        module, graphs, prediction_task="regression", prediction_level="node", device=torch.device("cpu"), **kwargs
    )


def _clean_step(module, graph):
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([graph])
    batch.mask = torch.zeros_like(batch.mask)
    return module._common_step(batch, "regression", "node")


def test_the_diagnostics_read_the_maps_the_step_returns():
    """The recorded maps are the ones a step returns when asked for them, and each token's
    coordinates are its own cell's, in µm -- placed here by the step's own token-to-cell index."""
    graphs = _graphs(sizes=(9, 5))
    module = _module(max_seq_len=6)  # the 9-cell graph is subsampled, with Python's random
    module.eval()
    expected = {}
    random.seed(5)
    for graph in graphs:
        module.return_attention_maps = True
        out = _clean_step(module, graph)
        module.return_attention_maps = False
        padding = out.view.src_padding_mask[:, :-1]
        pos = torch.zeros(1, padding.shape[1], 2, dtype=torch.float64)
        pos[0, ~padding[0]] = graph.pos[out.view.padded_node_idx].double() * 2.0
        for key, values in head_statistics(out.attn, padding, n_heads=N_HEADS, pos_um=pos, radii_um=(20.0,)).items():
            expected.setdefault(key, []).extend(values)

    random.seed(5)
    state = random.getstate()
    results = _diagnose(module, graphs, radii_um=(20.0,))

    assert random.getstate() == state
    # Beside the per-head maps, the token cosine of every layer and its mean (`attention.center_qk`).
    cosine = {name for name in results if "token_cosine" in name}
    assert cosine == {attention_metric_name("token_cosine", layer) for layer in (0, 1)} | {
        attention_metric_name("token_cosine")
    }
    assert results.keys() - cosine == summarize(expected).keys()
    for name, value in summarize(expected).items():
        assert results[name] == pytest.approx(value, abs=1e-6), name


def test_the_performer_reference_is_the_exact_softmax_it_estimates():
    """At one seed a performer and a dense model share every weight, so the first layer's
    reference (exact softmax on the performer's queries and keys) is the dense model's map."""
    graph = _graphs(sizes=(9,))[0]
    dense, performer = _module("dense"), _module("performer")
    dense.eval(), performer.eval()
    dense.return_attention_maps = True
    dense_maps = _clean_step(dense, graph).attn

    with performer.recording_attention(reference=True) as record:
        _clean_step(performer, graph)

    (entry,) = record
    assert torch.allclose(entry["reference"][0], dense_maps[0], atol=1e-6)
    assert not torch.allclose(entry["maps"][0], dense_maps[0], atol=1e-3)  # the estimate is not exact


def test_the_tv_distance_falls_with_the_feature_count():
    """Roughly as 1 / sqrt(m): measured 0.19, 0.041 and 0.014 here."""
    graphs = _graphs()
    tv = [
        _diagnose(_module("performer", num_features=m), graphs)[attention_metric_name("tv_softmax")]
        for m in (8, 512, 8192)
    ]

    assert tv[0] > tv[1] > tv[2] > 0.0
    assert tv[2] < 0.02


@pytest.mark.parametrize("kind", ["dense", "linear", "exphormer"])
def test_only_an_estimate_of_softmax_gets_a_tv_distance(kind):
    extra = {"head_dim": 32} if kind == "linear" else {}
    results = _diagnose(_module(kind, **extra), _graphs())

    assert attention_metric_name("cls_mass") in results
    assert not any("tv_softmax" in name for name in results)
    assert all(np.isfinite(v) for v in results.values())


@pytest.mark.parametrize("kind", ["dense", "performer", "linear", "exphormer"])
def test_recording_leaves_the_module_as_it_was(kind):
    extra = {"head_dim": 32} if kind == "linear" else {}
    module = _module(kind, **extra)
    module.eval()
    graph = _graphs(sizes=(9,))[0]
    before = _clean_step(module, graph).y_pred

    with pytest.raises(RuntimeError, match="inside"):
        with module.recording_attention(reference=True):
            _clean_step(module, graph)
            raise RuntimeError("inside")

    assert module.return_attention_maps is False and module._attention_record is None
    for layer in module.transformer_encoder.layers:
        assert getattr(layer.self_attn, "record_reference", False) is False
        assert getattr(layer.self_attn, "reference_weights", None) is None
    assert torch.equal(_clean_step(module, graph).y_pred, before)
    assert _clean_step(module, graph).attn is None


def test_recording_dense_attention_changes_no_number():
    module = _module("dense")
    module.eval()
    graph = _graphs(sizes=(9,))[0]
    before = _clean_step(module, graph).y_pred
    with module.recording_attention():
        during = _clean_step(module, graph).y_pred

    assert torch.equal(during, before)


def test_a_diagnostic_pass_leaves_every_global_generator_and_the_training_flag():
    graphs = _graphs(sizes=(40, 6))  # 40 > max_seq_len: pad_batch subsamples with Python's random
    module = _module("exphormer", max_seq_len=16)
    module.train()
    torch_state, py_state, np_state = torch.random.get_rng_state(), random.getstate(), np.random.get_state()

    _diagnose(module, graphs)

    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert module.training


def test_graphs_without_coordinates_skip_span_and_radii_with_a_warning():
    with pytest.warns(RuntimeWarning, match="no `pos`"):
        results = _diagnose(_module(), _graphs(pos=False), radii_um=(10.0,))

    assert attention_metric_name("cls_mass") in results
    assert attention_metric_name("effective_keys") in results
    assert not any("span_um" in name or "beyond" in name for name in results)


def test_only_the_first_graphs_are_read():
    graphs = _graphs(sizes=(9, 6, 7))
    assert _diagnose(_module(), graphs, max_graphs=1) == _diagnose(_module(), graphs[:1], max_graphs=0)


def test_a_module_without_a_transformer_gets_a_warning_and_nothing():
    with pytest.warns(RuntimeWarning, match="no transformer"):
        assert _diagnose(torch.nn.Linear(2, 2), _graphs()) == {}


# --------------------------------------------------------------------------- the probe callback


def test_the_probe_callback_logs_the_diagnostics_as_its_only_target():
    from interscale.config import _validate_probe, get_cfg_defaults
    from interscale.evaluation.online_probes import OnlineProbeCallback

    cfg = get_cfg_defaults()
    cfg.dataset.prediction_task = "regression"
    cfg.dataset.prediction_level = "node"
    cfg.dataset.spatial_unit_um = 2.0
    cfg.probe.use = True
    cfg.probe.attention_diagnostics = True
    cfg.probe.attention_graphs = 2
    cfg.probe.attention_radii_um = [20.0]
    _validate_probe(cfg)

    class _Datamodule:
        train_data, val_data = _graphs(seed=1), _graphs(sizes=(9, 6, 7))

    module = _module()
    results = OnlineProbeCallback(cfg, [f"g{i}" for i in range(8)]).run(
        module, _Datamodule(), device=torch.device("cpu")
    )

    assert results == _diagnose(module, _Datamodule.val_data[:2], radii_um=(20.0,), masked=True)
    assert attention_metric_name("beyond_20um", 0, 0) in results

    cfg.probe.attention_masked = False
    clean = OnlineProbeCallback(cfg, [f"g{i}" for i in range(8)]).run(module, _Datamodule(), device=torch.device("cpu"))
    assert clean == _diagnose(module, _Datamodule.val_data[:2], radii_um=(20.0,))


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("attention_graphs", -1, "attention_graphs"),
        ("attention_radii_um", [100.0, 0.0], "must all be > 0"),
        ("attention_radii_um", [100.0, 100], "twice"),
    ],
)
def test_the_validator_rejects_bad_diagnostic_settings(key, value, match):
    from interscale.config import _validate_probe, get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.probe.use = True
    cfg.probe.attention_diagnostics = True
    setattr(cfg.probe, key, value)
    with pytest.raises(ValueError, match=match):
        _validate_probe(cfg)
