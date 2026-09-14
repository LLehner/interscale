"""Online downstream probes: alignment, the leakage guard, and the local-vs-global pairing.

Three classes of silent failure are what these tests exist for, none of which raises:

* **Misalignment.** Row i of an embedding must be the same cell as row i of its target. Get it
  wrong and the probe scores at chance, which reads like a real negative result.
* **Leakage.** An unmasked cell had its own expression in the encoder input, so a readout can
  recover a target by inverting the embedding rather than by using anything the model learned.
  That inflates every probe -- including a structure-free control gene, which is precisely the
  number one would look at to decide the probe is trustworthy.
* **Unpaired columns.** The local and global halves of a pair must be scored on the same cells
  with the same readout, or the gap between them (the only quantity of interest) is noise.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.config import get_cfg_defaults
from interscale.evaluation.online_probes import (
    OnlineProbeCallback,
    ProbeBatchFeatures,
    collect_features,
    probe_metric_name,
    score_classification,
    score_regression,
)
from interscale.module.global_modules import TransformerNodeEncoderHook

N_GENES = 8
N_EMBED = 8
N_CELLTYPES = 3
GENES = [f"g{i}" for i in range(N_GENES)]


def build_module(max_seq_len=32, mask_percentage=0.3, mask_strategy="node"):
    return TransformerNodeEncoderHook(
        max_seq_len=max_seq_len,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=16,
        long_range_attention=False,
        local_mask_hops=1,
        n_input=N_GENES,
        n_output=N_GENES,
        n_embed=N_EMBED,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=mask_percentage,
        mask_strategy=mask_strategy,
        type_gex_embedding=None,
    )


def make_data_list(sizes=(6, 9), seed=0, mask_strategy="node"):
    """Graphs carrying precomputed embeddings plus a one-hot `celltype`, as the probe expects."""
    rng = np.random.default_rng(seed)
    datas = []
    for n in sizes:
        x = torch.tensor(rng.normal(size=(n, N_GENES)), dtype=torch.float32)
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        d = Data(x=x, edge_index=torch.tensor([src, dst], dtype=torch.long))
        d.embeddings = torch.tensor(rng.normal(size=(n, N_EMBED)), dtype=torch.float32)

        codes = rng.integers(0, N_CELLTYPES, size=n)
        one_hot = np.zeros((n, N_CELLTYPES), dtype=np.float32)
        one_hot[np.arange(n), codes] = 1.0
        d.celltype = torch.tensor(one_hot)

        if mask_strategy == "gene":
            gene_mask = torch.tensor(rng.random((n, N_GENES)) < 0.4)
            gene_mask[0, 0] = True
            d.gene_mask = gene_mask
            d.mask = gene_mask.any(dim=1)
        else:
            d.mask = torch.tensor(rng.random(n) < 0.4)
            d.mask[0] = True  # never an all-unmasked graph
        datas.append(d)
    return datas


def probe_cfg(**overrides):
    cfg = get_cfg_defaults()
    cfg.dataset.prediction_task = "regression"
    cfg.dataset.prediction_level = "node"
    cfg.dataset.batch_size = 2
    cfg.dataset.celltype_key = "cell_type"
    cfg.probe.use = True
    cfg.probe.classification_targets = ["celltype"]
    cfg.probe.regression_genes = ["g0", "g1"]
    cfg.probe.max_cells = 0
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        setattr(getattr(cfg, section), leaf, value)
    return cfg


class _Datamodule:
    """The two attributes the probe reads off a GraphAnnDataModule."""

    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.setup_called = True


def collect(data_list, module=None, mask_strategy="node"):
    return collect_features(
        module or build_module(mask_strategy=mask_strategy),
        data_list,
        prediction_task="regression",
        prediction_level="node",
        embeddings=("local", "global"),
        gene_index={"g0": 0, "g3": 3},
        categorical_targets=("celltype",),
        mask_strategy=mask_strategy,
        batch_size=2,
        device=torch.device("cpu"),
    )


# --------------------------------------------------------------------------- naming


def test_metric_name_round_trips_through_the_chart_grouping():
    """`chart_series` splits on the LAST underscore, so a gene name full of them must survive."""
    cfg = probe_cfg()
    cfg.probe.regression_genes = []
    callback = OnlineProbeCallback(cfg, GENES)

    name = probe_metric_name("gene_int_long", "mse", "global")
    series = callback.chart_series(None, {name: 0.5})

    assert series == {"gene_int_long_mse": {"global": 0.5}}


def test_chart_series_puts_both_embeddings_on_one_chart():
    """The whole point of the charts: one plot per target, one line per embedding."""
    callback = OnlineProbeCallback(probe_cfg(), GENES)

    series = callback.chart_series(
        None,
        {
            probe_metric_name("celltype", "precision", "local"): 0.4,
            probe_metric_name("celltype", "precision", "global"): 0.6,
            probe_metric_name("celltype", "recall", "local"): 0.3,
        },
    )

    assert series["celltype_precision"] == {"local": 0.4, "global": 0.6}
    assert series["celltype_recall"] == {"local": 0.3}


# --------------------------------------------------------------------------- alignment


def test_select_indexes_every_block_with_the_same_rows():
    features = ProbeBatchFeatures(
        embeddings={"local": np.arange(10).reshape(5, 2)},
        categorical={"celltype": np.arange(5)},
        continuous={"g0": np.arange(5) * 10.0},
        cell_masked=np.array([True, False, True, False, True]),
        gene_masked={"g0": np.array([True, False, True, False, True])},
    )

    picked = features.select(np.array([0, 3]))

    assert picked.n_cells() == 2
    np.testing.assert_array_equal(picked.categorical["celltype"], [0, 3])
    np.testing.assert_array_equal(picked.continuous["g0"], [0.0, 30.0])
    np.testing.assert_array_equal(picked.embeddings["local"], [[0, 1], [6, 7]])
    np.testing.assert_array_equal(picked.cell_masked, [True, False])


def test_subsample_is_a_no_op_below_the_cap():
    features = ProbeBatchFeatures(embeddings={"local": np.zeros((4, 2))})

    assert features.subsample(10, np.random.default_rng(0)) is features
    assert features.subsample(0, np.random.default_rng(0)) is features


def test_collected_targets_line_up_with_the_batch():
    """Targets are gathered by `padded_node_idx`, the same index that puts tokens in cell order."""
    data_list = make_data_list()
    module = build_module()
    features = collect(data_list, module)

    batch = Batch.from_data_list(data_list)
    out = module._common_step(batch, "regression", "node")
    idx = out.view.padded_node_idx

    np.testing.assert_allclose(features.continuous["g0"], batch.x[idx, 0].numpy(), rtol=1e-5)
    np.testing.assert_allclose(features.continuous["g3"], batch.x[idx, 3].numpy(), rtol=1e-5)
    np.testing.assert_array_equal(features.categorical["celltype"], batch.celltype[idx].argmax(dim=1).numpy())


def test_regression_targets_are_the_uncorrupted_expression():
    """`apply_mask` corrupts a clone, so a masked cell's target is still its real expression.

    Without this the probe would be asked to predict MASK_VALUE for exactly the cells it is
    restricted to -- a constant target, i.e. a probe that measures nothing at all.
    """
    from interscale.tl.masking import MASK_VALUE

    features = collect(make_data_list())
    masked_targets = features.continuous["g0"][features.cell_masked]

    assert len(masked_targets) > 0
    assert not np.any(masked_targets == MASK_VALUE)


def test_cell_masked_marks_the_reconstruction_targets():
    data_list = make_data_list()
    module = build_module()
    features = collect(data_list, module)

    batch = Batch.from_data_list(data_list)
    idx = module._common_step(batch, "regression", "node").view.padded_node_idx

    np.testing.assert_array_equal(features.cell_masked, batch.mask[idx].numpy())
    # Under node masking a masked cell has every gene blanked, so the per-gene view agrees.
    np.testing.assert_array_equal(features.gene_masked["g0"], features.cell_masked)


def test_gene_masking_gives_each_gene_its_own_held_out_rows():
    """Under `mask_strategy="gene"` the entries are drawn per (cell, gene), so g0 and g3 differ."""
    data_list = make_data_list(mask_strategy="gene", seed=3)
    features = collect(data_list, build_module(mask_strategy="gene"), mask_strategy="gene")

    assert not np.array_equal(features.gene_masked["g0"], features.gene_masked["g3"])
    # and neither is simply the cell-level OR
    assert not np.array_equal(features.gene_masked["g0"], features.cell_masked)


# --------------------------------------------------------------------------- the leakage guard


def test_masked_cells_only_restricts_the_probe_to_held_out_cells():
    """This is the guard that keeps a noise control honest -- see probe.masked_cells_only.

    An unmasked cell's own expression is in the encoder input, so including it lets the readout
    invert the embedding instead of using anything the model learned.
    """
    data_list = make_data_list(sizes=(20, 20))
    features = collect(data_list)

    restricted = OnlineProbeCallback._masked(features)

    assert restricted.n_cells() == int(features.cell_masked.sum())
    assert restricted.n_cells() < features.n_cells(), "fixture must contain unmasked cells"
    assert restricted.cell_masked.all()


def test_masked_helper_keeps_every_row_when_nothing_was_masked():
    """Graph-level runs leave val uncorrupted; dropping every row there would log nothing."""
    features = ProbeBatchFeatures(
        embeddings={"local": np.zeros((4, 2))},
        cell_masked=np.zeros(4, dtype=bool),
    )

    assert OnlineProbeCallback._masked(features) is features


# --------------------------------------------------------------------------- the readouts


def test_classification_readout_recovers_a_separable_label():
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 3, size=200)
    y_val = rng.integers(0, 3, size=100)
    # Feature 0 carries the class; the rest is noise.
    x_train = np.column_stack([y_train * 5.0, rng.normal(size=(200, 3))])
    x_val = np.column_stack([y_val * 5.0, rng.normal(size=(100, 3))])

    scores = score_classification(x_train, y_train, x_val, y_val, seed=0, max_iter=500)

    assert scores["precision"] > 0.9
    assert scores["recall"] > 0.9


def test_classification_readout_skips_a_single_class_split():
    x = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.warns(RuntimeWarning, match="single class"):
        assert score_classification(x, np.zeros(20), x, np.zeros(20), seed=0, max_iter=100) == {}


def test_regression_readout_separates_signal_from_noise():
    """A target the features carry scores near R2 1; an unrelated one near 0."""
    rng = np.random.default_rng(0)
    x_train, x_val = rng.normal(size=(300, 4)), rng.normal(size=(150, 4))

    signal = score_regression(x_train, x_train[:, 0] * 3.0, x_val, x_val[:, 0] * 3.0, alpha=1.0)
    noise = score_regression(x_train, rng.normal(size=300), x_val, rng.normal(size=150), alpha=1.0)

    assert signal["r2"] > 0.95
    assert noise["r2"] < 0.2


def test_regression_readout_skips_a_constant_target():
    x = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.warns(RuntimeWarning, match="constant"):
        assert score_regression(x, np.ones(20), x, np.ones(20), alpha=1.0) == {}


# --------------------------------------------------------------------------- end to end


def test_run_emits_a_paired_scalar_for_every_target_and_embedding():
    data_list = make_data_list(sizes=(40, 40))
    cfg = probe_cfg()
    cfg.probe.regression_genes = ["g0"]
    callback = OnlineProbeCallback(cfg, GENES)

    results = callback.run(
        build_module(),
        _Datamodule(data_list, make_data_list(sizes=(40,), seed=7)),
        device=torch.device("cpu"),
    )

    # A GlobalModule has no local component, so only the global column exists -- the probe
    # must degrade to that rather than erroring.
    assert probe_metric_name("celltype", "precision", "global") in results
    assert probe_metric_name("celltype", "recall", "global") in results
    assert probe_metric_name("gene_g0", "mse", "global") in results
    assert not any(name.endswith("_local") for name in results)
    assert all(np.isfinite(v) for v in results.values())


def test_unknown_gene_name_raises_rather_than_probing_the_wrong_column():
    cfg = probe_cfg()
    cfg.probe.regression_genes = ["not_a_gene"]

    with pytest.raises(KeyError, match="not in adata.var_names"):
        OnlineProbeCallback(cfg, GENES)


def test_var_names_mismatch_is_caught_before_the_forward_pass():
    cfg = probe_cfg()
    cfg.probe.regression_genes = ["g0"]
    callback = OnlineProbeCallback(cfg, [*GENES, "extra_gene"])

    with pytest.raises(ValueError, match="wrong column"):
        callback.run(build_module(), _Datamodule([], []), device=torch.device("cpu"))


# --------------------------------------------------------------------------- config validation


def test_probe_without_targets_is_rejected():
    from interscale.config import _validate_probe

    cfg = get_cfg_defaults()
    cfg.probe.use = True
    with pytest.raises(ValueError, match="names a target"):
        _validate_probe(cfg)


def test_celltype_target_without_celltype_key_is_rejected():
    """The annotation only reaches the graphs when dataset.celltype_key is set."""
    from interscale.config import _validate_probe

    cfg = get_cfg_defaults()
    cfg.probe.use = True
    cfg.probe.classification_targets = ["celltype"]
    with pytest.raises(ValueError, match="dataset.celltype_key is unset"):
        _validate_probe(cfg)


def test_unknown_embedding_is_rejected():
    from interscale.config import _validate_probe

    cfg = get_cfg_defaults()
    cfg.probe.use = True
    cfg.probe.classification_targets = ["celltype"]
    cfg.dataset.celltype_key = "cell_type"
    cfg.probe.embeddings = ["local", "cls"]
    with pytest.raises(ValueError, match="no such representation"):
        _validate_probe(cfg)


def test_disabled_probe_validates_anything():
    """Inert means inert: a nonsense probe block must not stop a run that has probes off."""
    from interscale.config import _validate_probe

    cfg = get_cfg_defaults()
    cfg.probe.use = False
    cfg.probe.embeddings = ["nonsense"]
    _validate_probe(cfg)
