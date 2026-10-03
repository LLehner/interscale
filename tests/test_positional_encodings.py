"""Node positional encodings: naive and 2D sinusoidal (PE_plan.md, Stages 0-2).

Pinned here: off is the model from before PEs; on changes neither the other initial weights nor
the first forward; each cell's encoding travels with its token through pad_batch's subsampling;
the coordinate handling (units, centring, rotation) does what the config says; and the config,
checkpoint name and sweep arms carry the choice.
"""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from interscale.config import get_cfg_defaults, load_config
from interscale.config.global_component_config import get_global_component_cfg
from interscale.config.local_component_config import get_local_component_cfg
from interscale.module.global_modules.positional_encodings import (
    PE_REGISTRY,
    SinusoidalPE,
    build_node_positional_encoding,
)

N_EMBED = 8


def _cfg(node=(), *, local=True, **pe):
    cfg = get_cfg_defaults()
    if local:
        cfg.model.local_component.name = "GCN"
        cfg = get_local_component_cfg(cfg, "GCN")
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.dataset.spatial_key = "spatial"
    cfg.model.global_component.parameters.pe.node = list(node)
    for key, value in pe.items():
        setattr(cfg.model.global_component.parameters.pe, key, value)
    return cfg


def _randomised(encoding):
    """Zero-initialised output layers make every encoder output 0; give them weights to measure."""
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in encoding.parameters():
            p.copy_(torch.randn(p.shape, generator=gen))
    return encoding


def _batch(*graphs, edges=True):
    """A batch-like object of graphs given as ``[n_i, 2]`` coordinate tensors."""
    pos = torch.cat(graphs)
    batch = torch.cat([torch.full((len(g),), i, dtype=torch.long) for i, g in enumerate(graphs)])
    n = len(pos)
    ns = SimpleNamespace(pos=pos, batch=batch, num_nodes=n, obs_names=torch.arange(n))
    ns.mask = torch.zeros(n, dtype=torch.bool)
    if edges:
        offsets = torch.cumsum(torch.tensor([0] + [len(g) for g in graphs[:-1]]), 0)
        src, dst = [], []
        for off, g in zip(offsets.tolist(), graphs, strict=True):
            src += list(range(off, off + len(g) - 1)) + list(range(off + 1, off + len(g)))
            dst += list(range(off + 1, off + len(g))) + list(range(off, off + len(g) - 1))
        ns.edge_index = torch.tensor([src, dst], dtype=torch.long)
    return ns


def _combined(cfg):
    from interscale.module.combined_module import CombinedModule

    return CombinedModule(
        cfg=cfg,
        n_input=6,
        n_output=6,
        n_embed=N_EMBED,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )


def _encode(encoding, *graphs):
    return encoding(_batch(*graphs, edges=False), dtype=torch.float32, device=torch.device("cpu"))


# --------------------------------------------------------------------------- off is the old model


def test_the_default_config_builds_nothing():
    module = _combined(_cfg())

    assert module.global_module.positional_encoding is None
    assert not [k for k in module.state_dict() if "positional" in k]


def test_enabling_a_pe_changes_no_other_initial_weight_and_not_the_first_forward():
    """Built in a forked RNG with a zero output layer: a PE run and a PE-free run at one seed start
    from the same weights and the same first forward, so what differs later is what the PE added."""
    torch.manual_seed(0)
    off = _combined(_cfg())
    torch.manual_seed(0)
    on = _combined(_cfg(["naive", "sinusoidal"]))

    on_state = on.state_dict()
    for key, value in off.state_dict().items():
        assert torch.equal(value, on_state[key]), key

    g = torch.rand(10, 2) * 200
    emb = torch.randn(10, N_EMBED)
    out_off = off.global_module.common_step_local_to_global(_batch(g), emb, eval_step=True)[0]
    out_on = on.global_module.common_step_local_to_global(_batch(g), emb, eval_step=True)[0]
    assert torch.equal(out_off, out_on)


# --------------------------------------------------------------------------- through pad_batch


def test_each_kept_token_carries_its_own_cells_encoding_when_pad_batch_subsamples():
    """pad_batch draws a random subset when a graph exceeds max_seq_len. The encoding is added
    before that, so whichever cells are kept, each token holds its own cell's encoding."""
    cfg = _cfg(["sinusoidal"])
    cfg.model.global_component.parameters.max_seq_len = 4
    module = _combined(cfg).global_module.eval()
    _randomised(module.positional_encoding)

    g = torch.rand(10, 2) * 300
    batch = _batch(g)
    expected = module.positional_encoding(batch, dtype=torch.float32, device=torch.device("cpu"))

    padded, _, index_nodes, _ = module.common_step_local_to_global(batch, torch.zeros(10, N_EMBED), eval_step=True)

    # LayerNorm maps the zero embedding to its bias, so token - bias is the encoding alone.
    tokens = padded[:, 0] - module.norm_input.bias.detach()
    assert len(index_nodes[0]) == 4
    assert torch.allclose(tokens, expected[index_nodes[0]], atol=1e-5)


# --------------------------------------------------------------------------- coordinates


def test_centring_makes_each_graph_translation_invariant():
    encoding = _randomised(build_node_positional_encoding(_cfg(["naive", "sinusoidal"]), N_EMBED))
    a, b = torch.rand(6, 2) * 100, torch.rand(5, 2) * 100

    before = _encode(encoding, a, b)
    moved = _encode(encoding, a + torch.tensor([500.0, -300.0]), b)

    assert torch.allclose(before, moved, atol=1e-4)


def test_without_centring_absolute_position_is_encoded():
    """The control for the invariance test above: it must be the centring that makes it pass."""
    encoding = _randomised(build_node_positional_encoding(_cfg(["naive"], center_coords=False), N_EMBED))
    a = torch.rand(6, 2) * 100

    assert not torch.allclose(_encode(encoding, a), _encode(encoding, a + 500.0), atol=1e-3)


def test_spatial_unit_converts_coordinates_to_micrometres():
    cfg_um = _cfg(["naive", "sinusoidal"])
    cfg_px = _cfg(["naive", "sinusoidal"])
    cfg_px.dataset.spatial_unit_um = 2.0
    in_um = _randomised(build_node_positional_encoding(cfg_um, N_EMBED))
    in_px = _randomised(build_node_positional_encoding(cfg_px, N_EMBED))
    pixels = torch.rand(7, 2) * 100

    assert torch.allclose(_encode(in_px, pixels), _encode(in_um, pixels * 2.0), atol=1e-4)


def test_rotation_is_training_only_and_uses_its_own_generator():
    encoding = _randomised(build_node_positional_encoding(_cfg(["naive"], rotate_train=True), N_EMBED))
    plain = _randomised(build_node_positional_encoding(_cfg(["naive"]), N_EMBED))
    g = torch.rand(8, 2) * 100
    batch = _batch(g, edges=False)
    cpu = torch.device("cpu")

    encoding.train()
    state = torch.get_rng_state()
    first = encoding.coordinates(batch, torch.float32, cpu)
    second = encoding.coordinates(batch, torch.float32, cpu)
    assert torch.equal(state, torch.get_rng_state()), "rotation drew from the global torch RNG"
    assert not torch.allclose(first, second), "each training call draws a new angle"
    assert torch.allclose(first.norm(dim=1), second.norm(dim=1), atol=1e-4), "a rotation keeps distances"

    encoding.eval()
    assert torch.allclose(_encode(encoding, g), _encode(plain.eval(), g), atol=1e-5)


def test_graphs_without_coordinates_fail_with_the_fix():
    encoding = build_node_positional_encoding(_cfg(["naive"]), N_EMBED)
    batch = _batch(torch.rand(3, 2), edges=False)
    del batch.pos

    with pytest.raises(ValueError, match="dataset.spatial_key"):
        encoding(batch, dtype=torch.float32, device=torch.device("cpu"))


# --------------------------------------------------------------------------- sinusoidal


def test_sinusoidal_features_match_the_closed_form():
    pe = SinusoidalPE(N_EMBED, dim=8, min_wavelength=10.0, max_wavelength=1000.0)
    x, y = 37.0, -12.5
    wavelengths = [10.0, 1000.0]  # dim/4 = 2 frequencies, geometric between the ends

    expected = []
    for coord in (x, y):
        expected += [math.sin(2 * math.pi * coord / w) for w in wavelengths]
        expected += [math.cos(2 * math.pi * coord / w) for w in wavelengths]

    assert torch.allclose(pe.features(torch.tensor([[x, y]])), torch.tensor([expected]), atol=1e-5)


def test_sinusoidal_similarity_falls_with_distance_up_to_the_short_scales():
    pe = SinusoidalPE(N_EMBED, dim=32, min_wavelength=10.0, max_wavelength=1000.0)
    origin = pe.features(torch.zeros(1, 2))

    def similarity(d):
        f = pe.features(torch.tensor([[d, 0.0]]))
        return float((origin * f).sum() / (origin.norm() * f.norm()))

    values = [similarity(d) for d in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)]
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------------------------- config, name, sweep


def _write(tmp_path, pe_lines, spatial=True):
    header = "model:\n  global_component:\n    name: self-attn-transformer\n    parameters:\n      pe:\n"
    text = header + "".join(f"        {line}\n" for line in pe_lines)
    if spatial:
        text += "dataset:\n  spatial_key: spatial\n"
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return path


def test_a_valid_pe_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, ["node: [naive, sinusoidal]"]))

    assert list(cfg.model.global_component.parameters.pe.node) == ["naive", "sinusoidal"]


@pytest.mark.parametrize(
    ("lines", "spatial", "match"),
    [
        (["node: [fourier]"], True, "unknown positional encodings"),
        (["node: [naive, naive]"], True, "more than once"),
        (["node: [naive]"], False, "dataset.spatial_key"),
        (["node: [sinusoidal]", "sinusoidal:", "  dim: 30"], True, "multiple of 4"),
        (["node: [sinusoidal]", "sinusoidal:", "  max_wavelength: 5.0"], True, "max_wavelength"),
        (["node: [naive]", "naive:", "  length_scale: 0.0"], True, "length_scale"),
        (["node: [lap]", "lap:", "  k: 0"], True, "lap.k"),
    ],
)
def test_invalid_pe_configs_fail_at_load(tmp_path, lines, spatial, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, lines, spatial=spatial))


def test_the_registry_holds_the_node_encodings_built_so_far():
    """What each encoding reads, and which precompute: a new entry has to be added here on purpose."""
    assert {name: (spec.kind, spec.requires, spec.precompute is not None) for name, spec in PE_REGISTRY.items()} == {
        "naive": ("node", "pos", False),
        "sinusoidal": ("node", "pos", False),
        "lap": ("node", "lap_pe", True),
    }


def test_the_checkpoint_name_carries_the_encodings_and_is_unchanged_without():
    from interscale.tl.utils import get_model_filename_prefix

    off = get_model_filename_prefix(_cfg(), True, True)
    on = get_model_filename_prefix(_cfg(["naive", "sinusoidal"]), True, True)

    assert "pe-" not in off
    assert on == off + "pe-naive+sinusoidal_"
    assert get_model_filename_prefix(_cfg(["naive"], local=False), False, True).endswith("pe-naive_")


def test_a_sweep_arm_switches_the_encoding():
    """List-valued keys go through arms, so the list never round-trips through wandb."""
    from interscale.config.sweep import apply_sweep_config

    key = "model.global_component.parameters.pe.node"
    arms = {"none": {key: []}, "sin": {key: ["sinusoidal"]}}
    cfg = _cfg()
    cfg.freeze()

    cfg, _ = apply_sweep_config(cfg, "hyperparmeter", {"arm": "sin"}, sweep_params=["arm"], arms=arms)

    assert list(cfg.model.global_component.parameters.pe.node) == ["sinusoidal"]


# --------------------------------------------------------------------------- training


def test_the_encodings_train_end_to_end():
    """Through the training plan: the loss stays finite and the zero-initialised output layers
    move, i.e. the gradient reaches the encodings."""
    import lightning.pytorch as pl
    from torch_geometric.data import Data

    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.module.global_modules import TransformerNodeEncoderHook
    from interscale.train._trainingplans import TrainingPlan

    def graph(n, seed):
        g = torch.Generator().manual_seed(seed)
        src = list(range(n - 1)) + list(range(1, n))
        d = Data(
            x=torch.randn(n, 6, generator=g),
            edge_index=torch.tensor([src, list(range(1, n)) + list(range(n - 1))], dtype=torch.long),
            pos=torch.rand(n, 2, generator=g) * 200,
        )
        d.embeddings = torch.randn(n, 4, generator=g)
        return d

    dm = GraphAnnDataModule(
        datas=[[graph(7, 0), graph(9, 1)], [graph(8, 2)], [graph(8, 3)]],
        batch_size=2,
        num_workers=0,
        mask_percentage=0.3,
        mask_strategy="node",
        learning_type="node",
    )
    encoding = build_node_positional_encoding(_cfg(["naive", "sinusoidal"], local=False), 4)
    module = TransformerNodeEncoderHook(
        max_seq_len=16,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=8,
        long_range_attention=False,
        local_mask_hops=0,
        positional_encoding=encoding,
        n_input=6,
        n_output=6,
        n_embed=4,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[8],
        mask_percentage=0.3,
        mask_strategy="node",
        type_gex_embedding=None,
    )
    outputs = [m for m in encoding.modules() if isinstance(m, nn.Linear) and not m.weight.any()]
    assert len(outputs) == 2, "both output layers start at zero"

    plan = TrainingPlan(
        module,
        "regression",
        "node",
        "MSELoss",
        "cell",
        batch_size=2,
        lr_scheduler="CosineWarmupScheduler",
        lr_warmup=1,
        lr_max_epochs=2,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    dm.setup(stage="fit")
    trainer.fit(plan, datamodule=dm)

    assert torch.isfinite(trainer.logged_metrics["train_loss"])
    assert all(layer.weight.any() for layer in outputs), "the gradient never reached an encoding"


# --------------------------------------------------------------------------- Laplacian eigenvectors


def _geometric_graph(n, radius, seed):
    """Undirected radius graph on random points in the unit square, as a PyG edge index."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 2, generator=g)
    close = (torch.cdist(pts, pts) < radius) & ~torch.eye(n, dtype=torch.bool)
    return close.nonzero().t().contiguous()


def _components(edge_index, n):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    src, dst = edge_index.numpy()
    return connected_components(coo_matrix(([1] * len(src), (src, dst)), shape=(n, n)), directed=False)


def _connected_geometric_graph(n, radius):
    for seed in range(50):
        edge_index = _geometric_graph(n, radius, seed)
        if _components(edge_index, n)[0] == 1:
            return edge_index
    raise AssertionError("no connected graph found")


def _same_up_to_sign(a, b, atol=1e-4):
    return all(
        torch.allclose(a[:, j], b[:, j], atol=atol) or torch.allclose(a[:, j], -b[:, j], atol=atol)
        for j in range(a.shape[1])
    )


@pytest.mark.parametrize("n", [60, 150])
def test_lap_pe_matches_pyg_on_a_connected_graph(n):
    """On the connected graph PyG's transform assumes, the two agree up to each column's sign. n=150
    takes PyG's sparse path; distinct eigenvalues make the comparison well defined."""
    from torch_geometric.data import Data
    from torch_geometric.transforms import AddLaplacianEigenvectorPE

    from interscale.tl import laplacian_pe

    k = 6
    edge_index = _connected_geometric_graph(n, radius=0.25)
    ours = laplacian_pe(edge_index, n, k)
    theirs = AddLaplacianEigenvectorPE(k, attr_name="pe", is_undirected=True)(
        Data(edge_index=edge_index, num_nodes=n)
    ).pe

    assert ours.shape == (n, k)
    assert _same_up_to_sign(ours, theirs.float())


def test_lap_pe_sparse_and_dense_solvers_agree():
    from interscale.tl import laplacian_pe

    n, k = 300, 6
    edge_index = _connected_geometric_graph(n, radius=0.15)

    dense = laplacian_pe(edge_index, n, k, dense_max_nodes=10_000)
    sparse = laplacian_pe(edge_index, n, k, dense_max_nodes=10)

    assert torch.allclose(dense, sparse, atol=1e-4), "canonical signs make the two solvers agree exactly"


def test_lap_pe_drops_one_trivial_eigenvector_per_component():
    """Two components plus an isolated cell: PyG would drop only the first zero eigenvector and keep
    a component indicator. Here no returned column lies in the zero eigenspace."""
    from interscale.tl import laplacian_pe

    a = _connected_geometric_graph(30, radius=0.35)
    b = _connected_geometric_graph(25, radius=0.35) + 30
    edge_index = torch.cat([a, b], dim=1)  # node 55 is isolated
    n, k = 56, 5
    assert _components(edge_index, n)[0] == 3

    pe = laplacian_pe(edge_index, n, k).double()
    adj = torch.zeros(n, n, dtype=torch.float64)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(1)
    inv = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    lap = torch.eye(n, dtype=torch.float64) - inv[:, None] * adj * inv[None, :]

    rayleigh = (pe * (lap @ pe)).sum(0) / (pe * pe).sum(0)
    assert bool((rayleigh > 1e-6).all()), f"a trivial eigenvector was kept: {rayleigh.tolist()}"
    assert torch.allclose(pe.norm(dim=0), torch.ones(k, dtype=torch.float64), atol=1e-4)
    assert torch.allclose(pe.t() @ pe, torch.eye(k, dtype=torch.float64), atol=1e-4), "columns orthonormal"


def test_lap_pe_zero_pads_a_graph_with_fewer_eigenvectors_than_k():
    from interscale.tl import laplacian_pe

    path = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    pe = laplacian_pe(path, 3, 8)

    assert pe.shape == (3, 8)
    assert pe[:, :2].abs().sum() > 0, "a 3-node path has two non-trivial eigenvectors"
    assert not pe[:, 2:].any()


def test_lap_pe_signs_are_canonical_so_rebuilt_graphs_match():
    """get_model_output rebuilds the graphs; the encoding must not depend on the solver's sign."""
    from interscale.tl import laplacian_pe

    edge_index = _connected_geometric_graph(80, radius=0.25)
    first, second = laplacian_pe(edge_index, 80, 6), laplacian_pe(edge_index, 80, 6)

    assert torch.equal(first, second)
    peak = first.abs().argmax(dim=0)
    assert bool((first[peak, torch.arange(6)] > 0).all()), "largest-magnitude entry is positive"


def _lap_batch(k=4, sizes=(6, 5)):
    graphs = [torch.rand(n, 2) for n in sizes]
    batch = _batch(*graphs, edges=False)
    batch.lap_pe = torch.randn(sum(sizes), k)
    return batch


def test_lap_sign_flips_are_per_graph_training_only_and_off_the_global_rng():
    encoding = _randomised(build_node_positional_encoding(_cfg(["lap"], lap=_lap_cfg(k=4)), N_EMBED))
    lap = encoding.encoders["lap"]
    batch = _lap_batch()
    cpu = torch.device("cpu")

    state = torch.get_rng_state()
    flipped = lap.augment(batch.lap_pe, batch.batch, encoding.generator)
    assert torch.equal(state, torch.get_rng_state()), "flips drew from the global torch RNG"

    ratio = flipped / batch.lap_pe
    assert torch.equal(ratio.abs(), torch.ones_like(ratio)), "a flip only changes signs"
    for g in (0, 1):
        rows = ratio[batch.batch == g]
        assert torch.equal(rows, rows[:1].expand_as(rows)), "one sign per eigenvector per graph"

    encoding.eval()
    plain = lap.proj(batch.lap_pe)
    assert torch.allclose(encoding(batch, dtype=torch.float32, device=cpu), plain), "no flips in eval"


def _lap_cfg(**values):
    from yacs.config import CfgNode as CN

    node = CN()
    node.k = values.get("k", 8)
    node.sign_flip = values.get("sign_flip", True)
    return node


def test_lap_without_sign_flip_is_deterministic_in_training():
    encoding = _randomised(build_node_positional_encoding(_cfg(["lap"], lap=_lap_cfg(k=4, sign_flip=False)), N_EMBED))
    batch = _lap_batch()
    cpu = torch.device("cpu")

    encoding.train()
    assert torch.equal(
        encoding(batch, dtype=torch.float32, device=cpu), encoding(batch, dtype=torch.float32, device=cpu)
    )


def test_lap_reports_graphs_built_without_its_eigenvectors():
    encoding = build_node_positional_encoding(_cfg(["lap"]), N_EMBED)

    with pytest.raises(ValueError, match="attach_positional_inputs"):
        encoding(_batch(torch.rand(4, 2), edges=False), dtype=torch.float32, device=torch.device("cpu"))


def test_lap_rejects_eigenvectors_from_a_different_k():
    encoding = build_node_positional_encoding(_cfg(["lap"], lap=_lap_cfg(k=8)), N_EMBED)

    with pytest.raises(ValueError, match="pe.lap.k"):
        encoding(_lap_batch(k=4), dtype=torch.float32, device=torch.device("cpu"))


def test_attach_positional_inputs_stores_eigenvectors_only_when_enabled():
    from torch_geometric.data import Data

    from interscale.module.global_modules.positional_encodings import attach_positional_inputs
    from interscale.tl import laplacian_pe

    edge_index = _connected_geometric_graph(40, radius=0.3)
    off, on = Data(edge_index=edge_index, num_nodes=40), Data(edge_index=edge_index, num_nodes=40)

    attach_positional_inputs([off], _cfg(["naive"]))
    attach_positional_inputs([on], _cfg(["lap"], lap=_lap_cfg(k=5)))

    assert getattr(off, "lap_pe", None) is None
    assert torch.equal(on.lap_pe, laplacian_pe(edge_index, 40, 5))
