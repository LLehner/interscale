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
        (["node: [rw]", "rw:", "  steps: 0"], True, "rw.steps"),
        (["node: [distance]"], True, "belong under pe.bias"),
        (["bias: [naive]"], True, "belong under pe.node"),
        (["bias: [distance]", "distance:", "  num_kernels: 1"], True, "num_kernels"),
        (["bias: [distance]", "distance:", "  max_dist: 0.0"], True, "max_dist"),
        (["bias: [distance]"], False, "dataset.spatial_key"),
        (["bias: [distance]", "distance:", "  kind: gaussian"], True, "distance.kind"),
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
        "rw": ("node", "rw_pe", True),
        "distance": ("bias", "pos", False),
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


# --------------------------------------------------------------------------- random-walk return probabilities


def _brute_force_rw(edge_index, n, steps):
    """diag((D^-1 A)^t) from the full dense powers -- the definition, independent of any shortcut."""
    adj = torch.zeros(n, n, dtype=torch.float64)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = ((adj + adj.t()) > 0).double()
    adj.fill_diagonal_(0)
    deg = adj.sum(1, keepdim=True)
    walk = torch.where(deg > 0, adj / deg.clamp(min=1), torch.zeros_like(adj))
    power, cols = torch.eye(n, dtype=torch.float64), []
    for _ in range(steps):
        power = power @ walk
        cols.append(power.diagonal().clone())
    return torch.stack(cols, dim=1)


def test_rw_pe_equals_the_definition_at_every_step():
    """The half-power shortcut against full dense powers: odd and even steps, and an isolated cell."""
    from interscale.tl import random_walk_pe

    edge_index = _geometric_graph(70, radius=0.2, seed=3)
    steps = 9

    ours = random_walk_pe(edge_index, 70, steps).double()

    assert torch.allclose(ours, _brute_force_rw(edge_index, 70, steps), atol=1e-6)
    assert not ours[:, 0].any(), "one step cannot return without a self loop"


def test_rw_pe_matches_pyg():
    from torch_geometric.data import Data
    from torch_geometric.transforms import AddRandomWalkPE

    from interscale.tl import random_walk_pe

    n, steps = 120, 8
    edge_index = _geometric_graph(n, radius=0.15, seed=4)

    ours = random_walk_pe(edge_index, n, steps)
    theirs = AddRandomWalkPE(steps, attr_name="pe")(Data(edge_index=edge_index, num_nodes=n)).pe

    assert torch.allclose(ours, theirs, atol=1e-5)


def test_rw_pe_gives_isolated_cells_zeros_and_walks_probabilities():
    from interscale.tl import random_walk_pe

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])  # a triangle; node 3 isolated
    pe = random_walk_pe(edge_index, 4, 4)

    assert not pe[3].any()
    # On a triangle: back after 2 steps with p 1/2, after 3 steps with p 1/4 (two orientations x 1/8).
    assert torch.allclose(pe[0], torch.tensor([0.0, 0.5, 0.25, 0.375]), atol=1e-6)


def test_rw_encoder_normalises_small_spreads_instead_of_squashing_them():
    """Late-step return probabilities vary across cells by ~1e-3, a variance of ~1e-6 -- below
    BatchNorm's default eps of 1e-5, which would keep only ~30% of that spread."""
    from interscale.tl import random_walk_pe

    encoding = build_node_positional_encoding(_cfg(["rw"]), N_EMBED)
    rw = encoding.encoders["rw"]
    probs = random_walk_pe(_geometric_graph(400, radius=0.12, seed=6), 400, 16)
    probs[:, 1:] = probs[:, 1:].mean(0) + (probs[:, 1:] - probs[:, 1:].mean(0)) * 0.1  # spreads ~1e-3

    rw.train()
    normed = rw.norm(probs)
    default = nn.BatchNorm1d(16).train()(probs)

    assert not normed[:, 0].any(), "step 1 is constant (no self loops) and stays at zero"
    assert torch.allclose(normed[:, 1:].mean(0), torch.zeros(15), atol=1e-4)
    assert bool((normed[:, 1:].std(0, unbiased=False) > 0.95).all()), "the spread is normalised to ~1"
    assert bool((default[:, -1].std(unbiased=False) < 0.5).all()), "BatchNorm's default eps would squash it"


def test_rw_rejects_probabilities_from_a_different_step_count():
    encoding = build_node_positional_encoding(_cfg(["rw"]), N_EMBED)
    batch = _batch(torch.rand(5, 2), edges=False)
    batch.rw_pe = torch.rand(5, 4)

    with pytest.raises(ValueError, match="pe.rw.steps"):
        encoding(batch, dtype=torch.float32, device=torch.device("cpu"))


def test_attach_positional_inputs_stores_return_probabilities_only_when_enabled():
    from torch_geometric.data import Data

    from interscale.module.global_modules.positional_encodings import attach_positional_inputs
    from interscale.tl import random_walk_pe

    edge_index = _geometric_graph(40, radius=0.3, seed=5)
    off, on = Data(edge_index=edge_index, num_nodes=40), Data(edge_index=edge_index, num_nodes=40)

    attach_positional_inputs([off], _cfg(["lap"]))
    attach_positional_inputs([on], _cfg(["rw"]))

    assert getattr(off, "rw_pe", None) is None
    assert torch.equal(on.rw_pe, random_walk_pe(edge_index, 40, 16))


# --------------------------------------------------------------------------- relative distance bias


def test_pad_like_lays_values_out_exactly_as_pad_batch_lays_out_tokens():
    """The layout the bias relies on, including the case that makes it non-trivial: pad_batch
    subsampling a graph longer than max_seq_len, with masked nodes it must keep."""
    import random

    from interscale.tl import pad_batch, pad_like

    sizes = (5, 9, 7)
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
    values = torch.randn(len(batch), 3)
    keep = torch.zeros(len(batch), dtype=torch.bool)
    keep[[6, 9, 12]] = True  # masked nodes of graph 1, which pad_batch has to keep

    for get_mask in (False, True):
        random.seed(0)
        padded, _, index_nodes, *_ = pad_batch(values, batch, 6, get_mask=get_mask, keep_indices=keep)
        assert torch.equal(pad_like(values, batch, index_nodes, padded.shape[0]), padded.permute(1, 0, 2))


def test_distance_bias_interpolates_between_knots_and_is_flat_beyond():
    from interscale.module.global_modules.positional_encodings import DistanceBias

    bias = DistanceBias(n_heads=2, num_kernels=5, max_dist=400.0)  # knots every 100 um
    with torch.no_grad():
        bias.table.copy_(torch.tensor([[0.0, 1.0, 3.0, 2.0, -1.0], [5.0, 4.0, 3.0, 2.0, 1.0]]))

    dist = torch.tensor([[[0.0, 100.0, 150.0, 400.0, 900.0]]])
    out = bias(dist)[0, :, 0]

    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 2.0, -1.0, -1.0]))
    assert torch.allclose(out[1], torch.tensor([5.0, 4.0, 3.5, 1.0, 1.0]))
    knots, values = bias.profile()
    assert torch.allclose(knots, torch.tensor([0.0, 100.0, 200.0, 300.0, 400.0]))
    assert torch.equal(values, bias.table.detach())


def _bias_cfg(kind="profile", **pe):
    pe.setdefault("bias", ["distance"])
    cfg = _cfg(**pe)
    cfg.model.global_component.parameters.pe.distance.kind = kind
    return cfg


def _random_bias(cfg, n_heads, seed=2):
    """The configured bias with random parameters -- at their zero init every bias is 0."""
    from interscale.module.global_modules.positional_encodings import build_attention_bias

    bias = build_attention_bias(cfg, n_heads)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in bias.parameters():
            param.copy_(torch.randn(param.shape, generator=gen))
    return bias


def test_the_linear_bias_is_a_slope_per_head_on_the_distance_in_mm():
    from interscale.module.global_modules.positional_encodings import LinearDistanceBias

    bias = LinearDistanceBias(n_heads=2)
    assert not bias.slope.any(), "starts at zero"
    with torch.no_grad():
        bias.slope.copy_(torch.tensor([-2.0, 0.5]))

    out = bias(torch.tensor([[[0.0, 500.0, 3000.0]]]))[0, :, 0]

    assert torch.allclose(out[0], torch.tensor([0.0, -1.0, -6.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.25, 1.5]))
    knots, values = bias.profile(max_dist=2000.0)
    assert torch.allclose(values, torch.tensor([[0.0, -4.0], [0.0, 1.0]]))


def test_the_distance_kind_selects_the_function_and_linear_ignores_the_profile_settings():
    from interscale.module.global_modules.positional_encodings import (
        DistanceBias,
        LinearDistanceBias,
        build_attention_bias,
    )

    assert isinstance(build_attention_bias(_bias_cfg("profile"), 2).encoders["distance"], DistanceBias)
    linear = _bias_cfg("linear")
    linear.model.global_component.parameters.pe.distance.num_kernels = 1  # would fail for a profile
    assert isinstance(build_attention_bias(linear, 2).encoders["distance"], LinearDistanceBias)


def test_the_bias_is_laid_out_like_the_mask_with_zero_cls_and_padding():
    """Entry (b * heads + h, i, j) is head h's bias between the cells at tokens i and j of graph b:
    left-padded, CLS last, graph-major -- the order mask.repeat_interleave(heads) uses."""
    n_heads = 3
    cfg = _bias_cfg()
    bias_module = _random_bias(cfg, n_heads)
    distance = bias_module.encoders["distance"]
    a, b = torch.rand(4, 2) * 1500, torch.rand(6, 2) * 1500
    batch = _batch(a, b, edges=False)
    padding = torch.tensor([[True, True, False, False, False, False], [False] * 6])

    out = bias_module(batch, [list(range(4)), list(range(6))], padding, torch.float32, torch.device("cpu"))

    assert out.shape == (2 * n_heads, 7, 7)
    for g, coords, lo in ((0, a, 2), (1, b, 0)):
        expected = distance(torch.cdist(coords, coords).unsqueeze(0))[0]  # [heads, n, n]
        for h in range(n_heads):
            block = out[g * n_heads + h]
            assert torch.allclose(block[lo:6, lo:6], expected[h], atol=1e-4), (g, h)
            assert not block[:lo].any() and not block[:, :lo].any(), "padding carries no bias"
            assert not block[6].any() and not block[:, 6].any(), "CLS carries no bias"


@pytest.mark.parametrize("kind", ["profile", "linear"])
def test_the_bias_is_translation_and_rotation_invariant(kind):
    bias_module = _random_bias(_bias_cfg(kind), 2)
    a = torch.rand(5, 2) * 1000
    angle = torch.tensor(0.7)
    rot = torch.stack([torch.stack([angle.cos(), -angle.sin()]), torch.stack([angle.sin(), angle.cos()])])
    padding = torch.zeros(1, 5, dtype=torch.bool)

    def run(coords):
        return bias_module(_batch(coords, edges=False), [list(range(5))], padding, torch.float32, torch.device("cpu"))

    assert torch.allclose(run(a), run(a @ rot.t() + torch.tensor([300.0, -200.0])), atol=1e-3)


def _hook(cfg, *, long_range, hops, bias=None):
    from interscale.module.global_modules import TransformerNodeEncoderHook

    return TransformerNodeEncoderHook(
        max_seq_len=64,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=2,
        dim_feedforward=16,
        long_range_attention=long_range,
        local_mask_hops=hops,
        attention_bias=bias,
        n_input=N_EMBED,
        n_output=N_EMBED,
        n_embed=N_EMBED,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )


def test_blocked_pairs_become_minus_inf_and_every_other_pair_keeps_its_bias():
    cfg = _bias_cfg()
    plain = _hook(cfg, long_range=True, hops=1)
    biased = _hook(cfg, long_range=True, hops=1, bias=_random_bias(cfg, 2))
    g = torch.rand(8, 2) * 500
    emb = torch.randn(8, N_EMBED)

    boolean = plain.common_step_local_to_global(_batch(g), emb, eval_step=True)[3]
    merged = biased.common_step_local_to_global(_batch(g), emb, eval_step=True)[3]
    alone = biased.attention_bias(
        _batch(g), [list(range(8))], torch.zeros(1, 8, dtype=torch.bool), torch.float32, torch.device("cpu")
    )

    assert merged.is_floating_point() and boolean.dtype == torch.bool
    assert torch.equal(torch.isinf(merged), boolean), "-inf exactly where the boolean mask blocks"
    assert torch.equal(merged[~boolean], alone[~boolean])


@pytest.mark.parametrize("kind", ["profile", "linear"])
def test_a_zero_bias_changes_nothing_and_raises_no_mask_type_warning(kind):
    """Both kinds start at zero: the float-mask path must then compute what the boolean one did,
    without torch's mixed-mask-type deprecation warning on every layer."""
    import warnings

    cfg = _bias_cfg(kind)
    torch.manual_seed(0)
    plain = _hook(cfg, long_range=True, hops=1).eval()
    torch.manual_seed(0)
    from interscale.module.global_modules.positional_encodings import build_attention_bias

    biased = _hook(cfg, long_range=True, hops=1, bias=build_attention_bias(cfg, 2)).eval()
    g = torch.rand(9, 2) * 500
    emb = torch.randn(9, N_EMBED)

    outs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for module in (plain, biased):
            padded, padding, _, mask = module.common_step_local_to_global(_batch(g), emb, eval_step=True)
            outs.append(module(padded, padding, mask, register_hook=False)[0])

    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert not [w for w in caught if "mismatched" in str(w.message).lower()]


@pytest.mark.parametrize("kind", ["profile", "linear"])
@pytest.mark.parametrize("hops", [1, 2, 3])
def test_a_biased_dense_graph_stays_finite_and_blocked_pairs_get_no_attention(hops, kind):
    """The long-range mask's NaN guarantee must survive the bias: on a complete graph every pair
    is blocked, so each row keeps only CLS. Rows still sum to one and the gradient is finite."""
    n = 10
    cfg = _bias_cfg(kind)
    module = _hook(cfg, long_range=True, hops=hops, bias=_random_bias(cfg, 2)).train()
    coords = torch.rand(n, 2) * 300
    batch = _batch(coords)
    full = torch.ones(n, n, dtype=torch.bool).nonzero().t()
    batch.edge_index = full[:, full[0] != full[1]]
    emb = torch.randn(n, N_EMBED, requires_grad=True)

    padded, padding, _, mask = module.common_step_local_to_global(batch, emb, eval_step=True)
    out, _, attn = module(padded, padding, mask, register_hook=True)
    out.sum().backward()

    assert torch.isfinite(out).all() and torch.isfinite(emb.grad).all()
    weights = attn[0]  # layer 0; one graph, so [heads, S+1, S+1]
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)
    assert not weights[:, :n, :n].any(), "every cell-to-cell pair is blocked on a complete graph"


def test_the_checkpoint_name_and_a_sweep_arm_carry_the_bias():
    from interscale.config.sweep import apply_sweep_config
    from interscale.tl.utils import get_model_filename_prefix

    assert get_model_filename_prefix(_cfg(["naive"], bias=["distance"]), True, True).endswith("pe-naive+distance_")
    assert get_model_filename_prefix(_bias_cfg("linear"), True, True).endswith("pe-distance-linear_"), (
        "the two kinds must not share a checkpoint name"
    )

    node, bias = "model.global_component.parameters.pe.node", "model.global_component.parameters.pe.bias"
    arms = {"none": {node: [], bias: []}, "distance": {node: [], bias: ["distance"]}}
    cfg = _cfg()
    cfg.freeze()
    cfg, _ = apply_sweep_config(cfg, "hyperparmeter", {"arm": "distance"}, sweep_params=["arm"], arms=arms)
    assert list(cfg.model.global_component.parameters.pe.bias) == ["distance"]


@pytest.mark.parametrize("kind", ["profile", "linear"])
def test_the_bias_trains_end_to_end(kind):
    """Through the training plan, long-range mask on: the zero parameters move, so the gradient reaches them."""
    import lightning.pytorch as pl
    from torch_geometric.data import Data

    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.module.global_modules import TransformerNodeEncoderHook
    from interscale.module.global_modules.positional_encodings import build_attention_bias
    from interscale.train._trainingplans import TrainingPlan

    def graph(n, seed):
        g = torch.Generator().manual_seed(seed)
        src = list(range(n - 1)) + list(range(1, n))
        d = Data(
            x=torch.randn(n, 6, generator=g),
            edge_index=torch.tensor([src, list(range(1, n)) + list(range(n - 1))], dtype=torch.long),
            pos=torch.rand(n, 2, generator=g) * 2000,
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
    bias = build_attention_bias(_bias_cfg(kind, local=False), 2)
    module = TransformerNodeEncoderHook(
        max_seq_len=16,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=8,
        long_range_attention=True,
        local_mask_hops=1,
        attention_bias=bias,
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
    assert all(param.any() for param in bias.parameters()), "the gradient never reached the bias"
