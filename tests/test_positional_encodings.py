"""Positional encodings of the global transformer (PE_plan.md, Stages 0-8).

Pinned here: off is the model from before PEs; on changes neither the other initial weights nor
(except RoPE) the first forward; each cell's encoding travels with its token through pad_batch's
subsampling; the coordinate handling (units, centring, rotation) does what the config says; each
encoding computes what its definition says; and the config, checkpoint name and sweep arms carry
the choice.
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

    padded, _, index_nodes, _, _ = module.common_step_local_to_global(batch, torch.zeros(10, N_EMBED), eval_step=True)

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
        (["bias: [distance]", "distance:", "  max_dist: -1.0"], True, "max_dist"),
        (["bias: [distance]"], False, "dataset.spatial_key"),
        (["bias: [distance]", "distance:", "  kind: gaussian"], True, "distance.kind"),
    ],
)
def test_invalid_pe_configs_fail_at_load(tmp_path, lines, spatial, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, lines, spatial=spatial))


def test_the_registry_holds_the_encodings_built_so_far():
    """What each encoding reads, and which precompute: a new entry has to be added here on purpose."""
    assert {name: (spec.kind, spec.requires, spec.precompute is not None) for name, spec in PE_REGISTRY.items()} == {
        "naive": ("node", "pos", False),
        "sinusoidal": ("node", "pos", False),
        "lap": ("node", "lap_pe", True),
        "rw": ("node", "rw_pe", True),
        "distance": ("bias", "pos", False),
        "spectral": ("bias", "spectral_pe", True),
        "rope": ("rotary", "pos", False),
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
    # The default 0 is derived from AnnData by the model; these tests build modules directly.
    cfg.model.global_component.parameters.pe.distance.max_dist = 2000.0
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


def _hook(cfg, *, long_range, hops, bias=None, rotary=None):
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
        rotary_encoding=rotary,
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
            padded, padding, _, mask, _ = module.common_step_local_to_global(_batch(g), emb, eval_step=True)
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

    padded, padding, _, mask, _ = module.common_step_local_to_global(batch, emb, eval_step=True)
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


# --------------------------------------------------------------------------- derived max_dist


def test_point_set_diameter_is_the_farthest_pair():
    import numpy as np
    from scipy.spatial.distance import pdist

    from interscale.tl.positional import point_set_diameter

    rng = np.random.default_rng(0)
    cloud = rng.uniform(0, 300, size=(400, 2))
    assert point_set_diameter(cloud) == pytest.approx(pdist(cloud).max())
    assert point_set_diameter(np.array([[0, 0], [3, 0], [0, 4], [3, 4]])) == pytest.approx(5.0)
    # No 2D hull: collinear points and pairs fall back to the bounding-box diagonal, exact here.
    assert point_set_diameter(np.array([[0, 0], [1, 1], [2, 2], [5, 5]])) == pytest.approx(5 * math.sqrt(2))
    assert point_set_diameter(np.array([[1, 1], [4, 5]])) == pytest.approx(5.0)
    assert point_set_diameter(np.array([[1, 1]])) == 0.0


def _slides_adata(model_cls=None, unit_offset=10_000.0):
    """Two slides far apart in obsm: a 30 x 40 rectangle (diameter 50) and a 6 x 8 one (10)."""
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    big = [[0, 0], [30, 0], [0, 40], [30, 40], [10, 10]]
    small = [[0, 0], [6, 0], [0, 8], [6, 8]]
    coords = np.array(big + [[x + unit_offset, y] for x, y in small], dtype=np.float32)
    adata = AnnData(X=np.random.default_rng(0).integers(0, 10, size=(len(coords), 6)).astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["sample"] = pd.Categorical(["s1"] * len(big) + ["s2"] * len(small))
    adata.obs["split"] = pd.Categorical(["train"] * 6 + ["val"] * 3)
    if model_cls is not None:
        model_cls._setup_anndata(
            adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
        )
    return adata


def _derived_cfg(kind="profile", unit=1.0):
    cfg = _cfg(bias=["distance"])
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.spatial_unit_um = unit
    cfg.model.global_component.parameters.pe.distance.kind = kind
    cfg.freeze()
    return cfg


def test_max_dist_zero_is_the_largest_diameter_within_one_slide():
    """Per slide, then the max: the 10 mm gap between the two slides is not a distance any pair
    of tokens ever has, since one graph is one slide."""
    from interscale.module.global_modules.positional_encodings import resolve_distance_range

    cfg = _derived_cfg(unit=2.0)
    resolved = resolve_distance_range(cfg, _slides_adata())

    assert resolved.model.global_component.parameters.pe.distance.max_dist == pytest.approx(100.0)
    assert cfg.model.global_component.parameters.pe.distance.max_dist == 0.0  # caller's cfg untouched


def test_max_dist_is_only_derived_when_asked():
    from interscale.module.global_modules.positional_encodings import resolve_distance_range

    adata = _slides_adata()
    linear = _derived_cfg(kind="linear")
    assert resolve_distance_range(linear, adata) is linear

    fixed = _derived_cfg().clone()
    fixed.defrost()
    fixed.model.global_component.parameters.pe.distance.max_dist = 700.0
    assert resolve_distance_range(fixed, adata) is fixed

    off = _cfg()
    assert resolve_distance_range(off, adata) is off  # no coordinates needed without the bias


def test_a_module_built_without_the_data_cannot_use_max_dist_zero():
    from interscale.module.global_modules.positional_encodings import build_attention_bias

    with pytest.raises(ValueError, match="nothing derived it"):
        build_attention_bias(_derived_cfg(), 2)


def test_the_model_builds_its_bias_over_the_derived_range():
    from interscale.model.combined_model import CombinedModel

    model = CombinedModel(_slides_adata(CombinedModel), cfg=_derived_cfg())
    distance = model.module.global_module.attention_bias.encoders["distance"]

    assert float(distance.max_dist) == pytest.approx(50.0)
    assert distance.profile()[0][-1].item() == pytest.approx(50.0)


def test_the_global_model_builds_from_the_derived_range_too():
    """GlobalModel used to build its module from the cfg it was given rather than the one with the
    derived range, so the default max_dist 0 raised at construction."""
    from interscale.model.global_model import GlobalModel

    cfg = _derived_cfg().clone()
    cfg.defrost()
    cfg.model.global_component.parameters.type_gex_embedding = "PCA"
    cfg.model.global_component.parameters.pe.rotary = ["rope"]
    cfg.freeze()
    model = GlobalModel(_slides_adata(GlobalModel), cfg=cfg)

    assert float(model.module.attention_bias.encoders["distance"].max_dist) == pytest.approx(50.0)
    assert model.module.rotary_encoding.encoders["rope"].wavelengths().max().item() == pytest.approx(100.0, rel=1e-5)


def test_the_checkpoint_keeps_the_range_the_table_was_trained_on():
    """Loaded onto other data the model re-derives another range; the stored one must win, or
    every learned knot is read at the wrong distance."""
    from interscale.module.global_modules.positional_encodings import DistanceBias

    trained = DistanceBias(n_heads=2, num_kernels=5, max_dist=400.0)
    assert "max_dist" in trained.state_dict()

    reloaded = DistanceBias(n_heads=2, num_kernels=5, max_dist=900.0)
    reloaded.load_state_dict(trained.state_dict())
    assert float(reloaded.max_dist) == 400.0
    assert reloaded.profile()[0][-1].item() == 400.0


def test_checkpoints_from_before_the_buffer_still_load():
    from interscale.model.base._base_model import _OPTIONAL_STATE_PREFIXES

    key = "global_module.attention_bias.encoders.distance.max_dist"
    assert key.rsplit(".", 1)[-1].startswith(_OPTIONAL_STATE_PREFIXES)


# --------------------------------------------------------------------------- LapPE size warning


def _sized_graphs(*sizes):
    return [SimpleNamespace(num_nodes=n) for n in sizes]


def test_lap_warns_when_graph_sizes_differ_enough_to_rescale_it(caplog):
    from interscale.module.global_modules.positional_encodings import warn_if_lap_scales_differ

    lap = _cfg(["lap"])
    with caplog.at_level("WARNING"):
        assert warn_if_lap_scales_differ(_sized_graphs(500, 400, 450), lap) == pytest.approx(1.25)
    assert not caplog.records

    with caplog.at_level("WARNING"):
        assert warn_if_lap_scales_differ(_sized_graphs(5000, 400, 450), lap) == pytest.approx(12.5)
    assert "LapPE" in caplog.text and "400-5000" in caplog.text

    caplog.clear()
    assert warn_if_lap_scales_differ(_sized_graphs(5000, 40), _cfg(["rw"])) is None
    assert not caplog.records


# --------------------------------------------------------------------------- rotary (RoPE)


def _rope_cfg(kind="axial", *, node=(), max_wavelength=4000.0, **pe):
    """RoPE on, with two heads so the test width N_EMBED (8) gives the 4-dim heads 2D RoPE needs."""
    cfg = _cfg(node, **pe)
    cfg.model.global_component.parameters.n_heads = 2
    cfg.model.global_component.parameters.pe.rotary = ["rope"]
    cfg.model.global_component.parameters.pe.rope.kind = kind
    # The default 0 is derived from AnnData by the model; these tests build modules directly.
    cfg.model.global_component.parameters.pe.rope.max_wavelength = max_wavelength
    return cfg


def _plain_cfg():
    """The RoPE-free twin of `_rope_cfg`: the same two heads."""
    cfg = _cfg()
    cfg.model.global_component.parameters.n_heads = 2
    return cfg


def _rope(kind="axial", n_heads=2, head_dim=8, max_wavelength=4000.0):
    from interscale.module.global_modules.positional_encodings import RotaryPE

    return RotaryPE(n_heads, head_dim, kind, 50.0, max_wavelength)


def _rotated_logits(rope, pos, q, k):
    from interscale.module.global_modules.transformer_utils import apply_rotary

    cos, sin = rope(pos, torch.float32)
    return apply_rotary(q, cos, sin) @ apply_rotary(k, cos, sin).transpose(-1, -2)


def test_apply_rotary_multiplies_each_pair_by_its_phase():
    """Pair i is the complex number x[i] + 1j * x[i + d/2], turned by exp(1j * theta_i)."""
    from interscale.module.global_modules.transformer_utils import apply_rotary

    x = torch.randn(3, 5, 8)
    theta = torch.randn(3, 5, 4)
    out = apply_rotary(x, theta.cos().repeat(1, 1, 2), theta.sin().repeat(1, 1, 2))

    z = torch.complex(x[..., :4], x[..., 4:]) * torch.polar(torch.ones_like(theta), theta)
    assert torch.allclose(out, torch.cat([z.real, z.imag], dim=-1), atol=1e-6)


def test_rope_deals_geometric_wavelengths_across_heads_and_sets_each_kinds_axes():
    from interscale.module.global_modules.positional_encodings import RotaryPE

    axial = RotaryPE(n_heads=2, head_dim=8, kind="axial", min_wavelength=10.0, max_wavelength=10000.0)
    # 2 heads x 2 wavelengths per axis, geometric over the range, head h taking every 2nd from h.
    expected = torch.tensor([[10.0, 1000.0, 10.0, 1000.0], [100.0, 10000.0, 100.0, 10000.0]])
    assert torch.allclose(axial.wavelengths(), expected, rtol=1e-5)
    freqs = axial.frequencies()
    assert not freqs[:, :2, 1].any() and not freqs[:, 2:, 0].any(), "axial: first half x, second half y, exactly"

    mixed = RotaryPE(n_heads=2, head_dim=8, kind="mixed", min_wavelength=10.0, max_wavelength=10000.0)
    assert [name for name, _ in mixed.named_parameters()] == ["direction"]
    assert not list(axial.parameters()), "axial has nothing to learn"
    assert torch.allclose(mixed.wavelengths(), expected, rtol=1e-5)
    f = mixed.frequencies()
    degrees = torch.rad2deg(torch.atan2(f[..., 1], f[..., 0]))
    # Head h's frame turned by h * 90 / n_heads degrees: the heads' axes spread over 0, 45, 90, 135.
    assert torch.allclose(degrees, torch.tensor([[0.0, 0.0, 90.0, 90.0], [45.0, 45.0, 135.0, 135.0]]), atol=1e-4)


@pytest.mark.parametrize("kind", ["axial", "mixed"])
def test_rope_logits_depend_only_on_the_offset_between_two_cells(kind):
    """The defining property: translate the slide and no logit moves; move one cell and only the
    logits that involve it do. Coordinates of several mm, on a quarter-µm grid so that the
    translated ones are exact in float32 -- otherwise their own rounding (~1e-3 µm) is what shows."""
    gen = torch.Generator().manual_seed(0)
    rope = _rope(kind)
    if kind == "mixed":
        with torch.no_grad():  # away from the init, as after training
            rope.direction.add_(0.3 * torch.randn(rope.direction.shape, generator=gen))
    pos = torch.randint(0, 12000, (1, 6, 2), generator=gen).float() * 0.25
    q, k = torch.randn(2, 6, 8, generator=gen), torch.randn(2, 6, 8, generator=gen)  # [graphs x heads, tokens, dim]

    base = _rotated_logits(rope, pos, q, k)
    assert torch.allclose(base, _rotated_logits(rope, pos + torch.tensor([5000.0, -2500.0]), q, k), atol=1e-4)
    assert not torch.allclose(base, q @ k.transpose(-1, -2), atol=1e-2), "the rotation acts at all"

    moved = pos.clone()
    moved[0, 0] += torch.tensor([300.0, 0.0])
    changed = (_rotated_logits(rope, moved, q, k) - base).abs() > 1e-4
    assert changed[:, 0].any() and changed[:, :, 0].any()
    assert not changed[:, 1:, 1:].any(), "pairs not involving the moved cell keep their logit"


def test_the_attention_rotates_queries_and_keys_so_it_sees_offsets_only():
    """Through the layer's attention itself: translated cells give the same output and weights.
    That only holds if queries and keys are both rotated -- rotating one leaves absolute angles."""
    from interscale.module.global_modules.transformer_utils import MultiHeadAttentionWithEdits

    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    attention = MultiHeadAttentionWithEdits(8, 2, dropout=0.0).eval()
    rope = _rope("mixed", n_heads=2, head_dim=4)
    x = torch.randn(6, 1, 8, generator=gen)  # [tokens, graphs, width], every token a cell
    pos = torch.randint(0, 12000, (1, 6, 2), generator=gen).float() * 0.25

    def attend(p):
        return attention(x, x, x, need_weights=True, average_attn_weights=False, rotary=rope(p, torch.float32))

    out, weights = attend(pos)
    out_moved, weights_moved = attend(pos + torch.tensor([4000.0, 1500.0]))
    assert torch.allclose(out, out_moved, atol=1e-5) and torch.allclose(weights, weights_moved, atol=1e-5)
    plain_out, _ = attention(x, x, x, need_weights=True)
    assert not torch.allclose(out, plain_out, atol=1e-3), "the rotation acts at all"


def test_axial_rope_sees_direction():
    """Unlike LapPE, RWPE and the distance bias: turning the slide by 90 degrees changes the logits."""
    rope = _rope("axial")
    pos = torch.rand(1, 6, 2) * 3000
    q, k = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    turned = torch.stack([-pos[..., 1], pos[..., 0]], dim=-1)

    assert not torch.allclose(_rotated_logits(rope, pos, q, k), _rotated_logits(rope, turned, q, k), atol=1e-2)


def test_the_rotation_follows_each_token_through_pad_batch_with_cls_and_padding_unrotated():
    """Row b * heads + h, token j: the rotation of the cell pad_batch put at token j of graph b.
    Graph a is subsampled (9 cells, 5 tokens), graph b padded (3 cells); CLS is the last token."""
    import random

    cfg = _rope_cfg(center_coords=False)
    cfg.model.global_component.parameters.max_seq_len = 5
    module = _combined(cfg).global_module.eval()
    rope = module.rotary_encoding.encoders["rope"]
    a, b = torch.rand(9, 2) * 2000, torch.rand(3, 2) * 2000

    random.seed(0)
    _, _, index_nodes, _, (cos, sin) = module.common_step_local_to_global(
        _batch(a, b), torch.zeros(12, N_EMBED), eval_step=True
    )

    assert cos.shape == sin.shape == (2 * 2, 6, 4)
    assert len(index_nodes[0]) == 5 and index_nodes[0] != list(range(5)), "a subsample that tells orders apart"
    for g, coords in ((0, a), (1, b)):
        kept = index_nodes[g]
        exp_cos, exp_sin = rope(coords[kept].unsqueeze(0), torch.float32)  # [heads, kept, head_dim]
        lo = 5 - len(kept)
        for h in range(2):
            row = g * 2 + h
            assert torch.allclose(cos[row, lo:5], exp_cos[h], atol=1e-6), (g, h)
            assert torch.allclose(sin[row, lo:5], exp_sin[h], atol=1e-6), (g, h)
            assert (cos[row, :lo] == 1).all() and not sin[row, :lo].any(), "padding is not rotated"
            assert (cos[row, 5] == 1).all() and not sin[row, 5].any(), "CLS, last, is not rotated"


def test_rope_off_builds_nothing():
    module = _combined(_plain_cfg())

    assert module.global_module.rotary_encoding is None
    assert not [k for k in module.state_dict() if "rotary" in k]


@pytest.mark.parametrize("kind", ["axial", "mixed"])
def test_rope_adds_its_frequencies_and_changes_no_other_initial_weight(kind):
    torch.manual_seed(0)
    off = _combined(_plain_cfg())
    torch.manual_seed(0)
    on = _combined(_rope_cfg(kind))

    off_state, on_state = off.state_dict(), on.state_dict()
    for key, value in off_state.items():
        assert torch.equal(value, on_state[key]), key
    prefix = "global_module.rotary_encoding.encoders.rope."
    assert sorted(set(on_state) - set(off_state)) == [prefix + "direction", prefix + "magnitude"]
    learned = sorted(set(dict(on.named_parameters())) - set(dict(off.named_parameters())))
    assert learned == ([prefix + "direction"] if kind == "mixed" else [])


def test_rope_rotates_with_the_node_encodings_and_draws_only_from_the_pe_generator():
    """rotate_train turns each graph once per pass: the rotary encoding must see the coordinates the
    node encodings saw, not a second draw, and nothing may come from the global torch RNG."""
    module = _combined(_rope_cfg(node=["naive"], rotate_train=True)).global_module.train()
    assert module.rotary_encoding.frame is module.positional_encoding.frame

    seen = {}
    encoders = {"node": module.positional_encoding.encoders["naive"], "rope": module.rotary_encoding.encoders["rope"]}
    for name, encoder in encoders.items():
        original = encoder.forward

        def spy(pos, *args, _name=name, _original=original):
            seen[_name] = pos.detach().clone()
            return _original(pos, *args)

        encoder.forward = spy
    g = torch.rand(6, 2) * 1000
    emb = torch.randn(6, N_EMBED)
    state = torch.get_rng_state()
    module.common_step_local_to_global(_batch(g), emb)

    assert torch.equal(state, torch.get_rng_state()), "drew from the global torch RNG"
    assert torch.allclose(seen["rope"][0, :6], seen["node"], atol=1e-6), "the same rotated coordinates"
    centred = (g - g.mean(0)).float()
    assert not torch.allclose(seen["node"], centred, atol=1e-2), "a rotation was applied"
    assert torch.allclose(seen["node"].norm(dim=1), centred.norm(dim=1), atol=1e-3)


def test_forward_needs_the_rotation_of_its_own_batch_and_leaves_none_behind():
    module = _combined(_rope_cfg()).global_module.eval()
    padded, padding, _, mask, rotary = module.common_step_local_to_global(
        _batch(torch.rand(6, 2) * 1000), torch.randn(6, N_EMBED), eval_step=True
    )
    layers = module.transformer_encoder.layers

    with pytest.raises(ValueError, match="needs the rotation"):
        module(padded, padding, mask)
    other = module.common_step_local_to_global(
        _batch(torch.rand(4, 2) * 1000), torch.randn(4, N_EMBED), eval_step=True
    ).rotary
    with pytest.raises(ValueError, match="another batch"):
        module(padded, padding, mask, rotary=other)
    with pytest.raises(ValueError, match="no rotary encoding"):
        _combined(_plain_cfg()).global_module.eval()(padded, padding, mask, rotary=rotary)

    module(padded, padding, mask, rotary=rotary)
    assert all(layer.rotary is None for layer in layers)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    layers[1].self_attn.forward = boom
    with pytest.raises(RuntimeError, match="boom"):
        module(padded, padding, mask, rotary=rotary)
    assert all(layer.rotary is None for layer in layers), "cleared even when a layer raises"


def test_rope_at_the_origin_is_the_plain_attention_and_anywhere_else_changes_it():
    """Same weights, one model with RoPE: at all-zero coordinates every angle is 0 and output and
    attention maps equal the plain model's exactly, so the rotation touches q and k and nothing
    else. Anywhere else the attention maps -- what the interpretability tools read -- change."""
    plain = _combined(_plain_cfg()).global_module.eval()
    rope = _combined(_rope_cfg(center_coords=False)).global_module.eval()
    missing, unexpected = rope.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all("rotary_encoding" in key for key in missing)
    emb = torch.randn(6, N_EMBED)

    def run(module, coords):
        gi = module.common_step_local_to_global(_batch(coords), emb, eval_step=True)
        out, _, attn = module(gi.padded_emb, gi.src_padding_mask, gi.attention_mask, rotary=gi.rotary)
        return out, attn

    out_plain, attn_plain = run(plain, torch.zeros(6, 2))
    out_origin, attn_origin = run(rope, torch.zeros(6, 2))
    assert torch.allclose(out_plain, out_origin, atol=1e-6) and torch.allclose(attn_plain, attn_origin, atol=1e-6)

    _, attn_placed = run(rope, torch.rand(6, 2) * 2000)
    assert not torch.allclose(attn_plain, attn_placed, atol=1e-3)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_rope_with_a_bias_on_a_dense_graph_stays_finite_and_blocked_pairs_get_no_attention(hops):
    """The long-range mask's NaN guarantee with both relative encodings on: every cell-to-cell
    pair of a complete graph is blocked, each row keeps only CLS, and the gradient is finite."""
    from interscale.module.global_modules.positional_encodings import build_rotary_encoding

    n = 10
    cfg = _rope_cfg(bias=["distance"], kind="mixed")
    cfg.model.global_component.parameters.pe.distance.max_dist = 2000.0
    rotary = build_rotary_encoding(cfg, 2, N_EMBED)
    module = _hook(cfg, long_range=True, hops=hops, bias=_random_bias(cfg, 2), rotary=rotary).train()
    batch = _batch(torch.rand(n, 2) * 300)
    full = torch.ones(n, n, dtype=torch.bool).nonzero().t()
    batch.edge_index = full[:, full[0] != full[1]]
    emb = torch.randn(n, N_EMBED, requires_grad=True)

    padded, padding, _, mask, rot = module.common_step_local_to_global(batch, emb, eval_step=True)
    out, _, attn = module(padded, padding, mask, register_hook=True, rotary=rot)
    out.sum().backward()

    assert torch.isfinite(out).all() and torch.isfinite(emb.grad).all()
    assert torch.isfinite(rotary.encoders["rope"].direction.grad).all()
    weights = attn[0]
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)
    assert not weights[:, :n, :n].any()


def test_a_valid_rope_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, ["rotary: [rope]", "rope:", "  kind: mixed"]))

    assert list(cfg.model.global_component.parameters.pe.rotary) == ["rope"]


@pytest.mark.parametrize(
    ("lines", "spatial", "overrides", "match"),
    [
        (["rotary: [rope]", "rope:", "  kind: polar"], True, None, "rope.kind"),
        (["rotary: [rope]", "rope:", "  min_wavelength: 0.0"], True, None, "rope.min_wavelength"),
        (["rotary: [rope]", "rope:", "  max_wavelength: 10.0"], True, None, "max_wavelength"),
        (["rotary: [rope]", "rope:", "  max_wavelength: -1.0"], True, None, "max_wavelength"),
        (["rotary: [rope]"], False, None, "dataset.spatial_key"),
        (["node: [rope]"], True, None, "belong under pe.rotary"),
        (["rotary: [naive]"], True, None, "belong under pe.node"),
        (["rotary: [rope, rope]"], True, None, "more than once"),
        (["rotary: [rope]"], True, ["model.global_component.parameters.n_heads", 8], "multiple of 4"),
        (["rotary: [rope]"], True, ["model.n_embed", 24], "multiple of 4"),
    ],
)
def test_invalid_rope_configs_fail_at_load(tmp_path, lines, spatial, overrides, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, lines, spatial=spatial), overrides=overrides)


def test_the_checkpoint_name_and_a_sweep_arm_carry_rope():
    from interscale.config.sweep import apply_sweep_config
    from interscale.tl.utils import get_model_filename_prefix

    assert get_model_filename_prefix(_rope_cfg(), True, True).endswith("pe-rope_")
    assert get_model_filename_prefix(_rope_cfg("mixed"), True, True).endswith("pe-rope-mixed_"), (
        "the two kinds must not share a checkpoint name"
    )
    both = _rope_cfg(node=["naive"], bias=["distance"])
    assert get_model_filename_prefix(both, True, True).endswith("pe-naive+distance+rope_")

    rotary = "model.global_component.parameters.pe.rotary"
    arms = {"none": {rotary: []}, "rope": {rotary: ["rope"]}}
    cfg = _cfg()
    cfg.freeze()
    cfg, _ = apply_sweep_config(cfg, "hyperparmeter", {"arm": "rope"}, sweep_params=["arm"], arms=arms)
    assert list(cfg.model.global_component.parameters.pe.rotary) == ["rope"]


def _derived_rope_cfg(unit=1.0, min_wavelength=50.0):
    cfg = _cfg()
    cfg.model.global_component.parameters.pe.rotary = ["rope"]
    cfg.model.global_component.parameters.pe.rope.min_wavelength = min_wavelength
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.spatial_unit_um = unit
    cfg.freeze()
    return cfg


def test_max_wavelength_zero_is_twice_the_largest_slide_diameter():
    from interscale.module.global_modules.positional_encodings import resolve_rope_range

    cfg = _derived_rope_cfg(unit=2.0)
    resolved = resolve_rope_range(cfg, _slides_adata())

    assert resolved.model.global_component.parameters.pe.rope.max_wavelength == pytest.approx(200.0)
    assert cfg.model.global_component.parameters.pe.rope.max_wavelength == 0.0  # caller's cfg untouched

    fixed = cfg.clone()
    fixed.defrost()
    fixed.model.global_component.parameters.pe.rope.max_wavelength = 700.0
    assert resolve_rope_range(fixed, _slides_adata()) is fixed
    off = _cfg()
    assert resolve_rope_range(off, _slides_adata()) is off

    with pytest.raises(ValueError, match="below pe.rope.min_wavelength"):
        resolve_rope_range(_derived_rope_cfg(min_wavelength=500.0), _slides_adata())


def test_a_module_built_without_the_data_cannot_use_max_wavelength_zero():
    from interscale.module.global_modules.positional_encodings import build_rotary_encoding

    with pytest.raises(ValueError, match="nothing derived it"):
        build_rotary_encoding(_derived_rope_cfg(), 4, 16)


def test_the_model_builds_rope_over_the_derived_range_and_the_checkpoint_keeps_it():
    from interscale.model.combined_model import CombinedModel
    from interscale.module.global_modules.positional_encodings import RotaryPE

    model = CombinedModel(_slides_adata(CombinedModel), cfg=_derived_rope_cfg())
    rope = model.module.global_module.rotary_encoding.encoders["rope"]
    assert rope.wavelengths().max().item() == pytest.approx(100.0, rel=1e-5)  # 2 x the 50 um slide

    reloaded = RotaryPE(4, 4, "axial", 50.0, 9000.0)
    reloaded.load_state_dict(rope.state_dict())
    assert torch.equal(reloaded.wavelengths(), rope.wavelengths()), "the trained frequencies win"


@pytest.mark.parametrize("kind", ["axial", "mixed"])
def test_rope_trains_end_to_end(kind):
    """Through the training plan with the long-range mask on: the loss stays finite, and the mixed
    kind's frequencies move, so the gradient reaches them through every layer."""
    import lightning.pytorch as pl
    from torch_geometric.data import Data

    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.module.global_modules import TransformerNodeEncoderHook
    from interscale.module.global_modules.positional_encodings import build_rotary_encoding
    from interscale.train._trainingplans import TrainingPlan

    def graph(n, seed):
        g = torch.Generator().manual_seed(seed)
        src = list(range(n - 1)) + list(range(1, n))
        d = Data(
            x=torch.randn(n, 6, generator=g),
            edge_index=torch.tensor([src, list(range(1, n)) + list(range(n - 1))], dtype=torch.long),
            pos=torch.rand(n, 2, generator=g) * 2000,
        )
        d.embeddings = torch.randn(n, 8, generator=g)
        return d

    dm = GraphAnnDataModule(
        datas=[[graph(7, 0), graph(9, 1)], [graph(8, 2)], [graph(8, 3)]],
        batch_size=2,
        num_workers=0,
        mask_percentage=0.3,
        mask_strategy="node",
        learning_type="node",
    )
    rotary = build_rotary_encoding(_rope_cfg(kind, local=False), 2, 8)
    initial = rotary.encoders["rope"].direction.detach().clone()
    module = TransformerNodeEncoderHook(
        max_seq_len=16,
        n_heads=2,
        dropout_global=0.0,
        act_func="relu",
        num_layers=2,
        dim_feedforward=8,
        long_range_attention=True,
        local_mask_hops=1,
        rotary_encoding=rotary,
        n_input=6,
        n_output=6,
        n_embed=8,
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
    moved = not torch.equal(rotary.encoders["rope"].direction.detach(), initial)
    assert moved == (kind == "mixed")


# --------------------------------------------------------------------------- spectral bias


def _lattice_edges(side):
    """A side x side 4-neighbour lattice, both directions."""
    idx = torch.arange(side * side).reshape(side, side)
    src = torch.cat([idx[:, :-1].reshape(-1), idx[:-1, :].reshape(-1)])
    dst = torch.cat([idx[:, 1:].reshape(-1), idx[1:, :].reshape(-1)])
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def _random_edges(n, extra, seed):
    """A path through n cells plus `extra` random chords: connected, and no two eigenvalues tie."""
    gen = torch.Generator().manual_seed(seed)
    src = list(range(n - 1)) + torch.randint(0, n, (extra,), generator=gen).tolist()
    dst = list(range(1, n)) + torch.randint(0, n, (extra,), generator=gen).tolist()
    return torch.tensor([src + dst, dst + src])


def _dense_spectrum(edge_index, n):
    from interscale.tl.positional import _adjacency, _normalised

    lap = torch.eye(n, dtype=torch.float64) - torch.from_numpy(_normalised(_adjacency(edge_index, n)).toarray())
    return torch.linalg.eigh(lap)


def _spectral_cfg(k=6, **pe):
    pe.setdefault("bias", ["spectral"])
    cfg = _cfg(**pe)
    cfg.model.global_component.parameters.pe.spectral.k = k
    return cfg


def _spectral_batch(cfg, *sizes, seed=0):
    """Real PyG graphs of the given sizes with the spectral inputs attached, batched by PyG."""
    from torch_geometric.data import Batch, Data

    from interscale.module.global_modules.positional_encodings import attach_positional_inputs

    datas = []
    for i, n in enumerate(sizes):
        gen = torch.Generator().manual_seed(seed + i)
        datas.append(
            Data(edge_index=_random_edges(n, n // 2, seed + i), num_nodes=n, pos=torch.rand(n, 2, generator=gen) * 1000)
        )
    attach_positional_inputs(datas, cfg)
    batch = Batch.from_data_list(datas)
    batch.mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    return batch, datas


def test_laplacian_spectrum_is_the_lowest_non_trivial_eigenpairs():
    from interscale.tl.positional import laplacian_spectrum

    n, k = 40, 6
    edges = _random_edges(n, 15, seed=0)
    vals, vecs = _dense_spectrum(edges, n)
    got_vecs, got_vals = laplacian_spectrum(edges, n, k)

    assert torch.allclose(got_vals.double(), vals[1 : k + 1], atol=1e-6)
    # Compared as projectors: the comparison must not depend on the solver's signs.
    expected = vecs[:, 1 : k + 1] @ vecs[:, 1 : k + 1].T
    assert torch.allclose(got_vecs.double() @ got_vecs.double().T, expected, atol=1e-5)

    sparse_vecs, sparse_vals = laplacian_spectrum(edges, n, k, dense_max_nodes=10)  # shift-invert eigsh
    assert torch.allclose(sparse_vals, got_vals, atol=1e-6)
    assert torch.allclose(sparse_vecs @ sparse_vecs.T, got_vecs @ got_vecs.T, atol=1e-5)

    tiny_vecs, tiny_vals = laplacian_spectrum(_random_edges(4, 0, seed=1), 4, k)  # 3 non-trivial
    assert not tiny_vecs[:, 3:].any() and not tiny_vals[3:].any(), "zero-padded beyond what the graph has"


def test_a_tied_eigenspace_at_the_cut_is_dropped_whole():
    """On a square lattice modes come in tied pairs (an x and a y mode). Cutting a pair would keep
    one direction at the solver's whim; it is dropped instead, so the kernel built from what is kept
    does not depend on how the graph's cells happen to be numbered."""
    from interscale.tl.positional import laplacian_spectrum

    side = 6
    n = side * side
    edges = _lattice_edges(side)
    vals = _dense_spectrum(edges, n)[0][1:]
    # Whole eigenspaces end where the next eigenvalue differs.
    ends = [i + 1 for i in range(len(vals) - 1) if vals[i + 1] - vals[i] > 1e-6 * vals[i + 1]]
    assert ends[:4] == [2, 3, 5, 7], "the lattice's tied pairs, as on synth_spot's"

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    inverse = torch.argsort(perm)
    renumbered = inverse[edges]  # cell perm[i] of the original is cell i of the renumbered graph
    for k in range(1, 12):
        v, lam = laplacian_spectrum(edges, n, k)
        kept = int((lam > 0).sum())
        assert kept == max([e for e in ends if e <= k], default=0), k
        kernel = (v * torch.exp(-50 * lam)) @ v.T
        v2, lam2 = laplacian_spectrum(renumbered, n, k)
        kernel2 = ((v2 * torch.exp(-50 * lam2)) @ v2.T)[inverse][:, inverse]
        assert torch.allclose(kernel, kernel2, atol=1e-5), f"k={k}: the kernel depends on the numbering"


def test_the_spectral_filter_is_piecewise_linear_in_log_eigenvalue_and_flat_beyond():
    from interscale.module.global_modules.positional_encodings import SpectralBias

    bias = SpectralBias(n_heads=2, num_knots=3, min_eigval=1e-4, max_eigval=1e-2)  # knots 1e-4, 1e-3, 1e-2
    assert not bias.table.any(), "starts at zero"
    with torch.no_grad():
        bias.table.copy_(torch.tensor([[0.0, 1.0, 3.0], [2.0, 2.0, -1.0]]))

    lam = torch.tensor([1e-4, 10**-3.5, 1e-3, 1e-2, 1e-6, 1.0, 0.0])
    out = bias.filter(lam)

    expected = torch.tensor([[0.0, 2.0], [0.5, 2.0], [1.0, 2.0], [3.0, -1.0], [0.0, 2.0], [3.0, -1.0], [0.0, 2.0]])
    assert torch.allclose(out, expected, atol=1e-5)
    knots, values = bias.profile()
    assert torch.allclose(knots, torch.tensor([1e-4, 1e-3, 1e-2]), rtol=1e-5)
    assert torch.equal(values, bias.table.detach())


def test_the_spectral_bias_is_the_filtered_projector_and_free_of_signs_and_bases():
    from interscale.module.global_modules.positional_encodings import SpectralBias

    gen = torch.Generator().manual_seed(0)
    bias = SpectralBias(n_heads=2, num_knots=8, min_eigval=1e-4, max_eigval=1.0)
    with torch.no_grad():
        bias.table.copy_(torch.randn(bias.table.shape, generator=gen))
    k = 4
    vecs = torch.randn(1, 7, k, generator=gen)
    vals = torch.tensor([[2e-3, 1e-2, 1e-2, 5e-2]])  # columns 1 and 2 share an eigenvalue

    out = bias(vecs, vals)

    h = bias.filter(vals)[0]  # [k, heads]
    for head in range(2):
        expected = sum(h[i, head] * torch.outer(vecs[0, :, i], vecs[0, :, i]) for i in range(k))
        assert torch.allclose(out[0, head], expected, atol=1e-5)

    flipped = vecs * torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert torch.allclose(bias(flipped, vals), out, atol=1e-5), "signs"

    def turn(v, i, j, angle):
        c, s_ = math.cos(angle), math.sin(angle)
        v = v.clone()
        v[..., i], v[..., j] = c * vecs[..., i] - s_ * vecs[..., j], s_ * vecs[..., i] + c * vecs[..., j]
        return v

    assert torch.allclose(bias(turn(vecs, 1, 2, 0.7), vals), out, atol=1e-5), "basis within a tied pair"
    assert not torch.allclose(bias(turn(vecs, 0, 1, 0.7), vals), out, atol=1e-3), "control: untied columns"


def test_the_spectral_inputs_are_scaled_per_cell_and_batched_per_graph():
    """sqrt(N)-scaled eigenvectors: at a flat filter of 1 each kept mode adds 1 to the bias's mean
    diagonal, whatever the graph's size -- here all k of them."""
    from interscale.module.global_modules.positional_encodings import SpectralBias

    k = 5
    batch, datas = _spectral_batch(_spectral_cfg(k), 30, 60)
    for d in datas:
        assert d.spectral_pe.shape == (d.num_nodes, k) and d.spectral_eigval.shape == (1, k)
        norms = d.spectral_pe.norm(dim=0)
        assert torch.allclose(norms, torch.full((k,), math.sqrt(d.num_nodes)), rtol=1e-4)
    assert batch.spectral_eigval.shape == (2, k)

    flat = SpectralBias(n_heads=1, num_knots=2, min_eigval=1e-6, max_eigval=2.0)
    with torch.no_grad():
        flat.table.fill_(1.0)
    for d in datas:
        diag = flat(d.spectral_pe.unsqueeze(0), d.spectral_eigval)[0, 0].diagonal()
        assert diag.mean().item() == pytest.approx(k, rel=1e-4)


def test_the_spectral_bias_is_laid_out_like_the_tokens():
    """Entry (b * heads + h, i, j): head h's bias between the cells at tokens i and j of graph b,
    with each graph's own eigenvalues. Graph 0 is subsampled (30 cells, 8 tokens), graph 1 padded."""
    import random

    from interscale.module.global_modules.positional_encodings import build_attention_bias

    n_heads, seq_len = 3, 8
    cfg = _spectral_cfg(5)
    bias_module = build_attention_bias(cfg, n_heads)
    with torch.no_grad():
        bias_module.encoders["spectral"].table.copy_(
            torch.randn(n_heads, 32, generator=torch.Generator().manual_seed(3))
        )
    batch, datas = _spectral_batch(cfg, 30, 6)
    random.seed(0)
    index_nodes = [sorted(random.sample(range(30), seq_len)), list(range(6))]
    padding = torch.tensor([[False] * seq_len, [True, True] + [False] * 6])

    out = bias_module(batch, index_nodes, padding, torch.float32, torch.device("cpu"))

    assert out.shape == (2 * n_heads, seq_len + 1, seq_len + 1)
    spectral = bias_module.encoders["spectral"]
    for g, (d, kept) in enumerate(zip(datas, index_nodes, strict=True)):
        expected = spectral(d.spectral_pe[kept].unsqueeze(0), d.spectral_eigval)[0]  # [heads, kept, kept]
        lo = seq_len - len(kept)
        for h in range(n_heads):
            block = out[g * n_heads + h]
            assert torch.allclose(block[lo:seq_len, lo:seq_len], expected[h], atol=1e-5), (g, h)
            assert not block[:lo].any() and not block[:, :lo].any(), "padding carries no bias"
            assert not block[seq_len].any() and not block[:, seq_len].any(), "CLS carries no bias"


def test_spectral_and_distance_biases_add_up():
    from interscale.module.global_modules.positional_encodings import build_attention_bias

    cfg = _spectral_cfg(5, bias=["distance", "spectral"])
    cfg.model.global_component.parameters.pe.distance.max_dist = 2000.0
    both = build_attention_bias(cfg, 2)
    gen = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for param in both.parameters():
            param.copy_(torch.randn(param.shape, generator=gen))
    batch, _ = _spectral_batch(cfg, 9)
    padding = torch.zeros(1, 9, dtype=torch.bool)
    out = both(batch, [list(range(9))], padding, torch.float32, torch.device("cpu"))

    parts = []
    for name in ("distance", "spectral"):
        only = cfg.clone()
        only.model.global_component.parameters.pe.bias = [name]
        single = build_attention_bias(only, 2)
        single.encoders[name].load_state_dict(both.encoders[name].state_dict())
        parts.append(single(batch, [list(range(9))], padding, torch.float32, torch.device("cpu")))
    assert torch.allclose(out, parts[0] + parts[1], atol=1e-5)


def test_a_zero_spectral_bias_changes_nothing():
    from interscale.module.global_modules.positional_encodings import build_attention_bias

    cfg = _spectral_cfg(5)
    torch.manual_seed(0)
    plain = _hook(cfg, long_range=True, hops=1).eval()
    torch.manual_seed(0)
    biased = _hook(cfg, long_range=True, hops=1, bias=build_attention_bias(cfg, 2)).eval()
    batch, _ = _spectral_batch(cfg, 12)
    emb = torch.randn(12, N_EMBED)

    outs = []
    for module in (plain, biased):
        padded, padding, _, mask, _ = module.common_step_local_to_global(batch, emb, eval_step=True)
        outs.append(module(padded, padding, mask, register_hook=False)[0])
    assert torch.allclose(outs[0], outs[1], atol=1e-6)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_a_spectrally_biased_dense_graph_stays_finite_and_blocked_pairs_get_no_attention(hops):
    """The NaN guarantee with both biases on. On a complete graph every non-trivial eigenvalue ties,
    so the cut drops them all and the spectral bias is zero -- still a valid path to run."""
    n = 10
    cfg = _spectral_cfg(4, bias=["distance", "spectral"])
    cfg.model.global_component.parameters.pe.distance.max_dist = 2000.0
    module = _hook(cfg, long_range=True, hops=hops, bias=_random_bias(cfg, 2)).train()
    batch, _ = _spectral_batch(cfg, n)
    full = torch.ones(n, n, dtype=torch.bool).nonzero().t()
    batch.edge_index = full[:, full[0] != full[1]]
    from interscale.module.global_modules.positional_encodings import attach_positional_inputs

    attach_positional_inputs([batch], cfg)  # recompute on the complete graph
    assert not batch.spectral_pe.any()
    emb = torch.randn(n, N_EMBED, requires_grad=True)

    padded, padding, _, mask, _ = module.common_step_local_to_global(batch, emb, eval_step=True)
    out, _, attn = module(padded, padding, mask, register_hook=True)
    out.sum().backward()

    assert torch.isfinite(out).all() and torch.isfinite(emb.grad).all()
    weights = attn[0]
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)
    assert not weights[:, :n, :n].any()


def test_spectral_configs_load_without_coordinates(tmp_path):
    cfg = load_config(
        _write(tmp_path, ["bias: [spectral]", "spectral:", "  k: 16", "  min_eigval: 1.0e-6"], spatial=False)
    )

    assert list(cfg.model.global_component.parameters.pe.bias) == ["spectral"]
    assert cfg.model.global_component.parameters.pe.spectral.min_eigval == 1e-6


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (["bias: [spectral]", "spectral:", "  k: 0"], "spectral.k"),
        (["bias: [spectral]", "spectral:", "  num_knots: 1"], "num_knots"),
        (["bias: [spectral]", "spectral:", "  min_eigval: 0.0"], "min_eigval"),
        (["bias: [spectral]", "spectral:", "  max_eigval: 1.0e-6"], "max_eigval"),
        (["node: [spectral]"], "belong under pe.bias"),
    ],
)
def test_invalid_spectral_configs_fail_at_load(tmp_path, lines, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, lines))


def test_the_checkpoint_name_and_the_reserved_fields_carry_the_spectral_bias():
    from interscale.tl.geome_utils import RESERVED_FIELD_NAMES
    from interscale.tl.utils import get_model_filename_prefix

    assert get_model_filename_prefix(_spectral_cfg(), True, True).endswith("pe-spectral_")
    both = _spectral_cfg(bias=["distance", "spectral"])
    assert get_model_filename_prefix(both, True, True).endswith("pe-distance+spectral_")
    assert {"spectral_pe", "spectral_eigval"} <= RESERVED_FIELD_NAMES


def test_the_run_reports_the_eigenvalues_against_the_knots(caplog):
    from interscale.module.global_modules.positional_encodings import report_spectral_range

    cfg = _spectral_cfg(5)
    _, datas = _spectral_batch(cfg, 30, 4)  # the 4-cell graph has only 3 non-trivial eigenpairs
    found = torch.cat([d.spectral_eigval.reshape(-1) for d in datas])
    found = found[found > 0]
    with caplog.at_level("INFO"):
        lo, hi = report_spectral_range(datas, cfg)
    assert (lo, hi) == (found.min().item(), found.max().item())
    assert hi == pytest.approx(2.0), "the 4-cell graph is a 4-cycle: bipartite, so its top eigenvalue is 2"
    assert "1 graph(s) keep fewer than k = 5" in caplog.text
    assert not [r for r in caplog.records if r.levelname == "WARNING"]

    narrow = cfg.clone()
    narrow.model.global_component.parameters.pe.spectral.min_eigval = hi / 2
    narrow.model.global_component.parameters.pe.spectral.max_eigval = hi * 2
    caplog.clear()
    with caplog.at_level("INFO"):
        report_spectral_range(datas, narrow)
    assert "reach beyond the filter's knots" in caplog.text
    assert report_spectral_range(datas, _cfg()) is None


def test_the_run_warns_when_the_kept_modes_miss_part_of_a_fragmented_graph(caplog):
    """A 20-cell path beside a separate 6-cell one: the 2 lowest modes both live on the long path
    (eigenvalues ~0.014 and ~0.05 against the short one's ~0.19), so 6 of 26 cells get no mode."""
    from torch_geometric.data import Data

    from interscale.module.global_modules.positional_encodings import attach_positional_inputs, report_spectral_range

    cfg = _spectral_cfg(2)
    path = list(range(19)) + list(range(20, 25))
    edges = torch.tensor([path + [p + 1 for p in path], [p + 1 for p in path] + path])
    data = Data(edge_index=edges, num_nodes=26)
    attach_positional_inputs([data], cfg)
    assert not data.spectral_pe[20:].abs().amax() > 1e-4, "the short path is in no kept mode"

    with caplog.at_level("INFO"):
        report_spectral_range([data], cfg)
    assert "23.1% of cells are in no kept mode" in caplog.text
    assert "fragmented" in caplog.text


def test_the_spectral_bias_trains_end_to_end():
    """Through the training plan with the long-range mask on: the zero table moves."""
    import lightning.pytorch as pl
    from torch_geometric.data import Data

    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.module.global_modules import TransformerNodeEncoderHook
    from interscale.module.global_modules.positional_encodings import attach_positional_inputs, build_attention_bias
    from interscale.train._trainingplans import TrainingPlan

    cfg = _spectral_cfg(4, local=False)

    def graph(n, seed):
        g = torch.Generator().manual_seed(seed)
        d = Data(x=torch.randn(n, 6, generator=g), edge_index=_random_edges(n, n // 2, seed), num_nodes=n)
        d.embeddings = torch.randn(n, 4, generator=g)
        return d

    splits = [[graph(12, 0), graph(14, 1)], [graph(13, 2)], [graph(13, 3)]]
    for split in splits:
        attach_positional_inputs(split, cfg)
    dm = GraphAnnDataModule(
        datas=splits, batch_size=2, num_workers=0, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    bias = build_attention_bias(cfg, 2)
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
    assert bias.encoders["spectral"].table.any(), "the gradient never reached the filter"
