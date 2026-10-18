"""Where a patch draws its predictions from, against the expression-permuted slide (``evaluation.attention_flow``).

Pinned here: patches are numbered on each graph's own grid; the permutation moves whole profiles
(expression and label) among the cells with a visible entry and leaves masks and positions; uniform
attention, which reads no content, is its own null exactly and gives wholly hidden cells nothing; a
fixed linear map's influence matches its exact squared Jacobian and stays at the null's noise floor;
a map gated by what a cell expresses draws from where that expression is, far above the null; label
flows sum to 1; draws add up; no generator is touched and the module is restored; the figures draw.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation.attention_flow import (
    _permuted,
    _scored,
    patch_flow,
    patch_index,
    plot_flow_field,
    plot_label_flow,
    plot_source_maps,
    regular_sources,
)
from interscale.tl.masking import MASK_VALUE

G, E = 4, 8
UNIT = 10.0  # µm per lattice step


def _graph(side=8, seed=0, hidden=None):
    gen = torch.Generator().manual_seed(seed)
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    n = side * side
    data = Data(x=torch.rand(n, G, generator=gen) + 0.5, edge_index=(torch.cdist(pos, pos) == 1).nonzero().t(), pos=pos)
    data.mask = (pos - pos.mean(0)).norm(dim=1) <= 1.6 if hidden is None else hidden
    data.gene_mask = data.mask[:, None].expand(-1, G).clone()
    data.half = torch.nn.functional.one_hot((pos[:, 0] >= side / 2).long(), 2).float()
    data.obs_names = torch.arange(n, dtype=torch.float32) + 1
    return data


def _flow(module, draws, **kwargs):
    options = {
        "prediction_task": "regression",
        "prediction_level": "node",
        "mask_strategy": "node",
        "device": torch.device("cpu"),
        "spatial_unit_um": UNIT,
        "patch_um": 20.0,
        "min_queries": 1,
    }
    return patch_flow(module, draws, **(options | kwargs))


def _summary(found, kind):
    table = found["summary"]
    return table[table["kind"] == kind].set_index("metric")["value"]


def test_patches_are_numbered_on_each_graphs_own_grid():
    pos = np.array([[100.0, 50.0], [119.0, 50.0], [121.0, 75.0], [100.0, 89.0]])
    patch, centres = patch_index(pos, 20.0)
    assert patch.tolist() == [0, 0, 3, 1]
    assert np.allclose(centres, [[110, 60], [110, 80], [130, 60], [130, 80]])


def test_the_permutation_moves_whole_profiles_among_open_cells_and_leaves_the_masks():
    data = _graph()
    batch = Batch.from_data_list([data])
    scored = _scored(batch, "node")
    copy, perm = _permuted(batch, scored, torch.Generator().manual_seed(0), "half")
    hidden = data.mask
    assert torch.equal(perm[hidden], torch.where(hidden)[0]), "a hidden cell's row moved"
    assert set(perm[~hidden].tolist()) == set(torch.where(~hidden)[0].tolist())
    assert (perm[~hidden] != torch.where(~hidden)[0]).any()
    assert torch.equal(copy.x, batch.x[perm]) and torch.equal(copy.half, batch.half[perm])
    assert torch.equal(copy.mask, batch.mask) and torch.equal(copy.pos, batch.pos)


# --------------------------------------------------------------------------- rollout


def _global_module(uniform=False):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 128
    cfg.model.global_component.parameters.n_heads = 2
    cfg.dataset.spatial_key = "spatial"
    torch.manual_seed(0)
    common = {"n_input": G, "n_output": G, "n_embed": E, "decoder_type": "linear", "dropout_decoder": 0.0}
    common |= {"decoder_hidden_dims": [8], "mask_percentage": 0.0, "mask_strategy": "node", "block_masking": True}
    module = GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **common)
    if uniform:
        with torch.no_grad():
            for layer in module.transformer_encoder.layers:
                inner = layer.self_attn.in_proj_weight.shape[0] // 3
                layer.self_attn.in_proj_weight[: 2 * inner] = 0.0
                layer.self_attn.in_proj_bias[: 2 * inner] = 0.0
    return module


def test_uniform_attention_is_its_own_null_and_gives_hidden_cells_nothing():
    datas = [_graph(8, 0), _graph(7, 1)]
    module = _global_module(uniform=True).train()  # the builder reseeds on purpose
    state = torch.get_rng_state()
    found = _flow(module, [datas], kinds=("rollout",), permutations=3, label_key="half")
    assert torch.equal(state, torch.get_rng_state()) and module.training
    patches = found["patches"]
    assert np.allclose(patches["observed"], patches["null_mean"])
    assert np.allclose(patches["null_std"].fillna(0.0), 0.0)
    summary = _summary(found, "rollout")
    assert summary["kl_null"] == pytest.approx(0.0, abs=1e-9) and summary["shift_um"] == pytest.approx(0.0, abs=1e-6)
    # Shares sum to 1 per source, and a patch of wholly hidden cells is no source of anything.
    totals = patches.groupby(["graph", "source"])["observed"].sum()
    assert np.allclose(totals, 1.0)
    for g, data in enumerate(datas):
        patch, _ = patch_index(data.pos.numpy() * UNIT, 20.0)
        all_hidden = [p for p in np.unique(patch) if data.mask.numpy()[patch == p].all()]
        targets = patches[(patches["graph"] == g)]
        assert not targets[targets["target"].isin(all_hidden)]["observed"].gt(0).any()
    labels = found["labels"]
    assert np.allclose(labels.groupby(["kind", "query_label"])["observed"].sum(), 1.0)


# --------------------------------------------------------------------------- influence


class _Planted(torch.nn.Module):
    """``y_i = sum_j A_ij x_j[0]`` (``gate``: ``x_j[0] * x_j[1]``) in every gene, on the masked input."""

    registered_global_component = False

    def __init__(self, weights, gate=False):
        super().__init__()
        self.weights, self.gate = weights, gate

    def _common_step(self, batch, prediction_task, prediction_level):
        x = batch.x.masked_fill(batch.mask.bool()[:, None], MASK_VALUE)
        drive = x[:, 0] * x[:, 1] if self.gate else x[:, 0]
        y = (self.weights @ drive)[:, None].expand(-1, G)
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=None, view=SimpleNamespace(padded_node_idx=None))


def test_a_fixed_maps_influence_is_its_squared_jacobian_and_stays_at_the_null_floor():
    data = _graph(8, 0)
    n = data.num_nodes
    dist = torch.cdist(data.pos, data.pos)
    weights = torch.exp(-dist / 2.0)
    found = _flow(_Planted(weights), [[data]], kinds=("influence",), permutations=4, n_projections=400)
    patches = found["patches"]
    patch, _ = patch_index(data.pos.numpy() * UNIT, 20.0)
    hidden = data.mask.numpy()
    exact = (weights.numpy() ** 2) * ~hidden[None, :]  # through visible cells only
    for source, sub in patches.groupby("source"):
        rows = np.flatnonzero(hidden & (patch == source))
        per_target = np.bincount(patch, weights=exact[rows].sum(0), minlength=patch.max() + 1)
        expected = per_target / per_target.sum()
        found_share = sub.set_index("target")["observed"].reindex(range(len(expected))).fillna(0.0).to_numpy()
        assert np.allclose(found_share, expected, atol=0.03)
    summary = _summary(found, "influence")
    # No content in the map: the observed flow is a permutation like any other.
    assert summary["kl_null"] < 1.5 * summary["kl_null_floor"] + 1e-3
    assert summary["shift_um"] < 1.5 * summary["shift_floor_um"] + 0.1
    assert summary["above_null"] < 0.15
    assert n == 64


def test_a_map_gated_by_what_cells_express_draws_from_where_it_is_expressed():
    data = _graph(16, 0)
    patch, _ = patch_index(data.pos.numpy() * UNIT, 20.0)
    corner = (data.pos[:, 0] >= 8) & (data.pos[:, 1] >= 8)  # a quadrant: 16 patches, 64 cells
    targets = np.unique(patch[corner.numpy()])
    x = torch.full_like(data.x, 1.0)
    x[:, 1] = 0.1
    x[corner, 1] = 5.0  # what the keys of one quadrant express
    data.x = x
    weights = torch.ones(256, 256) - torch.eye(256)
    found = _flow(_Planted(weights, gate=True), [[data]], kinds=("influence",), permutations=6, n_projections=50)
    patches = found["patches"].set_index(["source", "target"])
    drawn = (
        patches["observed"].groupby("source").apply(lambda v: v[v.index.get_level_values("target").isin(targets)].sum())
    )
    null = (
        patches["null_mean"]
        .groupby("source")
        .apply(lambda v: v[v.index.get_level_values("target").isin(targets)].sum())
    )
    assert (drawn > 0.85).all() and (null < 0.6).all()  # cold keys keep a little: x_j[1] stays in the gate
    summary = _summary(found, "influence")
    # The observed arrangement is far from every random one: the centre of mass moves several times
    # further than permutations move it among themselves (measured: 6x, KL 2.4x, 0.70 above the null).
    assert summary["shift_um"] > 4 * summary["shift_floor_um"]
    assert summary["kl_null"] > 2 * summary["kl_null_floor"]
    assert summary["above_null"] > 0.5


def test_draws_add_up_and_labels_follow_the_queries():
    first = _graph(8, 0)
    second = _graph(8, 0, hidden=(torch.arange(64) % 7 == 0))
    weights = torch.exp(-torch.cdist(first.pos, first.pos) / 2.0)
    one = _flow(_Planted(weights), [[first]], kinds=("influence",), permutations=2, label_key="half")
    both = _flow(_Planted(weights), [[first], [second]], kinds=("influence",), permutations=2, label_key="half")
    queries = lambda found: found["patches"].groupby("source")["n_queries"].first().sum()
    assert queries(both) == queries(one) + int(second.mask.sum())
    labels = both["labels"]
    assert np.allclose(labels.groupby("query_label")["observed"].sum(), 1.0)
    with pytest.raises(ValueError, match="same graphs"):
        _flow(_Planted(weights), [[first], [first, second]], kinds=("influence",))


def test_the_figures_draw():
    import matplotlib

    matplotlib.use("Agg")
    datas = [_graph(8, 0)]
    found = _flow(_global_module(), [datas], kinds=("rollout", "influence"), permutations=2, label_key="half")
    sources = regular_sources(found["patches"], found["centres"], 0, "influence", n=4)
    assert len(sources) == len(set(sources)) >= 2
    figures = [
        plot_flow_field(found["patches"], found["centres"], graph=0, kind="influence", patch_um=20.0),
        plot_source_maps(found["patches"], found["centres"], graph=0, kind="rollout", patch_um=20.0, sources=sources),
        plot_label_flow(found["labels"], kind="influence", names=["left", "right"]),
    ]
    assert [len(fig.axes) >= 2 for fig in figures] == [True, True, True]
    assert isinstance(found["centres"], pd.DataFrame)


class _Programme(torch.nn.Module):
    """Gene 3 hidden in every cell, predicted as ``y_i = sum_j A_ij x_j[0]`` from the visible gene 0."""

    registered_global_component = False

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def _common_step(self, batch, prediction_task, prediction_level):
        x = batch.x.masked_fill(batch.gene_mask.bool(), MASK_VALUE)
        y = torch.zeros_like(batch.x)
        y[:, 3] = self.weights @ x[:, 0]
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=None, view=SimpleNamespace(padded_node_idx=None))


def test_a_querys_own_input_is_left_out_without_biasing_its_patch_mates():
    data = _graph(8, 0)
    data.gene_mask = torch.zeros(64, G, dtype=torch.bool)
    data.gene_mask[:, 3] = True  # every cell a query, its own genes 0-2 visible
    data.mask = data.gene_mask.any(1)
    dist = torch.cdist(data.pos, data.pos)
    weights = torch.exp(-dist / 1.5) + 20.0 * torch.eye(64)  # a large pull on the cell itself
    found = _flow(
        _Programme(weights), [[data]], kinds=("influence",), permutations=2, n_projections=1500, mask_strategy="gene"
    )
    patch, _ = patch_index(data.pos.numpy() * UNIT, 20.0)
    exact = weights.numpy() ** 2
    np.fill_diagonal(exact, 0.0)
    for source, sub in found["patches"].groupby("source"):
        rows = np.flatnonzero(patch == source)
        per_target = np.bincount(patch, weights=exact[rows].sum(0), minlength=patch.max() + 1)
        expected = per_target / per_target.sum()
        share = sub.set_index("target")["observed"].reindex(range(len(expected))).fillna(0.0).to_numpy()
        assert np.allclose(share, expected, atol=0.03), f"source {source}"


def test_rollout_leaves_out_a_querys_own_token_too():
    data = _graph(8, 0)
    data.gene_mask = torch.zeros(64, G, dtype=torch.bool)
    data.gene_mask[:, 3] = True
    data.mask = data.gene_mask.any(1)
    module = _global_module(uniform=True)
    module.mask_strategy = "gene"
    found = _flow(module, [[data]], kinds=("rollout",), permutations=2, mask_strategy="gene")
    patch, _ = patch_index(data.pos.numpy() * UNIT, 20.0)
    counts = np.bincount(patch)
    own = found["patches"][found["patches"]["source"] == found["patches"]["target"]].set_index("source")["observed"]
    # Uniform attention spreads evenly over the other cells: a patch's own share is its other cells' share.
    expected = (counts[own.index] - 1) / 63
    assert np.allclose(own.to_numpy(), expected, rtol=0.25)


class _Leaky(torch.nn.Module):
    """``y_i = sum_j A_ij x_j[0]``, read from the *unmasked* input: a hidden gene 0 would reach it."""

    registered_global_component = False

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def _common_step(self, batch, prediction_task, prediction_level):
        y = (self.weights @ batch.x[:, 0])[:, None].expand(-1, G)
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=None, view=SimpleNamespace(padded_node_idx=None))


def test_a_cell_counts_through_its_visible_entries_only():
    data = _graph(8, 0)
    hidden_gene = torch.zeros(64, G, dtype=torch.bool)
    hidden_gene[:, 0] = True  # gene 0 hidden in every cell; the head reads it anyway
    hidden_gene[data.mask] = True  # and the centre's cells wholly, as queries
    data.gene_mask = hidden_gene
    weights = torch.ones(64, 64) - torch.eye(64)
    found = _flow(_Leaky(weights), [[data]], kinds=("influence",), permutations=2, mask_strategy="gene")
    # Nothing reaches a prediction through a visible entry: no source draws from anywhere.
    assert found["patches"].empty


class _Maps(torch.nn.Module):
    """A transformer stand-in whose two layers' maps are given: token k attends to token k + 1 only."""

    def __init__(self, n):
        super().__init__()
        shift = torch.zeros(n + 1, n + 1)
        for k in range(n):
            shift[k, (k + 1) % n] = 1.0
        shift[n, n] = 1.0  # CLS on itself
        self.maps = shift.expand(2, 1, n + 1, n + 1).clone()
        self.n = n
        self._record = None

    from contextlib import contextmanager

    @contextmanager
    def recording_attention(self, reference=False):
        record = []
        self._record = record
        try:
            yield record
        finally:
            self._record = None

    def _common_step(self, batch, prediction_task, prediction_level):
        padding = torch.zeros(1, self.n, dtype=torch.bool)
        self._record.append({"maps": self.maps, "src_padding_mask": padding, "index_nodes": [torch.arange(self.n)]})
        return None


def test_rollout_composes_the_layers_with_the_residual_path():
    from interscale.evaluation.attention_flow import _rollout

    batch = Batch.from_data_list([_graph(4, 0)])  # 16 cells
    flow = _rollout(_Maps(16), batch, "regression", "node")
    # ((A + I) / 2)^2 = (A^2 + 2A + I) / 4: a quarter on itself, a half one step on, a quarter two steps on.
    assert flow[3, 3] == pytest.approx(0.25) and flow[3, 4] == pytest.approx(0.5) and flow[3, 5] == pytest.approx(0.25)
    assert torch.allclose(flow.sum(1), torch.ones(16, dtype=torch.float64))
