"""Programme masking: hide a gene set (a signalling programme's responses) and ask what predicts it.

Pinned here:

* masking -- selectors resolve to ``var`` columns and refuse what matches nothing; the datamodule
  hides exactly the set, in the selected cells (all, a share, or the discs) and nothing else; the
  config and the checkpoint name know it; through a real model no hidden value reaches a prediction;
* fitted references -- ``ref_own`` reads within-cell co-expression, ``ref_own+context`` a field of
  the surrounding cells, neither reads a hidden value, genes never hidden fall back to their mean,
  and through ``collect_head_predictions`` every reference row is its own cell;
* response influence -- a planted sender ranks 1 among cells as far from its receiver, a distance
  kernel 0.5, a cell counts through its visible entries only, and no generator is touched.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation.masked_probing import (
    FittedReferences,
    collect_head_predictions,
    draw_masks,
    position_features,
    score_heads,
)
from interscale.evaluation.receptive_field import response_influence
from interscale.geome_dataloader import GraphAnnDataModule, gene_set_mask_kwargs
from interscale.tl.masking import MASK_VALUE, resolve_mask_genes, sample_gene_set_mask

G, UNIT = 6, 10.0


def _lattice(side, seed=0, x=None):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    if x is None:
        x = torch.rand(n, G, generator=gen) * 3.0
    edges = (torch.cdist(pos, pos) == 1).nonzero().t()
    data = Data(x=x, edge_index=edges, pos=pos)
    data.obs_names = torch.arange(n, dtype=torch.float32) + 1
    return data


# --------------------------------------------------------------------------- masking


def test_selectors_resolve_to_var_columns_and_refuse_what_matches_nothing():
    var = pd.DataFrame({"role": ["sender", "response", "receptor", "response", "noise"]}, index=list("abcde"))
    assert resolve_mask_genes(["role=response"], var) == [1, 3]
    assert resolve_mask_genes(["role=response", "c", "a"], var) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="matches no gene"):
        resolve_mask_genes(["role=resp"], var)
    with pytest.raises(ValueError, match="no column"):
        resolve_mask_genes(["kind=response"], var)
    with pytest.raises(ValueError, match="matches no gene"):
        resolve_mask_genes(["zz"], var)


def test_the_gene_set_is_hidden_in_the_selected_cells_and_nothing_else():
    every = sample_gene_set_mask(50, G, [1, 4], 1.0)
    assert every[:, [1, 4]].all() and not every[:, [0, 2, 3, 5]].any()
    some = sample_gene_set_mask(400, G, [2], 0.25, generator=torch.Generator().manual_seed(0))
    assert 0.18 < some[:, 2].float().mean() < 0.32 and not some[:, [0, 1, 3, 4, 5]].any()
    blocks = torch.zeros(30, dtype=torch.bool)
    blocks[:5] = True
    only_blocks = sample_gene_set_mask(30, G, [0], 0.0, blocks=blocks)
    assert torch.equal(only_blocks[:, 0], blocks) and not only_blocks[:, 1:].any()
    assert sample_gene_set_mask(30, G, [0], 0.0).any(), "every graph keeps a target"


def test_the_datamodule_draws_the_gene_set_redraws_it_and_refuses_node_masking():
    datas = [[_lattice(6, s) for s in range(3)], [_lattice(5, 9)]]
    dm = GraphAnnDataModule(
        datas=datas,
        num_workers=0,
        mask_strategy="gene",
        mask_percentage=0.5,
        mask_gene_columns=[1, 3],
        learning_type="node",
    )
    dm.setup(stage="fit")
    first = [d.gene_mask.clone() for d in dm.train_data]
    for d in dm.train_data:
        assert not d.gene_mask[:, [0, 2, 4, 5]].any()
        assert torch.equal(d.gene_mask[:, 1], d.gene_mask[:, 3]), "a selected cell hides the whole set"
        assert torch.equal(d.mask, d.gene_mask.any(1))
    dm.resample_train_mask()
    assert any(not torch.equal(a, d.gene_mask) for a, d in zip(first, dm.train_data, strict=True))
    discs = GraphAnnDataModule(
        datas=datas,
        num_workers=0,
        mask_strategy="gene",
        mask_percentage=0.0,
        mask_gene_columns=[2],
        mask_block_radius=1.5,
        mask_block_fraction=0.3,
        learning_type="node",
    )
    discs.setup(stage="fit")
    for d in discs.train_data:
        assert d.gene_mask[:, 2].any() and not d.gene_mask[:, [0, 1, 3, 4, 5]].any(), "discs pick cells, not rows"
    with pytest.raises(ValueError, match="mask_strategy 'gene'"):
        GraphAnnDataModule(datas=datas, num_workers=0, mask_strategy="node", mask_gene_columns=[1])


def _cfg(**dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.spatial_key = "spatial"
    for key, value in dataset.items():
        setattr(cfg.dataset, key, value)
    return cfg


def test_the_config_and_the_checkpoint_name_know_the_gene_set():
    from interscale.config import _validate_block_mask, _validate_masking
    from interscale.tl.utils import _training_tag

    cfg = _cfg(mask_strategy="gene", mask_percentage=1.0, mask_genes=["role=response", "role=receiver"])
    _validate_masking(cfg)
    assert _training_tag(cfg) == "var-gs1-role-response-role-receiver_"
    with pytest.raises(ValueError, match="needs mask_strategy 'gene'"):
        _validate_masking(_cfg(mask_strategy="node", mask_percentage=0.5, mask_genes=["role=response"]))
    discs = _cfg(
        mask_strategy="gene",
        mask_percentage=0.0,
        mask_genes=["MC1"],
        mask_block_radius_um=500.0,
        mask_block_fraction=0.3,
    )
    _validate_masking(discs)
    _validate_block_mask(discs)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        _validate_masking(_cfg(mask_strategy="gene", mask_percentage=0.0, mask_genes=["MC1"]))
    var = pd.DataFrame({"role": ["sender", "response"]}, index=["S", "R"])
    adata = SimpleNamespace(var=var)
    found = gene_set_mask_kwargs(_cfg(mask_strategy="gene", mask_genes=["role=response"]), adata)
    assert found == {"mask_gene_columns": [1]}
    assert gene_set_mask_kwargs(_cfg(), object()) == {}  # var is not read without a gene set


def _global_module(max_seq_len=64):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    cfg.model.global_component.parameters.n_heads = 2
    cfg.dataset.spatial_key = "spatial"
    torch.manual_seed(0)
    common = {"n_input": G, "n_output": G, "n_embed": 8, "decoder_type": "linear", "dropout_decoder": 0.0}
    common |= {"decoder_hidden_dims": [8], "mask_percentage": 1.0, "mask_strategy": "gene"}
    return GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **common).eval()


def test_no_hidden_response_reaches_a_prediction_and_the_drivers_do():
    module = _global_module()
    datas = [_lattice(6, 0), _lattice(5, 1)]
    draw_masks(datas, "gene_set", 1.0, seed=0, columns=[4, 5])
    batch = Batch.from_data_list(datas)
    with torch.no_grad():
        base = module._common_step(batch.clone(), "regression", "node").y_pred
        changed = batch.clone()
        changed.x = batch.x.clone()
        changed.x[:, [4, 5]] = torch.rand(batch.num_nodes, 2) * 9.0
        assert torch.equal(module._common_step(changed, "regression", "node").y_pred, base)
        drivers = batch.clone()
        drivers.x = batch.x.clone()
        drivers.x[:, 0] += 1.0
        assert not torch.allclose(module._common_step(drivers, "regression", "node").y_pred, base)


# --------------------------------------------------------------------------- fitted references


def _field_graph(side, seed):
    """Gene 1 = 2 x gene 0 of the same cell; gene 2 = a 20 um Gaussian smooth of gene 3 around the cell."""
    gen = torch.Generator().manual_seed(seed)
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    n = side * side
    x = torch.rand(n, G, generator=gen) * 2.0
    x[:, 1] = 2.0 * x[:, 0] + 0.05 * torch.rand(n, generator=gen)
    w = torch.exp(-((torch.cdist(pos, pos) * UNIT) ** 2) / (2 * 20.0**2))
    w.fill_diagonal_(0.0)
    x[:, 2] = (w @ x[:, 3]) / w.sum(1) * 3.0
    return _lattice(side, x=x)


def _fit(train, **kwargs):
    options = {"spatial_unit_um": UNIT, "kernels_um": (20.0, 60.0)} | kwargs
    return FittedReferences(train, **options)


def test_the_cells_own_genes_and_its_surroundings_each_predict_what_they_carry():
    train = [_field_graph(9, s) for s in range(6)]
    test = [_field_graph(9, s) for s in range(10, 13)]
    for d in (*train, *test):
        draw_masks([d], "gene_set", 1.0, seed=0, columns=[1, 2])
    refs = _fit(train, position=position_features(spatial_unit_um=UNIT, patch_um=(30.0,), max_offset_um=50.0))
    assert set(refs.models) == {"ref_own", "ref_own+context", "ref_own+context+position"}
    assert list(refs.target_columns) == [1, 2]
    heads = {}
    for d in test:
        batch = Batch.from_data_list([d])
        for name, value in refs(batch, batch.gene_mask.bool()).items():
            heads.setdefault(name, []).append(value.numpy())
    heads = {k: np.concatenate(v) for k, v in heads.items()}
    true = np.concatenate([d.x.numpy() for d in test])
    hidden = np.concatenate([d.gene_mask.numpy() for d in test])
    from interscale.evaluation.masked_probing import HeadPredictions

    scores = score_heads(
        HeadPredictions(heads=heads, true=true, hidden=hidden),
        [f"g{j}" for j in range(G)],
        center=np.concatenate([d.x.numpy() for d in train]).mean(0),
        gene_groups={"own": ["g1"], "field": ["g2"]},
    )["by_group"].set_index("group")
    assert scores.loc["own", "ref_own_cosine"] > 0.95
    # The wider sets keep it: their penalty is chosen on held-out training graphs.
    assert scores.loc["own", "ref_own+context+position_cosine"] > 0.93
    assert set(refs.alpha_) == set(refs.models)
    assert scores.loc["field", "ref_own+context_cosine"] > scores.loc["field", "ref_own_cosine"] + 0.3
    assert scores.loc["field", "ref_own+context_cosine"] > 0.8
    # A gene never hidden in training is predicted as its training mean.
    fill = np.concatenate([d.x.numpy() for d in train]).mean(0)
    assert np.allclose(heads["ref_own"][:, 0], fill[0])


def test_no_hidden_value_reaches_a_fitted_reference():
    train = [_field_graph(7, s) for s in range(4)]
    draw_masks(train, "gene_set", 1.0, seed=0, columns=[1, 2])
    refs = _fit(train)
    data = _field_graph(7, 20)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[1, 2])
    batch = Batch.from_data_list([data])
    found = refs(batch, batch.gene_mask.bool())
    changed = batch.clone()
    changed.x = batch.x.clone()
    changed.x[:, [1, 2]] += 7.0
    again = refs(changed, changed.gene_mask.bool())
    assert all(torch.equal(found[k], again[k]) for k in found)


def test_a_fit_saved_and_loaded_predicts_exactly_what_the_fit_did(tmp_path):
    """Jobs sharing one fit: the state goes through pickle, the position function is handed back in."""
    import pickle

    train = [_field_graph(7, s) for s in range(4)]
    draw_masks(train, "gene_set", 1.0, seed=0, columns=[1, 2])
    position = position_features(spatial_unit_um=UNIT, patch_um=(30.0,), max_offset_um=50.0)
    refs = _fit(train, position=position)
    path = tmp_path / "fit.pkl"
    path.write_bytes(pickle.dumps(refs.state()))
    loaded = FittedReferences.from_state(pickle.loads(path.read_bytes()), position=position)

    data = _field_graph(7, 20)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[1, 2])
    batch = Batch.from_data_list([data])
    hidden = batch.gene_mask.bool()
    found, again = refs(batch, hidden), loaded(batch, hidden)
    assert set(found) == set(again) and all(torch.equal(found[k], again[k]) for k in found)
    assert loaded.alpha_ == refs.alpha_ and list(loaded.target_columns) == list(refs.target_columns)
    with pytest.raises(ValueError, match="lacks"):
        FittedReferences.from_state({"fill": refs.fill})


def test_through_the_collection_every_fitted_reference_row_is_its_own_cell():
    train = [_field_graph(7, s) for s in range(3)]
    draw_masks(train, "gene_set", 1.0, seed=0, columns=[1, 2])
    refs = _fit(train)
    data = _field_graph(7, 30)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[1, 2])
    heads = collect_head_predictions(
        _global_module(max_seq_len=30),  # 49 cells: pad_batch keeps 30, in its own order
        [data],
        prediction_task="regression",
        prediction_level="node",
        mask_strategy="gene",
        batch_size=1,
        device=torch.device("cpu"),
        fitted_references=refs,
    )
    direct = refs(Batch.from_data_list([data]), data.gene_mask.bool())["ref_own"].numpy()
    rows = [int(np.flatnonzero((data.x.numpy() == row).all(1))[0]) for row in heads.true]
    assert len(rows) == 30 and len(set(rows)) == 30  # a subsample of the 49 cells
    assert np.allclose(heads.heads["ref_own"], direct[rows])


# --------------------------------------------------------------------------- response influence


class _Planted(torch.nn.Module):
    """A head whose responses (columns ``out``) read gene ``driver`` of other cells: ``y = A @ x[:, driver]``.

    ``leak`` reads the driver through the *unmasked* input, which only a broken pipeline would.
    """

    registered_global_component = False

    def __init__(self, weights, driver, out, leak=False):
        super().__init__()
        self.weights, self.driver, self.out, self.leak = weights, driver, out, leak
        self.scale = torch.nn.Parameter(torch.ones(()))

    def _common_step(self, batch, prediction_task, prediction_level):
        hidden = batch.gene_mask.bool()
        x = batch.x if self.leak else batch.x.masked_fill(hidden, MASK_VALUE)
        y = torch.zeros_like(batch.x)
        y[:, self.out] = (self.weights @ x[:, self.driver])[:, None] * self.scale
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=hidden, view=SimpleNamespace(padded_node_idx=None))


def _influence(module, data, edges, **kwargs):
    options = {
        "prediction_task": "regression",
        "prediction_level": "node",
        "device": torch.device("cpu"),
        "spatial_unit_um": UNIT,
        "edges": edges,
        "edge_genes": lambda program: [5],
        "n_projections": 4,
        "tolerance_um": 1.0,  # the same lattice distance only: steps are 10 um here
    }
    return response_influence(module, [data], **(options | kwargs))


def test_a_planted_sender_ranks_one_and_a_distance_kernel_one_half():
    data = _lattice(9, 0)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[5])
    n = data.num_nodes
    dist = torch.cdist(data.pos, data.pos)
    kernel = torch.exp(-dist / 3.0)
    centre = 40
    sender = centre + 3  # three steps away
    planted = kernel.clone()
    planted[centre, sender] += 1.0
    edges = pd.DataFrame({"sender": [sender], "receiver": [centre], "range_class": ["mid"], "program": ["P>R"]})
    state = torch.get_rng_state()
    found = _influence(_Planted(planted, driver=0, out=[5]), data, edges)["edges"]
    assert torch.equal(state, torch.get_rng_state())
    ranks = found.set_index(["range_class", "program"])["value"]
    assert ranks[("mid", "P>R")] == pytest.approx(1.0) and ranks[("mid", "any")] == pytest.approx(1.0)
    blind = _influence(_Planted(kernel, driver=0, out=[5]), data, edges)["edges"].set_index(["range_class", "program"])
    assert blind.loc[("mid", "P>R"), "value"] == pytest.approx(0.5)
    assert n == 81


def test_a_cell_counts_through_its_visible_entries_only():
    data = _lattice(9, 0)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[5])
    planted = torch.zeros(81, 81)
    planted[40, 43] = 1.0
    edges = pd.DataFrame({"sender": [43], "receiver": [40], "range_class": ["mid"], "program": ["P>R"]})
    # The head reads the sender's *hidden* gene 5 straight from the unmasked input: no visible entry carries it.
    leaking = _Planted(planted, driver=5, out=[5], leak=True)
    found = _influence(leaking, data, edges)["edges"].set_index(["range_class", "program"])
    assert found.loc[("mid", "P>R"), "value"] == pytest.approx(0.5), "influence through a hidden entry counted"


def test_receivers_without_their_response_hidden_are_not_queried():
    data = _lattice(9, 0)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[4])  # gene 5, the programme's response, stays visible
    edges = pd.DataFrame({"sender": [43], "receiver": [40], "range_class": ["mid"], "program": ["P>R"]})
    found = _influence(_Planted(torch.eye(81), driver=0, out=[5]), data, edges)["edges"]
    assert found.empty


class _TwoProgrammes(torch.nn.Module):
    """Response gene 5 reads the planted sender, response gene 4 (another programme) a distractor as far away."""

    registered_global_component = False

    def __init__(self, receiver, sender, distractor, n):
        super().__init__()
        self.own = torch.zeros(n, n)
        self.own[receiver, sender] = 1.0
        self.other = torch.zeros(n, n)
        self.other[receiver, distractor] = 10.0

    def _common_step(self, batch, prediction_task, prediction_level):
        hidden = batch.gene_mask.bool()
        x = batch.x.masked_fill(hidden, MASK_VALUE)
        y = torch.zeros_like(batch.x)
        y[:, 5] = self.own @ x[:, 0]
        y[:, 4] = self.other @ x[:, 0]
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=hidden, view=SimpleNamespace(padded_node_idx=None))


def test_a_receiver_is_read_on_its_programmes_response_genes_only():
    data = _lattice(9, 0)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[4, 5])
    edges = pd.DataFrame({"sender": [43], "receiver": [40], "range_class": ["mid"], "program": ["P>R"]})
    module = _TwoProgrammes(receiver=40, sender=43, distractor=37, n=81)  # 43 and 37: three steps either side
    found = _influence(module, data, edges)["edges"].set_index(["range_class", "program"])
    assert found.loc[("mid", "P>R"), "value"] == pytest.approx(1.0), "another programme's sender leaked in"
