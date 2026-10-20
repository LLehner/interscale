"""Sender attribution: what a receiver's predicted response takes from each cell.

The squared gradient of ``receptive_field`` is a *sensitivity*: it says where a model can look, not
what it used. Ranking planted senders by it gave a GCN's senders 0.41 while its responses were
predicted at 0.59, because a mean-aggregating GCN weighs all its neighbours alike. The default of
``response_influence`` is therefore an attribution, integrated gradients from a baseline cell.

Pinned here:

* the difference -- a uniform aggregator cannot rank its sender by gradient (0.5) and can by
  attribution (1); a cell the model is very sensitive to but that carries nothing beyond the baseline
  ranks 1 by gradient and 0 by attribution; the size of a contribution counts, not its sign;
* the contributions add up to the response's change from the baseline, and the check says when the
  integration is too coarse; a hidden entry does not move, so it contributes nothing;
* the null -- a distance kernel on random inputs ranks senders at chance;
* the path -- the passes along it see one subsample of a long graph; a model without a GCN is moved
  from its graph's mean token; a real GCN is attributed within its hops only, and a dual decoder's
  global head beyond them;
* the plumbing -- unknown measures and mis-sized baselines are refused, no generator is touched.
"""

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

from interscale.evaluation import receptive_field as rf
from interscale.evaluation.masked_probing import _model_heads, draw_masks
from interscale.evaluation.receptive_field import (
    INFLUENCE_MEASURES,
    _frozen_parameters,
    _integration_points,
    _program_attribution,
    response_influence,
)
from interscale.tl.masking import MASK_VALUE

G, E, UNIT = 6, 8, 10.0
RESPONSE = 5  # the gene hidden in every cell


def _lattice(side, seed=0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float()
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    data = Data(x=torch.rand(n, G, generator=gen) * 3.0, edge_index=(torch.cdist(pos, pos) == 1).nonzero().t(), pos=pos)
    data.obs_names = torch.arange(n, dtype=torch.float32) + 1
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[RESPONSE])
    return data


class _Head(torch.nn.Module):
    """``y[:, RESPONSE] = shape(weights @ x[:, driver])`` on the masked input: a response read off other cells' driver gene.

    ``leak`` reads the driver through the *unmasked* input, which only a broken pipeline would.
    """

    registered_global_component = False

    def __init__(self, weights, driver=0, shape=lambda z: z, leak=False):
        super().__init__()
        self.weights, self.driver, self.shape, self.leak = weights, driver, shape, leak
        self.scale = torch.nn.Parameter(torch.ones(()))

    def _common_step(self, batch, prediction_task, prediction_level):
        hidden = batch.gene_mask.bool()
        x = batch.x if self.leak else batch.x.masked_fill(hidden, MASK_VALUE)
        y = torch.zeros_like(batch.x)
        y[:, RESPONSE] = self.shape(self.weights @ x[:, self.driver]) * self.scale
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=hidden, view=SimpleNamespace(padded_node_idx=None))


def _edges(pairs, program="P>R", range_class="mid"):
    return pd.DataFrame(
        {
            "sender": [s for s, _ in pairs],
            "receiver": [r for _, r in pairs],
            "range_class": range_class,
            "program": program,
        }
    )


def _rank(module, data, edges, **kwargs):
    """The planted pairs' mean matched rank (programme ``any``), and the whole result."""
    options = {
        "prediction_task": "regression",
        "prediction_level": "node",
        "device": torch.device("cpu"),
        "spatial_unit_um": UNIT,
        "edges": edges,
        "edge_genes": lambda program: [RESPONSE],
        "tolerance_um": 0.5,  # the same lattice distance only: steps are 10 um here
        "n_projections": 4,
    }
    found = response_influence(module, [data], **(options | kwargs))
    ranks = found["edges"].set_index(["range_class", "program"])["value"]
    return float(ranks.xs("any", level="program").iloc[0]), found


def _attribute(module, data, queries, steps=8, baseline=None, genes=(RESPONSE,)):
    masks = [np.isin(np.arange(G), list(genes)) for _ in queries]
    with _frozen_parameters(module):
        return _program_attribution(
            module, Batch.from_data_list([data]), queries, masks, baseline, "regression", steps, _model_heads
        )


def _uniform_aggregator(dist, radius=3.0):
    """Row-normalised: every cell within ``radius`` weighs the same, like a mean-aggregating GCN."""
    near = ((dist <= radius) & (dist > 0)).float()
    return near / near.sum(1, keepdim=True)


CENTRE, SENDER = 40, 43  # on the 9 x 9 lattice, three steps apart


# --------------------------------------------------------------------------- the difference


def test_a_uniform_aggregator_cannot_rank_its_sender_by_gradient_but_can_by_attribution():
    data = _lattice(9)
    data.x[SENDER, 0] = 6.0  # the sender carries the driver; the other cells hold 0-3
    head = _Head(_uniform_aggregator(torch.cdist(data.pos, data.pos)))
    edges = _edges([(SENDER, CENTRE)])
    assert _rank(head, data, edges, measure="gradient")[0] == pytest.approx(0.5)  # equal weights: all ties
    assert _rank(head, data, edges, measure="attribution")[0] == pytest.approx(1.0)


def test_a_cell_the_model_is_sensitive_to_but_that_carries_nothing_new_is_not_attributed():
    data = _lattice(9)
    dist = torch.cdist(data.pos, data.pos)
    weights = torch.exp(-dist / 3.0)
    weights[CENTRE, SENDER] += 3.0  # the receiver's response is far more sensitive to the sender ...
    data.x[SENDER, 0] = 1.0  # ... which sits exactly at the baseline
    edges = _edges([(SENDER, CENTRE)])
    baseline = torch.full((G,), 1.0)
    assert _rank(_Head(weights), data, edges, measure="gradient", baseline=baseline)[0] == pytest.approx(1.0)
    assert _rank(_Head(weights), data, edges, measure="attribution", baseline=baseline)[0] == pytest.approx(0.0)


def test_the_size_of_a_contribution_ranks_not_its_sign():
    data = _lattice(9)
    data.x[SENDER, 0] = 6.0
    weights = _uniform_aggregator(torch.cdist(data.pos, data.pos))
    weights[CENTRE, SENDER] *= -1.0  # the sender pushes the response down
    contribution, _ = _attribute(_Head(weights), data, [CENTRE])["local"]
    assert contribution[0, SENDER] < 0
    assert _rank(_Head(weights), data, _edges([(SENDER, CENTRE)]))[0] == pytest.approx(1.0)


def test_a_distance_kernel_on_random_inputs_ranks_senders_at_chance_and_the_gradient_at_exactly_half():
    data = _lattice(21, seed=1)
    dist = torch.cdist(data.pos, data.pos)
    head = _Head(torch.exp(-dist / 3.0))
    receivers = [r * 21 + c for r in range(5, 16) for c in range(5, 16)]
    edges = _edges([(r + 3 * 21 + 4, r) for r in receivers])  # five steps away: a ring of 12 cells as far
    chance, _ = _rank(head, data, edges, queries_per_program=len(receivers))
    assert abs(chance - 0.5) < 0.08
    assert _rank(head, data, edges, measure="gradient", queries_per_program=len(receivers))[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- completeness


def test_the_integration_points_integrate_polynomials_exactly():
    for steps in (1, 2, 5):
        nodes, weights = _integration_points(steps)
        assert weights.sum() == pytest.approx(1.0) and ((nodes > 0) & (nodes < 1)).all()
        for degree in range(2 * steps):  # exact up to degree 2 * steps - 1
            assert (weights * nodes**degree).sum() == pytest.approx(1.0 / (degree + 1))
    with pytest.raises(ValueError, match="at least 1"):
        _integration_points(0)


def test_the_contributions_add_up_to_the_change_and_the_check_says_when_they_do_not():
    data = _lattice(9)
    data.x[SENDER, 0] = 600.0  # far from the baseline, where a cubic bends
    head = _Head(_uniform_aggregator(torch.cdist(data.pos, data.pos)), shape=lambda z: z**3)
    edges = _edges([(SENDER, CENTRE)])
    gaps = {}
    for steps in (1, 2, 8):
        _, found = _rank(head, data, edges, steps=steps)
        check = found["check"].set_index(["head", "program"])
        assert set(check.index.get_level_values("program")) == {"P>R", "any"} and check["n_queries"].min() == 1
        gaps[steps] = float(check.loc[("local", "any"), "value"])
    assert gaps[1] > 0.05  # one point does not integrate a cubic's gradient
    assert gaps[2] < 1e-4 and gaps[8] < 1e-4  # two do (degree 2 in the path)


class _TwoShapes(torch.nn.Module):
    """Response gene 5 is linear in the neighbours' driver, response gene 4 its cube: two programmes the integration reads differently."""

    registered_global_component = False

    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.scale = torch.nn.Parameter(torch.ones(()))

    def _common_step(self, batch, prediction_task, prediction_level):
        hidden = batch.gene_mask.bool()
        z = self.weights @ batch.x.masked_fill(hidden, MASK_VALUE)[:, 0]
        y = torch.zeros_like(batch.x)
        y[:, 5], y[:, 4] = z * self.scale, z**3 * self.scale
        return SimpleNamespace(y_pred=y, y_true=batch.x, entry_mask=hidden, view=SimpleNamespace(padded_node_idx=None))


def test_the_check_reads_each_programme_on_its_own_queries_and_all_of_them_together():
    data = _lattice(9)
    draw_masks([data], "gene_set", 1.0, seed=0, columns=[4, 5])
    data.x[SENDER, 0] = 600.0
    data.x[13, 0] = 6000.0  # a second sender, for the second receiver
    head = _TwoShapes(_uniform_aggregator(torch.cdist(data.pos, data.pos)))
    edges = pd.concat([_edges([(SENDER, CENTRE)], program="LIN"), _edges([(13, 10)], program="CUB")])
    genes = {"LIN": [5], "CUB": [4]}
    _, found = _rank(head, data, edges, steps=1, edge_genes=genes.__getitem__)
    gap = found["check"].set_index(["head", "program"])["value"]
    assert gap[("local", "LIN")] < 1e-4  # a linear response integrates exactly
    assert gap[("local", "CUB")] > 0.05  # a cubic one needs more than one point
    # The share missed over both queries lies between the two, weighted by how much each response moved.
    assert gap[("local", "LIN")] < gap[("local", "any")] < gap[("local", "CUB")]
    assert found["check"].set_index(["head", "program"])["n_queries"].tolist() == [1, 1, 2]


def test_queries_the_transformer_left_out_are_not_in_the_check():
    module = _global(max_seq_len=20)  # 25 cells: pad_batch keeps 20
    data = _lattice(5)
    edges = _edges([(r + 1, r) for r in range(24)])
    random.seed(3)
    ranked, found = _rank(module, data, edges, steps=4, queries_per_program=25)
    check = found["check"].set_index(["head", "program"])
    assert 15 <= check.loc[("global", "any"), "n_queries"] < 24  # 24 receivers asked; the cells pad_batch dropped have no row
    assert np.isfinite(check["value"]).all() and np.isfinite(ranked)


def test_a_hidden_entry_does_not_move_so_it_contributes_nothing():
    data = _lattice(9)
    data.x[SENDER, RESPONSE] = 6.0  # the hidden gene: the sender's value of it is never given to the model
    planted = torch.zeros(81, 81)
    planted[CENTRE, SENDER] = 1.0
    edges = _edges([(SENDER, CENTRE)])
    leaking = _Head(planted, driver=RESPONSE, leak=True)  # reads that value through the unmasked input
    ranked, found = _rank(leaking, data, edges)
    assert ranked == pytest.approx(0.5)
    contribution, _ = _attribute(leaking, data, [CENTRE])["local"]
    assert contribution.abs().max() == 0
    data.x[SENDER, 0] = 6.0
    assert _rank(_Head(planted, driver=0), data, edges)[0] == pytest.approx(1.0)  # a visible driver is read


def test_the_baseline_is_the_cell_the_contributions_are_measured_from():
    data = _lattice(9)
    head = _Head(_uniform_aggregator(torch.cdist(data.pos, data.pos)))
    low, _ = _attribute(head, data, [CENTRE], baseline=torch.zeros(G))["local"]
    high, _ = _attribute(head, data, [CENTRE], baseline=torch.full((G,), 3.0))["local"]
    assert (low[0] >= 0).all() and (high[0] <= 0).all()  # inputs lie between 0 and 3
    default, _ = _attribute(head, data, [CENTRE])["local"]
    assert torch.equal(default, _attribute(head, data, [CENTRE], baseline=data.x.mean(0))["local"][0])  # the graph's mean
    assert not torch.equal(default, low)
    with pytest.raises(ValueError, match="baseline holds 3 genes"):
        _attribute(head, data, [CENTRE], baseline=torch.zeros(3))


# --------------------------------------------------------------------------- the path


def _cfg():
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 256
    cfg.model.global_component.parameters.n_heads = 2
    cfg.dataset.spatial_key = "spatial"
    return cfg


_COMMON = {"n_input": G, "n_output": G, "n_embed": E, "decoder_type": "linear", "dropout_decoder": 0.0}
_COMMON |= {"decoder_hidden_dims": [16], "mask_percentage": 1.0, "mask_strategy": "gene"}


def _local():
    from interscale.module.base import LocalModule

    torch.manual_seed(0)
    return LocalModule.from_config(_cfg(), **_COMMON).eval()


def _global(max_seq_len=256):
    from interscale.module.base import GlobalModule

    cfg = _cfg()
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    torch.manual_seed(0)
    return GlobalModule.from_config(cfg, local_mask_hops=0, type_gex_embedding="Linear", **_COMMON).eval()


def _dual():
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = _cfg()
    cfg.model.decoder.dual_decoder = True
    torch.manual_seed(0)
    return DualDecoderCombinedModule(cfg=cfg, **(_COMMON | {"decoder_type": None})).eval()


@pytest.mark.parametrize("build", [_local, _dual], ids=["local", "dual"])
def test_a_gcn_head_is_attributed_within_its_hops_only_and_the_contributions_add_up(build):
    module = build()
    data = _lattice(9)
    found = _attribute(module, data, [CENTRE])
    beyond = torch.cdist(data.pos, data.pos)[CENTRE] > 2.01  # a 2-layer GCN on a 4-neighbour lattice: L1 distance 2
    assert set(found) == ({"local"} if build is _local else {"local", "global"})
    for head, (contribution, change) in found.items():
        assert contribution[0].sum().item() == pytest.approx(change[0].item(), rel=0.02), head
        if head == "local":
            assert contribution[0, beyond].abs().max() == 0 and contribution[0, ~beyond].abs().max() > 0
        else:
            assert contribution[0, beyond].abs().max() > 0  # the transformer looks past the GCN's hops


def test_a_model_without_a_gcn_is_moved_from_its_graphs_mean_token():
    module = _global()
    data = _lattice(6)
    contribution, change = _attribute(module, data, [14, 21])["global"]
    assert "gex_embedding" not in vars(module)  # the capture is gone
    assert change.abs().min() > 0
    # A ReLU feed-forward bends the path at kinks, so eight points land within a few percent, not exactly.
    assert torch.allclose(contribution.sum(1), change, rtol=0.05, atol=1e-7)
    assert contribution.shape == (2, 36)

    # The change is the response at the actual tokens minus the response with every cell at the mean token.
    batch = Batch.from_data_list([data])
    original = module.gex_embedding
    with torch.no_grad():
        at_input = module._common_step(batch, "regression", "node").y_pred
        module.gex_embedding = lambda b, hidden=None: original(b, hidden=hidden).mean(0, keepdim=True).expand(36, -1)
        try:
            at_mean = module._common_step(batch, "regression", "node").y_pred
        finally:
            del module.gex_embedding
    expected = (at_input - at_mean)[[14, 21], RESPONSE].double()
    assert torch.allclose(change, expected, atol=1e-6)


def test_the_passes_along_the_path_see_one_subsample_of_a_long_graph(monkeypatch):
    module = _global(max_seq_len=20)  # 25 cells: pad_batch keeps 20, drawing with Python's random
    data = _lattice(5)

    def gap() -> float:
        random.seed(3)
        contribution, change = _attribute(module, data, list(range(25)), steps=6)["global"]
        kept = torch.isfinite(change)
        assert 5 < int(kept.sum()) < 25  # some cells are left out, the others are queried
        return ((contribution.sum(1) - change).abs()[kept].sum() / change.abs()[kept].sum()).item()

    assert gap() < 0.02
    monkeypatch.setattr(rf, "_replayable_rng", lambda device: lambda: None)  # every pass draws its own subsample
    assert gap() > 0.1


def test_a_cell_the_transformer_left_out_has_no_row():
    module = _global(max_seq_len=20)
    data = _lattice(5)
    random.seed(3)
    contribution, change = _attribute(module, data, list(range(25)), steps=2)["global"]
    left_out = ~torch.isfinite(change)
    assert left_out.any() and torch.isnan(contribution[left_out]).all()
    assert torch.isfinite(contribution[~left_out]).all()


def test_response_influence_reads_both_heads_of_a_dual_decoder_and_checks_the_sum():
    module = _dual()
    data = _lattice(9)
    edges = pd.concat([_edges([(CENTRE + 3, CENTRE)], range_class="far"), _edges([(CENTRE + 1, CENTRE)], range_class="near")])
    _, found = _rank(module, data, edges, steps=8)
    ranks = found["edges"].set_index(["head", "range_class", "program"])["value"]
    assert ranks[("local", "far", "P>R")] == pytest.approx(0.5)  # three steps: past the GCN, nothing to rank
    assert set(ranks.index.get_level_values("head")) == {"local", "global"}
    check = found["check"].set_index(["head", "program"])["value"]
    assert (check < 0.05).all() and set(check.index.get_level_values("head")) == {"local", "global"}


# --------------------------------------------------------------------------- the plumbing


def test_unknown_measures_are_refused_and_the_known_ones_listed():
    assert INFLUENCE_MEASURES == ("attribution", "gradient")
    data = _lattice(9)
    with pytest.raises(ValueError, match="measure must come from"):
        _rank(_Head(torch.eye(81)), data, _edges([(SENDER, CENTRE)]), measure="ablation")


def test_the_attribution_touches_no_generator_and_restores_the_module():
    module = _global()
    module.train()
    flags = [p.requires_grad for p in module.parameters()]
    data = _lattice(6)
    edges = _edges([(15, 14)])
    states = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    first = _rank(module, data, edges, steps=3)[1]["edges"]
    assert module.training and flags == [p.requires_grad for p in module.parameters()]
    assert "gex_embedding" not in vars(module)
    assert torch.equal(states[0], torch.get_rng_state())
    assert np.array_equal(states[1], np.random.get_state()[1]) and states[2] == random.getstate()
    pd.testing.assert_frame_equal(first, _rank(module, data, edges, steps=3)[1]["edges"])
