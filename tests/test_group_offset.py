"""A field of view's offset as a known input (``model.decoder.group_offset``), the gene set hidden in disc cells only
(``draw_masks(kind="gene_set+blocks")``) and the group / composition parts of the fitted references (``.claude/ceiling_plan.md``,
2026-11-12: on CosMx the FOV mean carries more of the neighbourhood term than the neighbours do).

Pinned here:

* **the mean** -- per (graph, group), over visible entries only; the same code in two graphs is two groups;
* **the part** -- the prediction is the decoder's output plus the label mean plus ``group``, the group's visible mean of the residual
  from the label means times a weight per gene starting at 1; no hidden value reaches it;
* **the switch** -- off, the module holds no new state and draws no random number; validator and checkpoint tag;
* **the mask** -- ``gene_set+blocks`` hides the columns in exactly the cells a ``blocks`` draw hides, nothing else;
* **the references** -- ``group`` and ``composition`` parts read no hidden value, add their references, and survive a saved fit.

Every parameter is stirred off its initialisation (tests-stir-the-init).
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from anndata import AnnData
from torch_geometric.data import Batch, Data

E, G, LABELS, GROUPS = 8, 6, 3, 2


def _stir(module, seed=0, scale=0.1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))
    return module


def _graph(side, seed, *, hidden_genes=(0, 1), hidden_rows=None):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ii.flatten(), jj.flatten()], dim=1).float() * 100.0
    n = side * side
    gen = torch.Generator().manual_seed(seed)
    edges = (torch.cdist(pos, pos) <= 101.0).nonzero().t()
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.rand(n, G, generator=gen) * 3 + 1.0, edge_index=edges, pos=pos)
    data.cell_type = torch.nn.functional.one_hot(torch.arange(n) % LABELS, LABELS).float()
    data.fov = torch.nn.functional.one_hot((ii.flatten() >= side // 2).long(), GROUPS).float()  # two halves
    rows = torch.ones(n, dtype=torch.bool) if hidden_rows is None else hidden_rows
    data.gene_mask = torch.zeros(n, G, dtype=torch.bool)
    data.gene_mask[:, list(hidden_genes)] = rows[:, None]
    data.mask = data.gene_mask.any(1)
    data.obs_names = torch.arange(n)
    return data


def _cfg(group="visible_mean", key="fov", section="label_mean", extra=("cell_type", "fov")):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    cfg.model.decoder.dropout_decoder = 0.0
    cfg.model.decoder.section_offset = section
    cfg.model.decoder.label_offset_key = "cell_type" if section == "label_mean" else ""
    cfg.model.decoder.group_offset = group
    cfg.model.decoder.group_offset_key = key
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads, params.num_layers, params.dropout_global = 64, 2, 2, 0.0
    params.type_gex_embedding = "Linear"
    cfg.optim.loss = "SCELoss"
    cfg.optim.center_targets = section == "label_mean"
    cfg.dataset.mask_strategy = "gene"
    cfg.dataset.mask_percentage = 1.0
    cfg.dataset.mask_genes = ["g0", "g1"]
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    cfg.dataset.sample_key = ["sample"]
    cfg.dataset.extra_obs_keys = list(extra)
    return cfg


def _model(seed=0, **kwargs):
    from interscale.config import _validate
    from interscale.model.global_model import GlobalModel

    cfg = _cfg(**kwargs)
    _validate(cfg)
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(40, G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 40)
    adata.obs["split"] = pd.Categorical(["train"] * 38 + ["val"] * 2)
    adata.obs["cell_type"] = pd.Categorical(["a", "b"] * 20)
    adata.obs["fov"] = pd.Categorical(["f1", "f2"] * 20)
    adata.obsm["spatial"] = rng.uniform(0, 500, size=(40, 2))
    GlobalModel._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(seed)
    return GlobalModel(adata, cfg=cfg)


def _manual_group_mean(x, visible, graph, group):
    out = torch.zeros_like(x)
    for gr in graph.unique():
        for k in group.unique():
            rows = (graph == gr) & (group == k)
            if not rows.any():
                continue
            w = visible[rows].double()
            total, count = (x[rows].double() * w).sum(0), w.sum(0)
            out[rows] = (total / count.clamp_min(1.0)).to(x.dtype)
    return out


def test_the_group_mean_is_per_graph_and_group_over_visible_entries_only():
    from interscale.tl.masking import group_visible_mean

    gen = torch.Generator().manual_seed(0)
    x = torch.rand(12, G, generator=gen)
    visible = torch.rand(12, G, generator=gen) > 0.3
    graph = torch.tensor([0] * 6 + [1] * 6)
    group = torch.tensor([0, 0, 1, 1, 1, 0] * 2)  # the same codes in both graphs: four groups
    got = group_visible_mean(x, visible, graph, group)
    assert torch.allclose(got, _manual_group_mean(x, visible, graph, group), atol=1e-6)
    # A hidden entry's value never matters.
    y = x.clone()
    y[~visible] = 1e6
    assert torch.allclose(group_visible_mean(y, visible, graph, group), got, atol=1e-6)


def test_the_part_is_the_groups_visible_mean_of_the_residual_and_no_hidden_value_reaches_it():
    net = _model()
    module = _stir(net.module)
    graphs = [_graph(4, 1), _graph(5, 2)]
    module.set_label_means(graphs)
    module.eval()
    batch = Batch.from_data_list(graphs)
    out = module._common_step(batch, "regression", "node")
    parts = out.parts["global"]
    assert set(parts) >= {"context", "section", "group"}
    assert torch.allclose(sum(parts.values()), out.y_pred, atol=1e-5)

    # The part by hand: the residual from the label means, averaged over visible entries per (graph, fov), times the weight.
    masked, _, _ = module._common_step_masking(batch)
    visible = module.visible_entries(masked)
    codes = masked.cell_type.argmax(1)
    resid = masked.x.float() - module.label_means[codes].float()
    want = _manual_group_mean(resid, visible, masked.batch, masked.fov.argmax(1)) * module.group_offset_weight.detach()
    got = module.group_input(masked, masked=True)
    assert torch.allclose(got, want, atol=1e-5)

    # Hidden entries (genes 0, 1 in every cell) changed: the group part does not move.
    changed = [g.clone() for g in graphs]
    for g in changed:
        g.x[:, :2] = g.x[:, :2] * 7 + 3
    out2 = module._common_step(Batch.from_data_list(changed), "regression", "node")
    assert torch.allclose(out2.parts["global"]["group"], parts["group"], atol=1e-6)


def test_off_the_module_holds_no_new_state_and_draws_no_random_number():
    on = _model(seed=3)
    off = _model(seed=3, group="none", key="")
    extra = set(on.module.state_dict()) - set(off.module.state_dict())
    assert extra == {"group_offset_weight"}
    assert torch.equal(on.module.group_offset_weight.detach(), torch.ones(G))
    shared = set(off.module.state_dict())
    for name in shared:
        assert torch.equal(on.module.state_dict()[name], off.module.state_dict()[name]), name
    assert off.module.group_input(Batch.from_data_list([_graph(4, 1)])) is None


def test_the_weight_learns_where_the_gene_is_visible_in_other_cells_of_the_group():
    # Hidden in a third of the cells (as the conditional arms train): the group's other cells show the gene. Hidden in every cell
    # (programme masking) the part of a hidden gene is 0 and its weight gets no gradient, as the section offset's.
    net = _model()
    module = _stir(net.module)
    graphs = [_graph(4, 1, hidden_rows=torch.arange(16) % 3 == 0), _graph(5, 2, hidden_rows=torch.arange(25) % 3 == 0)]
    module.set_label_means(graphs)
    out = module._common_step(Batch.from_data_list(graphs), "regression", "node")
    loss = ((out.y_pred - out.y_true)[out.entry_mask] ** 2).mean()
    loss.backward()
    assert module.group_offset_weight.grad is not None and module.group_offset_weight.grad.abs().sum() > 0


def test_a_step_refuses_without_the_group_part():
    net = _model()
    module = net.module
    graphs = [_graph(4, 1)]
    module.set_label_means(graphs)
    module.group_input = lambda *args, **kwargs: None  # a caller that forgets the part
    with pytest.raises(ValueError, match="group_offset needs"):
        module._common_step(Batch.from_data_list(graphs), "regression", "node")


def test_the_validator_and_the_tag():
    from interscale.config import _validate_group_offset
    from interscale.tl.utils import _decoder_tag

    _validate_group_offset(_cfg())
    with pytest.raises(ValueError, match="group_offset must be one of"):
        _validate_group_offset(_cfg(group="median"))
    with pytest.raises(ValueError, match="needs model.decoder.group_offset_key"):
        _validate_group_offset(_cfg(key=""))
    with pytest.raises(ValueError, match="read under group_offset visible_mean"):
        _validate_group_offset(_cfg(group="none"))
    with pytest.raises(ValueError, match="not combined with section_offset visible_mean"):
        _validate_group_offset(_cfg(section="visible_mean"))
    with pytest.raises((ValueError, KeyError)):
        _validate_group_offset(_cfg(extra=("cell_type",)))  # the group column does not reach the graphs
    assert _decoder_tag(_cfg()) == "dec-linear-lab-cell-type-grp-fov_"
    assert "grp" not in _decoder_tag(_cfg(group="none", key=""))


def test_gene_set_in_discs_hides_the_columns_in_the_cells_a_disc_draw_hides():
    from interscale.evaluation.masked_probing import draw_masks

    graphs = [_graph(10, 1), _graph(12, 2)]
    discs = [g.clone() for g in graphs]
    draw_masks(discs, "blocks", 0.2, seed=5, radius=150.0)
    sets = [g.clone() for g in graphs]
    draw_masks(sets, "gene_set+blocks", 0.0, seed=5, radius=150.0, columns=[1, 3], block_rate=0.2)
    for a, b in zip(discs, sets, strict=True):
        assert torch.equal(a.mask, b.mask)  # the same cells
        assert torch.equal(b.gene_mask[:, [1, 3]], a.mask[:, None].expand(-1, 2))
        assert not b.gene_mask[:, [0, 2, 4, 5]].any()
    # With a cell rate the first graph keeps its discs and gains cells; later graphs' discs move (the generator ran on).
    more = [g.clone() for g in graphs]
    draw_masks(more, "gene_set+blocks", 0.3, seed=5, radius=150.0, columns=[1, 3], block_rate=0.2)
    assert (more[0].mask >= sets[0].mask).all() and more[0].mask.sum() > sets[0].mask.sum()
    with pytest.raises(ValueError, match="gene_set\\+blocks"):
        draw_masks([g.clone() for g in graphs], "gene_set+blocks", 0.0, seed=5, radius=150.0, columns=None, block_rate=0.2)


def test_the_group_and_composition_parts_read_no_hidden_value_and_add_their_references():
    from interscale.evaluation.masked_probing import FittedReferences, _reference_parts

    rows = torch.zeros(36, dtype=torch.bool)
    rows[::3] = True
    train = [_graph(6, s, hidden_rows=rows) for s in (1, 2, 3)]
    fitted = FittedReferences(train, spatial_unit_um=1.0, k_nearest=(4, 8), group_attr="fov", composition_attr="cell_type",
                              alphas=(1.0, 10.0), folds=3)
    assert {"ref_own+group", "ref_own+context+group", "ref_own+composition+group", "ref_own+context+composition+group"} <= set(fitted.models)
    plain = FittedReferences(train, spatial_unit_um=1.0, k_nearest=(4,), alphas=(1.0,), folds=3)
    assert not any("group" in n or "composition" in n for n in plain.models)

    test = Batch.from_data_list([_graph(6, 9, hidden_rows=rows)])
    hidden = test.gene_mask.bool()
    parts = _reference_parts(test, hidden, spatial_unit_um=1.0, kernels_um=(), position=None, k_nearest=(4, 8),
                             group_attr="fov", composition_attr="cell_type")
    assert parts["composition"].shape[1] == 2 * LABELS
    assert torch.allclose(parts["composition"].reshape(-1, 2, LABELS).sum(-1), torch.ones(36, 2, dtype=torch.float64))
    changed = test.clone()
    changed.x = torch.where(hidden, changed.x * 9 + 2, changed.x)
    again = _reference_parts(changed, hidden, spatial_unit_um=1.0, kernels_um=(), position=None, k_nearest=(4, 8),
                             group_attr="fov", composition_attr="cell_type")
    for name in parts:
        assert torch.allclose(parts[name], again[name]), name
    # A saved fit predicts the same.
    restored = FittedReferences.from_state(fitted.state())
    first, second = fitted(test, hidden), restored(test, hidden)
    for name in first:
        assert torch.allclose(first[name], second[name]), name
