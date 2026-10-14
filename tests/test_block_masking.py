"""Spatial blocks: whole cells hidden in discs (`dataset.mask_block_fraction`; attention_plan.md, round 3).

Cell and gene masking both leave a hidden entry's neighbourhood visible, so the GCN alone can fill
it in and the global component is never needed. Pinned here: each disc is every cell within the
radius of a centre cell, drawn until the fraction is covered; the discs join either strategy's mask
(masked cells, or whole masked rows) and are redrawn with it, the val mask staying fixed; with the
Bernoulli rate at 0 under "node" the discs are the only targets; with no discs nothing changes and
no random number is drawn; the config needs coordinates, a node-level reconstruction and
consistent keys; the radius reaches the datamodule in the units of `pos`; the checkpoint name
carries it.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Data

from interscale.geome_dataloader import GraphAnnDataModule, block_mask_kwargs
from interscale.tl.masking import sample_block_mask, sample_gene_mask, sample_node_mask

SIDE = 30  # a 30 x 30 lattice of unit spacing


def _lattice(side=SIDE):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=1).float()


def _graph(side=SIDE, genes=5, pos=True):
    n = side * side
    d = Data(x=torch.rand(n, genes), edge_index=torch.zeros((2, 0), dtype=torch.long))
    if pos:
        d.pos = _lattice(side)
    return d


def _dm(strategy="node", pct=0.0, radius=3.0, fraction=0.3, n_train=2):
    dm = GraphAnnDataModule(
        datas=[[_graph() for _ in range(n_train)], [_graph()]],
        batch_size=1,
        num_workers=0,
        mask_percentage=pct,
        mask_strategy=strategy,
        mask_block_radius=radius,
        mask_block_fraction=fraction,
    )
    dm.setup(stage="fit")
    return dm


def _is_one_disc(mask, pos, radius):
    """Whether `mask` is exactly every cell within `radius` of one of the cells."""
    dist = torch.cdist(pos.double(), pos.double())
    return any(torch.equal(mask, dist[c] <= radius) for c in torch.where(mask)[0])


# --------------------------------------------------------------------------- the draw


def test_a_block_is_every_cell_within_the_radius_of_its_centre():
    pos = _lattice()
    for seed in range(5):
        # One cell to cover: exactly one disc is drawn.
        mask = sample_block_mask(pos, 3.0, 1 / len(pos), generator=torch.Generator().manual_seed(seed))
        assert _is_one_disc(mask, pos, 3.0)
        assert 1 < int(mask.sum()) <= 29  # a full disc of radius 3 on the lattice holds 29 cells


def test_discs_are_drawn_until_the_fraction_is_covered():
    pos = _lattice()
    for fraction in (0.1, 0.3, 0.6):
        mask = sample_block_mask(pos, 3.0, fraction, generator=torch.Generator().manual_seed(0))
        covered = float(mask.float().mean())
        assert fraction <= covered <= fraction + 29 / len(pos)


def test_blocks_hide_whole_neighbourhoods_and_a_bernoulli_mask_does_not():
    """The point of the discs: cells whose every neighbour within two hops is hidden as well."""
    pos = _lattice()
    within_two = torch.cdist(pos, pos) <= 2.0 * 2**0.5  # a 2-hop reach on the lattice with diagonals
    gen = torch.Generator().manual_seed(0)

    def blind(mask):
        return int((mask & (within_two & ~mask).sum(1).eq(0)).sum())

    blocks = sample_block_mask(pos, 5.0, 0.3, generator=gen)
    bernoulli = torch.rand(len(pos), generator=gen) < float(blocks.float().mean())
    assert blind(blocks) >= 20
    assert blind(bernoulli) == 0


# --------------------------------------------------------------------------- in the datamodule


def test_under_node_masking_with_rate_zero_the_discs_are_the_only_targets():
    dm = _dm("node", pct=0.0, fraction=1 / SIDE**2)  # one disc per graph
    for data in dm.train_data + dm.val_data:
        assert _is_one_disc(data.mask, data.pos, 3.0)


def test_under_node_masking_the_discs_join_the_bernoulli_cells():
    torch.manual_seed(0)
    dm = _dm("node", pct=0.2, fraction=0.3)
    covered = torch.stack([d.mask.float().mean() for d in dm.train_data + dm.val_data])
    # The union: about 0.3 + 0.2 * 0.7 = 0.44, well above either part alone.
    assert torch.all(covered > 0.38) and torch.all(covered < 0.52)


def test_under_gene_masking_every_entry_of_a_disc_cell_is_masked():
    torch.manual_seed(0)
    dm = _dm("gene", pct=0.2, fraction=0.3)
    for data in dm.train_data + dm.val_data:
        full = data.gene_mask.all(dim=1)
        assert 0.3 <= float(full.float().mean()) < 0.3 + 0.1  # the discs (rows full by chance: 0.2**5)
        assert torch.equal(data.mask, data.gene_mask.any(dim=1))
        assert bool(data.mask.all())  # every other cell keeps its Bernoulli entries
        partial = data.gene_mask[~full].float().mean()
        assert 0.15 < float(partial) < 0.3


def test_the_discs_are_redrawn_for_train_and_fixed_for_val():
    torch.manual_seed(0)
    dm = _dm("node", pct=0.0, fraction=0.3)
    train, val = [d.mask.clone() for d in dm.train_data], [d.mask.clone() for d in dm.val_data]
    dm.resample_train_mask()
    assert all(not torch.equal(before, d.mask) for before, d in zip(train, dm.train_data))
    assert all(torch.equal(before, d.mask) for before, d in zip(val, dm.val_data))


@pytest.mark.parametrize("strategy", ["node", "gene"])
def test_without_discs_the_masks_and_the_random_state_are_unchanged(strategy):
    """Exactly the strategy's own draws, graph by graph, train then val, and nothing else."""
    graphs = [_graph(), _graph(), _graph()]
    n, genes = graphs[0].num_nodes, graphs[0].x.shape[1]
    torch.manual_seed(0)
    if strategy == "gene":
        expected = [sample_gene_mask(n, genes, 0.3) for _ in graphs]
        expected = [(m.any(dim=1), m) for m in expected]
    else:
        expected = [(sample_node_mask(n, 0.3), None) for _ in graphs]
    expected_after = torch.rand(3)

    for blocks in ({}, {"mask_block_radius": 3.0, "mask_block_fraction": 0.0}):
        torch.manual_seed(0)
        dm = GraphAnnDataModule(
            datas=[graphs[:2], graphs[2:]],
            batch_size=1,
            num_workers=0,
            mask_percentage=0.3,
            mask_strategy=strategy,
            **blocks,
        )
        dm.setup(stage="fit")
        assert torch.equal(torch.rand(3), expected_after)
        for data, (mask, gene_mask) in zip(dm.train_data + dm.val_data, expected):
            assert torch.equal(data.mask, mask)
            assert gene_mask is None or torch.equal(data.gene_mask, gene_mask)


def test_a_graph_without_coordinates_is_refused():
    with pytest.raises(ValueError, match="no `pos`"):
        GraphAnnDataModule(
            datas=[[_graph(pos=False)], [_graph(pos=False)]],
            num_workers=0,
            mask_block_radius=3.0,
            mask_block_fraction=0.3,
        ).setup(stage="fit")


@pytest.mark.parametrize(("radius", "fraction"), [(0.0, 0.3), (3.0, 1.0), (3.0, -0.1)])
def test_the_datamodule_refuses_an_inconsistent_block(radius, fraction):
    with pytest.raises(ValueError, match="Spatial blocks need"):
        GraphAnnDataModule(datas=[[_graph()], [_graph()]], mask_block_radius=radius, mask_block_fraction=fraction)


# --------------------------------------------------------------------------- config


def _cfg(**dataset):
    from interscale.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.mask_block_radius_um, cfg.dataset.mask_block_fraction = 500.0, 0.3
    for key, value in dataset.items():
        setattr(cfg.dataset, key, value)
    return cfg


@pytest.mark.parametrize(
    ("dataset", "match"),
    [
        ({"mask_block_fraction": 0.0}, "no disc would be drawn"),
        ({"mask_block_fraction": 1.0}, r"in \(0, 1\)"),
        ({"mask_block_radius_um": 0.0}, "mask_block_radius_um > 0"),
        ({"spatial_key": None}, "spatial_key"),
        ({"prediction_level": "graph"}, "prediction_level node"),
        ({"prediction_task": "classification"}, "regression"),
        ({"spatial_unit_um": 0.0}, "spatial_unit_um"),
        ({"mask_strategy": "gene", "mask_percentage": 0.0}, "mask_strategy node"),
    ],
)
def test_the_config_refuses_blocks_it_cannot_draw(dataset, match):
    from interscale.config import _validate_block_mask

    _validate_block_mask(_cfg())
    with pytest.raises(ValueError, match=match):
        _validate_block_mask(_cfg(**dataset))


def test_the_rate_may_be_zero_only_with_discs_under_node_masking():
    from interscale.config import _validate

    _validate(_cfg(mask_percentage=0.0))  # the discs alone
    with pytest.raises(ValueError, match="mask_percentage"):
        _validate(_cfg(mask_percentage=0.0, mask_block_fraction=0.0, mask_block_radius_um=0.0))
    with pytest.raises(ValueError, match="mask_strategy node"):
        _validate(_cfg(mask_percentage=0.0, mask_strategy="gene"))


def test_config_loading_runs_the_check():
    from interscale.config import _validate

    with pytest.raises(ValueError, match="spatial_key"):
        _validate(_cfg(spatial_key=None))


def test_the_radius_reaches_the_datamodule_in_the_units_of_pos():
    assert block_mask_kwargs(_cfg(spatial_unit_um=0.5)) == {"mask_block_radius": 1000.0, "mask_block_fraction": 0.3}
    assert block_mask_kwargs(_cfg(mask_block_fraction=0.0)) == {}


def test_both_entrypoints_hand_the_blocks_to_the_datamodule(monkeypatch):
    import interscale
    import interscale.main as main_module
    import interscale.main_sweep as main_sweep_module

    seen = []

    class _Model:
        @classmethod
        def _setup_anndata(cls, **kwargs):
            pass

        def __init__(self, adata, cfg):
            pass

        def train(self, **kwargs):
            pass

    class _Run:
        def __init__(self, config):
            self.config, self.name, self.summary = config, "run", {}

    for module in (main_module, main_sweep_module):
        monkeypatch.setattr(module.sc, "read_h5ad", lambda path: object())
        monkeypatch.setattr(module, "remove_zero_expression_cells", lambda adata: adata)
        monkeypatch.setattr(module, "build_probe_obsm", lambda adata, cfg: False)
        monkeypatch.setattr(module, "prepare_geome_dataset", lambda adata, cfg: ([[], [], []], None))
        monkeypatch.setattr(module, "GraphAnnDataModule", lambda **kwargs: seen.append(kwargs) or object())
    monkeypatch.setattr(interscale.model, "CombinedModel", _Model)
    monkeypatch.setattr(main_sweep_module, "wandb", types.SimpleNamespace(init=lambda **kw: _Run({}), config={}))

    from interscale.config.global_component_config import get_global_component_cfg

    cfg = _cfg(spatial_unit_um=2.0)
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    main_module.main(cfg.clone(), "CombinedModel")
    cfg.wandb.use = True
    main_sweep_module.main_sweep(lambda: cfg.clone(), "CombinedModel", "hyperparmeter")

    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs["mask_block_radius"] == 250.0 and kwargs["mask_block_fraction"] == 0.3


def test_the_checkpoint_name_carries_the_blocks():
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _cfg()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.dataset.name = "d"
    assert get_model_filename_prefix(cfg, True, True).endswith("var-blk500f0.3_")
    cfg.optim.center_targets = True
    assert get_model_filename_prefix(cfg, True, True).endswith("var-ctr+blk500f0.3_")


# --------------------------------------------------------------------------- the model really hides them


def _hidden_values_reach_predictions(module):
    """Max change of the predictions when ONLY the hidden cells' true values change."""
    from torch_geometric.data import Batch

    graphs = []
    for seed in (0, 1):
        g = _graph(side=6, genes=8)
        g.x = g.x + seed
        g.obs_names = torch.arange(g.num_nodes)
        g.mask = sample_block_mask(g.pos, 1.5, 0.3, generator=torch.Generator().manual_seed(seed))
        graphs.append(g)
    batch = Batch.from_data_list(graphs)
    batch.obs_names = torch.cat([g.obs_names for g in graphs])
    module.eval()
    with torch.no_grad():
        before = module._common_step(batch, "regression", "node").y_pred
        changed = batch.clone()
        changed.x = changed.x.clone()
        hidden = changed.mask.bool()
        changed.x[hidden] = changed.x[hidden] + 5.0
        after = module._common_step(changed, "regression", "node").y_pred
    return float((before - after).abs().max())


def _block_module(rate, block_masking):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder = True
    torch.manual_seed(0)
    return DualDecoderCombinedModule(
        cfg=cfg,
        n_input=8,
        n_output=8,
        n_embed=16,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=rate,
        mask_strategy="node",
        block_masking=block_masking,
    )


def test_discs_alone_corrupt_the_input_of_a_module():
    """With the Bernoulli rate at 0 the discs are the only corruption. A module that read the rate
    alone passed the input through, and trained an autoencoder: hidden spots were reconstructed from
    their own values (SCE 0.037 against a gene-mean floor of 0.082)."""
    module = _block_module(0.0, block_masking=True)
    assert _hidden_values_reach_predictions(module) == 0.0
    # The submodules too: the transformer keeps the hidden cells when it subsamples a long slide.
    assert module.global_module.masked_nodes and module.local_module.masked_nodes
    assert _hidden_values_reach_predictions(_block_module(0.3, block_masking=False)) == 0.0
    # What the bug looked like: no rate, no flag, and the hidden values come straight through.
    assert _hidden_values_reach_predictions(_block_module(0.0, block_masking=False)) > 0.01


def test_a_model_built_from_a_config_with_discs_alone_hides_them():
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    from interscale.model.combined_model import CombinedModel

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(8, 8)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * 4 + ["s2"] * 4)
    adata.obs["split"] = pd.Categorical(["train"] * 6 + ["val"] * 2)
    CombinedModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = _cfg(mask_percentage=0.0)
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder = True
    model = CombinedModel(adata, cfg=cfg)
    assert model.module.masked_nodes and model.module.block_masking
    assert _hidden_values_reach_predictions(model.module) == 0.0

    cfg.dataset.mask_block_fraction, cfg.dataset.mask_block_radius_um = 0.0, 0.0
    cfg.dataset.mask_percentage = 0.3
    assert CombinedModel(adata, cfg=cfg).module.block_masking is False


def test_the_masks_are_redrawn_every_epoch_with_discs_alone():
    """The resample callback was only built for a rate > 0, leaving one fixed set of discs."""
    from interscale.train._utils import redraws_masks

    assert redraws_masks(_cfg(mask_percentage=0.0))
    assert redraws_masks(_cfg(mask_percentage=0.3, mask_block_fraction=0.0, mask_block_radius_um=0.0))
    assert not redraws_masks(_cfg(mask_percentage=0.0, mask_block_fraction=0.0, mask_block_radius_um=0.0))
