"""Reference predictors on hidden cells (`masked_probing.reference_predictions`; global_plan.md, 0b).

The bar a model head must clear: averages of the visible entries of the same graph. Pinned here:
each reference is exact on the field it is built for (a constant per slide for the slide mean, a
smooth field for the kernel and the nearest cells) and scores ~0 on noise; graphs never mix; no
hidden entry ever enters an average, per entry under gene masking; a cell never averages itself;
far-away cells do not overflow the kernel; and through `collect_head_predictions` every reference
row is its own cell's, also on graphs the transformer subsamples.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interscale.evaluation.masked_probing import reference_predictions
from interscale.tl.masking import sample_block_mask


def _lattice(side, offset=0.0):
    ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=1).double() + offset


def _refs(x, hidden, pos, graph=None, **kwargs):
    graph = torch.zeros(len(x), dtype=torch.long) if graph is None else graph
    kwargs = {"spatial_unit_um": 100.0, "kernels_um": (150.0,), "k_nearest": (4,)} | kwargs
    return reference_predictions(x, hidden, pos, graph, **kwargs)


def _discs(pos, seed=0):
    return sample_block_mask(pos, 2.5, 0.3, generator=torch.Generator().manual_seed(seed))


def test_a_constant_per_slide_is_reproduced_and_slides_never_mix():
    pos = torch.cat([_lattice(12), _lattice(12)])  # same coordinates: only the graph index separates them
    graph = torch.repeat_interleave(torch.arange(2), 144)
    level = torch.tensor([[3.0, 1.0], [7.0, 2.0]])
    x = level[graph]
    hidden = torch.cat([_discs(pos[:144], 0), _discs(pos[144:], 1)])[:, None].expand(-1, 2)
    for name, pred in _refs(x, hidden, pos, graph).items():
        assert torch.allclose(pred, x.double()), name


def test_a_smooth_field_favours_the_local_averages_and_noise_scores_zero():
    pos = _lattice(30)
    hid = _discs(pos)
    hidden = hid[:, None].expand(-1, 3)
    smooth = torch.stack([pos[:, 0], pos[:, 1], pos[:, 0] + pos[:, 1]], dim=1)
    refs = _refs(smooth, hidden, pos)

    def error(name):
        return float((refs[name][hid] - smooth[hid]).pow(2).mean())

    assert error("ref_kernel_150um") < 0.2 * error("ref_slide_mean")
    assert error("ref_nearest_4") < 0.2 * error("ref_slide_mean")

    noise = torch.randn(len(pos), 50, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    for name, pred in _refs(noise, hid[:, None].expand(-1, 50), pos).items():
        p, t = pred[hid], noise[hid]
        cos = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1))
        assert abs(float(cos.mean())) < 0.1, name


def test_no_hidden_entry_ever_enters_an_average():
    pos = _lattice(15)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(len(pos), 4, generator=gen, dtype=torch.float64)
    hidden = torch.rand(len(pos), 4, generator=gen) < 0.3  # entry masking: per gene
    hidden[_discs(pos)] = True  # and some whole cells
    labels = (pos[:, 0] > 7).long()
    base = _refs(x, hidden, pos, labels=labels)
    changed = x.clone()
    changed[hidden] += 100.0
    again = _refs(changed, hidden, pos, labels=labels)
    for name in base:
        assert torch.allclose(base[name], again[name]), name


def test_entries_count_per_gene_and_a_cell_never_averages_itself():
    """Three cells on a line. Gene 0 of the middle cell is hidden, gene 1 is not: a wide kernel
    reads gene 0 from the far cell only, gene 1 from both; and nothing reads cell 0 itself."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    x = torch.tensor([[100.0, 100.0], [2.0, 4.0], [6.0, 8.0]], dtype=torch.float64)
    hidden = torch.tensor([[True, True], [True, False], [False, False]])
    refs = _refs(x, hidden, pos, kernels_um=(1e6,), k_nearest=(1,))
    assert torch.allclose(refs["ref_kernel_1e+06um"][0], torch.tensor([6.0, 6.0], dtype=torch.float64))
    assert torch.allclose(refs["ref_slide_mean"][0], torch.tensor([6.0, 6.0], dtype=torch.float64))
    # The nearest open cell is the middle one; its gene 0 is hidden, so gene 0 falls back to the slide mean.
    assert torch.allclose(refs["ref_nearest_1"][0], torch.tensor([6.0, 4.0], dtype=torch.float64))


def test_the_nearest_cells_skip_wholly_hidden_ones_and_the_label_mean_keeps_to_its_label():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    x = torch.tensor([[0.0], [10.0], [20.0], [30.0]], dtype=torch.float64)
    hidden = torch.tensor([[True], [True], [False], [False]])
    labels = torch.tensor([0, 1, 0, 1])
    refs = _refs(x, hidden, pos, k_nearest=(1,), labels=labels)
    assert float(refs["ref_nearest_1"][0, 0]) == 20.0  # cell 1 is wholly hidden: skipped
    assert float(refs["ref_label_mean"][0, 0]) == 20.0  # label 0: cell 2 only
    assert float(refs["ref_label_mean"][1, 0]) == 30.0  # label 1: cell 3 only


def test_far_away_cells_do_not_overflow_the_kernel():
    pos = torch.tensor([[0.0, 0.0], [50.0, 0.0], [51.0, 0.0]], dtype=torch.float64)  # 5 mm apart
    x = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float64)
    hidden = torch.tensor([[True], [False], [False]])
    refs = _refs(x, hidden, pos, kernels_um=(50.0,))
    assert torch.isfinite(refs["ref_kernel_50um"]).all()
    assert float(refs["ref_kernel_50um"][0, 0]) == pytest.approx(2.0)  # the nearest visible cell dominates


def test_through_the_collection_every_reference_row_is_its_own_cell():
    """On graphs the transformer subsamples, rows are a subset; each row's references must still be
    its own cell's, found here by matching the row's true values to a node."""
    import random

    from torch_geometric.data import Data

    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.evaluation.masked_probing import collect_head_predictions, draw_masks
    from interscale.module.base import GlobalModule

    datas = []
    for side, seed in ((10, 0), (7, 1)):
        pos = _lattice(side).float()
        d = Data(x=torch.rand(side * side, 5, generator=torch.Generator().manual_seed(seed)) + 1.0, pos=pos)
        d.edge_index = torch.zeros((2, 0), dtype=torch.long)
        d.obs_names = torch.arange(side * side)
        datas.append(d)
    draw_masks(datas, "blocks", 0.3, seed=0, radius=2.0)

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 40
    torch.manual_seed(0)
    module = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=5,
        n_output=5,
        n_embed=16,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.0,
        mask_strategy="node",
        block_masking=True,
        type_gex_embedding="Linear",
    )
    options = {"kernels_um": (150.0,), "k_nearest": (4,)}
    random.seed(0)
    pred = collect_head_predictions(
        module,
        datas,
        prediction_task="regression",
        prediction_level="node",
        mask_strategy="node",
        batch_size=2,
        device=torch.device("cpu"),
        spatial_unit_um=100.0,
        references=options,
    )
    assert len(pred.true) == 80
    assert {"global", "ref_slide_mean", "ref_kernel_150um", "ref_nearest_4"} <= set(pred.heads)
    for graph, d in enumerate(datas):
        hidden = d.mask[:, None].expand_as(d.x)
        direct = reference_predictions(
            d.x, hidden, d.pos, torch.zeros(d.num_nodes, dtype=torch.long), spatial_unit_um=100.0, **options
        )
        rows = slice(40 * graph, 40 * graph + 40)
        node = [int(torch.where((d.x == torch.tensor(t, dtype=d.x.dtype)).all(1))[0]) for t in pred.true[rows]]
        for name, value in direct.items():
            assert np.allclose(pred.heads[name][rows], value[node].numpy()), name


def test_a_partly_visible_cell_does_not_spend_a_neighbour_slot_on_itself():
    """Under gene masking a cell keeps some entries visible; its hidden ones must still be read from
    its k nearest *other* cells."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=torch.float64)
    x = torch.tensor([[100.0, 100.0], [2.0, 4.0], [6.0, 8.0]], dtype=torch.float64)
    hidden = torch.tensor([[True, False], [False, False], [False, False]])
    refs = _refs(x, hidden, pos, k_nearest=(1,))
    assert float(refs["ref_nearest_1"][0, 0]) == 2.0  # cell 1, not cell 0 itself (nor the slide mean, 4)
