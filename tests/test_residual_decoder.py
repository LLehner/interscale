"""The global head predicting what the local head misses (`model.decoder.global_residual`; round 3).

Under plain dual decoding both heads reconstruct the same target, so copying the local head is
already optimal for the global one. Pinned here: in residual mode the global prediction is the
local prediction plus the global decoder's output, for the same cells in the same order; no
gradient of the global loss reaches the local prediction; the config requires the dual decoder at
node level; and the checkpoint name carries it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

G = 8


def _cfg(residual):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder = True
    cfg.model.decoder.global_residual = residual
    return cfg


def _module(residual, seed=0):
    from interscale.module.combined_module import DualDecoderCombinedModule

    torch.manual_seed(seed)
    return DualDecoderCombinedModule(
        cfg=_cfg(residual),
        n_input=G,
        n_output=G,
        n_embed=16,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="gene",
    ).eval()


def _batch(sizes=(9, 12), seed=0):
    rng = np.random.default_rng(seed)
    datas = []
    for n in sizes:
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        d = Data(x=torch.tensor(rng.random((n, G)), dtype=torch.float32), edge_index=torch.tensor([src, dst]))
        d.gene_mask = torch.tensor(rng.random((n, G)) < 0.3)
        d.gene_mask[:, 0] = True
        d.mask = d.gene_mask.any(dim=1)
        datas.append(d)
    batch = Batch.from_data_list(datas)
    batch.obs_names = torch.arange(batch.num_nodes)
    return batch


def test_the_global_prediction_is_the_local_one_plus_the_global_decoders_output():
    plain, residual = _module(False), _module(True)  # the same weights at one seed
    batch = _batch()
    off = plain._common_step(batch, "regression", "node")
    on = residual._common_step(batch, "regression", "node")

    n = len(off.y_pred) // 2
    local_off, global_off = off.y_pred[:n], off.y_pred[n:]
    local_on, global_on = on.y_pred[:n], on.y_pred[n:]
    assert torch.allclose(local_on, local_off, atol=1e-6)  # the local head is untouched
    assert torch.allclose(global_on, local_off + global_off, atol=1e-5)
    assert torch.equal(on.y_true, off.y_true)


def test_no_gradient_of_the_global_loss_reaches_the_local_prediction():
    module = _module(True)
    out = module._common_step(_batch(), "regression", "node")
    n = len(out.y_pred) // 2
    out.y_pred[n:].pow(2).mean().backward()  # the global head's loss alone

    local_decoder = [p.grad for p in module.local_module.decoder.parameters()]
    global_decoder = [p.grad for p in module.global_module.decoder.parameters()]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in local_decoder)
    assert all(g is not None and torch.count_nonzero(g) > 0 for g in global_decoder)


def test_the_combination_is_the_one_inference_uses():
    module = _module(True)
    local, raw = torch.randn(5, G), torch.randn(5, G)
    assert torch.allclose(module.combine_global(local, raw), local + raw)
    assert torch.equal(_module(False).combine_global(local, raw), raw)


@pytest.mark.parametrize(
    ("key", "value", "match"), [("dual", False, "dual_decoder"), ("level", "graph", "prediction_level")]
)
def test_the_config_requires_the_dual_decoder_at_node_level(key, value, match):
    from interscale.config import _validate_global_residual

    cfg = _cfg(True)
    _validate_global_residual(cfg)
    if key == "dual":
        cfg.model.decoder.dual_decoder = value
    else:
        cfg.dataset.prediction_level = value
    with pytest.raises(ValueError, match=match):
        _validate_global_residual(cfg)


def test_config_loading_runs_the_check():
    from interscale.config import _validate

    cfg = _cfg(True)
    cfg.model.decoder.dual_decoder = False
    with pytest.raises(ValueError, match="dual_decoder"):
        _validate(cfg)


def test_the_checkpoint_name_carries_it():
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _cfg(True)
    cfg.dataset.name = "d"
    assert get_model_filename_prefix(cfg, True, True).endswith("var-res_")
    cfg.optim.center_targets = True
    assert get_model_filename_prefix(cfg, True, True).endswith("var-ctr+res_")


def test_inference_reports_the_same_sum_for_a_subsampled_slide(monkeypatch):
    """`get_model_output` decodes outside `_common_step`; a slide longer than the sequence keeps a
    subset of its cells, and each kept cell's global prediction is its own local one plus the global
    decoder's output."""
    import random

    import pandas as pd
    from anndata import AnnData

    import interscale.model.combined_model as combined_model
    from interscale.model.combined_model import CombinedModel

    n, genes = 12, 4
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(n, genes)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.random((n, 2))
    CombinedModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    cfg = _cfg(True)
    cfg.model.global_component.parameters.max_seq_len = 8  # fewer tokens than cells
    torch.manual_seed(0)
    model = CombinedModel(adata, cfg=cfg)
    model.module.eval()
    model.is_trained_ = True

    graph = Data(
        x=torch.tensor(adata.X),
        edge_index=torch.tensor([list(range(n - 1)), list(range(1, n))]),
        obs_names=torch.arange(1, n + 1),
    )
    monkeypatch.setattr(combined_model, "prepare_a2d_dataset", lambda cfg: lambda adata: [[graph], None])
    monkeypatch.setattr(combined_model, "attach_positional_inputs", lambda pyg, cfg: None)
    saved = {}
    monkeypatch.setattr(model, "save_evaluation_results", lambda adata, prefix, **out: saved.update(out) or adata)

    preds = {}
    for residual in (False, True):
        model.module.global_residual = residual
        random.seed(0)  # the same subsample both times
        model.get_model_output(adata.copy())
        preds[residual] = saved["y_pred_global_df"].to_numpy(dtype=float)
    local = saved["y_pred_local_df"].to_numpy(dtype=float)

    kept = ~np.isnan(preds[True]).any(axis=1)
    assert kept.sum() == 8
    assert np.allclose(preds[True][kept], local[kept] + preds[False][kept], atol=1e-5)
