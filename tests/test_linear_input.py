"""The learned linear input of a global-only model (`type_gex_embedding: Linear`; global_plan.md, A1).

Before it, a GlobalModel's tokens were a PCA fitted once and frozen, NMF, or precomputed: nothing
the transformer could shape. Pinned here: the layer is built and trained with the transformer;
the masked input is what it maps, so hidden values never reach a prediction; inference maps the
clean input through the same layer; the config accepts only inputs the module can build; and only
the global-only checkpoint name carries it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

G = 6


def _cfg(embedding="Linear"):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.global_component.parameters.type_gex_embedding = embedding
    return cfg


def _module(rate=0.3):
    from interscale.module.base import GlobalModule

    torch.manual_seed(0)
    return GlobalModule.from_config(
        _cfg(),
        local_mask_hops=0,
        n_input=G,
        n_output=G,
        n_embed=16,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=rate,
        mask_strategy="node",
        type_gex_embedding="Linear",
    )


def _batch(n=12, seed=0):
    gen = torch.Generator().manual_seed(seed)
    d = Data(x=torch.rand(n, G, generator=gen) + 1.0, edge_index=torch.zeros((2, 0), dtype=torch.long))
    d.mask = torch.zeros(n, dtype=torch.bool)
    d.mask[:4] = True
    batch = Batch.from_data_list([d])
    batch.obs_names = torch.arange(n)
    return batch


def test_the_layer_is_trained_with_the_transformer():
    module = _module()
    assert isinstance(module.input_embedding, torch.nn.Linear)
    out = module._common_step(_batch(), "regression", "node")
    out.y_pred.pow(2).mean().backward()
    assert module.input_embedding.weight.grad is not None
    assert torch.count_nonzero(module.input_embedding.weight.grad) > 0


def test_the_layer_maps_the_masked_input_so_hidden_values_never_reach_a_prediction():
    module = _module().eval()
    batch = _batch()
    with torch.no_grad():
        before = module._common_step(batch, "regression", "node").y_pred
        changed = batch.clone()
        changed.x = changed.x.clone()
        changed.x[:4] += 5.0  # the hidden cells only
        after = module._common_step(changed, "regression", "node").y_pred
        # ... while a visible cell's value does reach the predictions.
        visible = batch.clone()
        visible.x = visible.x.clone()
        visible.x[5] += 5.0
        moved = module._common_step(visible, "regression", "node").y_pred
    assert torch.allclose(before, after)
    assert not torch.allclose(before, moved)


def test_inference_maps_the_clean_input_through_the_same_layer(monkeypatch):
    import pandas as pd
    from anndata import AnnData

    import interscale.model.global_model as global_model
    from interscale.model.global_model import GlobalModel

    n = 10
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(n, G)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * n)
    adata.obs["split"] = pd.Categorical(["train"] * (n - 2) + ["val"] * 2)
    adata.obsm["spatial"] = rng.random((n, 2))
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(0)
    model = GlobalModel(adata, cfg=_cfg())
    model.module.eval()
    model.is_trained_ = True

    graph = Data(
        x=torch.tensor(adata.X), edge_index=torch.zeros((2, 0), dtype=torch.long), obs_names=torch.arange(1, n + 1)
    )
    monkeypatch.setattr(global_model, "prepare_a2d_dataset", lambda cfg: lambda adata: [[graph], None])
    monkeypatch.setattr(global_model, "attach_positional_inputs", lambda pyg, cfg: None)
    saved = {}
    monkeypatch.setattr(model, "save_evaluation_results", lambda adata, prefix, **out: saved.update(out) or adata)
    model.get_model_output(adata.copy())

    module = model.module
    tokens = module.input_embedding(graph.x)
    _, emb, padding, _, _ = module.evaluate(graph.clone(), tokens)
    expected = module.predict(emb, padding, "node").detach().numpy()
    assert np.allclose(saved["y_pred_global_df"].to_numpy(dtype=float), expected, atol=1e-5)


def test_the_config_accepts_only_inputs_the_module_builds():
    from interscale.config import _validate

    _validate(_cfg("Linear"))
    _validate(_cfg("PCA"))
    with pytest.raises(ValueError, match="type_gex_embedding"):
        _validate(_cfg("scvi"))


def test_only_the_global_only_checkpoint_name_carries_it():
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.tl.utils import get_model_filename_prefix

    cfg = _cfg("Linear")
    cfg.dataset.name = "d"
    assert get_model_filename_prefix(cfg, False, True).endswith("self-attn-transformer_emb-linear_")
    cfg.model.global_component.parameters.type_gex_embedding = "PCA"
    assert get_model_filename_prefix(cfg, False, True).endswith("self-attn-transformer_")  # unchanged
    cfg.model.global_component.parameters.type_gex_embedding = "Linear"
    cfg.model.local_component.name = "GCN"
    cfg = get_local_component_cfg(cfg, "GCN")
    assert "emb-" not in get_model_filename_prefix(cfg, True, True)  # a CombinedModel ignores it
