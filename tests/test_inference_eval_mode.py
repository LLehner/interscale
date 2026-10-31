"""``get_model_output`` reads the model with dropout off (``model.base._base_model.in_eval_mode``), and puts the module's mode back.

Before, a model read after ``train()`` -- its module left in training mode -- predicted with dropout on: another output every call.
"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _vk():
    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(vk, kind):
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel
    from interscale.model.local_model import LocalModel

    cfg = vk._config("dense", visible=False)
    cfg.model.global_component.parameters.dropout_global = 0.5
    cls = GlobalModel
    if kind != "global":
        cfg.model.local_component.name = "GCN"
        cfg = get_local_component_cfg(cfg, "GCN")
        cfg.model.local_component.parameters.dropout_local = 0.5
        cls = LocalModel if kind == "local" else CombinedModel
        cfg.model.decoder.dual_decoder = kind == "combined"
    adata = vk._build.__globals__["AnnData"](X=np.random.default_rng(0).random((36, vk.G)).astype(np.float32))
    adata.var_names = [f"g{k}" for k in range(vk.G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 36)
    adata.obs["split"] = pd.Categorical(["train"] * 34 + ["val"] * 2)
    adata.obsm["spatial"] = np.random.default_rng(1).uniform(0, 500, (36, 2))
    cls._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(0)
    net = cls(adata, cfg=cfg)
    net.module.train()  # as Lightning leaves it after fit
    net.is_trained_ = True
    return net, adata


@pytest.mark.parametrize("kind", ["global", "local", "combined"])
def test_two_reads_agree_and_the_module_stays_in_its_mode(kind, monkeypatch):
    import interscale.model.combined_model as combined_model
    import interscale.model.global_model as global_model
    import interscale.model.local_model as local_model

    vk = _vk()
    net, adata = _model(vk, kind)
    graph = vk._lattice(6, seed=6)
    graph.obs_names = torch.arange(1, 37)
    graph.x = torch.as_tensor(adata.X)
    where = {"global": global_model, "local": local_model, "combined": combined_model}[kind]
    monkeypatch.setattr(where, "prepare_a2d_dataset", lambda cfg: lambda a: [[graph.clone()], None])
    if hasattr(where, "attach_positional_inputs"):
        monkeypatch.setattr(where, "attach_positional_inputs", lambda pyg, cfg: None)
    key = "y_pred_local_df" if kind == "local" else "y_pred_global_df"
    reads = []
    for _ in range(2):
        saved = {}
        monkeypatch.setattr(net, "save_evaluation_results", lambda a, p, saved=saved, **out: saved.update(out) or a)
        random.seed(0)
        net.get_model_output(adata.copy())
        reads.append(saved[key].to_numpy(float))
    assert np.array_equal(reads[0], reads[1])  # dropout off: one function
    assert net.module.training  # the mode it was in

    def broken(cfg):
        raise RuntimeError("no graphs")

    monkeypatch.setattr(where, "prepare_a2d_dataset", broken)
    with pytest.raises(RuntimeError, match="no graphs"):
        net.get_model_output(adata.copy())
    assert net.module.training  # put back also when the read fails
