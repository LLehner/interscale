"""Inference without dense maps (``TransformerNodeEncoderHook.encode``, ``get_model_output(attention=False)``; roadmap Tier 3, item 8).

Pinned here:

* ``encode`` is :meth:`evaluate`'s clean pass without the relevance -- the same tokens out -- for dense attention, equal on the fused
  attention, and on a routed hierarchy (which ``evaluate`` cannot read) it forms no tensor over the cells' pairs;
* ``get_model_output(attention=False)`` writes the same embeddings and predictions as with the attention outputs, and neither the
  ``[cells, max_seq_len]`` table nor the CLS scores, for a GlobalModel and a CombinedModel.
"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode
from torch_geometric.data import Data


def _vk():
    spec = importlib.util.spec_from_file_location("vk", Path(__file__).with_name("test_visible_keys_backends.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph(vk, side=6):
    data = vk._lattice(side, seed=side)
    data.obs_names = torch.arange(1, data.num_nodes + 1)
    return data


def _encode(module, graph):
    with torch.no_grad():
        random.seed(0)
        return module.encode(graph.clone(), module.gex_embedding(graph))


def test_encode_is_the_clean_pass_of_evaluate_without_the_relevance():
    vk = _vk()
    # visible_keys_only and cells marked hidden: a masked pass would block their keys, the clean pass must not.
    module = vk._stir(vk._build(vk._config("dense", visible=True))).eval()
    graph = _graph(vk)
    graph.mask[:10] = True
    random.seed(0)
    _, out_eval, padding_eval, kept_eval, relevance = module.evaluate(graph.clone(), module.gex_embedding(graph))
    _, out_enc, padding_enc, kept_enc = _encode(module, graph)
    assert torch.allclose(out_enc, out_eval.detach(), atol=1e-6)
    assert torch.equal(padding_enc, padding_eval) and kept_enc == kept_eval
    assert relevance.shape[0] == graph.num_nodes + 1  # evaluate's S x S relevance; encode hands none back
    assert all(layer.get_attn_output_weights() is None for layer in module.transformer_encoder.layers)  # no map kept


@pytest.mark.parametrize("chunk_pairs", [2**24, 37 * 2], ids=["one block", "blocks"])
def test_the_fused_attention_encodes_as_dense_does(chunk_pairs):
    vk = _vk()
    dense = vk._stir(vk._build(vk._config("dense", visible=False))).eval()
    fused = vk._build(vk._config("fused", visible=False, chunk_pairs=chunk_pairs)).eval()
    fused.load_state_dict(dense.state_dict())
    graph = _graph(vk)
    assert torch.allclose(_encode(fused, graph)[1], _encode(dense, graph)[1], atol=1e-5)


class _Shapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(t, torch.Tensor):
                self.shapes.append(tuple(t.shape))
        return out


def test_a_routed_hierarchy_encodes_a_section_without_a_matrix_over_its_cells():
    vk = _vk()
    hierarchy = {"prior_mode": "soft", "levels": [3, 12], "readout": "concat", "candidates.mode": "routed", "candidates.topk": [2]}
    module = vk._stir(vk._build(vk._config("dense", visible=False, hierarchy=hierarchy, max_seq_len=400))).eval()
    graph = _graph(vk, side=20)  # 400 cells: more than any width of the model
    n = graph.num_nodes
    with _Shapes() as seen:
        _, out, _, kept = _encode(module, graph)
    assert out.shape[0] == n + 1 and len(kept[0]) == n
    assert not [s for s in seen.shapes if sum(d >= n for d in s) >= 2]  # nothing with two axes over the cells


def _trained(vk, kind):
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.model.combined_model import CombinedModel
    from interscale.model.global_model import GlobalModel

    cfg = vk._config("dense", visible=False)
    cls = GlobalModel
    if kind == "combined":
        cfg.model.local_component.name = "GCN"
        cfg = get_local_component_cfg(cfg, "GCN")
        cfg.model.decoder.dual_decoder = True
        cls = CombinedModel
    adata = vk._build.__globals__["AnnData"](X=np.random.default_rng(0).random((36, vk.G)).astype(np.float32))
    import pandas as pd

    adata.var_names = [f"g{k}" for k in range(vk.G)]
    adata.obs["sample"] = pd.Categorical(["s1"] * 36)
    adata.obs["split"] = pd.Categorical(["train"] * 34 + ["val"] * 2)
    adata.obsm["spatial"] = np.random.default_rng(1).uniform(0, 500, (36, 2))
    cls._setup_anndata(adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False)
    torch.manual_seed(0)
    net = cls(adata, cfg=cfg)
    vk._stir(net.module).eval()  # no dropout: the two calls compare one function
    net.is_trained_ = True
    return net, adata


@pytest.mark.parametrize("kind", ["global", "combined"])
def test_get_model_output_without_attention_writes_the_same_embeddings_and_predictions_and_no_maps(kind, monkeypatch):
    import interscale.model.combined_model as combined_model
    import interscale.model.global_model as global_model

    vk = _vk()
    net, adata = _trained(vk, kind)
    graph = _graph(vk)
    graph.x = torch.as_tensor(adata.X)
    where = global_model if kind == "global" else combined_model
    monkeypatch.setattr(where, "prepare_a2d_dataset", lambda cfg: lambda a: [[graph.clone()], None])
    monkeypatch.setattr(where, "attach_positional_inputs", lambda pyg, cfg: None)
    outputs = {}
    for attention in (True, False):
        saved = {}
        monkeypatch.setattr(net, "save_evaluation_results", lambda a, p, saved=saved, **out: saved.update(out) or a)
        random.seed(0)
        net.get_model_output(adata.copy(), attention=attention)
        outputs[attention] = saved
    with_maps, without = outputs[True], outputs[False]
    assert with_maps["attention_matrix_df"] is not None and with_maps["cls_token_horizontal"] is not None
    assert without["attention_matrix_df"] is None and without["cls_token_horizontal"] is None
    assert without["cls_token_vertical"] is None
    for key in ("global_embeddings_df", "y_pred_global_df") + (("y_pred_local_df",) if kind == "combined" else ()):
        assert np.allclose(without[key].to_numpy(float), with_maps[key].to_numpy(float), atol=1e-5), key
