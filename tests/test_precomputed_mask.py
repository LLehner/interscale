"""Hidden cells' precomputed embeddings never reach the model (`precomputed_mask_fill`; global_plan.md).

Embeddings computed outside InterScale (scVI, PCA, ...) reach a GlobalModel through
`type_gex_embedding: Precomputed`. Masking corrupts `batch.x` only, so a hidden cell used to hand
the transformer its own embedding -- a compressed copy of the expression it is scored on. Pinned
here, through a GlobalModel built from a config: the hidden rows are replaced by the configured
fill (a learned vector, or -1 everywhere) and never reach a prediction, while visible rows do;
the learned fill trains; a clean pass leaves every embedding as it is; the rest of the model is
initialised identically under both fills; a constant fill leaves the input LayerNorm as its bias;
the config and the module refuse gene masking and unknown fills; and the checkpoint name tells
the fills apart.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import Batch, Data

E, G, N = 8, 6, 12


def _cfg(fill="learned", strategy="node"):
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = E
    params = cfg.model.global_component.parameters
    params.max_seq_len, params.n_heads = 64, 2
    params.type_gex_embedding = "Precomputed"
    params.precomputed_mask_fill = fill
    cfg.model.global_component.latent_obsm_key = "X_scvi"
    cfg.dataset.mask_strategy = strategy
    cfg.dataset.mask_percentage = 0.3
    return cfg


def _model(fill="learned"):
    import pandas as pd
    from anndata import AnnData

    from interscale.model.global_model import GlobalModel

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(N, G)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * N)
    adata.obs["split"] = pd.Categorical(["train"] * (N - 2) + ["val"] * 2)
    GlobalModel._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    torch.manual_seed(0)
    return GlobalModel(adata, cfg=_cfg(fill))


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    d = Data(x=torch.rand(N, G, generator=gen) + 1.0, edge_index=torch.zeros((2, 0), dtype=torch.long))
    d.embeddings = torch.randn(N, E, generator=gen)
    d.mask = torch.zeros(N, dtype=torch.bool)
    d.mask[:4] = True
    batch = Batch.from_data_list([d])
    batch.obs_names = torch.arange(N)
    return batch


def _predict(module, batch):
    with torch.no_grad():
        return module._common_step(batch, "regression", "node").y_pred


@pytest.mark.parametrize("fill", ["learned", "minus_one"])
def test_hidden_cells_embeddings_never_reach_a_prediction(fill):
    module = _model(fill).module.eval()
    assert module.precomputed_mask_fill == fill  # from the config, through GlobalModel
    batch = _batch()
    before = _predict(module, batch)

    # Random, not constant, shifts: the input LayerNorm removes a constant added to every entry.
    shift = torch.randn(N, E, generator=torch.Generator().manual_seed(1)) * 3.0
    hidden = batch.clone()
    hidden.embeddings = hidden.embeddings.clone()
    hidden.embeddings[:4] += shift[:4]  # the hidden cells only
    visible = batch.clone()
    visible.embeddings = visible.embeddings.clone()
    visible.embeddings[6] += shift[6]
    assert torch.allclose(_predict(module, hidden), before)
    assert not torch.allclose(_predict(module, visible), before)


def test_the_learned_fill_replaces_hidden_rows_and_trains():
    module = _model("learned").module
    batch = _batch()
    tokens = module.gex_embedding(batch, hidden=batch.mask)
    assert torch.equal(tokens[:4], module.mask_embedding.detach().expand(4, -1))
    assert torch.equal(tokens[4:], batch.embeddings[4:])
    module._common_step(batch, "regression", "node").y_pred.pow(2).mean().backward()
    assert module.mask_embedding.grad is not None and torch.count_nonzero(module.mask_embedding.grad) > 0


def test_the_minus_one_fill_sets_every_entry_and_adds_no_parameter():
    module = _model("minus_one").module
    batch = _batch()
    tokens = module.gex_embedding(batch, hidden=batch.mask)
    assert torch.equal(tokens[:4], torch.full((4, E), -1.0))
    assert torch.equal(tokens[4:], batch.embeddings[4:])
    assert not hasattr(module, "mask_embedding")


@pytest.mark.parametrize("fill", ["learned", "minus_one"])
def test_a_clean_pass_leaves_every_embedding_as_it_is(fill):
    module = _model(fill).module
    batch = _batch()
    assert torch.equal(module.gex_embedding(batch), batch.embeddings)  # inference: no marker
    nothing = torch.zeros(N, dtype=torch.bool)
    assert torch.equal(module.gex_embedding(batch, hidden=nothing), batch.embeddings)


def test_both_fills_initialise_the_rest_of_the_model_identically():
    state = torch.get_rng_state()
    learned = _model("learned").module.state_dict()
    after_learned = torch.get_rng_state()
    torch.set_rng_state(state)
    minus_one = _model("minus_one").module.state_dict()
    assert torch.equal(after_learned, torch.get_rng_state())
    assert set(learned) - set(minus_one) == {"mask_embedding"}
    assert all(torch.equal(minus_one[k], learned[k]) for k in minus_one)


def test_a_constant_fill_comes_out_of_the_input_layernorm_as_its_bias():
    """Why -1 needs no out-of-range value here: any constant vector normalises to the bias."""
    module = _model("minus_one").module
    with torch.no_grad():
        module.norm_input.bias.normal_()
        out = module.norm_input(torch.full((1, E), -1.0))
        other = module.norm_input(torch.full((1, E), 3.0))
    assert torch.allclose(out[0], module.norm_input.bias, atol=1e-5)
    assert torch.allclose(out, other, atol=1e-5)


@pytest.mark.parametrize(
    ("change", "match"),
    [("fill", "precomputed_mask_fill"), ("gene", "mask_strategy gene"), ("obsm", "latent_obsm_key")],
)
def test_the_config_refuses_what_would_leak_or_cannot_be_built(change, match):
    from interscale.config import _validate

    _validate(_cfg())
    cfg = _cfg()
    if change == "fill":
        cfg.model.global_component.parameters.precomputed_mask_fill = "zeros"
    elif change == "gene":
        cfg.dataset.mask_strategy = "gene"
    else:
        cfg.model.global_component.latent_obsm_key = None
    with pytest.raises(ValueError, match=match):
        _validate(cfg)


def test_the_module_refuses_gene_masking_too():
    from interscale.module.base import GlobalModule

    with pytest.raises(ValueError, match="single genes"):
        GlobalModule.from_config(
            _cfg(strategy="gene"),
            local_mask_hops=0,
            n_input=G,
            n_output=G,
            n_embed=E,
            decoder_type="linear",
            dropout_decoder=0.0,
            decoder_hidden_dims=[8],
            mask_percentage=0.3,
            mask_strategy="gene",
            type_gex_embedding="Precomputed",
        )


def test_the_checkpoint_name_tells_the_fills_apart():
    from interscale.tl.utils import get_model_filename_prefix

    names = {}
    for fill in ("learned", "minus_one"):
        cfg = _cfg(fill)
        cfg.dataset.name = "d"
        names[fill] = get_model_filename_prefix(cfg, False, True)
    assert names["learned"].endswith("self-attn-transformer_emb-pre-Xscvi-learned_")
    assert names["minus_one"].endswith("self-attn-transformer_emb-pre-Xscvi-m1_")
    cfg = _cfg("learned")  # another embedding, another checkpoint
    cfg.dataset.name = "d"
    cfg.model.global_component.latent_obsm_key = "X_pca"
    assert get_model_filename_prefix(cfg, False, True).endswith("_emb-pre-Xpca-learned_")
