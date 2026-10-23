"""Orthogonality between the parts of a prediction, and within the global part: decoder_plan.md, D4.

``train.orthogonality`` (the measures) and the ``orth_*`` auxiliary losses (``train.aux_losses``). Pinned here:

* **the measures** -- each is its textbook definition written out, lies in [0, 1], is 1 for a copy and near 0 for
  independent data, and has the invariances it claims (linear CKA: scale and rotation; the canonical overlap: any
  invertible linear map; kernel CKA: scale); and the loopholes the plan names hold (an inflated unused direction lowers
  CKA but not the overlap; moving one part into the other's orthogonal complement zeroes OPL's term while one part still
  determines the other);
* **the plumbing** -- each term reads the cells the loss reads, centres each graph on its own, and subsamples from a
  generator of its own (the global ones untouched);
* **the models** -- each term accepts the models it reads and refuses the others with the reason, and through a real
  step reaches exactly the parameters it should; a term that reads cells refuses dropout on their path, with which
  training meets it by noise that evaluation removes;
* **inert by default** -- no term is built at weight 0, the module's state dict holds no back-reference, a deep copy's
  terms read the copy, and the checkpoint name carries every non-zero auxiliary weight;
* **the trainer** -- a dual additive model trains through ``train()`` with a term on, logs it, and ends with it lower
  than the same run without it.
"""

import copy
import math
import types

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scipy.linalg import subspace_angles
from scipy.spatial import cKDTree
from torch_geometric.data import Batch, Data

from interscale.config import get_cfg_defaults
from interscale.module.base import StepOutput, ViewOutput
from interscale.tl.utils import get_model_filename_prefix
from interscale.train.aux_losses import AUX_LOSSES, build_aux_losses
from interscale.train.orthogonality import (
    canonical_overlap,
    centre_by_group,
    cross_cosine,
    decorrelation,
    gene_correlation,
    kernel_cka,
    linear_cka,
    loading_orthogonality,
)

TERMS = (
    "orth_cka",
    "orth_cca",
    "orth_kernel_cka",
    "orth_opl",
    "orth_gene_corr",
    "orth_loadings",
    "orth_decorrelation",
    "orth_loading_ortho",
)
#: The terms that compare the dual model's two embeddings.
EMBEDDING_TERMS = ("orth_cka", "orth_cca", "orth_kernel_cka", "orth_opl")


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _centred(x):
    return x - x.mean(dim=0)


def _orthogonal(d, seed=0):
    q, _ = torch.linalg.qr(_randn(d, d, seed=seed))
    return q


# --------------------------------------------------------------------------- the measures


def test_each_group_is_centred_on_its_own():
    x = _randn(7, 3)
    groups = torch.tensor([0, 0, 1, 1, 1, 5, 5])
    c = centre_by_group(x, groups)
    for g in (0, 1, 5):
        assert torch.allclose(c[groups == g].mean(dim=0), torch.zeros(3, dtype=x.dtype), atol=1e-12)
    assert torch.allclose(centre_by_group(x).mean(dim=0), torch.zeros(3, dtype=x.dtype), atol=1e-12)


def test_linear_cka_is_its_textbook_formula_and_ignores_scale_and_rotation():
    x, y = _centred(_randn(200, 5)), _centred(_randn(200, 4, seed=1) + 0.5 * _randn(200, 5)[:, :4])
    xn, yn = x.numpy(), y.numpy()
    textbook = np.linalg.norm(xn.T @ yn) ** 2 / (np.linalg.norm(xn.T @ xn) * np.linalg.norm(yn.T @ yn))
    assert float(linear_cka(x, y)) == pytest.approx(textbook, rel=1e-10)
    assert float(linear_cka(3.0 * x @ _orthogonal(5), y)) == pytest.approx(textbook, rel=1e-10)
    assert float(linear_cka(x, 0.1 * x @ _orthogonal(5, seed=2))) == pytest.approx(1.0, rel=1e-10)
    assert float(linear_cka(x, _centred(_randn(200, 4, seed=3)))) < 0.1


def test_the_canonical_overlap_reads_through_any_invertible_linear_map():
    x = _centred(_randn(500, 6))
    mixed = _randn(6, 6, seed=1) + 3.0 * torch.eye(6, dtype=x.dtype)
    assert float(canonical_overlap(x, x @ mixed, ridge=1e-9)) == pytest.approx(1.0, abs=1e-6)
    y = _centred(0.7 * x[:, :3] @ _randn(3, 4, seed=2) + _randn(500, 4, seed=3))
    # the textbook: the canonical correlations are the singular values of Qx^T Qy
    qx, qy = np.linalg.qr(x.numpy())[0], np.linalg.qr(y.numpy())[0]
    rho = np.linalg.svd(qx.T @ qy, compute_uv=False)
    expected = (rho**2).sum() / 4
    assert float(canonical_overlap(x, y, ridge=1e-9)) == pytest.approx(expected, rel=1e-6)
    a, b = (
        _randn(6, 6, seed=4) + 3 * torch.eye(6, dtype=x.dtype),
        _randn(4, 4, seed=5) + 3 * torch.eye(4, dtype=x.dtype),
    )
    assert float(canonical_overlap(x @ a, y @ b, ridge=1e-9)) == pytest.approx(expected, rel=1e-6)
    assert float(canonical_overlap(x, _centred(_randn(500, 6, seed=6)))) < 0.05


def test_an_inflated_unused_direction_lowers_cka_but_not_the_overlap():
    """The loophole the plan names: CKA weighs directions by variance, so a copy can hide under a large unused one."""
    z = _centred(_randn(500, 3))
    copy_ = z @ (_randn(3, 3, seed=1) + 3 * torch.eye(3, dtype=z.dtype))
    hidden = _centred(torch.cat([0.01 * z, 100.0 * _randn(500, 1, seed=2)], dim=1))
    assert float(linear_cka(hidden, copy_)) < 0.01
    assert float(canonical_overlap(hidden, copy_)) > 0.99


def test_the_ridge_is_a_share_of_the_mean_variance():
    # orthogonal columns of very different sizes: after the unit scaling X^T X = n I, so each squared canonical
    # correlation of x with itself is (n / (n + ridge * n))^2 whatever n and the columns' sizes
    q, _ = torch.linalg.qr(_randn(400, 3))
    x = q * torch.tensor([1.0, 30.0, 0.01], dtype=q.dtype)
    assert float(canonical_overlap(x, x, ridge=0.1)) == pytest.approx(1 / 1.1**2, rel=1e-9)


def test_on_loadings_the_overlap_is_the_mean_squared_cosine_of_the_principal_angles():
    w, other = _randn(50, 4), _randn(50, 4, seed=1)
    angles = subspace_angles(w.numpy(), other.numpy())
    assert float(canonical_overlap(w, other, ridge=1e-12)) == pytest.approx(np.mean(np.cos(angles) ** 2), rel=1e-6)
    assert float(
        canonical_overlap(w, w @ (_randn(4, 4, seed=2) + 2 * torch.eye(4, dtype=w.dtype)), 1e-12)
    ) == pytest.approx(1.0, abs=1e-6)
    q, _ = torch.linalg.qr(w)
    complement = other - q @ (q.T @ other)
    assert float(canonical_overlap(w, complement, ridge=1e-12)) < 1e-10


def test_kernel_cka_sees_the_nonlinear_dependence_linear_cka_misses():
    x = 2.0 * _randn(300, 1)
    y = x**2 + 0.01 * _randn(300, 1, seed=1)
    assert float(linear_cka(_centred(x), _centred(y))) < 0.02
    assert float(kernel_cka(_centred(x), _centred(y))) > 0.3
    assert float(kernel_cka(_centred(x), _centred(_randn(300, 1, seed=2)))) < 0.02
    many = _randn(300, 3, seed=3)
    assert float(kernel_cka(10.0 * many, many**2)) == pytest.approx(float(kernel_cka(many, many**2)), rel=1e-9)


def test_kernel_cka_is_its_textbook_formula():
    x, y = _randn(30, 2), _randn(30, 3, seed=1)
    n = 30

    def gram(a):
        d2 = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)
        sigma2 = np.median(d2[~np.eye(n, dtype=bool)])
        return np.exp(-d2 / (2 * sigma2))

    h = np.eye(n) - 1.0 / n
    kx, ky = h @ gram(x.numpy()) @ h, h @ gram(y.numpy()) @ h
    expected = (kx * ky).sum() / np.sqrt((kx * kx).sum() * (ky * ky).sum())
    assert float(kernel_cka(x, y)) == pytest.approx(expected, rel=1e-8)


def test_oplsterm_is_the_mean_absolute_cosine_and_a_rotation_into_the_complement_zeroes_it():
    x, y = _randn(20, 4), _randn(15, 4, seed=1)
    by_hand = np.mean([abs(float(torch.nn.functional.cosine_similarity(a, b, dim=0))) for a in x for b in y])
    assert float(cross_cosine(x, y)) == pytest.approx(by_hand, rel=1e-10)
    # the control: one part a copy of the other, moved into its orthogonal complement of a shared 8-d space
    z = _randn(100, 4, seed=2)
    copy_ = z @ (_randn(4, 4, seed=3) + 3 * torch.eye(4, dtype=z.dtype))
    local = torch.cat([z, torch.zeros_like(z)], dim=1)
    glob = torch.cat([torch.zeros_like(z), copy_], dim=1)
    assert float(cross_cosine(local, glob)) == 0.0
    assert float(canonical_overlap(_centred(local[:, :4]), _centred(glob[:, 4:]))) > 0.99


def test_the_gene_correlation_is_per_gene_on_the_masked_rows_of_each_graph():
    a = _randn(60, 3)
    b = torch.stack([2.0 * a[:, 0] + 1.0, _randn(60, seed=1), torch.full((60,), 4.0, dtype=a.dtype)], dim=1)
    corr1 = np.corrcoef(a[:, 1].numpy(), b[:, 1].numpy())[0, 1]
    # gene 2 does not vary in b: left out of the mean
    assert float(gene_correlation(a, b)) == pytest.approx((1.0 + corr1**2) / 2, rel=1e-10)
    # entries outside the mask do not count
    mask = torch.ones(60, 3, dtype=torch.bool)
    mask[:10, 0] = False
    noisy_a, noisy_b = a.clone(), b.clone()
    noisy_a[:10, 0], noisy_b[:10, 0] = _randn(10, seed=2) * 50, _randn(10, seed=3) * 50
    assert float(gene_correlation(noisy_a, noisy_b, mask)) == pytest.approx(float(gene_correlation(a, b)), rel=1e-10)
    # an offset shared by both parts on one graph is not a correlation within the graph
    groups = torch.tensor([0] * 30 + [1] * 30)
    u, v = _randn(60, 1, seed=4), _randn(60, 1, seed=5)
    u[30:] += 10.0
    v[30:] += 10.0
    assert float(gene_correlation(u, v)) > 0.9
    assert float(gene_correlation(u, v, groups=groups)) < 0.05
    # nothing varies: a zero that keeps the graph
    leaf = _randn(60, 2).requires_grad_()
    zero = gene_correlation(leaf, torch.zeros(60, 2, dtype=leaf.dtype))
    assert float(zero) == 0.0 and zero.requires_grad


def test_decorrelation_reads_standardised_dimensions_of_each_graph():
    x = _randn(2000, 4)
    assert float(decorrelation(x)) < 0.01
    dup = torch.cat([x, 5.0 * x[:, :1]], dim=1)
    # the pair (0, 4) correlates fully: 2 of the 20 off-diagonal entries
    assert float(decorrelation(dup)) == pytest.approx(2 / 20, abs=0.01)
    scales = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=x.dtype)
    assert float(decorrelation(x * scales)) == pytest.approx(float(decorrelation(x)), rel=1e-9)
    groups = torch.tensor([0] * 1000 + [1] * 1000)
    shifted = x.clone()
    shifted[1000:, :2] += 20.0
    # pooled, the shared offset correlates columns 0 and 1: 2 of the 12 off-diagonal entries
    assert float(decorrelation(shifted)) == pytest.approx(2 / 12, abs=0.01)
    assert float(decorrelation(shifted, groups)) < 0.01
    constant = torch.cat([x, torch.ones(2000, 1, dtype=x.dtype)], dim=1)
    assert float(decorrelation(constant)) == pytest.approx(float(decorrelation(x)), rel=1e-9)


def test_loading_orthogonality_is_the_mean_squared_cosine_between_loading_vectors():
    q, _ = torch.linalg.qr(_randn(30, 4))
    assert float(loading_orthogonality(q)) < 1e-20
    w = torch.cat([q, 5.0 * q[:, :1]], dim=1)
    assert float(loading_orthogonality(w)) == pytest.approx(2 / 20, rel=1e-9)
    with_zero = torch.cat([q, torch.zeros(30, 1, dtype=q.dtype)], dim=1)
    assert float(loading_orthogonality(with_zero)) < 1e-20


# --------------------------------------------------------------------------- the plumbing

N_CELLS, E_DIM, N_GENES = 40, 4, 5


def _cfg(**weights):
    cfg = get_cfg_defaults()
    for name, w in weights.items():
        setattr(cfg.optim.aux_loss_weights, name, w)
    return cfg


def _hand_out(local, glob, hidden, parts=None):
    """A step output laid out as a dual model's: two graphs of 20 cells, no padding, CLS last."""
    g = glob.reshape(2, 20, E_DIM).permute(1, 0, 2)
    padded = torch.cat([g, torch.zeros(1, 2, E_DIM, dtype=glob.dtype)], dim=0)
    view = ViewOutput(
        local_embedding=local,
        global_embedding=padded,
        src_padding_mask=torch.zeros(2, 21, dtype=torch.bool),
        padded_node_idx=torch.arange(N_CELLS),
    )
    entry = hidden[:, None].expand(N_CELLS, N_GENES)
    y = torch.zeros(2 * N_CELLS, N_GENES, dtype=local.dtype)
    return StepOutput(y_pred=y, y_true=y, entry_mask=torch.cat([entry, entry]), parts=parts, views=[view])


def _hand_inputs(seed=0):
    local = _randn(N_CELLS, E_DIM, seed=seed)
    glob = 0.5 * local @ _randn(E_DIM, E_DIM, seed=seed + 1) + _randn(N_CELLS, E_DIM, seed=seed + 2)
    hidden = torch.zeros(N_CELLS, dtype=torch.bool)
    hidden[::2] = True
    batch = types.SimpleNamespace(batch=torch.tensor([0] * 20 + [1] * 20))
    return local, glob, hidden, batch


@pytest.mark.parametrize("name", [*EMBEDDING_TERMS, "orth_decorrelation"])
def test_a_term_reads_only_the_cells_of_the_loss(name):
    local, glob, hidden, batch = _hand_inputs()
    term = build_aux_losses(_cfg(**{name: 1.0})).terms[name]
    base = float(term(_hand_out(local, glob, hidden), batch)[name])
    moved_local, moved_glob = local.clone(), glob.clone()
    moved_local[~hidden] += 3.0 * _randn(int((~hidden).sum()), E_DIM, seed=9)
    moved_glob[~hidden] -= 3.0 * _randn(int((~hidden).sum()), E_DIM, seed=10)
    assert float(term(_hand_out(moved_local, moved_glob, hidden), batch)[name]) == pytest.approx(base, rel=1e-10)
    moved_glob[hidden] += _randn(int(hidden.sum()), E_DIM, seed=11)
    assert float(term(_hand_out(moved_local, moved_glob, hidden), batch)[name]) != pytest.approx(base, rel=1e-6)


@pytest.mark.parametrize("name", ["orth_cka", "orth_cca", "orth_kernel_cka", "orth_decorrelation"])
def test_a_term_centres_each_graph_on_its_own(name):
    local, glob, hidden, batch = _hand_inputs(seed=3)
    term = build_aux_losses(_cfg(**{name: 1.0})).terms[name]
    base = float(term(_hand_out(local, glob, hidden), batch)[name])
    shifted_local, shifted_glob = local.clone(), glob.clone()
    shifted_local[20:] += 25.0
    shifted_glob[20:] += 25.0
    assert float(term(_hand_out(shifted_local, shifted_glob, hidden), batch)[name]) == pytest.approx(base, rel=1e-8)


@pytest.mark.parametrize("name", EMBEDDING_TERMS)
def test_the_local_embedding_is_gathered_into_token_order(name):
    """``padded_node_idx`` names the cell behind each token; the local embedding (batch order) must follow it."""
    local, glob, hidden, batch = _hand_inputs(seed=5)
    order = torch.cat([torch.randperm(20, generator=torch.Generator().manual_seed(2)), 20 + torch.arange(20)])
    term = build_aux_losses(_cfg(**{name: 1.0})).terms[name]
    aligned = float(term(_hand_out(local, glob, hidden), batch)[name])
    # the same cells, the tokens in another order: node `order[i]` behind token `i`
    shuffled = _hand_out(torch.empty_like(local), glob, hidden)
    shuffled.views[0].local_embedding = torch.empty_like(local)
    shuffled.views[0].local_embedding[order] = local
    shuffled.views[0].padded_node_idx = order
    graph = types.SimpleNamespace(batch=torch.empty_like(batch.batch))
    graph.batch[order] = batch.batch
    assert float(term(shuffled, graph)[name]) == pytest.approx(aligned, rel=1e-10)


@pytest.mark.parametrize("name", [*EMBEDDING_TERMS, "orth_decorrelation"])
def test_a_step_with_too_few_cells_gives_a_zero_that_keeps_the_graph(name):
    local, glob, hidden, batch = _hand_inputs()
    hidden[:] = False
    hidden[:2] = True
    local, glob = local.requires_grad_(), glob.requires_grad_()
    term = build_aux_losses(_cfg(**{name: 1.0})).terms[name]
    value = term(_hand_out(local, glob, hidden), batch)[name]
    if name == "orth_opl":  # OPL reads raw pairs: two cells are enough
        assert float(value) > 0
    else:
        assert float(value) == 0.0
    assert value.requires_grad


def test_the_gene_correlation_term_reads_the_heads_parts_on_the_loss_entries():
    local, glob, hidden, batch = _hand_inputs()
    a, b = _randn(N_CELLS, N_GENES, seed=4), _randn(N_CELLS, N_GENES, seed=5)
    b[:, 0] = 3.0 * a[:, 0]
    term = build_aux_losses(_cfg(orth_gene_corr=1.0)).terms["orth_gene_corr"]
    out = _hand_out(local, glob, hidden, parts={"global": {"local": a, "global": b}})
    entries = hidden[:, None].expand(N_CELLS, N_GENES)
    expected = gene_correlation(a, b, entries, batch.batch)
    assert float(term(out, batch)["orth_gene_corr"]) == pytest.approx(float(expected), rel=1e-12)
    with pytest.raises(ValueError, match="two parts"):
        term(_hand_out(local, glob, hidden, parts=None), batch)


def test_the_pairwise_terms_subsample_from_a_generator_of_their_own():
    local, glob, hidden, batch = _hand_inputs()
    hidden[:] = True
    cfg = _cfg(orth_kernel_cka=1.0)
    cfg.optim.orthogonality.subsample = 8
    first, again = build_aux_losses(cfg).terms["orth_kernel_cka"], build_aux_losses(cfg).terms["orth_kernel_cka"]
    torch.manual_seed(5)
    state = torch.get_rng_state().clone()
    values = [float(first(_hand_out(local, glob, hidden), batch)["orth_kernel_cka"]) for _ in range(3)]
    assert torch.equal(torch.get_rng_state(), state)
    assert float(again(_hand_out(local, glob, hidden), batch)["orth_kernel_cka"]) == values[0]
    assert len(set(values)) == 3  # each step draws other cells


# --------------------------------------------------------------------------- the models

# Two 8-dimensional loadings fit orthogonally into 20 genes; a raw skip (20 dimensions) does not.
G_GENES, E_EMBED = 20, 8


def _stir(module, scale=0.1, seed=7):
    """Every parameter moved by seeded noise: a fresh module's zero biases, identity norms and zero skip hide paths."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=gen))
    return module


def _dual(additive=True, residual=False, decoder="linear"):
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg
    from interscale.module.combined_module import DualDecoderCombinedModule

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = "GCN"
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, "GCN")
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.decoder.dual_decoder, cfg.model.decoder.type = True, decoder
    cfg.model.decoder.additive, cfg.model.decoder.global_residual = additive, residual
    cfg.dataset.name = "d"
    torch.manual_seed(0)
    module = DualDecoderCombinedModule(
        cfg=cfg,
        n_input=G_GENES,
        n_output=G_GENES,
        n_embed=E_EMBED,
        decoder_type=decoder,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy="node",
    ).eval()
    return _stir(module)


def _global(skip=True, read="embedding", decoder="linear", strategy="node"):
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.max_seq_len = 64
    cfg.model.global_component.parameters.type_gex_embedding = "Linear"
    cfg.model.decoder.type, cfg.model.decoder.own_cell_skip = decoder, skip
    cfg.model.decoder.own_cell_skip_input = read
    cfg.dataset.name = "d"
    torch.manual_seed(0)
    module = GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=G_GENES,
        n_output=G_GENES,
        n_embed=E_EMBED,
        decoder_type=decoder,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.3,
        mask_strategy=strategy,
        type_gex_embedding="Linear",
    ).eval()
    return _stir(module)


def _batch(sizes=(14, 18), seed=0, strategy="node"):
    """Lattice slides with a 4-nearest-neighbour graph; whole cells hidden, or under ``gene`` a third of the entries."""
    rng = np.random.default_rng(seed)
    datas = []
    for n in sizes:
        side = math.ceil(math.sqrt(n))
        pos = np.array([(k % side, k // side) for k in range(n)], dtype=float) * 100.0
        _, j = cKDTree(pos).query(pos, k=5)
        src, dst = np.repeat(np.arange(n), 4), j[:, 1:].ravel()
        d = Data(
            x=torch.tensor(rng.random((n, G_GENES)) + 0.5, dtype=torch.float32),
            edge_index=torch.tensor(np.stack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
        )
        if strategy == "gene":
            d.gene_mask = torch.tensor(rng.random((n, G_GENES)) < 0.3)
            d.mask = d.gene_mask.any(dim=1)
        else:
            d.mask = torch.tensor(rng.random(n) < 0.5)
            d.mask[0] = True
        datas.append(d)
    return Batch.from_data_list(datas)


ACCEPTS = {
    "dual additive": (lambda: _dual(), set(TERMS)),
    "dual plain": (lambda: _dual(additive=False), set(TERMS) - {"orth_gene_corr", "orth_loadings"}),
    "dual residual": (lambda: _dual(additive=False, residual=True), set(TERMS)),
    "dual nonlinear": (
        lambda: _dual(decoder="nonlinear"),
        set(TERMS) - {"orth_loadings", "orth_loading_ortho"},
    ),
    "global skip": (
        lambda: _global(),
        {"orth_gene_corr", "orth_loadings", "orth_decorrelation", "orth_loading_ortho"},
    ),
    "global raw skip": (lambda: _global(read="raw"), {"orth_gene_corr", "orth_decorrelation", "orth_loading_ortho"}),
    "global plain": (lambda: _global(skip=False), {"orth_decorrelation", "orth_loading_ortho"}),
}


@pytest.mark.parametrize("model", sorted(ACCEPTS))
def test_each_term_accepts_the_models_it_reads_and_refuses_the_rest(model):
    build, accepted = ACCEPTS[model]
    module = build()
    for name in TERMS:
        if name in accepted:
            assert name in build_aux_losses(_cfg(**{name: 1.0}), module).terms
        else:
            with pytest.raises(ValueError, match=name):
                build_aux_losses(_cfg(**{name: 1.0}), module)


GCN, TRANSFORMER = "local_module.layers", "global_module.transformer_encoder"
LOCAL_DECODER, GLOBAL_DECODER = "local_module.decoder", "global_module.decoder"
#: Through a real dual additive step: what each term must move, and what it must not.
REACHES = {
    **{name: ({GCN, TRANSFORMER}, {LOCAL_DECODER, GLOBAL_DECODER}) for name in EMBEDDING_TERMS},
    "orth_gene_corr": ({LOCAL_DECODER, GLOBAL_DECODER, GCN, TRANSFORMER}, set()),
    "orth_loadings": ({LOCAL_DECODER, GLOBAL_DECODER}, {GCN, TRANSFORMER}),
    "orth_decorrelation": ({TRANSFORMER}, {LOCAL_DECODER, GLOBAL_DECODER}),
    "orth_loading_ortho": ({GLOBAL_DECODER}, {LOCAL_DECODER, GCN, TRANSFORMER}),
}


def _gradients(module, name, batch=None):
    batch = _batch() if batch is None else batch
    term = build_aux_losses(_cfg(**{name: 1.0}), module).terms[name]
    module.zero_grad()
    out = module._common_step(batch, "regression", "node")
    value = term(out, batch)[name]
    assert torch.isfinite(value) and 0.0 <= float(value) <= 1.0 + 1e-6
    value.backward()
    return {n: p.grad for n, p in module.named_parameters()}


def _moved(grads, prefix):
    found = [g for n, g in grads.items() if n.startswith(prefix)]
    assert found, f"no parameter under {prefix}"
    return any(g is not None and float(g.abs().sum()) > 0 for g in found)


@pytest.mark.parametrize("name", TERMS)
def test_a_term_reaches_exactly_the_parameters_it_should_through_a_real_dual_step(name):
    grads = _gradients(_dual(), name)
    must, must_not = REACHES[name]
    for prefix in must:
        assert _moved(grads, prefix), f"{name} does not reach {prefix}"
    for prefix in must_not:
        assert not _moved(grads, prefix), f"{name} reaches {prefix}"


def test_under_the_residual_decoder_the_gene_correlation_moves_the_global_part_only():
    grads = _gradients(_dual(additive=False, residual=True), "orth_gene_corr")
    assert _moved(grads, "global_module.decoder")
    assert not _moved(grads, "local_module.decoder")


def test_a_global_only_model_with_the_skip_splits_into_its_context_and_its_own_cell():
    # Gene masking: a cell hidden whole would hand the skip the fill, the same for every hidden cell.
    module, batch = _global(strategy="gene"), _batch(strategy="gene")
    for name in ("orth_gene_corr", "orth_loadings"):
        grads = _gradients(module, name, batch)
        assert _moved(grads, "own_skip_weight") and _moved(grads, "decoder")
    # under cell masking the own-cell part is constant on the loss's cells: the term is 0 and moves nothing
    still, cells = _global(), _batch()
    term = build_aux_losses(_cfg(orth_gene_corr=1.0), still).terms["orth_gene_corr"]
    assert float(term(still._common_step(cells, "regression", "node"), cells)["orth_gene_corr"]) == 0.0


def test_a_term_that_reads_cells_refuses_dropout_on_their_path():
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_local_component_cfg(_cfg(), "GCN")  # dropout_local 0.1
    for name in TERMS:
        setattr(cfg.optim.aux_loss_weights, name, 0.0)
    for name in ("orth_cka", "orth_cca", "orth_kernel_cka", "orth_opl", "orth_gene_corr", "orth_decorrelation"):
        setattr(cfg.optim.aux_loss_weights, name, 1.0)
        with pytest.raises(ValueError, match="dropout"):
            build_aux_losses(cfg)
        setattr(cfg.optim.aux_loss_weights, name, 0.0)
    for name in ("orth_loadings", "orth_loading_ortho"):  # weights only: dropout does not touch them
        setattr(cfg.optim.aux_loss_weights, name, 1.0)
        assert name in build_aux_losses(cfg).terms
        setattr(cfg.optim.aux_loss_weights, name, 0.0)
    # a nonlinear decoder's dropout counts for the parts it outputs
    clean = get_global_component_cfg(_cfg(orth_gene_corr=1.0), "self-attn-transformer")
    clean.model.global_component.parameters.dropout_global = 0.0
    clean.model.decoder.type, clean.model.decoder.dropout_decoder = "nonlinear", 0.1
    with pytest.raises(ValueError, match="dropout"):
        build_aux_losses(clean)
    clean.model.decoder.type = "linear"
    assert "orth_gene_corr" in build_aux_losses(clean).terms


def test_the_module_holds_no_back_reference_and_a_deep_copy_reads_its_own_decoders():
    module = _dual()
    module.aux_losses = build_aux_losses(_cfg(orth_loadings=1.0, orth_loading_ortho=1.0), module)
    keys = list(module.state_dict())
    assert not any("owner" in k for k in keys) and not any(k.startswith("aux_losses.") for k in keys)
    clone = copy.deepcopy(module)
    term = clone.aux_losses.terms["orth_loadings"]
    assert term.owner is clone and module.aux_losses.terms["orth_loadings"].owner is module
    with torch.no_grad():
        clone.global_module.decoder.decoder.weight.copy_(clone.local_module.decoder.decoder.weight)
    out = None  # the loading terms read no step output
    assert float(term(out, None)["orth_loadings"]) == pytest.approx(1.0, abs=0.01)  # below 1 by the ridge
    assert float(module.aux_losses.terms["orth_loadings"](out, None)["orth_loadings"]) < 0.99


# --------------------------------------------------------------------------- inert by default


def test_every_term_is_registered_and_off_by_default():
    cfg = get_cfg_defaults()
    for name in TERMS:
        assert name in AUX_LOSSES and float(getattr(cfg.optim.aux_loss_weights, name)) == 0.0
    assert not build_aux_losses(cfg, _dual())


def test_the_checkpoint_name_carries_every_non_zero_auxiliary_weight():
    def name(**weights):
        cfg = _cfg(**weights)
        cfg.dataset.name, cfg.model.decoder.dual_decoder = "d", True
        cfg.model.local_component.name, cfg.model.global_component.name = "GCN", "self-attn-transformer"
        return get_model_filename_prefix(cfg, True, True)

    plain = name()
    assert "aux-" not in plain
    assert "aux-orth_cka0.1" in name(orth_cka=0.1)
    assert name(orth_cka=0.1) != name(orth_cka=0.2) != name(orth_cca=0.1)
    assert "aux-orth_cca0.5-orth_cka0.1" in name(orth_cka=0.1, orth_cca=0.5)
    assert "aux-vicreg1" in name(vicreg=1.0)


def test_the_settings_are_validated_when_a_term_is_on():
    from interscale.config import _validate_orthogonality

    cfg = _cfg(orth_cca=1.0)
    _validate_orthogonality(cfg)
    cfg.optim.orthogonality.ridge = 0.0
    with pytest.raises(ValueError, match="ridge"):
        _validate_orthogonality(cfg)
    cfg.optim.orthogonality.ridge, cfg.optim.orthogonality.subsample = 1e-3, 2
    with pytest.raises(ValueError, match="subsample"):
        _validate_orthogonality(cfg)
    off = _cfg()
    off.optim.orthogonality.ridge = 0.0
    _validate_orthogonality(off)  # no term on: the settings are inert
    # and loading a config runs the check
    from interscale.config import _validate

    loaded = _cfg(orth_cca=1.0)
    loaded.optim.orthogonality.ridge = 0.0
    with pytest.raises(ValueError, match="ridge"):
        _validate(loaded)


# --------------------------------------------------------------------------- the trainer


def _trained(tmp_path, weight, term="orth_cca", epochs=15, per=30):
    """A dual additive model trained through ``train()`` on eight toy slides, ``term`` at ``weight``, no dropout.

    Dropout off in both runs: with it the term is met by noise that evaluation removes (and the term refuses it).
    """
    import scanpy as sc
    from anndata import AnnData

    from interscale.config import load_config
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.model.combined_model import CombinedModel
    from interscale.tl import prepare_geome_dataset

    rng = np.random.default_rng(0)
    samples = [f"s{k}" for k in range(10)]
    per = per
    n = per * len(samples)
    pos = rng.uniform(0, 500, size=(n, 2)).astype(np.float32)
    field = np.sin(pos[:, :1] / 80.0) + np.cos(pos[:, 1:] / 120.0)
    x = np.abs(field @ rng.normal(size=(1, G_GENES)) + 0.3 * rng.normal(size=(n, G_GENES))).astype(np.float32) + 0.05
    adata = AnnData(X=x)
    adata.var_names = [f"g{k}" for k in range(G_GENES)]
    adata.obsm["spatial"] = pos
    adata.obs["sample"] = pd.Categorical(np.repeat(samples, per))
    split = {s: ("val" if s == "s8" else "test" if s == "s9" else "train") for s in samples}
    adata.obs["split"] = pd.Categorical([split[s] for s in adata.obs["sample"]])
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    h5ad = tmp_path / f"orth{weight}.h5ad"
    adata.write_h5ad(h5ad)
    path = tmp_path / f"orth{weight}.yaml"
    path.write_text(
        "model:\n  n_embed: 8\n  local_component:\n    name: GCN\n    parameters:\n      dropout_local: 0.0\n"
        "  global_component:\n"
        "    name: self-attn-transformer\n    parameters:\n      max_seq_len: 64\n      num_layers: 1\n"
        "      n_heads: 2\n      long_range_attention: False\n      dropout_global: 0.0\n"
        f"  decoder:\n    type: linear\n    dual_decoder: True\n    additive: True\n  save: {tmp_path}/out{weight}/\n"
        "optim:\n  loss: SCELoss\n  sce_gamma: 1.0\n  lr: 0.01\n  seed: 42\n  accelerator: cpu\n"
        f"  n_epochs: {epochs}\n  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        "  loss_weights:\n    local: 0.0\n    global: 1.0\n"
        f"  aux_loss_weights:\n    {term}: {weight}\n"
        f"dataset:\n  h5ad_data: {h5ad}\n  name: orthtest\n  prediction_task: regression\n"
        "  prediction_level: node\n  layer_key: log1p_norm\n  sample_key: ['sample']\n  mask_strategy: node\n"
        "  mask_percentage: 0.3\n  spatial_key: spatial\n  batch_size: 2\n  evaluate_test: False\n"
        "  spatial_neigbors_kwargs:\n    radius: 120\n    library_key: sample\n"
    )
    cfg = load_config(str(path))
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    CombinedModel._setup_anndata(
        adata=adata,
        prediction_task="regression",
        layer_key="log1p_norm",
        sample_key_list=["sample"],
        prediction_obs=None,
    )
    net = CombinedModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=0, batch_size=2, mask_percentage=0.3, mask_strategy="node", learning_type="node"
    )
    net.train(max_epochs=epochs, datamodule=dm, early_stopping=False, wandb_use=False)
    return net, pyg, cfg


def _final_overlap(net, pyg, term="orth_cca"):
    """The term's value over all training slides at once (each centred on its own), under seeded masks the same for
    every run: one slide's few hidden cells against 16 dimensions would give canonical correlations high by chance."""
    module = net.module.eval()
    measure = build_aux_losses(_cfg(**{term: 1.0}), module).terms[term]
    gen = torch.Generator().manual_seed(1)
    graphs = []
    for data in pyg[0]:
        graph = data.clone()
        graph.mask = torch.rand(graph.num_nodes, generator=gen) < 0.3
        graphs.append(graph)
    batch = Batch.from_data_list(graphs).to(module.device)
    with torch.no_grad():
        return float(measure(module._common_step(batch, "regression", "node"), batch)[term])


def test_a_dual_model_trains_with_a_term_and_ends_with_it_lower(tmp_path):
    on, pyg_on, cfg_on = _trained(tmp_path, 1.0)
    rows = [row for row in on.history_.history if "val_orth_cca" in row]
    assert rows and all(np.isfinite(row["val_orth_cca"]) and np.isfinite(row["val_loss"]) for row in rows)
    assert any("aux-orth_cca1" in p.name for p in (tmp_path / "out1.0").iterdir())
    off, pyg_off, _ = _trained(tmp_path, 0.0)
    assert not any("val_orth_cca" in row for row in off.history_.history)
    assert _final_overlap(on, pyg_on) < _final_overlap(off, pyg_off) - 0.05
