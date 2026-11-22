"""Attribution of a receiver's prediction to the cells it reads (``.claude/attribution_plan.md``, A0-A6, 2026-11-22).

One interface (:func:`attribute`) for every method and reference, on any module whose input tokens come from ``gex_embedding``
(GlobalModel): the path runs in token space, the space the model reads (for the linear token embedding the same as expression space).

**Target.** Per receiver a fixed weight vector over the predicted genes (``weights`` ``[Q, G]``): the receiver's predicted deviation from
its type mean (:func:`deviation_targets`), a named gene set (:func:`gene_targets`), or a programme's response genes. Fixed weights keep
the target linear in the prediction, so attributions of several receivers add (a disc's target is the sum of its receivers').

**Methods** (:data:`METHODS`). ``decomposition``: the interaction read-out's exact per-sender share of the context part (no reference,
no gradient; :mod:`interaction_readout`). ``attention``: the read-out's head- and layer-mean weight (routing, not an effect; for
comparison). ``grad_input``: gradient at the input times the deviation from the reference. ``ig``: integrated gradients (Gauss-Legendre
points, a receiver whose completeness gap stays above ``gap_limit`` is integrated again with ``retry_steps``). ``deeplift``: DeepLIFT-
Rescale, i.e. gradient times deviation with every ReLU / GELU's derivative replaced by its secant between the reference and the input
(:func:`_secant_activations`); softmax and LayerNorm keep their gradients, so completeness is not guaranteed and is reported.
``conservative``: gradient x input with every softmax output and LayerNorm's scale held constant in the backward pass (Ali et al. 2022;
AttnLRP), which restores conservation through attention; it needs explicit attention weights, so the SDPA kernel is replaced by its explicit
form for the pass (the read-out and fused backends compute block by block, so a section of ~10^4 cells fits). ``occlusion``: one sender's
token set to its reference (or hidden by the model's own mask, ``removal="mask"``), one forward per sender (a whole column at once).

**References** (:data:`REFERENCES`). ``graph_mean`` (R1: the graph's mean token), ``type_mean`` (R2: the mean token of the sender's
label in the graph: what a sender expresses beyond its type, so composition drops out), ``sampled`` (R4: a random other cell of the
same label, ``draws`` times, averaged; the spread is kept).

**Fixed rows.** The receivers' own rows and cells hidden whole never move: a receiver's prediction is attributed to *other* cells
(its own row, and anything the module reads of it, stays as given), and a hidden cell's token is the mask fill, no measurement.

**Measures.** A1: :func:`planted_top_k`, :func:`agreement`. A3: :func:`strengths`, :func:`disc_strengths`, :func:`map_fields`. A4:
:func:`distance_profile`. A6: :func:`sensitivity_n` (hide random subsets of a receiver's nearby cells, correlate the change with the
attributions' sum, Ancona et al. 2018) and :func:`pairwise_interactions` (``Delta_jk - Delta_j - Delta_k``: zero for an additive
function). Both read many receivers per forward: receivers are chosen far apart (:func:`separated_receivers`) and each perturbs only its
own nearby cells, so one pass serves all of them (cross-talk between distant receivers is the stated approximation).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from interscale.evaluation.context_test import _hidden_entries
from interscale.evaluation.receptive_field import _integration_points, _replayable_rng

METHODS = ("decomposition", "attention", "grad_input", "ig", "deeplift", "conservative", "occlusion")
#: How occlusion, Sensitivity-n and pairwise interactions remove a sender: its token set to its reference, or the model's own removal (the cell
#: hidden whole: under ``visible_keys_only`` no key; node-masked models only).
REMOVALS = ("reference", "mask")
REFERENCES = ("graph_mean", "type_mean", "sampled")


@dataclass
class Attribution:
    """``contribution`` ``[Q, N]``: each cell's signed share of each receiver's target (the receiver's own column 0; NaN where not read,
    e.g. occlusion outside its senders). ``change`` ``[Q]``: the target at the input minus at the reference (decomposition: minus the
    decoder's intercept; NaN for occlusion and attention), ``gap`` ``|sum - change| / |change|``, ``steps`` per receiver (IG)."""

    method: str
    reference: str | None
    receivers: np.ndarray
    contribution: np.ndarray
    change: np.ndarray
    gap: np.ndarray
    seconds: float
    steps: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- targets


def deviation_targets(prediction: np.ndarray, labels, receivers) -> np.ndarray:
    """``[Q, G]``: each receiver's predicted profile minus the mean prediction of its label (over the graph's cells), unit length: "what
    moved this cell off its type". A receiver equal to its type mean gets zeros (no target)."""
    labels = np.asarray(labels).astype(str)
    pred = np.asarray(prediction, dtype=np.float64)
    out = np.zeros((len(receivers), pred.shape[1]))
    means = {t: np.nanmean(pred[labels == t], axis=0) for t in np.unique(labels[np.asarray(receivers)])}
    for k, q in enumerate(np.asarray(receivers)):
        d = np.nan_to_num(pred[q] - means[labels[q]])
        norm = np.linalg.norm(d)
        out[k] = d / norm if norm > 0 else 0.0
    return out


def gene_targets(columns, n_receivers: int, n_genes: int) -> np.ndarray:
    """``[Q, G]``: the same named genes (summed) for every receiver."""
    w = np.zeros((int(n_receivers), int(n_genes)))
    w[:, np.asarray(columns, dtype=np.int64)] = 1.0
    return w


# ----------------------------------------------------------------------------------------------- passes in token space


def _fixed_rows(batch, receivers) -> np.ndarray:
    """``[N]`` bool: rows that never move (the receivers, and cells hidden whole)."""
    fixed = _hidden_entries(batch, "gene").all(axis=1)
    fixed[np.asarray(receivers, dtype=np.int64)] = True
    return fixed


def _actual_tokens(module, batch) -> torch.Tensor:
    """The tokens ``gex_embedding`` hands the transformer for this batch, as the step would mask it."""
    with torch.no_grad():
        masked = module._common_step_masking(batch)[0] if getattr(module, "masked_nodes", False) else batch
        return module.gex_embedding(masked).detach()


def reference_tokens(tokens: torch.Tensor, labels, reference: str, *, rng: np.random.Generator | None = None, usable=None) -> torch.Tensor:
    """``[N, E]`` reference token per cell (:data:`REFERENCES`); ``usable`` ``[N]`` bool: the cells a mean or a draw may use (default all)."""
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}.")
    n = tokens.shape[0]
    usable = np.ones(n, dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    if reference == "graph_mean":
        return tokens[torch.as_tensor(usable)].mean(dim=0, keepdim=True).expand_as(tokens).clone()
    labels = np.asarray(labels).astype(str)
    out = tokens.clone()
    rng = rng if rng is not None else np.random.default_rng(0)
    for t in np.unique(labels):
        members = np.flatnonzero(labels == t)
        pool = members[usable[members]]
        if not len(pool):
            continue
        if reference == "type_mean":
            out[torch.as_tensor(members)] = tokens[torch.as_tensor(pool)].mean(dim=0, keepdim=True)
        else:
            picks = rng.choice(pool, size=len(members))
            if len(pool) > 1:  # never a cell's own token
                same = picks == members
                while same.any():
                    picks[same] = rng.choice(pool, size=int(same.sum()))
                    same = picks == members
            out[torch.as_tensor(members)] = tokens[torch.as_tensor(picks)]
    return out


@contextmanager
def _given_tokens(module, holder: dict):
    """Inside, ``gex_embedding`` returns ``holder["tokens"]`` (shape-checked) instead of computing them."""
    original = module.gex_embedding

    def given(b, hidden=None):
        tokens = holder["tokens"]
        expected = original(b, hidden=hidden)
        if tokens.shape != expected.shape:
            raise ValueError(f"given tokens {tuple(tokens.shape)} but the batch makes {tuple(expected.shape)}.")
        return tokens

    module.gex_embedding = given
    try:
        yield
    finally:
        del module.gex_embedding


class _Reader:
    """The receivers' targets for given tokens: one step of the module, the head ``head``, the weighted sum per receiver."""

    def __init__(self, module, batch, receivers, weights, head: str):
        from interscale.evaluation.masked_probing import _model_heads

        self.module, self.batch, self.head, self._heads = module, batch, head, _model_heads
        self.receivers = torch.as_tensor(np.asarray(receivers, dtype=np.int64))
        self.weights = torch.as_tensor(np.asarray(weights, dtype=np.float64))
        self.replay = _replayable_rng(batch.x.device)
        self.holder: dict = {}

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[Q]`` targets (float64; NaN for a receiver the step did not predict)."""
        self.holder["tokens"] = tokens
        self.replay()
        with _given_tokens(self.module, self.holder):
            out = self.module._common_step(self.batch, "regression", "node")
        heads, _, _ = self._heads(self.module, out)
        if self.head not in heads:
            raise ValueError(f"no head {self.head!r}; the module has {sorted(heads)}.")
        pred = heads[self.head]
        idx = out.view.padded_node_idx
        if idx is None:
            idx = torch.arange(self.batch.num_nodes, device=pred.device)
        position = torch.full((self.batch.num_nodes,), -1, dtype=torch.long, device=pred.device)
        position[idx] = torch.arange(len(idx), device=pred.device)
        rows = position[self.receivers.to(pred.device)]
        value = (pred[rows.clamp_min(0)].double() * self.weights.to(pred.device)).sum(1)
        return torch.where(rows >= 0, value, torch.full_like(value, float("nan")))


def _grad_rows(value: torch.Tensor, leaf: torch.Tensor, delta: torch.Tensor, rows=None) -> torch.Tensor:
    """``[Q, N]``: for each receiver, (gradient of its target w.r.t. ``leaf``) x ``delta``, summed over the token dimension."""
    q = value.shape[0]
    out = torch.full((q, leaf.shape[0]), float("nan"), dtype=torch.float64)
    for k in range(q) if rows is None else rows:
        if not torch.isfinite(value[k]) or not value.requires_grad:
            continue
        (grad,) = torch.autograd.grad(value[k], leaf, retain_graph=True, allow_unused=True)
        out[k] = 0.0 if grad is None else (grad.double() * delta).sum(1).cpu()
    return out


# ----------------------------------------------------------------------------------------------- DeepLIFT-Rescale


def _activation_kind(fn) -> str | None:
    if fn is F.relu or isinstance(fn, torch.nn.ReLU):
        return "relu"
    if fn is F.gelu or isinstance(fn, torch.nn.GELU):
        return "gelu"
    return None


def _apply(kind: str, x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) if kind == "relu" else F.gelu(x)


def _derivative(kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind == "relu":
        return (x > 0).to(x.dtype)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + x * torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class _SecantFn(torch.autograd.Function):
    """Forward: the activation of ``x``; backward: the secant slope ``(f(x) - f(x_ref)) / (x - x_ref)`` (the derivative where they meet)."""

    @staticmethod
    def forward(ctx, x, x_ref, kind_code):
        kind = "relu" if kind_code == 0 else "gelu"
        y = _apply(kind, x)
        ctx.save_for_backward(x, x_ref)
        ctx.kind = kind
        return y

    @staticmethod
    def backward(ctx, grad):
        x, x_ref = ctx.saved_tensors
        dx = x - x_ref
        close = dx.abs() < 1e-6
        slope = (_apply(ctx.kind, x) - _apply(ctx.kind, x_ref)) / torch.where(close, torch.ones_like(dx), dx)
        slope = torch.where(close, _derivative(ctx.kind, x), slope)
        return grad * slope, None, None


@contextmanager
def _secant_activations(module, mode: str, store: list):
    """Every ReLU / GELU of ``module`` (``nn.ReLU`` / ``nn.GELU`` modules and layers holding ``F.relu`` / ``F.gelu`` as ``activation``)
    wrapped: ``mode="record"`` appends each call's input to ``store`` (the reference pass); ``mode="secant"`` takes them back in the same
    order and applies :class:`_SecantFn`. The two passes must make the same calls in the same order (one batch, replayed generators)."""
    counter = {"k": 0}
    restore = []

    def wrap(kind):
        def act(x, *args, **kwargs):
            if mode == "record":
                store.append(x.detach())
                return _apply(kind, x)
            ref = store[counter["k"]]
            counter["k"] += 1
            if ref.shape != x.shape:
                raise RuntimeError("DeepLIFT: the reference pass made different activation calls than this pass.")
            return _SecantFn.apply(x, ref.to(x.dtype), 0 if kind == "relu" else 1)

        return act

    for m in module.modules():
        kind = _activation_kind(m)
        if kind is not None:
            restore.append((m, "forward", m.forward))
            m.forward = wrap(kind)
        act = getattr(m, "activation", None)
        kind = _activation_kind(act) if act is not None and not isinstance(act, torch.nn.Module) else None
        if kind is not None:
            restore.append((m, "activation", act))
            m.activation = wrap(kind)
    try:
        yield
    finally:
        for m, name, old in reversed(restore):
            if name == "forward":
                del m.forward
            else:
                m.activation = old


@contextmanager
def _conservative_rules():
    """Inside, softmax outputs (``torch.softmax``, ``F.softmax``, ``Tensor.softmax``; ``log_softmax`` likewise) are detached, the SDPA
    kernel is computed explicitly with detached weights, and ``F.layer_norm`` divides by a detached standard deviation: the forward values
    are unchanged, the backward treats attention weights and LayerNorm scales as constants."""
    t_softmax, f_softmax, f_log_softmax = torch.softmax, F.softmax, F.log_softmax
    m_softmax, m_log_softmax = torch.Tensor.softmax, torch.Tensor.log_softmax
    f_sdpa, f_layer_norm = F.scaled_dot_product_attention, F.layer_norm

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if is_causal:
            raise ValueError("conservative: causal attention is not supported.")
        scores = (q @ k.transpose(-2, -1)) * (scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
        return t_softmax(scores, dim=-1).detach() @ v

    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
        dims = tuple(range(-len(normalized_shape), 0))
        mu = x.mean(dim=dims, keepdim=True)
        sd = torch.sqrt(x.var(dim=dims, unbiased=False, keepdim=True) + eps).detach()
        y = (x - mu) / sd
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias
        return y

    torch.softmax = lambda *a, **k: t_softmax(*a, **k).detach()
    F.softmax = lambda *a, **k: f_softmax(*a, **k).detach()
    F.log_softmax = lambda *a, **k: f_log_softmax(*a, **k).detach()
    torch.Tensor.softmax = lambda self, *a, **k: m_softmax(self, *a, **k).detach()
    torch.Tensor.log_softmax = lambda self, *a, **k: m_log_softmax(self, *a, **k).detach()
    F.scaled_dot_product_attention = sdpa
    F.layer_norm = layer_norm
    try:
        yield
    finally:
        torch.softmax, F.softmax, F.log_softmax = t_softmax, f_softmax, f_log_softmax
        torch.Tensor.softmax, torch.Tensor.log_softmax = m_softmax, m_log_softmax
        F.scaled_dot_product_attention, F.layer_norm = f_sdpa, f_layer_norm


# ----------------------------------------------------------------------------------------------- the interface


def attribute(
    module,
    batch,
    receivers,
    weights,
    method: str,
    *,
    labels=None,
    reference: str = "graph_mean",
    head: str = "global",
    steps: int = 5,
    retry_steps: int = 20,
    gap_limit: float = 0.1,
    draws: int = 5,
    senders=None,
    eval_step: bool = False,
    spatial_unit_um: float = 1.0,
    removal: str = "reference",
    seed: int = 0,
) -> Attribution:
    """Attribute each receiver's target (``weights`` ``[Q, G]`` over the predicted genes) to every cell of the single graph ``batch``.

    ``batch`` as the model should read it (its masks apply: the training-style step). ``labels`` ``[N]``: for ``type_mean`` / ``sampled``
    references. ``senders`` (occlusion): the cells to remove one at a time (default: every movable cell). ``eval_step`` (decomposition,
    attention): the clean pass (True) or the masked one. The module is put in eval mode and its parameters frozen for the call; the
    global generators are left as found.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    receivers = np.asarray(receivers, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != len(receivers):
        raise ValueError("weights needs one row per receiver.")
    if getattr(batch, "batch", None) is None:
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
    started = time.time()
    if method in ("decomposition", "attention"):
        return _decomposition(module, batch, receivers, weights, method, eval_step, spatial_unit_um, started)
    if reference != "graph_mean" and labels is None:
        raise ValueError(f"reference {reference!r} needs labels.")
    _check_removal(module, removal)
    rng = np.random.default_rng(seed)
    was_training = module.training
    module.eval()
    saved = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    replay_outer = _replayable_rng(batch.x.device)
    try:
        tokens = _actual_tokens(module, batch)
        fixed = _fixed_rows(batch, receivers)
        movable = torch.as_tensor(~fixed, device=tokens.device)[:, None]
        n_ref = int(draws) if reference == "sampled" else 1
        reads, spread = [], []
        for d in range(n_ref):
            ref = reference_tokens(tokens, labels if labels is not None else np.zeros(len(tokens)), reference, rng=rng,
                                   usable=~_hidden_entries(batch, "gene").all(axis=1))  # fmt: skip
            ref = torch.where(movable, ref, tokens)
            reader = _Reader(module, batch, receivers, weights, head)
            reads.append(_one_reference(reader, tokens, ref, method, steps, retry_steps, gap_limit, senders, fixed, removal))
        contribution = np.mean([r[0] for r in reads], axis=0)
        change = np.mean([r[1] for r in reads], axis=0)
        used = reads[0][2]
        if n_ref > 1:
            spread = np.std([r[0] for r in reads], axis=0)
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)
        module.train(was_training)
        replay_outer()
    contribution[np.arange(len(receivers)), receivers] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(np.nansum(contribution, axis=1) - change) / np.maximum(np.abs(change), 1e-12)
    if method == "occlusion":
        gap = np.full(len(receivers), np.nan)
    extra = {"spread": spread} if len(spread) else {}
    return Attribution(method, reference, receivers, contribution.astype(np.float32), change, gap, time.time() - started, used, extra)


def _one_reference(reader, tokens, ref, method, steps, retry_steps, gap_limit, senders, fixed, removal="reference"):
    delta = (tokens - ref).double()
    q = len(reader.receivers)
    with torch.no_grad():
        end = reader(tokens.clone())
        start = reader(ref.clone())
    change = (end - start).cpu().numpy()
    if method in ("grad_input", "conservative"):
        leaf = tokens.clone().requires_grad_(True)
        rules = _conservative_rules() if method == "conservative" else _no_rules()
        with torch.enable_grad():
            with rules:
                value = reader(leaf)
            contribution = _grad_rows(value, leaf, delta).numpy()
        return contribution, change, None
    if method == "deeplift":
        store: list = []
        with torch.no_grad(), _secant_activations(reader.module, "record", store):
            reader(ref.clone())
        leaf = tokens.clone().requires_grad_(True)
        with torch.enable_grad(), _secant_activations(reader.module, "secant", store):
            value = reader(leaf)
        with torch.enable_grad():
            contribution = _grad_rows(value, leaf, delta).numpy()
        return contribution, change, None
    if method == "ig":

        def integrate(n_steps, rows):
            points, w = _integration_points(n_steps)
            acc = torch.zeros(q, tokens.shape[0], dtype=torch.float64)
            for alpha, weight in zip(points, w, strict=True):
                leaf = (ref + float(alpha) * (tokens - ref)).requires_grad_(True)
                with torch.enable_grad():
                    value = reader(leaf)
                    part = _grad_rows(value, leaf, delta, rows)
                acc[rows] += float(weight) * part[rows]
            return acc

        everyone = list(range(q))
        contribution = integrate(steps, everyone)
        used = np.full(q, int(steps))
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs(contribution.sum(1).numpy() - change) / np.maximum(np.abs(change), 1e-12)
        redo = [k for k in everyone if not gap[k] <= gap_limit]
        if redo and retry_steps > steps:
            again = integrate(retry_steps, redo)
            contribution[redo] = again[redo]
            used[redo] = int(retry_steps)
        return contribution.numpy(), change, used
    # occlusion: one forward per sender, the column for every receiver at once
    cells = np.flatnonzero(~fixed) if senders is None else np.setdiff1d(np.asarray(senders, dtype=np.int64), np.flatnonzero(fixed))
    contribution = np.full((q, tokens.shape[0]), np.nan)
    with torch.no_grad():
        for j in cells:
            if removal == "mask":
                contribution[:, j] = (end - _masked_read(reader, tokens, [j])).cpu().numpy()
                continue
            changed = tokens.clone()
            changed[j] = ref[j]
            contribution[:, j] = (end - reader(changed)).cpu().numpy()
    return contribution, np.full(q, np.nan), None


@contextmanager
def _no_rules():
    yield


def _check_removal(module, removal: str) -> None:
    if removal not in REMOVALS:
        raise ValueError(f"removal must be one of {REMOVALS}, got {removal!r}.")
    if removal == "mask" and getattr(module, "mask_strategy", "node") != "node":
        raise ValueError("removal='mask' hides cells whole: a node-masked model only (mask_strategy node).")


def _masked_read(reader, tokens, rows) -> torch.Tensor:
    """The receivers' targets with ``rows`` hidden whole by the model's own mask (their tokens are what the step makes of the masked cells;
    the other cells keep ``tokens``)."""
    batch = reader.batch
    copy = batch.clone()
    mask = (batch.mask.clone() if getattr(batch, "mask", None) is not None else torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)).bool()
    mask[torch.as_tensor(np.asarray(rows, dtype=np.int64), device=mask.device)] = True
    copy.mask = mask
    hidden_tokens = _actual_tokens(reader.module, copy)
    changed = tokens.clone()
    idx = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=tokens.device)
    changed[idx] = hidden_tokens[idx]
    sub = _Reader(reader.module, copy, reader.receivers.numpy(), reader.weights.numpy(), reader.head)
    return sub(changed)


def _decomposition(module, batch, receivers, weights, method, eval_step, unit, started) -> Attribution:
    """The interaction read-out's exact share of every sender (``decomposition``) or its mean attention (``attention``), on the targets."""
    from interscale.evaluation.interaction_readout import _block_weights, _values, readout_pass

    rp = readout_pass(module, batch, eval_step=eval_step, spatial_unit_um=unit)
    positions = rp.position_of(receivers)
    n = batch.num_nodes
    q = len(receivers)
    contribution = np.full((q, n), np.nan)
    change = np.full(q, np.nan)
    cells_pos = np.arange(rp.offset, rp.offset + len(rp.index_nodes))
    nodes = np.asarray(rp.index_nodes)
    with torch.no_grad():
        values = _values(module, rp)  # [LH, L, E]
        w_dec = module.decoder.decoder.weight.detach().to(values.dtype)  # [G, E]
        # The intercept: the decoder's bias plus every read layer's output-projection bias through the decoder (added once per read).
        bias = module.decoder.decoder.bias.detach().double().cpu().numpy() if module.decoder.decoder.bias is not None else np.zeros(w_dec.shape[0])
        from interscale.evaluation.interaction_readout import readout_layers

        for layer in readout_layers(module):
            if getattr(layer.out_proj, "bias", None) is not None:
                bias = bias + (w_dec.double() @ layer.out_proj.bias.detach().double()).cpu().numpy()
        for k, (node, p) in enumerate(zip(receivers, positions, strict=True)):
            if p < 0:
                continue
            cells, _ = _block_weights(module, rp, int(p), int(p) + 1)  # [1, LH, L]
            a = cells[0]
            if method == "attention":
                contribution[k, nodes] = a.mean(0)[cells_pos].double().cpu().numpy()
                continue
            target = torch.as_tensor(weights[k], dtype=values.dtype, device=values.device)
            u = values @ (w_dec.T @ target)  # [LH, L]
            share = (a * u).sum(0)[cells_pos]
            contribution[k, nodes] = share.double().cpu().numpy()
            change[k] = float(weights[k] @ (np.nan_to_num(rp.context[node]) - bias))
    if method == "decomposition":
        contribution = np.nan_to_num(contribution)
    contribution[np.arange(q), receivers] = 0.0 if method == "decomposition" else contribution[np.arange(q), receivers]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(np.nansum(contribution, axis=1) - change) / np.maximum(np.abs(change), 1e-12)
    return Attribution(method, None, receivers, contribution.astype(np.float32), change, gap, time.time() - started)


# ----------------------------------------------------------------------------------------------- choosing receivers


def separated_receivers(pos, candidates, labels=None, *, per_label: int = 50, min_distance: float = 150.0, seed: int = 0) -> np.ndarray:
    """Up to ``per_label`` receivers per label (rare labels first), every two at least ``min_distance`` apart (units of ``pos``): so a
    perturbation of one receiver's nearby cells hardly reaches another, and one forward serves all of them (:func:`sensitivity_n`)."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    candidates = np.asarray(candidates, dtype=bool)
    labels = np.zeros(len(pos), dtype=object) if labels is None else np.asarray(labels).astype(str)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    tree_points: list = []
    names, counts = np.unique(labels[candidates], return_counts=True)
    for name in names[np.argsort(counts, kind="stable")]:
        taken = 0
        for cell in rng.permutation(np.flatnonzero(candidates & (labels == name))):
            if taken >= per_label:
                break
            if tree_points and np.min(np.linalg.norm(np.asarray(tree_points) - pos[cell], axis=1)) < min_distance:
                continue
            chosen.append(int(cell))
            tree_points.append(pos[cell])
            taken += 1
    return np.asarray(sorted(chosen), dtype=np.int64)


def near_candidates(pos, receivers, movable, *, k: int = 32) -> np.ndarray:
    """``[Q, k]`` each receiver's ``k`` nearest movable cells (other than itself; -1 pads)."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    movable = np.flatnonzero(np.asarray(movable, dtype=bool))
    out = np.full((len(receivers), k), -1, dtype=np.int64)
    if not len(movable):
        return out
    tree = cKDTree(pos[movable])
    kk = min(k + 1, len(movable))
    _, near = tree.query(pos[np.asarray(receivers)], k=kk)
    near = movable[np.atleast_2d(near)]
    for i, q in enumerate(receivers):
        row = near[i][near[i] != q][:k]
        out[i, : len(row)] = row
    return out


# ----------------------------------------------------------------------------------------------- A1: against planted truth and other methods


def planted_top_k(attr: Attribution, planted: dict, *, k: int = 5) -> np.ndarray:
    """``[Q]`` bool (NaN-free; receivers without planted senders are False and should be left out): a planted sender of the receiver is
    among its ``k`` cells of largest ``|contribution|``. ``planted``: receiver node -> set of sender nodes."""
    out = np.zeros(len(attr.receivers), dtype=bool)
    for i, q in enumerate(attr.receivers):
        senders = planted.get(int(q))
        if not senders:
            continue
        row = np.nan_to_num(np.abs(attr.contribution[i].astype(np.float64)), nan=-1.0)
        row[q] = -1.0
        top = np.argsort(-row, kind="stable")[:k]
        out[i] = bool(set(top.tolist()) & set(senders))
    return out


def agreement(a: Attribution, b: Attribution, *, top: int = 32, k: int = 5) -> dict:
    """Per receiver, over the union of each method's ``top`` cells by ``|contribution|``: Spearman of the signed contributions, and the
    overlap of the two top-``k`` sets (share). Returns ``{"spearman": [Q], "top_overlap": [Q]}``."""
    if not np.array_equal(a.receivers, b.receivers):
        raise ValueError("the two attributions read different receivers.")
    rho, overlap = np.full(len(a.receivers), np.nan), np.full(len(a.receivers), np.nan)
    for i, q in enumerate(a.receivers):
        x, y = a.contribution[i].astype(np.float64), b.contribution[i].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        ok[q] = False
        cells = np.flatnonzero(ok)
        if len(cells) < 3:
            continue
        ta = cells[np.argsort(-np.abs(x[cells]), kind="stable")[:top]]
        tb = cells[np.argsort(-np.abs(y[cells]), kind="stable")[:top]]
        union = np.union1d(ta, tb)
        if len(union) >= 3 and np.std(x[union]) > 0 and np.std(y[union]) > 0:
            rho[i] = spearmanr(x[union], y[union]).statistic
        overlap[i] = len(set(ta[:k].tolist()) & set(tb[:k].tolist())) / max(k, 1)
    return {"spearman": rho, "top_overlap": overlap}


# ----------------------------------------------------------------------------------------------- A6: Sensitivity-n and pairwise interactions


def _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, replace_rows, removal: str = "reference") -> np.ndarray:
    """``[Q]`` targets with the cells ``replace_rows`` removed: tokens set to their references, or hidden by the model's mask (no gradient)."""
    if removal == "mask":
        with torch.no_grad():
            if not len(replace_rows):
                return _Reader(module, batch, receivers, weights, head)(tokens.clone()).cpu().numpy()
            return _masked_read(_Reader(module, batch, receivers, weights, head), tokens, replace_rows).cpu().numpy()
    changed = tokens.clone()
    if len(replace_rows):
        idx = torch.as_tensor(np.asarray(replace_rows, dtype=np.int64), device=tokens.device)
        changed[idx] = ref[idx]
    with torch.no_grad():
        return _Reader(module, batch, receivers, weights, head)(changed).cpu().numpy()


@contextmanager
def _evaluating(module):
    was = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was)


def sensitivity_n(
    module,
    batch,
    receivers,
    weights,
    attributions: dict,
    candidates: np.ndarray,
    *,
    ns=(1, 2, 4, 8, 16),
    draws: int = 30,
    labels=None,
    reference: str = "graph_mean",
    head: str = "global",
    removal: str = "reference",
    seed: int = 0,
) -> dict:
    """Ancona et al.'s Sensitivity-n on the receivers' own nearby cells: per draw and ``n``, every receiver has ``n`` random cells of its
    ``candidates`` row set to their reference tokens (all receivers in one forward), and the change of its target is correlated over draws
    with the sum of each attribution over the same cells. Returns ``{method: {n: [Q] Pearson r}}`` and ``"_change_sd"`` per n."""
    _check_removal(module, removal)
    rng = np.random.default_rng(seed)
    receivers = np.asarray(receivers, dtype=np.int64)
    with _evaluating(module), torch.no_grad():
        tokens = _actual_tokens(module, batch)
        ref = reference_tokens(tokens, labels if labels is not None else np.zeros(len(tokens)), reference, rng=rng)
        base = _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, [], removal)
        out = {name: {} for name in attributions}
        out["_change_sd"] = {}
        for n in ns:
            deltas, sums = [], {name: [] for name in attributions}
            for _ in range(int(draws)):
                picked = []
                for row in candidates:
                    pool = row[row >= 0]
                    picked.append(rng.choice(pool, size=min(int(n), len(pool)), replace=False) if len(pool) else np.zeros(0, np.int64))
                rows = np.unique(np.concatenate(picked)) if picked else np.zeros(0, np.int64)
                deltas.append(base - _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, rows, removal))
                for name, attr in attributions.items():
                    contrib = np.nan_to_num(attr.contribution.astype(np.float64))
                    sums[name].append(np.array([contrib[i, p].sum() for i, p in enumerate(picked)]))
            deltas = np.stack(deltas)  # [draws, Q]
            out["_change_sd"][n] = deltas.std(axis=0)
            for name in attributions:
                s = np.stack(sums[name])
                r = np.full(len(receivers), np.nan)
                for i in range(len(receivers)):
                    if deltas[:, i].std() > 0 and s[:, i].std() > 0:
                        r[i] = np.corrcoef(deltas[:, i], s[:, i])[0, 1]
                out[name][n] = r
    return out


def pairwise_interactions(
    module,
    batch,
    receivers,
    weights,
    candidates: np.ndarray,
    *,
    pairs: int = 6,
    labels=None,
    reference: str = "graph_mean",
    head: str = "global",
    removal: str = "reference",
    seed: int = 0,
) -> dict:
    """``I_jk = Delta_jk - Delta_j - Delta_k`` for ``pairs`` random pairs from each receiver's ``candidates`` (its nearby cells), ``Delta``
    the drop of the target when those cells take their reference tokens; all receivers in one forward per pair index and condition.
    Returns ``interaction``, ``single`` (``|Delta_j| + |Delta_k|``) and ``relative`` (``|I| / single``), each ``[Q, pairs]``."""
    _check_removal(module, removal)
    rng = np.random.default_rng(seed)
    receivers = np.asarray(receivers, dtype=np.int64)
    q = len(receivers)
    inter, single = np.full((q, pairs), np.nan), np.full((q, pairs), np.nan)
    with _evaluating(module), torch.no_grad():
        tokens = _actual_tokens(module, batch)
        ref = reference_tokens(tokens, labels if labels is not None else np.zeros(len(tokens)), reference, rng=rng)
        base = _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, [], removal)
        for p in range(pairs):
            js, ks = [], []
            for row in candidates:
                pool = row[row >= 0]
                if len(pool) >= 2:
                    j, k = rng.choice(pool, size=2, replace=False)
                else:
                    j = k = -1
                js.append(j), ks.append(k)
            js, ks = np.asarray(js), np.asarray(ks)
            ok = js >= 0
            d_j = base - _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, np.unique(js[ok]), removal)
            d_k = base - _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, np.unique(ks[ok]), removal)
            d_jk = base - _perturbed_targets(module, batch, receivers, weights, head, tokens, ref, np.unique(np.concatenate([js[ok], ks[ok]])), removal)
            inter[ok, p] = (d_jk - d_j - d_k)[ok]
            single[ok, p] = (np.abs(d_j) + np.abs(d_k))[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(single > 0, np.abs(inter) / single, np.nan)
    return {"interaction": inter, "single": single, "relative": relative}


# ----------------------------------------------------------------------------------------------- A3 / A4: maps and distance


def distance_profile(attr: Attribution, pos, bins) -> dict:
    """Per sender-receiver distance bin (``bins`` edges, units of ``pos``): ``mean_abs`` (mean ``|contribution|`` per cell in the bin),
    ``mean_signed``, ``share`` (the bin's share of all ``|contribution|``), ``cumulative``, ``cells`` (pairs in the bin)."""
    pos = np.asarray(pos, dtype=np.float64)[:, :2]
    bins = np.asarray(bins, dtype=np.float64)
    nb = len(bins) - 1
    total_abs, total_signed, count = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for i, q in enumerate(attr.receivers):
        c = attr.contribution[i].astype(np.float64)
        ok = np.isfinite(c)
        ok[q] = False
        d = np.linalg.norm(pos[ok] - pos[q], axis=1)
        b = np.digitize(d, bins) - 1
        inside = (b >= 0) & (b < nb)
        np.add.at(total_abs, b[inside], np.abs(c[ok][inside]))
        np.add.at(total_signed, b[inside], c[ok][inside])
        np.add.at(count, b[inside], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = total_abs / max(total_abs.sum(), 1e-30)
        return {"lower": bins[:-1], "upper": bins[1:], "mean_abs": total_abs / count, "mean_signed": total_signed / count,
                "share": share, "cumulative": np.cumsum(share), "cells": count}  # fmt: skip


def strengths(attr: Attribution, n: int, *, keep=None) -> dict:
    """Per cell: ``out`` (signed contribution sent, summed over the receivers), ``out_abs``, ``reads`` (receivers it was read for), and
    per receiver ``in`` (signed contribution received from other cells). ``keep`` ``[Q]`` bool: receivers to use (e.g. gap within limit)."""
    keep = np.ones(len(attr.receivers), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    c = np.nan_to_num(attr.contribution[keep].astype(np.float64))
    finite = np.isfinite(attr.contribution[keep])
    into = np.zeros(n)
    np.add.at(into, attr.receivers[keep], c.sum(axis=1))
    return {"out": c.sum(axis=0), "out_abs": np.abs(c).sum(axis=0), "reads": finite.sum(axis=0), "in": into}


def disc_strengths(attr: Attribution, groups, n: int, *, keep=None) -> dict:
    """A3 with disc targets: IG is linear in the target, so a disc's attribution is the sum of its receivers' rows. ``groups`` ``[Q]``
    disc index per receiver. ``out`` per cell: what it sends to discs it is not in; ``in`` per cell: its disc's received total from
    outside, divided over the disc's receivers."""
    keep = np.ones(len(attr.receivers), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    groups = np.asarray(groups)
    out, into = np.zeros(n), np.zeros(n)
    for g in np.unique(groups[keep]):
        rows = np.flatnonzero(keep & (groups == g))
        members = attr.receivers[rows]
        total = np.nan_to_num(attr.contribution[rows].astype(np.float64)).sum(axis=0)
        total[members] = 0.0
        out += total
        into[members] += total.sum() / len(members)
    return {"out": out, "in": into}


def map_fields(pos_um, out, into, visible, *, grid_um: float = 10.0) -> dict:
    """The fields :func:`flow_maps.plot_source_sink` reads, from per-cell out / in strengths (attribution in place of attention flow)."""
    pos = np.asarray(pos_um, dtype=np.float64)[:, :2]
    origin = pos.min(axis=0)
    shape = np.maximum(np.ceil((pos.max(axis=0) - origin) / grid_um).astype(int) + 1, 1)
    cells = np.zeros(tuple(shape))
    ij = np.floor((pos - origin) / grid_um).astype(int)
    np.add.at(cells, (ij[:, 0], ij[:, 1]), 1)
    return {"pos_um": pos, "flow_out": np.asarray(out, dtype=np.float64), "flow_in": np.asarray(into, dtype=np.float64),
            "visible": np.asarray(visible, dtype=np.int8), "flow_grid_um": float(grid_um), "flow_origin": origin, "flow_cells": cells}  # fmt: skip
