"""Readouts of the bilinear part (``model.decoder.bilinear``): the gene-to-gene map ``B = U V^T`` and what it says.

The part is ``y_i = sum_j a_ij (x_j V) U^T``: ``a_ij`` the last layer's attention of cell ``i`` over cell ``j``, ``V``
``[n_input, r]`` the sender side (which genes of a neighbour are read), ``U`` ``[n_output, r]`` the receiver side (which genes
of the cell they move). ``B[g, h] = sum_k U[g, k] V[h, k]`` is the weight with which gene ``h`` in the neighbourhood (attention-
weighted) moves gene ``g`` in the cell: the cross-cell gene map, read from the parameters alone, with no attribution pass. It
is the model-side counterpart of a band ridge's coefficient matrix (``.claude/ceiling_plan.md``), on the same footing: a linear
map from neighbours' genes to the cell's genes, the attention playing the band.

What is and is not in it: ``B`` is in the units of the input (log expression) times the output; a row ``g`` is a receiver
programme's loading on every sender gene; a column ``h`` a sender gene's effect on every receiver gene. The ``r`` components
(``U[:, k]`` receiver programme ``k``, ``V[:, k]`` sender programme ``k``) are one basis of many: a rotation of ``(U, V)`` by any
invertible ``R`` (``U R``, ``V R^-T``) leaves ``B`` unchanged, so read ``B`` and its SVD, and take the raw components as a
convenience only. Where the part reads is the attention's business (``flow_maps``); ``B`` says what it reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _module(model_or_module):
    module = getattr(model_or_module, "module", model_or_module)
    if not getattr(module, "bilinear_rank", 0):
        raise ValueError("this model has no bilinear part (model.decoder.bilinear.rank 0).")
    return module


def gene_map(model_or_module, var_names, output_names=None) -> pd.DataFrame:
    """``B = U V^T`` as a DataFrame, rows the receiver (output) genes, columns the sender (input) genes."""
    module = _module(model_or_module)
    U, V = module.bilinear_receiver.detach().cpu().double(), module.bilinear_sender.detach().cpu().double()
    B = (U @ V.T).numpy()
    var_names = list(var_names)
    output_names = var_names if output_names is None else list(output_names)
    if B.shape != (len(output_names), len(var_names)):
        raise ValueError(f"B is {B.shape}; got {len(output_names)} output and {len(var_names)} input names.")
    return pd.DataFrame(B, index=pd.Index(output_names, name="receiver_gene"), columns=pd.Index(var_names, name="sender_gene"))


def components(model_or_module, var_names, output_names=None, top: int = 10, rotation: str = "svd") -> pd.DataFrame:
    """The top genes of each component: one row per (component, side, rank).

    ``rotation`` ``"svd"``: the components are the singular vectors of ``B`` (unique up to sign, ordered by singular value; the
    reading that does not depend on the parametrisation); ``"raw"``: the columns of ``U`` and ``V`` as trained. Each component's
    largest-magnitude gene is made positive, so a sign is comparable between runs. ``weight`` is the component's singular value
    (``svd``) or ``||U[:, k]|| ||V[:, k]||`` (``raw``).
    """
    module = _module(model_or_module)
    var_names = list(var_names)
    output_names = var_names if output_names is None else list(output_names)
    U, V = module.bilinear_receiver.detach().cpu().double().numpy(), module.bilinear_sender.detach().cpu().double().numpy()
    if rotation == "svd":
        left, s, right_t = np.linalg.svd(U @ V.T, full_matrices=False)
        k = min(U.shape[1], len(s))
        receivers, senders, weights = left[:, :k], right_t[:k].T, s[:k]
    elif rotation == "raw":
        receivers, senders = U, V
        weights = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0)
    else:
        raise ValueError(f"rotation must be 'svd' or 'raw', got {rotation!r}.")
    rows = []
    for k in range(receivers.shape[1]):
        r, s_ = receivers[:, k].copy(), senders[:, k].copy()
        if r[np.argmax(np.abs(r))] < 0:  # sign: the receiver side's largest gene positive, the sender side follows
            r, s_ = -r, -s_
        for side, vec, names in (("receiver", r, output_names), ("sender", s_, var_names)):
            order = np.argsort(-np.abs(vec))[:top]
            for rank, g in enumerate(order):
                rows.append({"component": k, "weight": float(weights[k]), "side": side, "rank": rank, "gene": names[g],
                             "loading": float(vec[g])})
    return pd.DataFrame(rows)


def planted_recovery(B: pd.DataFrame, pairs) -> pd.DataFrame:
    """For planted (sender gene, receiver gene) pairs: the rank of the planted sender in the receiver's row of ``B``.

    ``pairs``: an iterable of ``(sender_gene, receiver_gene)``. Per pair: ``weight`` ``B[receiver, sender]``, ``rank`` of
    ``|B[receiver, sender]|`` among ``|B[receiver, :]|`` (1 = the largest), ``top_sender`` the gene with the largest magnitude in
    that row, ``hit`` whether it is the planted one, and ``share`` the pair's share of the row's squared mass. A row of zeros
    (an untrained part) ranks every gene equally: ``rank`` is then ``n_senders`` and ``hit`` False.
    """
    rows = []
    n = B.shape[1]
    for sender, receiver in pairs:
        if receiver not in B.index or sender not in B.columns:
            rows.append({"sender": sender, "receiver": receiver, "weight": np.nan, "rank": np.nan, "top_sender": None,
                         "hit": False, "share": np.nan})
            continue
        row = B.loc[receiver].to_numpy(dtype=float)
        mag = np.abs(row)
        w = float(B.loc[receiver, sender])
        if not np.any(mag > 0):
            rows.append({"sender": sender, "receiver": receiver, "weight": w, "rank": n, "top_sender": None, "hit": False,
                         "share": 0.0})
            continue
        rank = int((mag > abs(w)).sum()) + 1
        top = B.columns[int(np.argmax(mag))]
        rows.append({"sender": sender, "receiver": receiver, "weight": w, "rank": rank, "top_sender": top,
                     "hit": bool(top == sender), "share": float(w * w / (mag * mag).sum())})
    return pd.DataFrame(rows)


def compare_maps(B: pd.DataFrame, other: pd.DataFrame, top: int = 100) -> dict:
    """How two gene maps agree: ``other`` on the same gene names (a band ridge's coefficients, another seed's ``B``).

    ``cosine``: of the two matrices flattened, over the shared genes; ``row_cosine_median``: the median over receiver rows of
    the cosine of the two rows; ``top_overlap``: the share of the ``top`` largest-magnitude entries of ``B`` that are among
    the ``top`` largest of ``other``. Genes missing from either side are left out.
    """
    rows = B.index.intersection(other.index)
    cols = B.columns.intersection(other.columns)
    a, b = B.loc[rows, cols].to_numpy(dtype=float), other.loc[rows, cols].to_numpy(dtype=float)

    def cos(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")

    row_cos = [cos(a[i], b[i]) for i in range(a.shape[0]) if np.linalg.norm(a[i]) > 0 and np.linalg.norm(b[i]) > 0]
    k = min(top, a.size)
    top_a = set(np.argsort(-np.abs(a).ravel())[:k].tolist())
    top_b = set(np.argsort(-np.abs(b).ravel())[:k].tolist())
    return {
        "cosine": cos(a.ravel(), b.ravel()),
        "row_cosine_median": float(np.median(row_cos)) if row_cos else float("nan"),
        "top_overlap": len(top_a & top_b) / k if k else float("nan"),
        "n_receivers": int(len(rows)),
        "n_senders": int(len(cols)),
    }


def pair_weights(module, attn_last, x_visible: torch.Tensor, gene: int) -> torch.Tensor:
    """``[B, S, S]``: each pair's contribution to receiver gene ``gene``, ``a_ij * (x_j B)[gene]`` -- the arrows of a per-gene flow map.

    ``attn_last`` ``[B * heads, S + 1, S + 1]`` (the last layer's map), ``x_visible`` ``[B, S, n_input]`` the keys' visible genes in
    token layout (``tl.padding.pad_like``; a hidden entry 0). Head mean or the module's head, as the part reads. For
    ``flow_maps``: weight the pair arrows by this instead of the attention alone to draw where gene ``gene`` is received from.
    """
    n_graphs, seq_len = x_visible.shape[0], x_visible.shape[1]
    heads = attn_last.shape[0] // n_graphs
    maps = attn_last.view(n_graphs, heads, attn_last.shape[1], attn_last.shape[2])[:, :, :seq_len, :seq_len]
    a = maps.mean(dim=1) if module.bilinear_head < 0 else maps[:, module.bilinear_head]
    b_col = (module.bilinear_receiver[gene] @ module.bilinear_sender.T).to(x_visible.dtype)  # [n_input]: B[gene, :]
    sent = x_visible @ b_col  # [B, S]
    return a.to(sent.dtype) * sent[:, None, :]
