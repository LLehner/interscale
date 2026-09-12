"""Generate a synthetic spatial transcriptomics dataset with known niches and interactions.

The dataset is a benchmark for architecture experiments: every pattern the model is supposed to
find is written into ``adata`` as ground truth, so a change to the encoder can be scored rather
than eyeballed.

Layout (defaults): two conditions x 6 donors x 2 slides = 24 slides of 1500 cells on a
1000 x 1000 unit square, measured over 42 genes. One slide is one PyG graph and one transformer sequence -- no sliding
windows -- so ``cfg.model.global_component.parameters.max_seq_len`` must be >= the cells per
slide, otherwise :func:`interscale.tl.padding.pad_batch` subsamples tokens at random and the
token-to-cell mapping the attention regression depends on is lost.

The train/val/test split is assigned **per donor**, never per cell: ``prepare_geome_dataset``
subsets by split before building the neighbour graph, so a cell-level split would cut every graph
into three partial graphs. Splitting by donor also makes the condition label a genuine
out-of-sample question rather than a memorised slide identity.

See ``synth_data_0.md`` for a prose description of the generative model.

Run as a script to write the ``.h5ad``::

    python src/interscale/evaluation/synthetic_data.py --out data/synth_data_0.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import ndtr

CONDITIONS = ("healthy", "diseased")

#: Cell types. ``senderA``/``receiverA`` carry the short-range program, ``senderB``/``receiverB``
#: the mid-range one, ``hub`` is the tissue-scale source and ``stroma`` is inert filler.
CELL_TYPES = ("senderA", "receiverA", "senderB", "receiverB", "hub", "stroma")

NICHES = ("cortex", "medulla", "stroma_rich")

#: Cell-type composition per niche. ``medulla`` deliberately places ``senderB`` next to
#: ``receiverA`` although no senderB -> receiverA interaction exists: that pair is the negative
#: control that separates "attention follows co-occurrence" from "attention follows signalling".
NICHE_COMPOSITION = {
    "cortex": {"senderA": 0.30, "receiverA": 0.30, "senderB": 0.05, "receiverB": 0.05, "stroma": 0.30},
    "medulla": {"senderA": 0.05, "receiverA": 0.25, "senderB": 0.25, "receiverB": 0.25, "stroma": 0.20},
    "stroma_rich": {"senderA": 0.10, "receiverA": 0.10, "senderB": 0.10, "receiverB": 0.10, "stroma": 0.60},
}


def _gene_table(
    n_noise_genes: int, lr_range: float, mid_range: float, long_range: float, effect_scale: float
) -> pd.DataFrame:
    """Build the gene annotation, i.e. which gene belongs to which program.

    Parameters
    ----------
    n_noise_genes
        Number of structure-free genes. The remaining 18 genes are the annotated programs.
    lr_range, mid_range, long_range
        Kernel widths of the contact-range ligand/receptor pair and of the mid- and long-range
        interaction programs, in coordinate units.
    effect_scale
        Global multiplier on every log-fold effect; the knob for "how hard is this dataset".

    Returns
    -------
    pandas.DataFrame
        Indexed by gene name, with ``program``, ``effect_size``, ``true_length_scale``,
        ``target_cell_type`` and ``is_spatial``.
    """
    rows = []
    for i in range(n_noise_genes):
        rows.append({"gene": f"noise_{i:02d}", "program": "noise", "effect_size": 0.0})

    for ct in CELL_TYPES:
        rows.append({"gene": f"mark_{ct}", "program": "celltype_marker", "effect_size": 1.6, "target_cell_type": ct})

    # Two broad, monotone gradients in distance-from-centre and two with a sharper length scale,
    # so a Moran's-I style length-scale readout has something to separate.
    rows += [
        {"gene": "grad_up_broad", "program": "gradient", "effect_size": 1.2, "true_length_scale": np.inf},
        {"gene": "grad_down_broad", "program": "gradient", "effect_size": -1.2, "true_length_scale": np.inf},
        {"gene": "grad_center_sharp", "program": "gradient", "effect_size": 1.5, "true_length_scale": 150.0},
        {"gene": "grad_edge_sharp", "program": "gradient", "effect_size": 1.5, "true_length_scale": 300.0},
    ]

    # The interaction genes, shortest range first. The ligand/receptor pair leads them: it is the
    # most direct of the four, a coupling between the expression of two touching cells rather than
    # between two cell types, so no cell-type covariate can account for it.
    rows += [
        {"gene": "lig_LR1", "program": "interaction_lr", "effect_size": 1.2, "true_length_scale": lr_range},
        {"gene": "rec_LR1", "program": "interaction_lr", "effect_size": 1.2, "true_length_scale": lr_range},
        {
            "gene": "int_short",
            "program": "interaction_short",
            "effect_size": 0.8,
            "true_length_scale": 30.0,
            "target_cell_type": "receiverA",
        },
        {
            "gene": "int_mid",
            "program": "interaction_mid",
            "effect_size": 1.0,
            "true_length_scale": mid_range,
            "target_cell_type": "receiverB",
        },
        {
            "gene": "int_long",
            "program": "interaction_long",
            "effect_size": 1.3,
            "true_length_scale": long_range,
        },
    ]

    rows += [
        {"gene": "cond_up_1", "program": "condition_de", "effect_size": 1.0},
        {"gene": "cond_up_2", "program": "condition_de", "effect_size": 0.8},
        {"gene": "cond_down_1", "program": "condition_de", "effect_size": -1.0},
    ]

    genes = pd.DataFrame(rows).set_index("gene")
    genes["effect_size"] *= effect_scale
    genes["target_cell_type"] = genes.get("target_cell_type", pd.Series(dtype=object)).fillna("")
    genes["true_length_scale"] = genes.get("true_length_scale", pd.Series(dtype=float)).astype(float)
    genes["is_spatial"] = genes["program"].isin(
        ["gradient", "interaction_lr", "interaction_short", "interaction_mid", "interaction_long"]
    )
    return genes


def _sample_positions(rng: np.random.Generator, n: int, size: float) -> np.ndarray:
    """Sample cell coordinates with a smooth, slide-specific density modulation.

    The modulation is a nuisance on purpose: cell density drives neighbour counts and therefore
    both attention and QC statistics, so the downstream regression has a confounder to remove
    that has nothing to do with signalling.
    """
    centers = rng.uniform(0, size, size=(3, 2))
    widths = rng.uniform(0.25, 0.5, size=3) * size

    kept = []
    n_kept = 0
    while n_kept < n:
        cand = rng.uniform(0, size, size=(4 * n, 2))
        d2 = ((cand[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        density = np.exp(-d2 / (2 * widths[None, :] ** 2)).sum(1)
        accept = 0.4 + 0.6 * density / density.max()
        sel = cand[rng.random(len(cand)) < accept]
        kept.append(sel)
        n_kept += len(sel)
    return np.concatenate(kept)[:n]


def _assign_niches(rng: np.random.Generator, pos: np.ndarray, size: float) -> np.ndarray:
    """Assign each cell to one of :data:`NICHES` by a noisy nearest-centre rule."""
    centers = rng.uniform(0.2 * size, 0.8 * size, size=(len(NICHES), 2))
    d = cdist(pos, centers)
    # Noise on the distances softens the Voronoi boundaries, so niches interdigitate at their
    # borders the way real tissue domains do.
    d = d + rng.normal(0, 0.08 * size, size=d.shape)
    return np.asarray(NICHES, dtype=object)[d.argmin(1)]


def _assign_cell_types(rng: np.random.Generator, niche: np.ndarray, pos: np.ndarray, n_hub_cells: int) -> np.ndarray:
    """Draw a cell type per cell from its niche composition, then carve out the hub cluster."""
    cell_type = np.empty(len(niche), dtype=object)
    for nb in NICHES:
        m = niche == nb
        probs = np.array([NICHE_COMPOSITION[nb][ct] for ct in CELL_TYPES if ct != "hub"])
        probs = probs / probs.sum()
        types = [ct for ct in CELL_TYPES if ct != "hub"]
        cell_type[m] = rng.choice(types, size=int(m.sum()), p=probs)

    # The hub is one compact cluster per slide, not a dispersed cell type: it is the source of
    # the tissue-scale program, so its geometry has to be a single focus.
    hub_center = pos[rng.integers(len(pos))]
    hub_idx = np.argsort(((pos - hub_center) ** 2).sum(1))[:n_hub_cells]
    cell_type[hub_idx] = "hub"
    return cell_type


def _log_effects(
    genes: pd.DataFrame,
    obs: dict[str, np.ndarray],
    condition: str,
    slide_size: float,
) -> np.ndarray:
    """Per-cell, per-gene log-fold effect on the ZINB mean.

    Every program is an additive term in log-space, so the whole dataset stays one generative
    model and effect sizes are directly comparable across programs.
    """
    n = len(obs["cell_type"])
    eff = np.zeros((n, len(genes)), dtype=np.float64)
    diseased = condition == "diseased"
    s = obs["dist_to_center"] / (0.5 * slide_size)

    for j, (gene, row) in enumerate(genes.iterrows()):
        beta = row["effect_size"]
        program = row["program"]
        if program == "celltype_marker":
            eff[:, j] = beta * (obs["cell_type"] == row["target_cell_type"])
        elif program == "gradient":
            if gene == "grad_up_broad":
                eff[:, j] = beta * s
            elif gene == "grad_down_broad":
                eff[:, j] = beta * s
            elif gene == "grad_center_sharp":
                eff[:, j] = beta * np.exp(-obs["dist_to_center"] / row["true_length_scale"])
            elif gene == "grad_edge_sharp":
                eff[:, j] = beta * (1.0 - np.exp(-obs["dist_to_center"] / row["true_length_scale"]))
        elif program == "interaction_lr":
            # The ligand follows the cell's own tone, the receptor follows the tone of the cells
            # touching it. A cell with a high ligand therefore sits next to cells with a high
            # receptor, and a cell with a high receptor sits next to cells with a high ligand --
            # the coupling holds in both directions and never mentions cell type.
            eff[:, j] = beta * (obs["lr_tone"] if gene == "lig_LR1" else obs["lr_neighbor_tone"])
        elif program == "interaction_short":
            target = obs["cell_type"] == row["target_cell_type"]
            eff[:, j] = beta * np.log1p(obs["n_senderA_short"]) * target
        elif program == "interaction_mid":
            target = obs["cell_type"] == row["target_cell_type"]
            eff[:, j] = beta * np.log1p(obs["kern_senderB_mid"]) * target
        elif program == "interaction_long":
            # Only the diseased slides carry the tissue-scale program, so detecting it requires
            # both long-range context and the condition -- it cannot be explained by geometry alone.
            eff[:, j] = beta * obs["hub_response"] * diseased
        elif program == "condition_de":
            eff[:, j] = beta * diseased
    return eff


def _simulate_slide(
    rng: np.random.Generator,
    genes: pd.DataFrame,
    *,
    condition: str,
    donor: str,
    slide: str,
    n_cells: int,
    slide_size: float,
    lr_range: float,
    short_range: float,
    mid_range: float,
    long_range: float,
    n_hub_cells: int,
    edge_weight_cutoff: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Simulate one slide: coordinates, annotations, counts and the ground-truth edge list."""
    pos = _sample_positions(rng, n_cells, slide_size)
    niche = _assign_niches(rng, pos, slide_size)
    cell_type = _assign_cell_types(rng, niche, pos, n_hub_cells)

    dist = cdist(pos, pos)
    is_sender_a = cell_type == "senderA"
    is_sender_b = cell_type == "senderB"
    is_hub = cell_type == "hub"

    # Short range: a plain count of senderA inside `short_range`, i.e. inside the 2-hop reach of
    # the local GCN. Mid range: a Gaussian kernel wider than that reach but well inside a slide.
    # Contact-range ligand/receptor coupling. `lr_tone` is an independent draw per cell -- it is
    # not a smooth spatial field, so the pair is detectable only from the expression of individual
    # touching cells, never from the neighbourhood a cell sits in.
    lr_tone = rng.normal(0.0, 1.0, size=n_cells)
    contact = (dist <= lr_range) & ~np.eye(n_cells, dtype=bool)
    n_contacts = contact.sum(1)
    lr_neighbor_tone = (contact @ lr_tone) / np.maximum(n_contacts, 1)

    n_senderA_short = ((dist <= short_range) & is_sender_a[None, :]).sum(1) - is_sender_a
    kern_senderB_mid = (np.exp(-(dist**2) / (2 * mid_range**2)) * is_sender_b[None, :]).sum(1) - is_sender_b

    center = np.full(2, 0.5 * slide_size)
    dist_to_center = np.linalg.norm(pos - center, axis=1)
    hub_centroid = pos[is_hub].mean(0)
    dist_to_hub = np.linalg.norm(pos - hub_centroid, axis=1)
    hub_response = np.exp(-dist_to_hub / long_range)

    obs_arrays = {
        "cell_type": cell_type,
        "niche": niche,
        "lr_tone": lr_tone,
        "lr_neighbor_tone": lr_neighbor_tone,
        "dist_to_center": dist_to_center,
        "dist_to_hub": dist_to_hub,
        "hub_response": hub_response,
        "n_senderA_short": n_senderA_short.astype(float),
        "kern_senderB_mid": kern_senderB_mid,
    }

    log_eff = _log_effects(genes, obs_arrays, condition, slide_size)

    # Library size: a per-cell lognormal times a per-slide factor. The slide factor is the batch
    # effect the downstream regression is asked to partial out.
    slide_factor = np.exp(rng.normal(0, 0.35))
    lib_factor = np.exp(rng.normal(0, 0.25, size=n_cells)) * slide_factor

    mu = genes["base_mu"].to_numpy()[None, :] * np.exp(log_eff) * lib_factor[:, None]
    theta = genes["theta"].to_numpy()[None, :]
    rate = rng.gamma(shape=np.broadcast_to(theta, mu.shape), scale=mu / theta)
    counts = rng.poisson(rate).astype(np.float32)
    dropout = rng.random(counts.shape) < genes["pi"].to_numpy()[None, :]
    counts[dropout] = 0.0

    obs = pd.DataFrame(
        {
            "condition": condition,
            "donor": donor,
            "slide": slide,
            "cell_type": cell_type,
            "niche": niche,
            "lr_tone": lr_tone,
            "lr_neighbor_tone": lr_neighbor_tone,
            "n_contacts": n_contacts.astype(float),
            "dist_to_center": dist_to_center,
            "dist_to_hub": dist_to_hub,
            "hub_response": hub_response,
            "n_senderA_short": n_senderA_short.astype(float),
            "kern_senderB_mid": kern_senderB_mid,
            "lib_factor": lib_factor,
        }
    )

    edges = _ground_truth_edges(
        dist=dist,
        cell_type=cell_type,
        hub_response=hub_response,
        lr_tone=lr_tone,
        condition=condition,
        lr_range=lr_range,
        short_range=short_range,
        mid_range=mid_range,
        long_range=long_range,
        cutoff=edge_weight_cutoff,
    )
    edges["slide"] = slide
    return obs, pos, counts, edges


def _ground_truth_edges(
    *,
    dist: np.ndarray,
    cell_type: np.ndarray,
    hub_response: np.ndarray,
    lr_tone: np.ndarray,
    condition: str,
    lr_range: float,
    short_range: float,
    mid_range: float,
    long_range: float,
    cutoff: float,
) -> pd.DataFrame:
    """Slide-local list of pairs that actually influence each other.

    A pair is recorded when its kernel weight exceeds ``cutoff``, i.e. when the sender carries at
    least that fraction of the maximum possible per-pair effect. Indices are **positional within
    the slide**; :func:`make_synthetic` shifts them to positions in the full object, which stay
    valid even though ``prepare_geome_dataset`` renames ``obs_names``.
    """
    frames = []

    # Contact range: the ligand level of the sending cell is what raises the receptor in the cell
    # it touches, so the edge weight is that cell's ligand tone mapped onto [0, 1]. Every cell can
    # be a sender here -- this program is not restricted to a cell type.
    ligand_strength = ndtr(lr_tone)
    si, ri = np.nonzero((dist <= lr_range) & (ligand_strength[:, None] >= cutoff))
    if len(si):
        frames.append(
            pd.DataFrame({"sender": si, "receiver": ri, "weight": ligand_strength[si], "range_class": "contact"})
        )

    sa = np.flatnonzero(cell_type == "senderA")
    ra = np.flatnonzero(cell_type == "receiverA")
    if len(sa) and len(ra):
        w = (dist[np.ix_(sa, ra)] <= short_range).astype(float)
        si, ri = np.nonzero(w)
        frames.append(
            pd.DataFrame({"sender": sa[si], "receiver": ra[ri], "weight": 1.0, "range_class": "short"})
        )

    sb = np.flatnonzero(cell_type == "senderB")
    rb = np.flatnonzero(cell_type == "receiverB")
    if len(sb) and len(rb):
        w = np.exp(-(dist[np.ix_(sb, rb)] ** 2) / (2 * mid_range**2))
        si, ri = np.nonzero(w >= cutoff)
        frames.append(
            pd.DataFrame({"sender": sb[si], "receiver": rb[ri], "weight": w[si, ri], "range_class": "mid"})
        )

    if condition == "diseased":
        hub = np.flatnonzero(cell_type == "hub")
        resp = np.flatnonzero(hub_response >= cutoff)
        if len(hub) and len(resp):
            si, ri = np.meshgrid(np.arange(len(hub)), np.arange(len(resp)), indexing="ij")
            frames.append(
                pd.DataFrame(
                    {
                        "sender": hub[si.ravel()],
                        "receiver": resp[ri.ravel()],
                        "weight": hub_response[resp[ri.ravel()]],
                        "range_class": "long",
                    }
                )
            )

    if not frames:
        return pd.DataFrame(columns=["sender", "receiver", "weight", "range_class"])
    edges = pd.concat(frames, ignore_index=True)
    return edges[edges["sender"] != edges["receiver"]].reset_index(drop=True)


def make_synthetic(
    *,
    n_donors_per_condition: int = 6,
    n_slides_per_donor: int = 2,
    n_cells_per_slide: int = 1500,
    slide_size: float = 1000.0,
    n_noise_genes: int = 24,
    lr_range: float = 20.0,
    short_range: float = 30.0,
    mid_range: float = 150.0,
    long_range: float = 400.0,
    n_hub_cells: int = 15,
    effect_scale: float = 1.0,
    edge_weight_cutoff: float = 0.4,
    n_val_donors: int = 1,
    n_test_donors: int = 1,
    seed: int = 0,
) -> AnnData:
    """Simulate the full multi-slide dataset.

    Parameters
    ----------
    n_donors_per_condition, n_slides_per_donor, n_cells_per_slide
        Size of the design. One slide is one graph and one transformer sequence, so
        ``n_cells_per_slide`` must not exceed ``max_seq_len``.
    slide_size
        Side length of the square slide, in the same units as the interaction ranges.
    n_noise_genes
        Structure-free genes; the 18 annotated program genes are added on top.
    lr_range
        Contact radius of the ligand/receptor pair -- the distance at which two cells count as
        touching. Keep it below ``short_range``; at the default density it gives each cell about
        two contacts.
    short_range, mid_range, long_range
        The three cell-type-driven interaction length scales. ``short_range`` should sit inside the
        local component's reach (``spatial_neigbors_kwargs.radius`` x number of GCN layers) and
        ``mid_range`` outside it, so that the two scales are attributable to different components.
    n_hub_cells
        Size of the compact hub cluster that emits the tissue-scale program.
    effect_scale
        Multiplier on all log-fold effects; lower it to find an architecture's detection floor.
    edge_weight_cutoff
        Kernel weight above which a pair enters the ground-truth edge list.
    n_val_donors, n_test_donors
        Donors per condition held out for val and test. The rest are train.
    seed
        Seed for the single :class:`numpy.random.Generator` driving the simulation.

    Returns
    -------
    anndata.AnnData
        Normalised expression in ``X`` and ``layers['log1p_norm']``, raw counts in
        ``layers['counts']``, ground truth in ``var``, ``obs`` and ``uns['synthetic']``.
    """
    if n_donors_per_condition < n_val_donors + n_test_donors + 1:
        raise ValueError("need at least one train donor per condition")

    rng = np.random.default_rng(seed)

    genes = _gene_table(n_noise_genes, lr_range, mid_range, long_range, effect_scale)
    n_genes = len(genes)
    # Baseline abundance and overdispersion are gene properties, shared across all slides --
    # otherwise a "gene" would not mean the same thing in two slides.
    genes["base_mu"] = np.exp(rng.uniform(np.log(0.5), np.log(20.0), size=n_genes))
    genes["theta"] = np.exp(rng.uniform(np.log(1.0), np.log(10.0), size=n_genes))
    pi = rng.uniform(0.05, 0.35, size=n_genes)
    # Program genes get less dropout: a signal that is zero-inflated away is not a detectable
    # pattern, it is noise with extra steps.
    pi[genes["program"].to_numpy() != "noise"] = rng.uniform(0.02, 0.10, size=int((genes["program"] != "noise").sum()))
    genes["pi"] = pi

    obs_frames, pos_list, count_list, edge_frames = [], [], [], []
    offset = 0
    for condition in CONDITIONS:
        for d in range(n_donors_per_condition):
            donor = f"{condition[:1].upper()}{d + 1:02d}"
            if d < n_val_donors:
                split = "val"
            elif d < n_val_donors + n_test_donors:
                split = "test"
            else:
                split = "train"
            for s in range(n_slides_per_donor):
                slide = f"{donor}_s{s + 1}"
                obs, pos, counts, edges = _simulate_slide(
                    rng,
                    genes,
                    condition=condition,
                    donor=donor,
                    slide=slide,
                    n_cells=n_cells_per_slide,
                    slide_size=slide_size,
                    lr_range=lr_range,
                    short_range=short_range,
                    mid_range=mid_range,
                    long_range=long_range,
                    n_hub_cells=n_hub_cells,
                    edge_weight_cutoff=edge_weight_cutoff,
                )
                obs["split"] = split
                obs_frames.append(obs)
                pos_list.append(pos)
                count_list.append(counts)
                edges[["sender", "receiver"]] += offset
                edge_frames.append(edges)
                offset += n_cells_per_slide

    obs = pd.concat(obs_frames, ignore_index=True)
    X = np.concatenate(count_list, axis=0)
    spatial = np.concatenate(pos_list, axis=0)

    obs["total_counts"] = X.sum(1)
    obs["n_genes_by_counts"] = (X > 0).sum(1).astype(float)
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))], name=None)
    for col in ["condition", "donor", "slide", "cell_type", "niche", "split"]:
        obs[col] = pd.Categorical(obs[col])

    var = genes.drop(columns=["base_mu", "theta", "pi"]).copy()
    var.index.name = None

    # log1p of median-normalised counts. Must stay non-negative: tl.masking fills masked entries
    # with MASK_VALUE = -1, which has to sit outside the layer's range to be distinguishable.
    totals = np.asarray(X.sum(1))
    totals[totals == 0] = 1.0
    norm = np.log1p(X / totals[:, None] * np.median(totals)).astype(np.float32)

    # scanpy convention: .X holds the normalised values, raw counts live in layers["counts"].
    # log1p_norm is also kept under its own name, because the training configs address it by
    # name through cfg.dataset.layer_key rather than reading .X.
    adata = AnnData(X=csr_matrix(norm), obs=obs, var=var)
    adata.obsm["spatial"] = spatial.astype(np.float64)
    adata.layers["counts"] = csr_matrix(X)
    adata.layers["log1p_norm"] = csr_matrix(norm)

    edges = pd.concat(edge_frames, ignore_index=True)
    edges["sender"] = edges["sender"].astype(np.int32)
    edges["receiver"] = edges["receiver"].astype(np.int32)
    edges["weight"] = edges["weight"].astype(np.float32)
    edges["range_class"] = pd.Categorical(edges["range_class"])
    edges["slide"] = pd.Categorical(edges["slide"])

    adata.uns["synthetic"] = {
        "interaction_edges": edges,
        "params": {
            "n_donors_per_condition": n_donors_per_condition,
            "n_slides_per_donor": n_slides_per_donor,
            "n_cells_per_slide": n_cells_per_slide,
            "slide_size": slide_size,
            "lr_range": lr_range,
            "short_range": short_range,
            "mid_range": mid_range,
            "long_range": long_range,
            "n_hub_cells": n_hub_cells,
            "effect_scale": effect_scale,
            "edge_weight_cutoff": edge_weight_cutoff,
            "seed": seed,
        },
        "interactions": pd.DataFrame(
            [
                {"sender": "any", "receiver": "any touching cell", "gene": "lig_LR1/rec_LR1",
                 "range_class": "contact"},
                {"sender": "senderA", "receiver": "receiverA", "gene": "int_short", "range_class": "short"},
                {"sender": "senderB", "receiver": "receiverB", "gene": "int_mid", "range_class": "mid"},
                {"sender": "hub", "receiver": "all", "gene": "int_long", "range_class": "long"},
                {"sender": "senderB", "receiver": "receiverA", "gene": "", "range_class": "negative_control"},
            ]
        ),
        "niche_composition": pd.DataFrame(NICHE_COMPOSITION).T.fillna(0.0),
    }
    return adata


def _summary(adata: AnnData) -> str:
    """One-screen description of what was generated, printed by the CLI."""
    lines = [
        f"cells x genes: {adata.n_obs} x {adata.n_vars}",
        f"slides: {adata.obs['slide'].nunique()}  donors: {adata.obs['donor'].nunique()}",
        "split (cells): " + ", ".join(f"{k}={v}" for k, v in adata.obs["split"].value_counts().items()),
        "split (donors): "
        + ", ".join(
            f"{k}={v}" for k, v in adata.obs.groupby("split", observed=True)["donor"].nunique().items()
        ),
        "cell types: " + ", ".join(f"{k}={v}" for k, v in adata.obs["cell_type"].value_counts().items()),
        "programs: " + ", ".join(f"{k}={v}" for k, v in adata.var["program"].value_counts().items()),
        "ground-truth edges: "
        + ", ".join(
            f"{k}={v}" for k, v in adata.uns["synthetic"]["interaction_edges"]["range_class"].value_counts().items()
        ),
        f"median counts/cell: {np.median(adata.obs['total_counts']):.0f}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write the dataset to an ``.h5ad`` file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synth_data_0.h5ad"))
    parser.add_argument("--n-donors-per-condition", type=int, default=6)
    parser.add_argument("--n-slides-per-donor", type=int, default=2)
    parser.add_argument("--n-cells-per-slide", type=int, default=1500)
    parser.add_argument("--lr-range", type=float, default=20.0)
    parser.add_argument("--effect-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = make_synthetic(
        n_donors_per_condition=args.n_donors_per_condition,
        n_slides_per_donor=args.n_slides_per_donor,
        n_cells_per_slide=args.n_cells_per_slide,
        lr_range=args.lr_range,
        effect_scale=args.effect_scale,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    print(_summary(adata))
    print(f"\nwritten to {args.out}")
    print(f"set cfg.model.global_component.parameters.max_seq_len >= {args.n_cells_per_slide}")


if __name__ == "__main__":
    main()
