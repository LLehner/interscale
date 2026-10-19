"""Which genes a precomputed embedding was made from, and the guard for programme masking.

An embedding computed outside InterScale (PCA, scVI, ...) is read through ``type_gex_embedding:
Precomputed``. Masking corrupts ``batch.x`` only, so it cannot hide a gene from an embedding that
was computed from it: under programme masking (``dataset.mask_genes``, a gene set hidden in every
cell) a precomputed embedding is safe only if it never saw those genes. The genes an embedding saw
are recorded next to it, ``adata.uns["embedding_genes"][<obsm key>]``, by :func:`record_embedding_genes`
when it is computed, and :func:`check_embedding_genes` refuses a hidden set that overlaps them -- and
an embedding with no record, which cannot be verified.
"""

from __future__ import annotations

from interscale.tl.masking import resolve_mask_genes

#: ``adata.uns`` key holding ``{obsm key: [gene names]}``.
EMBEDDING_GENES_KEY = "embedding_genes"


def record_embedding_genes(adata, key: str, genes) -> None:
    """Record that ``adata.obsm[key]`` was computed from the genes ``genes`` (names in ``adata.var_names``) only.

    Stored as an array of strings, so it survives ``write_h5ad``.
    """
    names = [str(g) for g in genes]
    unknown = sorted(set(names) - set(map(str, adata.var_names)))
    if unknown:
        raise ValueError(f"embedding '{key}' names genes that are not in adata.var_names: {unknown[:5]}")
    recorded = adata.uns.get(EMBEDDING_GENES_KEY)
    recorded = dict(recorded) if recorded is not None else {}
    recorded[str(key)] = names
    adata.uns[EMBEDDING_GENES_KEY] = recorded


def embedding_genes(adata, key: str) -> list[str] | None:
    """The genes ``adata.obsm[key]`` was recorded as computed from, or None when nothing was recorded."""
    recorded = adata.uns.get(EMBEDDING_GENES_KEY)
    if recorded is None or str(key) not in recorded:
        return None
    return [str(g) for g in recorded[str(key)]]


def check_embedding_genes(adata, cfg) -> None:
    """Refuse a ``Precomputed`` embedding that saw a gene programme masking hides, or whose genes are unknown.

    A no-op unless the model reads ``Precomputed`` embeddings and ``dataset.mask_genes`` hides a gene
    set. The record is :func:`record_embedding_genes`'; without one the embedding cannot be shown to
    exclude the hidden genes, so it is refused rather than trusted.
    """
    params = cfg.model.global_component.get("parameters", None)
    if params is None or params.get("type_gex_embedding", None) != "Precomputed":
        return
    selectors = list(cfg.dataset.get("mask_genes", []) or [])
    if not selectors:
        return
    key = cfg.model.global_component.get("latent_obsm_key", None)
    seen = embedding_genes(adata, key)
    if seen is None:
        raise ValueError(
            f"dataset.mask_genes hides a gene set, so the precomputed embedding obsm[{key!r}] must be computed "
            f"without those genes, but adata.uns['{EMBEDDING_GENES_KEY}'][{key!r}] does not say which genes it "
            "was computed from. Record them with interscale.tl.record_embedding_genes(adata, key, genes) "
            "when the embedding is made."
        )
    hidden = [str(adata.var_names[k]) for k in resolve_mask_genes(selectors, adata.var)]
    overlap = sorted(set(hidden) & set(seen))
    if overlap:
        raise ValueError(
            f"the precomputed embedding obsm[{key!r}] was computed from {len(overlap)} of the genes "
            f"dataset.mask_genes hides ({overlap[:5]}{' ...' if len(overlap) > 5 else ''}), so the hidden values "
            "would reach the model through it. Compute it from the other genes."
        )
