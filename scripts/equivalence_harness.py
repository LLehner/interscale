"""Acceptance gate for refactors that must change no behaviour.

Runs four short, fully deterministic trainings on a small synthetic dataset and dumps the
per-epoch metric history to JSON, so a before/after pair can be compared exactly. The four cases
are chosen to cover every `_common_step` implementation (local, global, combined, dual-decoder)
and all three distinct loss paths (element-wise, row-structured, attention-consuming).

This is what "verified" means for the plumbing stages in `.claude/contrastive_plan.md`: a
refactor is done when the tests pass *and* this reports IDENTICAL. Tests alone do not cover it --
every stage so far has passed the suite while the question was whether any number moved.

    python scripts/equivalence_harness.py /tmp/before.json
    # ... refactor ...
    python scripts/equivalence_harness.py /tmp/after.json
    python scripts/equivalence_harness.py --compare /tmp/before.json /tmp/after.json

Run from the repo root. Takes about a minute. The harness is itself checked for determinism by
running it twice with no change in between -- do that first if a comparison ever surprises you.
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

warnings.filterwarnings("ignore")

import interscale  # noqa: E402
from interscale.config import load_config  # noqa: E402
from interscale.geome_dataloader import GraphAnnDataModule  # noqa: E402
from interscale.tl import prepare_geome_dataset, set_full_reproducibility  # noqa: E402

N_PER_SAMPLE = 60
N_GENES = 24
SAMPLES = ["s1", "s2", "s3", "s4"]


def make_adata(path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = N_PER_SAMPLE * len(SAMPLES)
    # Structured, not pure noise: a latent field drives expression so reconstruction is learnable
    # and the metrics actually move between epochs.
    coords = rng.uniform(0, 100, size=(n, 2))
    latent = np.stack([np.sin(coords[:, 0] / 12), np.cos(coords[:, 1] / 9)], axis=1)
    loadings = rng.normal(size=(2, N_GENES))
    X = np.abs(latent @ loadings + rng.normal(scale=0.3, size=(n, N_GENES))).astype(np.float32)
    X += 0.05  # no all-zero cells -- remove_zero_expression_cells would drop them

    adata = AnnData(X=X)
    adata.obsm["spatial"] = coords
    adata.obs["sample"] = pd.Categorical(np.repeat(SAMPLES, N_PER_SAMPLE))
    adata.obs["split"] = pd.Categorical(
        [{"s1": "train", "s2": "train", "s3": "val", "s4": "test"}[s] for s in adata.obs["sample"]]
    )
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    adata.layers["log1p_norm"] = np.log1p(adata.X)
    adata.write_h5ad(path)
    return path


CASES = {
    # name: (model_type, cfg overrides)
    "combined_dual_smoothl1_node": (
        "CombinedModel",
        {"dual_decoder": True, "loss": "SmoothL1", "mask_strategy": "node"},
    ),
    "combined_single_sce_gene": (
        "CombinedModel",
        {"dual_decoder": False, "loss": "SCELoss", "mask_strategy": "gene"},
    ),
    "global_sce_entropy_node": (
        "GlobalModel",
        {"dual_decoder": False, "loss": "SCE_EntropyATT_Loss", "mask_strategy": "node"},
    ),
    "local_mse_node": (
        "LocalModel",
        {"dual_decoder": False, "loss": "MSELoss", "mask_strategy": "node"},
    ),
}


def write_cfg(path: Path, h5ad: Path, model_type: str, o: dict) -> Path:
    local = "  local_component:\n    name: GCN\n" if model_type != "GlobalModel" else ""
    gex = "      type_gex_embedding: PCA\n" if model_type == "GlobalModel" else ""
    # The global block is written even for LocalModel: prepare_geome_dataset reads
    # cfg.model.global_component.parameters.type_gex_embedding unconditionally, and that node
    # only exists once a global component name is set. LocalModel ignores it.
    glob = (
        "  global_component:\n    name: self-attn-transformer\n"
        "    parameters:\n      max_seq_len: 64\n      num_layers: 1\n      n_heads: 2\n" + gex
    )
    path.write_text(
        "model:\n"
        f"  n_embed: 8\n{local}{glob}"
        f"  decoder:\n    type: linear\n    dual_decoder: {o['dual_decoder']}\n"
        "  save: null\n"
        "optim:\n"
        f"  loss: {o['loss']}\n"
        "  lr: 0.01\n  seed: 42\n  accelerator: cpu\n  n_epochs: 3\n"
        "  lr_warmup: 1\n  min_epochs: 2\n  early_stopping: False\n"
        "dataset:\n"
        f"  h5ad_data: {h5ad}\n"
        "  name: equivtest\n  prediction_task: regression\n  prediction_level: node\n"
        "  layer_key: log1p_norm\n  sample_key: ['sample']\n"
        f"  mask_strategy: {o['mask_strategy']}\n  mask_percentage: 0.3\n"
        "  batch_size: 2\n"
        "  spatial_neigbors_kwargs:\n    radius: 25\n    library_key: sample\n"
    )
    return path


def run_case(name: str, model_type: str, overrides: dict, h5ad: Path, tmp: Path) -> list[dict]:
    cfg = load_config(str(write_cfg(tmp / f"{name}.yaml", h5ad, model_type, overrides)))
    set_full_reproducibility(cfg.optim.seed)

    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    cls = getattr(interscale.model, model_type)
    cls._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
    )
    model = cls(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg,
        num_workers=0,
        batch_size=int(cfg.dataset.batch_size),
        mask_percentage=cfg.dataset.mask_percentage,
        mask_strategy=cfg.dataset.mask_strategy,
        learning_type=cfg.dataset.prediction_level,
    )
    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=False, wandb_use=False)
    return [{k: round(float(v), 6) for k, v in ep.items()} for ep in model.history_.history]


def compare(a_path: str, b_path: str) -> int:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    bad = []
    for case in sorted(set(a) | set(b)):
        if case not in a or case not in b:
            bad.append(f"{case}: present in only one run")
            continue
        if len(a[case]) != len(b[case]):
            bad.append(f"{case}: {len(a[case])} epochs vs {len(b[case])}")
            continue
        for i, (ea, eb) in enumerate(zip(a[case], b[case], strict=True)):
            for k in sorted(set(ea) | set(eb)):
                va, vb = ea.get(k), eb.get(k)
                if va != vb:
                    bad.append(f"{case}[epoch {i}].{k}: {va} != {vb}")
    if bad:
        print("MISMATCH:")
        for line in bad[:40]:
            print(" ", line)
        print(f"({len(bad)} differences)")
        return 1
    n = sum(len(v) for v in a.values())
    print(f"IDENTICAL across {len(a)} cases / {n} epoch records")
    return 0


def main() -> int:
    if sys.argv[1] == "--compare":
        return compare(sys.argv[2], sys.argv[3])

    out = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        h5ad = make_adata(tmp / "equiv.h5ad")
        results = {}
        for name, (model_type, overrides) in CASES.items():
            print(f"--- {name} ---", flush=True)
            results[name] = run_case(name, model_type, overrides, h5ad, tmp)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
