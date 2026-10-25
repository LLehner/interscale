"""Speed, memory and agreement of the fused attention against dense, on synthetic sections of growing size.

The check stage 6 of ``.claude/attention_plan.md`` leaves to the cluster: the global module of the windowed model -- dense
softmax attention with the distance window over ``[--scale-min, --scale-max]`` µm, RoPE, centred queries and keys, the blocked
diagonal -- built once with ``attention.kind: dense`` and once with ``fused`` (the same weights), on one synthetic section of
``n`` cells spread uniformly at ``--density`` cells per mm² around ``--offset-um`` (a section's global coordinates), with random
tokens. For each ``n`` it times a training step (forward and backward of every layer, dropout 0) and its peak GPU memory, and
where dense attention fits it reports how far the fused outputs and the window's gradients are from dense. Nothing is trained
and no data is read.

    python scripts/fused_attention_benchmark.py --cells 2500 10000 30000 50000 100000

Run from the repo root on a GPU node. Dense attention is skipped where it runs out of memory, and so from then on.
"""

from __future__ import annotations

import argparse
import math
import time

import torch


def build(kind: str, args, n_cells: int):
    """The global module of the windowed model, dense or fused."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.module.base import GlobalModule

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.n_embed = args.embed
    params = cfg.model.global_component.parameters
    params.n_heads, params.num_layers, params.dim_feedforward = args.heads, args.layers, 4 * args.embed
    params.dropout_global, params.max_seq_len = 0.0, n_cells
    params.attention.kind = kind
    params.attention.center_qk = True
    params.attention.fused.chunk_pairs = args.chunk_pairs
    params.pe.bias = ["distance"]
    params.pe.distance.kind = "scales"
    params.pe.distance.scale_min_um, params.pe.distance.scale_max_um = args.scale_min, args.scale_max
    params.pe.rotary = ["rope"]
    params.pe.rope.min_wavelength, params.pe.rope.max_wavelength = args.scale_min, 4 * args.scale_max
    cfg.dataset.spatial_key = "spatial"
    cfg.dataset.spatial_unit_um = 1.0
    return GlobalModule.from_config(
        cfg,
        local_mask_hops=0,
        n_input=args.embed,
        n_output=args.embed,
        n_embed=args.embed,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[args.embed],
        mask_percentage=0.1,
        mask_strategy="node",
    )


class Section:
    """One synthetic section: ``n`` cells uniform on a square at ``density`` per mm², around ``offset`` µm."""

    def __init__(self, n: int, density: float, offset: float, device, seed: int = 0):
        side_um = math.sqrt(n / density) * 1000.0
        gen = torch.Generator().manual_seed(seed)
        self.pos = (torch.rand(n, 2, generator=gen) * side_um + offset).to(device)
        self.batch = torch.zeros(n, dtype=torch.long, device=device)
        self.edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        self.mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.obs_names = torch.arange(n)
        self.num_nodes = n


def step(module, section, tokens):
    """A training step's forward and backward through every layer; returns the output."""
    module.zero_grad(set_to_none=True)
    padded, padding, _, mask, rotary = module.common_step_local_to_global(section, tokens, eval_step=True)
    out, _, _ = module.forward(padded, padding, mask, rotary=rotary)
    out.pow(2).mean().backward()
    return out.detach()


def window_grad(module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for n, p in module.named_parameters() if "log_scale" in n or "strength" in n])


def measure(module, section, tokens, steps: int, device) -> tuple[float, float, torch.Tensor]:
    """Mean seconds per step, peak GiB, and the last output."""
    step(module, section, tokens)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        out = step(module, section, tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / steps
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    return seconds, peak, out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", type=int, nargs="+", default=[2500, 10000, 30000, 50000, 100000])
    parser.add_argument("--density", type=float, default=2000.0, help="cells per mm²")
    parser.add_argument("--offset-um", type=float, default=0.0, help="where the section sits, as global coordinates")
    parser.add_argument("--scale-min", type=float, default=15.0, help="the narrowest window, µm")
    parser.add_argument("--scale-max", type=float, default=1000.0, help="the widest window, µm")
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--chunk-pairs", type=int, default=2**24)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    print(
        f"{args.heads} heads x {args.embed // args.heads}, {args.layers} layers, window {args.scale_min:g}-{args.scale_max:g} um, "
        f"{args.density:g} cells/mm2 around {args.offset_um:g} um, chunk_pairs {args.chunk_pairs}, {device}"
    )
    header = f"{'cells':>8} {'blocks':>7} {'dense s/step':>13} {'dense GiB':>10} {'fused s/step':>13} {'fused GiB':>10} {'max |out diff|':>15} {'window grad rel diff':>21}"
    print(header)
    dense_fits = True
    for n in args.cells:
        section = Section(n, args.density, args.offset_um, device)
        tokens = torch.randn(n, args.embed, generator=torch.Generator().manual_seed(1)).to(device)
        torch.manual_seed(0)
        fused = build("fused", args, n).to(device).train()
        with torch.no_grad():
            for p in fused.parameters():  # off the initialisation, where the window's strength is 1 and RoPE's tables 0
                p.add_(0.05 * torch.randn_like(p))
        blocks = math.ceil((n + 1) / fused.transformer_encoder.layers[0].self_attn.block_rows(1, n + 1))
        fused_s, fused_gib, out_fused = measure(fused, section, tokens, args.steps, device)
        grad_fused = window_grad(fused)

        dense_s = dense_gib = diff = rel = float("nan")
        if dense_fits:
            try:
                dense = build("dense", args, n).to(device).train()
                dense.load_state_dict(fused.state_dict())
                dense_s, dense_gib, out_dense = measure(dense, section, tokens, args.steps, device)
                diff = (out_fused - out_dense).abs().max().item()
                grad_dense = window_grad(dense)
                rel = ((grad_fused - grad_dense).norm() / grad_dense.norm().clamp_min(1e-30)).item()
                del dense, out_dense
            except torch.cuda.OutOfMemoryError:
                dense_fits = False
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dense_col = (
            f"{dense_s:13.3f} {dense_gib:10.2f}" if dense_fits or not math.isnan(dense_s) else f"{'OOM':>13} {'':>10}"
        )
        print(f"{n:8d} {blocks:7d} {dense_col} {fused_s:13.3f} {fused_gib:10.2f} {diff:15.2e} {rel:21.2e}", flush=True)
        del fused
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
