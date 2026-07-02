#!/usr/bin/env python3
"""Microbenchmark direct packed MXFP4 selected-expert matvec."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.gpu_runtime import _dequantize_mxfp4
from thinruntime.triton_kernels import TritonDecodeBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rows", type=int, default=5760)
    parser.add_argument("--cols", type=int, default=2880)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--block-m", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cols % 32:
        raise ValueError("--cols must be divisible by MXFP4 group size 32")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(17)
    blocks = torch.randint(
        0,
        256,
        (args.experts, args.rows, args.cols // 32, 16),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    )
    scales = torch.randint(
        124,
        130,
        (args.experts, args.rows, args.cols // 32),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    )
    vector = torch.randn(
        args.cols,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    indices = torch.arange(
        args.top_k,
        device=device,
        dtype=torch.long,
    )
    output = torch.empty(
        (args.top_k, args.rows),
        device=device,
        dtype=torch.bfloat16,
    )
    backend = TritonDecodeBackend(
        device,
        torch.bfloat16,
        args.cols,
        args.cols,
        args.cols,
        args.rows,
    )

    def direct() -> None:
        backend.mxfp4_selected_matvec(
            blocks,
            scales,
            vector,
            indices,
            output,
            block_m=args.block_m,
            num_warps=args.num_warps,
        )

    def expanded() -> None:
        weights = _dequantize_mxfp4(
            blocks.index_select(0, indices),
            scales.index_select(0, indices),
            dtype=torch.bfloat16,
        )
        output.copy_(torch.einsum("kcr,c->kr", weights, vector))

    direct_samples = measure(direct, args.warmup, args.repeats)
    expanded_samples = measure(
        expanded,
        max(1, args.warmup // 2),
        max(3, args.repeats // 4),
    )
    direct()
    direct_result = output.clone()
    expanded()
    expanded_result = output.clone()
    difference = (direct_result - expanded_result).abs()
    direct_ms = statistics.median(direct_samples)
    expanded_ms = statistics.median(expanded_samples)
    selected_weight_bytes = (
        args.top_k
        * args.rows
        * args.cols
        // 2
        + args.top_k * args.rows * args.cols // 32
    )
    report = {
        "experts": args.experts,
        "top_k": args.top_k,
        "rows": args.rows,
        "cols": args.cols,
        "block_m": args.block_m,
        "num_warps": args.num_warps,
        "direct_packed_median_ms": direct_ms,
        "expand_then_matvec_median_ms": expanded_ms,
        "speedup": expanded_ms / direct_ms,
        "selected_packed_bytes": selected_weight_bytes,
        "effective_bandwidth_gb_s": selected_weight_bytes / direct_ms / 1e6,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.float().mean()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                direct_result.float().flatten(),
                expanded_result.float().flatten(),
                dim=0,
            )
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")


def measure(
    operation: object,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


if __name__ == "__main__":
    main()
