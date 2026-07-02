#!/usr/bin/env python3
"""Run the GPU-first ThinTensor runtime smoke/benchmark path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime import benchmark_gpu_runtime


def main() -> None:
    args = parse_args()
    dtype = parse_dtype(args.dtype)
    result = benchmark_gpu_runtime(
        args.archive,
        token_id=args.token_id,
        steps=args.steps,
        device=args.device,
        dtype=dtype,
        layers=args.layers,
        top_k=args.top_k,
        max_gpu_temp=args.max_gpu_temp,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_text(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["archive", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-gpu-temp", type=int, default=87)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def parse_dtype(value: str) -> torch.dtype | None:
    if value == "archive":
        return None
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype {value}")


def print_text(result: dict[str, Any]) -> None:
    print("ThinTensor GPU Runtime")
    print(f"Archive: {result['archive']}")
    print(f"Device: {result['device']}")
    print(f"DType: {result['dtype']}")
    print(f"Layers: {result['layers'] if result['layers'] is not None else 'all'}")
    print(f"Token ID: {result['token_id']}")
    print()
    print("Load:")
    print(f"  total: {result['load_s']:.4f} s")
    print(f"  archive open: {result['archive_open_s']:.4f} s")
    print(f"  gpu weights: {result['gpu_weight_load_s']:.4f} s")
    print(f"  disk read/page fault: {result['disk_read_s']:.4f} s")
    print(f"  cpu staging: {result['cpu_stage_s']:.4f} s")
    print(f"  gpu transfer: {result['gpu_transfer_s']:.4f} s")
    print(f"  pages loaded: {result['pages_loaded']}")
    print(f"  physical fused pages loaded: {result['physical_pages_loaded']}")
    print(f"  fused logical views: {result['fused_logical_pages']}")
    print(f"  aliased pages: {result['aliased_pages']}")
    print(f"  physical weights: {format_bytes(result['physical_weight_bytes'])}")
    print(f"  unique GPU weights: {format_bytes(result['unique_gpu_weight_bytes'])}")
    print(f"  CPU staging bytes: {format_bytes(result['cpu_staging_bytes'])}")
    print(f"  GPU transfer bytes: {format_bytes(result['gpu_transfer_bytes'])}")
    print(f"  page faults: minor={result['minor_page_faults']} major={result['major_page_faults']}")
    print()
    print("Forward:")
    print(f"  first token: {result['first_forward_s']:.4f} s")
    print(f"  loop: {result['forward_loop_s']:.4f} s for {result['steps']} steps")
    print(f"  forwards/s: {result['forward_per_s']:.2f}")
    print()
    print("Memory:")
    print(f"  RSS delta: {format_optional_bytes(result['rss_delta_bytes'])}")
    print(f"  GPU peak allocated: {format_optional_bytes(result['gpu_peak_allocated_bytes'])}")
    print(f"  GPU peak reserved: {format_optional_bytes(result['gpu_peak_reserved_bytes'])}")
    if result["gpu_peak_temp_c"] is not None:
        print(f"  GPU peak temp: {result['gpu_peak_temp_c']} C")
    print(f"  thermal stop: {result['thermal_stop']}")
    print()
    print("Top logits:")
    for item in result["top_logits"]:
        print(f"  token {item['token_id']}: {item['logit']:.5f}")


def format_optional_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return format_bytes(value)


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


if __name__ == "__main__":
    main()
