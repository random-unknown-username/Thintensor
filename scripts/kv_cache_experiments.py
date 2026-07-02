#!/usr/bin/env python3
"""Plan and execute exact KV layout/page-size experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.archive import ThinArchive
from thinruntime.gpu_runtime import PagedKVCache
from thinruntime.model_arch import descriptor_from_manifest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "thin_runtime.py"
QUALITY_FLAGS = [
    "--gate-up-fp8",
    "--down-proj-fp8",
    "--down-fp8-layers",
    "8:28",
]


def csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "SmolLM3-3B.thin")
    parser.add_argument("--contexts", default="128,512,1024,2048,4096")
    parser.add_argument("--block-sizes", default="1,8,16,32")
    parser.add_argument(
        "--layouts",
        default=",".join(sorted(PagedKVCache.LAYOUTS)),
    )
    parser.add_argument(
        "--residencies",
        default="gpu_full",
        help="Comma-separated subset of gpu_full,cpu_exact,hybrid_recent",
    )
    parser.add_argument("--gpu-recent-tokens", type=int, default=256)
    parser.add_argument("--prefetch-pages", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "runtime_optimizer" / "kv_experiments.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "runtime_optimizer" / "kv",
    )
    return parser.parse_args()


def exact_projection(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    context: int,
    block_size: int,
    dtype_bytes: int = 2,
) -> dict[str, int | float]:
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    blocks_per_layer = (context + block_size - 1) // block_size
    reserved_tokens = blocks_per_layer * block_size
    used = context * bytes_per_token
    allocated = reserved_tokens * bytes_per_token
    wasted = allocated - used
    return {
        "context": context,
        "block_size": block_size,
        "blocks_per_layer": blocks_per_layer,
        "actual_sequence_length": context,
        "max_sequence_length_reserved": reserved_tokens,
        "bytes_per_token": bytes_per_token,
        "bytes_per_layer_per_token": bytes_per_token // layers,
        "bytes_per_head_per_token": 2 * head_dim * dtype_bytes,
        "used_bytes": used,
        "allocated_bytes": allocated,
        "wasted_padded_bytes": wasted,
        "fragmentation_ratio": wasted / allocated if allocated else 0.0,
    }


def validate_layouts() -> dict[str, Any]:
    reference: tuple[torch.Tensor, torch.Tensor] | None = None
    rows = []
    for layout in sorted(PagedKVCache.LAYOUTS):
        cache = PagedKVCache(
            layers=2,
            kv_heads=2,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            block_size=3,
            layout=layout,
        )
        for token in range(5):
            for layer in range(2):
                key = (
                    torch.arange(8, dtype=torch.bfloat16)
                    + token * 10
                    + layer * 100
                )
                cache.append(layer, key, key + 1000, token)
        observed = cache.history(0, 4)
        if reference is None:
            reference = observed
        exact = torch.equal(reference[0], observed[0]) and torch.equal(
            reference[1], observed[1]
        )
        rows.append(
            {
                "layout": layout,
                "exact_storage_parity": exact,
                "telemetry": cache.telemetry(),
            }
        )
    return {"all_exact": all(row["exact_storage_parity"] for row in rows), "rows": rows}


def validate_exact_residency() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "CUDA unavailable", "rows": []}
    rows = []
    for layout in sorted(PagedKVCache.LAYOUTS):
        reference: tuple[torch.Tensor, torch.Tensor] | None = None
        for residency in ("gpu_full", "hybrid_recent", "cpu_exact"):
            cache = PagedKVCache(
                layers=2,
                kv_heads=2,
                head_dim=4,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                block_size=2,
                layout=layout,
                residency=residency,
                gpu_recent_tokens=2,
                prefetch_pages=2,
            )
            observed = None
            for token in range(5):
                for layer in range(2):
                    key = (
                        torch.arange(
                            8,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        + token * 10
                        + layer * 100
                    )
                    cache.append(layer, key, key + 1000, token)
                observed = cache.history(0, token)
            assert observed is not None
            observed_cpu = (observed[0].cpu(), observed[1].cpu())
            if reference is None:
                reference = observed_cpu
            exact = torch.equal(reference[0], observed_cpu[0]) and torch.equal(
                reference[1], observed_cpu[1]
            )
            rows.append(
                {
                    "layout": layout,
                    "residency": residency,
                    "exact_storage_parity": exact,
                    "telemetry": cache.telemetry(),
                }
            )
    return {
        "skipped": False,
        "all_exact": all(row["exact_storage_parity"] for row in rows),
        "rows": rows,
    }


def benchmark_command(
    args: argparse.Namespace,
    context: int,
    block_size: int,
    layout: str,
    residency: str,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "run",
        str(args.archive),
        "--device",
        "cuda",
        "--dtype",
        "bf16",
        "--steps",
        str(context),
        "--warmup-steps",
        str(args.warmup_steps),
        "--residency",
        "all",
        "--kernel-backend",
        "triton",
        "--attention-mode",
        "causal_kv",
        "--lm-head-backend",
        "triton",
        "--kv-block-size",
        str(block_size),
        "--kv-data-layout",
        layout,
        "--kv-residency",
        residency,
        "--kv-gpu-recent-tokens",
        str(args.gpu_recent_tokens),
        "--kv-prefetch-pages",
        str(args.prefetch_pages),
        *QUALITY_FLAGS,
        "--json",
    ]


def execute_case(
    args: argparse.Namespace,
    context: int,
    block_size: int,
    layout: str,
    residency: str,
) -> dict[str, Any]:
    stem = f"ctx{context}_block{block_size}_{layout}_{residency}"
    output = args.raw_dir / f"{stem}.json"
    stderr_path = args.raw_dir / f"{stem}.stderr"
    command = benchmark_command(
        args,
        context,
        block_size,
        layout,
        residency,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "command": command,
            "stderr": str(stderr_path),
        }
    result = json.loads(completed.stdout)
    result["command"] = command
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    contexts = csv_ints(args.contexts)
    block_sizes = csv_ints(args.block_sizes)
    layouts = csv_strings(args.layouts)
    residencies = csv_strings(args.residencies)
    unknown = set(layouts) - PagedKVCache.LAYOUTS
    if unknown:
        raise ValueError(f"unsupported layouts: {sorted(unknown)}")
    unknown_residencies = set(residencies) - {
        "gpu_full",
        "cpu_exact",
        "hybrid_recent",
    }
    if unknown_residencies:
        raise ValueError(
            f"unsupported residencies: {sorted(unknown_residencies)}"
        )
    with ThinArchive(args.archive) as archive:
        descriptor = descriptor_from_manifest(archive.manifest)
    projections = []
    for context in contexts:
        for block_size in block_sizes:
            projection = exact_projection(
                layers=descriptor.num_hidden_layers,
                kv_heads=descriptor.num_key_value_heads,
                head_dim=descriptor.head_dim,
                context=context,
                block_size=block_size,
            )
            for layout in layouts:
                for residency in residencies:
                    row = dict(projection)
                    row["layout"] = layout
                    row["residency"] = residency
                    reserved_tokens = int(
                        row["max_sequence_length_reserved"]
                    )
                    if residency == "gpu_full":
                        gpu_tokens = reserved_tokens
                    elif residency == "cpu_exact":
                        gpu_tokens = min(block_size, reserved_tokens)
                    else:
                        gpu_tokens = min(
                            reserved_tokens,
                            (
                                (
                                    args.gpu_recent_tokens
                                    + block_size
                                    - 1
                                )
                                // block_size
                            )
                            * block_size,
                        )
                    bytes_per_token = int(row["bytes_per_token"])
                    row["projected_gpu_bytes"] = gpu_tokens * bytes_per_token
                    row["projected_cpu_bytes"] = (
                        reserved_tokens - gpu_tokens
                    ) * bytes_per_token
                    projections.append(row)
    payload: dict[str, Any] = {
        "archive": str(args.archive),
        "model": descriptor.as_dict(),
        "formula": "2 * layers * kv_heads * head_dim * dtype_bytes * tokens",
        "layout_validation": validate_layouts(),
        "exact_residency_validation": validate_exact_residency(),
        "projections": projections,
        "executed": args.execute,
        "benchmarks": [],
    }
    if args.execute:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        for row in projections:
            payload["benchmarks"].append(
                execute_case(
                    args,
                    int(row["context"]),
                    int(row["block_size"]),
                    str(row["layout"]),
                    str(row["residency"]),
                )
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
