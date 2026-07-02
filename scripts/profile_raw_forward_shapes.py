#!/usr/bin/env python3
"""Profile ThinTensor decode matvec shapes and layer scaling with CUDA events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from thinruntime.gpu_runtime import ThinGpuQwenRuntime, ThinGpuWeights, _layer_tensor


@torch.inference_mode()
def cuda_bench(fn, trials: int, warmup: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        fn()
    end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)
    return {
        "trials": trials,
        "total_ms": total_ms,
        "mean_ms": total_ms / trials,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "archive_positional",
        type=Path,
        nargs="?",
        default=None,
    )
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--fused-mlp", action="store_true")
    return parser.parse_args()


def calculate_per_shape_bandwidths(runtime, breakdown):
    per_shape_bandwidth = {}
    if not hasattr(runtime, "per_shape_backend_choices"):
        return per_shape_bandwidth
    for key in runtime.per_shape_backend_choices.keys():
        try:
            parts = key.split(":")
            shape_part = parts[0]
            rows, cols = map(int, shape_part.split("x"))
            element_size = 1 if ("float8" in key or "fp8" in key) else 2
            weight_bytes = rows * cols * element_size
            
            # Map time
            time_ms = 0.0
            if rows == runtime.intermediate_size and cols == runtime.hidden_size:
                time_ms = breakdown.get("gate_proj_time_ms", 0.0) + breakdown.get("up_proj_time_ms", 0.0)
            elif rows == runtime.hidden_size and cols == runtime.intermediate_size:
                time_ms = breakdown.get("down_proj_time_ms", 0.0)
            elif rows == runtime.hidden_size and cols == runtime.hidden_size:
                time_ms = breakdown.get("o_proj_time_ms", 0.0)
            elif rows == runtime.kv_dim and cols == runtime.hidden_size:
                time_ms = breakdown.get("qkv_time_ms", 0.0)
            elif cols == runtime.hidden_size:
                time_ms = breakdown.get("lm_head_argmax_time_ms", 0.0)
            
            num_executions = runtime.layers
            if "lm_head" in key or rows > 50000:
                num_executions = 1
            elif rows == runtime.intermediate_size and cols == runtime.hidden_size:
                num_executions = runtime.layers * (1 if runtime.fused_mlp_enabled else 2)
            
            time_per_exec_ms = time_ms / max(1, num_executions)
            effective_bandwidth = (weight_bytes / 1e9) / max(1e-12, (time_per_exec_ms / 1000.0))
            
            per_shape_bandwidth[key] = {
                "weight_bytes": weight_bytes,
                "time_ms": time_per_exec_ms,
                "effective_bandwidth_gb_s": effective_bandwidth
            }
        except Exception:
            continue
    return per_shape_bandwidth


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    archive_path = args.archive if args.archive is not None else args.archive_positional
    if archive_path is None:
        archive_path = Path("/tmp/thintensor-first-run/qwen3-0.6b.thin")

    device = "cuda"
    dtype = torch.bfloat16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    weights = ThinGpuWeights(archive_path, device=device, dtype=dtype)
    runtime = ThinGpuQwenRuntime(
        weights,
        kv_cache=None,
        kernel_backend="triton-matvec",
        persistent_buffers=True,
        fused_mlp=args.fused_mlp,
    )
    backend = runtime.kernel_backend
    assert backend is not None

    results = []

    def test_weight(label: str, weight_id: str) -> None:
        weight = weights.tensor(weight_id)
        rows, cols = map(int, weight.shape)
        x = torch.randn(cols, device=device, dtype=dtype)
        out = torch.empty(rows, device=device, dtype=dtype)
        block_results = {}
        for block_m in (8, 16, 32, 64):
            block_results[str(block_m)] = cuda_bench(
                lambda bm=block_m: backend.matvec(
                    weight, x, out, block_m=bm
                ),
                args.trials,
                args.warmup,
            )
        results.append(
            {
                "label": label,
                "weight_id": weight_id,
                "shape": [rows, cols],
                "stride": list(weight.stride()),
                "dtype": str(weight.dtype).replace("torch.", ""),
                "triton_static": cuda_bench(
                    lambda: backend.matvec(weight, x, out),
                    args.trials,
                    args.warmup,
                ),
                "torch_mv": cuda_bench(
                    lambda: torch.mv(weight, x, out=out),
                    args.trials,
                    args.warmup,
                ),
                "triton_block_m": block_results,
                "block_n": 1 << (cols - 1).bit_length(),
                "runtime_choice": runtime.per_shape_backend_choices.get(
                    f"{rows}x{cols}:{weight.dtype}:stride={weight.stride(0)},{weight.stride(1)}"
                ),
            }
        )

    layer = 0
    for label, suffix in [
        ("q_proj", "self_attn.q_proj.weight"),
        ("k_proj", "self_attn.k_proj.weight"),
        ("v_proj", "self_attn.v_proj.weight"),
        ("o_proj", "self_attn.o_proj.weight"),
        ("gate_proj", "mlp.gate_proj.weight"),
        ("up_proj", "mlp.up_proj.weight"),
        ("down_proj", "mlp.down_proj.weight"),
    ]:
        test_weight(label, _layer_tensor(layer, suffix))
    lm_head_id = (
        "lm_head.weight"
        if "lm_head.weight" in weights.tensors
        else "model.embed_tokens.weight"
    )
    test_weight("lm_head", lm_head_id)

    token = torch.zeros((), device=device, dtype=torch.long)
    layer_tests = []
    seen = set()
    for layer_count in [1, 4, 8, 16, runtime.layers]:
        layer_count = min(layer_count, runtime.layers)
        if layer_count in seen:
            continue
        seen.add(layer_count)
        layer_tests.append(
            {
                "layers": layer_count,
                **cuda_bench(
                    lambda count=layer_count: runtime.forward_token(
                        token, layers=count, token_index=0
                    ),
                    args.trials,
                    args.warmup,
                ),
            }
        )

    # Profiling pass to collect component timing breakdown
    runtime._profiler_enabled = True
    runtime._profile_steps = []
    # Warmup
    for i in range(5):
        hidden = runtime.forward_token(token, token_index=i)
        _ = runtime.next_token_tensor(hidden)
    runtime._profile_steps = []
    # Profile execution
    for i in range(20):
        hidden = runtime.forward_token(token, token_index=i)
        _ = runtime.next_token_tensor(hidden)

    breakdown = runtime.get_profiler_results()
    runtime._profiler_enabled = False

    per_shape_bandwidths = calculate_per_shape_bandwidths(runtime, breakdown)

    # Use average total forward token time to calculate overall effective bandwidth
    total_fwd_time_ms = breakdown.get("total_forward_token_time_ms")
    weight_read_bytes = runtime.estimated_weight_read_bytes_per_token
    if total_fwd_time_ms:
        effective_bandwidth_gb_s = (weight_read_bytes / 1e9) / max(1e-12, (total_fwd_time_ms / 1000.0))
    else:
        effective_bandwidth_gb_s = 0.0

    output = {
        "archive": str(archive_path),
        "hidden_size": runtime.hidden_size,
        "intermediate_size": runtime.intermediate_size,
        "heads": runtime.heads,
        "kv_heads": runtime.kv_heads,
        "head_dim": runtime.head_dim,
        "kernel_backend": runtime.kernel_backend_name,
        "autotune_enabled": runtime.autotune_enabled,
        "per_shape_backend_choices": runtime.per_shape_backend_choices,
        "per_shape_backend_benchmarks": getattr(runtime, "per_shape_backend_benchmarks", {}),
        "per_shape_bandwidths": per_shape_bandwidths,
        "launches_per_token": getattr(runtime, "launches_per_token", 0),
        "matvec_launches_per_token": getattr(runtime, "matvec_launches_per_token", 0),
        "elementwise_launches_per_token": getattr(runtime, "elementwise_launches_per_token", 0),
        "attention_launches_per_token": getattr(runtime, "attention_launches_per_token", 0),
        "estimated_weight_read_bytes_per_token": weight_read_bytes,
        "weight_bytes": weight_read_bytes,
        "effective_bandwidth_gb_s": effective_bandwidth_gb_s,
        "projection_benchmarks": results,
        "forward_layer_scaling": layer_tests,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),

        # Breakdown metrics requested by Task 1:
        "total_forward_token_time_ms": breakdown.get("total_forward_token_time_ms"),
        "per_layer_average_time_ms": breakdown.get("per_layer_average_time_ms"),
        "attention_time_ms": breakdown.get("attention_time_ms"),
        "qkv_time_ms": breakdown.get("qkv_time_ms"),
        "o_proj_time_ms": breakdown.get("o_proj_time_ms"),
        "mlp_total_time_ms": breakdown.get("mlp_total_time_ms"),
        "gate_proj_time_ms": breakdown.get("gate_proj_time_ms"),
        "up_proj_time_ms": breakdown.get("up_proj_time_ms"),
        "silu_mul_time_ms": breakdown.get("silu_mul_time_ms"),
        "down_proj_time_ms": breakdown.get("down_proj_time_ms"),
        "lm_head_argmax_time_ms": breakdown.get("lm_head_argmax_time_ms"),
        "fused_mlp_enabled": runtime.fused_mlp_enabled,
        "fused_mlp_supported_layers": runtime.fused_mlp_supported_layers,
        "fused_mlp_extra_bytes": runtime.fused_mlp_extra_bytes,
        "fused_mlp_fallbacks": runtime.fused_mlp_fallbacks,
    }
    print(json.dumps(output, indent=2))
    weights.close()


if __name__ == "__main__":
    main()
