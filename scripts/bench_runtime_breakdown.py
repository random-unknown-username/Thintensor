import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from thinruntime.gpu_runtime import ThinGpuWeights, ThinGpuQwenRuntime, PagedKVCache


DEVICE = "cuda"
DTYPE = torch.bfloat16


def parse_args():
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
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lm-head-fp8", action="store_true")
    parser.add_argument("--keep-bf16-lm-head", action="store_true")
    parser.add_argument("--fused-mlp", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def bench(name, fn, steps, warmup):
    for i in range(warmup):
        fn(i)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(steps):
        fn(warmup + i)
    end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)
    return {
        "name": name,
        "steps": steps,
        "seconds": total_ms / 1000.0,
        "tokens_per_s": steps * 1000.0 / total_ms,
        "ms_per_token": total_ms / steps,
    }


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
def main():
    args = parse_args()
    archive_path = args.archive if args.archive is not None else args.archive_positional
    if archive_path is None:
        archive_path = Path("/tmp/thintensor-first-run/qwen3-0.6b.thin")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()

    weights = ThinGpuWeights(
        archive_path,
        device=DEVICE,
        dtype=DTYPE,
    )

    runtime_no_kv = ThinGpuQwenRuntime(
        weights,
        kv_cache=None,
        kernel_backend="triton-matvec",
        persistent_buffers=True,
        lm_head_fp8=args.lm_head_fp8,
        keep_bf16_lm_head=args.keep_bf16_lm_head,
        fused_mlp=args.fused_mlp,
    )

    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start

    token0 = torch.zeros((), device=DEVICE, dtype=torch.long)

    # Compile/warm hidden once.
    hidden0 = runtime_no_kv.forward_token(token0, token_index=0)
    _ = runtime_no_kv.next_token_tensor(hidden0)
    torch.cuda.synchronize()

    results = []

    # 1. Raw forward only, no KV, no lm_head.
    results.append(
        bench(
            "raw_forward_no_kv_no_lm_head",
            lambda i: runtime_no_kv.forward_token(token0, token_index=i),
            args.steps,
            args.warmup,
        )
    )

    # 2. lm_head + argmax only on same hidden.
    results.append(
        bench(
            "lm_head_argmax_only",
            lambda i: runtime_no_kv.next_token_tensor(hidden0),
            args.steps,
            args.warmup,
        )
    )

    # 3. Greedy decode no KV: forward + lm_head/argmax, token stays GPU scalar.
    token_holder = {"token": token0}

    def greedy_no_kv(i):
        hidden = runtime_no_kv.forward_token(token_holder["token"], token_index=i)
        token_holder["token"] = runtime_no_kv.next_token_tensor(hidden)

    results.append(
        bench("greedy_no_kv", greedy_no_kv, args.steps, args.warmup)
    )

    # 4. Greedy with BF16 KV append, exact BF16, no q4.
    model_manifest = weights.manifest["model"]
    kv = PagedKVCache(
        layers=int(model_manifest["layers"]),
        kv_heads=int(model_manifest["kv_heads"]),
        head_dim=int(model_manifest.get("head_dim") or (int(model_manifest["hidden_size"]) // int(model_manifest["heads"]))),
        device=torch.device(DEVICE),
        dtype=DTYPE,
        block_size=16,
        recent_window=256,
        old_codec="bf16",
        policy="sink_recent_attention",
    )

    runtime_kv = ThinGpuQwenRuntime(
        weights,
        kv_cache=kv,
        kernel_backend="triton-matvec",
        persistent_buffers=True,
        lm_head_fp8=args.lm_head_fp8,
        keep_bf16_lm_head=args.keep_bf16_lm_head,
        fused_mlp=args.fused_mlp,
    )

    token_holder = {"token": token0}

    def greedy_with_kv(i):
        hidden = runtime_kv.forward_token(token_holder["token"], token_index=i)
        token_holder["token"] = runtime_kv.next_token_tensor(hidden)

    results.append(
        bench(
            "greedy_with_bf16_kv_append",
            greedy_with_kv,
            args.steps,
            args.warmup,
        )
    )

    torch.cuda.synchronize()
    top_logits = runtime_no_kv.topk(hidden0, args.top_k) if args.top_k > 0 else []
    by_name = {result["name"]: result for result in results}

    # Dedicated profiling pass to collect component metrics (e.g. MLP / attention breakdown)
    runtime_no_kv._profiler_enabled = True
    runtime_no_kv._profile_steps = []
    # Warmup
    for i in range(5):
        hidden = runtime_no_kv.forward_token(token0, token_index=i)
        _ = runtime_no_kv.next_token_tensor(hidden)
    runtime_no_kv._profile_steps = []
    # Profile execution
    for i in range(20):
        hidden = runtime_no_kv.forward_token(token0, token_index=i)
        _ = runtime_no_kv.next_token_tensor(hidden)

    breakdown = runtime_no_kv.get_profiler_results()
    runtime_no_kv._profiler_enabled = False

    per_shape_bandwidths = calculate_per_shape_bandwidths(runtime_no_kv, breakdown)

    greedy_kv_result = by_name.get("greedy_with_bf16_kv_append")
    if greedy_kv_result:
        ms_per_token = greedy_kv_result["ms_per_token"]
        weight_read_bytes = runtime_no_kv.estimated_weight_read_bytes_per_token
        effective_bandwidth_gb_s = (weight_read_bytes / 1e9) / max(1e-12, (ms_per_token / 1000.0))
    else:
        effective_bandwidth_gb_s = 0.0

    out = {
        "archive": str(archive_path),
        "device": DEVICE,
        "dtype": "bf16",
        "load_s": load_s,
        **by_name,
        "results": results,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "kv_cache": kv.telemetry(),
        "kv_old_codec": kv.old_codec,
        "kv_compressed_blocks": kv.telemetry()["kv_compressed_blocks"],
        "runtime_fusion_enabled": runtime_no_kv.runtime_fusion_enabled,
        "runtime_fusion_extra_bytes": runtime_no_kv.runtime_fusion_extra_bytes,
        "lm_head_fp8_enabled": runtime_no_kv.lm_head_fp8_enabled,
        "lm_head_fp8_extra_bytes": runtime_no_kv.lm_head_fp8_extra_bytes,
        "lm_head_fp8_bytes": runtime_no_kv.lm_head_fp8_bytes,
        "lm_head_bf16_resident": runtime_no_kv.lm_head_bf16_resident,
        "lm_head_bf16_bytes": runtime_no_kv.lm_head_bf16_bytes,
        "lm_head_memory_saved_bytes": runtime_no_kv.lm_head_memory_saved_bytes,
        "lm_head_net_extra_bytes": runtime_no_kv.lm_head_net_extra_bytes,
        "autotune_enabled": runtime_no_kv.autotune_enabled,
        "per_shape_backend_choices": runtime_no_kv.per_shape_backend_choices,
        "per_shape_backend_benchmarks": getattr(runtime_no_kv, "per_shape_backend_benchmarks", {}),
        "per_shape_bandwidths": per_shape_bandwidths,
        "launches_per_token": getattr(runtime_no_kv, "launches_per_token", 0),
        "matvec_launches_per_token": getattr(runtime_no_kv, "matvec_launches_per_token", 0),
        "elementwise_launches_per_token": getattr(runtime_no_kv, "elementwise_launches_per_token", 0),
        "attention_launches_per_token": getattr(runtime_no_kv, "attention_launches_per_token", 0),
        "estimated_weight_read_bytes_per_token": weight_read_bytes,
        "weight_bytes": weight_read_bytes,
        "effective_bandwidth_gb_s": effective_bandwidth_gb_s,
        **runtime_no_kv.geometry_telemetry,
        "top_logits": top_logits,
        "top_logits_timing": "after_all_timed_sections",
        
        # Component timing breakdown:
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
        "fused_mlp_enabled": runtime_no_kv.fused_mlp_enabled,
        "fused_mlp_supported_layers": runtime_no_kv.fused_mlp_supported_layers,
        "fused_mlp_extra_bytes": runtime_no_kv.fused_mlp_extra_bytes,
        "fused_mlp_fallbacks": runtime_no_kv.fused_mlp_fallbacks,

        "note": "No .item(), no topk, no CPU token sync inside timed loops.",
    }

    print(json.dumps(out, indent=2))

    weights.close()


if __name__ == "__main__":
    main()
