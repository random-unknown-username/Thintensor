#!/usr/bin/env python3
"""ThinTensor GPU runtime product commands.

This is the v0 runtime backend: it owns page loading, the resident VRAM pool,
execution-tape prefetch/eviction, paged KV telemetry, and runtime-profile
compilation. Math kernels still use PyTorch ops while the memory system is
ThinTensor controlled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.gpu_runtime import (
    PagedKVCache,
    TempGuard,
    ThinGpuCausalLMRuntime,
    ThinGpuPagePool,
    ThinGpuWeights,
    _gpu_peak_allocated,
    _gpu_peak_reserved,
    _rss_bytes,
)


def main() -> None:
    args = parse_args()
    if args.command == "gpu-load":
        result = cmd_gpu_load(args)
    elif args.command == "run":
        result = cmd_run(args)
    elif args.command == "compile-runtime":
        result = cmd_compile_runtime(args)
    elif args.command == "optimize-runtime":
        result = cmd_optimize_runtime(args)
    else:
        raise SystemExit(f"unknown command {args.command}")

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_text(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    gpu_load = sub.add_parser("gpu-load")
    gpu_load.add_argument("archive", type=Path)
    gpu_load.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    gpu_load.add_argument("--dtype", choices=["archive", "bf16", "fp16", "fp32"], default="bf16")
    gpu_load.add_argument("--prefetch", type=int, default=4)
    gpu_load.add_argument("--max-gpu-temp", type=int, default=87)
    gpu_load.add_argument("--json", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("archive", type=Path)
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run.add_argument("--dtype", choices=["archive", "bf16", "fp16", "fp32"], default="bf16")
    run.add_argument("--residency", choices=["all", "stream"], default="stream")
    run.add_argument("--vram", default="0")
    run.add_argument("--prefetch", type=int, default=1)
    run.add_argument("--steps", type=int, default=8)
    run.add_argument("--warmup-steps", type=int, default=0)
    run.add_argument("--token-id", type=int, default=0)
    run.add_argument("--layers", type=int)
    run.add_argument("--kv", default="bf16")
    run.add_argument("--kv-budget", default="0")
    run.add_argument("--kv-policy", default="sink_recent_attention")
    run.add_argument("--kv-layout", choices=["paged", "cuda-vmm"], default="paged")
    run.add_argument(
        "--kv-data-layout",
        choices=sorted(PagedKVCache.LAYOUTS),
        default="head_token_interleaved",
        help="Physical K/V organization inside each exact KV page",
    )
    run.add_argument("--recent-window", type=int, default=256)
    run.add_argument("--kv-block-size", type=int, default=16)
    run.add_argument(
        "--kv-residency",
        choices=["gpu_full", "cpu_exact", "hybrid_recent"],
        default="gpu_full",
    )
    run.add_argument("--kv-gpu-recent-tokens", type=int, default=256)
    run.add_argument("--kv-prefetch-pages", type=int, default=0)
    run.add_argument("--arena", action="store_true")
    run.add_argument("--cuda-graphs", action="store_true")
    run.add_argument("--gds", action="store_true")
    run.add_argument("--draft", type=Path)
    run.add_argument("--speculative", action="store_true")
    run.add_argument("--mode", choices=["decode", "embeddings"], default="decode")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--profile-runtime", type=Path)
    run.add_argument("--runtime-profile", type=Path)
    run.add_argument(
        "--kernel-backend",
        choices=["torch", "triton", "triton-matvec", "hybrid", "auto"],
        default="triton-matvec",
    )
    run.add_argument("--no-persistent-buffers", action="store_true")
    run.add_argument("--lm-head-fp8", action="store_true")
    run.add_argument("--keep-bf16-lm-head", action="store_true")
    run.add_argument("--exact-topk", action="store_true")
    run.add_argument("--max-gpu-temp", type=int, default=87)
    run.add_argument("--fused-mlp", action="store_true")
    run.add_argument(
        "--fused-scaled-mlp",
        action="store_true",
        help=(
            "Fuse scaled FP8 gate/up matvecs with SiLU multiply; "
            "experimental until real decode and correctness pass"
        ),
    )
    run.add_argument(
        "--fused-residual-norm",
        action="store_true",
        help="Fuse each residual add with its immediately following RMSNorm",
    )
    run.add_argument(
        "--fused-rope",
        action="store_true",
        help="Apply Q and K rotary embedding in one in-place Triton kernel",
    )
    run.add_argument(
        "--tuned-large-matvec",
        action="store_true",
        help="Use real-shape-tuned row tiles/warps for large decode matvecs",
    )
    run.add_argument(
        "--split-k-down-proj",
        action="store_true",
        help=(
            "Parallelize dense down-projection reduction across two exact "
            "column ranges; experimental until full decode gates pass"
        ),
    )
    run.add_argument("--cpu-offload", action="store_true")
    run.add_argument("--gpu-weight-budget", default="0")
    run.add_argument("--prefetch-layers", type=int, default=0)
    run.add_argument("--pin-cpu-pages", action="store_true")
    run.add_argument("--debug-stream-refs", action="store_true")
    run.add_argument("--down-proj-fp8", action="store_true")
    run.add_argument("--mlp-fp8", action="store_true")
    run.add_argument("--gate-up-fp8", action="store_true")
    run.add_argument("--attn-proj-fp8", action="store_true")
    run.add_argument("--qkv-fp8", action="store_true")
    run.add_argument("--o-proj-fp8", action="store_true")
    run.add_argument("--validate-lm-head-fp8", action="store_true")
    run.add_argument(
        "--fp8-layers",
        "--mlp-fp8-layers",
        dest="fp8_layers",
        help="Layer selection such as all, 2:34, or 0,1,8:16",
    )
    run.add_argument(
        "--down-fp8-layers",
        help="Optional down-projection-only layer selection",
    )
    run.add_argument(
        "--qkv-fp8-layers",
        help="Optional Q/K/V projection-only layer selection",
    )
    run.add_argument(
        "--o-fp8-layers",
        help="Optional attention output projection-only layer selection",
    )
    run.add_argument(
        "--fp8-scale-block",
        type=int,
        default=0,
        help="0 for row scales, or a power-of-two column block size",
    )
    run.add_argument(
        "--lm-head-fp8-scale-block",
        type=int,
        default=0,
        help=(
            "Head8 scale block independent of body FP8; 0 uses one scale "
            "per vocabulary row"
        ),
    )
    run.add_argument(
        "--attention-mode",
        choices=["causal_kv", "current_only", "current_only_smoke"],
        default="causal_kv",
    )
    run.add_argument(
        "--attention-backend",
        choices=["torch", "triton_fused"],
        default="torch",
        help=(
            "Single-token causal attention implementation; fused Triton "
            "remains opt-in until correctness and real decode both pass"
        ),
    )
    run.add_argument("--exact-hf-mode", action="store_true")
    run.add_argument(
        "--lm-head-backend",
        choices=[
            "torch_mv",
            "torch_matmul",
            "triton",
            "triton_loop_128",
            "triton_loop_256",
            "triton_loop_512",
            "row_block_m2",
            "row_block_m4",
            "row_block_m8",
        ],
    )
    run.add_argument(
        "--lm-head-argmax-mode",
        choices=["torch", "triton_two_stage", "triton_persistent"],
        default="torch",
        help=(
            "Greedy BF16 head reduction; triton_two_stage and triton_persistent are opt-in and "
            "must win real decode before use"
        ),
    )
    run.add_argument(
        "--lm-head-topk-guard",
        type=int,
        default=0,
        help=(
            "Use Head8 only to shortlist this many tokens, then select from "
            "BF16-recomputed candidate logits; full logits remain BF16"
        ),
    )
    run.add_argument(
        "--gate-up-backend",
        choices=[
            "torch_mv",
            "torch_matmul",
            "triton",
            "triton_loop_64",
            "triton_loop_128",
            "triton_loop_256",
            "triton_loop_512",
            "row_block_m2",
            "row_block_m4",
            "row_block_m8",
        ],
    )
    run.add_argument(
        "--down-proj-backend",
        choices=[
            "torch_mv",
            "torch_matmul",
            "triton",
            "triton_loop_64",
            "triton_loop_128",
            "triton_loop_256",
            "triton_loop_512",
            "row_block_m2",
            "row_block_m4",
            "row_block_m8",
        ],
    )
    run.add_argument(
        "--attn-proj-backend",
        choices=[
            "torch_mv",
            "torch_matmul",
            "triton",
            "triton_loop_64",
            "triton_loop_128",
            "triton_loop_256",
            "triton_loop_512",
            "row_block_m2",
            "row_block_m4",
            "row_block_m8",
        ],
    )
    run.add_argument("--json", action="store_true")

    compile_runtime = sub.add_parser("compile-runtime")
    compile_runtime.add_argument("archive", type=Path)
    compile_runtime.add_argument("--device", default="cuda")
    compile_runtime.add_argument("--vram", required=True)
    compile_runtime.add_argument("--ctx", type=int, default=8192)
    compile_runtime.add_argument("--batch", type=int, default=1)
    compile_runtime.add_argument("--kv", default="bf16")
    compile_runtime.add_argument("--recent-window", type=int, default=256)
    compile_runtime.add_argument("--out", type=Path)
    compile_runtime.add_argument("--json", action="store_true")

    optimize = sub.add_parser("optimize-runtime")
    optimize.add_argument("archive", type=Path)
    optimize.add_argument("--device", default="cuda")
    optimize.add_argument("--vram", required=True)
    optimize.add_argument("--ctx", type=int, default=8192)
    optimize.add_argument("--batch", type=int, default=1)
    optimize.add_argument("--out", type=Path)
    optimize.add_argument("--json", action="store_true")
    return parser.parse_args()


def cmd_gpu_load(args: argparse.Namespace) -> dict[str, Any]:
    dtype = parse_dtype(args.dtype)
    reset_gpu(args.device)
    rss0 = _rss_bytes()
    guard = TempGuard(args.device, args.max_gpu_temp)
    guard.start()
    try:
        start = time.perf_counter()
        weights = ThinGpuWeights(args.archive, device=args.device, dtype=dtype)
        sync(args.device)
        elapsed = time.perf_counter() - start
        rss1 = _rss_bytes()
        stats = weights.stats
        result = {
            "command": "gpu-load",
            "archive": str(args.archive),
            "device": args.device,
            "dtype": args.dtype,
            "prefetch": args.prefetch,
            "load_s": elapsed,
            "archive_open_s": stats.archive_open_s,
            "gpu_weight_load_s": stats.gpu_load_s,
            "disk_read_s": stats.disk_read_s,
            "cpu_stage_s": stats.cpu_stage_s,
            "gpu_transfer_s": stats.gpu_transfer_s,
            "cpu_staging_bytes": stats.cpu_staging_bytes,
            "gpu_transfer_bytes": stats.gpu_transfer_bytes,
            "physical_weight_bytes": stats.physical_weight_bytes,
            "unique_gpu_weight_bytes": stats.unique_gpu_weight_bytes,
            "pages_loaded": stats.pages_loaded,
            "physical_pages_loaded": stats.physical_pages_loaded,
            "fused_logical_pages": stats.fused_logical_pages,
            "aliased_pages": stats.aliased_pages,
            "minor_page_faults": stats.minor_page_faults,
            "major_page_faults": stats.major_page_faults,
            "peak_cpu_rss_delta_bytes": None if rss0 is None or rss1 is None else rss1 - rss0,
            "peak_vram_allocated_bytes": _gpu_peak_allocated(args.device),
            "peak_vram_reserved_bytes": _gpu_peak_reserved(args.device),
            "gpu_peak_temp_c": guard.peak_temp,
            "thermal_stop": guard.too_hot,
        }
        weights.close()
        return result
    finally:
        guard.stop()


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


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = parse_dtype(args.dtype)
    reset_gpu(args.device)
    rss0 = _rss_bytes()
    guard = TempGuard(args.device, args.max_gpu_temp)
    guard.start()
    unsupported = unsupported_runtime_flags(args)
    try:
        load_start = time.perf_counter()
        budget = parse_bytes(args.vram)
        if args.residency == "all":
            weights: ThinGpuWeights | ThinGpuPagePool = ThinGpuWeights(
                args.archive, device=args.device, dtype=dtype
            )
            pool_telemetry: dict[str, Any] = {}
        else:
            vram_budget = parse_bytes(args.gpu_weight_budget)
            if vram_budget == 0:
                vram_budget = parse_bytes(args.vram)
            prefetch_dist = args.prefetch_layers if args.prefetch_layers > 0 else args.prefetch
            
            weights = ThinGpuPagePool(
                args.archive,
                device=args.device,
                dtype=dtype,
                vram_budget_bytes=vram_budget if vram_budget > 0 else None,
                prefetch_distance=prefetch_dist,
                cpu_offload=args.cpu_offload,
                pin_cpu_pages=args.pin_cpu_pages,
                debug_stream_refs=args.debug_stream_refs,
                down_proj_fp8=args.down_proj_fp8 or args.mlp_fp8,
                gate_up_fp8=args.gate_up_fp8 or args.mlp_fp8,
                qkv_fp8=args.qkv_fp8 or args.attn_proj_fp8,
                o_proj_fp8=args.o_proj_fp8 or args.attn_proj_fp8,
                fp8_layer_spec=args.fp8_layers,
                down_fp8_layer_spec=args.down_fp8_layers,
                qkv_fp8_layer_spec=args.qkv_fp8_layers,
                o_fp8_layer_spec=args.o_fp8_layers,
                fp8_scale_block=args.fp8_scale_block,
                lm_head_fp8_scale_block=args.lm_head_fp8_scale_block,
            )
            weights.warm_start()
            pool_telemetry = weights.telemetry()

        model = weights.manifest["model"]
        kv_dtype = dtype or torch.bfloat16

        def build_runtime() -> tuple[PagedKVCache, ThinGpuCausalLMRuntime]:
            kv = PagedKVCache(
                layers=int(model["layers"]),
                kv_heads=int(model["kv_heads"]),
                head_dim=int(
                    model.get("head_dim")
                    or (int(model["hidden_size"]) // int(model["heads"]))
                ),
                device=torch.device(args.device),
                dtype=kv_dtype,
                block_size=args.kv_block_size,
                recent_window=args.recent_window,
                old_codec=args.kv,
                budget_bytes=parse_bytes(args.kv_budget) or None,
                policy=args.kv_policy,
                offload_old_to_cpu=args.kv_policy.endswith("offload"),
                layout=args.kv_data_layout,
                residency=args.kv_residency,
                gpu_recent_tokens=args.kv_gpu_recent_tokens,
                prefetch_pages=args.kv_prefetch_pages,
            )
            rt = ThinGpuCausalLMRuntime(
                weights,
                kv_cache=kv,
                prefetch_distance=args.prefetch if args.residency == "stream" else 0,
                evict_completed_layers=args.residency == "stream",
                kernel_backend="torch" if args.exact_hf_mode else args.kernel_backend,
                persistent_buffers=not args.no_persistent_buffers,
                lm_head_fp8=args.lm_head_fp8,
                keep_bf16_lm_head=(
                    args.keep_bf16_lm_head
                    or args.exact_topk
                    or args.validate_lm_head_fp8
                ),
                fused_mlp=args.fused_mlp,
                fused_scaled_mlp=args.fused_scaled_mlp,
                fused_residual_norm=args.fused_residual_norm,
                fused_rope=args.fused_rope,
                tuned_large_matvec=args.tuned_large_matvec,
                split_k_down_proj=args.split_k_down_proj,
                cuda_graphs=args.cuda_graphs,
                down_proj_fp8=args.down_proj_fp8,
                mlp_fp8=args.mlp_fp8,
                gate_up_fp8=args.gate_up_fp8,
                qkv_fp8=args.qkv_fp8,
                o_proj_fp8=args.o_proj_fp8,
                attn_proj_fp8=args.attn_proj_fp8,
                fp8_layer_spec=args.fp8_layers,
                down_fp8_layer_spec=args.down_fp8_layers,
                qkv_fp8_layer_spec=args.qkv_fp8_layers,
                o_fp8_layer_spec=args.o_fp8_layers,
                attention_mode=(
                    "current_only_smoke"
                    if args.attention_mode == "current_only"
                    else args.attention_mode
                ),
                attention_backend=args.attention_backend,
                exact_hf_mode=args.exact_hf_mode,
                fp8_scale_block=args.fp8_scale_block,
                lm_head_backend=args.lm_head_backend,
                lm_head_argmax_mode=args.lm_head_argmax_mode,
                lm_head_topk_guard=args.lm_head_topk_guard,
                gate_up_backend=args.gate_up_backend,
                down_proj_backend=args.down_proj_backend,
                attn_proj_backend=args.attn_proj_backend,
            )
            return kv, rt

        kv_cache, runtime = build_runtime()
        warmup_token = torch.full(
            (), args.token_id, device=args.device, dtype=torch.long
        )
        for warmup_step in range(max(0, args.warmup_steps)):
            if args.mode == "embeddings":
                _ = weights.tensor("model.embed_tokens.weight")[args.token_id].clone()
            else:
                hidden = runtime.forward_token(warmup_token, layers=args.layers, token_index=warmup_step)
                warmup_token = runtime.next_token_tensor(hidden)
        if args.warmup_steps > 0:
            sync(args.device)
            kv_cache.reset(reuse_pages=True)
        sync(args.device)
        load_s = time.perf_counter() - load_start

        guard.raise_if_hot("before run")
        token_id = torch.full((), args.token_id, device=args.device, dtype=torch.long)
        top_logits: list[dict[str, float | int]] = []
        lm_head_validation: dict[str, Any] = {}

        if getattr(args, "cuda_graphs", False) and args.residency == "all" and args.device == "cuda":
            runtime.capture_cuda_graph(args.steps)

        if args.device == "cuda":
            loop_start_event = torch.cuda.Event(enable_timing=True)
            first_end_event = torch.cuda.Event(enable_timing=True)
            loop_end_event = torch.cuda.Event(enable_timing=True)
            loop_start_event.record()
        else:
            loop_start = time.perf_counter()
        for step in range(max(1, args.steps)):
            if args.mode == "embeddings":
                hidden = weights.tensor("model.embed_tokens.weight").index_select(
                    0, token_id.reshape(1)
                ).reshape(-1)
                token_id = runtime.next_token_tensor(hidden)
            else:
                if getattr(runtime, "_cuda_graphs_enabled", False):
                    token_id = runtime.replay_cuda_graph(token_id)
                else:
                    hidden = runtime.forward_token(token_id, layers=args.layers, token_index=step)
                    token_id = runtime.next_token_tensor(hidden)
            if step == 0 and args.device == "cuda":
                first_end_event.record()
        if args.device == "cuda":
            loop_end_event.record()
            loop_end_event.synchronize()
            first_s = loop_start_event.elapsed_time(first_end_event) / 1000.0
            loop_s = loop_start_event.elapsed_time(loop_end_event) / 1000.0
        else:
            loop_s = time.perf_counter() - loop_start
            first_s = None

        if getattr(runtime, "_cuda_graphs_enabled", False):
            hidden = runtime._static_hidden
        guard.raise_if_hot("after timed run")
        timed_geometry = dict(runtime.geometry_telemetry)
        timed_kv_telemetry = dict(kv_cache.telemetry())

        # Compute top logits only after timing. topk() copies to CPU.
        if args.mode != "embeddings" and args.top_k > 0:
            top_logits = runtime.topk(
                hidden, args.top_k, exact=args.exact_topk
            )
            if args.validate_lm_head_fp8:
                if not args.lm_head_fp8:
                    raise RuntimeError(
                        "--validate-lm-head-fp8 requires --lm-head-fp8"
                    )
                approximate = runtime.topk(hidden, max(5, args.top_k))
                exact = runtime.topk(
                    hidden, max(5, args.top_k), exact=True
                )
                approx_ids = {row["token_id"] for row in approximate[:5]}
                exact_ids = {row["token_id"] for row in exact[:5]}
                lm_head_validation = {
                    "lm_head_fp8_safe": (
                        approximate[0]["token_id"] == exact[0]["token_id"]
                        and len(approx_ids & exact_ids) / 5.0 >= 0.8
                    ),
                    "lm_head_fp8_top1_agreement": (
                        approximate[0]["token_id"] == exact[0]["token_id"]
                    ),
                    "lm_head_fp8_top5_overlap": len(
                        approx_ids & exact_ids
                    )
                    / 5.0,
                }

        if args.mode != "embeddings" and args.device == "cuda":
            runtime._profiler_enabled = True
            runtime._profile_steps = []
            for i in range(10):
                hidden = runtime.forward_token(
                    token_id,
                    layers=args.layers,
                    token_index=max(1, args.steps) + i,
                )
                _ = runtime.next_token_tensor(hidden)
            breakdown = runtime.get_profiler_results()
            runtime._profiler_enabled = False
        else:
            breakdown = {}
        lm_head_components = (
            runtime.profile_lm_head_components(hidden)
            if args.mode != "embeddings" and args.device == "cuda"
            else {}
        )

        per_shape_bandwidths = calculate_per_shape_bandwidths(runtime, breakdown)

        ms_per_token = loop_s * 1000.0 / max(1, args.steps)
        weight_read_bytes = runtime.estimated_weight_read_bytes_per_token
        effective_bandwidth_gb_s = (weight_read_bytes / 1e9) / max(1e-12, (ms_per_token / 1000.0))

        rss1 = _rss_bytes()
        if isinstance(weights, ThinGpuPagePool):
            pool_telemetry = weights.telemetry()
        else:
            stats = weights.stats
            pool_telemetry = {
                "resident_pages": len(weights.tensors),
                "resident_bytes": stats.unique_gpu_weight_bytes,
                "gpu_transfer_bytes": stats.gpu_transfer_bytes,
                "cpu_staging_bytes": stats.cpu_staging_bytes,
                "minor_page_faults": stats.minor_page_faults,
                "major_page_faults": stats.major_page_faults,
                "fused_logical_pages": stats.fused_logical_pages,
                "aliased_pages": stats.aliased_pages,
            }
        result = {
            "command": "run",
            "archive": str(args.archive),
            "device": args.device,
            "dtype": args.dtype,
            "residency": args.residency,
            "mode": args.mode,
            "kernel_backend": runtime.kernel_backend_name,
            "persistent_buffers": not args.no_persistent_buffers,
            "steps": max(1, args.steps),
            "warmup_steps": max(0, args.warmup_steps),
            "layers": args.layers,
            "load_s": load_s,
            "first_token_s": first_s,
            "decode_s": loop_s,
            "tokens_per_s": max(1, args.steps) / loop_s if loop_s > 0 else 0.0,
            "ms_per_token": ms_per_token,
            "steady_decode_s": max(0.0, loop_s - (first_s or 0.0)),
            "steady_tokens_per_s": (
                (max(1, args.steps) - 1) / max(1e-12, loop_s - (first_s or 0.0))
                if max(1, args.steps) > 1
                else None
            ),
            "bytes_moved_per_token": weight_read_bytes,
            "estimated_weight_read_bytes_per_token": weight_read_bytes,
            "weight_bytes": weight_read_bytes,
            "effective_bandwidth_gb_s": effective_bandwidth_gb_s,
            "resident_weight_bytes": runtime.resident_weight_bytes,
            "kv_cache_bytes": timed_kv_telemetry["kv_allocated_bytes"],
            "gpu_kv_cache_bytes": timed_kv_telemetry["kv_gpu_bytes"],
            "cpu_kv_cache_bytes": timed_kv_telemetry["kv_cpu_bytes"],
            "temp_buffer_bytes": runtime.temp_buffer_bytes,
            "load_transfer_bytes": int(pool_telemetry.get("gpu_transfer_bytes", 0)),
            "runtime_capability_gates": unsupported,
            "cpu_offload_enabled": pool_telemetry.get("cpu_offload_enabled", False),
            "gpu_weight_budget_bytes": pool_telemetry.get("gpu_weight_budget_bytes", 0),
            "gpu_cache": pool_telemetry.get("gpu_cache", {}),
            "cpu_store": pool_telemetry.get("cpu_store", {}),
            "page_pool": pool_telemetry,
            "kv_cache": timed_kv_telemetry,
            "kv_old_codec": kv_cache.old_codec,
            "kv_compressed_blocks": timed_kv_telemetry["kv_compressed_blocks"],
            "runtime_fusion_enabled": runtime.runtime_fusion_enabled,
            "runtime_fusion_extra_bytes": runtime.runtime_fusion_extra_bytes,
            "fused_mlp_enabled": runtime.fused_mlp_enabled,
            "fused_scaled_mlp_enabled": (
                runtime.fused_scaled_mlp_enabled_flag
            ),
            "fused_residual_norm_enabled": (
                runtime.fused_residual_norm_enabled
            ),
            "fused_rope_enabled": runtime.fused_rope_enabled,
            "tuned_large_matvec_enabled": (
                runtime.tuned_large_matvec_enabled
            ),
            "split_k_down_proj_enabled": (
                runtime.split_k_down_proj_enabled
            ),
            "fused_mlp_supported_layers": runtime.fused_mlp_supported_layers,
            "fused_mlp_extra_bytes": runtime.fused_mlp_extra_bytes,
            "fused_mlp_fallbacks": runtime.fused_mlp_fallbacks,
            "lm_head_fp8_enabled": runtime.lm_head_fp8_enabled,
            "lm_head_fp8_extra_bytes": runtime.lm_head_fp8_extra_bytes,
            "lm_head_fp8_bytes": runtime.lm_head_fp8_bytes,
            "lm_head_bf16_resident": runtime.lm_head_bf16_resident,
            "lm_head_bf16_bytes": runtime.lm_head_bf16_bytes,
            "lm_head_memory_saved_bytes": runtime.lm_head_memory_saved_bytes,
            "lm_head_net_extra_bytes": runtime.lm_head_net_extra_bytes,
            "lm_head_tied_to_embeddings": runtime.lm_head_tied_to_embeddings,
            "lm_head_fp8_separate_execution_head": (
                runtime.lm_head_fp8_separate_execution_head
            ),
            "lm_head_fp8_safe": lm_head_validation.get("lm_head_fp8_safe"),
            "lm_head_fp8_top1_agreement": lm_head_validation.get(
                "lm_head_fp8_top1_agreement"
            ),
            "lm_head_fp8_top5_overlap": lm_head_validation.get(
                "lm_head_fp8_top5_overlap"
            ),
            "lm_head_fp8_validation_required": bool(
                args.validate_lm_head_fp8
            ),
            "body_fp8_original_bytes": runtime.body_fp8_original_bytes,
            "body_fp8_resident_bytes": runtime.body_fp8_resident_bytes,
            "body_fp8_memory_saved_bytes": runtime.body_fp8_memory_saved_bytes,
            "fp8_layers": args.fp8_layers or "all",
            "down_fp8_layers": (
                args.down_fp8_layers or args.fp8_layers or "all"
            ),
            "qkv_fp8_layers": (
                args.qkv_fp8_layers or args.fp8_layers or "all"
            ),
            "o_fp8_layers": (
                args.o_fp8_layers or args.fp8_layers or "all"
            ),
            "fp8_scale_block": args.fp8_scale_block,
            "lm_head_fp8_scale_block": args.lm_head_fp8_scale_block,
            "lm_head_backend_override": args.lm_head_backend,
            "lm_head_argmax_mode": args.lm_head_argmax_mode,
            "lm_head_topk_guard": args.lm_head_topk_guard,
            "gate_up_backend_override": args.gate_up_backend,
            "down_proj_backend_override": args.down_proj_backend,
            "attn_proj_backend_override": args.attn_proj_backend,
            "selective_quantization": any(
                (
                    args.lm_head_fp8,
                    args.down_proj_fp8,
                    args.gate_up_fp8,
                    args.mlp_fp8,
                    args.qkv_fp8,
                    args.o_proj_fp8,
                    args.attn_proj_fp8,
                )
            ),
            "exact_topk": args.exact_topk,
            "top_logits_computed_after_timing": True,
            "autotune_enabled": runtime.autotune_enabled,
            "per_shape_backend_choices": runtime.per_shape_backend_choices,
            "per_shape_backend_benchmarks": getattr(runtime, "per_shape_backend_benchmarks", {}),
            "per_shape_bandwidths": per_shape_bandwidths,
            "profile_breakdown": breakdown,
            **lm_head_components,
            "qk_time_ms": breakdown.get("qk_time_ms"),
            "softmax_time_ms": breakdown.get("softmax_time_ms"),
            "value_mix_time_ms": breakdown.get("value_mix_time_ms"),
            "attention_time_ms": breakdown.get("attention_time_ms"),
            "launches_per_token": getattr(runtime, "launches_per_token", 0),
            "matvec_launches_per_token": getattr(runtime, "matvec_launches_per_token", 0),
            "elementwise_launches_per_token": getattr(runtime, "elementwise_launches_per_token", 0),
            "attention_launches_per_token": getattr(runtime, "attention_launches_per_token", 0),
            "cuda_graphs_enabled": getattr(runtime, "_cuda_graphs_enabled", False),
            "cuda_graph_capture_s": getattr(runtime, "_cuda_graph_capture_s", 0.0),
            "cuda_graph_replay_tokens_per_s": (
                (max(1, args.steps) - 1) / max(1e-12, loop_s - (first_s or 0.0))
                if getattr(runtime, "_cuda_graphs_enabled", False) and max(1, args.steps) > 1 and loop_s > (first_s or 0.0)
                else None
            ),
            "cuda_graph_error": getattr(runtime, "_cuda_graph_error", None),
            **timed_geometry,
            "top_logits": top_logits,
            "rss_delta_bytes": None if rss0 is None or rss1 is None else rss1 - rss0,
            "gpu_peak_allocated_bytes": _gpu_peak_allocated(args.device),
            "gpu_peak_reserved_bytes": _gpu_peak_reserved(args.device),
            "gpu_peak_temp_c": guard.peak_temp,
            "thermal_stop": guard.too_hot,
        }
        if args.profile_runtime is not None:
            args.profile_runtime.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        weights.close()
        return result
    finally:
        guard.stop()


def cmd_compile_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from thinruntime.archive import ThinArchive

    archive = ThinArchive(args.archive)
    try:
        profile = build_runtime_profile(
            archive.manifest,
            str(args.archive),
            args.device,
            parse_bytes(args.vram),
            args.ctx,
            args.batch,
            args.kv,
            args.recent_window,
        )
    finally:
        archive.close()
    if args.out is not None:
        args.out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def cmd_optimize_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from thinruntime.archive import ThinArchive

    archive = ThinArchive(args.archive)
    candidates = []
    try:
        for residency in ["all", "stream"]:
            for kv in ["fp8", "q8", "q4", "q3", "q2"]:
                for prefetch in [0, 1, 2, 4]:
                    profile = build_runtime_profile(
                        archive.manifest,
                        str(args.archive),
                        args.device,
                        parse_bytes(args.vram),
                        args.ctx,
                        args.batch,
                        kv,
                        recent_window=256,
                        forced_residency=residency,
                        forced_prefetch=prefetch,
                    )
                    candidates.append(profile)
    finally:
        archive.close()
    candidates.sort(key=lambda item: (not item["fits"], item["expected_total_bytes"], item["expected_bytes_per_token"]))
    result = {
        "command": "optimize-runtime",
        "archive": str(args.archive),
        "device": args.device,
        "vram_bytes": parse_bytes(args.vram),
        "ctx": args.ctx,
        "batch": args.batch,
        "selected": candidates[0],
        "candidates": candidates[:12],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_runtime_profile(
    manifest: dict[str, Any],
    archive: str,
    device: str,
    vram_bytes: int,
    ctx: int,
    batch: int,
    kv: str,
    recent_window: int,
    forced_residency: str | None = None,
    forced_prefetch: int | None = None,
) -> dict[str, Any]:
    model = manifest["model"]
    pages = [page for page in manifest.get("pages", []) if page.get("kind") != "fused_physical"]
    globals_ = [page["id"] for page in pages if page.get("layer") is None]
    layer_pages = [page["id"] for page in pages if page.get("layer") is not None]
    unique_weight_bytes = unique_page_bytes(pages)
    global_bytes = unique_page_bytes([page for page in pages if page.get("layer") is None])
    scratch = int(manifest["memory_plan"]["scratch_bytes"]) * max(1, batch)
    kv_bytes = estimate_kv_bytes(manifest, ctx, batch, kv, recent_window)
    all_total = unique_weight_bytes + scratch + kv_bytes
    stream_peak_layer = max_layer_bytes(pages)
    stream_total = global_bytes + stream_peak_layer + scratch + kv_bytes
    residency = forced_residency or ("all" if all_total <= vram_bytes else "stream")
    expected_total = all_total if residency == "all" else stream_total
    prefetch = forced_prefetch if forced_prefetch is not None else (0 if residency == "all" else 2)
    fused_ops = detect_fused_ops(manifest)
    layouts = sorted({page.get("backend_layout", "generic") for page in manifest.get("pages", [])})
    native_layouts = [layout for layout in layouts if layout.startswith(device) or device in layout]
    streamed_bytes_per_token = 0 if residency == "all" else sum_page_bytes(pages, ids=layer_pages)
    kv_write_per_token = int(
        2
        * int(model["layers"])
        * int(model["kv_heads"])
        * (int(model["hidden_size"]) // int(model["heads"]))
        * 2
    )
    return {
        "command": "compile-runtime",
        "archive": archive,
        "device": device,
        "vram_bytes": vram_bytes,
        "ctx": ctx,
        "batch": batch,
        "fits": expected_total <= vram_bytes,
        "chosen_weight_layout": native_layouts[0] if native_layouts else "generic",
        "available_layouts": layouts,
        "chosen_kv_dtype": kv,
        "resident_pages": pages_to_ranges(globals_ if residency == "stream" else [page["id"] for page in pages]),
        "resident_page_count": len(globals_ if residency == "stream" else pages),
        "streamed_pages": pages_to_ranges(layer_pages if residency == "stream" else []),
        "streamed_page_count": len(layer_pages if residency == "stream" else []),
        "prefetch_distance": prefetch,
        "residency": residency,
        "fused_ops": fused_ops,
        "expected_weight_bytes": unique_weight_bytes if residency == "all" else global_bytes + stream_peak_layer,
        "expected_kv_bytes": kv_bytes,
        "expected_scratch_bytes": scratch,
        "expected_total_bytes": expected_total,
        "expected_bytes_per_token": streamed_bytes_per_token + kv_write_per_token,
        "expected_first_token_latency": "profile_required",
        "expected_max_ctx": max_context_for_budget(manifest, vram_bytes, batch, kv, recent_window, residency, global_bytes, stream_peak_layer, scratch),
        "feature_matrix": runtime_feature_matrix(device),
        "compatibility": compatibility_matrix(device),
    }


def detect_fused_ops(manifest: dict[str, Any]) -> list[str]:
    ids = {page["id"] for page in manifest.get("pages", []) if page.get("kind") == "fused_physical"}
    fused = []
    if any("attn_qkv_fused" in page_id for page_id in ids):
        fused.append("qkv")
    if any("mlp_gate_up_fused" in page_id for page_id in ids):
        fused.append("gate_up")
    return fused


def runtime_feature_matrix(device: str) -> dict[str, str]:
    cuda = device == "cuda" and torch.cuda.is_available()
    return {
        "direct_gpu_page_loader": "implemented",
        "gpu_resident_page_pool": "implemented",
        "execution_tape_prefetch": "implemented_sync_v0",
        "layer_streaming_runtime": "implemented",
        "kv_cache_manager": "implemented_v0",
        "paged_kv_cache": "implemented_v0",
        "kv_eviction_pruning": "implemented_v0",
        "cuda_vmm_kv": "capability_gate" if cuda else "unavailable",
        "fused_qkv_gpu_pages": "implemented_when_archive_fused",
        "fused_gate_up_gpu_pages": "implemented_when_archive_fused",
        "triton_decode_kernels": "implemented_experimental",
        "triton_matvec_backend": "implemented_experimental",
        "selected_default_kernel_backend": "triton_matvec",
        "backend_native_packed_layouts": "manifest_contract_v0",
        "multi_layout_archive": "manifest_contract_v0",
        "cuda_graph_capture": "capability_gate" if cuda else "unavailable",
        "persistent_decode_buffers": "arena_contract_v0",
        "static_vram_arena_allocator": "arena_contract_v0",
        "async_cpu_gpu_pipeline": "copy_stream_v0",
        "gpudirect_storage": "capability_gate" if cuda else "unavailable",
        "moe_hot_cold_residency": "profile_contract_v0",
        "speculative_decoding_slot": "profile_contract_v0",
        "mixed_precision_truth_budget": "page_dtype_contract_v0",
        "outlier_sidecar_pages": "page_kind_contract_v0",
        "runtime_verify_modes": "cli_contract_v0",
        "page_level_telemetry": "implemented",
        "runtime_profiler": "implemented",
        "runtime_optimizer": "implemented",
        "partial_tensor_loading": "implemented_embeddings_mode",
        "lora_overlays": "profile_contract_v0",
        "adapter_hot_swap": "profile_contract_v0",
        "quant_kv_weight_matrix": "implemented",
        "bytes_per_token_compiler": "implemented",
    }


def compatibility_matrix(device: str) -> dict[str, Any]:
    capability = None
    if device == "cuda" and torch.cuda.is_available():
        capability = ".".join(map(str, torch.cuda.get_device_capability()))
    return {
        "device": device,
        "cuda_capability": capability,
        "weights": ["fp32", "fp16", "bf16", "q8_contract", "q4_contract"],
        "kv": ["fp16", "bf16", "fp8", "q8", "q4", "q3", "q2"],
        "nvfp4_kv": bool(capability and int(capability.split(".")[0]) >= 10),
    }


def unsupported_runtime_flags(args: argparse.Namespace) -> dict[str, str]:
    gates = {}
    if args.kv_layout == "cuda-vmm":
        gates["cuda_vmm_kv"] = "requested; v0 records profile contract but uses paged KV backend"
    if args.cuda_graphs and (args.device != "cuda" or args.residency != "all"):
        gates["cuda_graphs"] = "requested; CUDA Graphs only supported in all-resident CUDA mode"
    if args.gds:
        gates["gpudirect_storage"] = "requested; v0 falls back to mmap plus pinned staging"
    if args.speculative or args.draft is not None:
        gates["speculative_decoding"] = "requested; v0 reserves profile slot but does not verify draft tokens"
    if args.arena:
        gates["arena"] = "requested; v0 reports pool/arena bytes but Torch allocator owns physical suballocation"
    if args.runtime_profile is not None:
        gates["runtime_profile"] = f"loaded request path {args.runtime_profile}; v0 does not replay every field yet"
    return gates


def unique_page_bytes(pages: list[dict[str, Any]]) -> int:
    seen = set()
    total = 0
    for page in pages:
        key = (page["checksum"], tuple(page["shape"]), page["dtype"])
        if key in seen:
            continue
        seen.add(key)
        total += int(page["size"])
    return total


def sum_page_bytes(pages: list[dict[str, Any]], ids: list[str]) -> int:
    wanted = set(ids)
    return unique_page_bytes([page for page in pages if page["id"] in wanted])


def max_layer_bytes(pages: list[dict[str, Any]]) -> int:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        if page.get("layer") is not None:
            by_layer.setdefault(int(page["layer"]), []).append(page)
    return max((unique_page_bytes(layer_pages) for layer_pages in by_layer.values()), default=0)


def estimate_kv_bytes(
    manifest: dict[str, Any],
    ctx: int,
    batch: int,
    codec: str,
    recent_window: int,
) -> int:
    model = manifest["model"]
    head_dim = int(
        model.get("head_dim")
        or (int(model["hidden_size"]) // int(model["heads"]))
    )
    scalars = 2 * int(model["layers"]) * int(model["kv_heads"]) * head_dim
    recent = min(ctx, recent_window)
    old = max(0, ctx - recent)
    return int((scalars * recent * 2 + scalars * old * codec_bytes(codec)) * max(1, batch))


def max_context_for_budget(
    manifest: dict[str, Any],
    vram: int,
    batch: int,
    codec: str,
    recent_window: int,
    residency: str,
    global_bytes: int,
    stream_peak_layer: int,
    scratch: int,
) -> int:
    fixed = scratch + (global_bytes + stream_peak_layer if residency == "stream" else unique_page_bytes([
        page for page in manifest.get("pages", []) if page.get("kind") != "fused_physical"
    ]))
    if fixed >= vram:
        return 0
    lo, hi = 0, 262144
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fixed + estimate_kv_bytes(manifest, mid, batch, codec, recent_window) <= vram:
            lo = mid
        else:
            hi = mid - 1
    return lo


def pages_to_ranges(page_ids: list[str]) -> list[str]:
    limit = 32
    return page_ids[:limit] + ([f"... {len(page_ids) - limit} more"] if len(page_ids) > limit else [])


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


def parse_bytes(value: str) -> int:
    text = str(value).strip()
    if not text or text == "0":
        return 0
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    number = ""
    unit = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            number += ch
        elif not ch.isspace():
            unit += ch.lower()
    if not number:
        raise ValueError(f"cannot parse byte value {value}")
    return int(float(number) * units.get(unit or "b", 1))


def codec_bytes(codec: str) -> float:
    return {
        "q2": 0.25,
        "q3": 0.375,
        "q4": 0.5,
        "nvfp4": 0.5,
        "q5": 0.625,
        "q6": 0.75,
        "q8": 1.0,
        "fp8": 1.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "fp32": 4.0,
    }.get(codec.lower(), 2.0)


def reset_gpu(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def print_text(result: dict[str, Any]) -> None:
    print(f"ThinTensor {result.get('command', 'runtime')}")
    for key, value in result.items():
        if isinstance(value, (dict, list)):
            continue
        if key.endswith("_bytes") and isinstance(value, int):
            print(f"{key}: {format_bytes(value)}")
        else:
            print(f"{key}: {value}")
    for key in ["page_pool", "kv_cache", "runtime_capability_gates", "selected"]:
        if key in result:
            print()
            print(f"{key}:")
            print(json.dumps(result[key], indent=2, sort_keys=True))


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


if __name__ == "__main__":
    main()
