#!/usr/bin/env python3
"""Compare HF and ThinTensor logits on an identical token trajectory."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.gpu_runtime import (
    PagedKVCache,
    ThinGpuPagePool,
    ThinGpuQwenRuntime,
    ThinGpuWeights,
)
from thinruntime.model_arch import descriptor_from_hf_config, descriptor_from_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", "--hf-path", dest="hf_model", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--kernel-backend",
        choices=["torch", "triton", "triton-matvec"],
        default="triton-matvec",
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--prefill-lens", default="1,8,32,128")
    parser.add_argument("--steps", default="1,10,50")
    parser.add_argument(
        "--attention-mode",
        choices=["causal_kv", "current_only", "current_only_smoke"],
        default="causal_kv",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["torch", "triton_fused"],
        default="torch",
    )
    parser.add_argument("--lm-head-fp8", action="store_true")
    parser.add_argument("--lm-head-topk-guard", type=int, default=0)
    parser.add_argument(
        "--lm-head-argmax-mode",
        choices=["torch", "triton_two_stage", "triton_persistent"],
        default="torch",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--kv-data-layout",
        choices=sorted(PagedKVCache.LAYOUTS),
        default="head_token_interleaved",
    )
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument(
        "--kv-residency",
        choices=["gpu_full", "cpu_exact", "hybrid_recent"],
        default="gpu_full",
    )
    parser.add_argument("--kv-gpu-recent-tokens", type=int, default=256)
    parser.add_argument("--kv-prefetch-pages", type=int, default=0)
    parser.add_argument(
        "--weight-residency",
        choices=["all", "stream"],
        default="all",
    )
    parser.add_argument("--gpu-weight-budget", default="0")
    parser.add_argument("--weight-prefetch-layers", type=int, default=1)
    parser.add_argument("--cpu-weight-offload", action="store_true")
    parser.add_argument("--pin-cpu-weight-pages", action="store_true")
    parser.add_argument("--mlp-fp8", action="store_true")
    parser.add_argument("--down-proj-fp8", action="store_true")
    parser.add_argument("--gate-up-fp8", action="store_true")
    parser.add_argument("--fused-scaled-mlp", action="store_true")
    parser.add_argument("--split-k-down-proj", action="store_true")
    parser.add_argument("--attn-proj-fp8", action="store_true")
    parser.add_argument("--qkv-fp8", action="store_true")
    parser.add_argument("--o-proj-fp8", action="store_true")
    parser.add_argument(
        "--fp8-layers",
        "--mlp-fp8-layers",
        dest="fp8_layers",
    )
    parser.add_argument("--down-fp8-layers")
    parser.add_argument("--qkv-fp8-layers")
    parser.add_argument("--o-fp8-layers")
    parser.add_argument("--fp8-scale-block", type=int, default=0)
    parser.add_argument("--lm-head-fp8-scale-block", type=int, default=0)
    parser.add_argument("--exact-hf-mode", action="store_true")
    parser.add_argument("--dump-layer-debug", type=int)
    parser.add_argument("--dump-token-debug", type=int)
    parser.add_argument("--compare-attention", action="store_true")
    parser.add_argument("--compare-hidden-states", action="store_true")
    parser.add_argument("--compare-components", action="store_true")
    parser.add_argument(
        "--find-first-divergence",
        action="store_true",
        help="Trace all decoder layers and report the first tensor below 0.999 cosine",
    )
    parser.add_argument("--out-debug-dir", default="/tmp/thin_debug")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default="hf_vs_thin_correctness_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    prefill_lens = parse_positive_csv(args.prefill_lens, "prefill-lens")
    requested_steps = parse_positive_csv(args.steps, "steps")
    max_steps = max(requested_steps)
    attention_mode = (
        "current_only_smoke"
        if args.attention_mode == "current_only"
        else args.attention_mode
    )

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    input_ids_by_length = {
        length: prompt_ids(tokenizer, args.prompt, length)
        for length in prefill_lens
    }
    hf_descriptor = descriptor_from_hf_config(Path(args.hf_model))
    hf_config_summary = summarize_hf_config(Path(args.hf_model))

    print(f"loading HF model {args.hf_model}", file=sys.stderr)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to("cpu")
    hf_model.eval()
    hf_attention_implementation = getattr(
        hf_model.config, "_attn_implementation", "unknown"
    )
    if args.exact_hf_mode:
        hf_model.config._attn_implementation = "eager"
        hf_attention_implementation = "eager"
    hf_runs: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for prefill_len, input_ids in input_ids_by_length.items():
            hf_runs[prefill_len] = run_hf_trajectory(
                hf_model,
                input_ids.to("cpu"),
                requested_steps,
                max_steps,
            )
    detailed_debug_requested = any(
        (
            args.compare_attention,
            args.compare_hidden_states,
            args.compare_components,
            args.dump_layer_debug is not None,
        )
    )
    debug_requested = detailed_debug_requested or args.find_first_divergence
    if debug_requested and args.weight_residency == "stream":
        raise ValueError(
            "component debug capture currently requires --weight-residency all"
        )
    hf_debug: dict[str, torch.Tensor] | None = None
    hf_layer_trace: dict[str, torch.Tensor] | None = None
    debug_sequence: torch.Tensor | None = None
    debug_layer = args.dump_layer_debug if args.dump_layer_debug is not None else 0
    debug_token = args.dump_token_debug if args.dump_token_debug is not None else 0
    if debug_requested:
        debug_prefill = prefill_lens[0]
        debug_sequence = build_debug_sequence(
            input_ids_by_length[debug_prefill],
            hf_runs[debug_prefill]["tokens"],
            debug_token,
        )
        if args.find_first_divergence:
            hf_layer_trace = capture_hf_layer_trace(
                hf_model,
                debug_sequence.to("cpu"),
                token_position=debug_token,
                descriptor=hf_descriptor,
            )
        if detailed_debug_requested:
            hf_debug = capture_hf_components(
                hf_model,
                debug_sequence.to("cpu"),
                layer_index=debug_layer,
                token_position=debug_token,
                descriptor=hf_descriptor,
            )
    del hf_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    os.environ.setdefault("THINTENSOR_DISABLE_AUTOTUNE", "1")
    print(f"loading ThinTensor archive {args.archive}", file=sys.stderr)
    if args.weight_residency == "stream":
        weight_budget = parse_bytes(args.gpu_weight_budget)
        if weight_budget <= 0:
            raise ValueError(
                "--weight-residency stream requires --gpu-weight-budget"
            )
        weights: ThinGpuWeights | ThinGpuPagePool = ThinGpuPagePool(
            args.archive,
            device=args.device,
            dtype=dtype,
            vram_budget_bytes=weight_budget,
            prefetch_distance=args.weight_prefetch_layers,
            cpu_offload=args.cpu_weight_offload,
            pin_cpu_pages=args.pin_cpu_weight_pages,
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
    else:
        weights = ThinGpuWeights(
            args.archive,
            device=args.device,
            dtype=dtype,
        )
    thin_descriptor = descriptor_from_manifest(weights.manifest)
    thin_config_summary = summarize_thin_config(
        weights.manifest,
        thin_descriptor,
    )
    config_mismatches = compare_config_summaries(
        hf_config_summary,
        thin_config_summary,
    )
    descriptor_mismatches = compare_descriptors(hf_descriptor, thin_descriptor)
    if descriptor_mismatches:
        weights.close()
        raise RuntimeError(
            "HF config and .thin manifest disagree: "
            + "; ".join(descriptor_mismatches)
        )

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prefill_len, input_ids in input_ids_by_length.items():
            hf_run = hf_runs[prefill_len]
            thin_run = run_thin_trajectory(
                weights=weights,
                input_ids=input_ids.to(device),
                teacher_tokens=hf_run["tokens"],
                requested_steps=requested_steps,
                max_steps=max_steps,
                attention_mode=attention_mode,
                attention_backend=args.attention_backend,
                lm_head_fp8=args.lm_head_fp8,
                mlp_fp8=args.mlp_fp8,
                down_proj_fp8=args.down_proj_fp8,
                gate_up_fp8=args.gate_up_fp8,
                attn_proj_fp8=args.attn_proj_fp8,
                qkv_fp8=args.qkv_fp8,
                o_proj_fp8=args.o_proj_fp8,
                fp8_layer_spec=args.fp8_layers,
                down_fp8_layer_spec=args.down_fp8_layers,
                qkv_fp8_layer_spec=args.qkv_fp8_layers,
                o_fp8_layer_spec=args.o_fp8_layers,
                kernel_backend=args.kernel_backend,
                exact_hf_mode=args.exact_hf_mode,
                fp8_scale_block=args.fp8_scale_block,
                lm_head_fp8_scale_block=args.lm_head_fp8_scale_block,
                dtype=dtype,
                lm_head_topk_guard=args.lm_head_topk_guard,
                kv_data_layout=args.kv_data_layout,
                kv_block_size=args.kv_block_size,
                kv_residency=args.kv_residency,
                kv_gpu_recent_tokens=args.kv_gpu_recent_tokens,
                kv_prefetch_pages=args.kv_prefetch_pages,
                lm_head_backend=args.lm_head_backend,
                fused_scaled_mlp=args.fused_scaled_mlp,
                split_k_down_proj=args.split_k_down_proj,
                lm_head_argmax_mode=args.lm_head_argmax_mode,
                stream_weights=args.weight_residency == "stream",
                weight_prefetch_layers=args.weight_prefetch_layers,
            )
            for step in requested_steps:
                records.append(
                    comparison_record(
                        args=args,
                        prefill_len=prefill_len,
                        step=step,
                        hf_logits=hf_run["logits"][step],
                        thin_logits=thin_run["logits"][step],
                        hf_tokens=hf_run["tokens"][:step],
                        thin_tokens=thin_run["tokens"][:step],
                        attention_mode=attention_mode,
                        kv_tokens_attended=thin_run["kv_tokens_attended"][step],
                        kv_read_bytes=thin_run["kv_read_bytes"][step],
                    )
                )
    debug_report = None
    first_divergence_report = None
    if detailed_debug_requested:
        assert hf_debug is not None and debug_sequence is not None
        thin_debug = capture_thin_components(
            weights=weights,
            token_ids=debug_sequence.to(device),
            layer_index=debug_layer,
            token_position=debug_token,
            attention_mode=attention_mode,
            lm_head_fp8=args.lm_head_fp8,
            mlp_fp8=args.mlp_fp8,
            down_proj_fp8=args.down_proj_fp8,
            gate_up_fp8=args.gate_up_fp8,
            attn_proj_fp8=args.attn_proj_fp8,
            qkv_fp8=args.qkv_fp8,
            o_proj_fp8=args.o_proj_fp8,
            fp8_layer_spec=args.fp8_layers,
            down_fp8_layer_spec=args.down_fp8_layers,
            qkv_fp8_layer_spec=args.qkv_fp8_layers,
            o_fp8_layer_spec=args.o_fp8_layers,
            kernel_backend=args.kernel_backend,
            exact_hf_mode=args.exact_hf_mode,
            fp8_scale_block=args.fp8_scale_block,
            dtype=dtype,
        )
        debug_report = compare_component_sets(
            hf_debug,
            thin_debug,
            layer_index=debug_layer,
            token_position=debug_token,
            out_dir=Path(args.out_debug_dir),
        )
    if args.find_first_divergence:
        assert hf_layer_trace is not None and debug_sequence is not None
        thin_layer_trace = capture_thin_all_components(
            weights=weights,
            token_ids=debug_sequence.to(device),
            token_position=debug_token,
            attention_mode=attention_mode,
            lm_head_fp8=args.lm_head_fp8,
            mlp_fp8=args.mlp_fp8,
            down_proj_fp8=args.down_proj_fp8,
            gate_up_fp8=args.gate_up_fp8,
            attn_proj_fp8=args.attn_proj_fp8,
            qkv_fp8=args.qkv_fp8,
            o_proj_fp8=args.o_proj_fp8,
            fp8_layer_spec=args.fp8_layers,
            down_fp8_layer_spec=args.down_fp8_layers,
            qkv_fp8_layer_spec=args.qkv_fp8_layers,
            o_fp8_layer_spec=args.o_fp8_layers,
            kernel_backend=args.kernel_backend,
            exact_hf_mode=args.exact_hf_mode,
            fp8_scale_block=args.fp8_scale_block,
            dtype=dtype,
        )
        first_divergence_report = compare_layer_traces(
            hf_layer_trace,
            thin_layer_trace,
            layers=thin_descriptor.num_hidden_layers,
            token_position=debug_token,
            out_dir=Path(args.out_debug_dir),
        )
    if debug_requested:
        write_attention_parity_report(
            debug_report,
            first_divergence_report=first_divergence_report,
        )
    weights.close()

    summary = build_summary(
        args,
        hf_descriptor.as_dict(),
        thin_descriptor.as_dict(),
        hf_config_summary,
        thin_config_summary,
        config_mismatches,
        records,
        hf_attention_implementation,
    )
    if debug_report is not None:
        summary["attention_parity_report"] = debug_report
    if first_divergence_report is not None:
        summary["first_divergence_report"] = first_divergence_report
    out_path = Path(args.out)
    out_path.write_text(render_markdown(summary), encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"wrote {out_path} and {json_path}; "
            f"hf_equivalent={summary['hf_equivalent']}"
        )


def run_hf_trajectory(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    requested_steps: list[int],
    max_steps: int,
) -> dict[str, Any]:
    # Full recomputation is the semantic gold reference. HF cached decode can
    # differ measurably from HF full recomputation in BF16 because prefill and
    # decode use different reduction shapes; that kernel-ordering drift must
    # not be misdiagnosed as a ThinTensor attention error.
    current_ids = input_ids
    outputs = model(input_ids=current_ids, use_cache=False)
    logits = outputs.logits[0, -1].float()
    captured: dict[int, torch.Tensor] = {}
    tokens: list[torch.Tensor] = []
    for step in range(1, max_steps + 1):
        if step in requested_steps:
            captured[step] = logits.detach().cpu()
        token = torch.argmax(logits).reshape(())
        tokens.append(token.detach().cpu())
        if step < max_steps:
            current_ids = torch.cat(
                (current_ids, token.reshape(1, 1)),
                dim=1,
            )
            outputs = model(
                input_ids=current_ids,
                use_cache=False,
            )
            logits = outputs.logits[0, -1].float()
    return {"logits": captured, "tokens": tokens}


def run_thin_trajectory(
    weights: ThinGpuWeights | ThinGpuPagePool,
    input_ids: torch.Tensor,
    teacher_tokens: list[torch.Tensor],
    requested_steps: list[int],
    max_steps: int,
    attention_mode: str,
    attention_backend: str,
    lm_head_fp8: bool,
    mlp_fp8: bool,
    down_proj_fp8: bool,
    gate_up_fp8: bool,
    attn_proj_fp8: bool,
    qkv_fp8: bool,
    o_proj_fp8: bool,
    fp8_layer_spec: str | None,
    down_fp8_layer_spec: str | None,
    qkv_fp8_layer_spec: str | None,
    o_fp8_layer_spec: str | None,
    kernel_backend: str,
    exact_hf_mode: bool,
    fp8_scale_block: int,
    lm_head_fp8_scale_block: int,
    dtype: torch.dtype,
    lm_head_topk_guard: int,
    kv_data_layout: str,
    kv_block_size: int,
    kv_residency: str,
    kv_gpu_recent_tokens: int,
    kv_prefetch_pages: int,
    lm_head_backend: str | None,
    fused_scaled_mlp: bool,
    split_k_down_proj: bool,
    lm_head_argmax_mode: str = "torch",
    stream_weights: bool = False,
    weight_prefetch_layers: int = 1,
) -> dict[str, Any]:
    model = weights.manifest["model"]
    cache = PagedKVCache(
        layers=int(model["layers"]),
        kv_heads=int(model["kv_heads"]),
        head_dim=int(
            model.get("head_dim")
            or int(model["hidden_size"]) // int(model["heads"])
        ),
        device=weights.device,
        dtype=dtype,
        policy="full",
        old_codec="bf16",
        layout=kv_data_layout,
        block_size=kv_block_size,
        residency=kv_residency,
        gpu_recent_tokens=kv_gpu_recent_tokens,
        prefetch_pages=kv_prefetch_pages,
    )
    runtime = ThinGpuQwenRuntime(
        weights,
        kv_cache=cache,
        prefetch_distance=weight_prefetch_layers if stream_weights else 0,
        evict_completed_layers=stream_weights,
        kernel_backend=(
            "torch"
            if exact_hf_mode or weights.device.type != "cuda"
            else kernel_backend
        ),
        lm_head_fp8=lm_head_fp8,
        lm_head_topk_guard=lm_head_topk_guard,
        lm_head_backend=lm_head_backend,
        mlp_fp8=mlp_fp8,
        down_proj_fp8=down_proj_fp8,
        gate_up_fp8=gate_up_fp8,
        fused_scaled_mlp=fused_scaled_mlp,
        split_k_down_proj=split_k_down_proj,
        attn_proj_fp8=attn_proj_fp8,
        qkv_fp8=qkv_fp8,
        o_proj_fp8=o_proj_fp8,
        fp8_layer_spec=fp8_layer_spec,
        down_fp8_layer_spec=down_fp8_layer_spec,
        qkv_fp8_layer_spec=qkv_fp8_layer_spec,
        o_fp8_layer_spec=o_fp8_layer_spec,
        exact_hf_mode=exact_hf_mode,
        fp8_scale_block=fp8_scale_block,
        lm_head_fp8_scale_block=lm_head_fp8_scale_block,
        attention_mode=attention_mode,
        attention_backend=attention_backend,
        lm_head_argmax_mode=lm_head_argmax_mode,
    )
    hidden = None
    for position in range(int(input_ids.shape[1])):
        hidden = runtime.forward_token(
            input_ids[0, position],
            token_index=position,
        )
    assert hidden is not None

    captured: dict[int, torch.Tensor] = {}
    tokens: list[torch.Tensor] = []
    attended: dict[int, int] = {}
    read_bytes: dict[int, int] = {}
    for step in range(1, max_steps + 1):
        logits = runtime.logits(hidden).float()
        if step in requested_steps:
            captured[step] = logits.detach().cpu()
            attended[step] = runtime._last_kv_tokens_attended
            read_bytes[step] = cache.read_bytes_per_token
        tokens.append(runtime.next_token_tensor(hidden).detach().cpu())
        if step < max_steps:
            teacher_token = teacher_tokens[step - 1].to(
                device=weights.device,
                dtype=torch.long,
            )
            hidden = runtime.forward_token(
                teacher_token,
                token_index=int(input_ids.shape[1]) + step - 1,
            )
    return {
        "logits": captured,
        "tokens": tokens,
        "kv_tokens_attended": attended,
        "kv_read_bytes": read_bytes,
    }


def build_debug_sequence(
    prompt_ids_tensor: torch.Tensor,
    generated_tokens: list[torch.Tensor],
    token_position: int,
) -> torch.Tensor:
    if token_position < 0:
        raise ValueError("--dump-token-debug must be non-negative")
    sequence = prompt_ids_tensor.clone()
    if token_position >= int(sequence.shape[1]):
        needed = token_position - int(sequence.shape[1]) + 1
        if needed > len(generated_tokens):
            raise ValueError(
                f"debug token {token_position} requires {needed} generated "
                f"tokens, but --steps only produced {len(generated_tokens)}"
            )
        continuation = torch.stack(generated_tokens[:needed]).reshape(1, -1)
        sequence = torch.cat((sequence, continuation), dim=1)
    return sequence[:, : token_position + 1].to(dtype=torch.long)


def capture_hf_components(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    layer_index: int,
    token_position: int,
    descriptor: Any,
) -> dict[str, torch.Tensor]:
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    layers = getattr(base, "layers", None) or getattr(base, "h", None)
    if layers is None or layer_index < 0 or layer_index >= len(layers):
        raise RuntimeError(
            f"HF component hooks do not support {type(model).__name__}: "
            "decoder layers were not found"
        )
    layer = layers[layer_index]
    attention = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    required = {
        "input_layernorm": getattr(layer, "input_layernorm", None),
        "post_attention_layernorm": getattr(
            layer, "post_attention_layernorm", None
        ),
        "q_proj": getattr(attention, "q_proj", None),
        "k_proj": getattr(attention, "k_proj", None),
        "v_proj": getattr(attention, "v_proj", None),
        "o_proj": getattr(attention, "o_proj", None),
        "gate_proj": getattr(mlp, "gate_proj", None),
        "up_proj": getattr(mlp, "up_proj", None),
        "down_proj": getattr(mlp, "down_proj", None),
    }
    missing = [name for name, module in required.items() if module is None]
    if missing:
        raise RuntimeError(
            f"unsupported HF layer {type(layer).__name__}; missing modules: "
            + ", ".join(missing)
        )

    captured: dict[str, torch.Tensor] = {}
    full: dict[str, torch.Tensor] = {}
    handles = []

    def tensor_output(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(
            output[0], torch.Tensor
        ):
            return output[0]
        raise RuntimeError(f"hook output is not tensor-like: {type(output)}")

    def select_position(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() >= 3 and tensor.shape[0] == 1:
            return tensor[0, token_position].reshape(-1)
        if tensor.dim() >= 2 and tensor.shape[0] == 1:
            return tensor[0].reshape(-1)
        return tensor.reshape(-1)

    def save_output(name: str, keep_full: bool = False):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any):
            tensor = tensor_output(output).detach()
            captured[name] = select_position(tensor).clone()
            if keep_full:
                full[name] = tensor.clone()

        return hook

    def save_pre(name: str, keep_full: bool = False):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]):
            tensor = tensor_output(inputs).detach()
            captured[name] = select_position(tensor).clone()
            if keep_full:
                full[name] = tensor.clone()

        return hook

    handles.append(layer.register_forward_pre_hook(save_pre("layer_input")))
    handles.append(
        required["input_layernorm"].register_forward_hook(
            save_output("post_input_rmsnorm")
        )
    )
    for name in ("q_proj", "k_proj", "v_proj"):
        handles.append(
            required[name].register_forward_hook(
                save_output(
                    {
                        "q_proj": "q_projection",
                        "k_proj": "k_projection",
                        "v_proj": "v_projection",
                    }[name],
                    keep_full=True,
                )
            )
        )
    handles.append(
        required["o_proj"].register_forward_pre_hook(
            save_pre("attention_output_before_o_proj")
        )
    )
    handles.append(
        required["o_proj"].register_forward_hook(save_output("o_proj_output"))
    )
    handles.append(
        required["post_attention_layernorm"].register_forward_hook(
            save_output("post_attention_rmsnorm")
        )
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        handles.append(
            required[name].register_forward_hook(
                save_output(
                    {
                        "gate_proj": "gate_projection",
                        "up_proj": "up_projection",
                        "down_proj": "down_projection",
                    }[name]
                )
            )
        )
    handles.append(
        layer.register_forward_hook(save_output("final_residual_after_mlp"))
    )
    final_norm = getattr(base, "norm", None)
    if final_norm is not None:
        handles.append(
            final_norm.register_forward_hook(save_output("final_norm"))
        )

    previous_attention = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=token_ids,
                use_cache=False,
                output_attentions=False,
            )
    finally:
        for handle in handles:
            handle.remove()
        if previous_attention is not None:
            model.config._attn_implementation = previous_attention

    captured["post_attention_residual"] = (
        captured["layer_input"] + captured["o_proj_output"]
    )
    if descriptor.activation != "silu":
        raise RuntimeError(
            f"debug gated activation does not support {descriptor.activation!r}"
        )
    captured["gated_activation"] = F.silu(
        captured["gate_projection"]
    ) * captured["up_projection"]
    captured["final_logits"] = outputs.logits[
        0, token_position
    ].detach().clone()

    q = full["q_projection"].reshape(
        1,
        int(token_ids.shape[1]),
        descriptor.num_attention_heads,
        descriptor.head_dim,
    )[0]
    k = full["k_projection"].reshape(
        1,
        int(token_ids.shape[1]),
        descriptor.num_key_value_heads,
        descriptor.head_dim,
    )[0]
    v = full["v_projection"].reshape(
        1,
        int(token_ids.shape[1]),
        descriptor.num_key_value_heads,
        descriptor.head_dim,
    )[0]
    q_norm_module = getattr(attention, "q_norm", None)
    k_norm_module = getattr(attention, "k_norm", None)
    if q_norm_module is not None:
        q = q_norm_module(q)
    if k_norm_module is not None:
        k = k_norm_module(k)
    captured["q_after_q_norm"] = q[token_position].reshape(-1).detach().clone()
    captured["k_after_k_norm"] = k[token_position].reshape(-1).detach().clone()
    q, k = apply_rope_reference(
        q,
        k,
        descriptor,
        layer_index,
    )
    captured["q_after_rope"] = q[token_position].reshape(-1).detach().clone()
    captured["k_after_rope"] = k[token_position].reshape(-1).detach().clone()

    keys = k[: token_position + 1].permute(1, 0, 2)
    values = v[: token_position + 1].permute(1, 0, 2)
    repeated_keys = keys.repeat_interleave(
        descriptor.num_attention_heads
        // descriptor.num_key_value_heads,
        dim=0,
    )
    repeated_values = values.repeat_interleave(
        descriptor.num_attention_heads
        // descriptor.num_key_value_heads,
        dim=0,
    )
    query = q[token_position]
    scores = torch.matmul(
        query.unsqueeze(1), repeated_keys.transpose(1, 2)
    ).squeeze(1)
    scores.mul_(descriptor.head_dim**-0.5)
    probabilities = torch.softmax(
        scores, dim=-1, dtype=torch.float32
    ).to(dtype=query.dtype)
    mixed = torch.matmul(
        probabilities.unsqueeze(1), repeated_values
    ).squeeze(1)
    captured["attention_scores"] = scores.detach().clone()
    captured["attention_probs"] = probabilities.detach().clone()
    # The hook is authoritative, but retain the recomputed tensor for auditing.
    captured["attention_output_recomputed"] = mixed.reshape(-1).detach().clone()
    return {name: tensor.cpu() for name, tensor in captured.items()}


def capture_hf_layer_trace(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    *,
    token_position: int,
    descriptor: Any,
) -> dict[str, torch.Tensor]:
    """Capture selected-token tensors across all HF decoder layers."""
    base = getattr(model, "model", None) or getattr(model, "transformer", None)
    layers = getattr(base, "layers", None) or getattr(base, "h", None)
    if layers is None:
        raise RuntimeError(
            f"HF layer tracing does not support {type(model).__name__}: "
            "decoder layers were not found"
        )

    captured: dict[str, torch.Tensor] = {}
    handles = []

    def tensor_output(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(
            output[0], torch.Tensor
        ):
            return output[0]
        raise RuntimeError(f"hook output is not tensor-like: {type(output)}")

    def select_position(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() >= 3 and tensor.shape[0] == 1:
            return tensor[0, token_position].reshape(-1)
        if tensor.dim() >= 2 and tensor.shape[0] == 1:
            return tensor[0].reshape(-1)
        return tensor.reshape(-1)

    def save_output(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any):
            captured[name] = select_position(
                tensor_output(output).detach()
            ).clone()

        return hook

    def save_pre(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]):
            captured[name] = select_position(
                tensor_output(inputs).detach()
            ).clone()

        return hook

    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        modules = {
            "post_input_rmsnorm": getattr(layer, "input_layernorm", None),
            "post_attention_rmsnorm": getattr(
                layer, "post_attention_layernorm", None
            ),
            "q_projection": getattr(attention, "q_proj", None),
            "k_projection": getattr(attention, "k_proj", None),
            "v_projection": getattr(attention, "v_proj", None),
            "o_proj": getattr(attention, "o_proj", None),
            "gate_projection": getattr(mlp, "gate_proj", None),
            "up_projection": getattr(mlp, "up_proj", None),
            "down_projection": getattr(mlp, "down_proj", None),
        }
        missing = [name for name, module in modules.items() if module is None]
        if missing:
            raise RuntimeError(
                f"unsupported HF layer {type(layer).__name__} at "
                f"index {layer_index}; missing modules: {', '.join(missing)}"
            )
        prefix = f"layer_{layer_index}."
        handles.append(
            layer.register_forward_pre_hook(save_pre(prefix + "layer_input"))
        )
        handles.append(
            modules["post_input_rmsnorm"].register_forward_hook(
                save_output(prefix + "post_input_rmsnorm")
            )
        )
        for name in ("q_projection", "k_projection", "v_projection"):
            handles.append(
                modules[name].register_forward_hook(save_output(prefix + name))
            )
        handles.append(
            modules["o_proj"].register_forward_pre_hook(
                save_pre(prefix + "attention_output_before_o_proj")
            )
        )
        handles.append(
            modules["o_proj"].register_forward_hook(
                save_output(prefix + "o_proj_output")
            )
        )
        handles.append(
            modules["post_attention_rmsnorm"].register_forward_hook(
                save_output(prefix + "post_attention_rmsnorm")
            )
        )
        for name in ("gate_projection", "up_projection", "down_projection"):
            handles.append(
                modules[name].register_forward_hook(save_output(prefix + name))
            )
        handles.append(
            layer.register_forward_hook(
                save_output(prefix + "final_residual_after_mlp")
            )
        )

    final_norm = getattr(base, "norm", None)
    if final_norm is not None:
        handles.append(
            final_norm.register_forward_hook(save_output("final_norm"))
        )
    previous_attention = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=token_ids,
                use_cache=False,
                output_attentions=False,
            )
    finally:
        for handle in handles:
            handle.remove()
        if previous_attention is not None:
            model.config._attn_implementation = previous_attention

    if descriptor.activation != "silu":
        raise RuntimeError(
            f"layer trace gated activation does not support "
            f"{descriptor.activation!r}"
        )
    for layer_index in range(len(layers)):
        prefix = f"layer_{layer_index}."
        captured[prefix + "post_attention_residual"] = (
            captured[prefix + "layer_input"]
            + captured[prefix + "o_proj_output"]
        )
        captured[prefix + "gated_activation"] = F.silu(
            captured[prefix + "gate_projection"]
        ) * captured[prefix + "up_projection"]
    captured["final_logits"] = outputs.logits[
        0, token_position
    ].detach().clone()
    return {name: tensor.cpu() for name, tensor in captured.items()}


def apply_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    descriptor: Any,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not descriptor.layer_uses_rope(layer_index):
        return q, k
    if descriptor.rope_scaling not in (None, {}):
        raise RuntimeError(
            f"debug RoPE scaling is unsupported: {descriptor.rope_scaling!r}"
        )
    positions = torch.arange(q.shape[0], device=q.device, dtype=torch.float32)
    indices = torch.arange(
        0, descriptor.head_dim, 2, device=q.device, dtype=torch.float32
    )
    inv_freq = descriptor.rope_theta ** (-indices / descriptor.head_dim)
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(dtype=q.dtype).unsqueeze(1)
    sin = embedding.sin().to(dtype=q.dtype).unsqueeze(1)
    half = descriptor.head_dim // 2
    q_rotated = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    k_rotated = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return q * cos + q_rotated * sin, k * cos + k_rotated * sin


def capture_thin_components(
    *,
    weights: ThinGpuWeights,
    token_ids: torch.Tensor,
    layer_index: int,
    token_position: int,
    attention_mode: str,
    lm_head_fp8: bool,
    mlp_fp8: bool,
    down_proj_fp8: bool,
    gate_up_fp8: bool,
    attn_proj_fp8: bool,
    qkv_fp8: bool,
    o_proj_fp8: bool,
    fp8_layer_spec: str | None,
    down_fp8_layer_spec: str | None,
    qkv_fp8_layer_spec: str | None,
    o_fp8_layer_spec: str | None,
    kernel_backend: str,
    exact_hf_mode: bool,
    fp8_scale_block: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    model = weights.manifest["model"]
    cache = PagedKVCache(
        layers=int(model["layers"]),
        kv_heads=int(model["kv_heads"]),
        head_dim=int(
            model.get("head_dim")
            or descriptor_from_manifest(weights.manifest).head_dim
        ),
        device=weights.device,
        dtype=dtype,
        policy="full",
        old_codec="bf16",
    )
    runtime = ThinGpuQwenRuntime(
        weights,
        kv_cache=cache,
        kernel_backend=(
            "torch"
            if exact_hf_mode or weights.device.type != "cuda"
            else kernel_backend
        ),
        lm_head_fp8=lm_head_fp8,
        mlp_fp8=mlp_fp8,
        down_proj_fp8=down_proj_fp8,
        gate_up_fp8=gate_up_fp8,
        attn_proj_fp8=attn_proj_fp8,
        qkv_fp8=qkv_fp8,
        o_proj_fp8=o_proj_fp8,
        fp8_layer_spec=fp8_layer_spec,
        down_fp8_layer_spec=down_fp8_layer_spec,
        qkv_fp8_layer_spec=qkv_fp8_layer_spec,
        o_fp8_layer_spec=o_fp8_layer_spec,
        attention_mode=attention_mode,
        exact_hf_mode=exact_hf_mode,
        fp8_scale_block=fp8_scale_block,
    )
    result = {}
    for position in range(int(token_ids.shape[1])):
        if position == token_position:
            result = runtime.forward_token_debug(
                token_ids[0, position],
                layer=layer_index,
                token_index=position,
            )
        else:
            runtime.forward_token(token_ids[0, position], token_index=position)
    return {name: tensor.detach().cpu() for name, tensor in result.items()}


def capture_thin_all_components(
    *,
    weights: ThinGpuWeights,
    token_ids: torch.Tensor,
    token_position: int,
    attention_mode: str,
    lm_head_fp8: bool,
    mlp_fp8: bool,
    down_proj_fp8: bool,
    gate_up_fp8: bool,
    attn_proj_fp8: bool,
    qkv_fp8: bool,
    o_proj_fp8: bool,
    fp8_layer_spec: str | None,
    down_fp8_layer_spec: str | None,
    qkv_fp8_layer_spec: str | None,
    o_fp8_layer_spec: str | None,
    kernel_backend: str,
    exact_hf_mode: bool,
    fp8_scale_block: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    model = weights.manifest["model"]
    cache = PagedKVCache(
        layers=int(model["layers"]),
        kv_heads=int(model["kv_heads"]),
        head_dim=int(
            model.get("head_dim")
            or descriptor_from_manifest(weights.manifest).head_dim
        ),
        device=weights.device,
        dtype=dtype,
        policy="full",
        old_codec="bf16",
    )
    runtime = ThinGpuQwenRuntime(
        weights,
        kv_cache=cache,
        kernel_backend=(
            "torch"
            if exact_hf_mode or weights.device.type != "cuda"
            else kernel_backend
        ),
        lm_head_fp8=lm_head_fp8,
        mlp_fp8=mlp_fp8,
        down_proj_fp8=down_proj_fp8,
        gate_up_fp8=gate_up_fp8,
        attn_proj_fp8=attn_proj_fp8,
        qkv_fp8=qkv_fp8,
        o_proj_fp8=o_proj_fp8,
        fp8_layer_spec=fp8_layer_spec,
        down_fp8_layer_spec=down_fp8_layer_spec,
        qkv_fp8_layer_spec=qkv_fp8_layer_spec,
        o_fp8_layer_spec=o_fp8_layer_spec,
        attention_mode=attention_mode,
        exact_hf_mode=exact_hf_mode,
        fp8_scale_block=fp8_scale_block,
    )
    result = {}
    for position in range(int(token_ids.shape[1])):
        if position == token_position:
            result = runtime.forward_token_debug_all(
                token_ids[0, position],
                token_index=position,
            )
        else:
            runtime.forward_token(token_ids[0, position], token_index=position)
    return {name: tensor.detach().cpu() for name, tensor in result.items()}


def compare_component_sets(
    hf: dict[str, torch.Tensor],
    thin: dict[str, torch.Tensor],
    *,
    layer_index: int,
    token_position: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hf, out_dir / "hf_components.pt")
    torch.save(thin, out_dir / "thin_components.pt")
    order = [
        "layer_input",
        "post_input_rmsnorm",
        "q_projection",
        "k_projection",
        "v_projection",
        "q_after_q_norm",
        "k_after_k_norm",
        "q_after_rope",
        "k_after_rope",
        "attention_scores",
        "attention_probs",
        "attention_output_before_o_proj",
        "o_proj_output",
        "post_attention_residual",
        "post_attention_rmsnorm",
        "gate_projection",
        "up_projection",
        "gated_activation",
        "down_projection",
        "final_residual_after_mlp",
        "final_norm",
        "final_logits",
    ]
    rows = []
    for name in order:
        if name not in hf or name not in thin:
            rows.append(
                {
                    "component": name,
                    "supported": False,
                    "reason": (
                        "missing from HF capture"
                        if name not in hf
                        else "missing from ThinTensor capture"
                    ),
                }
            )
            continue
        left = hf[name]
        right = thin[name]
        row: dict[str, Any] = {
            "component": name,
            "supported": True,
            "hf_shape": list(left.shape),
            "thin_shape": list(right.shape),
            "hf_dtype": str(left.dtype).replace("torch.", ""),
            "thin_dtype": str(right.dtype).replace("torch.", ""),
            "compared_in": "fp32",
        }
        if left.numel() != right.numel():
            row.update(
                {
                    "threshold_pass": False,
                    "reason": "shape/element count mismatch",
                }
            )
            rows.append(row)
            continue
        left_float = left.reshape(-1).float()
        right_float = right.reshape(-1).float()
        difference = (left_float - right_float).abs()
        cosine = float(F.cosine_similarity(left_float, right_float, dim=0))
        count = min(5, difference.numel())
        values, indices = torch.topk(difference, count)
        row.update(
            {
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "cosine_similarity": cosine,
                "top_offending_indices": [
                    {"flat_index": int(index), "abs_error": float(value)}
                    for value, index in zip(values, indices)
                ],
                "threshold": {"cosine_similarity": 0.999},
                "threshold_pass": cosine >= 0.999,
            }
        )
        rows.append(row)
    first_failure = next(
        (
            row["component"]
            for row in rows
            if row.get("supported") and not row.get("threshold_pass", False)
        ),
        None,
    )
    return {
        "layer": layer_index,
        "token_position": token_position,
        "components": rows,
        "first_failing_component": first_failure,
        "hf_dump": str(out_dir / "hf_components.pt"),
        "thin_dump": str(out_dir / "thin_components.pt"),
    }


def compare_layer_traces(
    hf: dict[str, torch.Tensor],
    thin: dict[str, torch.Tensor],
    *,
    layers: int,
    token_position: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(hf, out_dir / "hf_layer_trace.pt")
    torch.save(thin, out_dir / "thin_layer_trace.pt")
    component_order = [
        "layer_input",
        "post_input_rmsnorm",
        "q_projection",
        "k_projection",
        "v_projection",
        "attention_output_before_o_proj",
        "o_proj_output",
        "post_attention_residual",
        "post_attention_rmsnorm",
        "gate_projection",
        "up_projection",
        "gated_activation",
        "down_projection",
        "final_residual_after_mlp",
    ]
    ordered_keys = [
        f"layer_{layer}.{component}"
        for layer in range(layers)
        for component in component_order
    ] + ["final_norm", "final_logits"]
    rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        left = hf.get(key)
        right = thin.get(key)
        layer = None
        component = key
        if key.startswith("layer_"):
            prefix, component = key.split(".", 1)
            layer = int(prefix.removeprefix("layer_"))
        if left is None or right is None:
            rows.append(
                {
                    "key": key,
                    "layer": layer,
                    "component": component,
                    "supported": False,
                    "reason": (
                        "missing from HF trace"
                        if left is None
                        else "missing from ThinTensor trace"
                    ),
                    "threshold_pass": False,
                }
            )
            continue
        left_float = left.reshape(-1).float()
        right_float = right.reshape(-1).float()
        if left_float.numel() != right_float.numel():
            rows.append(
                {
                    "key": key,
                    "layer": layer,
                    "component": component,
                    "supported": True,
                    "hf_shape": list(left.shape),
                    "thin_shape": list(right.shape),
                    "reason": "shape/element count mismatch",
                    "threshold_pass": False,
                }
            )
            continue
        difference = (left_float - right_float).abs()
        cosine = float(F.cosine_similarity(left_float, right_float, dim=0))
        rows.append(
            {
                "key": key,
                "layer": layer,
                "component": component,
                "supported": True,
                "hf_shape": list(left.shape),
                "thin_shape": list(right.shape),
                "hf_dtype": str(left.dtype).replace("torch.", ""),
                "thin_dtype": str(right.dtype).replace("torch.", ""),
                "compared_in": "fp32",
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "cosine_similarity": cosine,
                "threshold": {"cosine_similarity": 0.999},
                "threshold_pass": cosine >= 0.999,
            }
        )
    first_failure = next(
        (row for row in rows if not row.get("threshold_pass", False)),
        None,
    )
    layer_summaries = []
    for layer in range(layers):
        layer_rows = [row for row in rows if row.get("layer") == layer]
        input_row = next(
            row
            for row in layer_rows
            if row["component"] == "layer_input"
        )
        output_row = next(
            row
            for row in layer_rows
            if row["component"] == "final_residual_after_mlp"
        )
        layer_summaries.append(
            {
                "layer": layer,
                "layer_input_cosine": input_row.get("cosine_similarity"),
                "layer_output_cosine": output_row.get("cosine_similarity"),
                "first_failing_component": next(
                    (
                        row["component"]
                        for row in layer_rows
                        if not row.get("threshold_pass", False)
                    ),
                    None,
                ),
            }
        )
    return {
        "token_position": token_position,
        "threshold": {"cosine_similarity": 0.999},
        "first_failing_layer": (
            first_failure.get("layer") if first_failure is not None else None
        ),
        "first_failing_component": (
            first_failure.get("component")
            if first_failure is not None
            else None
        ),
        "first_failure": first_failure,
        "layer_summaries": layer_summaries,
        "components": rows,
        "hf_dump": str(out_dir / "hf_layer_trace.pt"),
        "thin_dump": str(out_dir / "thin_layer_trace.pt"),
    }


def write_attention_parity_report(
    report: dict[str, Any] | None,
    *,
    first_divergence_report: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {}
    if report is not None:
        payload["selected_layer"] = report
    if first_divergence_report is not None:
        payload["first_divergence"] = first_divergence_report
    Path("attention_parity_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Attention parity report", ""]
    if first_divergence_report is not None:
        failure = first_divergence_report["first_failure"]
        lines.extend(
            [
                "## First-divergence trace",
                "",
                f"- Token position: `{first_divergence_report['token_position']}`",
                f"- Threshold: cosine `>= {first_divergence_report['threshold']['cosine_similarity']}`",
                f"- First failing layer: `{first_divergence_report['first_failing_layer']}`",
                f"- First failing component: `{first_divergence_report['first_failing_component']}`",
            ]
        )
        if failure is not None:
            lines.extend(
                [
                    f"- First-failure cosine: `{failure.get('cosine_similarity')}`",
                    f"- First-failure mean absolute error: `{failure.get('mean_abs_error')}`",
                    f"- First-failure max absolute error: `{failure.get('max_abs_error')}`",
                ]
            )
        lines.extend(
            [
                "",
                "| layer | input cosine | output cosine | first failing component |",
                "|---:|---:|---:|:---|",
            ]
        )
        for row in first_divergence_report["layer_summaries"]:
            lines.append(
                f"| {row['layer']} | "
                f"{row['layer_input_cosine']:.6f} | "
                f"{row['layer_output_cosine']:.6f} | "
                f"{row['first_failing_component'] or '-'} |"
            )
        lines.append("")
    if report is not None:
        lines.extend(
            [
                "## Selected-layer component audit",
                "",
                f"- Layer: `{report['layer']}`",
                f"- Token position: `{report['token_position']}`",
                f"- First failing component: `{report['first_failing_component']}`",
                "- Interpretation: attention inputs, RoPE, scores, probabilities, and "
                "value mixing are compared independently from later projection drift.",
                "- A failure first appearing at final norm/logits after earlier "
                "attention components pass is accumulated projection/quantization "
                "drift, not a causal-KV semantic mismatch.",
                "",
                "| component | HF shape | Thin shape | HF dtype | Thin dtype | max abs | mean abs | cosine | pass |",
                "|:---|:---|:---|:---|:---|---:|---:|---:|:---:|",
            ]
        )
        for row in report["components"]:
            if not row.get("supported"):
                lines.append(
                    f"| {row['component']} | unsupported | unsupported | - | - | - | - | - | False |"
                )
                continue
            lines.append(
                f"| {row['component']} | {row['hf_shape']} | {row['thin_shape']} | "
                f"{row['hf_dtype']} | {row['thin_dtype']} | "
                f"{row.get('max_abs_error', float('nan')):.6f} | "
                f"{row.get('mean_abs_error', float('nan')):.6f} | "
                f"{row.get('cosine_similarity', float('nan')):.6f} | "
                f"{row.get('threshold_pass', False)} |"
            )
    Path("attention_parity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def comparison_record(
    args: argparse.Namespace,
    prefill_len: int,
    step: int,
    hf_logits: torch.Tensor,
    thin_logits: torch.Tensor,
    hf_tokens: list[torch.Tensor],
    thin_tokens: list[torch.Tensor],
    attention_mode: str,
    kv_tokens_attended: int,
    kv_read_bytes: int,
) -> dict[str, Any]:
    difference = (hf_logits - thin_logits).abs()
    hf_values, hf_indices = torch.topk(hf_logits, 5)
    thin_values, thin_indices = torch.topk(thin_logits, 5)
    hf_top5 = top_entries(hf_values, hf_indices)
    thin_top5 = top_entries(thin_values, thin_indices)
    hf_ids = {entry["token_id"] for entry in hf_top5}
    thin_ids = {entry["token_id"] for entry in thin_top5}
    equivalent_attention = attention_mode == "causal_kv"
    reason = None
    if not equivalent_attention:
        reason = "historical KV is not read in current_only_smoke mode"
    return {
        "model_name": str(args.hf_model),
        "prompt": args.prompt,
        "prefill_length": prefill_len,
        "decode_step": step,
        "hf_top5": hf_top5,
        "thin_top5": thin_top5,
        "top1_same": hf_top5[0]["token_id"] == thin_top5[0]["token_id"],
        "top5_overlap": len(hf_ids & thin_ids) / 5.0,
        "max_abs_logit_error": float(difference.max()),
        "mean_abs_logit_error": float(difference.mean()),
        "cosine_similarity": float(
            F.cosine_similarity(hf_logits, thin_logits, dim=0)
        ),
        "generated_hf_tokens": [int(token) for token in hf_tokens],
        "generated_thin_tokens": [int(token) for token in thin_tokens],
        "generated_token_match_rate": sum(
            int(left) == int(right)
            for left, right in zip(hf_tokens, thin_tokens)
        )
        / max(1, len(hf_tokens)),
        "token_trajectory": "hf_greedy_teacher_forced",
        "attention_mode": attention_mode,
        "kv_tokens_attended": kv_tokens_attended,
        "kv_cache_read_bytes_per_token": kv_read_bytes,
        "not_hf_equivalent": not equivalent_attention,
        "reason_not_equivalent": reason,
    }


def build_summary(
    args: argparse.Namespace,
    hf_descriptor: dict[str, Any],
    thin_descriptor: dict[str, Any],
    hf_config_summary: dict[str, Any],
    thin_config_summary: dict[str, Any],
    config_mismatches: list[dict[str, Any]],
    records: list[dict[str, Any]],
    hf_attention_implementation: str,
) -> dict[str, Any]:
    attention_equivalent = all(not row["not_hf_equivalent"] for row in records)
    any_fp8 = any(
        (
            args.lm_head_fp8,
            args.mlp_fp8,
            args.down_proj_fp8,
            args.gate_up_fp8,
            args.attn_proj_fp8,
            args.qkv_fp8,
            args.o_proj_fp8,
        )
    )
    body_fp8 = any(
        (
            args.mlp_fp8,
            args.down_proj_fp8,
            args.gate_up_fp8,
            args.attn_proj_fp8,
            args.qkv_fp8,
            args.o_proj_fp8,
        )
    )
    top1_all = all(row["top1_same"] for row in records)
    top5_high = all(row["top5_overlap"] >= 0.8 for row in records)
    generated_match = all(
        row["generated_token_match_rate"] >= 0.8 for row in records
    )
    strict_cosine = all(
        row["cosine_similarity"] >= 0.999 for row in records
    )
    exact_pass = (
        attention_equivalent
        and not any_fp8
        and top1_all
        and top5_high
        and generated_match
        and strict_cosine
        and not config_mismatches
    )
    ranking_pass = (
        attention_equivalent and top1_all and top5_high and generated_match
    )
    if exact_pass:
        tier = "exact_pass"
    elif ranking_pass:
        tier = "ranking_pass"
    elif attention_equivalent and top5_high:
        tier = "experimental"
    else:
        tier = "fail"
    return {
        "model_name": args.hf_model,
        "archive": args.archive,
        "prompt": args.prompt,
        "dtype": args.dtype,
        "kernel_backend": (
            "torch" if args.exact_hf_mode else args.kernel_backend
        ),
        "weight_residency": args.weight_residency,
        "gpu_weight_budget_bytes": (
            parse_bytes(args.gpu_weight_budget)
            if args.weight_residency == "stream"
            else 0
        ),
        "cpu_weight_offload": args.cpu_weight_offload,
        "pin_cpu_weight_pages": args.pin_cpu_weight_pages,
        "lm_head_fp8": args.lm_head_fp8,
        "mlp_fp8": args.mlp_fp8,
        "down_proj_fp8": args.down_proj_fp8,
        "gate_up_fp8": args.gate_up_fp8,
        "attn_proj_fp8": args.attn_proj_fp8,
        "qkv_fp8": args.qkv_fp8,
        "o_proj_fp8": args.o_proj_fp8,
        "fp8_layers": args.fp8_layers or "all",
        "down_fp8_layers": args.down_fp8_layers or args.fp8_layers or "all",
        "qkv_fp8_layers": args.qkv_fp8_layers or args.fp8_layers or "all",
        "o_fp8_layers": args.o_fp8_layers or args.fp8_layers or "all",
        "exact_hf_mode": args.exact_hf_mode,
        "hf_reference_mode": "full_recompute_causal",
        "hf_attention_implementation": hf_attention_implementation,
        "fp8_scale_block": args.fp8_scale_block,
        "hf_descriptor": hf_descriptor,
        "thin_descriptor": thin_descriptor,
        "hf_config_summary": hf_config_summary,
        "thin_config_summary": thin_config_summary,
        "config_mismatches": config_mismatches,
        "correctness_tier": tier,
        "attention_equivalent": attention_equivalent,
        "ranking_equivalent": ranking_pass,
        "hf_equivalent": exact_pass,
        "quantized_experimental": body_fp8,
        "acceptance": {
            "attention_equivalent": attention_equivalent,
            "top1_same_all_records": top1_all,
            "top5_overlap_high_all_records": top5_high,
            "strict_cosine_all_records": strict_cosine,
            "short_generation_mostly_matches": generated_match,
            "minimum_top5_overlap": 0.8,
            "exact_minimum_cosine_similarity": 0.999,
            "minimum_generated_token_match_rate": 0.8,
        },
        "records": records,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# HF vs ThinTensor correctness report",
        "",
        f"- Model: `{summary['model_name']}`",
        f"- Archive: `{summary['archive']}`",
        f"- Dtype: `{summary['dtype']}`",
        f"- Kernel backend: `{summary['kernel_backend']}`",
        f"- Weight residency: `{summary['weight_residency']}`",
        f"- GPU weight budget: `{summary['gpu_weight_budget_bytes']}` bytes",
        f"- Prompt: `{summary['prompt']}`",
        f"- HF attention implementation: `{summary['hf_attention_implementation']}`",
        f"- HF-equivalent: `{str(summary['hf_equivalent']).lower()}`",
        f"- Correctness tier: `{summary['correctness_tier']}`",
        f"- Attention-equivalent: `{str(summary['attention_equivalent']).lower()}`",
        f"- Ranking-equivalent: `{str(summary['ranking_equivalent']).lower()}`",
        f"- Quantized experimental: `{str(summary['quantized_experimental']).lower()}`",
        f"- LM-head FP8: `{str(summary['lm_head_fp8']).lower()}`",
        f"- MLP FP8: `{str(summary['mlp_fp8']).lower()}`",
        f"- Config mismatches: `{len(summary['config_mismatches'])}`",
        "",
        "| prefill | step | top1 | top5 overlap | cosine | mean abs err | max abs err | token match | attention | KV tokens | not HF equivalent |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|:---|---:|:---:|",
    ]
    for row in summary["records"]:
        lines.append(
            f"| {row['prefill_length']} | {row['decode_step']} | "
            f"{row['top1_same']} | {row['top5_overlap']:.3f} | "
            f"{row['cosine_similarity']:.6f} | "
            f"{row['mean_abs_logit_error']:.6f} | "
            f"{row['max_abs_logit_error']:.6f} | "
            f"{row['generated_token_match_rate']:.3f} | "
            f"{row['attention_mode']} | {row['kv_tokens_attended']} | "
            f"{row['not_hf_equivalent']} |"
        )
        lines.extend(
            [
                "",
                f"HF top5 ({row['prefill_length']}/{row['decode_step']}): "
                f"`{json.dumps(row['hf_top5'])}`",
                "",
                f"ThinTensor top5 ({row['prefill_length']}/{row['decode_step']}): "
                f"`{json.dumps(row['thin_top5'])}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def top_entries(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> list[dict[str, int | float]]:
    return [
        {"token_id": int(index), "logit": float(value)}
        for value, index in zip(values, indices)
    ]


def prompt_ids(tokenizer: Any, prompt: str, length: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    if encoded.numel() == 0:
        raise ValueError("prompt tokenized to an empty sequence")
    repeats = (length + int(encoded.shape[1]) - 1) // int(encoded.shape[1])
    return encoded.repeat(1, repeats)[:, :length].contiguous()


def compare_descriptors(hf: Any, thin: Any) -> list[str]:
    fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    )
    return [
        f"{field}: hf={getattr(hf, field)} thin={getattr(thin, field)}"
        for field in fields
        if getattr(hf, field) != getattr(thin, field)
    ]


def summarize_hf_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    descriptor = descriptor_from_hf_config(config)
    rope_scaling = config.get("rope_scaling")
    rope_variant = None
    if isinstance(rope_scaling, dict):
        rope_variant = rope_scaling.get("rope_type") or rope_scaling.get("type")
    if descriptor.model_type == "smollm3":
        rope_variant = rope_variant or "smollm3_selective_rope"
    return {
        "model_type": descriptor.model_type,
        "hidden_size": descriptor.hidden_size,
        "intermediate_size": descriptor.intermediate_size,
        "num_hidden_layers": descriptor.num_hidden_layers,
        "num_attention_heads": descriptor.num_attention_heads,
        "num_key_value_heads": descriptor.num_key_value_heads,
        "head_dim": descriptor.head_dim,
        "rope_theta": descriptor.rope_theta,
        "rope_scaling": rope_scaling,
        "rms_norm_eps": descriptor.rms_norm_eps,
        "attention_bias": bool(config.get("attention_bias", False)),
        "qkv_bias": descriptor.qkv_bias,
        "tie_word_embeddings": descriptor.tie_word_embeddings,
        "hidden_act": descriptor.activation,
        "torch_dtype": str(config.get("torch_dtype", "unknown")),
        "sliding_window": config.get("sliding_window"),
        "use_sliding_window": bool(config.get("use_sliding_window", False)),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "rope_variant": rope_variant or "default",
        "no_rope_layers": list(descriptor.no_rope_layers),
    }


def summarize_thin_config(
    manifest: dict[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    model = manifest["model"]
    page_ids = {page["id"] for page in manifest.get("pages", [])}
    has_qkv_bias = any(
        f"model.layers.0.self_attn.{name}_proj.bias" in page_ids
        for name in ("q", "k", "v")
    )
    tied = model.get("tie_word_embeddings")
    if tied is None:
        tied = "lm_head.weight" not in page_ids
    rope_variant = model.get("rope_variant")
    if descriptor.model_type == "smollm3":
        rope_variant = rope_variant or "smollm3_selective_rope"
    missing = [
        field
        for field in (
            "model_type",
            "rope_scaling",
            "tie_word_embeddings",
            "activation",
            "attention_bias",
            "sliding_window",
            "use_sliding_window",
            "max_position_embeddings",
        )
        if field not in model
    ]
    return {
        "model_type": descriptor.model_type,
        "hidden_size": descriptor.hidden_size,
        "intermediate_size": descriptor.intermediate_size,
        "num_hidden_layers": descriptor.num_hidden_layers,
        "num_attention_heads": descriptor.num_attention_heads,
        "num_key_value_heads": descriptor.num_key_value_heads,
        "head_dim": descriptor.head_dim,
        "rope_theta": descriptor.rope_theta,
        "rope_scaling": model.get("rope_scaling"),
        "rms_norm_eps": descriptor.rms_norm_eps,
        "attention_bias": bool(model.get("attention_bias", has_qkv_bias)),
        "qkv_bias": bool(model.get("qkv_bias", has_qkv_bias)),
        "tie_word_embeddings": bool(tied),
        "hidden_act": str(model.get("activation", descriptor.activation)),
        "torch_dtype": str(
            model.get("source_dtype", model.get("dtype", "unknown"))
        ),
        "sliding_window": model.get("sliding_window"),
        "use_sliding_window": bool(model.get("use_sliding_window", False)),
        "max_position_embeddings": model.get("max_position_embeddings"),
        "rope_variant": rope_variant or "default",
        "no_rope_layers": list(descriptor.no_rope_layers),
        "manifest_metadata_missing": missing,
    }


def compare_config_summaries(
    hf: dict[str, Any],
    thin: dict[str, Any],
) -> list[dict[str, Any]]:
    mismatches = []
    for field, hf_value in hf.items():
        if field == "manifest_metadata_missing":
            continue
        thin_value = thin.get(field)
        if field == "torch_dtype":
            aliases = {
                "bf16": "bfloat16",
                "torch.bfloat16": "bfloat16",
                "fp16": "float16",
            }
            hf_value = aliases.get(str(hf_value).lower(), hf_value)
            thin_value = aliases.get(str(thin_value).lower(), thin_value)
        if thin_value != hf_value:
            mismatches.append(
                {"field": field, "hf": hf_value, "thin": thin_value}
            )
    return mismatches


def parse_positive_csv(value: str, name: str) -> list[int]:
    result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not result or any(number <= 0 for number in result):
        raise ValueError(f"--{name} must contain positive comma-separated integers")
    return result


def parse_dtype(value: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[value]


def parse_bytes(value: str) -> int:
    text = value.strip().upper()
    units = {
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "B": 1,
    }
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            return int(float(number) * multiplier)
    return int(text)


if __name__ == "__main__":
    main()
