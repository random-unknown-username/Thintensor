#!/usr/bin/env python3
"""Post-optimization behavioral stress validation for ThinTensor modes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.gpu_runtime import PagedKVCache, ThinGpuQwenRuntime, ThinGpuWeights


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 1234

    @property
    def name(self) -> str:
        if self.temperature <= 0:
            return "greedy"
        return (
            f"temp={self.temperature:g},top_p={self.top_p:g},"
            f"top_k={self.top_k}"
        )


@dataclass
class Case:
    name: str
    input_ids: torch.Tensor
    steps: int
    checkpoints: tuple[int, ...]
    sampling: Sampling
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
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
        "--modes",
        default="bf16_exact,retained_fp8",
        help=(
            "Comma-separated subset of bf16_exact,bf16_triton,"
            "bf16_triton_guarded,gate_up_fp8,"
            "hybrid_mlp_fp8,hybrid_mlp_8_28,hybrid_mlp_10_26,"
            "retained_fp8,quality_8_28,quality_10_26,"
            "quality_8_28_head8_block128,"
            "quality_8_28_head8_topk_guard,"
            "quality_8_28_head8_topk_guard_split_k,full_mlp_fp8,"
            "quality_o_fp8_4_32,max_body_fp8_4_32"
        ),
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Optional comma-separated case names or categories to run",
    )
    parser.add_argument("--out", default="stress_validation_report.md")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    cases = build_cases(tokenizer)
    if args.cases:
        selected = {
            value.strip() for value in args.cases.split(",") if value.strip()
        }
        cases = [
            case
            for case in cases
            if case.name in selected or case.category in selected
        ]
        if not cases:
            raise ValueError(f"--cases matched no stress cases: {sorted(selected)}")

    print("loading HF stress reference", file=sys.stderr)
    if "cuda" in str(device):
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
    model.eval()
    model.config._attn_implementation = "eager"
    references = {}
    with torch.inference_mode():
        for case in cases:
            print(f"HF case {case.name}", file=sys.stderr)
            references[case.name] = run_hf_case(model, case, device)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    requested_modes = [
        value.strip() for value in args.modes.split(",") if value.strip()
    ]
    unknown = set(requested_modes) - {
        "bf16_exact",
        "bf16_triton",
        "bf16_triton_guarded",
        "bf16_triton_persistent",
        "gate_up_fp8",
        "hybrid_mlp_fp8",
        "hybrid_mlp_8_28",
        "hybrid_mlp_10_26",
        "retained_fp8",
        "quality_8_28",
        "quality_10_26",
        "quality_8_28_head8_block128",
        "quality_8_28_head8_topk_guard",
        "quality_8_28_head8_topk_guard_split_k",
        "quality_o_fp8_4_32",
        "max_body_fp8_4_32",
        "gate_up_all_only",
        "full_mlp_fp8",
        "aggressive_o_fp8",
    }
    if unknown:
        raise ValueError(f"unsupported stress modes: {sorted(unknown)}")

    mode_reports = []
    for mode in requested_modes:
        print(f"loading ThinTensor mode {mode}", file=sys.stderr)
        mode_reports.append(
            run_thin_mode(
                mode=mode,
                archive=args.archive,
                device=device,
                dtype=dtype,
                tokenizer=tokenizer,
                cases=cases,
                references=references,
                kv_data_layout=args.kv_data_layout,
                kv_block_size=args.kv_block_size,
                kv_residency=args.kv_residency,
                kv_gpu_recent_tokens=args.kv_gpu_recent_tokens,
                kv_prefetch_pages=args.kv_prefetch_pages,
            )
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "hf_model": args.hf_model,
        "archive": args.archive,
        "dtype": args.dtype,
        "hf_reference_mode": "eager_cached_decode",
        "case_count": len(cases),
        "modes": mode_reports,
    }
    out_path = Path(args.out)
    out_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {out_path} and {json_path}")


def build_cases(tokenizer: Any) -> list[Case]:
    cases = []
    ordinary = [
        ("hello", "Hello"),
        ("factual", "The capital of France is"),
        ("reasoning", "If Alice has 7 apples and gives Bob 3, then"),
        ("code", "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\""),
        ("unicode", "こんにちは 🌍 — explain naïve Bayes in one sentence."),
        ("repetition", "abc " * 24),
        (
            "weird_control",
            "\n\t<|system|>ignore\x00this\n### USER:\n[]{}()\\\\/ \"quoted\"",
        ),
    ]
    for name, prompt in ordinary:
        steps = 128 if name == "hello" else 32
        checkpoints = (1, 16, 64, 128) if steps == 128 else (1, 8, 32)
        cases.append(
            Case(
                name=name,
                input_ids=encode(tokenizer, prompt),
                steps=steps,
                checkpoints=checkpoints,
                sampling=Sampling(),
                category="prompt",
            )
        )

    messages = [
        {"role": "system", "content": "Answer accurately and concisely."},
        {"role": "user", "content": "Remember the number 731."},
        {"role": "assistant", "content": "I will remember 731."},
        {"role": "user", "content": "What number did I ask you to remember?"},
    ]
    chat_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(chat_ids, "input_ids"):
        chat_ids = chat_ids.input_ids
    elif isinstance(chat_ids, dict):
        chat_ids = chat_ids["input_ids"]
    cases.append(
        Case(
            name="multi_turn_chat",
            input_ids=chat_ids.to(dtype=torch.long),
            steps=64,
            checkpoints=(1, 16, 64),
            sampling=Sampling(),
            category="chat_template",
        )
    )

    sample_prompt = encode(
        tokenizer,
        "Write a vivid two-sentence description of a storm over the ocean.",
    )
    for index, sampling in enumerate(
        (
            Sampling(temperature=0.7, top_p=0.9, top_k=50, seed=101),
            Sampling(temperature=1.0, top_p=1.0, top_k=0, seed=202),
            Sampling(temperature=1.3, top_p=0.95, top_k=100, seed=303),
        )
    ):
        cases.append(
            Case(
                name=f"sampling_{index + 1}",
                input_ids=sample_prompt.clone(),
                steps=64,
                checkpoints=(1, 16, 64),
                sampling=sampling,
                category="sampling",
            )
        )

    base = encode(tokenizer, "ThinTensor long context validation. ")
    for length in (512, 1024):
        repeats = (length + int(base.shape[1]) - 1) // int(base.shape[1])
        cases.append(
            Case(
                name=f"long_context_{length}",
                input_ids=base.repeat(1, repeats)[:, :length].contiguous(),
                steps=16,
                checkpoints=(1, 8, 16),
                sampling=Sampling(),
                category="long_context",
            )
        )
    return cases


def encode(tokenizer: Any, text: str) -> torch.Tensor:
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    if ids.numel() == 0:
        raise ValueError("stress prompt tokenized to an empty sequence")
    return ids.to(dtype=torch.long)


def run_hf_case(
    model: torch.nn.Module,
    case: Case,
    device: torch.device,
) -> dict[str, Any]:
    current = case.input_ids.to(device)
    output = model(input_ids=current, use_cache=True)
    past = output.past_key_values
    logits = output.logits[0, -1].float()
    captured = {}
    tokens = []
    for step in range(1, case.steps + 1):
        if step in case.checkpoints:
            captured[step] = logits.detach().cpu()
        token = select_token(logits.detach().cpu(), case.sampling, step)
        tokens.append(token)
        if step < case.steps:
            token_gpu = torch.tensor(
                [[token]], device=device, dtype=torch.long
            )
            output = model(
                input_ids=token_gpu,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor(
                    [int(case.input_ids.shape[1]) + step - 1],
                    device=device,
                ),
            )
            past = output.past_key_values
            logits = output.logits[0, -1].float()
    return {"logits": captured, "tokens": tokens}


def run_thin_mode(
    *,
    mode: str,
    archive: str,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer: Any,
    cases: list[Case],
    references: dict[str, dict[str, Any]],
    kv_data_layout: str,
    kv_block_size: int,
    kv_residency: str,
    kv_gpu_recent_tokens: int,
    kv_prefetch_pages: int,
) -> dict[str, Any]:
    os.environ.setdefault("THINTENSOR_DISABLE_AUTOTUNE", "0")
    weights = ThinGpuWeights(archive, device=str(device), dtype=dtype)
    model_config = weights.manifest["model"]
    initial_cache = make_cache(
        model_config,
        device,
        dtype,
        layout=kv_data_layout,
        block_size=kv_block_size,
        residency=kv_residency,
        gpu_recent_tokens=kv_gpu_recent_tokens,
        prefetch_pages=kv_prefetch_pages,
    )
    if mode == "bf16_exact":
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="torch",
            attention_mode="causal_kv",
            exact_hf_mode=True,
        )
        mode_config = {
            "kernel_backend": "torch",
            "precision": "bf16",
        }
    elif mode in {"bf16_triton", "bf16_triton_guarded", "bf16_triton_persistent"}:
        guarded = mode.endswith("_guarded")
        persistent = mode.endswith("_persistent")
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            lm_head_fp8=guarded,
            keep_bf16_lm_head=guarded,
            lm_head_topk_guard=64 if guarded else 0,
            lm_head_backend="triton",
            lm_head_argmax_mode="triton_persistent" if persistent else "torch",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "bf16",
            "lm_head_fp8_shortlist": guarded,
            "lm_head_topk_guard": 64 if guarded else 0,
            "lm_head_argmax_mode": "triton_persistent" if persistent else "torch",
        }
    elif mode in {"gate_up_fp8", "gate_up_all_only"}:
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            gate_up_fp8=True,
            lm_head_backend="triton",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
        }
    elif mode == "hybrid_mlp_fp8":
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec="6:30",
            lm_head_backend="triton",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
            "down_fp8_layers": "6:30",
        }
    elif mode in {
        "hybrid_mlp_8_28",
        "hybrid_mlp_10_26",
        "quality_8_28",
        "quality_10_26",
        "retained_fp8",
    }:
        down_layers = (
            "10:26"
            if mode in {"hybrid_mlp_10_26", "quality_10_26"}
            else "8:28"
        )
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec=down_layers,
            lm_head_backend="triton",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
            "down_fp8_layers": down_layers,
        }
    elif mode in {
        "quality_8_28_head8_block128",
        "quality_8_28_head8_topk_guard",
        "quality_8_28_head8_topk_guard_split_k",
    }:
        guarded = "topk_guard" in mode
        split_k = mode.endswith("_split_k")
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec="8:28",
            lm_head_fp8=True,
            keep_bf16_lm_head=guarded,
            lm_head_fp8_scale_block=0 if guarded else 128,
            lm_head_topk_guard=64 if guarded else 0,
            lm_head_backend="triton",
            split_k_down_proj=split_k,
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
            "down_fp8_layers": "8:28",
            "lm_head_fp8": True,
            "lm_head_fp8_scale_block": 0 if guarded else 128,
            "lm_head_topk_guard": 64 if guarded else 0,
            "split_k_down_proj": split_k,
        }
    elif mode in {"quality_o_fp8_4_32", "max_body_fp8_4_32"}:
        down_layers = (
            "4:32" if mode == "max_body_fp8_4_32" else "8:28"
        )
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec=down_layers,
            o_proj_fp8=True,
            o_fp8_layer_spec="4:32",
            lm_head_fp8=True,
            keep_bf16_lm_head=True,
            lm_head_topk_guard=64,
            lm_head_backend="triton",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
            "down_fp8_layers": down_layers,
            "o_fp8_layers": "4:32",
            "lm_head_fp8": True,
            "lm_head_topk_guard": 64,
        }
    elif mode == "full_mlp_fp8":
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            mlp_fp8=True,
            lm_head_backend="triton",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
            "down_fp8_layers": "all",
        }
    else:
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=initial_cache,
            kernel_backend="triton",
            attention_mode="causal_kv",
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec="6:30",
            o_proj_fp8=True,
            o_fp8_layer_spec="4:32",
            lm_head_backend="row_block_m2",
        )
        mode_config = {
            "kernel_backend": "triton",
            "precision": "selective_fp8",
            "gate_up_fp8_layers": "all",
            "down_fp8_layers": "6:30",
            "o_fp8_layers": "4:32",
        }

    records = []
    for case in cases:
        print(f"Thin {mode} case {case.name}", file=sys.stderr)
        cache = make_cache(
            model_config,
            device,
            dtype,
            layout=kv_data_layout,
            block_size=kv_block_size,
            residency=kv_residency,
            gpu_recent_tokens=kv_gpu_recent_tokens,
            prefetch_pages=kv_prefetch_pages,
        )
        runtime.kv_cache = cache
        runtime._rope_cos_sin_cache.clear()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        hidden = None
        with torch.inference_mode():
            for position in range(int(case.input_ids.shape[1])):
                hidden = runtime.forward_token(
                    case.input_ids[0, position].to(device),
                    token_index=position,
                )
            assert hidden is not None
            thin_tokens = []
            comparisons = []
            reference = references[case.name]
            for step in range(1, case.steps + 1):
                logits = runtime.logits(hidden).float().detach().cpu()
                thin_tokens.append(select_token(logits, case.sampling, step))
                if step in case.checkpoints:
                    comparisons.append(
                        compare_logits(
                            reference["logits"][step],
                            logits,
                            step,
                            case.sampling,
                        )
                    )
                if step < case.steps:
                    teacher = torch.tensor(
                        reference["tokens"][step - 1],
                        device=device,
                        dtype=torch.long,
                    )
                    hidden = runtime.forward_token(
                        teacher,
                        token_index=int(case.input_ids.shape[1]) + step - 1,
                    )
        telemetry = cache.telemetry()
        reference_tokens = reference["tokens"]
        teacher_forced_token_match_rate = sum(
            left == right
            for left, right in zip(reference_tokens, thin_tokens)
        ) / max(1, len(reference_tokens))
        free_running_tokens = run_thin_generation(
            runtime=runtime,
            model_config=model_config,
                case=case,
                device=device,
                dtype=dtype,
                kv_data_layout=kv_data_layout,
                kv_block_size=kv_block_size,
                kv_residency=kv_residency,
                kv_gpu_recent_tokens=kv_gpu_recent_tokens,
                kv_prefetch_pages=kv_prefetch_pages,
        )
        free_running_token_match_rate = sum(
            left == right
            for left, right in zip(reference_tokens, free_running_tokens)
        ) / max(1, len(reference_tokens))
        matched_prefix_tokens = matching_prefix_length(
            reference_tokens,
            free_running_tokens,
        )
        sampling_comparisons = [
            row
            for row in comparisons
            if "sampling_total_variation" in row
        ]
        records.append(
            {
                "case": case.name,
                "category": case.category,
                "prompt_tokens": int(case.input_ids.shape[1]),
                "steps": case.steps,
                "sampling": case.sampling.name,
                "teacher_forced_token_match_rate": (
                    teacher_forced_token_match_rate
                ),
                "free_running_token_match_rate": free_running_token_match_rate,
                "free_running_matched_prefix_tokens": matched_prefix_tokens,
                "hf_generated_token_ids": reference_tokens,
                "thin_generated_token_ids": free_running_tokens,
                "hf_generated_text": tokenizer.decode(reference_tokens),
                "thin_generated_text": tokenizer.decode(free_running_tokens),
                "comparisons": comparisons,
                "minimum_cosine_similarity": min(
                    row["cosine_similarity"] for row in comparisons
                ),
                "top1_agreement_rate": sum(
                    row["top1_same"] for row in comparisons
                )
                / len(comparisons),
                "minimum_top5_overlap": min(
                    row["top5_overlap"] for row in comparisons
                ),
                "maximum_sampling_total_variation": (
                    max(
                        row["sampling_total_variation"]
                        for row in sampling_comparisons
                    )
                    if sampling_comparisons
                    else None
                ),
                "maximum_sampling_js_divergence": (
                    max(
                        row["sampling_js_divergence"]
                        for row in sampling_comparisons
                    )
                    if sampling_comparisons
                    else None
                ),
                "kv_tokens_attended": telemetry["kv_tokens_attended"],
                "kv_bytes": telemetry["kv_bytes"],
                "kv_peak_bytes": telemetry["kv_peak_bytes"],
                "kv_cache_read_bytes_per_token": telemetry[
                    "kv_cache_read_bytes_per_token"
                ],
                "gpu_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            }
        )
    weights.close()
    return {
        "mode": mode,
        "config": mode_config,
        "minimum_cosine_similarity": min(
            row["minimum_cosine_similarity"] for row in records
        ),
        "minimum_top5_overlap": min(
            row["minimum_top5_overlap"] for row in records
        ),
        "minimum_teacher_forced_token_match_rate": min(
            row["teacher_forced_token_match_rate"] for row in records
        ),
        "minimum_free_running_token_match_rate": min(
            row["free_running_token_match_rate"] for row in records
        ),
        "maximum_sampling_total_variation": max(
            (
                row["maximum_sampling_total_variation"]
                for row in records
                if row["maximum_sampling_total_variation"] is not None
            ),
            default=None,
        ),
        "maximum_sampling_js_divergence": max(
            (
                row["maximum_sampling_js_divergence"]
                for row in records
                if row["maximum_sampling_js_divergence"] is not None
            ),
            default=None,
        ),
        "all_checkpoint_top1_same": all(
            comparison["top1_same"]
            for row in records
            for comparison in row["comparisons"]
        ),
        "maximum_gpu_peak_allocated_bytes": max(
            row["gpu_peak_allocated_bytes"] or 0 for row in records
        ),
        "records": records,
    }


def run_thin_generation(
    *,
    runtime: ThinGpuQwenRuntime,
    model_config: dict[str, Any],
    case: Case,
    device: torch.device,
    dtype: torch.dtype,
    kv_data_layout: str,
    kv_block_size: int,
    kv_residency: str,
    kv_gpu_recent_tokens: int,
    kv_prefetch_pages: int,
) -> list[int]:
    runtime.kv_cache = make_cache(
        model_config,
        device,
        dtype,
        layout=kv_data_layout,
        block_size=kv_block_size,
        residency=kv_residency,
        gpu_recent_tokens=kv_gpu_recent_tokens,
        prefetch_pages=kv_prefetch_pages,
    )
    runtime._rope_cos_sin_cache.clear()
    hidden = None
    with torch.inference_mode():
        for position in range(int(case.input_ids.shape[1])):
            hidden = runtime.forward_token(
                case.input_ids[0, position].to(device),
                token_index=position,
            )
        assert hidden is not None
        tokens = []
        for step in range(1, case.steps + 1):
            logits = runtime.logits(hidden).float().detach().cpu()
            token = select_token(logits, case.sampling, step)
            tokens.append(token)
            if step < case.steps:
                hidden = runtime.forward_token(
                    torch.tensor(token, device=device, dtype=torch.long),
                    token_index=int(case.input_ids.shape[1]) + step - 1,
                )
    return tokens


def matching_prefix_length(left: list[int], right: list[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


def make_cache(
    model: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    *,
    layout: str = "head_token_interleaved",
    block_size: int = 16,
    residency: str = "gpu_full",
    gpu_recent_tokens: int = 256,
    prefetch_pages: int = 0,
) -> PagedKVCache:
    return PagedKVCache(
        layers=int(model["layers"]),
        kv_heads=int(model["kv_heads"]),
        head_dim=int(
            model.get("head_dim")
            or int(model["hidden_size"]) // int(model["heads"])
        ),
        device=device,
        dtype=dtype,
        policy="full",
        old_codec="bf16",
        layout=layout,
        block_size=block_size,
        residency=residency,
        gpu_recent_tokens=gpu_recent_tokens,
        prefetch_pages=prefetch_pages,
    )


def select_token(logits: torch.Tensor, sampling: Sampling, step: int) -> int:
    if sampling.temperature <= 0:
        return int(torch.argmax(logits))
    probabilities = sampling_probabilities(logits, sampling)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling.seed + step)
    return int(torch.multinomial(probabilities, 1, generator=generator))


def sampling_probabilities(
    logits: torch.Tensor,
    sampling: Sampling,
) -> torch.Tensor:
    scores = logits.float().clone()
    scores.div_(sampling.temperature)
    if sampling.top_k > 0 and sampling.top_k < scores.numel():
        threshold = torch.topk(scores, sampling.top_k).values[-1]
        scores[scores < threshold] = -torch.inf
    if sampling.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = torch.cumsum(probabilities, dim=-1) > sampling.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = -torch.inf
    return torch.softmax(scores, dim=-1)


def compare_logits(
    hf_logits: torch.Tensor,
    thin_logits: torch.Tensor,
    step: int,
    sampling: Sampling,
) -> dict[str, Any]:
    difference = (hf_logits - thin_logits).abs()
    hf_top = torch.topk(hf_logits, 5).indices
    thin_top = torch.topk(thin_logits, 5).indices
    hf_ids = {int(value) for value in hf_top}
    thin_ids = {int(value) for value in thin_top}
    result = {
        "step": step,
        "top1_same": int(hf_top[0]) == int(thin_top[0]),
        "top5_overlap": len(hf_ids & thin_ids) / 5.0,
        "cosine_similarity": float(
            F.cosine_similarity(hf_logits, thin_logits, dim=0)
        ),
        "mean_abs_logit_error": float(difference.mean()),
        "max_abs_logit_error": float(difference.max()),
    }
    if sampling.temperature > 0:
        hf_probabilities = sampling_probabilities(hf_logits, sampling)
        thin_probabilities = sampling_probabilities(thin_logits, sampling)
        midpoint = (hf_probabilities + thin_probabilities) * 0.5
        epsilon = torch.finfo(torch.float32).tiny
        hf_safe = hf_probabilities.clamp_min(epsilon)
        thin_safe = thin_probabilities.clamp_min(epsilon)
        midpoint_safe = midpoint.clamp_min(epsilon)
        result["sampling_total_variation"] = float(
            0.5 * (hf_probabilities - thin_probabilities).abs().sum()
        )
        result["sampling_js_divergence"] = float(
            0.5
            * (
                (hf_safe * (hf_safe.log() - midpoint_safe.log())).sum()
                + (
                    thin_safe
                    * (thin_safe.log() - midpoint_safe.log())
                ).sum()
            )
        )
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ThinTensor stress validation",
        "",
        f"- HF model: `{report['hf_model']}`",
        f"- Archive: `{report['archive']}`",
        f"- HF reference: `{report['hf_reference_mode']}`",
        f"- Cases: `{report['case_count']}`",
        "",
    ]
    for mode in report["modes"]:
        lines.extend(
            [
                f"## {mode['mode']}",
                "",
                f"- Minimum cosine: `{mode['minimum_cosine_similarity']:.6f}`",
                f"- Minimum top-5 overlap: `{mode['minimum_top5_overlap']:.3f}`",
                f"- All checkpoint top-1 same: `{mode['all_checkpoint_top1_same']}`",
                f"- Minimum teacher-forced token match: `{mode['minimum_teacher_forced_token_match_rate']:.3f}`",
                f"- Minimum free-running token match: `{mode['minimum_free_running_token_match_rate']:.3f}`",
                f"- Maximum sampling total variation: `{format_optional(mode['maximum_sampling_total_variation'])}`",
                f"- Maximum sampling JS divergence: `{format_optional(mode['maximum_sampling_js_divergence'])}`",
                f"- Maximum peak VRAM: `{mode['maximum_gpu_peak_allocated_bytes']}` bytes",
                "",
                "| case | category | prompt tokens | steps | sampling | min cosine | top1 rate | min top5 | teacher match | free match | prefix | max TV | max JS | KV tokens | KV bytes | peak VRAM |",
                "|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mode["records"]:
            lines.append(
                f"| {row['case']} | {row['category']} | "
                f"{row['prompt_tokens']} | {row['steps']} | "
                f"{row['sampling']} | {row['minimum_cosine_similarity']:.6f} | "
                f"{row['top1_agreement_rate']:.3f} | "
                f"{row['minimum_top5_overlap']:.3f} | "
                f"{row['teacher_forced_token_match_rate']:.3f} | "
                f"{row['free_running_token_match_rate']:.3f} | "
                f"{row['free_running_matched_prefix_tokens']} | "
                f"{format_optional(row['maximum_sampling_total_variation'])} | "
                f"{format_optional(row['maximum_sampling_js_divergence'])} | "
                f"{row['kv_tokens_attended']} | {row['kv_bytes']} | "
                f"{row['gpu_peak_allocated_bytes']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
