#!/usr/bin/env python3
"""Compare real int8 and ThinTensor modes against HF BF16 logits."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from stress_validate_thin import (  # noqa: E402
    Case,
    build_cases,
    make_cache,
    select_token,
)
from thinruntime.gpu_runtime import (  # noqa: E402
    ThinGpuCausalLMRuntime,
    ThinGpuWeights,
)


MODE_LABELS = {
    "hf_bf16": "HF BF16",
    "hf_bnb_int8": "HF bitsandbytes int8/Q8-ish",
    "thin_bf16": "ThinTensor BF16",
    "thin_quality_fp8": "ThinTensor quality FP8",
    "thin_quality_10_26": "ThinTensor quality FP8 10:26",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="SmolLM3-3B")
    parser.add_argument("--archive", default="SmolLM3-3B.thin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--modes",
        default="hf_bf16,hf_bnb_int8,thin_bf16,thin_quality_fp8",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case names/categories; empty runs the full grid",
    )
    parser.add_argument(
        "--out",
        default="correctness_results/q8_baseline_report.md",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
    )
    cases = select_cases(build_cases(tokenizer), args.cases)
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(modes) - set(MODE_LABELS)
    if unknown:
        raise ValueError(f"unsupported modes: {sorted(unknown)}")
    if "hf_bf16" not in modes:
        modes.insert(0, "hf_bf16")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "hf_model": args.hf_model,
        "archive": args.archive,
        "reference": "HF BF16 eager cached decode",
        "case_count": len(cases),
        "case_names": [case.name for case in cases],
        "modes": [],
        "notes": [
            "bitsandbytes int8 is a Q8-ish HF baseline, not GGUF Q8_0",
            "all candidate logits are compared with HF BF16",
            "sampling trajectories use deterministic HF BF16 teacher tokens",
        ],
    }

    print("Q8 comparison: HF BF16 reference", file=sys.stderr)
    references, reference_record = run_hf_reference(
        args.hf_model,
        cases,
        device,
        dtype,
    )
    report["modes"].append(reference_record)
    checkpoint_report(report, output)

    for mode in modes:
        if mode == "hf_bf16":
            continue
        print(f"Q8 comparison: {MODE_LABELS[mode]}", file=sys.stderr)
        if mode == "hf_bnb_int8":
            record = run_hf_candidate(
                args.hf_model,
                cases,
                references,
                device,
                dtype,
            )
        else:
            record = run_thin_candidate(
                mode,
                args.archive,
                cases,
                references,
                device,
                dtype,
            )
        report["modes"].append(record)
        checkpoint_report(report, output)

    checkpoint_report(report, output)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {output} and {output.with_suffix('.json')}")


def select_cases(cases: list[Case], selector: str) -> list[Case]:
    if not selector:
        return cases
    selected = {value.strip() for value in selector.split(",") if value.strip()}
    result = [
        case
        for case in cases
        if case.name in selected or case.category in selected
    ]
    if not result:
        raise ValueError(f"--cases matched no cases: {sorted(selected)}")
    return result


def load_hf(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    int8: bool,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if int8:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not int8:
        model = model.to(device)
    model.eval()
    model.config._attn_implementation = "eager"
    return model


def run_hf_reference(
    model_path: str,
    cases: list[Case],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    model = load_hf(model_path, device, dtype, int8=False)
    references: dict[str, dict[str, Any]] = {}
    records = []
    for case in cases:
        print(f"  HF BF16 {case.name}", file=sys.stderr)
        try:
            result = run_hf_case(model, case, device, teacher_tokens=None)
            references[case.name] = result
            records.append(
                reference_case_record(case, result, device)
            )
        except torch.OutOfMemoryError as exc:
            records.append(skipped_case(case, f"CUDA OOM: {exc}"))
            recover_cuda()
    record = aggregate_mode("hf_bf16", records)
    del model
    recover_cuda()
    return references, record


def run_hf_candidate(
    model_path: str,
    cases: list[Case],
    references: dict[str, dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = load_hf(model_path, device, dtype, int8=True)
    records = []
    for case in cases:
        reference = references.get(case.name)
        if reference is None:
            records.append(skipped_case(case, "HF BF16 reference unavailable"))
            continue
        print(f"  bitsandbytes int8 {case.name}", file=sys.stderr)
        try:
            candidate = run_hf_case(
                model,
                case,
                device,
                teacher_tokens=reference["tokens"],
            )
            records.append(
                candidate_case_record(
                    case,
                    reference,
                    candidate,
                    device,
                )
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            records.append(skipped_case(case, f"CUDA OOM: {exc}"))
            recover_cuda()
    record = aggregate_mode("hf_bnb_int8", records)
    record["quantization"] = {
        "implementation": "transformers BitsAndBytesConfig(load_in_8bit=True)",
        "label": "HF bitsandbytes int8/Q8-ish",
        "not_equivalent_to_gguf_q8_0": True,
    }
    del model
    recover_cuda()
    return record


def run_hf_case(
    model: torch.nn.Module,
    case: Case,
    device: torch.device,
    teacher_tokens: list[int] | None,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    input_ids = case.input_ids.to(device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        past = output.past_key_values
        logits = output.logits[0, -1].float()
        captured: dict[int, torch.Tensor] = {}
        tokens: list[int] = []
        for step in range(1, case.steps + 1):
            if step in case.checkpoints:
                captured[step] = logits.detach().cpu()
            token = (
                teacher_tokens[step - 1]
                if teacher_tokens is not None
                else select_token(logits.detach().cpu(), case.sampling, step)
            )
            tokens.append(int(token))
            if step == case.steps:
                break
            token_gpu = torch.tensor([[token]], device=device, dtype=torch.long)
            output = model(
                input_ids=token_gpu,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor(
                    [int(input_ids.shape[1]) + step - 1],
                    device=device,
                ),
            )
            past = output.past_key_values
            logits = output.logits[0, -1].float()
    return {
        "logits": captured,
        "tokens": tokens,
        "gpu_peak_allocated_bytes": peak_allocated(device),
    }


def run_thin_candidate(
    mode: str,
    archive: str,
    cases: list[Case],
    references: dict[str, dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    weights = ThinGpuWeights(archive, device=str(device), dtype=dtype)
    model_config = weights.manifest["model"]
    runtime_kwargs: dict[str, Any] = {
        "kernel_backend": "triton",
        "attention_mode": "causal_kv",
        "lm_head_backend": "triton",
    }
    if mode in {"thin_quality_fp8", "thin_quality_10_26"}:
        runtime_kwargs.update(
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec=(
                "10:26" if mode == "thin_quality_10_26" else "8:28"
            ),
        )
    runtime = ThinGpuCausalLMRuntime(
        weights,
        kv_cache=make_cache(model_config, device, dtype),
        **runtime_kwargs,
    )
    records = []
    for case in cases:
        reference = references.get(case.name)
        if reference is None:
            records.append(skipped_case(case, "HF BF16 reference unavailable"))
            continue
        print(f"  {MODE_LABELS[mode]} {case.name}", file=sys.stderr)
        try:
            candidate = run_thin_case(
                runtime,
                model_config,
                case,
                reference["tokens"],
                device,
                dtype,
            )
            records.append(
                candidate_case_record(
                    case,
                    reference,
                    candidate,
                    device,
                )
            )
        except torch.OutOfMemoryError as exc:
            records.append(skipped_case(case, f"CUDA OOM: {exc}"))
            recover_cuda()
    record = aggregate_mode(mode, records)
    record["resident_weight_bytes"] = (
        weights.stats.unique_gpu_weight_bytes
        - runtime.body_fp8_memory_saved_bytes
        + runtime.lm_head_net_extra_bytes
    )
    record["saved_weight_bytes"] = runtime.body_fp8_memory_saved_bytes
    record["config"] = {
        "kernel_backend": "triton",
        "attention_mode": "causal_kv",
        "precision": (
            "BF16" if mode == "thin_bf16" else "selective scaled FP8"
        ),
        "gate_up_fp8_layers": (
            "none" if mode == "thin_bf16" else "all"
        ),
        "down_fp8_layers": (
            "none"
            if mode == "thin_bf16"
            else ("10:26" if mode == "thin_quality_10_26" else "8:28")
        ),
        "o_proj": "BF16",
        "lm_head": "BF16",
    }
    weights.close()
    del runtime
    del weights
    recover_cuda()
    return record


def run_thin_case(
    runtime: ThinGpuCausalLMRuntime,
    model: dict[str, Any],
    case: Case,
    teacher_tokens: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    runtime.kv_cache = make_cache(model, device, dtype)
    runtime._rope_cos_sin_cache.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        hidden = None
        for position in range(int(case.input_ids.shape[1])):
            hidden = runtime.forward_token(
                case.input_ids[0, position].to(device),
                token_index=position,
            )
        assert hidden is not None
        captured: dict[int, torch.Tensor] = {}
        tokens: list[int] = []
        for step in range(1, case.steps + 1):
            logits = runtime.logits(hidden).float()
            if step in case.checkpoints:
                captured[step] = logits.detach().cpu()
            token = int(teacher_tokens[step - 1])
            tokens.append(token)
            if step == case.steps:
                break
            hidden = runtime.forward_token(
                torch.tensor(token, device=device, dtype=torch.long),
                token_index=int(case.input_ids.shape[1]) + step - 1,
            )
    return {
        "logits": captured,
        "tokens": tokens,
        "gpu_peak_allocated_bytes": peak_allocated(device),
    }


def reference_case_record(
    case: Case,
    result: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    comparisons = [
        distribution_metrics(logits, logits, step)
        for step, logits in sorted(result["logits"].items())
    ]
    return finish_case_record(
        case,
        comparisons,
        result["gpu_peak_allocated_bytes"],
    )


def candidate_case_record(
    case: Case,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    del device
    comparisons = [
        distribution_metrics(
            reference["logits"][step],
            candidate["logits"][step],
            step,
        )
        for step in case.checkpoints
        if step in reference["logits"] and step in candidate["logits"]
    ]
    return finish_case_record(
        case,
        comparisons,
        candidate["gpu_peak_allocated_bytes"],
    )


def distribution_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    step: int,
) -> dict[str, Any]:
    identical_storage = (
        reference.device == candidate.device
        and reference.data_ptr() == candidate.data_ptr()
        and reference.shape == candidate.shape
    )
    if identical_storage:
        return {
            "step": step,
            "cosine_similarity": 1.0,
            "cosine_loss": 0.0,
            "top1_same": True,
            "top5_overlap": 1.0,
            "jensen_shannon_divergence": 0.0,
            "jensen_shannon_distance": 0.0,
            "total_variation_distance": 0.0,
            "max_abs_logit_error": 0.0,
            "mean_abs_logit_error": 0.0,
        }
    reference = reference.float()
    candidate = candidate.float()
    difference = (reference - candidate).abs()
    ref_top = torch.topk(reference, 5).indices
    candidate_top = torch.topk(candidate, 5).indices
    ref_ids = {int(value) for value in ref_top}
    candidate_ids = {int(value) for value in candidate_top}
    ref_prob = torch.softmax(reference, dim=-1)
    candidate_prob = torch.softmax(candidate, dim=-1)
    midpoint = (ref_prob + candidate_prob) * 0.5
    epsilon = torch.finfo(torch.float32).tiny
    ref_safe = ref_prob.clamp_min(epsilon)
    candidate_safe = candidate_prob.clamp_min(epsilon)
    midpoint_safe = midpoint.clamp_min(epsilon)
    js_divergence = 0.5 * (
        (ref_safe * (ref_safe.log() - midpoint_safe.log())).sum()
        + (
            candidate_safe
            * (candidate_safe.log() - midpoint_safe.log())
        ).sum()
    )
    cosine = float(F.cosine_similarity(reference, candidate, dim=0))
    return {
        "step": step,
        "cosine_similarity": cosine,
        "cosine_loss": 1.0 - cosine,
        "top1_same": int(ref_top[0]) == int(candidate_top[0]),
        "top5_overlap": len(ref_ids & candidate_ids) / 5.0,
        "jensen_shannon_divergence": float(js_divergence),
        "jensen_shannon_distance": math.sqrt(
            max(0.0, float(js_divergence))
        ),
        "total_variation_distance": float(
            0.5 * (ref_prob - candidate_prob).abs().sum()
        ),
        "max_abs_logit_error": float(difference.max()),
        "mean_abs_logit_error": float(difference.mean()),
    }


def finish_case_record(
    case: Case,
    comparisons: list[dict[str, Any]],
    peak_bytes: int | None,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "category": case.category,
        "prompt_tokens": int(case.input_ids.shape[1]),
        "generation_steps": case.steps,
        "sampling": case.sampling.name,
        "comparisons": comparisons,
        "minimum_cosine_similarity": min(
            row["cosine_similarity"] for row in comparisons
        ),
        "maximum_cosine_loss": max(
            row["cosine_loss"] for row in comparisons
        ),
        "all_top1_same": all(row["top1_same"] for row in comparisons),
        "minimum_top5_overlap": min(
            row["top5_overlap"] for row in comparisons
        ),
        "maximum_jensen_shannon_distance": max(
            row["jensen_shannon_distance"] for row in comparisons
        ),
        "maximum_total_variation_distance": max(
            row["total_variation_distance"] for row in comparisons
        ),
        "maximum_abs_logit_error": max(
            row["max_abs_logit_error"] for row in comparisons
        ),
        "maximum_mean_abs_logit_error": max(
            row["mean_abs_logit_error"] for row in comparisons
        ),
        "gpu_peak_allocated_bytes": peak_bytes,
        "skipped": False,
    }


def skipped_case(case: Case, reason: str) -> dict[str, Any]:
    return {
        "case": case.name,
        "category": case.category,
        "prompt_tokens": int(case.input_ids.shape[1]),
        "generation_steps": case.steps,
        "skipped": True,
        "skip_reason": reason,
    }


def aggregate_mode(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in records if not row["skipped"]]
    if not completed:
        return {
            "mode": mode,
            "label": MODE_LABELS[mode],
            "available": False,
            "records": records,
        }
    return {
        "mode": mode,
        "label": MODE_LABELS[mode],
        "available": True,
        "minimum_cosine_similarity": min(
            row["minimum_cosine_similarity"] for row in completed
        ),
        "maximum_cosine_loss": max(
            row["maximum_cosine_loss"] for row in completed
        ),
        "all_top1_same": all(
            row["all_top1_same"] for row in completed
        ),
        "minimum_top5_overlap": min(
            row["minimum_top5_overlap"] for row in completed
        ),
        "maximum_jensen_shannon_distance": max(
            row["maximum_jensen_shannon_distance"] for row in completed
        ),
        "maximum_total_variation_distance": max(
            row["maximum_total_variation_distance"] for row in completed
        ),
        "maximum_abs_logit_error": max(
            row["maximum_abs_logit_error"] for row in completed
        ),
        "maximum_mean_abs_logit_error": max(
            row["maximum_mean_abs_logit_error"] for row in completed
        ),
        "maximum_gpu_peak_allocated_bytes": max(
            row["gpu_peak_allocated_bytes"] or 0 for row in completed
        ),
        "completed_cases": len(completed),
        "skipped_cases": len(records) - len(completed),
        "records": records,
    }


def checkpoint_report(report: dict[str, Any], output: Path) -> None:
    output.write_text(render_markdown(report), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SmolLM3 Q8 quality baseline",
        "",
        f"- HF model: `{report['hf_model']}`",
        f"- Archive: `{report['archive']}`",
        f"- Reference: `{report['reference']}`",
        f"- Stress cases: `{report['case_count']}`",
        "",
        "| Mode | Min cosine | Max cosine loss | Top1 all pass | Min top5 | Max JS distance | Max TV | Peak VRAM | Cases |",
        "|:---|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for mode in report["modes"]:
        if not mode.get("available"):
            lines.append(
                f"| {mode['label']} | - | - | - | - | - | - | - | unavailable |"
            )
            continue
        lines.append(
            f"| {mode['label']} | {mode['minimum_cosine_similarity']:.6f} | "
            f"{mode['maximum_cosine_loss']:.6f} | {mode['all_top1_same']} | "
            f"{mode['minimum_top5_overlap']:.3f} | "
            f"{mode['maximum_jensen_shannon_distance']:.6f} | "
            f"{mode['maximum_total_variation_distance']:.6f} | "
            f"{mode['maximum_gpu_peak_allocated_bytes']} | "
            f"{mode['completed_cases']} |"
        )
    lines.extend(
        [
            "",
            "HF bitsandbytes int8 is a Q8-ish quality baseline. It is not "
            "llama.cpp GGUF Q8_0.",
            "",
        ]
    )
    for mode in report["modes"]:
        lines.extend([f"## {mode['label']}", ""])
        for row in mode.get("records", []):
            if row["skipped"]:
                lines.append(
                    f"- `{row['case']}` skipped: {row['skip_reason']}"
                )
            else:
                lines.append(
                    f"- `{row['case']}`: cosine "
                    f"{row['minimum_cosine_similarity']:.6f}, "
                    f"top1={row['all_top1_same']}, "
                    f"top5={row['minimum_top5_overlap']:.3f}, "
                    f"JS distance={row['maximum_jensen_shannon_distance']:.6f}, "
                    f"TV={row['maximum_total_variation_distance']:.6f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def recover_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def peak_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


if __name__ == "__main__":
    main()
