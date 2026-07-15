#!/usr/bin/env python3
"""Compare one ThinTensor runtime mode against an exact ThinTensor reference."""

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
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.gpu_runtime import PagedKVCache, ThinGpuPagePool, ThinGpuQwenRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--trajectory",
        help="Fixed token trajectory JSON from dump_llamacpp_full_logits.py",
    )
    parser.add_argument(
        "--reference-logits-npz",
        help="Use saved step_<N> full-vocabulary reference vectors instead of rerunning BF16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--prefill-lens", default="1,8")
    parser.add_argument("--steps", default="1,4")
    parser.add_argument("--gpu-weight-budget", required=True)
    parser.add_argument(
        "--reference-gpu-weight-budget",
        help="Optional lower residency budget for the exact BF16 reference",
    )
    parser.add_argument("--kv-block-size", type=int, default=512)
    parser.add_argument("--kernel-backend", default="triton-matvec")
    parser.add_argument("--attention-backend", default="torch")
    parser.add_argument("--lm-head-backend", default="triton")
    parser.add_argument("--lm-head-fp8", action="store_true")
    parser.add_argument("--lm-head-int4-group-size", type=int, default=0)
    parser.add_argument("--keep-bf16-lm-head", action="store_true")
    parser.add_argument("--mlp-fp8", action="store_true")
    parser.add_argument("--gate-up-fp8", action="store_true")
    parser.add_argument("--down-proj-fp8", action="store_true")
    parser.add_argument("--qkv-fp8", action="store_true")
    parser.add_argument("--o-proj-fp8", action="store_true")
    parser.add_argument("--embed-fp8", action="store_true")
    parser.add_argument("--cpu-embed", action="store_true")
    parser.add_argument("--dense-int4", action="store_true")
    parser.add_argument("--dense-int4-layers")
    parser.add_argument("--dense-int4-group-size", type=int, default=128)
    parser.add_argument("--dense-int4-cpu-layers")
    parser.add_argument("--dense-int4-calibration-json")
    parser.add_argument("--dense-int4-calibration-npz")
    parser.add_argument("--dense-int4-calibration-covariance-npz")
    parser.add_argument("--dense-int4-gpu-ops")
    parser.add_argument("--dense-int4-gpu-suffixes")
    parser.add_argument("--dense-lowbit-bits", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--dense-lowbit-layers")
    parser.add_argument("--dense-lowbit-suffixes")
    parser.add_argument("--mxfp4-gate-up-layers")
    parser.add_argument("--mxfp4-down-layers")
    parser.add_argument("--mxfp4-qkv-layers")
    parser.add_argument("--mxfp4-o-layers")
    parser.add_argument("--moe-top-k-limit", type=int, default=0)
    parser.add_argument("--moe-top-k-no-renorm", action="store_true")
    parser.add_argument("--mxfp4-selected-tensorcore", action="store_true")
    parser.add_argument("--fp8-layers")
    parser.add_argument("--down-fp8-layers")
    parser.add_argument("--qkv-fp8-layers")
    parser.add_argument("--o-fp8-layers")
    parser.add_argument("--packed-expert-q2-layers")
    parser.add_argument("--packed-expert-q1-layers")
    parser.add_argument("--fused-scaled-mlp", action="store_true")
    parser.add_argument("--fused-residual-norm", action="store_true")
    parser.add_argument("--fused-rope", action="store_true")
    parser.add_argument("--no-pinned-staging", action="store_true")
    parser.add_argument("--out", default="correctness_results/thin_modes/latest.md")
    parser.add_argument(
        "--save-logits-npz",
        help="Persist reference and candidate full-vocabulary vectors",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dense_int4_gpu_ops:
        os.environ["THINTENSOR_DENSE_INT4_GPU_OPS"] = args.dense_int4_gpu_ops
    if args.dense_int4_gpu_suffixes:
        os.environ["THINTENSOR_DENSE_INT4_GPU_SUFFIXES"] = (
            args.dense_int4_gpu_suffixes
        )
    if args.dense_lowbit_suffixes:
        os.environ["THINTENSOR_DENSE_LOWBIT_SUFFIXES"] = args.dense_lowbit_suffixes
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    prefill_lens = parse_positive_csv(args.prefill_lens, "prefill-lens")
    requested_steps = parse_positive_csv(args.steps, "steps")
    max_steps = max(requested_steps)
    budget = parse_bytes(args.gpu_weight_budget)
    if budget <= 0:
        raise ValueError("--gpu-weight-budget must be positive")
    os.environ["THINTENSOR_PINNED_STAGING"] = (
        "0" if args.no_pinned_staging else "1"
    )

    fixed_teacher_tokens: list[torch.Tensor] | None = None
    if args.trajectory:
        trajectory = json.loads(Path(args.trajectory).read_text(encoding="utf-8"))
        input_ids = torch.tensor(
            [trajectory["prompt_token_ids"]], dtype=torch.long
        )
        inputs = {int(input_ids.shape[1]): input_ids}
        fixed_teacher_tokens = [
            torch.tensor(int(value), dtype=torch.long)
            for value in trajectory["teacher_token_ids"]
        ]
        requested_steps = [int(value) for value in trajectory["steps"]]
        max_steps = max(requested_steps)
    else:
        if not args.tokenizer:
            raise ValueError("--tokenizer is required without --trajectory")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
        inputs = {
            length: prompt_ids(tokenizer, args.prompt, length)
            for length in prefill_lens
        }

    records: list[dict[str, Any]] = []
    saved_logits: dict[str, Any] = {}
    saved_reference = None
    if args.reference_logits_npz:
        archive = np.load(args.reference_logits_npz)
        saved_reference = {
            step: torch.from_numpy(np.asarray(archive[f"step_{step}"])).float()
            for step in requested_steps
        }
    for prefill_len, input_ids in inputs.items():
        if saved_reference is None:
            print(f"reference prefill={prefill_len}", file=sys.stderr)
            reference = run_mode(
                args=args,
                input_ids=input_ids,
                teacher_tokens=fixed_teacher_tokens,
                requested_steps=requested_steps,
                max_steps=max_steps,
                exact=True,
                dtype=dtype,
                device=device,
                budget=(
                    parse_bytes(args.reference_gpu_weight_budget)
                    if args.reference_gpu_weight_budget else budget
                ),
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"saved reference prefill={prefill_len}", file=sys.stderr)
            reference = {
                "logits": saved_reference,
                "tokens": [],
                "kv_tokens_attended": {},
            }

        print(f"candidate prefill={prefill_len}", file=sys.stderr)
        candidate = run_mode(
            args=args,
            input_ids=input_ids,
            teacher_tokens=(fixed_teacher_tokens or reference["tokens"]),
            requested_steps=requested_steps,
            max_steps=max_steps,
            exact=False,
            dtype=dtype,
            device=device,
            budget=budget,
        )
        records.extend(
            comparison_record(
                args=args,
                prefill_len=prefill_len,
                step=step,
                reference=reference,
                candidate=candidate,
            )
            for step in requested_steps
        )
        for step in requested_steps:
            saved_logits[f"reference_step_{step}"] = (
                reference["logits"][step].numpy()
            )
            saved_logits[f"candidate_step_{step}"] = (
                candidate["logits"][step].numpy()
            )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = build_summary(args, records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(summary), encoding="utf-8")
    out_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.save_logits_npz:
        logits_path = Path(args.save_logits_npz)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(logits_path, **saved_logits)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"wrote {out_path} and {out_path.with_suffix('.json')}")


def run_mode(
    *,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    teacher_tokens: list[torch.Tensor] | None,
    requested_steps: list[int],
    max_steps: int,
    exact: bool,
    dtype: torch.dtype,
    device: torch.device,
    budget: int,
) -> dict[str, Any]:
    os.environ["THINTENSOR_MXFP4_SELECTED_TENSORCORE"] = (
        "1" if args.mxfp4_selected_tensorcore and not exact else "0"
    )
    weights = ThinGpuPagePool(
        args.archive,
        device=args.device,
        dtype=dtype,
        vram_budget_bytes=budget,
        prefetch_distance=0,
        cpu_offload=False,
        pin_cpu_pages=False,
        gate_up_fp8=False if exact else args.gate_up_fp8 or args.mlp_fp8,
        down_proj_fp8=False if exact else args.down_proj_fp8 or args.mlp_fp8,
        qkv_fp8=False if exact else args.qkv_fp8,
        o_proj_fp8=False if exact else args.o_proj_fp8,
        embed_fp8=False if exact else args.embed_fp8,
        cpu_embed=args.cpu_embed,
        fp8_layer_spec=None if exact else args.fp8_layers,
        down_fp8_layer_spec=None if exact else args.down_fp8_layers,
        qkv_fp8_layer_spec=None if exact else args.qkv_fp8_layers,
        o_fp8_layer_spec=None if exact else args.o_fp8_layers,
        lm_head_fp8=False if exact else args.lm_head_fp8,
        lm_head_int4_group_size=(
            0 if exact else args.lm_head_int4_group_size
        ),
        dense_int4=False if exact else args.dense_int4,
        dense_int4_layer_spec=(
            None if exact else args.dense_int4_layers
        ),
        dense_int4_group_size=args.dense_int4_group_size,
        dense_int4_cpu_layer_spec=(
            None if exact else args.dense_int4_cpu_layers
        ),
        dense_int4_calibration_json=(
            None if exact else args.dense_int4_calibration_json
        ),
        dense_int4_calibration_npz=(
            None if exact else args.dense_int4_calibration_npz
        ),
        dense_int4_calibration_covariance_npz=(
            None if exact else args.dense_int4_calibration_covariance_npz
        ),
        dense_lowbit_bits=0 if exact else args.dense_lowbit_bits,
        dense_lowbit_layer_spec=(
            None if exact else args.dense_lowbit_layers
        ),
        mxfp4_gate_up_layer_spec=(
            None if exact else args.mxfp4_gate_up_layers
        ),
        mxfp4_down_layer_spec=(
            None if exact else args.mxfp4_down_layers
        ),
        mxfp4_qkv_layer_spec=(
            None if exact else args.mxfp4_qkv_layers
        ),
        mxfp4_o_layer_spec=(
            None if exact else args.mxfp4_o_layers
        ),
        packed_expert_q2_layer_spec=(
            None if exact else args.packed_expert_q2_layers
        ),
        packed_expert_q1_layer_spec=(
            None if exact else args.packed_expert_q1_layers
        ),
    )
    weights.warm_start()
    try:
        model = weights.manifest["model"]
        cache = PagedKVCache(
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
            layout="head_token_interleaved",
            block_size=args.kv_block_size,
            residency="gpu_full",
            gpu_recent_tokens=256,
            prefetch_pages=0,
        )
        runtime = ThinGpuQwenRuntime(
            weights,
            kv_cache=cache,
            prefetch_distance=0,
            evict_completed_layers=True,
            kernel_backend=args.kernel_backend,
            attention_mode="causal_kv",
            attention_backend=args.attention_backend,
            lm_head_backend=args.lm_head_backend,
            lm_head_fp8=False if exact else args.lm_head_fp8,
            keep_bf16_lm_head=False if exact else args.keep_bf16_lm_head,
            lm_head_int4_group_size=(
                0 if exact else args.lm_head_int4_group_size
            ),
            moe_top_k_limit=0 if exact else args.moe_top_k_limit,
            moe_top_k_renorm=True if exact else not args.moe_top_k_no_renorm,
            gate_up_fp8=False if exact else args.gate_up_fp8 or args.mlp_fp8,
            down_proj_fp8=False if exact else args.down_proj_fp8 or args.mlp_fp8,
            qkv_fp8=False if exact else args.qkv_fp8,
            o_proj_fp8=False if exact else args.o_proj_fp8,
            fp8_layer_spec=None if exact else args.fp8_layers,
            down_fp8_layer_spec=None if exact else args.down_fp8_layers,
            qkv_fp8_layer_spec=None if exact else args.qkv_fp8_layers,
            o_fp8_layer_spec=None if exact else args.o_fp8_layers,
            fused_scaled_mlp=False if exact else args.fused_scaled_mlp,
            fused_residual_norm=False if exact else args.fused_residual_norm,
            fused_rope=False if exact else args.fused_rope,
            mxfp4_gate_up_layers=(
                None if exact else args.mxfp4_gate_up_layers
            ),
            mxfp4_down_layers=(
                None if exact else args.mxfp4_down_layers
            ),
            mxfp4_qkv_layers=(
                None if exact else args.mxfp4_qkv_layers
            ),
            mxfp4_o_layers=(
                None if exact else args.mxfp4_o_layers
            ),
        )
        hidden = None
        with torch.inference_mode():
            for position in range(int(input_ids.shape[1])):
                hidden = runtime.forward_token(
                    input_ids[0, position].to(device),
                    token_index=position,
                )
            assert hidden is not None
            runtime.begin_decode(int(input_ids.shape[1]))
            captured: dict[int, torch.Tensor] = {}
            tokens: list[torch.Tensor] = []
            kv_tokens: dict[int, int] = {}
            for step in range(1, max_steps + 1):
                logits = runtime.logits(hidden).float().detach().cpu()
                greedy = torch.argmax(logits).detach().cpu()
                tokens.append(greedy)
                if step in requested_steps:
                    captured[step] = logits
                    kv_tokens[step] = runtime._last_kv_tokens_attended
                if step < max_steps:
                    feed = (
                        teacher_tokens[step - 1]
                        if teacher_tokens is not None
                        else greedy
                    )
                    hidden = runtime.forward_token(
                        feed.to(device=device, dtype=torch.long),
                        token_index=int(input_ids.shape[1]) + step - 1,
                    )
        return {
            "logits": captured,
            "tokens": tokens,
            "kv_tokens_attended": kv_tokens,
        }
    finally:
        weights.close()
        if "runtime" in locals():
            del runtime
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def comparison_record(
    *,
    args: argparse.Namespace,
    prefill_len: int,
    step: int,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    ref_logits = reference["logits"][step]
    cand_logits = candidate["logits"][step]
    difference = (ref_logits - cand_logits).abs()
    ref_values, ref_indices = torch.topk(ref_logits, 5)
    cand_values, cand_indices = torch.topk(cand_logits, 5)
    ref_top5 = top_entries(ref_values, ref_indices)
    cand_top5 = top_entries(cand_values, cand_indices)
    ref_ids = {row["token_id"] for row in ref_top5}
    cand_ids = {row["token_id"] for row in cand_top5}
    return {
        "archive": args.archive,
        "prompt": args.prompt,
        "prefill_length": prefill_len,
        "decode_step": step,
        "reference_top5": ref_top5,
        "candidate_top5": cand_top5,
        "top1_same": ref_top5[0]["token_id"] == cand_top5[0]["token_id"],
        "top5_overlap": len(ref_ids & cand_ids) / 5.0,
        "cosine_similarity": float(F.cosine_similarity(ref_logits, cand_logits, dim=0)),
        "mean_abs_logit_error": float(difference.mean()),
        "max_abs_logit_error": float(difference.max()),
        "reference_tokens": [int(token) for token in reference["tokens"]],
        "candidate_tokens": [int(token) for token in candidate["tokens"]],
        "generated_token_match_rate": (
            sum(
                int(left) == int(right)
                for left, right in zip(reference["tokens"], candidate["tokens"])
            )
            / len(reference["tokens"])
            if reference["tokens"]
            else None
        ),
        "kv_tokens_attended": candidate["kv_tokens_attended"][step],
        "kv_tokens_expected": prefill_len + step - 1,
    }


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "archive": args.archive,
        "tokenizer": args.tokenizer,
        "trajectory": args.trajectory,
        "reference_logits_npz": args.reference_logits_npz,
        "reference": (
            f"saved full-vocabulary vectors: {args.reference_logits_npz}"
            if args.reference_logits_npz
            else "ThinTensor BF16 streamed weights"
        ),
        "candidate": "ThinTensor selected runtime flags",
        "candidate_flags": {
            "mlp_fp8": args.mlp_fp8,
            "gate_up_fp8": args.gate_up_fp8,
            "down_proj_fp8": args.down_proj_fp8,
            "qkv_fp8": args.qkv_fp8,
            "o_proj_fp8": args.o_proj_fp8,
            "embed_fp8": args.embed_fp8,
            "cpu_embed": args.cpu_embed,
            "moe_top_k_limit": args.moe_top_k_limit,
            "moe_top_k_renorm": not args.moe_top_k_no_renorm,
            "mxfp4_selected_tensorcore": args.mxfp4_selected_tensorcore,
            "lm_head_int4_group_size": args.lm_head_int4_group_size,
            "dense_int4": args.dense_int4,
            "dense_int4_layers": args.dense_int4_layers,
            "dense_int4_group_size": args.dense_int4_group_size,
            "dense_int4_cpu_layers": args.dense_int4_cpu_layers,
            "dense_int4_calibration_json": args.dense_int4_calibration_json,
            "dense_int4_calibration_npz": args.dense_int4_calibration_npz,
            "dense_int4_calibration_covariance_npz": (
                args.dense_int4_calibration_covariance_npz
            ),
            "dense_int4_gpu_ops": args.dense_int4_gpu_ops,
            "dense_int4_gpu_suffixes": args.dense_int4_gpu_suffixes,
            "dense_lowbit_bits": args.dense_lowbit_bits,
            "dense_lowbit_layers": args.dense_lowbit_layers,
            "dense_lowbit_suffixes": args.dense_lowbit_suffixes,
            "mxfp4_gate_up_layers": args.mxfp4_gate_up_layers,
            "mxfp4_down_layers": args.mxfp4_down_layers,
            "mxfp4_qkv_layers": args.mxfp4_qkv_layers,
            "mxfp4_o_layers": args.mxfp4_o_layers,
            "fp8_layers": args.fp8_layers,
            "down_fp8_layers": args.down_fp8_layers,
            "qkv_fp8_layers": args.qkv_fp8_layers,
            "o_fp8_layers": args.o_fp8_layers,
            "packed_expert_q2_layers": args.packed_expert_q2_layers,
            "packed_expert_q1_layers": args.packed_expert_q1_layers,
            "fused_scaled_mlp": args.fused_scaled_mlp,
            "fused_residual_norm": args.fused_residual_norm,
            "fused_rope": args.fused_rope,
            "pinned_staging": not args.no_pinned_staging,
        },
        "record_count": len(records),
        "minimum_cosine_similarity": min(row["cosine_similarity"] for row in records),
        "minimum_top5_overlap": min(row["top5_overlap"] for row in records),
        "top1_same_all_records": all(row["top1_same"] for row in records),
        "minimum_generated_token_match_rate": min(
            (
                row["generated_token_match_rate"]
                for row in records
                if row["generated_token_match_rate"] is not None
            ),
            default=None,
        ),
        "records": records,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ThinTensor mode comparison",
        "",
        f"- Archive: `{summary['archive']}`",
        f"- Reference: `{summary['reference']}`",
        f"- Candidate: `{summary['candidate']}`",
        f"- Minimum cosine: `{summary['minimum_cosine_similarity']:.6f}`",
        f"- Minimum top-5 overlap: `{summary['minimum_top5_overlap']:.3f}`",
        f"- Top-1 same for all records: `{summary['top1_same_all_records']}`",
        "- Minimum generated-token match: `"
        + (
            f"{summary['minimum_generated_token_match_rate']:.3f}"
            if summary["minimum_generated_token_match_rate"] is not None
            else "not measured for saved fixed-teacher reference"
        )
        + "`",
        "",
        "| prefill | step | cosine | top1 | top5 | token match | mean abs err | max abs err | KV |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["records"]:
        lines.append(
            f"| {row['prefill_length']} | {row['decode_step']} | "
            f"{row['cosine_similarity']:.6f} | {row['top1_same']} | "
            f"{row['top5_overlap']:.3f} | "
            + (
                f"{row['generated_token_match_rate']:.3f}"
                if row["generated_token_match_rate"] is not None
                else "n/a"
            )
            + " | "
            f"{row['mean_abs_logit_error']:.6f} | "
            f"{row['max_abs_logit_error']:.6f} | "
            f"{row['kv_tokens_attended']}/{row['kv_tokens_expected']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def top_entries(values: torch.Tensor, indices: torch.Tensor) -> list[dict[str, int | float]]:
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


def parse_positive_csv(value: str, name: str) -> list[int]:
    result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not result or any(number <= 0 for number in result):
        raise ValueError(f"--{name} must contain positive comma-separated integers")
    return result


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
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


if __name__ == "__main__":
    main()
