#!/usr/bin/env python3
"""Bounded correctness-gated ThinTensor configuration optimizer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.archive import ThinArchive
from thinruntime.execution_plan import compile_execution_plan
from thinruntime.model_arch import descriptor_from_manifest


MODE_FLAGS = {
    "bf16": [],
    "head8": ["--lm-head-fp8"],
    "down_fp8": ["--down-proj-fp8"],
    "gate_up_fp8": ["--gate-up-fp8"],
    "mlp_fp8": ["--mlp-fp8"],
    "qkv_fp8": ["--qkv-fp8"],
    "o_proj_fp8": ["--o-proj-fp8"],
    "attn_proj_fp8": ["--attn-proj-fp8"],
    "quality_fp8": ["--gate-up-fp8", "--down-proj-fp8"],
    "fast_body_fp8": ["--mlp-fp8", "--o-proj-fp8"],
    "native_source": [],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--hf-model")
    parser.add_argument("--target-toks", type=float, default=0.0)
    parser.add_argument("--target-speedup", type=float, default=1.30)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default="auto")
    parser.add_argument("--allow-unsafe", default="false")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--correctness-prefill-lens", default="1,8")
    parser.add_argument("--correctness-steps", default="1,10")
    parser.add_argument(
        "--residency",
        choices=["auto", "all", "stream"],
        default="auto",
    )
    parser.add_argument("--gpu-weight-budget", default="6gb")
    parser.add_argument(
        "--out-dir",
        default=f"runs/auto_opt_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    allow_unsafe = parse_bool(args.allow_unsafe)
    archive_reader = ThinArchive(args.archive)
    descriptor = descriptor_from_manifest(archive_reader.manifest)
    execution_plan = compile_execution_plan(
        descriptor,
        archive_reader.manifest_pages,
        cuda_capability=(
            torch.cuda.get_device_capability()
            if torch.cuda.is_available()
            else None
        ),
        available_quant_kernels=(
            {"fp8_scaled_matvec", "mxfp4_matvec"}
            if torch.cuda.is_available()
            else set()
        ),
    )
    archive_weight_bytes = sum(
        int(page.get("size", 0))
        for page in archive_reader.manifest.get("pages", [])
        if page.get("kind") != "fused_logical"
    )
    archive_reader.close()
    modes = resolve_modes(args.modes, descriptor, execution_plan.schema)
    unknown = [mode for mode in modes if mode not in MODE_FLAGS]
    if unknown:
        raise SystemExit(f"unknown modes: {', '.join(unknown)}")
    args.resolved_residency = resolve_residency(
        args.residency,
        archive_weight_bytes,
        args.device,
    )
    layer_count = descriptor.num_hidden_layers

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command_log = out_dir / "commands.log"
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    no_improvement_rounds = 0
    stop_reason = "max_rounds"

    for round_number, mode in enumerate(modes[: args.max_rounds], start=1):
        flags = mode_flags(mode, layer_count)
        experiment: dict[str, Any] = {"round": round_number, "mode": mode}
        try:
            correctness = run_correctness(
                repo, out_dir, command_log, args, mode, flags
            )
            correctness_passed = bool(
                correctness
                and (
                    correctness.get("hf_equivalent")
                    or (
                        allow_unsafe
                        and correctness.get("ranking_equivalent")
                    )
                )
            )
            experiment["correctness_passed"] = correctness_passed
            experiment["correctness_tier"] = (
                correctness.get("correctness_tier")
                if correctness
                else "unavailable"
            )
            is_baseline = round_number == 1
            if (
                not correctness_passed
                and not allow_unsafe
                and not is_baseline
            ):
                experiment["rejection_reason"] = "correctness gate failed or unavailable"
                rejected.append(experiment)
                no_improvement_rounds += 1
            else:
                benchmark = run_benchmark(
                    repo, out_dir, command_log, args, mode, flags
                )
                experiment.update(
                    {
                        "tokens_per_s": benchmark["tokens_per_s"],
                        "ms_per_token": benchmark["ms_per_token"],
                        "peak_vram_bytes": benchmark.get(
                            "gpu_peak_allocated_bytes"
                        ),
                        "effective_bandwidth_gb_s": benchmark.get(
                            "effective_bandwidth_gb_s"
                        ),
                        "profile_breakdown": benchmark.get(
                            "profile_breakdown", {}
                        ),
                        "benchmark_json": str(
                            out_dir / f"round_{round_number:02d}_{mode}.benchmark.json"
                        ),
                        "residency": args.resolved_residency,
                    }
                )
                if baseline is None:
                    baseline = experiment
                improvement = (
                    float("inf")
                    if best is None
                    else benchmark["tokens_per_s"] / best["tokens_per_s"] - 1.0
                )
                experiment["improvement_over_best"] = improvement
                if best is None or improvement >= 0.02:
                    accepted.append(experiment)
                    best = experiment
                    no_improvement_rounds = 0
                else:
                    experiment["rejection_reason"] = "real decode improvement below 2%"
                    rejected.append(experiment)
                    no_improvement_rounds += 1

                speedup_over_baseline = (
                    best["tokens_per_s"] / baseline["tokens_per_s"]
                    if best and baseline
                    else 1.0
                )
                experiment["speedup_over_baseline"] = speedup_over_baseline
                absolute_reached = (
                    args.target_toks > 0
                    and best
                    and best["tokens_per_s"] >= args.target_toks
                )
                relative_reached = (
                    speedup_over_baseline >= args.target_speedup
                )
                if absolute_reached or relative_reached:
                    stop_reason = (
                        "absolute_target_reached"
                        if absolute_reached
                        else "speedup_target_reached"
                    )
                    break
        except torch_oom_or_runtime() as exc:
            experiment["rejection_reason"] = f"{type(exc).__name__}: {exc}"
            rejected.append(experiment)
            no_improvement_rounds += 1

        if no_improvement_rounds >= 3:
            stop_reason = "no_2_percent_improvement_for_3_rounds"
            break
    else:
        if len(modes) < args.max_rounds:
            stop_reason = "candidate_modes_exhausted"

    summary = {
        "archive": args.archive,
        "hf_model": args.hf_model,
        "target_tokens_per_s": args.target_toks,
        "target_speedup": args.target_speedup,
        "descriptor": descriptor.as_dict(),
        "execution_schema": execution_plan.schema,
        "residency": args.resolved_residency,
        "baseline": baseline,
        "stop_reason": stop_reason,
        "best": best,
        "accepted_configs": accepted,
        "rejected_configs": rejected,
        "safe_claims": safe_claims(best),
    }
    (out_dir / "chosen_config.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rejected_configs.json").write_text(
        json.dumps(rejected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "auto_opt_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "auto_opt_summary.md").write_text(
        render_summary(summary), encoding="utf-8"
    )
    (out_dir / "safe_claims.md").write_text(
        "# Safe claims\n\n"
        + "\n".join(f"- {claim}" for claim in summary["safe_claims"])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_correctness(
    repo: Path,
    out_dir: Path,
    log: Path,
    args: argparse.Namespace,
    mode: str,
    flags: list[str],
) -> dict[str, Any] | None:
    if not args.hf_model or not (Path(args.hf_model) / "config.json").exists():
        return None
    output = out_dir / f"{mode}.correctness.md"
    command = [
        sys.executable,
        str(repo / "scripts/compare_hf_thin_logits.py"),
        "--hf-model",
        args.hf_model,
        "--archive",
        args.archive,
        "--device",
        args.device,
        "--dtype",
        "bf16",
        "--prefill-lens",
        args.correctness_prefill_lens,
        "--steps",
        args.correctness_steps,
        "--attention-mode",
        "causal_kv",
        "--out",
        str(output),
        "--json",
        *flags,
    ]
    return run_json(command, repo, log)


def run_benchmark(
    repo: Path,
    out_dir: Path,
    log: Path,
    args: argparse.Namespace,
    mode: str,
    flags: list[str],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(repo / "scripts/thin_runtime.py"),
        "run",
        args.archive,
        "--device",
        args.device,
        "--dtype",
        "bf16",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--residency",
        args.resolved_residency,
        "--kernel-backend",
        "triton-matvec",
        "--attention-mode",
        "causal_kv",
        "--json",
        *flags,
    ]
    if args.resolved_residency == "stream":
        command.extend(
            [
                "--cpu-offload",
                "--gpu-weight-budget",
                args.gpu_weight_budget,
            ]
        )
    result = run_json(command, repo, log)
    round_number = sum(1 for _ in out_dir.glob("round_*.benchmark.json")) + 1
    (out_dir / f"round_{round_number:02d}_{mode}.benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_json(command: list[str], cwd: Path, log: Path) -> dict[str, Any]:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-4000:])
    return json.loads(process.stdout)


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# ThinTensor automatic optimization summary",
        "",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Target: `{summary['target_tokens_per_s']:.2f} tok/s`",
        f"- Target speedup: `{summary['target_speedup']:.2f}x`",
        f"- Residency: `{summary['residency']}`",
    ]
    best = summary["best"]
    if best:
        lines.extend(
            [
                f"- Chosen mode: `{best['mode']}`",
                f"- Measured speed: `{best['tokens_per_s']:.3f} tok/s`",
                f"- Correctness passed: `{best['correctness_passed']}`",
            ]
        )
    else:
        lines.append("- Chosen mode: none")
    lines.extend(["", "## Rejected configurations", ""])
    for row in summary["rejected_configs"]:
        lines.append(
            f"- `{row['mode']}`: {row.get('rejection_reason', 'rejected')}"
        )
    return "\n".join(lines) + "\n"


def safe_claims(best: dict[str, Any] | None) -> list[str]:
    if best is None or not best.get("correctness_passed"):
        return []
    return [
        f"{best['mode']} reached {best['tokens_per_s']:.3f} tok/s in "
        "causal-KV correctness-gated decode."
    ]


def resolve_modes(
    raw: str,
    descriptor: Any,
    execution_schema: str,
) -> list[str]:
    requested = [value.strip() for value in raw.split(",") if value.strip()]
    if requested != ["auto"]:
        return requested
    modes = [
        "native_source" if descriptor.quantization.is_quantized else "bf16"
    ]
    modes.append("head8")
    if (
        descriptor.mlp_kind == "gated_dense"
        and execution_schema == "separate_qkv_gated_dense"
    ):
        modes.extend(
            (
                "gate_up_fp8",
                "quality_fp8",
                "mlp_fp8",
                "fast_body_fp8",
            )
        )
    return modes


def mode_flags(mode: str, layers: int) -> list[str]:
    flags = list(MODE_FLAGS[mode])
    if mode == "quality_fp8":
        start = max(0, round(layers * 0.22))
        end = min(layers, round(layers * 0.78))
        flags.extend(["--down-fp8-layers", f"{start}:{end}"])
    return flags


def resolve_residency(
    requested: str,
    weight_bytes: int,
    device: str,
) -> str:
    if requested != "auto":
        return requested
    if device != "cuda" or not torch.cuda.is_available():
        return "all"
    total = torch.cuda.get_device_properties(device).total_memory
    return "stream" if weight_bytes > int(total * 0.72) else "all"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def torch_oom_or_runtime() -> tuple[type[Exception], ...]:
    return (RuntimeError, ValueError, OSError, subprocess.SubprocessError)


if __name__ == "__main__":
    main()
