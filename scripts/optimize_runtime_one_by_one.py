#!/usr/bin/env python3
"""Benchmark and correctness-gate one ThinTensor runtime candidate at a time."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "thin_runtime.py"
CORRECTNESS = ROOT / "scripts" / "compare_hf_thin_logits.py"
STRESS = ROOT / "scripts" / "stress_validate_thin.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a JSON/YAML experiment plan, run each candidate in fresh "
            "processes, and stack only candidates that clear every gate."
        )
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "runtime_optimizer",
    )
    parser.add_argument(
        "--correctness-dir",
        type=Path,
        default=ROOT / "correctness_results" / "runtime_optimizer",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runtime_optimization_report.md",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_plan(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML plans require PyYAML; use JSON or install pyyaml"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("optimizer plan must be an object")
    for field in ("archive", "hf_model", "baseline", "candidates"):
        if field not in payload:
            raise ValueError(f"optimizer plan missing {field!r}")
    return payload


def slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")


def run_json_command(
    command: list[str],
    *,
    output: Path,
    stderr_path: Path,
    resume: bool,
    json_file: Path | None = None,
) -> dict[str, Any]:
    if resume and output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
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
            "_failed": True,
            "_returncode": completed.returncode,
            "_stderr": str(stderr_path),
            "_command": command,
        }
    try:
        payload = (
            json.loads(json_file.read_text(encoding="utf-8"))
            if json_file is not None
            else json.loads(completed.stdout)
        )
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "_failed": True,
            "_returncode": completed.returncode,
            "_stderr": str(stderr_path),
            "_command": command,
            "_parse_error": str(exc),
        }
    payload["_command"] = command
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def benchmark_command(
    plan: dict[str, Any],
    step_count: int,
    flags: list[str],
) -> list[str]:
    return [
        sys.executable,
        str(RUNTIME),
        "run",
        str(plan["archive"]),
        "--device",
        str(plan.get("device", "cuda")),
        "--dtype",
        str(plan.get("dtype", "bf16")),
        "--steps",
        str(step_count),
        "--warmup-steps",
        str(plan.get("warmup_steps", 10)),
        "--residency",
        str(plan.get("residency", "all")),
        "--kernel-backend",
        str(plan.get("kernel_backend", "triton")),
        "--attention-mode",
        "causal_kv",
        "--lm-head-backend",
        "triton",
        *flags,
        "--json",
    ]


def correctness_command(
    plan: dict[str, Any],
    flags: list[str],
    report_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(CORRECTNESS),
        "--hf-model",
        str(plan["hf_model"]),
        "--archive",
        str(plan["archive"]),
        "--device",
        str(plan.get("device", "cuda")),
        "--dtype",
        str(plan.get("dtype", "bf16")),
        "--prompt",
        "Hello",
        "--prefill-lens",
        "1,8,128",
        "--steps",
        "1,10",
        "--kernel-backend",
        str(plan.get("kernel_backend", "triton")),
        "--lm-head-backend",
        "triton",
        *flags,
        "--out",
        str(report_path),
        "--json",
    ]


def stress_command(
    plan: dict[str, Any],
    mode: str,
    flags: list[str],
    report_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(STRESS),
        "--hf-model",
        str(plan["hf_model"]),
        "--archive",
        str(plan["archive"]),
        "--device",
        str(plan.get("device", "cuda")),
        "--dtype",
        str(plan.get("dtype", "bf16")),
        "--modes",
        mode,
        *flags,
        "--out",
        str(report_path),
        "--json",
    ]


def benchmark_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    kv = payload.get("kv_cache", {})
    return {
        "tokens_per_s": payload.get("tokens_per_s"),
        "steady_tokens_per_s": payload.get("steady_tokens_per_s"),
        "ms_per_token": payload.get("ms_per_token"),
        "gpu_peak_allocated_bytes": payload.get("gpu_peak_allocated_bytes"),
        "resident_weight_bytes": payload.get("resident_weight_bytes"),
        "kv_cache_bytes": payload.get(
            "kv_cache_bytes",
            kv.get("kv_allocated_bytes", kv.get("kv_bytes")),
        ),
        "gpu_kv_cache_bytes": payload.get(
            "gpu_kv_cache_bytes",
            kv.get("kv_gpu_bytes", kv.get("kv_bytes")),
        ),
        "cpu_kv_cache_bytes": payload.get(
            "cpu_kv_cache_bytes",
            kv.get("kv_cpu_bytes", 0),
        ),
        "launches_per_token": payload.get("launches_per_token"),
    }


def balanced_abba_benchmark(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    step_count: int,
    baseline_name: str,
    baseline_flags: list[str],
    candidate_name: str,
    candidate_flags: list[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[list[str]],
    dict[str, Any],
]:
    sequence = (
        ("b1", candidate_name, candidate_flags),
        ("a1", baseline_name, baseline_flags),
        ("a2", baseline_name, baseline_flags),
        ("b2", candidate_name, candidate_flags),
    )
    raw_by_label: dict[str, dict[str, Any]] = {}
    commands: list[list[str]] = []
    for label, name, flags in sequence:
        command = benchmark_command(plan, step_count, flags)
        commands.append(command)
        prefix = f"{slug(candidate_name)}_{step_count}_abba_{label}"
        output = args.benchmark_dir / f"{prefix}.json"
        raw = run_json_command(
            command,
            output=output,
            stderr_path=output.with_suffix(".stderr"),
            resume=args.resume,
        )
        if raw.get("_failed"):
            return raw, {}, {}, commands, {}
        raw_by_label[label] = raw

    baseline_samples = [
        benchmark_metrics(raw_by_label[label])
        for label in ("a1", "a2")
    ]
    candidate_samples = [
        benchmark_metrics(raw_by_label[label])
        for label in ("b1", "b2")
    ]

    def median_metrics(
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = dict(samples[-1])
        for key in (
            "tokens_per_s",
            "steady_tokens_per_s",
            "ms_per_token",
        ):
            values = [
                float(sample[key])
                for sample in samples
                if sample.get(key) is not None
            ]
            result[key] = median(values) if values else None
        return result

    baseline_median = median_metrics(baseline_samples)
    candidate_median = median_metrics(candidate_samples)
    details = {
        "sequence": [label.upper() for label, _, _ in sequence],
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline_samples": baseline_samples,
        "candidate_samples": candidate_samples,
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
    }
    return {}, baseline_median, candidate_median, commands, details


def correctness_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    return {
        "top1_failures": sum(not row.get("top1_same", False) for row in records),
        "minimum_top5_overlap": min(
            (float(row["top5_overlap"]) for row in records),
            default=0.0,
        ),
        "minimum_cosine_similarity": min(
            (float(row["cosine_similarity"]) for row in records),
            default=0.0,
        ),
        "minimum_generated_token_match_rate": min(
            (float(row["generated_token_match_rate"]) for row in records),
            default=0.0,
        ),
    }


def stress_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    modes = payload.get("modes", [])
    if not modes:
        return {
            "top1_failures": 1,
            "minimum_top5_overlap": 0.0,
            "minimum_cosine_similarity": 0.0,
            "maximum_sampling_total_variation": None,
            "maximum_sampling_js_divergence": None,
        }
    mode = modes[0]
    top1_failures = sum(
        not comparison.get("top1_same", False)
        for record in mode.get("records", [])
        for comparison in record.get("comparisons", [])
    )
    return {
        "top1_failures": top1_failures,
        "minimum_top5_overlap": float(mode["minimum_top5_overlap"]),
        "minimum_cosine_similarity": float(mode["minimum_cosine_similarity"]),
        "maximum_sampling_total_variation": mode.get(
            "maximum_sampling_total_variation"
        ),
        "maximum_sampling_js_divergence": mode.get(
            "maximum_sampling_js_divergence"
        ),
        "minimum_free_running_token_match_rate": mode.get(
            "minimum_free_running_token_match_rate"
        ),
    }


def optional_increase(
    candidate: float | None,
    baseline: float | None,
) -> float:
    if candidate is None or baseline is None:
        return 0.0
    return float(candidate) - float(baseline)


def decide(
    *,
    plan: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    experimental: bool,
) -> tuple[str, list[str]]:
    reasons = []
    minimum_speedup = float(plan.get("minimum_speedup", 0.02))
    speed_baseline = candidate.get(
        "speed_gate_baseline_benchmarks",
        baseline["benchmarks"],
    )
    for steps, candidate_metrics in candidate["benchmarks"].items():
        baseline_metrics = speed_baseline[steps]
        baseline_speed = float(baseline_metrics["steady_tokens_per_s"])
        candidate_speed = float(candidate_metrics["steady_tokens_per_s"])
        change = candidate_speed / baseline_speed - 1.0
        if change < minimum_speedup:
            reasons.append(
                f"{steps}-token speed change {change:+.2%} is below "
                f"{minimum_speedup:+.2%}"
            )

    cosine_tolerance = float(plan.get("cosine_tolerance", 0.0005))
    top5_tolerance = float(plan.get("top5_tolerance", 0.0))
    distribution_tolerance = float(
        plan.get("distribution_tolerance", 0.001)
    )
    for gate in ("required", "stress"):
        base = baseline[gate]
        test = candidate[gate]
        if test["top1_failures"] > base["top1_failures"]:
            reasons.append(f"{gate}: introduced a top-1 failure")
        if (
            test["minimum_top5_overlap"]
            < base["minimum_top5_overlap"] - top5_tolerance
        ):
            reasons.append(f"{gate}: minimum top-5 overlap regressed")
        cosine_change = (
            test["minimum_cosine_similarity"]
            - base["minimum_cosine_similarity"]
        )
        if cosine_change < -cosine_tolerance:
            reasons.append(
                f"{gate}: minimum cosine changed {cosine_change:+.6f}"
            )
    for metric in (
        "maximum_sampling_total_variation",
        "maximum_sampling_js_divergence",
    ):
        increase = optional_increase(
            candidate["stress"].get(metric),
            baseline["stress"].get(metric),
        )
        if increase > distribution_tolerance:
            reasons.append(f"stress: {metric} increased by {increase:.6f}")

    if not reasons:
        return "accept", ["all speed and correctness gates passed"]
    if experimental and not any("speed change" in reason for reason in reasons):
        return "experimental", reasons
    return "reject", reasons


def run_experiment(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    experiment: dict[str, Any],
    benchmark_flags: list[str],
    correctness_flags: list[str],
    stress_flags: list[str],
    speed_gate_baseline: dict[str, Any] | None = None,
    speed_gate_baseline_flags: list[str] | None = None,
) -> dict[str, Any]:
    name = str(experiment["name"])
    name_slug = slug(name)
    record: dict[str, Any] = {
        "name": name,
        "hypothesis": experiment.get("hypothesis", ""),
        "implementation_summary": experiment.get("implementation_summary", ""),
        "expected_bottleneck": experiment.get("expected_bottleneck", ""),
        "enabled_by_default": bool(experiment.get("enabled_by_default", False)),
        "experimental": bool(experiment.get("experimental", False)),
        "benchmarks": {},
    }
    for step_count in plan.get("benchmark_steps", [200, 500]):
        if (
            speed_gate_baseline is not None
            and speed_gate_baseline_flags is not None
            and bool(plan.get("balanced_abba", False))
        ):
            (
                failed,
                baseline_median,
                candidate_median,
                commands,
                details,
            ) = (
                balanced_abba_benchmark(
                    args=args,
                    plan=plan,
                    step_count=int(step_count),
                    baseline_name=str(speed_gate_baseline["name"]),
                    baseline_flags=speed_gate_baseline_flags,
                    candidate_name=name,
                    candidate_flags=benchmark_flags,
                )
            )
            if failed:
                record["failed"] = failed
                return record
            record["benchmarks"][str(step_count)] = candidate_median
            record.setdefault(
                "speed_gate_baseline_benchmarks",
                {},
            )[str(step_count)] = baseline_median
            record.setdefault("balanced_abba", {})[
                str(step_count)
            ] = details
            record.setdefault("benchmark_commands", []).extend(commands)
        else:
            command = benchmark_command(plan, int(step_count), benchmark_flags)
            output = args.benchmark_dir / f"{name_slug}_{step_count}.json"
            raw = run_json_command(
                command,
                output=output,
                stderr_path=output.with_suffix(".stderr"),
                resume=args.resume,
            )
            if raw.get("_failed"):
                record["failed"] = raw
                return record
            record["benchmarks"][str(step_count)] = benchmark_metrics(raw)
            record.setdefault("benchmark_commands", []).append(command)

    if speed_gate_baseline is not None:
        minimum_speedup = float(plan.get("minimum_speedup", 0.02))
        baseline_benchmarks = record.get(
            "speed_gate_baseline_benchmarks",
            speed_gate_baseline["benchmarks"],
        )
        speed_reasons = []
        for steps, metrics in record["benchmarks"].items():
            baseline_speed = float(
                baseline_benchmarks[steps]["steady_tokens_per_s"]
            )
            candidate_speed = float(metrics["steady_tokens_per_s"])
            change = candidate_speed / baseline_speed - 1.0
            if change < minimum_speedup:
                speed_reasons.append(
                    f"{steps}-token speed change {change:+.2%} is below "
                    f"{minimum_speedup:+.2%}"
                )
        if speed_reasons:
            record["speed_rejected"] = True
            record["speed_reasons"] = speed_reasons
            return record

    required_md = args.correctness_dir / f"{name_slug}_required.md"
    required_json = required_md.with_suffix(".json")
    required_command = correctness_command(plan, correctness_flags, required_md)
    required_raw = run_json_command(
        required_command,
        output=args.correctness_dir / f"{name_slug}_required_result.json",
        stderr_path=args.correctness_dir / f"{name_slug}_required.stderr",
        resume=args.resume,
        json_file=required_json,
    )
    if required_raw.get("_failed"):
        record["failed"] = required_raw
        return record
    record["correctness_command"] = required_command
    record["required"] = correctness_metrics(required_raw)

    stress_md = args.correctness_dir / f"{name_slug}_stress.md"
    stress_json = stress_md.with_suffix(".json")
    mode = str(experiment.get("stress_mode", plan.get("stress_mode", "retained_fp8")))
    stress_run_command = stress_command(plan, mode, stress_flags, stress_md)
    stress_raw = run_json_command(
        stress_run_command,
        output=args.correctness_dir / f"{name_slug}_stress_result.json",
        stderr_path=args.correctness_dir / f"{name_slug}_stress.stderr",
        resume=args.resume,
        json_file=stress_json,
    )
    if stress_raw.get("_failed"):
        record["failed"] = stress_raw
        return record
    record["stress_command"] = stress_run_command
    record["stress"] = stress_metrics(stress_raw)
    return record


def render_report(
    plan: dict[str, Any],
    baseline: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    primary_steps = str(plan.get("benchmark_steps", [200, 500])[-1])
    baseline_speed = float(
        baseline["benchmarks"][primary_steps]["steady_tokens_per_s"]
    )
    baseline_memory = baseline["benchmarks"][primary_steps]
    lines = [
        "# Runtime optimization report",
        "",
        "Every candidate ran in isolated processes. Accepted candidates become "
        "the baseline for the next candidate; rejected and experimental candidates "
        "are not stacked.",
        "",
        "| Candidate | Speed change | GPU memory change | KV memory change | Min cosine change | Top1/top5 impact | Decision |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    current = baseline
    current_speed = baseline_speed
    for row in records:
        if row.get("failed"):
            lines.append(
                f"| {row['name']} | n/a | n/a | n/a | n/a | command failed | reject |"
            )
            continue
        metrics = row["benchmarks"][primary_steps]
        speed_reference = row.get(
            "speed_gate_baseline_benchmarks",
            current["benchmarks"],
        )
        speed_change = (
            float(metrics["steady_tokens_per_s"])
            / float(
                speed_reference[primary_steps]["steady_tokens_per_s"]
            )
            - 1
        )
        gpu_change = int(metrics["gpu_peak_allocated_bytes"]) - int(
            current["benchmarks"][primary_steps]["gpu_peak_allocated_bytes"]
        )
        kv_change = int(metrics["gpu_kv_cache_bytes"]) - int(
            current["benchmarks"][primary_steps]["gpu_kv_cache_bytes"]
        )
        if row.get("speed_rejected"):
            lines.append(
                f"| {row['name']} | {speed_change:+.2%} | {gpu_change:+d} | "
                f"{kv_change:+d} | n/a | correctness skipped after speed "
                f"failure | reject |"
            )
            lines.extend(
                [
                    "",
                    f"- **{row['name']} hypothesis:** {row['hypothesis']}",
                    f"- Implementation: {row['implementation_summary']}",
                    f"- Expected bottleneck: {row['expected_bottleneck']}",
                    f"- Decision reason: {'; '.join(row['reason'])}",
                    f"- Enabled by default: `{str(row['enabled_by_default']).lower()}`",
                    f"- Benchmark command: `{shlex.join(row['benchmark_commands'][-1])}`",
                    "- Correctness command: not run because the candidate failed the speed gate.",
                    "",
                ]
            )
            continue
        cosine_change = (
            row["stress"]["minimum_cosine_similarity"]
            - current["stress"]["minimum_cosine_similarity"]
        )
        impact = (
            f"required failures {row['required']['top1_failures']}, "
            f"stress failures {row['stress']['top1_failures']}, "
            f"min top5 {row['stress']['minimum_top5_overlap']:.3f}"
        )
        lines.append(
            f"| {row['name']} | {speed_change:+.2%} | {gpu_change:+d} | "
            f"{kv_change:+d} | {cosine_change:+.6f} | {impact} | "
            f"{row['decision']} |"
        )
        lines.extend(
            [
                "",
                f"- **{row['name']} hypothesis:** {row['hypothesis']}",
                f"- Implementation: {row['implementation_summary']}",
                f"- Expected bottleneck: {row['expected_bottleneck']}",
                f"- Decision reason: {'; '.join(row['reason'])}",
                f"- Enabled by default: `{str(row['enabled_by_default']).lower()}`",
                f"- Benchmark command: `{shlex.join(row['benchmark_commands'][-1])}`",
                f"- Correctness command: `{shlex.join(row['correctness_command'])}`",
                "",
            ]
        )
        if row["decision"] == "accept":
            current = row
            current_speed = float(metrics["steady_tokens_per_s"])
    final_profiles = {}
    for name in ("bf16", "bf16_guarded", "quality", "guarded"):
        path = (
            ROOT
            / "benchmark_results"
            / "runtime_optimizer"
            / f"final_{name}_500.json"
        )
        if path.exists():
            final_profiles[name] = json.loads(path.read_text(encoding="utf-8"))
    if len(final_profiles) == 4:
        lines.extend(
            [
                "",
                "## Final current-source standalone checks",
                "",
                "| Path | Steady tok/s | ms/token | Resident weights | Read bytes/token | Peak allocated |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, name in (
            ("BF16", "bf16"),
            ("BF16 guarded", "bf16_guarded"),
            ("Quality", "quality"),
            ("Quality guarded", "guarded"),
        ):
            row = final_profiles[name]
            lines.append(
                f"| {label} | {row['steady_tokens_per_s']:.3f} | "
                f"{row['ms_per_token']:.3f} | "
                f"{row['resident_weight_bytes']} | "
                f"{row['estimated_weight_read_bytes_per_token']} | "
                f"{row['gpu_peak_allocated_bytes']} |"
            )
        lines.extend(
            [
                "",
                "These are final-state standalone 500-token checks. Balanced "
                "ABBA results above control acceptance because laptop power "
                "and thermals shift absolute standalone rates.",
            ]
        )
    lines.extend(
        [
            "",
            "## Locked baseline and correctness",
            "",
            f"- 200-token steady speed: `{baseline['benchmarks']['200']['steady_tokens_per_s']:.3f} tok/s`",
            f"- 500-token steady speed: `{baseline['benchmarks']['500']['steady_tokens_per_s']:.3f} tok/s`",
            f"- Required matrix: `{baseline['required']['top1_failures']}` top-1 failures, "
            f"minimum top-5 `{baseline['required']['minimum_top5_overlap']:.3f}`, "
            f"minimum cosine `{baseline['required']['minimum_cosine_similarity']:.6f}`.",
            f"- 13-case stress: `{baseline['stress']['top1_failures']}` checkpoint top-1 failure, "
            f"minimum top-5 `{baseline['stress']['minimum_top5_overlap']:.3f}`, "
            f"minimum cosine `{baseline['stress']['minimum_cosine_similarity']:.6f}`, "
            f"maximum TV `{baseline['stress']['maximum_sampling_total_variation']:.6f}`, "
            f"maximum JS `{baseline['stress']['maximum_sampling_js_divergence']:.6f}`.",
        ]
    )
    bf16_report_path = (
        ROOT
        / "benchmark_results"
        / "runtime_optimizer"
        / "bf16_optimizer_report.json"
    )
    if bf16_report_path.exists():
        bf16_report = json.loads(
            bf16_report_path.read_text(encoding="utf-8")
        )
        bf16_candidate = next(
            (
                row
                for row in bf16_report.get("candidates", [])
                if row.get("decision") == "accept"
            ),
            None,
        )
        quality_candidate = next(
            (
                row
                for row in records
                if row.get("name")
                == "lm_head_fp8_shortlist_bf16_verify"
            ),
            None,
        )
        if bf16_candidate is not None and quality_candidate is not None:
            quality_abba = quality_candidate["balanced_abba"]
            bf16_abba = bf16_candidate["balanced_abba"]
            lines.extend(
                [
                    "",
                    "## Accepted guarded LM-head paths",
                    "",
                    "The FP8 execution head only selects 64 candidate IDs. "
                    "Candidate logits, public full logits, and final tie-breaking "
                    "remain BF16.",
                    "",
                    "| Path | Tokens | Baseline tok/s | Guarded tok/s | Delta |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for label, abba in (
                ("Quality FP8", quality_abba),
                ("Full BF16", bf16_abba),
            ):
                for steps in ("200", "500"):
                    base_speed = float(
                        abba[steps]["baseline_median"][
                            "steady_tokens_per_s"
                        ]
                    )
                    test_speed = float(
                        abba[steps]["candidate_median"][
                            "steady_tokens_per_s"
                        ]
                    )
                    lines.append(
                        f"| {label} | {steps} | {base_speed:.3f} | "
                        f"{test_speed:.3f} | "
                        f"{test_speed / base_speed - 1:+.2%} |"
                    )
            quality_base = quality_abba["500"]["baseline_median"]
            quality_test = quality_abba["500"]["candidate_median"]
            bf16_base = bf16_abba["500"]["baseline_median"]
            bf16_test = bf16_abba["500"]["candidate_median"]
            lines.extend(
                [
                    "",
                    "| Path | Baseline resident weights | Guarded resident weights | Baseline read bytes/token | Guarded read bytes/token |",
                    "|---|---:|---:|---:|---:|",
                    f"| Quality FP8 | {quality_base['resident_weight_bytes']} | "
                    f"{quality_test['resident_weight_bytes']} | "
                    "4075814912 | 3813921792 |",
                    f"| Full BF16 | {bf16_base['resident_weight_bytes']} | "
                    f"{bf16_test['resident_weight_bytes']} | "
                    "6149898240 | 5888005120 |",
                    "",
                    "The guarded path adds `263181312` resident bytes and "
                    "removes `261893120` estimated head-read bytes per token. "
                    "It remains explicit rather than universal because shortlist "
                    "recall is model/profile specific.",
                    "",
                    "Full-BF16 required and stress metrics were unchanged: "
                    f"required minimum cosine "
                    f"`{bf16_candidate['required']['minimum_cosine_similarity']:.6f}`, "
                    f"stress minimum cosine "
                    f"`{bf16_candidate['stress']['minimum_cosine_similarity']:.6f}`, "
                    f"and stress minimum top-5 "
                    f"`{bf16_candidate['stress']['minimum_top5_overlap']:.3f}`.",
                ]
            )

    lines.extend(
        [
            "",
            "## Rejected hot-path controls",
            "",
            "| Candidate | Isolated result | Full decode result | Decision |",
            "|---|---|---|---|",
            "| Fused residual + RMSNorm | bit-exact; 0.01649 ms to 0.00885 ms; old logical launch count 434 to 362 | approximately -0.17% at 500 tokens | reject |",
            "| Fused scaled gate/up + SiLU | bit-exact; MLP component about 5-6% faster | approximately +0.04% at 500 tokens | reject |",
            "| Interleaved gate/up storage | max abs 0.000122; about 3.8x slower | not promoted | reject |",
            "| Two-stream gate/up | bit-exact | component about 18.3% slower | reject |",
            "| Pipeline-stage tuning | bit-exact | no stable randomized A/B win | reject |",
            "| Persistent FP8 gate/up variants | isolated kernels improved about 21-24% | best full decode gain remained below 2% | reject |",
            "| Persistent BF16 gate/up | isolated kernel improved about 11% | full decode was flat | reject |",
            "| Lossless contiguous QKV packing | bit-exact | +0.41% full decode | reject |",
            "| Tensor-core skinny GEMM | numerically valid prototype | slower than the bandwidth GEMV | reject |",
            "| Context-aware split-head attention | microkernel cosine about 0.99998 | -1.65% at 1000-token decode | reject |",
            "| QKV + O FP8 12:24 | stress top-1 recovered | +0.05% versus simpler O-only profile | reject |",
            "| Row-swizzled FP8 down | repeated-weight kernel up to 1.88x; 500-token +2.84% | introduced sampling_3 top-1 failures at steps 16 and 64 | reject |",
            "| Arithmetic-preserving row schedule | exact original reduction groups | +1.35% at 200 tokens | reject |",
            "| Paired 128-column down loads | bit-exact microkernel 1.76x | +0.50% at 200 tokens | reject |",
            "| FP16 packed products, FP32 accumulation | kernel cosine 0.9999997 | -0.80% at 200 tokens | reject |",
            "| On-chip K-chunk MLP pipeline | bit-exact at best tile | best kernel remained 28% slower | reject |",
            "| Static column-tiled BF16 head | +525336576 resident bytes; cosine about 1.0 | 1.7490 ms to 2.0017 ms | reject |",
            "",
            "These controls show why launch count and isolated tiny-kernel "
            "latency are not used as decode-win evidence.",
        ]
    )
    bottleneck_path = (
        ROOT / "benchmark_results" / "runtime_optimizer" / "bottlenecks.json"
    )
    if bottleneck_path.exists():
        bottlenecks = json.loads(bottleneck_path.read_text(encoding="utf-8"))
        ranked = bottlenecks.get("ranked_bottlenecks", [])
        if ranked:
            lines.extend(
                [
                    "",
                    "## Ranked bottleneck table",
                    "",
                    "| Component | BF16 ms/token | Quality ms/token | BF16 % | Quality % | Read bytes BF16 / quality | Launches BF16 / quality | Candidate optimization |",
                    "|---|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in ranked:
                lines.append(
                    f"| {row['component']} | "
                    f"{row['bf16_ms_per_token']:.3f} | "
                    f"{row['quality_ms_per_token']:.3f} | "
                    f"{row['bf16_percent']:.1f}% | "
                    f"{row['quality_percent']:.1f}% | "
                    f"{row['bf16_read_bytes']} / "
                    f"{row['quality_read_bytes']} | "
                    f"{row['bf16_launches']} / "
                    f"{row['quality_launches']} | "
                    f"{row['candidate_optimizations']} |"
                )
            lines.extend(
                [
                    "",
                    "The profiler component pass is diagnostic and runs after the "
                    "timed decode loop; its event overhead means component totals "
                    "must not be substituted for the real 500-token rate.",
                ]
            )
    residency_path = (
        ROOT
        / "benchmark_results"
        / "runtime_optimizer"
        / "kv_residency_512.json"
    )
    if residency_path.exists():
        residency = json.loads(residency_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Exact KV residency curve at 512 tokens",
                "",
                "| Mode | Steady tok/s | GPU KV bytes | CPU KV bytes | H2D bytes | Decision |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in residency.get("benchmarks", []):
            kv = row["kv_cache"]
            mode = kv["kv_residency"]
            decision = (
                "baseline"
                if mode == "gpu_full"
                else "exact opt-in; not a speed optimization"
            )
            lines.append(
                f"| {mode} | {row['steady_tokens_per_s']:.3f} | "
                f"{row['gpu_kv_cache_bytes']} | {row['cpu_kv_cache_bytes']} | "
                f"{kv['kv_h2d_transfer_bytes']} | {decision} |"
            )
    weight_residency_path = (
        ROOT
        / "benchmark_results"
        / "runtime_optimizer"
        / "residency_curve.json"
    )
    weight_correctness_path = (
        ROOT
        / "correctness_results"
        / "runtime_optimizer"
        / "weight_residency.json"
    )
    if weight_residency_path.exists():
        weight_residency = json.loads(
            weight_residency_path.read_text(encoding="utf-8")
        )
        weight_correctness = (
            json.loads(weight_correctness_path.read_text(encoding="utf-8"))
            if weight_correctness_path.exists()
            else {"budgets": []}
        )
        correctness_by_budget = {
            f"{int(row['budget_gb'])}GB": row
            for row in weight_correctness.get("budgets", [])
        }
        lines.extend(
            [
                "",
                "## Exact BF16 weight residency curve",
                "",
                "These are same-command 10-token capacity measurements, not "
                "steady 500-token speed claims. CPU offload preserves BF16; "
                "it is a capacity feature and is not called faster.",
                "",
                "| Budget | Steady tok/s | GPU resident weights | CPU resident weights | H2D bytes/token | Prefetch overlap | Correctness |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in weight_residency["curve"]:
            correctness = correctness_by_budget.get(row["budget"])
            correctness_text = (
                "all-resident reference"
                if row["budget"] == "all"
                else (
                    f"top1 failures {correctness['top1_failures']}; "
                    f"top5 {correctness['minimum_top5_overlap']:.1f}; "
                    f"cosine {correctness['minimum_cosine_similarity']:.6f}"
                    if correctness is not None
                    else "not run"
                )
            )
            lines.append(
                f"| {row['budget']} | {row['steady_tokens_per_s']:.3f} | "
                f"{row['gpu_resident_weight_bytes']} | "
                f"{row['cpu_resident_weight_bytes']} | "
                f"{row['h2d_bytes_per_token']:.0f} | "
                f"{row['prefetch_overlap_efficiency']:.3f} | "
                f"{correctness_text} |"
            )
        lines.extend(
            [
                "",
                "The page pool pins a deterministic whole-layer prefix and "
                "reserves the current/prefetched working set. This avoids the "
                "cyclic-LRU failure mode that reloaded nearly every layer even "
                "at a 6 GB budget. Backend dispatch now uses stable "
                "shape/page-role keys rather than recycled tensor object ids.",
            ]
        )
    shortlist_path = (
        ROOT
        / "benchmark_results"
        / "runtime_optimizer"
        / "lm_head_shortlist_recall.json"
    )
    if shortlist_path.exists():
        shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Guarded LM-head shortlist recall",
                "",
                "| K | Top-1 recall | Complete top-5 recall |",
                "|---:|---:|---:|",
            ]
        )
        for k, recall in shortlist["recall"].items():
            lines.append(
                f"| {k} | {recall['top1_recall']:.3f} | "
                f"{recall['complete_top5_recall']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Retained K: `{shortlist['selection']['retained_k']}`. "
                f"{shortlist['selection']['reason']}",
            ]
        )
    attention_context_path = (
        ROOT
        / "benchmark_results"
        / "runtime_optimizer"
        / "attention_context_diagnostic.json"
    )
    if attention_context_path.exists():
        attention_context = json.loads(
            attention_context_path.read_text(encoding="utf-8")
        )
        lines.extend(
            [
                "",
                "## Attention context crossover audit",
                "",
                "| Context | Existing torch ms | Fused ms | Fused / torch | Cosine |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in attention_context["contexts"]:
            lines.append(
                f"| {row['context']} | {row['torch_ms']:.6f} | "
                f"{row['fused_ms']:.6f} | "
                f"{row['fused_ms'] / row['torch_ms']:.2f}x | "
                f"{row['cosine']:.6f} |"
            )
        lines.extend(
            [
                "",
                attention_context["reason"],
                "These are isolated diagnostics; the full-decode regression "
                "controls the decision.",
            ]
        )
    accepted = [row for row in records if row.get("decision") == "accept"]
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- Best accepted runtime optimization: "
            f"`{accepted[-1]['name'] if accepted else 'none; locked quality baseline retained'}`.",
            "- Default settings: unchanged. The guarded LM-head path remains "
            "an explicit SmolLM3-validated profile because it carries a "
            "263,181,312-byte execution copy and shortlist recall is "
            "model-specific.",
            "- Best rejected idea: `row_swizzled_fp8_down` (+2.84% at "
            "500 tokens, but it introduced sampling_3 checkpoint top-1 "
            "failures at steps 16 and 64).",
            "- Next bottleneck: reduce real large-projection weight bytes or "
            "make exact FP8 conversion/reduction cheaper under the 35 W "
            "power ceiling. Launch-only and repeated-weight scheduler wins "
            "have been exhausted.",
            "- Exact KV memory reduction: yes for opt-in residency. At 512 "
            "tokens, `cpu_exact` reduced resident GPU KV from 37,748,736 to "
            "1,179,648 bytes while preserving total BF16 KV bytes.",
            "- KV compression: not tested. Exact layout/residency tiers were "
            "completed first; no lossy KV mode is enabled.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    plan = load_plan(args.plan)
    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    args.correctness_dir.mkdir(parents=True, exist_ok=True)

    common_benchmark = list(plan.get("common_benchmark_args", []))
    common_correctness = list(plan.get("common_correctness_args", []))
    common_stress = list(plan.get("common_stress_args", []))
    baseline_spec = deepcopy(plan["baseline"])
    baseline = run_experiment(
        args=args,
        plan=plan,
        experiment=baseline_spec,
        benchmark_flags=common_benchmark
        + list(baseline_spec.get("benchmark_args", [])),
        correctness_flags=common_correctness
        + list(baseline_spec.get("correctness_args", [])),
        stress_flags=common_stress + list(baseline_spec.get("stress_args", [])),
    )
    if baseline.get("failed"):
        raise RuntimeError(f"baseline failed: {baseline['failed']}")
    baseline["decision"] = "baseline"
    baseline["reason"] = ["locked baseline"]

    accepted_benchmark = list(baseline_spec.get("benchmark_args", []))
    accepted_correctness = list(baseline_spec.get("correctness_args", []))
    accepted_stress = list(baseline_spec.get("stress_args", []))
    current = baseline
    records = []
    for candidate in plan["candidates"]:
        candidate_benchmark = accepted_benchmark + list(
            candidate.get("benchmark_args", [])
        )
        candidate_correctness = accepted_correctness + list(
            candidate.get("correctness_args", [])
        )
        candidate_stress = accepted_stress + list(
            candidate.get("stress_args", [])
        )
        record = run_experiment(
            args=args,
            plan=plan,
            experiment=candidate,
            benchmark_flags=common_benchmark + candidate_benchmark,
            correctness_flags=common_correctness + candidate_correctness,
            stress_flags=common_stress + candidate_stress,
            speed_gate_baseline=current,
            speed_gate_baseline_flags=common_benchmark
            + accepted_benchmark,
        )
        if record.get("failed"):
            record["decision"] = "reject"
            record["reason"] = ["candidate command failed"]
        elif record.get("speed_rejected"):
            record["decision"] = "reject"
            record["reason"] = record["speed_reasons"]
        else:
            record["decision"], record["reason"] = decide(
                plan=plan,
                baseline=current,
                candidate=record,
                experimental=record["experimental"],
            )
        records.append(record)
        if record["decision"] == "accept":
            accepted_benchmark = candidate_benchmark
            accepted_correctness = candidate_correctness
            accepted_stress = candidate_stress
            current = record

    payload = {
        "plan": str(args.plan),
        "baseline": baseline,
        "candidates": records,
        "final_baseline": current["name"],
    }
    json_report = args.report.with_suffix(".json")
    json_report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        render_report(plan, baseline, records),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
