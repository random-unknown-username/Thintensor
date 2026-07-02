#!/usr/bin/env python3
"""Run locked ThinTensor decode profiles and summarize the hot path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thinruntime.archive import ThinArchive


RUNNER = ROOT / "scripts" / "thin_runtime.py"
QUALITY_FLAGS = [
    "--gate-up-fp8",
    "--down-proj-fp8",
    "--down-fp8-layers",
    "8:28",
]
REQUIRED_FIELDS = (
    "tokens_per_s",
    "steady_tokens_per_s",
    "ms_per_token",
    "launches_per_token",
    "estimated_weight_read_bytes_per_token",
    "effective_bandwidth_gb_s",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "resident_weight_bytes",
    "kv_cache_bytes",
    "temp_buffer_bytes",
    "lm_head_time_ms",
    "argmax_time_ms",
    "attention_time_ms",
)
TIMING_FIELDS = (
    "qkv_time_ms",
    "qk_time_ms",
    "softmax_time_ms",
    "value_mix_time_ms",
    "o_proj_time_ms",
    "gate_proj_time_ms",
    "up_proj_time_ms",
    "down_proj_time_ms",
    "lm_head_argmax_time_ms",
    "per_layer_average_time_ms",
    "attention_time_ms",
    "mlp_total_time_ms",
    "total_forward_token_time_ms",
    "silu_mul_time_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "SmolLM3-3B.thin")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--profiles",
        default="bf16,quality",
        help="Comma-separated subset of bf16,quality",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "runtime_optimizer",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "benchmark_results" / "runtime_optimizer" / "bottlenecks.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "runtime_optimizer" / "bottlenecks.md",
    )
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def profile_command(args: argparse.Namespace, profile: str) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "run",
        str(args.archive),
        "--device",
        "cuda",
        "--dtype",
        "bf16",
        "--steps",
        str(args.steps),
        "--residency",
        "all",
        "--kernel-backend",
        "triton",
        "--warmup-steps",
        str(args.warmup_steps),
        "--attention-mode",
        "causal_kv",
        "--lm-head-backend",
        "triton",
    ]
    if profile == "quality":
        command.extend(QUALITY_FLAGS)
    command.append("--json")
    return command


def run_profile(
    args: argparse.Namespace,
    profile: str,
) -> dict[str, Any]:
    output = args.out_dir / f"profile_{profile}_{args.steps}.json"
    stderr_path = output.with_suffix(".stderr")
    if args.reuse and output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    command = profile_command(args, profile)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{profile} profile failed with exit {completed.returncode}; "
            f"see {stderr_path}"
        )
    result = json.loads(completed.stdout)
    result["profile_name"] = profile
    result["benchmark_command"] = command
    missing = [field for field in REQUIRED_FIELDS if result.get(field) is None]
    breakdown = result.get("profile_breakdown", {})
    missing.extend(
        f"profile_breakdown.{field}"
        for field in TIMING_FIELDS
        if breakdown.get(field) is None
    )
    if missing:
        raise RuntimeError(
            f"{profile} profile lacks required telemetry: {', '.join(missing)}"
        )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def timing(result: dict[str, Any], field: str) -> float:
    return float(result.get("profile_breakdown", {}).get(field) or 0.0)


def model_geometry(archive_path: Path) -> dict[str, int]:
    with ThinArchive(archive_path) as archive:
        model = archive.manifest["model"]
    hidden = int(model["hidden_size"])
    heads = int(model["heads"])
    kv_heads = int(model["kv_heads"])
    head_dim = int(model.get("head_dim") or hidden // heads)
    return {
        "layers": int(model["layers"]),
        "hidden": hidden,
        "intermediate": int(model["intermediate_size"]),
        "q_dim": heads * head_dim,
        "kv_dim": kv_heads * head_dim,
        "vocab": int(model["vocab_size"]),
    }


def role_read_bytes(profile: str, model: dict[str, int]) -> dict[str, int]:
    layers = model["layers"]
    hidden = model["hidden"]
    intermediate = model["intermediate"]
    q_dim = model["q_dim"]
    kv_dim = model["kv_dim"]
    if profile == "quality":
        gate_up = layers * (
            2 * intermediate * hidden + 2 * intermediate * 4
        )
        down = (
            20 * (hidden * intermediate + hidden * 4)
            + (layers - 20) * hidden * intermediate * 2
        )
    else:
        gate_up = layers * 2 * intermediate * hidden * 2
        down = layers * hidden * intermediate * 2
    return {
        "lm_head": model["vocab"] * hidden * 2,
        "gate_up": gate_up,
        "down": down,
        "o_proj": layers * hidden * q_dim * 2,
        "qkv": layers * (q_dim + 2 * kv_dim) * hidden * 2,
    }


def component_rows(
    results: list[dict[str, Any]],
    model: dict[str, int],
) -> list[dict[str, Any]]:
    by_name = {row["profile_name"]: row for row in results}
    if set(by_name) != {"bf16", "quality"}:
        return []
    layers = model["layers"]
    specs = (
        (
            "LM head + argmax",
            lambda row: float(row["lm_head_time_ms"])
            + float(row["argmax_time_ms"]),
            "lm_head",
            (2, 2),
            "guarded shortlist, persistent vocab reduction, prepacked head",
        ),
        (
            "Gate/up projections",
            lambda row: layers
            * (timing(row, "gate_proj_time_ms") + timing(row, "up_proj_time_ms")),
            "gate_up",
            (layers, 2 * layers),
            "persistent FP8 GEMV, role-specific row scheduling",
        ),
        (
            "Down projection",
            lambda row: layers * timing(row, "down_proj_time_ms"),
            "down",
            (layers, layers),
            "loop-tile tuning, split-K only if end-to-end wins",
        ),
        (
            "O projection",
            lambda row: layers * timing(row, "o_proj_time_ms"),
            "o_proj",
            (layers, layers),
            "role-specific GEMV, exact layout packing",
        ),
        (
            "QKV projections",
            lambda row: layers * timing(row, "qkv_time_ms"),
            "qkv",
            (layers, layers),
            "net-zero contiguous packing, persistent grouped GEMV",
        ),
        (
            "Attention core",
            lambda row: layers
            * (
                timing(row, "qk_time_ms")
                + timing(row, "softmax_time_ms")
                + timing(row, "value_mix_time_ms")
            ),
            None,
            (3 * layers, 3 * layers),
            "context-aware block-split attention at long context",
        ),
    )
    reads = {
        name: role_read_bytes(name, model)
        for name in ("bf16", "quality")
    }
    rows = []
    for label, duration, read_role, launches, candidates in specs:
        bf16_ms = duration(by_name["bf16"])
        quality_ms = duration(by_name["quality"])
        rows.append(
            {
                "component": label,
                "bf16_ms_per_token": bf16_ms,
                "quality_ms_per_token": quality_ms,
                "bf16_percent": 100
                * bf16_ms
                / timing(by_name["bf16"], "total_forward_token_time_ms"),
                "quality_percent": 100
                * quality_ms
                / timing(by_name["quality"], "total_forward_token_time_ms"),
                "bf16_read_bytes": (
                    reads["bf16"][read_role]
                    if read_role is not None
                    else int(by_name["bf16"].get("kv_cache_read_bytes_per_token") or 0)
                ),
                "quality_read_bytes": (
                    reads["quality"][read_role]
                    if read_role is not None
                    else int(by_name["quality"].get("kv_cache_read_bytes_per_token") or 0)
                ),
                "bf16_launches": launches[0],
                "quality_launches": launches[1],
                "candidate_optimizations": candidates,
            }
        )
    rows.sort(
        key=lambda row: max(
            row["bf16_ms_per_token"],
            row["quality_ms_per_token"],
        ),
        reverse=True,
    )
    return rows


def render_markdown(
    results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
) -> str:
    lines = [
        "# Decode bottleneck profiles",
        "",
        "| profile | tok/s | steady tok/s | ms/token | qkv | qk | softmax | value mix | o proj | gate | up | down | lm head | argmax | launches | resident weights | KV allocated | temp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['profile_name']} | {row['tokens_per_s']:.3f} | "
            f"{row['steady_tokens_per_s']:.3f} | {row['ms_per_token']:.3f} | "
            f"{timing(row, 'qkv_time_ms'):.3f} | "
            f"{timing(row, 'qk_time_ms'):.3f} | "
            f"{timing(row, 'softmax_time_ms'):.3f} | "
            f"{timing(row, 'value_mix_time_ms'):.3f} | "
            f"{timing(row, 'o_proj_time_ms'):.3f} | "
            f"{timing(row, 'gate_proj_time_ms'):.3f} | "
            f"{timing(row, 'up_proj_time_ms'):.3f} | "
            f"{timing(row, 'down_proj_time_ms'):.3f} | "
            f"{row['lm_head_time_ms']:.3f} | "
            f"{row['argmax_time_ms']:.3f} | "
            f"{row['launches_per_token']} | "
            f"{row['resident_weight_bytes']} | {row['kv_cache_bytes']} | "
            f"{row['temp_buffer_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Ranked bottlenecks",
            "",
            "| Component | BF16 ms/token | Quality FP8 ms/token | BF16 % | Quality % | Read estimate BF16 / quality | Launches BF16 / quality | Candidate optimizations |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ranked:
        lines.append(
            f"| {row['component']} | {row['bf16_ms_per_token']:.3f} | "
            f"{row['quality_ms_per_token']:.3f} | {row['bf16_percent']:.1f}% | "
            f"{row['quality_percent']:.1f}% | {row['bf16_read_bytes']} / "
            f"{row['quality_read_bytes']} | {row['bf16_launches']} / "
            f"{row['quality_launches']} | {row['candidate_optimizations']} |"
        )
    lines.extend(
        [
            "",
            "Component timings come from an event-instrumented diagnostic pass "
            "after the real timed decode loop. Percentages rank work within that "
            "diagnostic pass; only steady full-decode tok/s decides retention.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = set(profiles) - {"bf16", "quality"}
    if unknown:
        raise ValueError(f"unsupported profiles: {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_profile(args, profile) for profile in profiles]
    geometry = model_geometry(args.archive)
    ranked = component_rows(results, geometry)
    payload = {
        "archive": str(args.archive),
        "model_geometry": geometry,
        "profiles": results,
        "ranked_bottlenecks": ranked,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_md.write_text(
        render_markdown(results, ranked),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
