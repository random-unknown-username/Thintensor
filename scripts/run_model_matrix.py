#!/usr/bin/env python3
"""Run a model/mode matrix with correctness-gated claims."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


MODE_FLAGS: dict[str, list[str]] = {
    "bf16": [],
    "head8": ["--lm-head-fp8"],
    "down_fp8": ["--down-proj-fp8"],
    "gate_up_fp8": ["--gate-up-fp8"],
    "mlp_fp8": ["--mlp-fp8"],
    "attn_proj_fp8": ["--attn-proj-fp8"],
    "qkv_fp8": ["--qkv-fp8"],
    "o_proj_fp8": ["--o-proj-fp8"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--out-dir", default="runs/model_matrix")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--correctness-prefill-lens", default="1,8")
    parser.add_argument("--correctness-steps", default="1,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config = load_config(Path(args.matrix))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command_log = out_dir / "commands.log"
    rows: list[dict[str, Any]] = []

    for model in config["models"]:
        for mode in config.get("modes", ["bf16"]):
            row = run_case(repo, out_dir, command_log, model, mode, args)
            rows.append(row)
            if row.get("error") and not args.continue_on_error:
                write_outputs(out_dir, rows)
                raise SystemExit(row["error"])

    write_outputs(out_dir, rows)
    print(f"wrote matrix outputs to {out_dir}")


def run_case(
    repo: Path,
    out_dir: Path,
    command_log: Path,
    model: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    name = model["name"]
    archive = Path(model["archive"])
    hf_dir = Path(model["hf_dir"]) if model.get("hf_dir") else None
    prefix = f"{slug(name)}__{slug(mode)}"
    row: dict[str, Any] = {
        "model": name,
        "mode": mode,
        "archive": str(archive),
        "hf_dir": str(hf_dir) if hf_dir else None,
    }
    try:
        residency, flags = mode_configuration(mode)
        correctness = None
        if hf_dir is not None and (hf_dir / "config.json").exists():
            correctness_path = out_dir / f"{prefix}.correctness.md"
            command = [
                sys.executable,
                str(repo / "scripts/compare_hf_thin_logits.py"),
                "--hf-model",
                str(hf_dir),
                "--archive",
                str(archive),
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
                str(correctness_path),
                "--json",
                *correctness_flags(flags),
            ]
            correctness = run_json(command, repo, command_log)
            row["correctness_json"] = str(correctness_path.with_suffix(".json"))
            row["hf_equivalent"] = bool(correctness.get("hf_equivalent"))
        else:
            row["correctness_skipped"] = "HF config/weights not available"
            row["hf_equivalent"] = False

        benchmark_path = out_dir / f"{prefix}.benchmark.json"
        command = [
            sys.executable,
            str(repo / "scripts/thin_runtime.py"),
            "run",
            str(archive),
            "--device",
            args.device,
            "--dtype",
            "bf16",
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--residency",
            residency,
            "--kernel-backend",
            "triton-matvec",
            "--attention-mode",
            "causal_kv",
            "--json",
            *flags,
        ]
        benchmark = run_json(command, repo, command_log)
        benchmark_path.write_text(
            json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        row.update(
            {
                "benchmark_json": str(benchmark_path),
                "tokens_per_s": benchmark.get("tokens_per_s"),
                "ms_per_token": benchmark.get("ms_per_token"),
                "peak_vram_bytes": benchmark.get("gpu_peak_allocated_bytes"),
                "effective_bandwidth_gb_s": benchmark.get(
                    "effective_bandwidth_gb_s"
                ),
                "top_logits": benchmark.get("top_logits"),
                "attention_mode": benchmark.get("attention_mode"),
                "not_hf_equivalent": benchmark.get("not_hf_equivalent"),
                "steady_state_eligible": (
                    args.steps >= 200 and args.warmup_steps >= 10
                ),
                "body_fp8_memory_saved_bytes": benchmark.get(
                    "body_fp8_memory_saved_bytes"
                ),
                "lm_head_memory_saved_bytes": benchmark.get(
                    "lm_head_memory_saved_bytes"
                ),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def mode_configuration(mode: str) -> tuple[str, list[str]]:
    if mode in MODE_FLAGS:
        return "all", list(MODE_FLAGS[mode])
    prefix = "stream_cpu_offload_"
    if mode.startswith(prefix):
        budget = mode[len(prefix) :]
        return "stream", ["--cpu-offload", "--gpu-weight-budget", budget]
    raise ValueError(f"unknown matrix mode {mode!r}")


def correctness_flags(flags: list[str]) -> list[str]:
    supported = {
        "--lm-head-fp8",
        "--mlp-fp8",
        "--down-proj-fp8",
        "--gate-up-fp8",
        "--attn-proj-fp8",
        "--qkv-fp8",
        "--o-proj-fp8",
    }
    return [flag for flag in flags if flag in supported]


def run_json(command: list[str], cwd: Path, log: Path) -> dict[str, Any]:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{dt.datetime.now(dt.timezone.utc).isoformat()} "
            f"{shlex.join(command)}\n"
        )
    env = os.environ.copy()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {shlex.join(command)}\n"
            f"{process.stderr[-4000:]}"
        )
    return json.loads(process.stdout)


def write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    matrix = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": rows}
    (out_dir / "benchmark_matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# ThinTensor benchmark matrix",
        "",
        "| model | mode | tok/s | ms/token | peak VRAM GiB | GB/s | correctness | status |",
        "|:---|:---|---:|---:|---:|---:|:---:|:---|",
    ]
    for row in rows:
        peak = row.get("peak_vram_bytes")
        markdown.append(
            f"| {row['model']} | {row['mode']} | "
            f"{number(row.get('tokens_per_s'))} | "
            f"{number(row.get('ms_per_token'))} | "
            f"{number(peak / 1024**3 if peak else None)} | "
            f"{number(row.get('effective_bandwidth_gb_s'))} | "
            f"{row.get('hf_equivalent', False)} | "
            f"{row.get('error', 'measured')} |"
        )
    (out_dir / "benchmark_matrix.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    safe: list[str] = ["# Safe claims", ""]
    unsafe: list[str] = ["# Unsafe claims", ""]
    for row in rows:
        label = f"{row['model']} {row['mode']}"
        if (
            not row.get("error")
            and row.get("hf_equivalent")
            and row.get("attention_mode") == "causal_kv"
            and not row.get("not_hf_equivalent")
            and row.get("steady_state_eligible")
        ):
            safe.append(
                f"- {label}: {number(row.get('tokens_per_s'))} tok/s in "
                "correctness-gated causal-KV decode."
            )
        else:
            unsafe.append(
                f"- Do not make HF-equivalent speed claims for {label}: "
                f"{row.get('error') or row.get('correctness_skipped') or 'correctness gate failed'}."
            )
    (out_dir / "safe_claims.md").write_text(
        "\n".join(safe) + "\n", encoding="utf-8"
    )
    (out_dir / "unsafe_claims.md").write_text(
        "\n".join(unsafe) + "\n", encoding="utf-8"
    )


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML input requires PyYAML; use JSON instead") from exc
    return yaml.safe_load(text)


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
