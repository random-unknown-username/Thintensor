#!/usr/bin/env python3
"""Merge Q8 correctness and speed evidence into the safe comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness",
        default="correctness_results/q8_baseline_report.json",
    )
    parser.add_argument(
        "--benchmark",
        default="benchmark_results/q8_baseline_bench.json",
    )
    parser.add_argument(
        "--head8-correctness",
        default="correctness_results/quality_body_head8.json",
    )
    parser.add_argument(
        "--head8-benchmark",
        default="benchmark_results/quality_body_head8_real_300.json",
    )
    parser.add_argument(
        "--tuned-correctness",
        default="correctness_results/q8_tune_10_26_report.json",
    )
    parser.add_argument(
        "--tuned-benchmark",
        default="benchmark_results/q8_tune_10_26_bench.json",
    )
    parser.add_argument("--out", default="q8_vs_thintensor_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    correctness = load(args.correctness)
    benchmark = load(args.benchmark)
    quality = by_key(correctness["modes"], "mode")
    speed = by_key(benchmark["results"], "mode")
    rows = [
        make_row("HF BF16", quality.get("hf_bf16"), speed.get("hf_bf16")),
        make_row(
            "HF bitsandbytes int8/Q8-ish",
            quality.get("hf_bnb_int8"),
            speed.get("hf_bnb_int8"),
        ),
        make_row(
            "ThinTensor BF16",
            quality.get("thin_bf16"),
            speed.get("thin_bf16"),
        ),
        make_row(
            "ThinTensor quality FP8",
            quality.get("thin_quality_fp8"),
            speed.get("thin_quality_fp8"),
        ),
    ]
    if Path(args.tuned_correctness).is_file() and Path(
        args.tuned_benchmark
    ).is_file():
        tuned_quality = by_key(
            load(args.tuned_correctness)["modes"],
            "mode",
        ).get("thin_quality_10_26")
        tuned_speed = by_key(
            load(args.tuned_benchmark)["results"],
            "mode",
        ).get("thin_quality_10_26")
        rows.append(
            make_row(
                "ThinTensor quality FP8 10:26",
                tuned_quality,
                tuned_speed,
            )
        )
    if Path(args.head8_correctness).is_file() and Path(
        args.head8_benchmark
    ).is_file():
        rows.append(
            make_row(
                "ThinTensor quality FP8 + Head8 experimental",
                load(args.head8_correctness),
                load(args.head8_benchmark),
                head8=True,
            )
        )
    deployment = speed.get("ollama_q8_0") or speed.get("llamacpp_q8_0")
    if deployment is not None:
        rows.append(
            make_row(
                "Ollama/llama.cpp Q8_0",
                None,
                deployment,
            )
        )

    bnb = next(row for row in rows if row["name"].startswith("HF bits"))
    thin_candidates = [
        row
        for row in rows
        if row["name"].startswith("ThinTensor quality FP8")
        and "Head8" not in row["name"]
    ]
    passing = [row for row in thin_candidates if row["top1"] is True]
    thin = max(
        passing or thin_candidates,
        key=lambda row: row["speed"] or 0.0,
    )
    decision = decide(thin, bnb)
    output = Path(args.out)
    output.write_text(
        render(rows, decision, benchmark),
        encoding="utf-8",
    )
    print(f"wrote {output}")


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def by_key(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def make_row(
    name: str,
    quality: dict[str, Any] | None,
    speed: dict[str, Any] | None,
    *,
    head8: bool = False,
) -> dict[str, Any]:
    minimum_cosine = None
    max_loss = None
    top1 = None
    top5 = None
    if quality is not None:
        records = quality.get("records")
        if "minimum_cosine_similarity" in quality:
            minimum_cosine = quality["minimum_cosine_similarity"]
            max_loss = quality.get("maximum_cosine_loss")
            top1 = quality.get("all_top1_same")
            top5 = quality.get("minimum_top5_overlap")
        elif records:
            minimum_cosine = min(
                row["cosine_similarity"] for row in records
            )
            max_loss = 1.0 - minimum_cosine
            top1 = all(row["top1_same"] for row in records)
            top5 = min(row["top5_overlap"] for row in records)
    speed_value = speed.get("tokens_per_s") if speed else None
    saved = speed.get("saved_weight_bytes") if speed else None
    peak = speed.get("gpu_peak_allocated_bytes") if speed else None
    verdict = "reference" if name == "HF BF16" else "measured"
    if head8:
        verdict = "experimental: long-case top1 changed"
    if quality is None:
        verdict = "speed-only; full logits unavailable"
    return {
        "name": name,
        "speed": speed_value,
        "minimum_cosine": minimum_cosine,
        "maximum_cosine_loss": max_loss,
        "top1": top1,
        "top5": top5,
        "peak_vram": peak,
        "saved_weight_bytes": saved,
        "verdict": verdict,
    }


def decide(
    thin: dict[str, Any],
    q8: dict[str, Any],
) -> str:
    required = (
        thin["speed"],
        thin["minimum_cosine"],
        q8["speed"],
        q8["minimum_cosine"],
    )
    if any(value is None for value in required):
        return "No speed/quality decision: required evidence is missing."
    if (
        thin["speed"] > q8["speed"]
        and thin["minimum_cosine"] >= q8["minimum_cosine"]
        and thin["top1"] is True
    ):
        return (
            "ThinTensor quality FP8 beats this HF bitsandbytes int8/Q8-ish "
            "baseline on the measured speed/quality tradeoff."
        )
    if (
        q8["minimum_cosine"] > thin["minimum_cosine"]
        and q8["speed"] < thin["speed"]
    ):
        return (
            "ThinTensor trades small extra drift for speed and selective "
            "residency savings."
        )
    if (
        q8["minimum_cosine"] > thin["minimum_cosine"]
        and q8["speed"] > thin["speed"]
    ):
        return (
            "Q8 baseline wins; ThinTensor needs more kernel work or a better "
            "precision policy."
        )
    return "The measured modes have mixed results; see the full table."


def render(
    rows: list[dict[str, Any]],
    decision: str,
    benchmark: dict[str, Any],
) -> str:
    lines = [
        "# Q8 vs ThinTensor summary",
        "",
        f"Decision: **{decision}**",
        "",
        "| Mode | Speed tok/s | Min cosine vs HF BF16 | Max cosine loss | Top1 all pass | Min top5 | Peak VRAM | Resident weight saved | Verdict |",
        "|:---|---:|---:|---:|:---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {fmt(row['speed'])} | "
            f"{fmt(row['minimum_cosine'], 6)} | "
            f"{fmt(row['maximum_cosine_loss'], 6)} | "
            f"{fmt(row['top1'])} | {fmt(row['top5'], 3)} | "
            f"{fmt(row['peak_vram'])} | "
            f"{fmt(row['saved_weight_bytes'])} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "HF bitsandbytes int8 is used as a Q8-ish quality baseline; it is "
            "not llama.cpp Q8_0.",
            "",
            "Ollama Q8_0 is reported separately as a GGUF deployment-speed "
            "baseline. Ollama exposes top log-probability shortlists but not "
            "the complete raw logit vector, so a full-vector cosine is not "
            "claimed.",
            "",
            "ThinTensor quality FP8 is selective quantization, not "
            "BF16-equivalent.",
            "",
            f"Ollama note: {benchmark.get('ollama_note', '-')}.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if __name__ == "__main__":
    main()
