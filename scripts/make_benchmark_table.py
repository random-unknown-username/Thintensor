#!/usr/bin/env python3
"""Merge benchmark, correctness, and external baseline JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    paths = expand_paths(args.inputs)
    documents = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    correctness = index_correctness(documents)
    rows: list[dict[str, Any]] = []
    for path, document in documents:
        if is_thin_benchmark(document):
            rows.append(thin_row(path, document, correctness))
        elif is_external_baseline(document):
            rows.append(external_row(path, document))
        elif "rows" in document and isinstance(document["rows"], list):
            rows.extend(document["rows"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README_benchmark_table.md").write_text(
        render_table(rows), encoding="utf-8"
    )
    safe, unsafe = classify_claims(rows)
    (out_dir / "safe_claims.md").write_text(
        "# Safe claims\n\n" + "\n".join(f"- {claim}" for claim in safe) + "\n",
        encoding="utf-8",
    )
    (out_dir / "unsafe_claims.md").write_text(
        "# Unsafe claims\n\n"
        + "\n".join(f"- {claim}" for claim in unsafe)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote benchmark table and claim files to {out_dir}")


def index_correctness(
    documents: list[tuple[Path, dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for _, document in documents:
        if "hf_equivalent" not in document or "records" not in document:
            continue
        key = (
            str(Path(document.get("archive", "")).resolve()),
            precision_mode(document),
        )
        result[key] = document
    return result


def thin_row(
    path: Path,
    document: dict[str, Any],
    correctness: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    mode = precision_mode(document)
    key = (str(Path(document["archive"]).resolve()), mode)
    gate = correctness.get(key)
    return {
        "source": str(path),
        "backend": "ThinTensor",
        "model": Path(document["archive"]).stem,
        "format": ".thin",
        "precision": mode,
        "residency": document.get("residency"),
        "decode_tokens_per_s": document.get("tokens_per_s"),
        "peak_vram_bytes": document.get("gpu_peak_allocated_bytes"),
        "attention_mode": document.get("attention_mode"),
        "not_hf_equivalent": document.get("not_hf_equivalent", True),
        "correctness_passed": bool(gate and gate.get("hf_equivalent")),
        "selective_quantization": document.get("selective_quantization", False),
        "h2d_bytes_per_token": (
            document.get("gpu_cache", {}).get("h2d_transfer_bytes_per_token")
        ),
    }


def external_row(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(path),
        "backend": document["backend"],
        "model": document["model"],
        "format": document["format"],
        "precision": document["quantization"],
        "residency": f"gpu_layers={document.get('gpu_offload_layers')}",
        "decode_tokens_per_s": document.get("decode_eval_tokens_per_s"),
        "peak_vram_bytes": document.get("peak_vram_bytes"),
        "attention_mode": "causal_kv",
        "not_hf_equivalent": False,
        "correctness_passed": None,
        "selective_quantization": document["quantization"].upper()
        not in {"F16", "BF16"},
        "h2d_bytes_per_token": None,
    }


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark table",
        "",
        "Rates are batch-1 decode rates. Quantized and BF16 rows are distinct precision classes.",
        "",
        "| backend | model | format | precision | residency | tok/s | peak VRAM GiB | causal KV | correctness gate |",
        "|:---|:---|:---|:---|:---|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        peak = row.get("peak_vram_bytes")
        lines.append(
            f"| {row.get('backend')} | {row.get('model')} | {row.get('format')} | "
            f"{row.get('precision')} | {row.get('residency')} | "
            f"{number(row.get('decode_tokens_per_s'))} | "
            f"{number(peak / 1024**3 if peak else None)} | "
            f"{row.get('attention_mode') == 'causal_kv' and not row.get('not_hf_equivalent')} | "
            f"{row.get('correctness_passed')} |"
        )
    return "\n".join(lines) + "\n"


def classify_claims(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    safe: list[str] = []
    unsafe: list[str] = []
    for row in rows:
        label = (
            f"{row.get('backend')} {row.get('model')} {row.get('precision')}"
        )
        if row.get("backend") != "ThinTensor":
            safe.append(
                f"{label} measured {number(row.get('decode_tokens_per_s'))} tok/s "
                "as an external baseline."
            )
            continue
        if (
            row.get("attention_mode") == "causal_kv"
            and not row.get("not_hf_equivalent")
            and row.get("correctness_passed")
        ):
            qualifier = (
                " selective quantization"
                if row.get("selective_quantization")
                else ""
            )
            safe.append(
                f"{label} measured {number(row.get('decode_tokens_per_s'))} tok/s "
                f"in correctness-gated causal decode{qualifier}."
            )
        else:
            unsafe.append(
                f"Do not claim HF-equivalent performance for {label}; causal "
                "attention and correctness evidence are incomplete."
            )

    stream_rows = [
        row
        for row in rows
        if row.get("backend") == "ThinTensor"
        and row.get("residency") == "stream"
        and row.get("h2d_bytes_per_token") is not None
    ]
    if len(stream_rows) >= 2:
        by_vram = sorted(
            stream_rows,
            key=lambda row: row.get("peak_vram_bytes") or 0,
        )
        transfers = [row["h2d_bytes_per_token"] for row in by_vram]
        if all(right <= left for left, right in zip(transfers, transfers[1:])):
            safe.append(
                "The measured CPU-offload ladder reduces H2D transfer per token "
                "as the GPU budget increases."
            )
        else:
            unsafe.append(
                "Do not claim the CPU-offload ladder reduces transfer per token; "
                "the measured series is not monotonic."
            )
    return safe, unsafe


def precision_mode(document: dict[str, Any]) -> str:
    if document.get("mlp_fp8") or (
        document.get("body_fp8_original_bytes", 0)
        and document.get("body_fp8_memory_saved_bytes", 0)
    ):
        return "selective_fp8_body"
    if document.get("lm_head_fp8_enabled") or document.get("lm_head_fp8"):
        return "head8"
    return str(document.get("dtype", "bf16"))


def is_thin_benchmark(document: dict[str, Any]) -> bool:
    return document.get("command") == "run" and "tokens_per_s" in document


def is_external_baseline(document: dict[str, Any]) -> bool:
    return (
        document.get("backend") in {"llamacpp", "ollama"}
        and "decode_eval_tokens_per_s" in document
    )


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = glob.glob(value)
        paths.extend(Path(match) for match in (matches or [value]))
    return paths


def number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
