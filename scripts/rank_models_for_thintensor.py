#!/usr/bin/env python3
"""Rank HF configurations for ThinTensor batch-1 decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.model_arch import descriptor_from_hf_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--vram", default="8gb")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = [score_config(Path(path), parse_bytes(args.vram)) for path in args.configs]
    result = {
        "vram_bytes": parse_bytes(args.vram),
        "models": sorted(rows, key=lambda row: row["overall_score"], reverse=True),
        "best_head8_targets": names(rows, "head8_score", reverse=True),
        "best_bf16_body_targets": names(rows, "bf16_body_score", reverse=True),
        "best_cpu_offload_targets": names(rows, "cpu_offload_score", reverse=True),
        "worst_fat_mlp_targets": names(rows, "mlp_ratio", reverse=True),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered if args.json or not args.out else f"wrote {args.out}")


def score_config(path: Path, vram_bytes: int) -> dict[str, Any]:
    descriptor = descriptor_from_hf_config(path)
    hidden = descriptor.hidden_size
    intermediate = descriptor.intermediate_size
    layers = descriptor.num_hidden_layers
    vocab = descriptor.vocab_size
    kv_heads = descriptor.num_key_value_heads
    head_dim = descriptor.head_dim

    embedding_params = vocab * hidden
    per_layer_params = (
        hidden * (descriptor.num_attention_heads * head_dim)
        + 2 * hidden * (kv_heads * head_dim)
        + hidden * (descriptor.num_attention_heads * head_dim)
        + 3 * hidden * intermediate
    )
    body_params = per_layer_params * layers
    unique_head_params = 0 if descriptor.tie_word_embeddings else embedding_params
    total_params = embedding_params + body_params + unique_head_params
    bf16_bytes = total_params * 2
    expected_bytes_per_token = body_params * 2 + embedding_params * 2
    head_share = embedding_params / max(1, body_params + embedding_params)
    mlp_ratio = intermediate / hidden
    head8_saved = 0 if descriptor.tie_word_embeddings else embedding_params
    mlp_params = 3 * hidden * intermediate * layers
    mlp_fp8_saved = mlp_params
    fits = bf16_bytes <= vram_bytes

    return {
        "name": path.parent.name if path.name == "config.json" else path.name,
        "config": str(path),
        "model_type": descriptor.model_type,
        "mlp_ratio": mlp_ratio,
        "lm_head_share": head_share,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "layers": layers,
        "kv_heads": kv_heads,
        "expected_bf16_bytes_per_token": expected_bytes_per_token,
        "expected_bf16_resident_bytes": bf16_bytes,
        "likely_head8_benefit_bytes": head8_saved,
        "likely_mlp_fp8_benefit_bytes": mlp_fp8_saved,
        "likely_all_resident_fit_8gb": fits,
        "head8_score": head_share * (0.25 if descriptor.tie_word_embeddings else 1.0),
        "bf16_body_score": (1.0 / max(1.0, mlp_ratio)) * (1.0 if fits else 0.2),
        "cpu_offload_score": bf16_bytes / max(1, vram_bytes),
        "overall_score": (
            (1.0 / max(1.0, mlp_ratio))
            + head_share
            + (1.0 if fits else 0.0)
        ),
    }


def names(rows: list[dict[str, Any]], key: str, reverse: bool) -> list[str]:
    return [
        row["name"]
        for row in sorted(rows, key=lambda row: row[key], reverse=reverse)
    ]


def parse_bytes(value: str) -> int:
    normalized = value.strip().lower()
    units = {
        "gb": 1024**3,
        "gib": 1024**3,
        "mb": 1024**2,
        "mib": 1024**2,
    }
    for suffix, multiplier in units.items():
        if normalized.endswith(suffix):
            return int(float(normalized[: -len(suffix)]) * multiplier)
    return int(normalized)


if __name__ == "__main__":
    main()
