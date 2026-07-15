#!/usr/bin/env python3
"""Compare persisted full-vocabulary vectors on identical checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_llamacpp_q8_logits import full_distribution_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "LABEL=NPZ, LABEL:logprobs=NPZ, or "
            "LABEL@ARRAY_PREFIX=NPZ"
        ),
    )
    parser.add_argument("--reference-prefix", default="step_")
    parser.add_argument("--candidate-prefix", default="step_")
    parser.add_argument("--steps", default="1,8,32")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    reference = np.load(args.reference)
    modes = []
    for spec in args.candidate:
        label_spec, path = spec.split("=", 1)
        logprobs = label_spec.endswith(":logprobs")
        label_spec = label_spec.removesuffix(":logprobs")
        if "@" in label_spec:
            label, candidate_prefix = label_spec.split("@", 1)
        else:
            label = label_spec
            candidate_prefix = args.candidate_prefix
        if not label or not candidate_prefix:
            raise ValueError(f"invalid --candidate spec {spec!r}")
        candidate = np.load(path)
        records = []
        for step in steps:
            ref = torch.from_numpy(reference[f"{args.reference_prefix}{step}"])
            cand = torch.from_numpy(candidate[f"{candidate_prefix}{step}"])
            records.append(
                full_distribution_metrics(
                    ref,
                    cand,
                    step,
                    raw_logits_available=not logprobs,
                )
            )
        cosines = np.asarray(
            [row["centered_logit_cosine"] for row in records],
            dtype=np.float64,
        )
        modes.append(
            {
                "label": label,
                "path": str(Path(path).resolve()),
                "array_prefix": candidate_prefix,
                "values": "log_probabilities" if logprobs else "logits",
                "minimum_centered_logit_cosine": min(
                    row["centered_logit_cosine"] for row in records
                ),
                "mean_centered_logit_cosine": float(cosines.mean()),
                "centered_logit_cosine_percentiles": {
                    "p05": float(np.percentile(cosines, 5)),
                    "p50": float(np.percentile(cosines, 50)),
                    "p95": float(np.percentile(cosines, 95)),
                },
                "minimum_raw_logit_cosine": min(
                    (
                        row["raw_logit_cosine"]
                        for row in records
                        if row["raw_logit_cosine"] is not None
                    ),
                    default=None,
                ),
                "top1_same_all": all(row["top1_same"] for row in records),
                "top1_agreement_rate": sum(
                    bool(row["top1_same"]) for row in records
                )
                / len(records),
                "greedy_token_match_rate_at_checkpoints": sum(
                    bool(row["top1_same"]) for row in records
                )
                / len(records),
                "first_top1_divergence_step": next(
                    (
                        int(row["step"])
                        for row in records
                        if not row["top1_same"]
                    ),
                    None,
                ),
                "minimum_top5_overlap": min(
                    row["top5_overlap"] for row in records
                ),
                "mean_top5_overlap": sum(
                    float(row["top5_overlap"]) for row in records
                )
                / len(records),
                "top5_exact_order_rate": sum(
                    bool(row["top5_exact_order"]) for row in records
                )
                / len(records),
                "maximum_centered_abs_error": max(
                    float(row["maximum_centered_abs_error"])
                    for row in records
                ),
                "mean_centered_abs_error": sum(
                    float(row["mean_centered_abs_error"])
                    for row in records
                )
                / len(records),
                "maximum_jensen_shannon_distance": max(
                    row["jensen_shannon_distance"] for row in records
                ),
                "maximum_total_variation_distance": max(
                    row["total_variation_distance"] for row in records
                ),
                "records": records,
            }
        )
    report = {
        "reference": str(Path(args.reference).resolve()),
        "steps": steps,
        "metric": (
            "full-vocabulary centered cosine; additive log-softmax constants "
            "are removed for llama.cpp log-probabilities"
        ),
        "modes": modes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
