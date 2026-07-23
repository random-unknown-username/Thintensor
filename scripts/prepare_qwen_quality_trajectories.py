#!/usr/bin/env python3
"""Create deterministic full-logit trajectories for Qwen text validation.

The sequential reference runner consumes token IDs rather than tokenizer
state.  Keeping the prompt and a fixed teacher continuation together makes
each report reproducible after source SafeTensor shards have been consumed by
the verified streaming converter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


CASES = {
    "general_explanation": {
        "prompt": (
            "Explain, in clear everyday language, why a metal spoon feels colder "
            "than a wooden spoon in the same room. Include one practical example."
        ),
        "continuation": (
            " Heat moves through metal more readily than through wood, so the "
            "metal draws energy from your hand faster even though both objects "
            "started at the same temperature. This is the key physical mechanism."
        ),
    },
    "coding": {
        "prompt": (
            "Write a small Python function that returns the first non-repeating "
            "character in a string. State its time complexity and handle an empty input."
        ),
        "continuation": (
            " A dictionary can count each character in one pass, then a second "
            "pass over the original string preserves order and returns the first "
            "character with count one. This uses linear time and linear extra space."
        ),
    },
    "mathematics": {
        "prompt": (
            "A rectangle has perimeter 30 and area 56. Find its side lengths, "
            "showing the algebraic steps instead of guessing."
        ),
        "continuation": (
            " Let the sides be x and y. Then x plus y is fifteen and xy is "
            "fifty-six, so the quadratic t squared minus fifteen t plus fifty-six "
            "equals zero has roots seven and eight. Both values satisfy the area."
        ),
    },
    "long_continuation": {
        "prompt": (
            "Continue this technical note in a precise but readable style: A "
            "benchmark is useful only when it measures the same workload that users "
            "will run. For autoregressive inference, this means that"
        ),
        "continuation": (
            " prompt processing must be separated from the steady decode loop, "
            "the model must execute every language layer for each accepted token, "
            "and warmup must finish kernel compilation before latency statistics "
            "are collected. Repeating a fixed protocol in fresh processes also "
            "prevents allocator state from being mistaken for an optimization. "
            "These details make comparisons reproducible."
        ),
    },
    "instruction_style": {
        "prompt": (
            "Give a concise numbered checklist for safely cleaning up a large "
            "download after its benchmark evidence has been saved. Do not delete "
            "unrelated cache entries."
        ),
        "continuation": (
            " First identify the exact model and temporary paths. Then stop every "
            "process using them, save the result files and checksums, record each "
            "deletion, remove only those paths, and verify the newly available disk space. "
            "This preserves a clear audit trail."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--steps", default="1,8,32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    if not steps or min(steps) < 1:
        raise ValueError("--steps must contain positive decode positions")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, case in CASES.items():
        prompt_ids = tokenizer.encode(case["prompt"], add_special_tokens=True)
        teacher_ids = tokenizer.encode(case["continuation"], add_special_tokens=False)
        if len(teacher_ids) < max(steps):
            raise ValueError(
                f"{name} continuation has {len(teacher_ids)} tokens; needs {max(steps)}"
            )
        payload = {
            "schema": "thintensor.quality_trajectory.v1",
            "name": name,
            "prompt": case["prompt"],
            "continuation": case["continuation"],
            "prompt_token_ids": prompt_ids,
            "teacher_token_ids": teacher_ids,
            "steps": steps,
        }
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(prompt_ids)} prompt / {len(teacher_ids)} teacher tokens)")


if __name__ == "__main__":
    main()
