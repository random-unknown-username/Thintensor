#!/usr/bin/env python3
"""Run or parse fair batch-1 llama.cpp and Ollama decode baselines."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["llamacpp", "ollama"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", default="GGUF")
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--binary", default="llama-cli")
    parser.add_argument("--ollama-binary", default="ollama")
    parser.add_argument("--parse-file")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.parse_file:
        text = Path(args.parse_file).read_text(encoding="utf-8")
        command = ["parse", args.parse_file]
    else:
        command = build_command(args)
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        text = process.stdout
        if process.returncode:
            raise SystemExit(
                f"baseline command failed ({process.returncode}):\n{text}"
            )

    rates = parse_rates(text, args.backend)
    result: dict[str, Any] = {
        "backend": args.backend,
        "model": args.model,
        "format": args.format,
        "quantization": args.quantization,
        "gpu_offload_layers": args.gpu_layers,
        "prompt": args.prompt,
        "generated_tokens": args.tokens,
        "batch": 1,
        "greedy_requested": True,
        "prompt_eval_tokens_per_s": rates.get("prompt_eval_tokens_per_s"),
        "decode_eval_tokens_per_s": rates.get("decode_eval_tokens_per_s"),
        "command": shlex.join(command),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    if args.json or not args.out:
        print(rendered)


def build_command(args: argparse.Namespace) -> list[str]:
    if args.backend == "ollama":
        return [
            args.ollama_binary,
            "run",
            args.model,
            args.prompt,
            "--verbose",
        ]
    return [
        args.binary,
        "-m",
        args.model,
        "-p",
        args.prompt,
        "-n",
        str(args.tokens),
        "-b",
        "1",
        "-ub",
        "1",
        "-ngl",
        str(args.gpu_layers),
        "--temp",
        "0",
    ]


def parse_rates(text: str, backend: str) -> dict[str, float]:
    if backend == "ollama":
        prompt = match_float(text, r"prompt eval rate:\s*([\d.]+)\s+tokens/s")
        decode = match_float(text, r"eval rate:\s*([\d.]+)\s+tokens/s")
    else:
        prompt = match_float(
            text,
            r"prompt eval time\s*=.*?\(\s*([\d.]+)\s+tokens per second\)",
        )
        decode = match_float(
            text,
            r"eval time\s*=.*?\(\s*([\d.]+)\s+tokens per second\)",
        )
    if decode is None:
        raise ValueError("could not parse decode evaluation rate")
    return {
        "prompt_eval_tokens_per_s": prompt,
        "decode_eval_tokens_per_s": decode,
    }


def match_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return float(match.group(1)) if match else None


if __name__ == "__main__":
    main()
