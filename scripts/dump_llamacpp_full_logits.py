#!/usr/bin/env python3
"""Persist teacher-forced llama.cpp full-vocabulary log-probabilities.

The saved NPZ keeps the Q4 evidence usable after its disposable GGUF is
deleted.  A fixed textual continuation supplies engine-independent teacher
tokens, so HF, ThinTensor, and llama.cpp can later be evaluated at identical
positions without requiring any engine's free-running trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from compare_llamacpp_q8_logits import request_full_logprobs


DEFAULT_PROMPT = "Question: What is the capital of France?\nAnswer:"
DEFAULT_TEACHER = (
    " Paris is the capital and largest city of France. It stands on the River "
    "Seine and is known for the Eiffel Tower, the Louvre, and its long history "
    "as a center of art, science, and culture."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--device", default="Vulkan1")
    parser.add_argument(
        "--gpu-layers",
        default="auto",
        help="Layer placement limit accepted by llama.cpp: auto, all, or an integer",
    )
    parser.add_argument("--fit-margin-mib", type=int, default=512)
    parser.add_argument("--fit-context", type=int, default=2048)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Prompt-evaluation batch for correctness capture (not a speed claim)",
    )
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--teacher-text", default=DEFAULT_TEACHER)
    parser.add_argument("--steps", default="1,8,32")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--gguf-repo")
    parser.add_argument("--gguf-revision")
    parser.add_argument("--gguf-sha256")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = sorted({int(value) for value in args.steps.split(",")})
    if not steps or steps[0] <= 0:
        raise ValueError("--steps must contain positive integers")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out.with_suffix(".npz")

    command = server_command(args)
    server = start_server(command, args.port)
    try:
        prompt_tokens = tokenize(args.port, args.prompt, add_special=True)
        teacher_tokens = tokenize(args.port, args.teacher_text, add_special=False)
        if len(teacher_tokens) < max(steps) - 1:
            raise RuntimeError(
                f"teacher text produced {len(teacher_tokens)} tokens; "
                f"need at least {max(steps) - 1}"
            )
        arrays: dict[str, np.ndarray] = {}
        checkpoints: list[dict[str, Any]] = []
        for step in steps:
            context = prompt_tokens + teacher_tokens[: step - 1]
            values = request_full_logprobs(args.port, context, args.vocab_size)
            array = values.numpy().astype(np.float32, copy=False)
            arrays[f"step_{step}"] = array
            top = np.argpartition(array, -5)[-5:]
            top = top[np.argsort(array[top])[::-1]]
            checkpoints.append(
                {
                    "step": step,
                    "context_token_count": len(context),
                    "next_teacher_token_id": (
                        teacher_tokens[step - 1]
                        if step - 1 < len(teacher_tokens)
                        else None
                    ),
                    "top5_token_ids": [int(value) for value in top],
                    "top5_logprobabilities": [float(array[value]) for value in top],
                    "all_values_finite": bool(np.isfinite(array).all()),
                }
            )
        np.savez_compressed(arrays_path, **arrays)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    report = {
        "schema": "thintensor.llamacpp_full_logprobs.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gguf": str(Path(args.gguf).resolve()),
        "gguf_repo": args.gguf_repo,
        "gguf_revision": args.gguf_revision,
        "gguf_sha256": args.gguf_sha256,
        "llama_server": str(Path(args.llama_server).resolve()),
        "server_command": command,
        "device": args.device,
        "prompt": args.prompt,
        "prompt_token_ids": prompt_tokens,
        "teacher_text": args.teacher_text,
        "teacher_token_ids": teacher_tokens,
        "steps": steps,
        "vocab_size": args.vocab_size,
        "stored_values": "llama.cpp full-vocabulary pre-sampling log-probabilities",
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "checkpoints": checkpoints,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} and {arrays_path}")


def server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(Path(args.llama_server).resolve()),
        "-m",
        str(Path(args.gguf).resolve()),
        "-ngl",
        str(args.gpu_layers),
        "-dev",
        args.device,
        "-fitt",
        str(args.fit_margin_mib),
        "-fitc",
        str(args.fit_context),
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.batch_size),
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-webui",
    ]


def start_server(command: list[str], port: int) -> subprocess.Popen[str]:
    server = Path(command[0])
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(server.parent), env.get("LD_LIBRARY_PATH", "")]
    )
    process = subprocess.Popen(command, env=env, text=True)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup: {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return process
        except Exception:
            time.sleep(0.5)
    process.terminate()
    raise TimeoutError("llama-server did not become healthy")


def tokenize(port: int, content: str, *, add_special: bool) -> list[int]:
    payload = {
        "content": content,
        "add_special": add_special,
        "parse_special": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(value, int) for value in tokens):
        raise RuntimeError("llama.cpp /tokenize returned an unexpected token list")
    return [int(value) for value in tokens]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
