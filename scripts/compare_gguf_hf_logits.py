#!/usr/bin/env python3
"""Compare a GGUF llama.cpp distribution against HF reference logits."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from compare_hf_thin_logits import (  # noqa: E402
    parse_dtype,
    parse_positive_csv,
    prompt_ids,
    run_hf_trajectory,
)
from compare_llamacpp_q8_logits import (  # noqa: E402
    aggregate,
    case_record,
    full_distribution_metrics,
    request_full_logprobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--prefill-lens", default="1")
    parser.add_argument("--steps", default="1")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", default="correctness_results/gguf_vs_hf.json")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    prefill_lens = parse_positive_csv(args.prefill_lens, "prefill-lens")
    requested_steps = parse_positive_csv(args.steps, "steps")
    max_steps = max(requested_steps)

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=args.trust_remote_code,
    )
    input_ids_by_length = {
        length: prompt_ids(tokenizer, args.prompt, length)
        for length in prefill_lens
    }

    print(f"loading HF model {args.hf_model}", file=sys.stderr)
    hf_model, hf_device = load_hf_model(args, dtype, device)
    hf_runs: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        for prefill_len, input_ids in input_ids_by_length.items():
            print(f"HF reference prefill={prefill_len}", file=sys.stderr)
            hf_runs[prefill_len] = run_hf_trajectory(
                hf_model,
                input_ids.to(hf_device),
                requested_steps,
                max_steps,
            )
    vocab_size = int(hf_runs[prefill_lens[0]]["logits"][requested_steps[0]].numel())
    del hf_model
    recover_cuda()

    server = start_llama_server(args)
    try:
        records = []
        for prefill_len, input_ids in input_ids_by_length.items():
            base_tokens = [int(value) for value in input_ids.reshape(-1)]
            hf_run = hf_runs[prefill_len]
            comparisons = []
            for step in requested_steps:
                print(
                    f"llama.cpp GGUF prefill={prefill_len} step={step}",
                    file=sys.stderr,
                )
                prompt_tokens = (
                    base_tokens
                    + [int(value) for value in hf_run["tokens"][: step - 1]]
                )
                gguf_log_probs = request_full_logprobs(
                    args.port,
                    prompt_tokens,
                    vocab_size,
                )
                comparisons.append(
                    full_distribution_metrics(
                        hf_run["logits"][step],
                        gguf_log_probs,
                        step,
                        raw_logits_available=False,
                    )
                )
            records.append(
                case_record(
                    SimpleCase(
                        name=f"prefill_{prefill_len}",
                        category="prompt_length",
                        input_ids=input_ids,
                    ),
                    comparisons,
                )
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    report = {
        "hf_model": args.hf_model,
        "gguf": args.gguf,
        "llama_server": args.llama_server,
        "prefill_lens": prefill_lens,
        "steps": requested_steps,
        "metric_definition": {
            "centered_logit_cosine": (
                "cosine after subtracting the per-vocabulary mean; llama.cpp "
                "returns log-probabilities, so this removes the additive "
                "normalization constant"
            ),
            "raw_logit_cosine": None,
        },
        "mode": aggregate("llamacpp_gguf", records),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {out} and {out.with_suffix('.md')}")


class SimpleCase:
    def __init__(self, name: str, category: str, input_ids: torch.Tensor) -> None:
        self.name = name
        self.category = category
        self.input_ids = input_ids


def load_hf_model(
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Any, torch.device]:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            device_map="auto" if device.type == "cuda" else None,
        )
        model.eval()
        if device.type == "cuda":
            with torch.inference_mode():
                model(torch.zeros((1, 1), dtype=torch.long, device=device))
            return model, device
        return model.to(device), device
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if device.type != "cuda":
            raise
        print(
            f"HF CUDA load failed ({exc}); falling back to CPU reference",
            file=sys.stderr,
        )
        recover_cuda()
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            device_map={"": "cpu"},
        )
        model.eval()
        return model, torch.device("cpu")


def start_llama_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    server = Path(args.llama_server)
    env = dict(os.environ)
    library_dirs = [
        str(server.parent),
        str(Path(torch.__file__).parent / "lib"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(
        [*library_dirs, env.get("LD_LIBRARY_PATH", "")]
    )
    command = [
        str(server),
        "-m",
        args.gguf,
        "-ngl",
        str(args.gpu_layers),
        "-c",
        str(args.ctx_size),
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-webui",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup: {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    return process
        except Exception:
            time.sleep(0.5)
    process.terminate()
    raise TimeoutError("llama-server did not become healthy")


def recover_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def render_markdown(report: dict[str, Any]) -> str:
    mode = report["mode"]
    lines = [
        "# GGUF vs HF full-logprob comparison",
        "",
        f"- HF model: `{report['hf_model']}`",
        f"- GGUF: `{report['gguf']}`",
        f"- Min centered-logit cosine: `{mode['minimum_centered_logit_cosine']:.6f}`",
        f"- Top1 all pass: `{mode['all_top1_same']}`",
        f"- Min top5 overlap: `{mode['minimum_top5_overlap']:.3f}`",
        f"- Max JS distance: `{mode['maximum_jensen_shannon_distance']:.6f}`",
        f"- Max total variation: `{mode['maximum_total_variation_distance']:.6f}`",
        "",
        "| case | min centered cosine | top1 all pass | min top5 | max JS | max TV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in mode["records"]:
        lines.append(
            f"| {row['case']} | {row['minimum_centered_logit_cosine']:.6f} | "
            f"{row['all_top1_same']} | {row['minimum_top5_overlap']:.3f} | "
            f"{row['maximum_jensen_shannon_distance']:.6f} | "
            f"{row['maximum_total_variation_distance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "llama.cpp returns full vocabulary log-probabilities, not raw logits; "
            "centered-logit cosine removes the additive log-softmax constant.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
