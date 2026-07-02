#!/usr/bin/env python3
"""Benchmark HF BF16/int8, ThinTensor BF16/FP8, and optional GGUF Q8."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from thinruntime.gpu_runtime import (  # noqa: E402
    PagedKVCache,
    ThinGpuCausalLMRuntime,
    ThinGpuWeights,
)


MODE_DESCRIPTIONS = {
    "hf_bf16": "HF Transformers BF16 cached greedy decode",
    "hf_bnb_int8": (
        "HF Transformers bitsandbytes int8/Q8-ish cached greedy decode"
    ),
    "thin_bf16": "ThinTensor optimized BF16 causal-KV decode",
    "thin_quality_fp8": (
        "ThinTensor selective scaled FP8 gate/up all + down layers 8:28; "
        "BF16 O projection and LM head"
    ),
    "thin_quality_10_26": (
        "ThinTensor selective scaled FP8 gate/up all + down layers 10:26; "
        "BF16 O projection and LM head"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="SmolLM3-3B")
    parser.add_argument("--archive", default="SmolLM3-3B.thin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--modes",
        default="hf_bf16,hf_bnb_int8,thin_bf16,thin_quality_fp8",
    )
    parser.add_argument(
        "--gguf-dir",
        default="benchmark_models",
    )
    parser.add_argument(
        "--llama-bench",
        default="/home/satvik/Projects/llama.cpp/build/bin/llama-bench",
    )
    parser.add_argument(
        "--ollama-model",
        default="thintensor-smollm3-q8",
    )
    parser.add_argument(
        "--out",
        default="benchmark_results/q8_baseline_bench.json",
    )
    parser.add_argument("--skip-llamacpp", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--worker-mode", choices=sorted(MODE_DESCRIPTIONS))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        print(json.dumps(run_worker(args), indent=2, sort_keys=True))
        return

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(modes) - set(MODE_DESCRIPTIONS)
    if unknown:
        raise ValueError(f"unsupported benchmark modes: {sorted(unknown)}")

    results = []
    for mode in modes:
        print(f"benchmarking {mode}", file=sys.stderr)
        results.append(run_worker_subprocess(args, mode))

    llama_results, llama_note = (
        ([], "llama.cpp baseline disabled by --skip-llamacpp")
        if args.skip_llamacpp
        else run_llamacpp_baselines(args)
    )
    results.extend(llama_results)
    ollama_result, ollama_note = (
        (None, "Ollama baseline disabled by --skip-ollama")
        if args.skip_ollama
        else run_ollama_baseline(args)
    )
    if ollama_result is not None:
        results.append(ollama_result)
    report = {
        "hf_model": args.hf_model,
        "archive": args.archive,
        "prompt_tokens": args.prompt_len,
        "decode_steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch": 1,
        "greedy": True,
        "results": results,
        "llamacpp_note": llama_note,
        "ollama_note": ollama_note,
        "fairness": {
            "same_model_family": "SmolLM3-3B",
            "same_prompt_token_count": True,
            "same_decode_count": True,
            "thin_token_ids_stay_on_gpu_inside_timed_loop": True,
            "bnb_int8_is_not_gguf_q8_0": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = out.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {out} and {markdown}")


def run_worker_subprocess(
    args: argparse.Namespace,
    mode: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--hf-model",
        args.hf_model,
        "--archive",
        args.archive,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--prompt-len",
        str(args.prompt_len),
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
    ]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    if process.returncode:
        return {
            "mode": mode,
            "mode_description": MODE_DESCRIPTIONS[mode],
            "available": False,
            "error": process.stdout[-2000:] + process.stderr[-4000:],
            "command": command,
        }
    result = json.loads(process.stdout)
    result["command"] = command
    return result


@torch.inference_mode()
def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda":
        raise ValueError("Q8 comparison benchmark currently requires CUDA")
    mode = args.worker_mode
    assert mode is not None
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
    )
    prompt_ids = fixed_length_prompt(tokenizer, args.prompt_len).to(device)
    load_start = time.perf_counter()
    if mode.startswith("hf_"):
        result = benchmark_hf(mode, args, prompt_ids, device, dtype)
    else:
        result = benchmark_thin(mode, args, prompt_ids, device, dtype)
    result["load_s"] = time.perf_counter() - load_start - result["decode_s"]
    result["mode"] = mode
    result["mode_description"] = MODE_DESCRIPTIONS[mode]
    result["available"] = True
    return result


def fixed_length_prompt(tokenizer: Any, length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("--prompt-len must be positive")
    text = "ThinTensor Q8 fair benchmark prompt. " * (length + 1)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if int(encoded.shape[1]) < length:
        raise RuntimeError("benchmark prompt construction produced too few tokens")
    result = encoded[:, :length].contiguous()
    roundtrip_text = tokenizer.decode(
        result[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    roundtrip = tokenizer(
        roundtrip_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if int(roundtrip.shape[1]) != length:
        raise RuntimeError(
            "benchmark prompt does not round-trip to the requested token "
            f"length: {int(roundtrip.shape[1])} != {length}"
        )
    return result


def benchmark_hf(
    mode: str,
    args: argparse.Namespace,
    prompt_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if mode == "hf_bnb_int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **kwargs)
    if mode == "hf_bf16":
        model = model.to(device)
    model.eval()
    output = model(input_ids=prompt_ids, use_cache=True)
    past = output.past_key_values
    token = torch.argmax(output.logits[:, -1, :], dim=-1).view(1, 1)
    for _ in range(args.warmup_steps):
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        token = torch.argmax(output.logits[:, -1, :], dim=-1).view(1, 1)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        token = torch.argmax(output.logits[:, -1, :], dim=-1).view(1, 1)
    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - start
    resident = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )
    return timing_result(
        args,
        decode_s,
        device,
        resident_weight_bytes=resident,
        saved_weight_bytes=None,
        final_token=int(token.detach().cpu()),
    )


def benchmark_thin(
    mode: str,
    args: argparse.Namespace,
    prompt_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    weights = ThinGpuWeights(args.archive, device=str(device), dtype=dtype)
    model = weights.manifest["model"]
    head_dim = int(
        model.get("head_dim")
        or int(model["hidden_size"]) // int(model["heads"])
    )

    def cache() -> PagedKVCache:
        return PagedKVCache(
            layers=int(model["layers"]),
            kv_heads=int(model["kv_heads"]),
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            policy="full",
            old_codec="bf16",
        )

    kwargs: dict[str, Any] = {
        "kernel_backend": "triton",
        "attention_mode": "causal_kv",
        "lm_head_backend": "triton",
    }
    if mode in {"thin_quality_fp8", "thin_quality_10_26"}:
        kwargs.update(
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec=(
                "10:26" if mode == "thin_quality_10_26" else "8:28"
            ),
        )
    runtime = ThinGpuCausalLMRuntime(weights, kv_cache=cache(), **kwargs)
    hidden = None
    for position in range(int(prompt_ids.shape[1])):
        hidden = runtime.forward_token(
            prompt_ids[0, position],
            token_index=position,
        )
    assert hidden is not None
    token = runtime.next_token_tensor(hidden)
    position = int(prompt_ids.shape[1])
    for _ in range(args.warmup_steps):
        hidden = runtime.forward_token(token, token_index=position)
        token = runtime.next_token_tensor(hidden)
        position += 1
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        hidden = runtime.forward_token(token, token_index=position)
        token = runtime.next_token_tensor(hidden)
        position += 1
    torch.cuda.synchronize(device)
    decode_s = time.perf_counter() - start
    resident = (
        weights.stats.unique_gpu_weight_bytes
        - runtime.body_fp8_memory_saved_bytes
        + runtime.lm_head_net_extra_bytes
        + runtime.runtime_fusion_extra_bytes
    )
    result = timing_result(
        args,
        decode_s,
        device,
        resident_weight_bytes=resident,
        saved_weight_bytes=runtime.body_fp8_memory_saved_bytes,
        final_token=int(token.detach().cpu()),
    )
    result["body_fp8_memory_saved_bytes"] = (
        runtime.body_fp8_memory_saved_bytes
    )
    result["lm_head_fp8_enabled"] = runtime.lm_head_fp8_enabled
    result["attention_mode"] = runtime.attention_mode
    result["selective_quantization"] = mode != "thin_bf16"
    weights.close()
    return result


def timing_result(
    args: argparse.Namespace,
    decode_s: float,
    device: torch.device,
    *,
    resident_weight_bytes: int | None,
    saved_weight_bytes: int | None,
    final_token: int,
) -> dict[str, Any]:
    return {
        "prompt_tokens": args.prompt_len,
        "warmup_steps": args.warmup_steps,
        "decode_steps": args.steps,
        "decode_s": decode_s,
        "tokens_per_s": args.steps / decode_s,
        "ms_per_token": decode_s * 1000.0 / args.steps,
        "gpu_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "gpu_peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "resident_weight_bytes": resident_weight_bytes,
        "saved_weight_bytes": saved_weight_bytes,
        "final_token_after_timing": final_token,
    }


def run_llamacpp_baselines(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    binary = Path(args.llama_bench)
    gguf_dir = Path(args.gguf_dir)
    candidates = [
        ("llamacpp_f16", "F16/BF16", ("*F16*.gguf", "*BF16*.gguf")),
        ("llamacpp_q8_0", "Q8_0", ("*Q8_0*.gguf", "*q8_0*.gguf")),
        ("llamacpp_q5_k_m", "Q5_K_M", ("*Q5_K_M*.gguf", "*q5_k_m*.gguf")),
        ("llamacpp_q4_k_m", "Q4_K_M", ("*Q4_K_M*.gguf", "*q4_k_m*.gguf")),
    ]
    if not binary.is_file():
        return [], "llama.cpp GGUF baseline not run: missing model or binary"
    results = []
    for mode, quantization, patterns in candidates:
        model = first_matching_file(gguf_dir, patterns)
        if model is None:
            continue
        command = [
            str(binary),
            "-m",
            str(model),
            "-p",
            str(args.prompt_len),
            "-n",
            str(args.steps),
            "-b",
            "1",
            "-ub",
            "1",
            "-ngl",
            "99",
            "-r",
            "3",
            "-o",
            "json",
        ]
        process = subprocess.run(
            command,
            cwd=binary.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode:
            results.append(
                {
                    "mode": mode,
                    "mode_description": (
                        f"llama.cpp GGUF {quantization} deployment baseline"
                    ),
                    "available": False,
                    "error": process.stderr[-4000:],
                    "command": command,
                }
            )
            continue
        result = parse_llama_bench_json(process.stdout, args.steps)
        result.update(
            {
                "mode": mode,
                "mode_description": (
                    f"llama.cpp GGUF {quantization} deployment baseline"
                ),
                "quantization": quantization,
                "model_file": str(model),
                "available": True,
                "command": command,
                "quality_metrics_available": False,
            }
        )
        results.append(result)
    note = (
        "llama.cpp GGUF baselines are speed-only; logits were not extracted"
        if results
        else "llama.cpp GGUF baseline not run: missing model or binary"
    )
    return results, note


def run_ollama_baseline(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
    )
    prompt_ids = fixed_length_prompt(tokenizer, args.prompt_len)
    prompt = tokenizer.decode(
        prompt_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    options = {
        "temperature": 0,
        "seed": 0,
        "num_ctx": max(
            2048,
            args.prompt_len + args.warmup_steps + args.steps + 32,
        ),
    }
    try:
        ollama_generate(
            args.ollama_model,
            prompt,
            options | {"num_predict": args.warmup_steps},
        )
        response = ollama_generate(
            args.ollama_model,
            prompt,
            options | {"num_predict": args.steps},
        )
        ps = ollama_api("/api/ps", None)
    except (urllib.error.URLError, RuntimeError, KeyError) as exc:
        return None, f"Ollama Q8_0 baseline not run: {exc}"
    eval_count = int(response["eval_count"])
    eval_duration_ns = int(response["eval_duration"])
    tokens_per_s = eval_count * 1e9 / eval_duration_ns
    model_info = next(
        (
            row
            for row in ps.get("models", [])
            if row.get("name", "").split(":")[0]
            == args.ollama_model.split(":")[0]
        ),
        {},
    )
    processor = "unknown"
    size_vram = model_info.get("size_vram")
    if size_vram:
        processor = "GPU-resident"
    result = {
        "mode": "ollama_q8_0",
        "mode_description": "Ollama/llama.cpp GGUF Q8_0 deployment baseline",
        "available": True,
        "quantization": "Q8_0",
        "model": args.ollama_model,
        "prompt_tokens": int(response.get("prompt_eval_count", 0)),
        "decode_steps": eval_count,
        "tokens_per_s": tokens_per_s,
        "ms_per_token": 1000.0 / tokens_per_s,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "resident_weight_bytes": size_vram,
        "saved_weight_bytes": None,
        "quality_metrics_available": False,
        "ollama_eval_duration_ns": eval_duration_ns,
        "ollama_total_duration_ns": int(response["total_duration"]),
        "processor": processor,
        "raw_prompt": True,
    }
    ollama_api(
        "/api/generate",
        {
            "model": args.ollama_model,
            "keep_alive": 0,
        },
    )
    return (
        result,
        "Ollama Q8_0 is a speed-only GGUF deployment baseline; logits were not extracted",
    )


def ollama_generate(
    model: str,
    prompt: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    return ollama_api(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        },
    )


def ollama_api(
    path: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    data = (
        None
        if payload is None
        else json.dumps(payload).encode("utf-8")
    )
    request = urllib.request.Request(
        "http://127.0.0.1:11434" + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def first_matching_file(
    directory: Path,
    patterns: tuple[str, ...],
) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_llama_bench_json(text: str, decode_steps: int) -> dict[str, Any]:
    payload = json.loads(text)
    rows = payload if isinstance(payload, list) else [payload]
    decode = [
        row
        for row in rows
        if int(row.get("n_gen", 0)) > 0
    ]
    if not decode:
        raise ValueError("llama-bench JSON has no decode row")
    row = decode[-1]
    tokens_per_s = float(
        row.get("avg_ts")
        or row.get("tokens_per_second")
        or row.get("ts")
    )
    return {
        "prompt_tokens": int(row.get("n_prompt", 0)),
        "decode_steps": int(row.get("n_gen", decode_steps)),
        "tokens_per_s": tokens_per_s,
        "ms_per_token": 1000.0 / tokens_per_s,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "resident_weight_bytes": None,
        "saved_weight_bytes": None,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SmolLM3 Q8 speed baseline",
        "",
        f"- Prompt tokens: `{report['prompt_tokens']}`",
        f"- Decode tokens: `{report['decode_steps']}`",
        f"- Warmup tokens: `{report['warmup_steps']}`",
        f"- llama.cpp: {report['llamacpp_note']}",
        f"- Ollama: {report['ollama_note']}",
        "",
        "| Mode | tok/s | ms/token | Peak allocated | Peak reserved | Resident weights | Saved weights |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        if not row.get("available"):
            lines.append(
                f"| {row['mode_description']} | unavailable | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {row['mode_description']} | {row['tokens_per_s']:.3f} | "
            f"{row['ms_per_token']:.3f} | "
            f"{format_optional(row.get('gpu_peak_allocated_bytes'))} | "
            f"{format_optional(row.get('gpu_peak_reserved_bytes'))} | "
            f"{format_optional(row.get('resident_weight_bytes'))} | "
            f"{format_optional(row.get('saved_weight_bytes'))} |"
        )
    lines.extend(
        [
            "",
            "HF bitsandbytes int8 is a Q8-ish baseline. llama.cpp Q8_0 is a "
            "separate GGUF deployment baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_optional(value: Any) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":
    main()
