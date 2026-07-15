#!/usr/bin/env python3
"""Compare full Q8_0 distributions from prebuilt llama.cpp with ThinTensor."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from compare_q8_baseline import (  # noqa: E402
    load_hf,
    run_hf_case,
    run_thin_case,
    select_cases,
)
from stress_validate_thin import build_cases, make_cache  # noqa: E402
from thinruntime.gpu_runtime import (  # noqa: E402
    ThinGpuPagePool,
    ThinGpuCausalLMRuntime,
    ThinGpuWeights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="SmolLM3-3B")
    parser.add_argument("--archive", default="SmolLM3-3B.thin")
    parser.add_argument("--q8-gguf", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument(
        "--llama-device",
        help="Optional llama.cpp device selector such as Vulkan1 or CUDA0",
    )
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--reference-source", choices=["hf", "thin"], default="hf"
    )
    parser.add_argument(
        "--reference-device",
        default="cuda",
        help="Device for the full-vocabulary reference trajectory (for example cpu or cuda)",
    )
    parser.add_argument("--reference-budget", default="5GiB")
    parser.add_argument("--llamacpp-label", default="llamacpp_q8_0")
    parser.add_argument("--only-reference-and-llamacpp", action="store_true")
    parser.add_argument("--cases", default="")
    parser.add_argument(
        "--out",
        default="correctness_results/llamacpp_q8_logits_report.md",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.reference_device)
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
    )
    cases = select_cases(build_cases(tokenizer), args.cases)

    if args.reference_source == "thin":
        references = load_thin_references(
            args.archive, cases, device, dtype, parse_bytes(args.reference_budget)
        )
        reference_label = "thin_bf16_source"
    else:
        print("full-logit comparison: HF BF16 reference", file=sys.stderr)
        hf_model = load_hf(args.hf_model, device, dtype, int8=False)
        references = {}
        for case in cases:
            print(f"  HF {case.name}", file=sys.stderr)
            references[case.name] = run_hf_case(
                hf_model, case, device, teacher_tokens=None
            )
        del hf_model
        recover_cuda()
        reference_label = "hf_bf16"

    mode_reports = [
        exact_reference_report(cases, reference_label),
    ]
    if not args.only_reference_and_llamacpp:
        mode_reports.extend([
        run_thin_mode(
            "thin_bf16",
            args.archive,
            cases,
            references,
            device,
            dtype,
        ),
        run_thin_mode(
            "thin_quality_8_28",
            args.archive,
            cases,
            references,
            device,
            dtype,
        ),
        run_thin_mode(
            "thin_quality_10_26",
            args.archive,
            cases,
            references,
            device,
            dtype,
        ),
        ])

    print("full-logit comparison: llama.cpp Q8_0", file=sys.stderr)
    server = start_server(args)
    try:
        mode_reports.append(
            run_q8_mode(
                args.port,
                cases,
                references,
                vocab_size=int(
                    references[cases[0].name]["logits"][
                        cases[0].checkpoints[0]
                    ].numel()
                ),
                mode_label=args.llamacpp_label,
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
        "archive": args.archive,
        "q8_gguf": args.q8_gguf,
        "case_count": len(cases),
        "metric_definition": {
            "centered_logit_cosine": (
                "cosine after subtracting each vocabulary vector mean; "
                "this removes the unknown additive log-softmax constant and "
                "is directly comparable across HF, ThinTensor, and llama.cpp"
            ),
            "raw_logit_cosine": (
                "reported only when raw logits are available"
            ),
            "q8_distribution": (
                "all vocabulary log-probabilities returned by llama.cpp "
                "n_probs=vocab_size"
            ),
        },
        "modes": mode_reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {out} and {out.with_suffix('.json')}")


def run_thin_mode(
    mode: str,
    archive: str,
    cases: list[Any],
    references: dict[str, dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    print(f"full-logit comparison: {mode}", file=sys.stderr)
    weights = ThinGpuWeights(archive, device=str(device), dtype=dtype)
    model = weights.manifest["model"]
    kwargs: dict[str, Any] = {
        "kernel_backend": "triton",
        "attention_mode": "causal_kv",
        "lm_head_backend": "triton",
    }
    if mode != "thin_bf16":
        kwargs.update(
            gate_up_fp8=True,
            down_proj_fp8=True,
            down_fp8_layer_spec=(
                "10:26" if mode.endswith("10_26") else "8:28"
            ),
        )
    runtime = ThinGpuCausalLMRuntime(
        weights,
        kv_cache=make_cache(model, device, dtype),
        **kwargs,
    )
    records = []
    for case in cases:
        print(f"  {mode} {case.name}", file=sys.stderr)
        reference = references[case.name]
        candidate = run_thin_case(
            runtime,
            model,
            case,
            reference["tokens"],
            device,
            dtype,
        )
        comparisons = [
            full_distribution_metrics(
                reference["logits"][step],
                candidate["logits"][step],
                step,
                raw_logits_available=True,
            )
            for step in case.checkpoints
        ]
        records.append(case_record(case, comparisons))
    report = aggregate(mode, records)
    report["saved_weight_bytes"] = runtime.body_fp8_memory_saved_bytes
    weights.close()
    del runtime
    del weights
    recover_cuda()
    return report


def load_thin_references(
    archive: str,
    cases: list[Any],
    device: torch.device,
    dtype: torch.dtype,
    budget: int,
) -> dict[str, dict[str, Any]]:
    print("full-logit comparison: streamed BF16 source reference", file=sys.stderr)
    weights = ThinGpuPagePool(
        archive,
        device=str(device),
        dtype=dtype,
        vram_budget_bytes=budget,
        prefetch_distance=0,
        cpu_offload=False,
        pin_cpu_pages=False,
    )
    weights.warm_start()
    model = weights.manifest["model"]
    runtime = ThinGpuCausalLMRuntime(
        weights,
        kv_cache=make_cache(model, device, dtype),
        kernel_backend="triton",
        attention_mode="causal_kv",
        attention_backend="torch",
        lm_head_backend="triton",
    )
    references: dict[str, dict[str, Any]] = {}
    try:
        for case in cases:
            print(f"  BF16 source {case.name}", file=sys.stderr)
            references[case.name] = run_thin_reference_case(
                runtime, model, case, device, dtype
            )
    finally:
        weights.close()
        del runtime
        del weights
        recover_cuda()
    return references


def run_thin_reference_case(
    runtime: ThinGpuCausalLMRuntime,
    model: dict[str, Any],
    case: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    runtime.kv_cache = make_cache(model, device, dtype)
    runtime._rope_cos_sin_cache.clear()
    with torch.inference_mode():
        hidden = None
        for position in range(int(case.input_ids.shape[1])):
            hidden = runtime.forward_token(
                case.input_ids[0, position].to(device), token_index=position
            )
        assert hidden is not None
        captured: dict[int, torch.Tensor] = {}
        tokens: list[int] = []
        for step in range(1, case.steps + 1):
            logits = runtime.logits(hidden).float()
            if step in case.checkpoints:
                captured[step] = logits.detach().cpu()
            token = int(torch.argmax(logits))
            tokens.append(token)
            if step < case.steps:
                hidden = runtime.forward_token(
                    torch.tensor(token, device=device, dtype=torch.long),
                    token_index=int(case.input_ids.shape[1]) + step - 1,
                )
    return {"logits": captured, "tokens": tokens}


def parse_bytes(value: str) -> int:
    text = value.strip().upper()
    for suffix, multiplier in (
        ("GIB", 1024**3), ("MIB", 1024**2), ("GB", 1000**3),
        ("MB", 1000**2), ("B", 1),
    ):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def exact_reference_report(cases: list[Any], label: str = "hf_bf16") -> dict[str, Any]:
    records = []
    for case in cases:
        comparisons = [
            {
                "step": step,
                "centered_logit_cosine": 1.0,
                "raw_logit_cosine": 1.0,
                "top1_same": True,
                "top5_overlap": 1.0,
                "jensen_shannon_distance": 0.0,
                "total_variation_distance": 0.0,
                "maximum_centered_abs_error": 0.0,
                "mean_centered_abs_error": 0.0,
            }
            for step in case.checkpoints
        ]
        records.append(case_record(case, comparisons))
    return aggregate(label, records)


def run_q8_mode(
    port: int,
    cases: list[Any],
    references: dict[str, dict[str, Any]],
    *,
    vocab_size: int,
    mode_label: str = "llamacpp_q8_0",
) -> dict[str, Any]:
    records = []
    for case in cases:
        print(f"  llama.cpp Q8_0 {case.name}", file=sys.stderr)
        reference = references[case.name]
        comparisons = []
        base_tokens = [
            int(value) for value in case.input_ids.reshape(-1)
        ]
        for step in case.checkpoints:
            prompt_tokens = (
                base_tokens
                + [int(value) for value in reference["tokens"][: step - 1]]
            )
            q8_log_probabilities = request_full_logprobs(
                port,
                prompt_tokens,
                vocab_size,
            )
            comparisons.append(
                full_distribution_metrics(
                    reference["logits"][step],
                    q8_log_probabilities,
                    step,
                    raw_logits_available=False,
                )
            )
        records.append(case_record(case, comparisons))
    return aggregate(mode_label, records)


def request_full_logprobs(
    port: int,
    prompt_tokens: list[int],
    vocab_size: int,
) -> torch.Tensor:
    payload = {
        "prompt": prompt_tokens,
        "n_predict": 1,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "n_probs": vocab_size,
        "post_sampling_probs": False,
        "stream": False,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    rows = result["completion_probabilities"][0]["top_logprobs"]
    if len(rows) != vocab_size:
        raise RuntimeError(
            f"llama.cpp returned {len(rows)} probabilities, expected {vocab_size}"
        )
    values = torch.empty(vocab_size, dtype=torch.float32)
    seen = torch.zeros(vocab_size, dtype=torch.bool)
    for row in rows:
        token_id = int(row["id"])
        values[token_id] = float(row["logprob"])
        seen[token_id] = True
    if not bool(torch.all(seen)):
        raise RuntimeError("llama.cpp full probability response omitted token IDs")
    return values


def full_distribution_metrics(
    reference_logits: torch.Tensor,
    candidate: torch.Tensor,
    step: int,
    *,
    raw_logits_available: bool,
) -> dict[str, Any]:
    reference_logits = reference_logits.float()
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    if raw_logits_available:
        candidate_logits = candidate.float()
        candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
        raw_cosine = float(
            F.cosine_similarity(
                reference_logits,
                candidate_logits,
                dim=0,
            )
        )
    else:
        candidate_logits = candidate.float()
        candidate_log_probs = candidate_logits
        raw_cosine = None
    reference_centered = reference_logits - reference_logits.mean()
    candidate_centered = candidate_logits - candidate_logits.mean()
    centered_difference = (reference_centered - candidate_centered).abs()
    reference_prob = reference_log_probs.exp()
    candidate_prob = candidate_log_probs.exp()
    candidate_prob.div_(candidate_prob.sum())
    midpoint = (reference_prob + candidate_prob) * 0.5
    epsilon = torch.finfo(torch.float32).tiny
    ref_safe = reference_prob.clamp_min(epsilon)
    candidate_safe = candidate_prob.clamp_min(epsilon)
    midpoint_safe = midpoint.clamp_min(epsilon)
    js = 0.5 * (
        (ref_safe * (ref_safe.log() - midpoint_safe.log())).sum()
        + (
            candidate_safe
            * (candidate_safe.log() - midpoint_safe.log())
        ).sum()
    )
    ref_top = torch.topk(reference_logits, 5).indices
    candidate_top = torch.topk(candidate_logits, 5).indices
    return {
        "step": step,
        "centered_logit_cosine": float(
            F.cosine_similarity(
                reference_centered,
                candidate_centered,
                dim=0,
            )
        ),
        "raw_logit_cosine": raw_cosine,
        "top1_same": int(ref_top[0]) == int(candidate_top[0]),
        "top5_exact_order": [int(value) for value in ref_top]
        == [int(value) for value in candidate_top],
        "top5_overlap": len(
            {int(value) for value in ref_top}
            & {int(value) for value in candidate_top}
        )
        / 5.0,
        "jensen_shannon_distance": math.sqrt(max(0.0, float(js))),
        "total_variation_distance": float(
            0.5 * (reference_prob - candidate_prob).abs().sum()
        ),
        "maximum_centered_abs_error": float(centered_difference.max()),
        "mean_centered_abs_error": float(centered_difference.mean()),
    }


def case_record(
    case: Any,
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case": case.name,
        "category": case.category,
        "prompt_tokens": int(case.input_ids.shape[1]),
        "comparisons": comparisons,
        "minimum_centered_logit_cosine": min(
            row["centered_logit_cosine"] for row in comparisons
        ),
        "minimum_raw_logit_cosine": min(
            (
                row["raw_logit_cosine"]
                for row in comparisons
                if row["raw_logit_cosine"] is not None
            ),
            default=None,
        ),
        "all_top1_same": all(row["top1_same"] for row in comparisons),
        "minimum_top5_overlap": min(
            row["top5_overlap"] for row in comparisons
        ),
        "maximum_jensen_shannon_distance": max(
            row["jensen_shannon_distance"] for row in comparisons
        ),
        "maximum_total_variation_distance": max(
            row["total_variation_distance"] for row in comparisons
        ),
    }


def aggregate(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": mode,
        "minimum_centered_logit_cosine": min(
            row["minimum_centered_logit_cosine"] for row in records
        ),
        "minimum_raw_logit_cosine": min(
            (
                row["minimum_raw_logit_cosine"]
                for row in records
                if row["minimum_raw_logit_cosine"] is not None
            ),
            default=None,
        ),
        "all_top1_same": all(row["all_top1_same"] for row in records),
        "minimum_top5_overlap": min(
            row["minimum_top5_overlap"] for row in records
        ),
        "maximum_jensen_shannon_distance": max(
            row["maximum_jensen_shannon_distance"] for row in records
        ),
        "maximum_total_variation_distance": max(
            row["maximum_total_variation_distance"] for row in records
        ),
        "records": records,
    }


def start_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = dict(os.environ)
    cuda_libraries = [
        "/tmp/llama-cuda-prebuilt",
        str(
            Path(torch.__file__).parent
            / "lib"
        ),
        "/home/satvik/.local/lib/python3.14/site-packages/nvidia/cuda_runtime/lib",
        "/home/satvik/.local/lib/python3.14/site-packages/nvidia/cublas/lib",
        "/home/satvik/.local/lib/python3.14/site-packages/nvidia/nccl/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_libraries)
    command = [
        args.llama_server,
        "-m",
        args.q8_gguf,
        "-ngl",
        "auto",
        "-fitt",
        "512",
        "-fitc",
        "2048",
        "-c",
        "2048",
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-webui",
    ]
    if args.llama_device:
        command.extend(["-dev", args.llama_device])
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


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full-logit llama.cpp Q8_0 comparison",
        "",
        "All modes use identical HF token IDs and HF BF16 teacher trajectories.",
        "",
        "| Mode | Min centered-logit cosine | Min raw-logit cosine | Top1 all pass | Min top5 | Max JS distance | Max TV |",
        "|:---|---:|---:|:---:|---:|---:|---:|",
    ]
    for mode in report["modes"]:
        lines.append(
            f"| {mode['mode']} | "
            f"{mode['minimum_centered_logit_cosine']:.6f} | "
            f"{fmt(mode['minimum_raw_logit_cosine'])} | "
            f"{mode['all_top1_same']} | "
            f"{mode['minimum_top5_overlap']:.3f} | "
            f"{mode['maximum_jensen_shannon_distance']:.6f} | "
            f"{mode['maximum_total_variation_distance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "llama.cpp returns complete log-probabilities, not raw logits. "
            "Centered-logit cosine removes the unknown additive log-softmax "
            "constant and is therefore the direct cross-runtime cosine.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def recover_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
