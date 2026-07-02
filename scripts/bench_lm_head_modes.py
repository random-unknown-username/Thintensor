import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from thinruntime.gpu_runtime import ThinGpuCausalLMRuntime, ThinGpuWeights


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def stat(name, times, batch_size):
    times = sorted(times)
    return {
        "name": name,
        "samples": len(times),
        "calls_per_sample": batch_size,
        "min_ms": min(times),
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "max_ms": max(times),
        "calls_per_s_median": 1000.0 / times[len(times) // 2],
    }


@torch.inference_mode()
def bench(name, fn, *, warmup, samples, batch_size):
    for _ in range(warmup):
        fn()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        for _ in range(batch_size):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / batch_size)

    return stat(name, times, batch_size)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.device != "cuda":
        raise ValueError("LM-head CUDA microbenchmark requires --device cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    weights = ThinGpuWeights(
        args.archive,
        device=args.device,
        dtype=DTYPES[args.dtype],
    )
    runtime = ThinGpuCausalLMRuntime(
        weights,
        kv_cache=None,
        kernel_backend="triton-matvec",
        persistent_buffers=True,
    )

    head = runtime.weights.tensors.get("lm_head.weight")
    if head is None:
        head = runtime.weights.tensor("model.embed_tokens.weight")
    hidden = torch.randn(
        int(head.shape[1]),
        device=args.device,
        dtype=DTYPES[args.dtype],
    )

    rows = int(head.shape[0])
    logits = torch.empty(rows, device=args.device, dtype=hidden.dtype)

    backend = runtime.kernel_backend
    assert backend is not None

    results = []
    bench_args = {
        "warmup": args.warmup,
        "samples": args.samples,
        "batch_size": args.batch_size,
    }

    # 1. Triton matvec only
    results.append(bench(
        "triton_matvec_only",
        lambda: backend.matvec(head, hidden, logits),
        **bench_args,
    ))

    # 2. torch.mv only
    results.append(bench(
        "torch_mv_only",
        lambda: torch.mv(head, hidden, out=logits),
        **bench_args,
    ))

    # 3. torch.matmul only
    results.append(bench(
        "torch_matmul_only",
        lambda: torch.matmul(head, hidden, out=logits),
        **bench_args,
    ))

    # Prepare logits once for argmax-only.
    backend.matvec(head, hidden, logits)

    # 4. argmax only BF16 logits
    results.append(bench(
        "torch_argmax_only_bf16_logits",
        lambda: torch.argmax(logits),
        **bench_args,
    ))

    # 5. argmax only FP32 logits
    logits_f32 = logits.float()
    results.append(bench(
        "torch_argmax_only_fp32_logits",
        lambda: torch.argmax(logits_f32),
        **bench_args,
    ))

    # 6. Triton matvec + torch.argmax
    results.append(bench(
        "triton_matvec_plus_torch_argmax",
        lambda: torch.argmax(backend.matvec(head, hidden, logits)),
        **bench_args,
    ))

    # 7. torch.mv + torch.argmax
    results.append(bench(
        "torch_mv_plus_torch_argmax",
        lambda: torch.argmax(torch.mv(head, hidden, out=logits)),
        **bench_args,
    ))

    # 8. backend default next_token_tensor
    results.append(bench(
        "runtime_next_token_tensor_default",
        lambda: runtime.next_token_tensor(hidden),
        **bench_args,
    ))

    # 9. Triton two-stage, if available
    old_mode = getattr(backend, "argmax_mode", "torch")
    backend.argmax_mode = "triton_two_stage"
    try:
        results.append(bench(
            "triton_two_stage_matvec_argmax",
            lambda: backend.matvec_argmax_tensor(head, hidden),
            **bench_args,
        ))
    except Exception as e:
        results.append({
            "name": "triton_two_stage_matvec_argmax",
            "error": repr(e),
        })
    finally:
        backend.argmax_mode = old_mode

    out = {
        "archive": args.archive,
        "head_shape": list(head.shape),
        "head_dtype": str(head.dtype).replace("torch.", ""),
        "head_stride": list(head.stride()),
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype).replace("torch.", ""),
        "results": results,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    rendered = json.dumps(out, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    weights.close()


if __name__ == "__main__":
    main()
