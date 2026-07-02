import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import torch

from thinruntime.gpu_runtime import ThinGpuWeights, ThinGpuQwenRuntime, PagedKVCache


ARCHIVE = "/tmp/thintensor-first-run/qwen3-0.6b.thin"
STEPS = 500
WARMUP = 10
DEVICE = "cuda"
DTYPE = torch.bfloat16


@torch.inference_mode()
def main():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_t0 = time.perf_counter()

    weights = ThinGpuWeights(
        ARCHIVE,
        device=DEVICE,
        dtype=DTYPE,
    )

    # IMPORTANT: BF16 exact KV default. No q4.
    kv = PagedKVCache(
        layers=int(weights.manifest["model"]["layers"]),
        kv_heads=int(weights.manifest["model"]["kv_heads"]),
        head_dim=128,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        block_size=16,
        recent_window=256,
        old_codec="bf16",
        policy="sink_recent_attention",
    )

    runtime = ThinGpuQwenRuntime(
        weights,
        kv_cache=kv,
        kernel_backend="triton-matvec",
        persistent_buffers=True,
    )

    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_t0

    # GPU scalar token. No .item() in loop.
    token = torch.zeros((), device=DEVICE, dtype=torch.long)

    # Warmup: no timing.
    for i in range(WARMUP):
        hidden = runtime.forward_token(token, token_index=i)
        token = runtime.next_token_tensor(hidden)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(STEPS):
        hidden = runtime.forward_token(token, token_index=WARMUP + i)
        token = runtime.next_token_tensor(hidden)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # OK after timing.
    final_token = int(token.item())

    out = {
        "archive": ARCHIVE,
        "steps": STEPS,
        "warmup": WARMUP,
        "load_s": load_s,
        "decode_s": t1 - t0,
        "tokens_per_s": STEPS / (t1 - t0),
        "final_token": final_token,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "kv_cache": kv.telemetry(),
        "kernel_backend": runtime.kernel_backend_name,
        "note": "No .item(), no topk, no CPU token sync inside timed loop.",
    }

    print(json.dumps(out, indent=2))

    weights.close()


if __name__ == "__main__":
    main()
