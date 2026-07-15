import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    uses_cuda = "cuda" in args.device
    if uses_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if "cuda" in args.device:
        load_kwargs["device_map"] = "auto"
    attention_implementation = "sdpa"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=attention_implementation,
            **load_kwargs,
        )
    except ValueError as exc:
        if "does not support an attention implementation" not in str(exc):
            raise
        attention_implementation = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation=attention_implementation,
            **load_kwargs,
        )
    if "cuda" not in args.device:
        model.to(args.device)
    model.eval()

    if uses_cuda:
        torch.cuda.synchronize()
    load_s = time.perf_counter() - load_t0

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)

    # Prefill prompt.
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    token = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)

    # Warmup decode, GPU token stays GPU.
    for _ in range(args.warmup_steps):
        out = model(input_ids=token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        token = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)

    if uses_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.steps):
        out = model(input_ids=token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        token = torch.argmax(out.logits[:, -1, :], dim=-1).view(1, 1)

    if uses_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # After timing only.
    final_token = int(token.item())

    dt = t1 - t0
    print(json.dumps({
        "runtime": "hf_transformers",
        "model": args.model,
        "dtype": args.dtype,
        "steps": args.steps,
        "prompt": args.prompt,
        "load_s": load_s,
        "decode_s": dt,
        "tokens_per_s": args.steps / dt,
        "ms_per_token": 1000.0 * dt / args.steps,
        "final_token": final_token,
        "attention_implementation": attention_implementation,
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if uses_cuda else None
        ),
        "gpu_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if uses_cuda else None
        ),
        "note": "Custom greedy decode loop. No token .item() inside timed loop.",
    }, indent=2))


if __name__ == "__main__":
    main()
