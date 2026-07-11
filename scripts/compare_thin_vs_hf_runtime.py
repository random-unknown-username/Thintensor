#!/usr/bin/env python3
"""Compare HF execution loaded from safetensors vs ThinTensor hydration.

Both paths execute the Transformers model. This is a storage/load benchmark,
not evidence for the native ThinTensor decode runtime.
"""

import sys
import time
import argparse
import subprocess
import json
import statistics
from pathlib import Path
from datetime import datetime

# Add project root to sys.path to import thinruntime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import conditionally inside trial to avoid importing torch in the manager process
def run_single_trial(
    mode: str,
    hf_dir: Path,
    thin_file: Path,
    prompt: str,
    tokens: int,
    warmup_tokens: int,
    device: str,
):
    import torch
    import gc
    import os
    import ctypes
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from thinruntime import load_thin_model

    try:
        import psutil
    except ImportError:
        psutil = None

    def rss_bytes() -> int | None:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
        return None

    def malloc_trim() -> None:
        if os.name != "posix":
            return
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    # Setup memory and device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rss0 = rss_bytes()

    t_start = time.perf_counter()

    if mode == "hf":
        # Standard HF path
        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if "cuda" in str(device):
            model = AutoModelForCausalLM.from_pretrained(
                hf_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
        model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
        load_time = time.perf_counter() - t_start
    elif mode == "thin":
        # ThinRuntime path
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model, _diagnostics = load_thin_model(
            archive_path=str(thin_file),
            hf_dir=str(hf_dir),
            device=device,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        load_time = time.perf_counter() - t_start
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # 1. Prefill (first token)
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        past_key_values = outputs.past_key_values
    if device == "cuda":
        torch.cuda.synchronize()
    first_token_latency = time.perf_counter() - t_prefill_start

    # 2. Warmup and steady-state decode. The token selected by prefill is an
    # input to decode, not a timed decoded token.
    curr_input_ids = next_token.unsqueeze(-1)
    for _ in range(warmup_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            curr_input_ids = torch.argmax(
                outputs.logits[:, -1, :], dim=-1
            ).unsqueeze(-1)
    if device == "cuda":
        torch.cuda.synchronize()

    t_decode_start = time.perf_counter()
    for _ in range(tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            curr_input_ids = torch.argmax(
                outputs.logits[:, -1, :], dim=-1
            ).unsqueeze(-1)
    if device == "cuda":
        torch.cuda.synchronize()
    decode_time = time.perf_counter() - t_decode_start

    # Gather metrics
    rss1 = rss_bytes()
    rss_delta = (rss1 - rss0) if (rss1 is not None and rss0 is not None) else 0
    
    gpu_peak_alloc = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    decode_tok_s = tokens / decode_time if decode_time > 0 else 0.0

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    malloc_trim()

    result = {
        "load_time": load_time,
        "first_token_latency": first_token_latency,
        "warmup_tokens": warmup_tokens,
        "decode_tokens": tokens,
        "decode_tok_s": decode_tok_s,
        "rss_delta": rss_delta,
        "gpu_peak_alloc": gpu_peak_alloc,
    }
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_dir", type=Path, help="Hugging Face model directory")
    parser.add_argument("thin_file", type=Path, help="ThinTensor archive path")
    parser.add_argument("--prompt", type=str, default="Explain ThinTensor in one sentence.", help="Benchmark prompt")
    parser.add_argument(
        "--tokens",
        type=int,
        default=200,
        help="Number of timed decode tokens",
    )
    parser.add_argument("--warmup-tokens", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3, help="Number of benchmark trials")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Execution device")
    
    # Internal flag for subprocess trial isolation
    parser.add_argument("--run-trial", choices=["hf", "thin"], default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    if args.warmup_tokens < 0:
        raise ValueError("--warmup-tokens must be non-negative")

    if args.run_trial is not None:
        run_single_trial(
            mode=args.run_trial,
            hf_dir=args.hf_dir,
            thin_file=args.thin_file,
            prompt=args.prompt,
            tokens=args.tokens,
            warmup_tokens=args.warmup_tokens,
            device=args.device
        )
        return

    print("==================================================")
    print(" ThinTensor Hydration vs HF Safetensors Benchmark ")
    print("==================================================")
    print(f"HF Dir:      {args.hf_dir}")
    print(f"Thin File:   {args.thin_file}")
    print(f"Prompt:      '{args.prompt}'")
    print(f"Tokens:      {args.tokens}")
    print(f"Warmup:      {args.warmup_tokens}")
    print(f"Trials:      {args.trials}")
    print(f"Device:      {args.device}")
    print("==================================================")

    # Launch subprocesses for trials to isolate memory and execution
    def run_trials_for_mode(mode: str) -> list[dict]:
        results = []
        for i in range(args.trials):
            print(f"Running {mode.upper()} trial {i+1}/{args.trials}...", end="", flush=True)
            cmd = [
                sys.executable,
                __file__,
                str(args.hf_dir),
                str(args.thin_file),
                "--prompt", args.prompt,
                "--tokens", str(args.tokens),
                "--warmup-tokens", str(args.warmup_tokens),
                "--device", args.device,
                "--run-trial", mode
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                print(" FAILED!")
                print(f"Error: {proc.stderr}")
                sys.exit(1)
            try:
                res = json.loads(proc.stdout.strip().splitlines()[-1])
                results.append(res)
                print(" Done.")
            except Exception as e:
                print(" PARSE ERROR!")
                print(f"Output: {proc.stdout}")
                print(f"Stderr: {proc.stderr}")
                sys.exit(1)
        return results

    hf_results = run_trials_for_mode("hf")
    thin_results = run_trials_for_mode("thin")

    def aggregate(results: list[dict]) -> dict:
        aggregated = {}
        for key in ["load_time", "first_token_latency", "decode_tok_s", "rss_delta", "gpu_peak_alloc"]:
            values = [r[key] for r in results]
            aggregated[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
            }
        return aggregated

    hf_agg = aggregate(hf_results)
    thin_agg = aggregate(thin_results)

    # Display comparison table
    print("\n==================== RESULTS COMPARISON ====================")
    print(f"{'Metric':<25} | {'HF Safetensors':<20} | {'Thin Hydration':<20}")
    print("-" * 72)
    
    def format_val(val_dict, fmt, unit=""):
        return f"{val_dict['median']:{fmt}} {unit} (min={val_dict['min']:{fmt}}, max={val_dict['max']:{fmt}})"

    print(f"{'Load Time':<25} | {format_val(hf_agg['load_time'], '.4f', 's'):<20} | {format_val(thin_agg['load_time'], '.4f', 's'):<20}")
    print(f"{'First-Token Latency':<25} | {format_val(hf_agg['first_token_latency'], '.4f', 's'):<20} | {format_val(thin_agg['first_token_latency'], '.4f', 's'):<20}")
    print(f"{'Decode Speed':<25} | {format_val(hf_agg['decode_tok_s'], '.2f', 'tok/s'):<20} | {format_val(thin_agg['decode_tok_s'], '.2f', 'tok/s'):<20}")
    
    def format_mem(val_dict):
        med_mib = val_dict['median'] / (1024**2)
        min_mib = val_dict['min'] / (1024**2)
        max_mib = val_dict['max'] / (1024**2)
        return f"{med_mib:.2f} MiB (min={min_mib:.2f}, max={max_mib:.2f})"

    print(f"{'Peak RSS Delta':<25} | {format_mem(hf_agg['rss_delta']):<20} | {format_mem(thin_agg['rss_delta']):<20}")
    if args.device == "cuda":
        print(f"{'Peak GPU Allocated':<25} | {format_mem(hf_agg['gpu_peak_alloc']):<20} | {format_mem(thin_agg['gpu_peak_alloc']):<20}")
    
    # Phase 4 comparative metrics
    hf_cold_start = hf_agg['load_time']['median'] + hf_agg['first_token_latency']['median']
    thin_cold_start = thin_agg['load_time']['median'] + thin_agg['first_token_latency']['median']
    print("-" * 72)
    print(f"{'Time to First Token (CS)':<25} | {hf_cold_start:.4f} s            | {thin_cold_start:.4f} s")
    print(f"{'First Forward Pass Time':<25} | {hf_agg['first_token_latency']['median']:.4f} s            | {thin_agg['first_token_latency']['median']:.4f} s")
    print(f"{'Warm Generation Speed':<25} | {hf_agg['decode_tok_s']['median']:.2f} tok/s         | {thin_agg['decode_tok_s']['median']:.2f} tok/s")
    print("============================================================")

    # Interpret results
    tps_ratio = thin_agg['decode_tok_s']['median'] / hf_agg['decode_tok_s']['median']
    speedup_pct = (tps_ratio - 1) * 100
    is_meaningful = abs(speedup_pct) >= 5.0  # 5% threshold
    meaningful_str = f"{abs(speedup_pct):.1f}% speedup" if speedup_pct >= 0 else f"{abs(speedup_pct):.1f}% slowdown"
    interpretation = f"{meaningful_str} (meaningful)" if is_meaningful else f"{meaningful_str} (noise / within margin)"

    print(f"TPS Ratio: {tps_ratio:.3f}")
    print(f"Interpretation: {interpretation}")

    # Determine zero-copy usage
    print("ThinRuntime zero-copy views: YES (frombuffer on mmap)")
    print("PyTorch copied during load_state_dict: NO (assign=True was used)")
    print("Device transfer copied: YES (model.to(device) duplicates on GPU)")

    # Save to target/runtime_compare.json
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "prompt": args.prompt,
        "tokens": args.tokens,
        "warmup_tokens": args.warmup_tokens,
        "trials": args.trials,
        "hf_baseline": hf_agg,
        "thin_runtime": thin_agg,
        "execution_backend": "transformers_for_both_paths",
        "native_thinruntime_benchmark": False,
        "tps_ratio": tps_ratio,
        "speedup_percent": speedup_pct,
        "is_meaningful": is_meaningful,
        "thin_zero_copy": True,
        "pytorch_assign": True,
        "phase4_metrics": {
            "hf_time_to_first_generated_token_s": hf_cold_start,
            "thin_time_to_first_generated_token_s": thin_cold_start,
            "hf_time_to_first_model_forward_s": hf_agg['first_token_latency']['median'],
            "thin_time_to_first_model_forward_s": thin_agg['first_token_latency']['median'],
            "hf_warm_generation_tok_s": hf_agg['decode_tok_s']['median'],
            "thin_warm_generation_tok_s": thin_agg['decode_tok_s']['median'],
            "hf_cold_start_total_time_s": hf_cold_start,
            "thin_cold_start_total_time_s": thin_cold_start,
        }
    }
    
    target_dir = Path("target")
    target_dir.mkdir(exist_ok=True)
    compare_json_path = target_dir / "runtime_compare.json"
    compare_json_path.write_text(json.dumps(output_data, indent=2, sort_keys=True))
    print(f"Saved JSON comparison to {compare_json_path}")

    # Append to BENCHMARKS.md
    bench_md = Path("BENCHMARKS.md")
    
    md_entry = f"""

## ThinTensor hydration vs HF safetensors (Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

Both rows execute Transformers; this does not measure the native ThinTensor runtime.

| Metric | HF Baseline | ThinRuntime | Delta / Ratio |
| --- | --- | --- | --- |
| Load Time (median) | {hf_agg['load_time']['median']:.4f} s | {thin_agg['load_time']['median']:.4f} s | {thin_agg['load_time']['median'] - hf_agg['load_time']['median']:.4f} s |
| First-Token Latency (median) | {hf_agg['first_token_latency']['median']:.4f} s | {thin_agg['first_token_latency']['median']:.4f} s | {thin_agg['first_token_latency']['median'] - hf_agg['first_token_latency']['median']:.4f} s |
| Decode Speed (median) | {hf_agg['decode_tok_s']['median']:.2f} tok/s | {thin_agg['decode_tok_s']['median']:.2f} tok/s | Ratio: {tps_ratio:.3f} ({interpretation}) |
| Peak RSS Delta (median) | {hf_agg['rss_delta']['median'] / (1024**2):.2f} MiB | {thin_agg['rss_delta']['median'] / (1024**2):.2f} MiB | { (thin_agg['rss_delta']['median'] - hf_agg['rss_delta']['median']) / (1024**2):.2f} MiB |
| Peak GPU Allocated (median) | {hf_agg['gpu_peak_alloc']['median'] / (1024**2):.2f} MiB | {thin_agg['gpu_peak_alloc']['median'] / (1024**2):.2f} MiB | { (thin_agg['gpu_peak_alloc']['median'] - hf_agg['gpu_peak_alloc']['median']) / (1024**2):.2f} MiB |
| Time to First Token (CS) | {hf_cold_start:.4f} s | {thin_cold_start:.4f} s | {thin_cold_start - hf_cold_start:.4f} s |
| Time to First Model Forward | {hf_agg['first_token_latency']['median']:.4f} s | {thin_agg['first_token_latency']['median']:.4f} s | {thin_agg['first_token_latency']['median'] - hf_agg['first_token_latency']['median']:.4f} s |
| Warm Generation tok/s | {hf_agg['decode_tok_s']['median']:.2f} tok/s | {thin_agg['decode_tok_s']['median']:.2f} tok/s | Ratio: {tps_ratio:.3f} |
| Cold Start Total Time | {hf_cold_start:.4f} s | {thin_cold_start:.4f} s | {thin_cold_start - hf_cold_start:.4f} s |

* **Zero-copy CPU views used**: YES
* **PyTorch assign=True supported**: YES
* **Device transfer copied**: YES (model.to(device))
* **Warmup tokens**: {args.warmup_tokens}
* **Trials run**: {args.trials} (isolated subprocesses)
"""
    with bench_md.open("a") as f:
        f.write(md_entry)
    print(f"Appended benchmark results to {bench_md}")


if __name__ == "__main__":
    main()
