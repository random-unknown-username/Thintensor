#!/usr/bin/env python3
import sys
import time
import argparse
import subprocess
import gc
import os
import ctypes
import warnings
import threading
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Add project root to sys.path to import thinruntime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime import ThinArchive, load_tensor_view, load_thin_model

try:
    import psutil
except ImportError:
    psutil = None

def malloc_trim() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

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

def reset_gpu_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

def gpu_peak_allocated(device: str) -> int | None:
    if device != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())

def gpu_peak_reserved(device: str) -> int | None:
    if device != "cuda":
        return None
    return int(torch.cuda.max_memory_reserved())

class TempMonitor:
    def __init__(self, device: str, poll_sec: float = 0.5) -> None:
        self.device = device
        self.poll_sec = poll_sec
        self.peak_temp: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device != "cuda" or not self._has_nvidia_smi():
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            self._sample()

    def _sample(self) -> None:
        temp = self._read_gpu_temp()
        if temp is None:
            return
        self.peak_temp = temp if self.peak_temp is None else max(self.peak_temp, temp)

    def _has_nvidia_smi(self) -> bool:
        return subprocess.run(
            ["bash", "-lc", "command -v nvidia-smi >/dev/null"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0

    def _read_gpu_temp(self) -> int | None:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode != 0:
            return None
        first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        try:
            return int(first)
        except ValueError:
            return None

def find_hf_dir(thin_path: Path) -> Path:
    # Look in common places
    candidates = [
        Path("./models/Qwen3-0.6B"),
        Path("/tmp/thintensor-qwen3-0.6b"),
        Path("/tmp/thintensor-first-run/Qwen--Qwen3-0.6B"),
        Path("/tmp/thintensor-first-run/Qwen3-0.6B"),
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return c
    raise RuntimeError(
        "Could not auto-detect HF model directory. Please provide it via --hf-dir."
    )

def main():
    parser = argparse.ArgumentParser(description="PyTorch model construction from .thin archive.")
    parser.add_argument("thin_file", type=Path, help="Path to .thin archive file")
    parser.add_argument("--prompt", type=str, default="Explain ThinTensor in one sentence.", help="Generation prompt")
    parser.add_argument("--tokens", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Execution device")
    parser.add_argument("--hf-dir", type=Path, default=None, help="Path to HuggingFace directory containing config/tokenizer")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        device = "cpu"

    thin_file = args.thin_file
    if not thin_file.exists():
        print(f"Error: {thin_file} does not exist.")
        sys.exit(1)

    hf_dir = args.hf_dir
    if hf_dir is None:
        try:
            hf_dir = find_hf_dir(thin_file)
            print(f"Auto-resolved HF directory: {hf_dir}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Begin Measurements
    print("\n--- Starting ThinRuntime Verification and Load ---")
    reset_gpu_peak(device)
    rss0 = rss_bytes()
    monitor = TempMonitor(device)
    monitor.start()

    # 1. Archive verify time
    print("Verifying archive via CLI...")
    t_verify_start = time.perf_counter()
    cli_path = None
    for p in ["target/release/thintensor", "target/debug/thintensor"]:
        if Path(p).exists():
            cli_path = p
            break
    if cli_path:
        try:
            subprocess.run([cli_path, "verify", str(thin_file)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            t_verify = time.perf_counter() - t_verify_start
            print(f"Archive verified in {t_verify:.4f} s")
        except subprocess.CalledProcessError as e:
            print(f"CLI verify failed: {e.stderr.decode('utf-8', errors='replace')}")
            sys.exit(1)
    else:
        print("CLI thintensor binary not found, skipping verify timing.")
        t_verify = 0.0

    # Load model and gather diagnostics
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    t_load_start = time.perf_counter()
    model, diagnostics = load_thin_model(
        archive_path=str(thin_file),
        hf_dir=str(hf_dir),
        device=device,
        dtype=dtype,
    )
    t_load = time.perf_counter() - t_load_start

    t_parse = diagnostics["archive_open_time_s"]
    t_views = diagnostics["tensor_views_creation_time_s"]
    t_construct = diagnostics["meta_model_creation_time_s"]
    t_assign = diagnostics["weight_assignment_time_s"]
    t_transfer = diagnostics["device_transfer_time_s"]
    tensors_count = diagnostics["zero_copy_views_count"] + diagnostics["copied_views_count"]
    print(f"Loaded {tensors_count} tensors in {t_load:.4f} s using load_thin_model")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Run inference
    model.eval()
    print("\n--- Running Generation ---")
    
    # Measure first-token latency (prefill)
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        past_key_values = outputs.past_key_values
    if device == "cuda":
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t_prefill_start

    # Measure decode
    generated_token_tensors = [next_token]
    curr_input_ids = next_token.unsqueeze(-1)
    
    t_decode_start = time.perf_counter()
    for _ in range(args.tokens - 1):
        with torch.no_grad():
            outputs = model(input_ids=curr_input_ids, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            past_key_values = outputs.past_key_values
            generated_token_tensors.append(next_token)
            curr_input_ids = next_token.unsqueeze(-1)
    if device == "cuda":
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t_decode_start

    monitor.stop()
    rss1 = rss_bytes()

    # Decode generated output
    generated_tokens = (
        torch.cat(generated_token_tensors).detach().cpu().tolist()
    )
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Prompt: {args.prompt}")
    print(f"Generated text: {decoded_text}")

    # Metrics Calculations
    decode_tok_s = len(generated_tokens) / t_decode if t_decode > 0 else 0.0
    rss_delta = (rss1 - rss0) if (rss1 is not None and rss0 is not None) else 0
    gpu_peak_alloc = gpu_peak_allocated(device)
    gpu_peak_res = gpu_peak_reserved(device)
    
    print("\n--- Measured Metrics ---")
    print(f"Archive Verify Time:             {t_verify:.4f} s")
    print(f"Thin Loader Parse Time:          {t_parse:.4f} s")
    print(f"Tensor View Creation:            {t_views:.4f} s")
    print(f"Model Construction:              {t_construct:.4f} s")
    print(f"Weight Assignment:               {t_assign:.4f} s")
    print(f"Device Transfer Time:            {t_transfer:.4f} s")
    print(f"First-Token Latency:             {t_prefill:.4f} s")
    print(f"Decode speed:                    {decode_tok_s:.2f} tok/s")
    print(f"Peak RSS Delta:                  {rss_delta / (1024**2):.2f} MiB")
    
    # Phase 4 explicit measurements
    cold_start_total = t_load + t_prefill
    print(f"Time to first generated token:   {cold_start_total:.4f} s")
    print(f"Time to first model forward:     {t_prefill:.4f} s")
    print(f"Warm generation tok/s:           {decode_tok_s:.2f} tok/s")
    print(f"Cold start total time:           {cold_start_total:.4f} s")

    if device == "cuda":
        print(f"Peak GPU Allocated:              {gpu_peak_alloc / (1024**2):.2f} MiB")
        print(f"Peak GPU Reserved:               {gpu_peak_res / (1024**2):.2f} MiB")
        if monitor.peak_temp is not None:
            print(f"Peak GPU Temp:                   {monitor.peak_temp} C")

if __name__ == "__main__":
    main()
