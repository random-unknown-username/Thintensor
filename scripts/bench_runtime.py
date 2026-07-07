#!/usr/bin/env python3
"""Runtime benchmark for HF safetensors vs ThinTensor-backed hydration.

This intentionally reuses Transformers for model execution. ThinTensor is tested
as the storage/hydration layer, not as a hand-written Qwen/Llama kernel stack.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import mmap
import os
import struct
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dep
    psutil = None


HEADER = struct.Struct("<8sIIQQQQQ32s")
PAGE_PREFIX = struct.Struct("<H")
PAGE_TAIL = struct.Struct("<QQQI")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    if args.mode == "hf":
        result = bench_hf(args.path, args.prompt, args.tokens, device, dtype, args)
    else:
        if args.hf_dir is None:
            raise SystemExit("--hf-dir is required for thin mode")
        result = bench_thin(args.path, args.hf_dir, args.prompt, args.tokens, device, dtype, args)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_text(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hf", "thin"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--hf-dir", type=Path)
    parser.add_argument("--prompt", default="Write one short paragraph about tensor layouts.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max-gpu-temp", type=int, default=87)
    parser.add_argument("--temp-poll-sec", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def bench_hf(
    path: Path,
    prompt: str,
    tokens: int,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reset_gpu_peak(device)
    rss0 = rss_bytes()
    monitor = TempMonitor(device, args.max_gpu_temp, args.temp_poll_sec)
    monitor.start()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path)
    if "cuda" in str(device):
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
    model.eval()
    sync(device)
    load_s = time.perf_counter() - t0

    gen = run_generate(model, tokenizer, prompt, tokens, device, monitor)
    monitor.stop()
    return result("hf", path, device, dtype, load_s, gen, rss0, monitor)


def bench_thin(
    path: Path,
    hf_dir: Path,
    prompt: str,
    tokens: int,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reset_gpu_peak(device)
    rss0 = rss_bytes()
    monitor = TempMonitor(device, args.max_gpu_temp, args.temp_poll_sec)
    monitor.start()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    config = AutoConfig.from_pretrained(hf_dir)
    thin = read_thin(path)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    state_dict = build_state_dict(thin)
    tied_lm_head = bool(getattr(config, "tie_word_embeddings", False))
    if tied_lm_head and "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    bad_missing = [
        name
        for name in missing
        if not allowed_missing_tensor(name, tied_lm_head)
    ]
    if bad_missing:
        raise RuntimeError(f"missing model tensors: {bad_missing[:16]}")
    if unexpected:
        raise RuntimeError(f"unexpected tensors: {unexpected[:16]}")
    if tied_lm_head:
        model.tie_weights()

    materialize_meta_buffers(model, dtype)
    assert_no_meta_parameters(model)
    model.to(device)
    del state_dict
    close_thin(thin)
    gc.collect()
    malloc_trim()
    model.eval()
    sync(device)
    load_s = time.perf_counter() - t0

    gen = run_generate(model, tokenizer, prompt, tokens, device, monitor)
    monitor.stop()
    out = result("thin", path, device, dtype, load_s, gen, rss0, monitor)
    out["thin_pages"] = len(thin["records"])
    out["thin_manifest_bytes"] = thin["manifest_len"]
    return out


def run_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    tokens: int,
    device: str,
    monitor: "TempMonitor",
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    sync(device)
    monitor.raise_if_hot("before generation")
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([ThermalStop(monitor)]),
        )
    sync(device)
    elapsed_s = time.perf_counter() - t0
    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(output.shape[-1])
    new_tokens = max(0, output_tokens - input_tokens)
    return {
        "prompt_tokens": input_tokens,
        "new_tokens": new_tokens,
        "generate_s": elapsed_s,
        "tokens_per_s": new_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "thermal_stop": monitor.too_hot,
    }


def read_thin(path: Path) -> dict[str, Any]:
    file = path.open("rb")
    mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    magic, header_len, version, manifest_off, manifest_len, page_table_off, page_count, data_off, _ = (
        HEADER.unpack_from(mm, 0)
    )
    if magic != b"THINv0\0\0" or header_len != 88 or version != 0:
        raise ValueError("not a ThinTensor v0 archive")

    manifest = json.loads(mm[manifest_off : manifest_off + manifest_len])
    pos = page_table_off
    records = {}
    for _ in range(page_count):
        (id_len,) = PAGE_PREFIX.unpack_from(mm, pos)
        pos += PAGE_PREFIX.size
        page_id = mm[pos : pos + id_len].decode("utf-8")
        pos += id_len
        offset, stored_size, raw_size, flags = PAGE_TAIL.unpack_from(mm, pos)
        pos += PAGE_TAIL.size
        checksum = mm[pos : pos + 32]
        pos += 32
        if flags != 0 or stored_size != raw_size or offset < data_off:
            raise ValueError(f"unsupported page table record for {page_id}")
        records[page_id] = {
            "offset": offset,
            "size": stored_size,
            "checksum": checksum,
        }

    return {
        "file": file,
        "mmap": mm,
        "manifest": manifest,
        "manifest_len": manifest_len,
        "records": records,
    }


def close_thin(thin: dict[str, Any]) -> None:
    thin["mmap"].close()
    thin["file"].close()


def build_state_dict(thin: dict[str, Any]) -> dict[str, torch.Tensor]:
    warnings.filterwarnings("ignore", message="The given buffer is not writable")
    mm = thin["mmap"]
    records = thin["records"]
    state = {}
    for page in ordered_pages(thin["manifest"]):
        record = records[page["id"]]
        dtype = page_dtype(page["dtype"])
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        count = record["size"] // elem_bytes
        tensor = torch.frombuffer(mm, dtype=dtype, count=count, offset=record["offset"])
        state[page["id"]] = tensor.reshape(page["shape"])
    return state


def ordered_pages(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pages = {page["id"]: page for page in manifest["pages"]}
    seen = set()
    ordered = []
    for stage in manifest.get("execution_tape", []):
        for page_id in stage.get("page_refs", []):
            page = pages.get(page_id)
            if page is not None and page_id not in seen:
                seen.add(page_id)
                ordered.append(page)
    for page in manifest["pages"]:
        if page["id"] not in seen:
            seen.add(page["id"])
            ordered.append(page)
    return ordered


def allowed_missing_tensor(name: str, tied_lm_head: bool) -> bool:
    if name.endswith("rotary_emb.inv_freq") or "rotary_emb" in name:
        return True
    if tied_lm_head and name == "lm_head.weight":
        return True
    return False


def materialize_meta_buffers(model: torch.nn.Module, dtype: torch.dtype) -> None:
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and getattr(buffer, "is_meta", False):
                module._buffers[name] = torch.empty(buffer.shape, dtype=dtype, device="cpu")


def assert_no_meta_parameters(model: torch.nn.Module) -> None:
    meta = [name for name, param in model.named_parameters() if getattr(param, "is_meta", False)]
    if meta:
        raise RuntimeError(f"unmaterialized meta parameters: {meta[:16]}")


def page_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"f16", "fp16", "float16"}:
        return torch.float16
    if normalized in {"f32", "fp32", "float32"}:
        return torch.float32
    if normalized in {"f64", "fp64", "float64"}:
        return torch.float64
    if normalized in {"i64", "int64"}:
        return torch.int64
    if normalized in {"i32", "int32"}:
        return torch.int32
    if normalized in {"u8", "uint8"}:
        return torch.uint8
    raise ValueError(f"unsupported ThinTensor page dtype {name}")


def result(
    mode: str,
    path: Path,
    device: str,
    dtype: torch.dtype,
    load_s: float,
    gen: dict[str, Any],
    rss0: int | None,
    monitor: "TempMonitor",
) -> dict[str, Any]:
    rss1 = rss_bytes()
    return {
        "mode": mode,
        "path": str(path),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "load_s": load_s,
        "prompt_tokens": gen["prompt_tokens"],
        "new_tokens": gen["new_tokens"],
        "generate_s": gen["generate_s"],
        "tokens_per_s": gen["tokens_per_s"],
        "rss_start_bytes": rss0,
        "rss_end_bytes": rss1,
        "rss_delta_bytes": None if rss0 is None or rss1 is None else rss1 - rss0,
        "gpu_peak_allocated_bytes": gpu_peak_allocated(device),
        "gpu_peak_reserved_bytes": gpu_peak_reserved(device),
        "gpu_peak_temp_c": monitor.peak_temp,
        "gpu_max_temp_c": monitor.max_temp,
        "thermal_stop": gen["thermal_stop"],
    }


class ThermalStop(StoppingCriteria):
    def __init__(self, monitor: "TempMonitor") -> None:
        self.monitor = monitor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        return self.monitor.too_hot


class TempMonitor:
    def __init__(self, device: str, max_temp: int, poll_sec: float) -> None:
        self.device = device
        self.max_temp = max_temp
        self.poll_sec = poll_sec
        self.peak_temp: int | None = None
        self.too_hot = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device != "cuda" or not has_nvidia_smi():
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def raise_if_hot(self, where: str) -> None:
        self._sample()
        if self.too_hot:
            raise RuntimeError(
                f"GPU temperature reached {self.peak_temp}C {where}; "
                f"limit is {self.max_temp}C"
            )

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            self._sample()

    def _sample(self) -> None:
        temp = read_gpu_temp()
        if temp is None:
            return
        self.peak_temp = temp if self.peak_temp is None else max(self.peak_temp, temp)
        if temp >= self.max_temp:
            self.too_hot = True


def has_nvidia_smi() -> bool:
    return subprocess.run(
        ["bash", "-lc", "command -v nvidia-smi >/dev/null"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def read_gpu_temp() -> int | None:
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


def print_text(result: dict[str, Any]) -> None:
    for key, value in result.items():
        print(f"{key}: {value}")


def choose_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return value


def choose_dtype(value: str, device: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


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
        return


if __name__ == "__main__":
    main()
