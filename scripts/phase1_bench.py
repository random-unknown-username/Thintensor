import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import time
import json
import subprocess
from thinruntime.cli import (
    _parse_bytes, _automatic_fit_plan,
    _adapt_profile_for_device, _tokenize_chat_messages, _print, _kv
)
from thinruntime.profile_presets import get_profile, apply_overrides
from thinruntime import ThinArchive, ThinGpuWeights, ThinGpuPagePool, PagedKVCache, ThinGpuCausalLMRuntime

def get_vram_info():
    """Returns (free, total) in bytes via NVML or nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        free, total = map(lambda x: int(x.strip()) * 1024 * 1024, out.strip().split(","))
        return free, total
    except Exception:
        return 0, 0

def run_bench(profile_name, context_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    archive_path = "/home/satvik/.cache/thintensor/archives/Qwen--Qwen2.5-3B-Instruct.thin"
    free_before, total_vram = get_vram_info()
    
    t0 = time.perf_counter()
    archive = ThinArchive(archive_path)
    manifest = archive.manifest
    model_meta = manifest["model"]
    
    # Emulate cli.py profile logic
    profile = get_profile(profile_name, model=model_meta)
    profile = apply_overrides(profile)
    profile = _adapt_profile_for_device(profile, requested_name=profile_name, model=model_meta, device="cuda")
    
    auto_quant = "auto"
    if auto_quant == "auto":
        from thinruntime.cli import _profile_auto_quant_mode
        auto_quant = _profile_auto_quant_mode(profile, auto_quant)
        
    auto_fit_plan = _automatic_fit_plan(
        archive_path,
        device="cuda",
        context=context_len,
        budget_text="0",
        auto_quant=auto_quant,
        lm_head_fp8=profile.get("lm_head_fp8", False)
    )
    
    from thinruntime.cli import _profile_with_auto_fit
    profile = _profile_with_auto_fit(profile, auto_fit_plan)
    
    # Load weights
    from thinruntime.profile_presets import profile_to_runtime_kwargs
    runtime_kwargs = profile_to_runtime_kwargs(profile)
    
    # Free VRAM before loading
    free_before_load, _ = get_vram_info()
    
    try:
        weights = ThinGpuWeights(archive_path, device="cuda", dtype=torch.bfloat16)
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
        
    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    peak_load = torch.cuda.max_memory_allocated()
    alloc_after_load = torch.cuda.memory_allocated()
    res_after_load = torch.cuda.memory_reserved()
    
    # KV Cache
    kv_cache = PagedKVCache(
        layers=int(model_meta["layers"]),
        kv_heads=int(model_meta.get("kv_heads", model_meta.get("heads", 1))),
        head_dim=int(model_meta.get("head_dim", int(model_meta["hidden_size"]) // int(model_meta["heads"]))),
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        block_size=profile.get("kv_block_size", 16),
        residency="gpu_full",
        budget_bytes=None
    )
    
    # Runtime
    runtime = ThinGpuCausalLMRuntime(
        weights,
        kv_cache=kv_cache,
        prefetch_distance=0,
        evict_completed_layers=False,
        **runtime_kwargs
    )
    
    torch.cuda.synchronize()
    mem_before_prefill = torch.cuda.memory_allocated()
    
    # Emulate chat loop first token
    messages = [{"role": "user", "content": "HEYO"}]
    token_ids = _tokenize_chat_messages(messages, archive_path, manifest, tokenizer_source=None)
    
    prefill_peak = 0
    first_token_peak = 0
    
    try:
        torch.cuda.reset_peak_memory_stats()
        # Process all tokens
        for i, tid in enumerate(token_ids):
            hidden = runtime.forward_token(tid, token_index=i)
        
        prefill_peak = torch.cuda.max_memory_allocated()
        
        # Generate 1 token
        torch.cuda.reset_peak_memory_stats()
        logits = runtime.next_token_tensor(hidden)
        next_tid = int(logits.argmax(dim=-1).item())
        first_token_peak = torch.cuda.max_memory_allocated()
        
        print(f"Success for {profile_name}. Prefill Peak: {prefill_peak / 1024**2:.2f} MB, First Token Peak: {first_token_peak / 1024**2:.2f} MB")
        
    except RuntimeError as e:
        print(f"OOM for {profile_name}: {e}")

if __name__ == "__main__":
    for prof in ["safe", "balanced", "max-performance", "max-max-perf"]:
        print(f"=== {prof} ===")
        run_bench(prof, 2048)
