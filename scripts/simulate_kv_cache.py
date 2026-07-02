#!/usr/bin/env python3
import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path to import thinruntime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime import ThinArchive

def codec_bytes(codec: str) -> float:
    c = codec.lower()
    if c == "q2":
        return 0.25
    elif c == "q3":
        return 0.375
    elif c == "q4":
        return 0.5
    elif c == "q5":
        return 0.625
    elif c == "q6":
        return 0.75
    elif c in ("q8", "fp8"):
        return 1.0
    elif c in ("fp16", "bf16"):
        return 2.0
    elif c == "fp32":
        return 4.0
    else:
        return 2.0

def estimate_kv_cache_bytes(
    layers: int,
    kv_heads: int,
    head_dim: int,
    ctx: int,
    batch_size: int,
    kv_dtype: str,
    recent_precision_tokens: int = 256
) -> float:
    # 2 scalars (key and value) per head dimension
    scalars_per_token = layers * kv_heads * head_dim * 2
    
    recent = min(recent_precision_tokens, ctx)
    old = max(0, ctx - recent)
    
    # 2.0 bytes per scalar for high precision (fp16/bf16)
    recent_bytes = scalars_per_token * recent * 2.0
    old_bytes = scalars_per_token * old * codec_bytes(kv_dtype)
    
    return (recent_bytes + old_bytes) * batch_size

def main():
    parser = argparse.ArgumentParser(description="KV Cache Planning Simulation")
    parser.add_argument("thin_file", type=Path, help="Path to .thin archive")
    parser.add_argument("--vram", type=str, default="4GB", help="GPU VRAM target (e.g. 4GB, 8GB)")
    args = parser.parse_args()

    if not args.thin_file.exists():
        print(f"Error: {args.thin_file} not found.")
        sys.exit(1)

    # Parse VRAM limit
    vram_str = args.vram.upper()
    if vram_str.endswith("GB"):
        vram_limit = float(vram_str.replace("GB", "")) * 1024**3
    elif vram_str.endswith("MIB"):
        vram_limit = float(vram_str.replace("MIB", "")) * 1024**2
    else:
        vram_limit = float(vram_str)

    archive = ThinArchive(args.thin_file)
    model_spec = archive.manifest.get("model", {})
    layers = int(model_spec.get("layers", 28))
    kv_heads = int(model_spec.get("kv_heads", 8))
    heads = int(model_spec.get("heads", 16))
    hidden_size = int(model_spec.get("hidden_size", 1024))
    head_dim = int(model_spec.get("head_dim") or (hidden_size // heads))
    
    # Unique weight bytes
    unique_weight_bytes = 0
    seen_checksums = set()
    for page in archive.manifest.get("pages", []):
        if page.get("kind") != "fused_physical":
            checksum = page["checksum"]
            if checksum not in seen_checksums:
                seen_checksums.add(checksum)
                unique_weight_bytes += page["size"]
                
    scratch_bytes = archive.manifest.get("memory_plan", {}).get("scratch_bytes", 64 * 1024**2)

    print("==========================================================")
    print("           ThinTensor KV Cache Simulation Report          ")
    print("==========================================================")
    print(f"Model: {model_spec.get('arch')} (layers={layers}, kv_heads={kv_heads}, head_dim={head_dim})")
    print(f"Unique Weights: {unique_weight_bytes / 1024**2:.2f} MiB")
    print(f"VRAM Target:    {args.vram} ({vram_limit / 1024**2:.2f} MiB)")
    print("==========================================================")

    # Simulation matrix
    contexts = [1024, 2048, 4096, 8192]
    batches = [1, 4, 8]
    kv_dtypes = ["fp16", "q8", "q4", "q3", "q2"]

    print(f"\n| {'Ctx':<5} | {'Batch':<5} | {'KV Dtype':<8} | {'Est KV (MiB)':<13} | {'Total Est (MiB)':<16} | {'Status':<6} |")
    print("|" + "-"*7 + "|" + "-"*7 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*18 + "|" + "-"*8 + "|")

    for ctx in contexts:
        for batch in batches:
            for dtype in kv_dtypes:
                kv_bytes = estimate_kv_cache_bytes(
                    layers=layers,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    ctx=ctx,
                    batch_size=batch,
                    kv_dtype=dtype
                )
                
                # Total estimated VRAM
                total_est = unique_weight_bytes + scratch_bytes * batch + kv_bytes
                status = "FIT" if total_est <= vram_limit else "FAIL"
                
                print(f"| {ctx:<5} | {batch:<5} | {dtype:<8} | {kv_bytes / 1024**2:13.2f} | {total_est / 1024**2:16.2f} | {status:<6} |")
                
    print("==========================================================")
    print("Note: The estimation assumes a hybrid policy where the first 256 tokens are held in FP16.")
    
    archive.close()

if __name__ == "__main__":
    main()
