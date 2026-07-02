#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path to import thinruntime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.archive import ThinArchive
from thinruntime.torch_loader import load_tensor_view

def main():
    if len(sys.argv) < 2:
        print("Usage: check_thin_loader.py <model.thin>")
        sys.exit(1)

    thin_path = Path(sys.argv[1])
    if not thin_path.exists():
        print(f"Error: {thin_path} does not exist.")
        sys.exit(1)

    print(f"Opening archive: {thin_path}")
    
    # Open ThinArchive (by default try to mmap)
    archive = ThinArchive(thin_path)
    
    model_spec = archive.manifest.get("model", {})
    arch = model_spec.get("arch", "unknown")
    pages = archive.manifest.get("pages", [])
    
    page_count = len(archive.pages)
    tensor_count = len(pages)
    
    print("\n--- Model Information ---")
    print(f"Model Arch: {arch}")
    print(f"Page Count in Archive: {page_count}")
    print(f"Tensor Count in Manifest: {tensor_count}")
    
    zero_copy_count = 0
    copied_count = 0
    total_bytes = 0
    tensors_info = []
    
    for i, p in enumerate(pages):
        if p.get("kind") == "fused_physical":
            continue
        page_id = p["id"]
        try:
            tensor, is_zero = load_tensor_view(archive, page_id)
            if is_zero:
                zero_copy_count += 1
            else:
                copied_count += 1
            
            t_bytes = tensor.nelement() * tensor.element_size()
            total_bytes += t_bytes
            
            if i < 10:
                tensors_info.append((page_id, p["shape"], p["dtype"], is_zero))
        except Exception as e:
            print(f"Warning: Failed to load tensor {page_id}: {e}")

    print(f"Total Tensor Bytes: {total_bytes} bytes ({total_bytes / (1024**3):.2f} GiB)")
    print(f"Number of Zero-Copy Tensor Views: {zero_copy_count}")
    print(f"Number of Copied Tensor Views: {copied_count}")
    
    print("\n--- First 10 Tensors ---")
    for name, shape, dtype, is_zero in tensors_info:
        zero_str = "zero-copy" if is_zero else "copied"
        print(f"  {name}: shape={shape}, dtype={dtype} ({zero_str})")
        
    print("\n--- Checksum Spot Check (First 3 Pages) ---")
    spot_pages = pages[:3]
    for p in spot_pages:
        page_id = p["id"]
        record = archive.pages[page_id]
        stored_checksum_hex = record["checksum"].hex()
        manifest_checksum_hex = p["checksum"]
        print(f"  Page: {page_id}")
        print(f"    Page Table Checksum: {stored_checksum_hex}")
        print(f"    Manifest Checksum:   {manifest_checksum_hex}")
        if stored_checksum_hex == manifest_checksum_hex:
            print("    Status: Match")
        else:
            print("    Status: MISMATCH")

    tensors_info.clear()
    if "tensor" in locals():
        del tensor
    archive.close()

if __name__ == "__main__":
    main()
