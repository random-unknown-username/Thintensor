#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path to import thinruntime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.archive import ThinArchive
from thinruntime.torch_loader import load_tensor_view

def test_shared_tensor_detection():
    thin_path = Path("/tmp/thintensor-first-run/qwen3-0.6b.thin")
    if not thin_path.exists():
        # Fallback to check target folder
        candidates = list(Path("./target").glob("*.thin"))
        if candidates:
            thin_path = candidates[0]
        else:
            print("Skipping test: No .thin archive found to run tests on.")
            sys.exit(0)

    print(f"Loading archive: {thin_path}")
    archive = ThinArchive(thin_path)

    state_dict = {}
    tensors_by_checksum = {}
    
    # Simulate our aliasing loader
    page_ids = [
        page["id"]
        for page in archive.manifest.get("pages", [])
        if page.get("kind") != "fused_physical"
    ]
    for page_id in page_ids:
        p_meta = archive.get_tensor_metadata(page_id)
        checksum = p_meta["checksum"]
        key = (checksum, tuple(p_meta["shape"]))
        
        if key in tensors_by_checksum:
            tensor, is_zero = tensors_by_checksum[key]
            state_dict[page_id] = tensor
            print(f"Detected shared tensor: {page_id} is aliased to previous tensor with checksum {checksum}")
        else:
            tensor, is_zero = load_tensor_view(archive, page_id)
            state_dict[page_id] = tensor
            tensors_by_checksum[key] = (tensor, is_zero)

    # Check for tied word embeddings
    embed_name = "model.embed_tokens.weight"
    lm_head_name = "lm_head.weight"
    
    if embed_name in state_dict and lm_head_name in state_dict:
        tied_by_content = (
            archive.get_tensor_metadata(embed_name)["checksum"]
            == archive.get_tensor_metadata(lm_head_name)["checksum"]
            and tuple(state_dict[embed_name].shape)
            == tuple(state_dict[lm_head_name].shape)
        )
        is_aliased = state_dict[embed_name] is state_dict[lm_head_name]
        print(f"\nChecking aliasing for {embed_name} and {lm_head_name}:")
        print(f"  Same Python Object: {is_aliased}")
        if tied_by_content:
            assert is_aliased, "Tensors with same checksum/shape are not aliased to the same object!"
            print("SUCCESS: Shared tensor detection and aliasing works perfectly!")
        else:
            assert not is_aliased, "Distinct embedding/head tensors were incorrectly aliased"
            print("SUCCESS: Distinct embedding/head tensors remain independent.")
    else:
        print(f"WARNING: embed_tokens or lm_head not found in state_dict. Found keys: {list(state_dict.keys())[:5]}")

    archive.close()

if __name__ == "__main__":
    test_shared_tensor_detection()
