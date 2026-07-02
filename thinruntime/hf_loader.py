import torch
import gc
import ctypes
import os
import time
from typing import Dict, Tuple, Any, List, Optional
from transformers import AutoConfig, AutoModelForCausalLM
from .archive import ThinArchive
from .torch_loader import load_tensor_view, map_dtype

def malloc_trim() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

def allowed_missing_tensor(name: str, tied_lm_head: bool) -> bool:
    if name.endswith("rotary_emb.inv_freq") or "rotary_emb" in name:
        return True
    if tied_lm_head and name == "lm_head.weight":
        return True
    return False

def materialize_meta_buffers(model: torch.nn.Module, dtype: torch.dtype) -> None:
    config = model.config
    rope_theta = getattr(config, "rope_theta", 10000.0)

    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and getattr(buffer, "is_meta", False):
                module._buffers[name] = torch.empty(buffer.shape, dtype=dtype, device="cpu")
                if "rotary_emb" in name or "inv_freq" in name:
                    num_elements = buffer.numel()
                    dim = 2 * num_elements
                    base = getattr(module, "base", rope_theta)
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                    module._buffers[name].copy_(inv_freq.to(dtype=module._buffers[name].dtype))

def load_thin_model(
    archive_path: str,
    hf_dir: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Loads a HF model from .thin archive weights.
    Returns the loaded model and a dictionary of loading diagnostics.
    """
    diagnostics = {}
    
    # 1. Open the archive
    t_start = time.perf_counter()
    archive = ThinArchive(archive_path)
    diagnostics["archive_open_time_s"] = time.perf_counter() - t_start
    
    # 2. Parse/load HF config
    t_config = time.perf_counter()
    config = AutoConfig.from_pretrained(hf_dir)
    diagnostics["config_load_time_s"] = time.perf_counter() - t_config
    
    # 3. Create model on meta device
    t_meta = time.perf_counter()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    diagnostics["meta_model_creation_time_s"] = time.perf_counter() - t_meta
    
    # 4. Expose tensor views
    t_views = time.perf_counter()
    state_dict = {}
    zero_copy_count = 0
    copied_count = 0
    total_tensor_bytes = 0
    
    page_ids = [p["id"] for p in archive.manifest.get("pages", [])]
    tensors_by_checksum = {}
    
    for page_id in page_ids:
        p_spec = next(p for p in archive.manifest.get("pages", []) if p["id"] == page_id)
        if p_spec.get("kind") == "fused_physical":
            continue

        p_meta = archive.get_tensor_metadata(page_id)
        checksum = p_meta["checksum"]
        key = (checksum, tuple(p_meta["shape"]))
        
        if key in tensors_by_checksum:
            tensor, is_zero = tensors_by_checksum[key]
            state_dict[page_id] = tensor
        else:
            tensor, is_zero = load_tensor_view(archive, page_id)
            state_dict[page_id] = tensor
            tensors_by_checksum[key] = (tensor, is_zero)
            
            if is_zero:
                zero_copy_count += 1
            else:
                copied_count += 1
            total_tensor_bytes += tensor.nelement() * tensor.element_size()
        
    diagnostics["tensor_views_creation_time_s"] = time.perf_counter() - t_views
    diagnostics["zero_copy_views_count"] = zero_copy_count
    diagnostics["copied_views_count"] = copied_count
    diagnostics["total_tensor_bytes"] = total_tensor_bytes
    
    # 5. Handle tied weights
    t_assign = time.perf_counter()
    tied_lm_head = bool(getattr(config, "tie_word_embeddings", False))
    if tied_lm_head and "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]
        
    # Materialize meta buffers (like rotary embeddings, etc.) on CPU
    materialize_meta_buffers(model, dtype or torch.float32)
                
    # 6. Load state_dict using assign=True if supported
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    except Exception:
        # Fallback if assign=True is not supported
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=False)
        
    bad_missing = [name for name in missing if not allowed_missing_tensor(name, tied_lm_head)]
    if bad_missing:
        raise RuntimeError(f"Missing model tensors from .thin: {bad_missing[:16]}")
    if unexpected:
        raise RuntimeError(f"Unexpected tensors found in .thin: {unexpected[:16]}")
        
    if tied_lm_head:
        model.tie_weights()
        
    # Check for unmaterialized meta parameters
    meta_params = [name for name, param in model.named_parameters() if getattr(param, "is_meta", False)]
    if meta_params:
        raise RuntimeError(f"Unmaterialized meta parameters: {meta_params[:16]}")
        
    diagnostics["weight_assignment_time_s"] = time.perf_counter() - t_assign
    
    # 7. Move to device
    t_device = time.perf_counter()
    model.to(device)
    diagnostics["device_transfer_time_s"] = time.perf_counter() - t_device
    
    # 8. Clean up
    del state_dict
    archive.close()
    gc.collect()
    malloc_trim()
    
    diagnostics["total_load_time_s"] = time.perf_counter() - t_start
    return model, diagnostics
