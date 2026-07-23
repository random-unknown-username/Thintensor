from dataclasses import dataclass, asdict
from typing import Any, Mapping

@dataclass
class MemoryLedger:
    source_weight_bytes: int = 0
    gpu_bf16_weights: int = 0
    gpu_fp8_weights: int = 0
    gpu_int8_weights: int = 0
    gpu_int4_packed_weights: int = 0
    quantization_scales: int = 0
    sparse_residual_sidecars: int = 0
    low_rank_correction_sidecars: int = 0
    adaptive_exact_bf16_copies: int = 0
    bf16_lm_head_copies: int = 0
    quantized_lm_head_copies: int = 0
    tied_embedding_head_bytes: int = 0
    fused_qkv_matrices: int = 0
    fused_gate_up_matrices: int = 0
    persistent_runtime_buffers: int = 0
    temporary_quant_buffers: int = 0
    triton_workspaces: int = 0
    cuda_graph_pools: int = 0
    kv_cache_current: int = 0
    kv_cache_max: int = 0
    recurrent_model_state: int = 0
    attention_scratch: int = 0
    logits_and_shortlist: int = 0
    page_pool_staging: int = 0
    allocator_reserved: int = 0
    cuda_context: int = 0
    external_gpu_usage: int = 0
    
    # Aggregates
    peak_initialization: int = 0
    peak_first_token: int = 0
    peak_steady_state: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def build_memory_plan(
    archive_path: str,
    device: str,
    context: int,
    profile_name: str,
    budget_text: str = "0",
) -> dict[str, Any]:
    from .cli import _adapt_profile_for_device, _automatic_fit_plan, _profile_auto_quant_mode, _profile_with_auto_fit
    from .archive import ThinArchive
    from .profile_presets import profile_to_runtime_kwargs, get_profile, apply_overrides
    
    archive = ThinArchive(archive_path)
    model_meta = archive.manifest.get("model", {})
    profile = get_profile(profile_name, model=model_meta)
    profile = apply_overrides(profile)
    profile = _adapt_profile_for_device(profile, requested_name=profile_name, model=model_meta, device=device)
    
    auto_quant = _profile_auto_quant_mode(profile, "auto")
    auto_fit_plan = _automatic_fit_plan(
        archive_path,
        device=device,
        context=context,
        budget_text=budget_text,
        auto_quant=auto_quant,
        lm_head_fp8=profile.get("lm_head_fp8", False)
    )
    profile = _profile_with_auto_fit(profile, auto_fit_plan)
    
    # Analyze weights
    ledger = MemoryLedger()
    pages = [p for p in archive.manifest.get("pages", []) if p.get("kind") != "fused_physical"]
    for page in pages:
        ledger.source_weight_bytes += page.get("size", 0)
        
    layers = int(model_meta.get("layers", 28))
    hidden = int(model_meta.get("hidden_size", 2048))
    vocab = int(model_meta.get("vocab_size", 151936))
    
    # Estimate KV cache
    kv_heads = int(model_meta.get("kv_heads", model_meta.get("heads", 8)))
    head_dim = int(model_meta.get("head_dim", hidden // int(model_meta.get("heads", 8))))
    kv_cache_bytes = context * layers * 2 * kv_heads * head_dim * 2
    ledger.kv_cache_max = kv_cache_bytes
    
    # Estimate LM Head
    lm_head_bytes = vocab * hidden * 2
    ledger.bf16_lm_head_copies = lm_head_bytes
    if profile.get("lm_head_fp8"):
        ledger.quantized_lm_head_copies = lm_head_bytes // 2
        if not profile.get("keep_bf16_lm_head", True):
            ledger.bf16_lm_head_copies = 0
            
    # Estimate body weights based on profile
    # _adaptive_body_int8 enabled?
    if profile.get("adaptive_body_int8_start_token", -1) >= 0:
        ledger.adaptive_exact_bf16_copies = ledger.source_weight_bytes - lm_head_bytes
        ledger.gpu_int8_weights = ledger.adaptive_exact_bf16_copies // 2
        
    ledger.peak_initialization = (
        ledger.source_weight_bytes + 
        ledger.adaptive_exact_bf16_copies + 
        ledger.gpu_int8_weights + 
        ledger.gpu_fp8_weights +
        ledger.bf16_lm_head_copies +
        ledger.quantized_lm_head_copies
    )
    
    # Add dummy margins
    ledger.cuda_context = 600 * 1024**2
    ledger.triton_workspaces = 100 * 1024**2
    
    ledger.peak_first_token = ledger.peak_initialization + ledger.kv_cache_max + ledger.cuda_context + ledger.triton_workspaces
    
    return {
        "profile": profile_name,
        "context": context,
        "auto_fit_plan": getattr(auto_fit_plan, "mode", "unknown") if auto_fit_plan else "none",
        "ledger": ledger.as_dict()
    }
