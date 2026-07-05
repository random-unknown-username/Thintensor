"""Explicit native architecture coverage and performance evidence.

An architecture is not considered performance-supported merely because its
weights can be loaded.  ``verified`` means the native runtime has both a
correctness result and an end-to-end result faster than the matched
Transformers loop. ``candidate`` means semantic/tensor support exists but the
performance gate has not been established. ``fallback`` remains runnable via
Transformers but is not a ThinTensor speed claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArchitectureStatus:
    name: str
    aliases: tuple[str, ...]
    family: str
    tensor_schema: str
    normalization: str
    attention: str
    mlp: str
    native_status: str
    validated_profile: str | None = None
    minimum_cosine: float | None = None
    top1_exact: bool | None = None
    top5_set_exact: bool | None = None
    top5_ordered_exact: bool | None = None
    thin_tokens_per_s: float | None = None
    hf_tokens_per_s: float | None = None
    speedup_vs_hf: float | None = None
    evidence_scope: str | None = None
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        payload["blockers"] = list(self.blockers)
        return payload


ARCHITECTURES: tuple[ArchitectureStatus, ...] = (
    ArchitectureStatus(
        name="smollm3",
        aliases=("smollm3",),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="rms_norm",
        attention="gqa_selective_rope",
        mlp="swiglu",
        native_status="verified",
        validated_profile="max-performance",
        minimum_cosine=0.997369766,
        top1_exact=True,
        top5_set_exact=True,
        top5_ordered_exact=False,
        thin_tokens_per_s=94.000319,
        hf_tokens_per_s=47.540605,
        speedup_vs_hf=1.977264,
        evidence_scope=(
            "SmolLM3-3B, RTX 5050 Laptop GPU, matched 500-token warmed causal "
            "decode versus Transformers BF16, full BF16 KV"
        ),
    ),
    ArchitectureStatus(
        name="stablelm",
        aliases=("stablelm",),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="layer_norm_bias",
        attention="mha_partial_rope",
        mlp="swiglu",
        native_status="verified",
        validated_profile="safe",
        minimum_cosine=0.998577237,
        top1_exact=True,
        top5_set_exact=True,
        top5_ordered_exact=False,
        thin_tokens_per_s=55.762293,
        hf_tokens_per_s=48.283997,
        speedup_vs_hf=1.154881,
        evidence_scope=(
            "StableLM-3B-4E1T, RTX 5050 Laptop GPU, 200-token warmed causal "
            "decode, full BF16 KV"
        ),
    ),
    ArchitectureStatus(
        name="llama",
        aliases=("llama", "tinyllama"),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="rms_norm",
        attention="mha_or_gqa_rope",
        mlp="swiglu",
        native_status="candidate",
        minimum_cosine=0.997788191,
        top1_exact=True,
        top5_set_exact=False,
        top5_ordered_exact=False,
        thin_tokens_per_s=180.284765,
        hf_tokens_per_s=132.364647,
        speedup_vs_hf=1.362031,
        evidence_scope=(
            "TinyLlama-1.1B-Chat-v1.0, RTX 5050 Laptop GPU, all public "
            "profiles at 200 tokens plus 1000-token prefill/50-step full-KV "
            "retention; max profile is faster but not promoted"
        ),
        blockers=(
            "strict short-suite top-5 set equality misses one BF16 cutoff tie",
            "a larger Llama checkpoint still needs an independent gate",
        ),
    ),
    ArchitectureStatus(
        name="mistral",
        aliases=("mistral",),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="rms_norm",
        attention="gqa_sliding_rope",
        mlp="swiglu",
        native_status="candidate",
        blockers=("matched correctness and HF speed gates not recorded",),
    ),
    ArchitectureStatus(
        name="qwen2",
        aliases=("qwen2", "qwen2_5"),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="rms_norm",
        attention="gqa_rope",
        mlp="swiglu",
        native_status="verified",
        validated_profile="max-performance",
        minimum_cosine=0.998892486,
        top1_exact=True,
        top5_set_exact=True,
        top5_ordered_exact=True,
        thin_tokens_per_s=93.079443,
        hf_tokens_per_s=48.183090,
        speedup_vs_hf=1.931787,
        evidence_scope=(
            "Qwen2.5-3B-Instruct, RTX 5050 Laptop GPU, repeated 500-token "
            "warmed full-causal decode plus all public profiles at 200 tokens"
        ),
    ),
    ArchitectureStatus(
        name="qwen3",
        aliases=("qwen3",),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="rms_norm_with_qk_norm",
        attention="gqa_rope_qk_norm",
        mlp="swiglu",
        native_status="candidate",
        blockers=("matched Qwen3 correctness and HF speed gates are incomplete",),
    ),
    ArchitectureStatus(
        name="phi",
        aliases=("phi3", "phi4"),
        family="dense_decoder",
        tensor_schema="hf_fused_qkv_fused_gate_up",
        normalization="rms_norm",
        attention="mha_or_gqa_partial_or_long_rope",
        mlp="swiglu",
        native_status="verified",
        validated_profile="max-performance",
        minimum_cosine=0.99990356,
        top1_exact=True,
        top5_set_exact=True,
        top5_ordered_exact=True,
        thin_tokens_per_s=53.797862,
        hf_tokens_per_s=39.088266,
        speedup_vs_hf=1.376317,
        evidence_scope=(
            "Phi-4-mini-instruct, RTX 5050 Laptop GPU, matched 200-token "
            "warmed causal decode versus Transformers BF16; public quick "
            "prefill 1/128 correctness plus targeted prefill 1/8, full BF16 KV"
        ),
    ),
    ArchitectureStatus(
        name="gpt-oss",
        aliases=("gpt_oss",),
        family="moe_decoder",
        tensor_schema="hf_packed_expert_moe",
        normalization="rms_norm",
        attention="gqa_attention_sinks",
        mlp="topk_swiglu_moe",
        native_status="candidate",
        blockers=("matched correctness and HF speed gates not recorded",),
    ),
    ArchitectureStatus(
        name="mixtral",
        aliases=("mixtral",),
        family="moe_decoder",
        tensor_schema="hf_separate_expert_moe",
        normalization="rms_norm",
        attention="gqa_sliding_rope",
        mlp="topk_swiglu_moe",
        native_status="candidate",
        blockers=("separate-expert repack and matched performance gate pending",),
    ),
    ArchitectureStatus(
        name="gemma2",
        aliases=("gemma2",),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="unit_offset_rms_norm",
        attention="gqa_interleaved_local_global",
        mlp="geglu",
        native_status="verified",
        validated_profile="max-performance",
        minimum_cosine=0.999554813,
        top1_exact=True,
        top5_set_exact=True,
        top5_ordered_exact=True,
        thin_tokens_per_s=57.289261,
        hf_tokens_per_s=53.580533,
        speedup_vs_hf=1.069218,
        evidence_scope=(
            "Gemma-2-2B-IT, RTX 5050 Laptop GPU, repeated 200-token warmed "
            "full-causal decode plus all public profiles; targeted 128-token "
            "prefill and 50-step adaptive-path validation, full BF16 KV"
        ),
    ),
    ArchitectureStatus(
        name="gemma",
        aliases=("gemma",),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="unit_offset_rms_norm",
        attention="gqa_rope",
        mlp="geglu",
        native_status="fallback",
        blockers=("Gemma 1 semantics need an independent correctness gate",),
    ),
    ArchitectureStatus(
        name="gemma3",
        aliases=("gemma3", "gemma3_text"),
        family="dense_decoder",
        tensor_schema="hf_separate_qkv_gated_mlp",
        normalization="unit_offset_rms_norm_with_qk_norm",
        attention="gqa_interleaved_local_global",
        mlp="geglu",
        native_status="fallback",
        blockers=(
            "Gemma 3 QK normalization and scaling need native implementation",
            "matched correctness and HF speed gates not recorded",
        ),
    ),
    ArchitectureStatus(
        name="deepseek",
        aliases=("deepseek_v2", "deepseek_v3", "deepseek"),
        family="moe_decoder",
        tensor_schema="deepseek_mla_moe",
        normalization="rms_norm",
        attention="multihead_latent_attention",
        mlp="shared_and_routed_moe",
        native_status="fallback",
        blockers=("MLA and shared-expert execution are not implemented",),
    ),
    ArchitectureStatus(
        name="gpt-neox",
        aliases=("gpt_neox",),
        family="dense_decoder",
        tensor_schema="transformer_h_fused_qkv",
        normalization="layer_norm_bias",
        attention="mha_partial_rope",
        mlp="gelu",
        native_status="fallback",
        blockers=("parallel residual and GELU MLP execution are not implemented",),
    ),
    ArchitectureStatus(
        name="falcon",
        aliases=("falcon", "refinedweb", "refinedwebmodel"),
        family="dense_decoder",
        tensor_schema="transformer_h_fused_qkv",
        normalization="layer_norm_bias",
        attention="mqa_or_gqa_rope",
        mlp="gelu",
        native_status="fallback",
        blockers=("Falcon parallel residual semantics are not implemented",),
    ),
    ArchitectureStatus(
        name="opt",
        aliases=("opt",),
        family="dense_decoder",
        tensor_schema="decoder_layers_separate_qkv",
        normalization="layer_norm_bias",
        attention="mha_learned_positions",
        mlp="relu",
        native_status="fallback",
        blockers=("learned positions and ReLU MLP are not implemented",),
    ),
    ArchitectureStatus(
        name="bloom",
        aliases=("bloom",),
        family="dense_decoder",
        tensor_schema="transformer_h_fused_qkv",
        normalization="layer_norm_bias",
        attention="mha_alibi",
        mlp="gelu",
        native_status="fallback",
        blockers=("ALiBi and BLOOM residual semantics are not implemented",),
    ),
    ArchitectureStatus(
        name="gpt2",
        aliases=("gpt2", "gpt_bigcode", "starcoder2"),
        family="dense_decoder",
        tensor_schema="transformer_h_conv1d",
        normalization="layer_norm_bias",
        attention="mha_learned_positions",
        mlp="gelu",
        native_status="fallback",
        blockers=("Conv1D layout, learned positions, and GELU are not implemented",),
    ),
    ArchitectureStatus(
        name="mpt",
        aliases=("mpt",),
        family="dense_decoder",
        tensor_schema="transformer_blocks_fused_qkv",
        normalization="layer_norm",
        attention="mha_alibi",
        mlp="gelu",
        native_status="fallback",
        blockers=("MPT block schema and ALiBi are not implemented",),
    ),
)


def normalize_architecture(value: str) -> str:
    compact = value.lower().replace("-", "").replace("_", "")
    for entry in ARCHITECTURES:
        for alias in entry.aliases:
            if alias.replace("_", "") in compact:
                return entry.name
    return value.lower()


def architecture_status(value: str) -> ArchitectureStatus | None:
    normalized = normalize_architecture(value)
    return next(
        (entry for entry in ARCHITECTURES if entry.name == normalized),
        None,
    )


def architecture_rows() -> list[dict[str, Any]]:
    return [entry.as_dict() for entry in ARCHITECTURES]
