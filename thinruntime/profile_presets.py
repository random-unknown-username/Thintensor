"""Portable runtime profiles and their user-facing contracts.

Profiles describe intent and semantic capabilities, not one model's tensor
dimensions.  Model-specific benchmark results may be attached as evidence, but
they never define whether another compatible decoder is allowed to run.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


DENSE_GATED_CAPABILITY = {
    "architecture_family": (
        "decoder_dense",
        "decoder_moe",
        "hybrid_decoder",
    ),
    "activation": ("silu", "swish", "gelu_pytorch_tanh"),
}


PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
        "label": "Safe BF16",
        "description": (
            "Portable BF16 profile with full causal history and no weight "
            "quantization."
        ),
        "kernel_backend": "triton-matvec",
        "gate_up_fp8": False,
        "down_proj_fp8": False,
        "o_proj_fp8": False,
        "qkv_fp8": False,
        "lm_head_fp8": False,
        "keep_bf16_lm_head": True,
        "lm_head_topk_guard": 0,
        "attention_backend": "torch",
        "fused_scaled_mlp": False,
        "fused_residual_norm": False,
        "experimental": False,
        "required_capabilities": None,
        "weight_residency": "all",
        "intent": "Highest fidelity and broadest native-runtime compatibility.",
        "quality_contract": (
            "BF16 weights and BF16 KV values. Different kernel reduction order "
            "can still produce small differences from Transformers."
        ),
        "retention_contract": "Full causal history; no KV compression or eviction.",
        "speed_contract": "Portable baseline; benchmark against HF on this machine.",
        "recommended_for": "First run, new architectures, and correctness checks.",
        "tradeoffs": ("Highest weight bandwidth and VRAM use.",),
    },
    "balanced": {
        "label": "Balanced native kernels",
        "description": (
            "Portable BF16-weight profile using ThinTensor matvec and fused "
            "causal-attention kernels without quantizing model weights."
        ),
        "kernel_backend": "triton",
        "gate_up_fp8": False,
        "down_proj_fp8": False,
        "o_proj_fp8": False,
        "qkv_fp8": False,
        "lm_head_fp8": False,
        "keep_bf16_lm_head": True,
        "lm_head_topk_guard": 0,
        "attention_backend": "triton_fused",
        "fused_scaled_mlp": False,
        "fused_residual_norm": False,
        "experimental": False,
        "required_capabilities": DENSE_GATED_CAPABILITY,
        "intent": "Use native kernels for speed without approximate weight storage.",
        "quality_contract": (
            "No weight quantization. Kernel-order rounding is possible; full "
            "model validation is still required before claiming HF equivalence."
        ),
        "retention_contract": "Full causal history; no KV compression or eviction.",
        "speed_contract": "Model and hardware dependent; run `thintensor bench`.",
        "recommended_for": "Portable default for supported dense gated decoders.",
        "tradeoffs": (
            "Requires a CUDA/Triton-compatible native decoder.",
            "Not every HF architecture has a native ThinTensor engine yet.",
        ),
    },
    "max-performance": {
        "label": "Maximum single-stream performance",
        "description": (
            "Portable opt-in speed profile for compatible dense gated decoders: "
            "exact prefill and early generated tokens, adaptive INT8 body "
            "weights, fused causal attention, guarded FP8 head, and guarded "
            "INT8 tensor-core dispatch."
        ),
        "kernel_backend": "triton",
        "gate_up_fp8": False,
        "down_proj_fp8": False,
        "o_proj_fp8": False,
        "qkv_fp8": False,
        "lm_head_fp8": True,
        "keep_bf16_lm_head": True,
        "lm_head_backend": "triton",
        "lm_head_topk_guard": 64,
        "attention_backend": "triton_fused",
        "kv_block_size": 512,
        "fused_rope": True,
        "exact_prefill": True,
        "adaptive_body_int8_start_token": 18,
        "experimental_int8_tensorcore": True,
        "fused_scaled_mlp": False,
        "fused_residual_norm": False,
        "experimental": True,
        "required_capabilities": DENSE_GATED_CAPABILITY,
        "intent": "Minimize single-stream weight bandwidth on supported GPUs.",
        "quality_contract": (
            "Prefill and the first 18 generated-token positions use exact body "
            "weights; later positions use approximate INT8 body weights. "
            "Quality must be validated per model; exact ordered top-5 is not "
            "promised."
        ),
        "retention_contract": "Full uncompressed BF16 KV history.",
        "speed_contract": (
            "No portable tok/s promise. The retained matched 3B run measured "
            "94.00 tok/s at 500 tokens versus 47.54 tok/s in Transformers."
        ),
        "measured_results": {
            "scope": "one retained 3B dense-SiLU model on the development Blackwell laptop",
            "tokens_per_second": {"500": 94.000319},
            "transformers_tokens_per_second": {"500": 47.540605},
            "speedup_vs_transformers": {"500": 1.977264},
            "minimum_short_cosine": 0.997369766,
            "long_1000_cosine": 0.999871254,
            "short_top1_exact": True,
            "short_top5_set_exact": True,
            "ordered_top5_exact": False,
            "kv_retention": "full_uncompressed_bf16",
        },
        "recommended_for": (
            "Opt-in local tuning after `thintensor validate`; never assume the "
            "retained measurement transfers to another model."
        ),
        "tradeoffs": (
            "Quality is model-dependent.",
            "Tensor-core gains are shape and GPU dependent.",
            "Throughput declines as full causal attention grows.",
        ),
    },
    "max-max-perf": {
        "label": "Maximum aggressive performance",
        "description": (
            "Aggressive speed profile with FP8 body weights (MLP/attention projections), "
            "fused rope, exact prefill, adaptive body INT8, lm head FP8, and fused scaled MLP/residual norm."
        ),
        "kernel_backend": "triton",
        "gate_up_fp8": True,
        "down_proj_fp8": True,
        "o_proj_fp8": True,
        "qkv_fp8": True,
        "lm_head_fp8": True,
        "keep_bf16_lm_head": True,
        "lm_head_backend": "triton",
        "lm_head_topk_guard": 64,
        "attention_backend": "triton_fused",
        "kv_block_size": 512,
        "fused_rope": True,
        "exact_prefill": True,
        "adaptive_body_int8_start_token": 18,
        "experimental_int8_tensorcore": True,
        "fused_scaled_mlp": True,
        "fused_residual_norm": True,
        "experimental": True,
        "required_capabilities": DENSE_GATED_CAPABILITY,
        "intent": "Minimize weight loading bandwidth through aggressive FP8 body quantization.",
        "quality_contract": (
            "MLP and attention projections quantized to FP8. Prefill and early generated "
            "tokens are exact; later tokens use adaptive INT8. Cosine similarity targets 0.995+."
        ),
        "retention_contract": "Full uncompressed BF16 KV history.",
        "speed_contract": "Max speed; benchmark against max-performance locally.",
        "recommended_for": "Local execution on Blackwell or RTX GPUs for maximum speedup.",
        "tradeoffs": (
            "Higher quantization noise than max-performance.",
            "Requires CUDA compatibility and sufficient VRAM headroom.",
        ),
    },
    "lab": {
        "label": "Experimental workbench",
        "description": (
            "Portable BF16 starting point that permits explicit low-level "
            "experimental overrides."
        ),
        "kernel_backend": "triton-matvec",
        "gate_up_fp8": False,
        "down_proj_fp8": False,
        "o_proj_fp8": False,
        "qkv_fp8": False,
        "lm_head_fp8": False,
        "keep_bf16_lm_head": True,
        "lm_head_topk_guard": 0,
        "attention_backend": "torch",
        "fused_scaled_mlp": False,
        "fused_residual_norm": False,
        "experimental": True,
        "required_capabilities": None,
        "intent": "Manual kernel and precision experiments.",
        "quality_contract": "No quality or performance guarantee.",
        "retention_contract": "Depends on explicit overrides.",
        "speed_contract": "No speed guarantee.",
        "recommended_for": "Developers running an explicit validation plan.",
        "tradeoffs": ("Every override must be benchmarked and correctness-gated.",),
    },
}

# Preserve the exact behavior of previously published opt-in names without
# making model-era presets the primary product interface.
PROFILES["legacy-fast-60"] = {
    **PROFILES["max-performance"],
    "label": "Legacy adaptive INT8 profile",
    "description": (
        "Compatibility preset for the original fast-60 CLI behavior."
    ),
    "experimental_int8_tensorcore": False,
    "speed_contract": "Legacy opt-in; no portable throughput promise.",
    "hidden": True,
}
PROFILES["legacy-fast-80"] = {
    **PROFILES["legacy-fast-60"],
    "label": "Legacy MXFP4 profile",
    "description": (
        "Compatibility preset for the original fast-80 MXFP4 experiment."
    ),
    "mxfp4_gate_up_layers": "6:12",
    "quality_contract": (
        "Known model-specific short-context regression; retained only for "
        "reproducibility."
    ),
    "hidden": True,
}


# Old names remain accepted so existing commands/scripts do not break. Public
# documentation leads with the four intent-based names above.
PROFILE_ALIASES = {
    "bf16": "safe",
    "quality": "balanced",
    "quality-guarded": "balanced",
    "fast": "balanced",
    "fast-60": "legacy-fast-60",
    "fast-80": "legacy-fast-80",
    "speed": "max-performance",
    "fast-90": "max-performance",
    "experimental": "lab",
}

EXPERIMENTAL_WARNING = (
    "This profile can change logits or regress decode speed on a different "
    "model. Benchmark and validate it locally before deployment."
)

CPU_KV_WARNING = (
    "CPU KV residency preserves exact values but is a memory-pressure mode, "
    "not a measured speed optimization."
)


def profile_names(*, include_aliases: bool = False) -> tuple[str, ...]:
    names = list(PROFILES)
    if include_aliases:
        names.extend(PROFILE_ALIASES)
    return tuple(names)


def _canonical_name(name: str) -> str:
    normalized = name.strip().lower()
    return PROFILE_ALIASES.get(normalized, normalized)


def _model_value(model: Mapping[str, Any], key: str) -> Any:
    if key == "activation":
        return model.get("activation") or model.get("hidden_act") or "silu"
    if key == "architecture_family":
        if model.get(key):
            return model[key]
        experts = int(
            model.get("num_local_experts")
            or model.get("num_experts")
            or 0
        )
        return "decoder_moe" if experts else "decoder_dense"
    if key == "norm_kind":
        if model.get("norm_kind"):
            return model["norm_kind"]
        if model.get("rms_norm_eps") is not None:
            return "rms_norm"
        if model.get("layer_norm_eps") is not None:
            return "layer_norm"
        return "unknown"
    return model.get(key)


def geometry_matches(
    model: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Backward-compatible capability matcher."""
    return all(
        _model_value(model, key) in value
        if isinstance(value, (tuple, list, set, frozenset))
        else _model_value(model, key) == value
        for key, value in expected.items()
    )


def profile_compatibility(
    profile: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[bool, str | None]:
    expected = profile.get("required_capabilities")
    if expected is None:
        return True, None
    if geometry_matches(model, expected):
        return True, None
    expected_text = ", ".join(
        (
            f"{key} in {tuple(value)}"
            if isinstance(value, (tuple, list, set, frozenset))
            else f"{key}={value}"
        )
        for key, value in expected.items()
    )
    actual_text = ", ".join(
        f"{key}={_model_value(model, key)!r}" for key in expected
    )
    return (
        False,
        f"profile requires {expected_text}; model exposes {actual_text}",
    )


def get_profile(
    name: str,
    *,
    model: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve a named profile and enforce semantic capability requirements."""
    normalized = name.strip().lower()
    requested_auto = normalized == "auto"
    if normalized == "auto":
        normalized = (
            "safe"
            if model is not None
            and _model_value(model, "norm_kind") == "layer_norm"
            else "balanced"
        )
    canonical = _canonical_name(normalized)
    if canonical not in PROFILES:
        available = ", ".join(("auto", *profile_names(include_aliases=True)))
        raise ValueError(
            f"unknown profile {name!r}; available profiles: {available}"
        )
    result = dict(PROFILES[canonical])
    if (
        canonical == "max-performance"
        and model is not None
        and _model_value(model, "architecture_family") == "decoder_moe"
    ):
        # Dense adaptive-INT8 flags are semantically invalid for MoE. MoE
        # throughput comes from full-device expert packs and batched routing.
        result.update(
            {
                "label": "Maximum MoE single-stream performance",
                "description": (
                    "Full-VRAM selected-expert packing with fused causal "
                    "attention and RoPE."
                ),
                "gate_up_fp8": False,
                "down_proj_fp8": False,
                "o_proj_fp8": False,
                "qkv_fp8": False,
                "lm_head_fp8": False,
                "keep_bf16_lm_head": True,
                "lm_head_backend": None,
                "lm_head_topk_guard": 0,
                "exact_prefill": False,
                "adaptive_body_int8_start_token": -1,
                "experimental_int8_tensorcore": False,
                "fused_rope": True,
                "required_capabilities": {
                    "architecture_family": "decoder_moe",
                    "activation": ("silu", "swish"),
                },
                "quality_contract": (
                    "Expert precision is selected by the device-aware fit "
                    "plan; full causal KV history remains exact."
                ),
                "speed_contract": (
                    "Uses the full safe VRAM envelope and batched selected-"
                    "expert kernels."
                ),
            }
        )
    if (
        canonical == "max-max-perf"
        and model is not None
        and _model_value(model, "architecture_family") == "decoder_moe"
    ):
        # FP8 attention-projection scale tensors push OLMoE-class 7B models
        # over the 8 GiB VRAM budget when all expert weights are resident.
        # Fused residual norm and scaled MLP reduce kernel launch overhead
        # across all 16 layers without adding any extra VRAM.
        result.update(
            {
                "label": "Maximum aggressive MoE performance",
                "description": (
                    "Full-VRAM selected-expert packing with fused residual "
                    "norm, fused scaled MLP, and RoPE."
                ),
                "gate_up_fp8": False,
                "down_proj_fp8": False,
                "o_proj_fp8": False,
                "qkv_fp8": False,
                "lm_head_fp8": False,
                "keep_bf16_lm_head": True,
                "lm_head_backend": None,
                "lm_head_topk_guard": 0,
                "exact_prefill": False,
                "adaptive_body_int8_start_token": -1,
                "experimental_int8_tensorcore": False,
                "fused_rope": True,
                "fused_residual_norm": True,
                "fused_scaled_mlp": True,
                "required_capabilities": {
                    "architecture_family": "decoder_moe",
                    "activation": ("silu", "swish"),
                },
                "quality_contract": (
                    "All weights remain BF16. Fused residual norm and scaled "
                    "MLP reduce per-layer kernel launch overhead."
                ),
                "speed_contract": (
                    "Full-VRAM expert packing with fused element-wise ops; "
                    "FP8 attention projections require more VRAM than a 7B "
                    "MoE can spare on 8 GiB GPUs."
                ),
            }
        )

    if (
        model is not None
        and str(model.get("model_type") or "") == "qwen3_5"
        and canonical in {"balanced", "max-performance"}
    ):
        # Qwen3.5's retained winner is its exact-BF16 fused recurrent path.
        # Dense adaptive quantization and fused short-context GQA both regress
        # this shape, so do not inherit model-specific legacy defaults.
        result.update(
            {
                "kernel_backend": "triton-matvec",
                "attention_backend": "torch",
                "gate_up_fp8": False,
                "down_proj_fp8": False,
                "o_proj_fp8": False,
                "qkv_fp8": False,
                "lm_head_fp8": False,
                "keep_bf16_lm_head": True,
                "lm_head_topk_guard": 0,
                "exact_prefill": False,
                "adaptive_body_int8_start_token": -1,
                "experimental_int8_tensorcore": False,
                "fused_rope": False,
                "preferred_auto_quant": "off",
                "quality_contract": (
                    "BF16 weights, FP32 recurrent DeltaNet state, and full "
                    "uncompressed BF16 KV history."
                ),
                "speed_contract": (
                    "Uses the architecture-native fused recurrent kernel; "
                    "quantized dense modes are disabled because they lost "
                    "end-to-end throughput on the retained checkpoint."
                ),
            }
        )
    if (
        model is not None
        and str(model.get("model_type") or "") == "qwen3_5"
        and canonical == "max-max-perf"
    ):
        result.update(
            {
                "kernel_backend": "triton-matvec",
                "attention_backend": "torch",
                "gate_up_fp8": False,
                "down_proj_fp8": False,
                "o_proj_fp8": True,
                "qkv_fp8": True,
                "lm_head_fp8": False,
                "keep_bf16_lm_head": True,
                "lm_head_topk_guard": 0,
                "exact_prefill": False,
                "adaptive_body_int8_start_token": -1,
                "experimental_int8_tensorcore": False,
                "fused_rope": True,
                "fused_residual_norm": True,
                "preferred_auto_quant": "off",
            }
        )
    if (
        model is not None
        and str(model.get("model_type") or "") == "gemma4"
        and canonical in {"balanced", "max-performance"}
    ):
        result.update(
            {
                "kernel_backend": "triton-matvec",
                "attention_backend": "torch",
                "gate_up_fp8": False,
                "down_proj_fp8": False,
                "o_proj_fp8": False,
                "qkv_fp8": False,
                "lm_head_fp8": False,
                "keep_bf16_lm_head": True,
                "lm_head_topk_guard": 0,
                "exact_prefill": False,
                "adaptive_body_int8_start_token": -1,
                "experimental_int8_tensorcore": False,
                "fused_rope": False,
                "preferred_auto_quant": "on",
                "quality_contract": (
                    "All projection weights remain BF16. Only the 4.375-GiB "
                    "token-indexed PLE table uses row-scaled INT8 storage."
                ),
            }
        )
    if (
        model is not None
        and str(model.get("model_type") or "") == "gemma4"
        and canonical == "max-max-perf"
    ):
        # Gemma-4's unique forward path does not go through the standard
        # adaptive-INT8 layer plan. On RTX 5050 / Blackwell, the FP8 body
        # weight path forces a slower scaled-matvec kernel that more than
        # erases any bandwidth saving (benchmark: FP8 33.3 vs BF16 35.0 tok/s).
        # triton-matvec is the fastest kernel backend for this model.
        # Fused residual norm and scaled MLP reduce kernel launch count
        # across the 35-layer forward pass.
        result.update(
            {
                "kernel_backend": "triton-matvec",
                "attention_backend": "triton_fused",
                "gate_up_fp8": False,
                "down_proj_fp8": False,
                "o_proj_fp8": False,
                "qkv_fp8": False,
                "lm_head_fp8": True,
                "keep_bf16_lm_head": True,
                "lm_head_backend": "triton",
                "lm_head_topk_guard": 64,
                "exact_prefill": False,
                "adaptive_body_int8_start_token": -1,
                "experimental_int8_tensorcore": False,
                "fused_rope": True,
                "fused_residual_norm": True,
                "fused_scaled_mlp": True,
                "preferred_auto_quant": "on",
                "quality_contract": (
                    "All body projection weights remain BF16 (FP8 body "
                    "regresses speed on this GPU). LM head is FP8-guarded. "
                    "Fused residual norm and scaled MLP reduce kernel launch "
                    "overhead across the 35-layer Gemma-4 forward pass."
                ),
                "speed_contract": (
                    "triton-matvec + fused kernels; benchmark vs lab/safe on "
                    "this machine."
                ),
            }
        )
    result["name"] = canonical
    if normalized in PROFILE_ALIASES:
        result["alias_used"] = normalized
    if model is not None:
        compatible, reason = profile_compatibility(result, model)
        if not compatible and requested_auto:
            result = dict(PROFILES["safe"])
            result["name"] = "safe"
            result["auto_fallback_reason"] = reason
        elif not compatible and not force:
            raise ValueError(
                f"profile {canonical!r} is not compatible with this model: "
                f"{reason}. Use --profile safe or --force-profile for an "
                "explicit experiment."
            )
        result["compatibility_warning"] = reason if not compatible else None
    return result


def apply_overrides(
    profile: dict[str, Any],
    *,
    head8: bool = False,
    o_proj_fp8: bool = False,
    fused_scaled_mlp: bool = False,
    fused_residual_norm: bool = False,
    allow_experimental: bool = False,
) -> dict[str, Any]:
    requested = {
        "head8": head8,
        "o_proj_fp8": o_proj_fp8,
        "fused_scaled_mlp": fused_scaled_mlp,
        "fused_residual_norm": fused_residual_norm,
    }
    used = [name for name, enabled in requested.items() if enabled]
    if used and not allow_experimental and not profile.get("experimental"):
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in used)
        raise ValueError(
            f"{flags} require --allow-experimental because they are not "
            "validated defaults"
        )
    result = dict(profile)
    if head8:
        result.update(lm_head_fp8=True, keep_bf16_lm_head=False, lm_head_topk_guard=0)
    if o_proj_fp8:
        result["o_proj_fp8"] = True
    if fused_scaled_mlp:
        result["fused_scaled_mlp"] = True
    if fused_residual_norm:
        result["fused_residual_norm"] = True
    if used:
        result["experimental"] = True
        result["experimental_overrides"] = used
    return result


def profile_to_runtime_kwargs(profile: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "kernel_backend",
        "gate_up_fp8",
        "down_proj_fp8",
        "down_fp8_layer_spec",
        "o_proj_fp8",
        "qkv_fp8",
        "lm_head_fp8",
        "keep_bf16_lm_head",
        "lm_head_backend",
        "lm_head_topk_guard",
        "attention_backend",
        "fused_rope",
        "adaptive_body_int8_start_token",
        "mxfp4_gate_up_layers",
        "mxfp4_down_layers",
        "mxfp4_qkv_layers",
        "mxfp4_o_layers",
        "fused_scaled_mlp",
        "fused_residual_norm",
    )
    return {key: profile[key] for key in keys if key in profile}


def profile_to_runtime_flags(profile: Mapping[str, Any]) -> list[str]:
    flags: list[str] = [
        "--kernel-backend",
        str(profile.get("kernel_backend", "triton")),
        "--attention-backend",
        str(profile.get("attention_backend", "torch")),
    ]
    if profile.get("lm_head_backend"):
        flags.extend(["--lm-head-backend", str(profile["lm_head_backend"])])
    booleans = {
        "gate_up_fp8": "--gate-up-fp8",
        "down_proj_fp8": "--down-proj-fp8",
        "o_proj_fp8": "--o-proj-fp8",
        "qkv_fp8": "--qkv-fp8",
        "lm_head_fp8": "--lm-head-fp8",
        "keep_bf16_lm_head": "--keep-bf16-lm-head",
        "fused_scaled_mlp": "--fused-scaled-mlp",
        "fused_residual_norm": "--fused-residual-norm",
        "fused_rope": "--fused-rope",
        "exact_prefill": "--exact-prefill",
    }
    for key, flag in booleans.items():
        if profile.get(key):
            flags.append(flag)
    if profile.get("down_fp8_layer_spec"):
        flags.extend(["--down-fp8-layers", str(profile["down_fp8_layer_spec"])])
    guard = int(profile.get("lm_head_topk_guard") or 0)
    if guard:
        flags.extend(["--lm-head-topk-guard", str(guard)])
    adaptive_start = int(profile.get("adaptive_body_int8_start_token", -1))
    if adaptive_start >= 0:
        flags.extend(["--adaptive-body-int8-start-token", str(adaptive_start)])
    kv_block_size = int(profile.get("kv_block_size") or 0)
    if kv_block_size > 0:
        flags.extend(["--kv-block-size", str(kv_block_size)])
    for key, flag in (
        ("mxfp4_gate_up_layers", "--mxfp4-gate-up-layers"),
        ("mxfp4_down_layers", "--mxfp4-down-layers"),
        ("mxfp4_qkv_layers", "--mxfp4-qkv-layers"),
        ("mxfp4_o_layers", "--mxfp4-o-layers"),
    ):
        value = profile.get(key)
        if value is not None:
            flags.extend([flag, str(value)])
    return flags


_PROFILE_ENV_KEYS = (
    "THINTENSOR_INT8_TENSORCORE",
    "THINTENSOR_INT8_TC_BLOCK_N",
    "THINTENSOR_INT8_TC_BLOCK_M",
    "THINTENSOR_INT8_TC_BLOCK_K",
    "THINTENSOR_TENSORCORE_MATVEC",
)


def profile_environment(profile: Mapping[str, Any]) -> dict[str, str]:
    if not profile.get("experimental_int8_tensorcore"):
        return {}
    return {
        "THINTENSOR_INT8_TENSORCORE": "1",
        "THINTENSOR_INT8_TC_BLOCK_N": "2",
        "THINTENSOR_INT8_TC_BLOCK_M": "64",
        "THINTENSOR_INT8_TC_BLOCK_K": "256",
        "THINTENSOR_TENSORCORE_MATVEC": "1",
    }


def activate_profile_environment(profile: Mapping[str, Any]) -> None:
    for key in _PROFILE_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(profile_environment(profile))


def subprocess_environment(profile: Mapping[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    for key in _PROFILE_ENV_KEYS:
        environment.pop(key, None)
    environment.update(profile_environment(profile))
    return environment


def profile_public_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "label",
        "description",
        "intent",
        "quality_contract",
        "retention_contract",
        "speed_contract",
        "recommended_for",
        "tradeoffs",
        "experimental",
        "required_capabilities",
        "measured_results",
    )
    return {key: profile.get(key) for key in keys if key in profile}


def recommend_profile(
    goal: str,
    *,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = goal.strip().lower()
    names = {
        "quality": "safe",
        "balanced": "balanced",
        "speed": "max-performance",
    }
    if normalized not in names:
        raise ValueError("goal must be one of: quality, balanced, speed")
    try:
        return get_profile(names[normalized], model=model)
    except ValueError:
        return get_profile("safe", model=model)
