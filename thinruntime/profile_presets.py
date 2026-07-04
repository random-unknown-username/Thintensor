"""Validated runtime profiles shared by the ThinTensor CLI workflows."""

from __future__ import annotations

from typing import Any, Mapping


SMOLLM3_3B_GEOMETRY = {
    "model_type": "smollm3",
    "layers": 36,
    "hidden_size": 2048,
    "intermediate_size": 11008,
}


PROFILES: dict[str, dict[str, Any]] = {
    "bf16": {
        "label": "BF16 reference precision",
        "description": (
            "Universal BF16 execution profile with BF16 weights and "
            "exact-value BF16 KV storage."
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
        "validated_geometry": None,
    },
    "quality": {
        "label": "SmolLM3 quality FP8",
        "description": (
            "Validated SmolLM3-3B profile: FP8 gate/up in every layer and "
            "FP8 down projection in layers 8:28; attention, O-proj, KV, and "
            "LM head remain BF16."
        ),
        "kernel_backend": "triton",
        "gate_up_fp8": True,
        "down_proj_fp8": True,
        "down_fp8_layer_spec": "8:28",
        "o_proj_fp8": False,
        "qkv_fp8": False,
        "lm_head_fp8": False,
        "keep_bf16_lm_head": True,
        "lm_head_topk_guard": 0,
        "attention_backend": "torch",
        "fused_scaled_mlp": False,
        "fused_residual_norm": False,
        "experimental": False,
        "validated_geometry": SMOLLM3_3B_GEOMETRY,
    },
    "quality-guarded": {
        "label": "SmolLM3 quality FP8 + guarded head",
        "description": (
            "The quality profile plus an FP8 shortlist execution head and "
            "exact BF16 verification over the top 64 candidates."
        ),
        "kernel_backend": "triton",
        "gate_up_fp8": True,
        "down_proj_fp8": True,
        "down_fp8_layer_spec": "8:28",
        "o_proj_fp8": False,
        "qkv_fp8": False,
        "lm_head_fp8": True,
        "keep_bf16_lm_head": True,
        "lm_head_backend": "triton",
        "lm_head_topk_guard": 64,
        "attention_backend": "torch",
        "fused_scaled_mlp": False,
        "fused_residual_norm": False,
        "experimental": False,
        "validated_geometry": SMOLLM3_3B_GEOMETRY,
    },
    "experimental": {
        "label": "Experimental workbench",
        "description": (
            "Starts from BF16 and permits explicit experimental overrides. "
            "It does not silently stack rejected optimizations."
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
        "validated_geometry": None,
    },
}

# Compatibility alias retained for the first CLI prototype. It resolves to the
# measured guarded profile instead of pretending to be a distinct mode.
PROFILE_ALIASES = {"fast": "quality-guarded"}

EXPERIMENTAL_WARNING = (
    "Experimental overrides can change logits or regress decode speed. "
    "Benchmark and validate them before making claims."
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


def geometry_matches(
    model: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(model.get(key) == value for key, value in expected.items())


def profile_compatibility(
    profile: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[bool, str | None]:
    expected = profile.get("validated_geometry")
    if expected is None:
        return True, None
    if geometry_matches(model, expected):
        return True, None
    expected_text = ", ".join(
        f"{key}={value}" for key, value in expected.items()
    )
    actual_text = ", ".join(
        f"{key}={model.get(key, '?')}" for key in expected
    )
    return (
        False,
        f"profile is validated for {expected_text}; archive has {actual_text}",
    )


def get_profile(
    name: str,
    *,
    model: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve a named or automatic profile and enforce its validation scope."""
    normalized = name.strip().lower()
    if normalized == "auto":
        normalized = (
            "quality-guarded"
            if model is not None
            and geometry_matches(model, SMOLLM3_3B_GEOMETRY)
            else "bf16"
        )
    canonical = _canonical_name(normalized)
    if canonical not in PROFILES:
        available = ", ".join(("auto", *profile_names(include_aliases=True)))
        raise ValueError(
            f"unknown profile {name!r}; available profiles: {available}"
        )
    result = dict(PROFILES[canonical])
    result["name"] = canonical
    if normalized in PROFILE_ALIASES:
        result["alias_used"] = normalized
    if model is not None:
        compatible, reason = profile_compatibility(result, model)
        if not compatible and not force:
            raise ValueError(
                f"profile {canonical!r} is not compatible with this archive: "
                f"{reason}. Use --force-profile only for an explicit experiment."
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
    """Apply explicit unsafe overrides without silently changing a safe profile."""
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
        result.update(
            lm_head_fp8=True,
            keep_bf16_lm_head=False,
            lm_head_topk_guard=0,
        )
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
    """Convert a profile into ThinGpuQwenRuntime constructor arguments."""
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
        "fused_scaled_mlp",
        "fused_residual_norm",
    )
    return {key: profile[key] for key in keys if key in profile}


def profile_to_runtime_flags(profile: Mapping[str, Any]) -> list[str]:
    """Translate a profile into scripts/thin_runtime.py CLI flags."""
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
    }
    for key, flag in booleans.items():
        if profile.get(key):
            flags.append(flag)
    if profile.get("down_fp8_layer_spec"):
        flags.extend(
            ["--down-fp8-layers", str(profile["down_fp8_layer_spec"])]
        )
    guard = int(profile.get("lm_head_topk_guard") or 0)
    if guard:
        flags.extend(["--lm-head-topk-guard", str(guard)])
    return flags
