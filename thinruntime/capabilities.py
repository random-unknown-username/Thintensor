"""Operator-driven native capability discovery.

Model-family names carry evidence labels only. Native execution is accepted
when geometry, tensor roles, and every required operator have an implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model_arch import required_operators_from_config
from .operator_registry import unsupported_operators


@dataclass(frozen=True)
class NativeSupport:
    supported: bool
    engine: str
    model_type: str
    architecture: str
    architecture_family: str
    activation: str
    attention: str
    reasons: tuple[str, ...]
    capabilities: tuple[str, ...]
    architecture_status: str = "unregistered"
    performance_supported: bool = False
    validated_profile: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["capabilities"] = list(self.capabilities)
        return payload


def analyze_hf_directory(path: str | Path) -> NativeSupport:
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.is_file():
        return _unsupported("unknown", "unknown", ("config.json is missing",))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _unsupported(
            "unknown",
            "unknown",
            (f"config.json cannot be read: {exc}",),
        )

    tensor_names = _tensor_names(root)
    text = config.get("text_config")
    if isinstance(text, dict):
        effective = dict(text)
        for key in ("architectures", "tie_word_embeddings", "model_type"):
            if key in config:
                effective[key] = config[key]
        skipped_layers = len(_cross_attention_layers(config))
        if skipped_layers:
            layer_count = int(
                _config_value(
                    effective,
                    "num_hidden_layers",
                    "n_layer",
                    "num_layers",
                )
                or 0
            )
            if layer_count:
                effective["num_hidden_layers"] = max(0, layer_count - skipped_layers)
        config = effective
        tensor_names = _canonical_text_tensor_names(config_path, tensor_names)
    return analyze_config(config, tensor_names=tensor_names)


def analyze_archive_model(model: dict[str, Any]) -> NativeSupport:
    model_type = str(model.get("model_type") or model.get("arch") or "unknown")
    architecture = str(model.get("raw_arch") or model.get("arch") or model_type)
    activation = str(model.get("activation") or "silu").lower()
    experts = int(model.get("num_local_experts") or 0)
    family = str(
        model.get("architecture_family")
        or ("decoder_moe" if experts else "decoder_dense")
    )
    reasons = _semantic_reasons(
        model,
        activation=activation,
        family=family,
        tensor_names=(),
        archive=True,
    )
    return _result(
        model_type,
        architecture,
        activation,
        family,
        reasons,
        model,
    )


def analyze_config(
    config: dict[str, Any],
    *,
    tensor_names: tuple[str, ...] = (),
) -> NativeSupport:
    model_type = str(config.get("model_type") or "unknown")
    architectures = config.get("architectures") or ()
    architecture = str(architectures[0] if architectures else model_type)
    activation = str(
        config.get("hidden_act")
        or config.get("hidden_activation")
        or "silu"
    ).lower()
    experts = int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or 0
    )
    family = (
        "decoder_moe"
        if experts
        else "hybrid_decoder"
        if "linear_attention" in (config.get("layer_types") or ())
        else "decoder_dense"
    )
    reasons = _semantic_reasons(
        config,
        activation=activation,
        family=family,
        tensor_names=tensor_names,
        archive=False,
    )
    return _result(
        model_type,
        architecture,
        activation,
        family,
        reasons,
        config,
    )


def _semantic_reasons(
    config: dict[str, Any],
    *,
    activation: str,
    family: str,
    tensor_names: tuple[str, ...],
    archive: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    required_config = (
        ("hidden_size", "hidden_size", "n_embd", "d_model"),
        ("num_hidden_layers", "num_hidden_layers", "layers", "n_layer", "num_layers"),
        ("num_attention_heads", "num_attention_heads", "heads", "n_head", "num_heads"),
        ("vocab_size", "vocab_size", "n_vocab", "padded_vocab_size"),
    )
    for field, *aliases in required_config:
        if _config_value(config, *aliases) is None:
            reasons.append(f"missing required geometry field {field}")

    supported_activations = {"silu", "swish", "gelu_pytorch_tanh"}
    if activation not in supported_activations:
        reasons.append(
            f"native gated-MLP engine does not implement activation {activation!r}"
        )
    if (
        config.get("rms_norm_eps") is None
        and _config_value(config, "layer_norm_eps", "layer_norm_epsilon") is None
        and config.get("norm_eps") is None
        and str(config.get("model_type") or "").lower() != "olmoe"
    ):
        reasons.append(
            "native engine requires RMSNorm or LayerNorm epsilon metadata"
        )

    heads = int(
        _config_value(
            config, "num_attention_heads", "heads", "n_head", "num_heads"
        )
        or 0
    )
    hidden = int(_config_value(config, "hidden_size", "n_embd", "d_model") or 0)
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
    if not heads or not head_dim or head_dim % 2:
        reasons.append("attention head geometry is missing or has odd head_dim")

    if family not in {"decoder_dense", "decoder_moe", "hybrid_decoder"}:
        reasons.append(f"unsupported architecture family {family!r}")

    try:
        required_operators = tuple(config.get("required_operators") or ())
        if not required_operators:
            required_operators = required_operators_from_config(
                config, tensor_names
            )
        missing_operators = unsupported_operators(required_operators)
        if missing_operators:
            reasons.append(
                "native operator registry is missing: "
                + ", ".join(missing_operators)
            )
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(f"cannot compile required operator contract: {exc}")

    if family == "hybrid_decoder":
        for field in (
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        ):
            if not int(config.get(field) or 0):
                reasons.append(f"hybrid decoder is missing {field}")

    if tensor_names and not archive:
        names = set(tensor_names)
        required = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ]
        if family == "hybrid_decoder":
            required.extend(
                f"model.layers.0.linear_attn.{name}"
                for name in (
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "in_proj_a.weight",
                    "in_proj_b.weight",
                    "conv1d.weight",
                    "dt_bias",
                    "A_log",
                    "norm.weight",
                    "out_proj.weight",
                )
            )
        else:
            required.append("model.layers.0.self_attn.o_proj.weight")
        for name in required:
            if name not in names:
                reasons.append(f"native tensor layout is missing {name}")
        if family != "hybrid_decoder":
            separate_qkv = all(
                f"model.layers.0.self_attn.{part}_proj.weight" in names
                for part in ("q", "k", "v")
            )
            fused_qkv = "model.layers.0.self_attn.qkv_proj.weight" in names
            if not separate_qkv and not fused_qkv:
                reasons.append("native tensor layout requires separate or fused QKV")
        if family in {"decoder_dense", "hybrid_decoder"}:
            separate_mlp = all(
                f"model.layers.0.mlp.{part}_proj.weight" in names
                for part in ("gate", "up", "down")
            )
            fused_mlp = (
                "model.layers.0.mlp.gate_up_proj.weight" in names
                and "model.layers.0.mlp.down_proj.weight" in names
            )
            if not separate_mlp and not fused_mlp:
                reasons.append("native tensor layout requires a gated MLP")
        elif not any(
            name.startswith("model.layers.0.mlp.experts.")
            or name.startswith("model.layers.0.block_sparse_moe.experts.")
            for name in names
        ):
            reasons.append("native MoE layout requires packed or separate experts")

    return tuple(reasons)


def _result(
    model_type: str,
    architecture: str,
    activation: str,
    family: str,
    reasons: tuple[str, ...],
    config: dict[str, Any],
) -> NativeSupport:
    from .architectures import architecture_status

    registry = architecture_status(model_type or architecture)
    heads = int(
        _config_value(
            config, "num_attention_heads", "heads", "n_head", "num_heads"
        )
        or 0
    )
    kv_heads = int(
        _config_value(
            config,
            "num_key_value_heads",
            "kv_heads",
            "num_kv_heads",
            "n_head_kv",
        )
        or heads
    )
    attention = (
        "mha" if heads and heads == kv_heads else ("mqa" if kv_heads == 1 else "gqa")
    )
    capabilities = [
        family,
        activation,
        attention,
        "full_causal_kv",
        "bf16",
        (
            "layer_norm"
            if config.get("norm_kind") == "layer_norm"
            or (
                config.get("rms_norm_eps") is None
                and _config_value(config, "layer_norm_eps", "layer_norm_epsilon") is not None
            )
            else "rms_norm"
        ),
    ]
    partial_rotary = float(
        config.get("partial_rotary_factor")
        or (config.get("rope_parameters") or {}).get(
            "partial_rotary_factor"
        )
        or 1.0
    )
    if partial_rotary < 1.0:
        capabilities.append(f"partial_rope:{partial_rotary:g}")
    if family in {"decoder_dense", "hybrid_decoder"}:
        capabilities.extend(("fp8_projections", "adaptive_int8"))
    if family == "hybrid_decoder":
        capabilities.extend(
            (
                "gated_deltanet",
                "recurrent_state_cache",
                "gated_full_attention",
            )
        )
    if config.get("layer_types") or (
        config.get("sliding_window") is not None
        and bool(config.get("use_sliding_window", True))
    ):
        capabilities.append("sliding_attention")
    if config.get("attention_sinks"):
        capabilities.append("attention_sinks")
    return NativeSupport(
        supported=not reasons,
        engine="native" if not reasons else "unsupported",
        model_type=model_type,
        architecture=architecture,
        architecture_family=family,
        activation=activation,
        attention=attention,
        reasons=reasons,
        capabilities=tuple(capabilities),
        architecture_status=(
            registry.native_status if registry is not None else "unregistered"
        ),
        performance_supported=bool(
            registry is not None and registry.native_status == "verified"
        ),
        validated_profile=(
            registry.validated_profile if registry is not None else None
        ),
    )


def _config_value(config: dict[str, Any], *names: str) -> Any:
    return next(
        (config.get(name) for name in names if config.get(name) is not None),
        None,
    )


def _unsupported(
    model_type: str,
    architecture: str,
    reasons: tuple[str, ...],
) -> NativeSupport:
    return NativeSupport(
        supported=False,
        engine="unsupported",
        model_type=model_type,
        architecture=architecture,
        architecture_family="unknown",
        activation="unknown",
        attention="unknown",
        reasons=reasons,
        capabilities=(),
    )


def _tensor_names(root: Path) -> tuple[str, ...]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if isinstance(weight_map, dict):
                return tuple(str(name) for name in weight_map)
        except (OSError, json.JSONDecodeError):
            return ()
    single = root / "model.safetensors"
    if single.is_file():
        try:
            from safetensors import safe_open

            with safe_open(single, framework="pt", device="cpu") as handle:
                return tuple(handle.keys())
        except Exception:
            return ()
    return ()


def _canonical_text_tensor_names(config_path: Path, names: tuple[str, ...]) -> tuple[str, ...]:
    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        source_config = {}
    prefixes = ("model.language_model.", "language_model.")
    cross_layers = _cross_attention_layers(source_config)
    canonical: list[str] = []
    for name in names:
        suffix = next(
            (name.removeprefix(prefix) for prefix in prefixes if name.startswith(prefix)),
            None,
        )
        if suffix is None:
            continue
        if _is_text_adapter_tensor(suffix):
            continue
        compact = _canonical_text_tensor_name(suffix, cross_layers)
        if compact is not None:
            canonical.append(compact)
    return tuple(canonical)


def _cross_attention_layers(config: dict[str, Any]) -> set[int]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        return set()
    layers = text.get("cross_attention_layers")
    if not isinstance(layers, list):
        return set()
    result: set[int] = set()
    for layer in layers:
        try:
            result.add(int(layer))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cross_attention_layers contains invalid layer id {layer!r}"
            ) from exc
    return result


def _canonical_text_tensor_name(suffix: str, cross_layers: set[int]) -> str | None:
    if suffix == "lm_head.weight":
        return suffix
    normalized = suffix if suffix.startswith("model.") else f"model.{suffix}"
    prefix = "model.layers."
    if not normalized.startswith(prefix):
        return normalized
    rest = normalized.removeprefix(prefix)
    layer_text, sep, layer_suffix = rest.partition(".")
    if not sep:
        return normalized
    try:
        layer = int(layer_text)
    except ValueError:
        return normalized
    if layer in cross_layers:
        return None
    skipped_before = sum(1 for skipped in cross_layers if skipped < layer)
    return f"model.layers.{layer - skipped_before}.{layer_suffix}"


def _is_text_adapter_tensor(suffix: str) -> bool:
    return (
        ".cross_attn." in suffix
        or ".cross_attn_attn_gate" in suffix
        or ".cross_attn_mlp_gate" in suffix
    )
