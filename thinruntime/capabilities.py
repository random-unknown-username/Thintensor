"""Architecture capability discovery for engine routing.

ThinTensor accepts any Transformers causal-LM source.  This module decides
whether the native decoder can execute it or whether the CLI should retain the
source as a Transformers fallback.  Decisions are based on semantics and
tensor structure, never a hard-coded model repository name.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
        config = effective
        prefix = "model.language_model."
        tensor_names = tuple(
            f"model.{name.removeprefix(prefix)}"
            for name in tensor_names
            if name.startswith(prefix)
        )
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
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    for field in required_config:
        archive_field = {
            "num_hidden_layers": "layers",
            "num_attention_heads": "heads",
        }.get(field, field)
        if config.get(field) is None and config.get(archive_field) is None:
            reasons.append(f"missing required geometry field {field}")

    supported_activations = {"silu", "swish"}
    if str(config.get("model_type") or "").lower() == "gemma2" or str(
        config.get("model_type") or ""
    ).lower().startswith("gemma4"):
        supported_activations.add("gelu_pytorch_tanh")
    if activation not in supported_activations:
        reasons.append(
            f"native gated-MLP engine does not implement activation {activation!r}"
        )
    if (
        config.get("rms_norm_eps") is None
        and config.get("layer_norm_eps") is None
        and config.get("norm_eps") is None
        and str(config.get("model_type") or "").lower() != "olmoe"
    ):
        reasons.append(
            "native engine requires RMSNorm or LayerNorm epsilon metadata"
        )

    heads = int(
        config.get("num_attention_heads")
        or config.get("heads")
        or 0
    )
    hidden = int(config.get("hidden_size") or 0)
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
    if not heads or not head_dim or head_dim % 2:
        reasons.append("attention head geometry is missing or has odd head_dim")

    if family not in {"decoder_dense", "decoder_moe", "hybrid_decoder"}:
        reasons.append(f"unsupported architecture family {family!r}")

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
            if str(config.get("model_type") or "").lower() == "gemma2":
                for suffix in (
                    "pre_feedforward_layernorm.weight",
                    "post_feedforward_layernorm.weight",
                ):
                    name = f"model.layers.0.{suffix}"
                    if name not in names:
                        reasons.append(
                            f"Gemma2 native layout is missing {name}"
                        )
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
        config.get("num_attention_heads")
        or config.get("heads")
        or 0
    )
    kv_heads = int(
        config.get("num_key_value_heads")
        or config.get("kv_heads")
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
                and config.get("layer_norm_eps") is not None
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
        engine="native" if not reasons else "transformers",
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


def _unsupported(
    model_type: str,
    architecture: str,
    reasons: tuple[str, ...],
) -> NativeSupport:
    return NativeSupport(
        supported=False,
        engine="transformers",
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
