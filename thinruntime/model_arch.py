"""Architecture discovery shared by conversion, runtime, and benchmark tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from .quantization import QuantizationDescriptor, descriptor_from_config


@dataclass(frozen=True)
class ModelDescriptor:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    rms_norm_eps: float
    norm_kind: str
    norm_eps: float
    partial_rotary_factor: float
    activation: str
    qkv_bias: bool
    tensor_naming_scheme: str
    attention_variants: tuple[str, ...]
    supported_precision_modes: tuple[str, ...]
    no_rope_layers: tuple[bool, ...]
    architecture_family: str = "dense_decoder"
    attention_kind: str = "gqa"
    mlp_kind: str = "gated_dense"
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    max_position_embeddings: int | None = None
    original_max_position_embeddings: int | None = None
    rope_variant: str | None = None
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_sinks: bool = False
    num_local_experts: int = 0
    num_experts_per_token: int = 0
    norm_topk_prob: bool = True
    swiglu_alpha: float = 1.0
    swiglu_limit: float | None = None
    norm_weight_offset: float = 0.0
    embedding_scale: float = 1.0
    query_pre_attn_scalar: float | None = None
    attention_logit_softcap: float | None = None
    final_logit_softcap: float | None = None
    linear_conv_kernel_dim: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    attention_output_gate: bool = False
    global_head_dim: int = 0
    num_global_key_value_heads: int = 0
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 0
    use_double_wide_mlp: bool = False
    required_operators: tuple[str, ...] = ()
    quantization: QuantizationDescriptor = field(
        default_factory=lambda: descriptor_from_config({})
    )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["attention_variants"] = list(self.attention_variants)
        result["supported_precision_modes"] = list(
            self.supported_precision_modes
        )
        result["no_rope_layers"] = list(self.no_rope_layers)
        result["layer_types"] = list(self.layer_types)
        result["required_operators"] = list(self.required_operators)
        result["quantization"] = self.quantization.as_dict()
        return result

    def layer_uses_rope(self, layer: int) -> bool:
        if layer >= len(self.no_rope_layers):
            return True
        # SmolLM3's upstream field is historically named no_rope_layers, but
        # its attention module assigns use_rope = no_rope_layers[layer].
        if self.model_type == "smollm3":
            return self.no_rope_layers[layer]
        return not self.no_rope_layers[layer]

    @property
    def is_moe(self) -> bool:
        return self.mlp_kind == "sparse_moe"

    def layer_attention_window(self, layer: int) -> int | None:
        if layer < len(self.layer_types):
            layer_type = self.layer_types[layer].lower()
            if "sliding" not in layer_type:
                return None
        return self.sliding_window


def descriptor_from_hf_config(config_or_path: dict[str, Any] | str | Path) -> ModelDescriptor:
    tensor_names: tuple[str, ...] = ()
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path)
        if path.is_dir():
            tensor_names = _hf_tensor_names(path)
            config_path = path / "config.json"
        else:
            config_path = path
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
        config = _effective_text_config(source_config)
        tensor_names = _canonical_text_tensor_names(
            source_config, tensor_names
        )
    else:
        config = _effective_text_config(config_or_path)

    model_type = _normalize_model_type(
        str(config.get("model_type") or _raw_arch(config))
    )
    default_norm_eps = 1e-5 if model_type == "olmoe" else 1e-6
    hidden = _required_int(config, "hidden_size")
    heads = _required_int(config, "num_attention_heads")
    layers = _required_int(config, "num_hidden_layers")
    kv_heads = int(
        _first_config_value(
            config, "num_key_value_heads", "num_kv_heads", "n_head_kv"
        )
        or heads
    )
    head_dim = int(config.get("head_dim") or hidden // heads)
    no_rope = tuple(bool(value) for value in config.get("no_rope_layers", []))
    if not no_rope and model_type == "smollm3":
        interval = int(config.get("no_rope_layer_interval") or 4)
        no_rope = tuple((layer + 1) % interval != 0 for layer in range(layers))
    traits = _semantic_traits(config, tensor_names, heads, kv_heads, layers)

    return ModelDescriptor(
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=_intermediate_size(config, hidden),
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_required_int(config, "vocab_size"),
        tie_word_embeddings=bool(
            config.get(
                "tie_word_embeddings",
                model_type in {"gemma", "gemma2"},
            )
        ),
        rope_theta=float(
            config.get("rope_theta")
            or (config.get("rope_parameters") or {}).get("rope_theta")
            or 10_000.0
        ),
        rope_scaling=config.get("rope_scaling") or config.get("rope_parameters"),
        rms_norm_eps=float(
            config.get("rms_norm_eps")
            or _first_config_value(config, "layer_norm_eps", "layer_norm_epsilon")
            or default_norm_eps
        ),
        norm_kind=(
            "rms_norm"
            if config.get("rms_norm_eps") is not None
            else "layer_norm"
            if _first_config_value(config, "layer_norm_eps", "layer_norm_epsilon") is not None
            else "rms_norm"
        ),
        norm_eps=float(
            config.get("rms_norm_eps")
            or _first_config_value(config, "layer_norm_eps", "layer_norm_epsilon")
            or default_norm_eps
        ),
        partial_rotary_factor=float(
            config.get("partial_rotary_factor")
            or (config.get("rope_parameters") or {}).get(
                "partial_rotary_factor"
            )
            or 1.0
        ),
        activation=str(
            config.get("hidden_act")
            or config.get("hidden_activation")
            or "silu"
        ),
        qkv_bias=bool(
            config.get("attention_bias", config.get("qkv_bias", False))
        ),
        tensor_naming_scheme=traits["tensor_naming_scheme"],
        attention_variants=traits["attention_variants"],
        supported_precision_modes=traits["precision_modes"],
        no_rope_layers=no_rope,
        architecture_family=traits["architecture_family"],
        attention_kind=traits["attention_kind"],
        mlp_kind=traits["mlp_kind"],
        layer_types=traits["layer_types"],
        sliding_window=traits["sliding_window"],
        max_position_embeddings=_optional_int(
            config.get("max_position_embeddings")
        ),
        original_max_position_embeddings=_optional_int(
            config.get("original_max_position_embeddings")
        ),
        rope_variant=traits["rope_variant"],
        rope_parameters=traits["rope_parameters"],
        attention_bias=bool(config.get("attention_bias", False)),
        mlp_bias=traits["mlp_bias"],
        attention_sinks=traits["attention_sinks"],
        num_local_experts=traits["num_local_experts"],
        num_experts_per_token=traits["num_experts_per_token"],
        norm_topk_prob=bool(config.get("norm_topk_prob", True)),
        swiglu_alpha=float(config.get("swiglu_alpha", 1.702)),
        swiglu_limit=(
            float(config["swiglu_limit"])
            if config.get("swiglu_limit") is not None
            else None
        ),
        norm_weight_offset=(
            1.0 if model_type in {"gemma2", "qwen3_5"} else 0.0
        ),
        embedding_scale=(
            hidden**0.5
            if model_type in {"gemma2", "gemma4"}
            else 1.0
        ),
        query_pre_attn_scalar=(
            float(config["query_pre_attn_scalar"])
            if config.get("query_pre_attn_scalar") is not None
            else None
        ),
        attention_logit_softcap=(
            float(config["attn_logit_softcapping"])
            if config.get("attn_logit_softcapping") is not None
            else None
        ),
        final_logit_softcap=(
            float(config["final_logit_softcapping"])
            if config.get("final_logit_softcapping") is not None
            else None
        ),
        linear_conv_kernel_dim=int(
            config.get("linear_conv_kernel_dim") or 0
        ),
        linear_key_head_dim=int(
            config.get("linear_key_head_dim") or 0
        ),
        linear_value_head_dim=int(
            config.get("linear_value_head_dim") or 0
        ),
        linear_num_key_heads=int(
            config.get("linear_num_key_heads") or 0
        ),
        linear_num_value_heads=int(
            config.get("linear_num_value_heads") or 0
        ),
        attention_output_gate=bool(
            config.get("attn_output_gate")
            or config.get("attention_output_gate")
        ),
        global_head_dim=int(config.get("global_head_dim") or 0),
        num_global_key_value_heads=int(
            config.get("num_global_key_value_heads") or 0
        ),
        num_kv_shared_layers=int(
            config.get("num_kv_shared_layers") or 0
        ),
        hidden_size_per_layer_input=int(
            config.get("hidden_size_per_layer_input") or 0
        ),
        vocab_size_per_layer_input=int(
            config.get("vocab_size_per_layer_input") or 0
        ),
        use_double_wide_mlp=bool(
            config.get("use_double_wide_mlp", False)
        ),
        required_operators=traits["required_operators"],
        quantization=descriptor_from_config(config, tensor_names),
    )


def required_operators_from_config(
    config: dict[str, Any], tensor_names: tuple[str, ...] = ()
) -> tuple[str, ...]:
    effective = _effective_text_config(config)
    heads = _required_int(effective, "num_attention_heads")
    layers = _required_int(effective, "num_hidden_layers")
    kv_heads = int(
        _first_config_value(
            effective, "num_key_value_heads", "num_kv_heads", "n_head_kv"
        )
        or heads
    )
    return _semantic_traits(
        effective, tensor_names, heads, kv_heads, layers
    )["required_operators"]


def descriptor_from_manifest(manifest_or_model: dict[str, Any]) -> ModelDescriptor:
    if "model" in manifest_or_model and "pages" in manifest_or_model:
        manifest = manifest_or_model
        model = manifest["model"]
    else:
        manifest = None
        model = manifest_or_model
    raw_type = str(
        model.get("model_type")
        or model.get("raw_arch")
        or model.get("arch")
        or ""
    )
    model_type = _normalize_model_type(raw_type)
    default_norm_eps = 1e-5 if model_type == "olmoe" else 1e-6

    hidden = int(model["hidden_size"])
    heads = int(model["heads"])
    layers = int(model["layers"])
    kv_heads = int(model.get("kv_heads") or heads)
    head_dim = model.get("head_dim")
    if head_dim is None and manifest is not None:
        k_name = "model.layers.0.self_attn.k_proj.weight"
        k_page = next(
            (page for page in manifest["pages"] if page["id"] == k_name),
            None,
        )
        if k_page is not None:
            head_dim = int(k_page["shape"][0]) // kv_heads
    if head_dim is None:
        head_dim = hidden // heads
    no_rope = tuple(bool(value) for value in model.get("no_rope_layers", []))
    if not no_rope and model_type == "smollm3":
        interval = int(model.get("no_rope_layer_interval") or 4)
        no_rope = tuple((layer + 1) % interval != 0 for layer in range(layers))
    tensor_names = tuple(
        str(page["id"]) for page in manifest["pages"]
    ) if manifest is not None else ()
    traits = _semantic_traits(model, tensor_names, heads, kv_heads, layers)

    return ModelDescriptor(
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=int(model.get("intermediate_size") or 0),
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=int(head_dim),
        vocab_size=int(model.get("vocab_size") or 0),
        tie_word_embeddings=bool(model.get("tie_word_embeddings", False)),
        rope_theta=float(model.get("rope_theta") or 10_000.0),
        rope_scaling=model.get("rope_scaling") or model.get("rope_parameters"),
        rms_norm_eps=float(
            model.get("norm_eps")
            or model.get("rms_norm_eps")
            or default_norm_eps
        ),
        norm_kind=str(
            model.get("norm_kind")
            or (
                "rms_norm"
                if model.get("rms_norm_eps") is not None
                else "layer_norm"
                if model.get("layer_norm_eps") is not None
                else "rms_norm"
            )
        ),
        norm_eps=float(
            model.get("norm_eps")
            or model.get("rms_norm_eps")
            or model.get("layer_norm_eps")
            or default_norm_eps
        ),
        partial_rotary_factor=float(
            model.get("partial_rotary_factor") or 1.0
        ),
        activation=str(model.get("activation") or "silu"),
        qkv_bias=bool(model.get("qkv_bias", False)),
        tensor_naming_scheme=str(
            model.get("tensor_naming_scheme") or "hf_decoder_layers"
        ),
        attention_variants=tuple(
            model.get("attention_variants") or traits["attention_variants"]
        ),
        supported_precision_modes=tuple(
            model.get("supported_precision_modes")
            or ("bf16", "head8", "down_fp8", "gate_up_fp8", "mlp_fp8")
        ),
        no_rope_layers=no_rope,
        architecture_family=str(
            model.get("architecture_family")
            or traits["architecture_family"]
        ),
        attention_kind=str(
            model.get("attention_kind") or traits["attention_kind"]
        ),
        mlp_kind=str(model.get("mlp_kind") or traits["mlp_kind"]),
        layer_types=tuple(model.get("layer_types") or traits["layer_types"]),
        sliding_window=traits["sliding_window"],
        max_position_embeddings=_optional_int(
            model.get("max_position_embeddings")
        ),
        original_max_position_embeddings=_optional_int(
            model.get("original_max_position_embeddings")
        ),
        rope_variant=str(
            model.get("rope_variant") or traits["rope_variant"] or ""
        ) or None,
        rope_parameters=(
            model.get("rope_parameters") or traits["rope_parameters"]
        ),
        attention_bias=bool(model.get("attention_bias", False)),
        mlp_bias=bool(model.get("mlp_bias", traits["mlp_bias"])),
        attention_sinks=bool(
            model.get("attention_sinks", traits["attention_sinks"])
        ),
        num_local_experts=int(
            model.get("num_local_experts")
            or traits["num_local_experts"]
        ),
        num_experts_per_token=int(
            model.get("num_experts_per_token")
            or traits["num_experts_per_token"]
        ),
        norm_topk_prob=bool(model.get("norm_topk_prob", True)),
        swiglu_alpha=float(model.get("swiglu_alpha") or 1.702),
        swiglu_limit=(
            float(model["swiglu_limit"])
            if model.get("swiglu_limit") is not None
            else None
        ),
        norm_weight_offset=float(
            model.get("norm_weight_offset")
            if model.get("norm_weight_offset") is not None
            else (
                1.0
                if model_type in {"gemma2", "qwen3_5"}
                else 0.0
            )
        ),
        embedding_scale=float(
            model.get("embedding_scale")
            if model.get("embedding_scale") is not None
            else (
                hidden**0.5
                if model_type in {"gemma2", "gemma4"}
                else 1.0
            )
        ),
        query_pre_attn_scalar=(
            float(model["query_pre_attn_scalar"])
            if model.get("query_pre_attn_scalar") is not None
            else None
        ),
        attention_logit_softcap=(
            float(model["attention_logit_softcap"])
            if model.get("attention_logit_softcap") is not None
            else None
        ),
        final_logit_softcap=(
            float(model["final_logit_softcap"])
            if model.get("final_logit_softcap") is not None
            else None
        ),
        linear_conv_kernel_dim=int(
            model.get("linear_conv_kernel_dim") or 0
        ),
        linear_key_head_dim=int(
            model.get("linear_key_head_dim") or 0
        ),
        linear_value_head_dim=int(
            model.get("linear_value_head_dim") or 0
        ),
        linear_num_key_heads=int(
            model.get("linear_num_key_heads") or 0
        ),
        linear_num_value_heads=int(
            model.get("linear_num_value_heads") or 0
        ),
        attention_output_gate=bool(
            model.get("attention_output_gate", False)
        ),
        global_head_dim=int(model.get("global_head_dim") or 0),
        num_global_key_value_heads=int(
            model.get("num_global_key_value_heads") or 0
        ),
        num_kv_shared_layers=int(
            model.get("num_kv_shared_layers") or 0
        ),
        hidden_size_per_layer_input=int(
            model.get("hidden_size_per_layer_input") or 0
        ),
        vocab_size_per_layer_input=int(
            model.get("vocab_size_per_layer_input") or 0
        ),
        use_double_wide_mlp=bool(
            model.get("use_double_wide_mlp", False)
        ),
        required_operators=tuple(
            model.get("required_operators")
            or traits["required_operators"]
        ),
        quantization=descriptor_from_config(
            {
                "torch_dtype": model.get("source_dtype")
                or model.get("dtype"),
                "quantization_config": model.get("quantization_config"),
            },
            tensor_names,
        ),
    )


def _required_int(config: dict[str, Any], key: str) -> int:
    aliases = {
        "hidden_size": ("hidden_size", "n_embd", "d_model"),
        "intermediate_size": (
            "intermediate_size",
            "ffn_dim",
            "ffn_hidden_size",
            "n_inner",
            "d_ff",
            "moe_intermediate_size",
            "expert_intermediate_size",
        ),
        "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
        "vocab_size": ("vocab_size", "n_vocab", "padded_vocab_size"),
    }
    value = next(
        (config.get(name) for name in aliases.get(key, (key,)) if config.get(name) is not None),
        None,
    )
    if value is None:
        raise ValueError(f"config.json is missing required field {key!r}")
    return int(value)


def _first_config_value(config: dict[str, Any], *names: str) -> Any:
    return next(
        (config.get(name) for name in names if config.get(name) is not None),
        None,
    )


def _intermediate_size(config: dict[str, Any], hidden_size: int) -> int:
    try:
        return _required_int(config, "intermediate_size")
    except ValueError:
        # GPT-style non-gated MLPs conventionally omit n_inner when it is 4*d_model.
        # Tensor-backed manifests replace this fallback with the converter-inferred shape.
        return 4 * hidden_size


def _raw_arch(config: dict[str, Any]) -> str:
    architectures = config.get("architectures") or []
    return str(architectures[0] if architectures else "")


def _effective_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        return config
    effective = dict(text)
    for key in ("architectures", "tie_word_embeddings", "model_type"):
        if key in config:
            effective[key] = config[key]
    skipped_layers = len(_cross_attention_layers(config))
    if skipped_layers:
        layer_count = int(
            _first_config_value(
                effective, "num_hidden_layers", "n_layer", "num_layers"
            )
            or 0
        )
        if layer_count:
            effective["num_hidden_layers"] = max(0, layer_count - skipped_layers)
    return effective


def _canonical_text_tensor_names(
    config: dict[str, Any],
    names: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(config.get("text_config"), dict):
        return names
    prefixes = ("model.language_model.", "language_model.")
    cross_layers = _cross_attention_layers(config)
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
        except (TypeError, ValueError):
            pass
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


def _normalize_model_type(value: str) -> str:
    compact = value.lower().replace("-", "").replace("_", "")
    if "smollm3" in compact:
        return "smollm3"
    if "smollm" in compact:
        return "smollm"
    if "tinyllama" in compact:
        return "tinyllama"
    if "qwen35" in compact:
        return "qwen3_5"
    if "qwen3" in compact:
        return "qwen3"
    if "qwen25" in compact:
        return "qwen2_5"
    if "qwen2" in compact or compact == "qwen":
        return "qwen2"
    if "phi3" in compact:
        return "phi3"
    if "olmoe" in compact:
        return "olmoe"
    if "gptoss" in compact:
        return "gpt_oss"
    if "gemma4" in compact:
        return "gemma4"
    if "gemma2" in compact:
        return "gemma2"
    if "gemma" in compact:
        return "gemma"
    if "llama" in compact:
        return "llama"
    return value.lower()


def _semantic_traits(
    config: dict[str, Any],
    tensor_names: tuple[str, ...],
    heads: int,
    kv_heads: int,
    layers: int,
) -> dict[str, Any]:
    num_experts = int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or 0
    )
    experts_per_token = int(
        config.get("num_experts_per_token")
        or config.get("experts_per_token")
        or config.get("num_experts_per_tok")
        or config.get("num_selected_experts")
        or 0
    )
    has_moe_tensors = any(".mlp.experts." in name for name in tensor_names)
    mlp_kind = "sparse_moe" if num_experts or has_moe_tensors else "gated_dense"
    layer_types = tuple(
        str(value) for value in (config.get("layer_types") or ())
    )
    raw_model_type = _normalize_model_type(
        str(config.get("model_type") or _raw_arch(config))
    )
    if not layer_types and raw_model_type == "gemma2":
        layer_types = tuple(
            "sliding_attention" if layer % 2 == 0 else "full_attention"
            for layer in range(layers)
        )
    if not layer_types and config.get("sliding_window") is not None:
        use_sliding = bool(config.get("use_sliding_window", True))
        if use_sliding:
            layer_types = tuple("sliding_attention" for _ in range(layers))
    attention_sinks = bool(config.get("attention_sinks", False)) or any(
        name.endswith(".self_attn.sinks") for name in tensor_names
    )
    rope_parameters = (
        config.get("rope_parameters") or config.get("rope_scaling")
    )
    rope_variant = None
    if isinstance(rope_parameters, dict):
        rope_variant = (
            rope_parameters.get("rope_type") or rope_parameters.get("type")
        )
    attention_variants = ["causal_kv", "current_only_smoke"]
    use_sliding = bool(config.get("use_sliding_window", True))
    if layer_types or (
        config.get("sliding_window") is not None and use_sliding
    ):
        attention_variants.append("sliding_causal_kv")
    required = [
        "embedding",
        "layer_norm"
        if _first_config_value(config, "layer_norm_eps", "layer_norm_epsilon") is not None
        and config.get("rms_norm_eps") is None
        else "rms_norm",
        "qkv_projection",
        "rope",
        (
            "mha_attention"
            if kv_heads == heads
            else "mqa_attention"
            if kv_heads == 1
            else "gqa_attention"
        ),
        "o_projection",
        "lm_head",
    ]
    if attention_sinks:
        required.append("attention_sinks")
    if layer_types:
        required.append("per_layer_attention_window")
    if "linear_attention" in layer_types:
        required.extend(
            (
                "gated_delta_net",
                "depthwise_causal_conv1d",
                "recurrent_state_cache",
                "gated_full_attention",
            )
        )
    if any(
        int(config.get(field) or 0) > 0
        for field in (
            "hidden_size_per_layer_input",
            "vocab_size_per_layer_input",
            "num_kv_shared_layers",
            "global_head_dim",
        )
    ):
        required.extend(
            (
                "per_layer_embeddings",
                "variable_head_dim_attention",
                "shared_kv_attention",
                "post_attention_norm",
                "post_feedforward_norm",
            )
        )
    if any(
        name.endswith("pre_feedforward_layernorm.weight")
        for name in tensor_names
    ) and any(
        name.endswith("post_feedforward_layernorm.weight")
        for name in tensor_names
    ):
        required.extend(("post_attention_norm", "post_feedforward_norm"))
    if mlp_kind == "sparse_moe":
        required.extend(("topk_router", "sparse_experts", "expert_weighted_sum"))
    else:
        required.extend(("gated_activation", "down_projection"))
    quant = descriptor_from_config(config, tensor_names)
    precision_modes = ["bf16", "head8"]
    if mlp_kind == "gated_dense":
        precision_modes.extend(
            ("down_fp8", "gate_up_fp8", "mlp_fp8", "attn_proj_fp8")
        )
    if quant.is_quantized:
        precision_modes.insert(0, f"native_{quant.method}")
    return {
        "architecture_family": (
            "decoder_moe" if mlp_kind == "sparse_moe" else "decoder_dense"
        ),
        "attention_kind": (
            "mha" if kv_heads == heads else ("mqa" if kv_heads == 1 else "gqa")
        ),
        "mlp_kind": mlp_kind,
        "layer_types": layer_types,
        "sliding_window": (
            _optional_int(config.get("sliding_window"))
            if layer_types or use_sliding
            else None
        ),
        "rope_variant": str(rope_variant) if rope_variant else None,
        "rope_parameters": rope_parameters,
        "attention_sinks": attention_sinks,
        "num_local_experts": num_experts,
        "num_experts_per_token": experts_per_token,
        "mlp_bias": bool(config.get("mlp_bias", False)) or any(
            ".mlp." in name and name.endswith("bias") for name in tensor_names
        ),
        "tensor_naming_scheme": _tensor_naming_scheme(tensor_names),
        "attention_variants": tuple(attention_variants),
        "precision_modes": tuple(precision_modes),
        "required_operators": tuple(required),
    }


def _tensor_naming_scheme(tensor_names: tuple[str, ...]) -> str:
    if any(name.startswith("model.layers.") for name in tensor_names):
        return "hf_decoder_layers"
    if any(name.startswith("transformer.h.") for name in tensor_names):
        return "hf_transformer_h"
    return "hf_decoder_layers"


def _hf_tensor_names(path: Path) -> tuple[str, ...]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            return tuple(str(name) for name in weight_map)
    single = path / "model.safetensors"
    if single.exists():
        try:
            from safetensors import safe_open

            with safe_open(single, framework="pt", device="cpu") as handle:
                return tuple(handle.keys())
        except Exception:
            return ()
    return ()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
