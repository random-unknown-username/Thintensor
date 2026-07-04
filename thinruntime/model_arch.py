"""Architecture discovery shared by conversion, runtime, and benchmark tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from .quantization import QuantizationDescriptor, descriptor_from_config


KNOWN_MODEL_TYPES = {
    "llama",
    "qwen2",
    "qwen2_5",
    "qwen3",
    "smollm",
    "smollm3",
    "tinyllama",
    "phi3",
    "gpt_oss",
}

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
    rope_variant: str | None = None
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_sinks: bool = False
    num_local_experts: int = 0
    num_experts_per_token: int = 0
    swiglu_alpha: float = 1.0
    swiglu_limit: float | None = None
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
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = config_or_path

    model_type = _normalize_model_type(
        str(config.get("model_type") or _raw_arch(config))
    )
    hidden = _required_int(config, "hidden_size")
    heads = _required_int(config, "num_attention_heads")
    layers = _required_int(config, "num_hidden_layers")
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or hidden // heads)
    no_rope = tuple(bool(value) for value in config.get("no_rope_layers", []))
    if not no_rope and model_type == "smollm3":
        interval = int(config.get("no_rope_layer_interval") or 4)
        no_rope = tuple((layer + 1) % interval != 0 for layer in range(layers))
    traits = _semantic_traits(config, tensor_names, heads, kv_heads, layers)

    return ModelDescriptor(
        model_type=model_type,
        hidden_size=hidden,
        intermediate_size=_required_int(config, "intermediate_size"),
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_required_int(config, "vocab_size"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        rope_theta=float(config.get("rope_theta", 10_000.0)),
        rope_scaling=config.get("rope_scaling") or config.get("rope_parameters"),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
        activation=str(config.get("hidden_act", "silu")),
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
        rope_variant=traits["rope_variant"],
        rope_parameters=traits["rope_parameters"],
        attention_bias=bool(config.get("attention_bias", False)),
        mlp_bias=traits["mlp_bias"],
        attention_sinks=traits["attention_sinks"],
        num_local_experts=traits["num_local_experts"],
        num_experts_per_token=traits["num_experts_per_token"],
        swiglu_alpha=float(config.get("swiglu_alpha", 1.702)),
        swiglu_limit=(
            float(config["swiglu_limit"])
            if config.get("swiglu_limit") is not None
            else None
        ),
        required_operators=traits["required_operators"],
        quantization=descriptor_from_config(config, tensor_names),
    )


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
        rms_norm_eps=float(model.get("rms_norm_eps") or 1e-6),
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
        sliding_window=_optional_int(model.get("sliding_window")),
        max_position_embeddings=_optional_int(
            model.get("max_position_embeddings")
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
        swiglu_alpha=float(model.get("swiglu_alpha") or 1.702),
        swiglu_limit=(
            float(model["swiglu_limit"])
            if model.get("swiglu_limit") is not None
            else None
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
    value = config.get(key)
    if value is None:
        raise ValueError(f"config.json is missing required field {key!r}")
    return int(value)


def _raw_arch(config: dict[str, Any]) -> str:
    architectures = config.get("architectures") or []
    return str(architectures[0] if architectures else "")


def _normalize_model_type(value: str) -> str:
    compact = value.lower().replace("-", "").replace("_", "")
    if "smollm3" in compact:
        return "smollm3"
    if "smollm" in compact:
        return "smollm"
    if "tinyllama" in compact:
        return "tinyllama"
    if "qwen3" in compact:
        return "qwen3"
    if "qwen25" in compact:
        return "qwen2_5"
    if "qwen2" in compact or compact == "qwen":
        return "qwen2"
    if "phi3" in compact:
        return "phi3"
    if "gptoss" in compact:
        return "gpt_oss"
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
    if layer_types or config.get("sliding_window") is not None:
        attention_variants.append("sliding_causal_kv")
    required = [
        "embedding",
        "rms_norm",
        "qkv_projection",
        "rope",
        "gqa_attention" if kv_heads < heads else "mha_attention",
        "o_projection",
        "lm_head",
    ]
    if attention_sinks:
        required.append("attention_sinks")
    if layer_types:
        required.append("per_layer_attention_window")
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
        "sliding_window": _optional_int(config.get("sliding_window")),
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
