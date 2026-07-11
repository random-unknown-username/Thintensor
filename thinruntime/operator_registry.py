"""Native operator contracts independent of model-family names."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OperatorCapability:
    name: str
    category: str
    implementation: str
    causal: bool = True
    quantizable_weights: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_OPERATORS = (
    OperatorCapability("embedding", "global", "token_embedding"),
    OperatorCapability("lm_head", "global", "vocabulary_projection"),
    OperatorCapability("rms_norm", "normalization", "rms_norm"),
    OperatorCapability("layer_norm", "normalization", "layer_norm"),
    OperatorCapability("qkv_projection", "attention", "role_projection", quantizable_weights=True),
    OperatorCapability("o_projection", "attention", "role_projection", quantizable_weights=True),
    OperatorCapability("rope", "position", "rotary_embedding"),
    OperatorCapability("mha_attention", "attention", "causal_attention"),
    OperatorCapability("mqa_attention", "attention", "causal_attention"),
    OperatorCapability("gqa_attention", "attention", "causal_attention"),
    OperatorCapability("attention_sinks", "attention", "causal_attention_sinks"),
    OperatorCapability("per_layer_attention_window", "attention", "sliding_causal_attention"),
    OperatorCapability("gated_activation", "mlp", "gated_activation"),
    OperatorCapability("down_projection", "mlp", "role_projection", quantizable_weights=True),
    OperatorCapability("topk_router", "moe", "exact_topk_router"),
    OperatorCapability("sparse_experts", "moe", "selected_expert_matvec", quantizable_weights=True),
    OperatorCapability("expert_weighted_sum", "moe", "weighted_expert_sum"),
    OperatorCapability("gated_delta_net", "recurrent", "gated_delta_net"),
    OperatorCapability("depthwise_causal_conv1d", "recurrent", "depthwise_causal_conv1d"),
    OperatorCapability("recurrent_state_cache", "recurrent", "recurrent_state_cache"),
    OperatorCapability("gated_full_attention", "attention", "gated_causal_attention"),
    OperatorCapability("per_layer_embeddings", "global", "per_layer_token_embedding"),
    OperatorCapability("variable_head_dim_attention", "attention", "variable_head_attention"),
    OperatorCapability("shared_kv_attention", "attention", "shared_kv_attention"),
    OperatorCapability("post_attention_norm", "normalization", "post_attention_norm"),
    OperatorCapability("post_feedforward_norm", "normalization", "post_feedforward_norm"),
)

OPERATORS = {operator.name: operator for operator in _OPERATORS}


@dataclass(frozen=True)
class RuntimeExecutor:
    name: str
    layer_schemas: tuple[str, ...]
    required_operators: tuple[str, ...]


def select_runtime_executor(
    layer_schemas: Iterable[str],
    required_operators: Iterable[str],
) -> RuntimeExecutor:
    """Select a native executor from semantic schemas, never family names."""
    schemas = tuple(dict.fromkeys(map(str, layer_schemas)))
    operators = frozenset(map(str, required_operators))
    if {"per_layer_embeddings", "shared_kv_attention"} & operators:
        return RuntimeExecutor(
            "shared_kv_decoder",
            schemas,
            ("per_layer_embeddings", "shared_kv_attention"),
        )
    if "gated_delta_net" in operators:
        return RuntimeExecutor(
            "recurrent_hybrid_decoder",
            schemas,
            ("gated_delta_net", "recurrent_state_cache"),
        )
    moe = tuple(schema for schema in schemas if schema.endswith("_moe"))
    dense = tuple(schema for schema in schemas if not schema.endswith("_moe"))
    if moe and dense:
        raise ValueError(
            "native per-layer graph execution is missing for heterogeneous "
            f"dense/MoE schemas: {', '.join(schemas)}"
        )
    if moe:
        return RuntimeExecutor(
            "sparse_moe_decoder",
            schemas,
            ("topk_router", "sparse_experts", "expert_weighted_sum"),
        )
    if not schemas:
        raise ValueError("execution plan contains no decoder layer schemas")
    if any("fused_" in schema for schema in schemas) or {
        "post_attention_norm",
        "post_feedforward_norm",
    }.issubset(operators):
        return RuntimeExecutor("generic_dense_decoder", schemas, ())
    return RuntimeExecutor("optimized_dense_decoder", schemas, ())


def unsupported_operators(required: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(name) for name in required if str(name) not in OPERATORS}))


def operator_capabilities(required: Iterable[str]) -> tuple[OperatorCapability, ...]:
    return tuple(OPERATORS[name] for name in dict.fromkeys(map(str, required)) if name in OPERATORS)
