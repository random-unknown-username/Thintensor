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
    layers: tuple["RuntimeLayerGraph", ...]


@dataclass(frozen=True)
class RuntimeLayerGraph:
    """Executable semantic blocks for one decoder layer."""

    layer: int
    schema: str
    normalization: str
    sequence: str
    feedforward: str
    operators: tuple[str, ...]


def select_runtime_executor(
    layer_schemas: Iterable[str],
    required_operators: Iterable[str],
    layer_operators: Iterable[Iterable[str]] | None = None,
) -> RuntimeExecutor:
    """Compile a native per-layer graph from semantic operator contracts."""
    schema_sequence = tuple(map(str, layer_schemas))
    schemas = tuple(dict.fromkeys(schema_sequence))
    operators = frozenset(map(str, required_operators))
    per_layer = tuple(
        tuple(map(str, values)) for values in (layer_operators or ())
    )
    if not schemas:
        raise ValueError("execution plan contains no decoder layer schemas")
    if not per_layer:
        # Compatibility for callers that only need selection metadata.
        per_layer = tuple(tuple(operators) for _ in schemas)
    graphs = []
    for layer, layer_ops in enumerate(per_layer):
        contract = frozenset(layer_ops)
        normalization = next(
            (name for name in ("rms_norm", "layer_norm") if name in contract),
            None,
        )
        if normalization is None:
            raise ValueError(
                f"layer {layer} declares no supported normalization operator"
            )
        if "shared_kv_attention" in contract:
            sequence = "shared_kv_attention"
        elif "gated_delta_net" in contract:
            sequence = "gated_delta_net"
        elif "per_layer_attention_window" in contract:
            sequence = "sliding_causal_attention"
        else:
            sequence = next(
                (
                    name
                    for name in (
                        "mha_attention",
                        "mqa_attention",
                        "gqa_attention",
                        "variable_head_dim_attention",
                    )
                    if name in contract
                ),
                None,
            )
            if sequence is None:
                raise ValueError(
                    f"layer {layer} declares no supported sequence operator"
                )
        if "sparse_experts" in contract:
            feedforward = "sparse_moe"
        elif "gated_activation" in contract and "down_projection" in contract:
            feedforward = "gated_mlp"
        else:
            raise ValueError(
                f"layer {layer} declares no complete feed-forward operator contract"
            )
        schema = schema_sequence[min(layer, len(schema_sequence) - 1)]
        graphs.append(
            RuntimeLayerGraph(
                layer=layer,
                schema=schema,
                normalization=normalization,
                sequence=sequence,
                feedforward=feedforward,
                operators=layer_ops,
            )
        )
    return RuntimeExecutor(
        "operator_graph_decoder",
        schemas,
        tuple(sorted(operators)),
        tuple(graphs),
    )


def unsupported_operators(required: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(name) for name in required if str(name) not in OPERATORS}))


def operator_capabilities(required: Iterable[str]) -> tuple[OperatorCapability, ...]:
    return tuple(OPERATORS[name] for name in dict.fromkeys(map(str, required)) if name in OPERATORS)
