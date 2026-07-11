"""Compile common Hugging Face causal-LM tensor schemas into ThinTensor roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Iterable

from .model_arch import ModelDescriptor
from .quantization import QuantExecutionPlan, plan_quantization


@dataclass(frozen=True)
class LayerExecutionPlan:
    layer: int
    schema: str
    attention_window: int | None
    tensors: dict[str, str]
    optional_tensors: dict[str, str]
    tensor_groups: dict[str, tuple[str, ...]]
    operators: tuple[str, ...]
    missing_required_roles: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["missing_required_roles"] = list(
            self.missing_required_roles
        )
        result["tensor_groups"] = {
            role: list(values) for role, values in self.tensor_groups.items()
        }
        result["operators"] = list(self.operators)
        return result


@dataclass(frozen=True)
class CompiledExecutionPlan:
    descriptor: ModelDescriptor
    schema: str
    globals: dict[str, str]
    layers: tuple[LayerExecutionPlan, ...]
    quantization: QuantExecutionPlan
    supported: bool
    unsupported_reasons: tuple[str, ...]
    optimization_candidates: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.as_dict(),
            "schema": self.schema,
            "globals": self.globals,
            "layers": [layer.as_dict() for layer in self.layers],
            "quantization": self.quantization.as_dict(),
            "supported": self.supported,
            "unsupported_reasons": list(self.unsupported_reasons),
            "optimization_candidates": list(
                self.optimization_candidates
            ),
        }


def compile_execution_plan(
    descriptor: ModelDescriptor,
    tensor_names: Iterable[str],
    *,
    requested_quantization: str = "auto",
    cuda_capability: tuple[int, int] | None = None,
    available_quant_kernels: Iterable[str] = (),
    allow_requantize: bool = False,
) -> CompiledExecutionPlan:
    names = set(str(name) for name in tensor_names)
    globals_map = _global_roles(names)
    layer_schemas = tuple(
        _detect_schema(descriptor, _names_for_layer(names, layer), layer)
        for layer in range(descriptor.num_hidden_layers)
    )
    layer_plans = tuple(
        _compile_layer(descriptor, names, layer, layer_schemas[layer])
        for layer in range(descriptor.num_hidden_layers)
    )
    unique_schemas = tuple(dict.fromkeys(layer_schemas))
    schema = (
        unique_schemas[0]
        if len(unique_schemas) == 1
        else "heterogeneous:" + ",".join(unique_schemas)
    )
    reasons = []
    for role in ("embed_tokens", "final_norm", "lm_head"):
        if role not in globals_map:
            if role == "lm_head" and descriptor.tie_word_embeddings:
                continue
            reasons.append(f"missing global tensor role {role!r}")
    for layer in layer_plans:
        if layer.missing_required_roles:
            reasons.append(
                f"layer {layer.layer} missing roles: "
                + ", ".join(layer.missing_required_roles)
            )
    quant = plan_quantization(
        descriptor.quantization,
        requested=requested_quantization,
        cuda_capability=cuda_capability,
        available_kernels=available_quant_kernels,
        allow_requantize=allow_requantize,
    )
    if (
        any(layer.schema == "packed_expert_moe" for layer in layer_plans)
        and descriptor.quantization.method == "mxfp4"
        and quant.supported
        and not quant.exact_storage_preserved
    ):
        quant = replace(
            quant,
            storage_action="dequantize_selected_experts_on_demand",
            reason=(
                "preserve MXFP4 archive pages and expand only GPU-selected "
                "experts when a native kernel is unavailable"
            ),
        )
    if not quant.supported:
        reasons.append(f"quantization: {quant.reason}")
    return CompiledExecutionPlan(
        descriptor=descriptor,
        schema=schema,
        globals=globals_map,
        layers=layer_plans,
        quantization=quant,
        supported=not reasons,
        unsupported_reasons=tuple(reasons),
        optimization_candidates=_optimization_candidates(
            descriptor, schema, quant
        ),
    )


def _detect_schema(
    descriptor: ModelDescriptor, names: set[str], layer: int
) -> str:
    if any(".linear_attn." in name for name in names) or (
        layer < len(descriptor.layer_types)
        and descriptor.layer_types[layer] == "linear_attention"
    ):
        return "gated_delta_net_dense"
    if any(".block_sparse_moe.experts." in name for name in names):
        return "separate_expert_moe"
    if any(
        re.search(r"\.mlp\.experts\.\d+\.", name) for name in names
    ):
        return "separate_expert_moe"
    if any(".mlp.experts." in name for name in names):
        return "packed_expert_moe"
    fused_qkv = any(".self_attn.qkv_proj." in name for name in names) or any(
        ".attn.c_attn." in name for name in names
    )
    fused_gate_up = any(
        ".mlp.gate_up_proj." in name for name in names
    )
    if fused_qkv and fused_gate_up:
        return "fused_qkv_fused_gate_up_dense"
    if fused_qkv:
        return "fused_qkv_dense"
    if fused_gate_up:
        return "separate_qkv_fused_gate_up_dense"
    return "separate_qkv_gated_dense"


def _global_roles(names: set[str]) -> dict[str, str]:
    aliases = {
        "embed_tokens": (
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "model.decoder.embed_tokens.weight",
        ),
        "final_norm": (
            "model.norm.weight",
            "transformer.ln_f.weight",
            "model.decoder.final_layer_norm.weight",
        ),
        "lm_head": ("lm_head.weight", "output.weight"),
    }
    result = {}
    for role, candidates in aliases.items():
        value = _first_present(names, candidates)
        if value is not None:
            result[role] = value
    return result


def _compile_layer(
    descriptor: ModelDescriptor,
    names: set[str],
    layer: int,
    schema: str,
) -> LayerExecutionPlan:
    prefixes = (
        f"model.layers.{layer}.",
        f"transformer.h.{layer}.",
        f"model.decoder.layers.{layer}.",
    )
    prefix = next(
        (value for value in prefixes if any(name.startswith(value) for name in names)),
        prefixes[0],
    )
    tensors: dict[str, str] = {}
    optional: dict[str, str] = {}
    groups: dict[str, tuple[str, ...]] = {}

    def required(role: str, *suffixes: str) -> None:
        value = _first_present(
            names, tuple(prefix + suffix for suffix in suffixes)
        )
        if value is not None:
            tensors[role] = value

    def optional_role(role: str, *suffixes: str) -> None:
        value = _first_present(
            names, tuple(prefix + suffix for suffix in suffixes)
        )
        if value is not None:
            optional[role] = value

    required("input_norm", "input_layernorm.weight", "ln_1.weight")
    required(
        "post_attention_norm",
        "post_attention_layernorm.weight",
        "ln_2.weight",
    )
    if schema.startswith("fused_qkv"):
        required(
            "qkv_weight",
            "self_attn.qkv_proj.weight",
            "self_attn.qkv_proj.qweight",
            "attn.c_attn.weight",
            "attn.c_attn.qweight",
        )
        optional_role("qkv_bias", "self_attn.qkv_proj.bias", "attn.c_attn.bias")
    else:
        for role, projection in (
            ("q_weight", "q_proj"),
            ("k_weight", "k_proj"),
            ("v_weight", "v_proj"),
        ):
            required(
                role,
                f"self_attn.{projection}.weight",
                f"self_attn.{projection}.qweight",
                f"self_attn.{projection}.weight_packed",
            )
            optional_role(
                role.replace("_weight", "_bias"),
                f"self_attn.{projection}.bias",
            )
            for suffix, quant_role in (
                ("scales", "scales"),
                ("qzeros", "zeros"),
                ("g_idx", "group_index"),
                ("weight_scale", "weight_scale"),
            ):
                optional_role(
                    f"{role}_{quant_role}",
                    f"self_attn.{projection}.{suffix}",
                )
    required(
        "o_weight",
        "self_attn.o_proj.weight",
        "self_attn.o_proj.qweight",
        "self_attn.o_proj.weight_packed",
        "attn.c_proj.weight",
        "attn.c_proj.qweight",
    )
    optional_role("o_bias", "self_attn.o_proj.bias", "attn.c_proj.bias")
    optional_role("q_norm", "self_attn.q_norm.weight")
    optional_role("k_norm", "self_attn.k_norm.weight")
    optional_role("attention_sinks", "self_attn.sinks")

    if schema == "gated_delta_net_dense":
        for role, suffix in (
            ("linear_qkv_weight", "linear_attn.in_proj_qkv.weight"),
            ("linear_z_weight", "linear_attn.in_proj_z.weight"),
            ("linear_a_weight", "linear_attn.in_proj_a.weight"),
            ("linear_b_weight", "linear_attn.in_proj_b.weight"),
            ("linear_conv_weight", "linear_attn.conv1d.weight"),
            ("linear_dt_bias", "linear_attn.dt_bias"),
            ("linear_a_log", "linear_attn.A_log"),
            ("linear_norm_weight", "linear_attn.norm.weight"),
            ("linear_out_weight", "linear_attn.out_proj.weight"),
        ):
            required(role, suffix)

    if schema == "packed_expert_moe":
        required("router_weight", "mlp.router.weight", "block_sparse_moe.gate.weight")
        optional_role("router_bias", "mlp.router.bias", "block_sparse_moe.gate.bias")
        required(
            "experts_gate_up",
            "mlp.experts.gate_up_proj",
            "mlp.experts.gate_up_proj_blocks",
        )
        optional_role(
            "experts_gate_up_scales",
            "mlp.experts.gate_up_proj_scales",
        )
        optional_role(
            "experts_gate_up_bias",
            "mlp.experts.gate_up_proj_bias",
        )
        required(
            "experts_down",
            "mlp.experts.down_proj",
            "mlp.experts.down_proj_blocks",
        )
        optional_role(
            "experts_down_scales",
            "mlp.experts.down_proj_scales",
        )
        optional_role("experts_down_bias", "mlp.experts.down_proj_bias")
    elif schema == "separate_expert_moe":
        required("router_weight", "block_sparse_moe.gate.weight", "mlp.gate.weight")
        expert_roles: dict[str, list[str]] = {
            "experts_gate": [], "experts_up": [], "experts_down": []
        }
        for expert in range(descriptor.num_local_experts):
            for role, suffixes in (
                (
                    "experts_gate",
                    (
                        f"block_sparse_moe.experts.{expert}.w1.weight",
                        f"mlp.experts.{expert}.gate_proj.weight",
                    ),
                ),
                (
                    "experts_up",
                    (
                        f"block_sparse_moe.experts.{expert}.w3.weight",
                        f"mlp.experts.{expert}.up_proj.weight",
                    ),
                ),
                (
                    "experts_down",
                    (
                        f"block_sparse_moe.experts.{expert}.w2.weight",
                        f"mlp.experts.{expert}.down_proj.weight",
                    ),
                ),
            ):
                value = _first_present(
                    names, tuple(prefix + suffix for suffix in suffixes)
                )
                if value is not None:
                    expert_roles[role].append(value)
        groups.update(
            (role, tuple(values)) for role, values in expert_roles.items()
        )
    elif "fused_gate_up" in schema:
        required(
            "gate_up_weight",
            "mlp.gate_up_proj.weight",
            "mlp.gate_up_proj.qweight",
            "mlp.gate_up_proj.weight_packed",
        )
        optional_role("gate_up_bias", "mlp.gate_up_proj.bias")
        required(
            "down_weight",
            "mlp.down_proj.weight",
            "mlp.down_proj.qweight",
            "mlp.down_proj.weight_packed",
        )
        optional_role("down_bias", "mlp.down_proj.bias")
    else:
        required(
            "gate_weight",
            "mlp.gate_proj.weight",
            "mlp.gate_proj.qweight",
            "mlp.gate_proj.weight_packed",
        )
        required(
            "up_weight",
            "mlp.up_proj.weight",
            "mlp.up_proj.qweight",
            "mlp.up_proj.weight_packed",
        )
        required(
            "down_weight",
            "mlp.down_proj.weight",
            "mlp.down_proj.qweight",
            "mlp.down_proj.weight_packed",
        )
        for role, projection in (
            ("gate_bias", "gate_proj"),
            ("up_bias", "up_proj"),
            ("down_bias", "down_proj"),
        ):
            optional_role(role, f"mlp.{projection}.bias")

    missing_list = [
        role
        for role in _required_roles(schema, descriptor.num_local_experts)
        if role not in tensors
    ]
    if schema == "separate_expert_moe":
        for role in ("experts_gate", "experts_up", "experts_down"):
            if len(groups.get(role, ())) != descriptor.num_local_experts:
                missing_list.append(
                    f"{role}[{descriptor.num_local_experts}]"
                )
    operators = _layer_operators(descriptor, schema, layer)
    return LayerExecutionPlan(
        layer=layer,
        schema=schema,
        attention_window=descriptor.layer_attention_window(layer),
        tensors=tensors,
        optional_tensors=optional,
        tensor_groups=groups,
        operators=operators,
        missing_required_roles=tuple(missing_list),
    )


def _required_roles(schema: str, experts: int) -> tuple[str, ...]:
    common = ("input_norm", "post_attention_norm")
    if schema == "gated_delta_net_dense":
        attention = (
            "linear_qkv_weight",
            "linear_z_weight",
            "linear_a_weight",
            "linear_b_weight",
            "linear_conv_weight",
            "linear_dt_bias",
            "linear_a_log",
            "linear_norm_weight",
            "linear_out_weight",
        )
    else:
        common += ("o_weight",)
        attention = (
            ("qkv_weight",)
            if schema.startswith("fused_qkv")
            else ("q_weight", "k_weight", "v_weight")
        )
    if schema == "packed_expert_moe":
        mlp = ("router_weight", "experts_gate_up", "experts_down")
    elif schema == "separate_expert_moe":
        mlp = ("router_weight",)
    elif "fused_gate_up" in schema:
        mlp = ("gate_up_weight", "down_weight")
    else:
        mlp = ("gate_weight", "up_weight", "down_weight")
    return common + attention + mlp


def _optimization_candidates(
    descriptor: ModelDescriptor,
    schema: str,
    quant: QuantExecutionPlan,
) -> tuple[str, ...]:
    candidates = [
        "shape_autotune",
        "role_aware_matvec",
        "lm_head_backend_autotune",
        "causal_kv_layout",
    ]
    if descriptor.tie_word_embeddings:
        candidates.append("separate_quantized_execution_head")
    if descriptor.layer_types:
        candidates.append("layer_window_aware_kv")
    if "moe" in schema:
        candidates.extend(
            (
                "router_topk_gpu",
                "expert_hotset_cache",
                "selected_expert_prefetch",
                "grouped_expert_matvec",
            )
        )
    else:
        candidates.extend(("gate_up_multi_matvec", "selective_mlp_precision"))
    if quant.source.is_quantized:
        candidates.append(f"native_{quant.source.method}_execution")
    return tuple(candidates)


def _first_present(
    names: set[str],
    candidates: tuple[str, ...],
) -> str | None:
    return next((value for value in candidates if value in names), None)


def _names_for_layer(names: set[str], layer: int) -> set[str]:
    prefixes = (
        f"model.layers.{layer}.",
        f"transformer.h.{layer}.",
        f"model.decoder.layers.{layer}.",
    )
    return {name for name in names if name.startswith(prefixes)}


def _layer_operators(
    descriptor: ModelDescriptor, schema: str, layer: int
) -> tuple[str, ...]:
    if schema == "gated_delta_net_dense":
        operators = [
            descriptor.norm_kind,
            "gated_delta_net",
            "depthwise_causal_conv1d",
            "recurrent_state_cache",
        ]
    else:
        attention = (
            "mha_attention"
            if descriptor.num_key_value_heads == descriptor.num_attention_heads
            else "mqa_attention"
            if descriptor.num_key_value_heads == 1
            else "gqa_attention"
        )
        operators = [
            descriptor.norm_kind,
            "qkv_projection",
            "rope",
            attention,
            "o_projection",
        ]
    if descriptor.layer_attention_window(layer) is not None:
        operators.append("per_layer_attention_window")
    if schema.endswith("_moe"):
        operators.extend(("topk_router", "sparse_experts", "expert_weighted_sum"))
    else:
        operators.extend(("gated_activation", "down_projection"))
    return tuple(operators)
