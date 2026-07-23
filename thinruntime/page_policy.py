"""Architecture-neutral page roles used by planning and execution.

The archive ``op`` field is the contract.  Tensor-name inference is retained
only for old v0 archives whose converter emitted ``unknown``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class PageRole(str, Enum):
    GLOBAL = "global"
    NORM = "norm"
    ATTENTION = "attention"
    ROUTER = "router"
    EXPERT = "expert"
    MLP = "mlp"
    STATE_SPACE = "state_space"
    SCALE = "scale"
    OTHER = "other"


_ATTENTION_OPS = {
    "attn_q_proj", "attn_k_proj", "attn_v_proj", "attn_qkv_proj",
    "attn_o_proj", "linear_attn_input_projection",
    "linear_attn_output_projection", "attention_sinks",
}
_NORM_OPS = {
    "input_layernorm", "post_attention_layernorm",
    "pre_feedforward_layernorm", "post_feedforward_layernorm",
    "attn_q_norm", "attn_k_norm", "linear_attn_gated_norm",
    "post_per_layer_input_norm", "per_layer_projection_norm",
}
_MLP_OPS = {"mlp_gate_proj", "mlp_up_proj", "mlp_gate_up_proj", "mlp_down_proj"}
_EXPERT_OPS = {"moe_gate_up", "moe_down", "moe_expert"}
_STATE_OPS = {
    "linear_attn_depthwise_conv", "linear_attn_recurrence_parameters",
    "state_space_input_projection", "state_space_output_projection",
    "state_space_recurrence_parameters", "state_space_convolution",
}
_GLOBAL_OPS = {
    "embed_tokens", "final_norm", "lm_head", "per_layer_token_embeddings",
    "per_layer_model_projection", "per_layer_projection_norm",
}


def page_role(page: Mapping[str, Any]) -> PageRole:
    op = str(page.get("op") or "unknown").lower()
    if page.get("layer") is None or op in _GLOBAL_OPS:
        return PageRole.GLOBAL
    if op in _NORM_OPS or "norm" in op:
        return PageRole.NORM
    if op == "moe_router" or "router" in op:
        return PageRole.ROUTER
    if op in _EXPERT_OPS or op.startswith("moe_expert"):
        return PageRole.EXPERT
    if op in _ATTENTION_OPS or op.startswith("attn_"):
        return PageRole.ATTENTION
    if op in _MLP_OPS or op.startswith("mlp_"):
        return PageRole.MLP
    if op in _STATE_OPS or op.startswith(("state_space_", "linear_attn_")):
        return PageRole.STATE_SPACE
    if op == "scale" or str(page.get("kind") or "") == "scale":
        return PageRole.SCALE
    return _legacy_name_role(str(page.get("id") or ""))


def is_body_weight(page: Mapping[str, Any]) -> bool:
    """Return whether a page is a quantizable decoder-body matrix."""
    if page_role(page) not in {
        PageRole.ATTENTION, PageRole.EXPERT, PageRole.MLP, PageRole.STATE_SPACE
    }:
        return False
    shape = page.get("shape") or ()
    # Runtime FP8/INT4 codecs and their matvec kernels consume 2-D matrices.
    # Recurrent architectures also expose higher-rank depthwise-convolution
    # weights under a body operator role; those must remain in their source
    # layout instead of being flattened implicitly by the matrix codec.
    return len(shape) == 2 and not _is_auxiliary_quant_page(page)


def is_expert_weight(page: Mapping[str, Any]) -> bool:
    return page_role(page) == PageRole.EXPERT and (is_body_weight(page) or is_native_packed_weight(page))


def is_attention_weight(page: Mapping[str, Any]) -> bool:
    return page_role(page) in {PageRole.ATTENTION, PageRole.STATE_SPACE} and is_body_weight(page)


def is_mlp_weight(page: Mapping[str, Any]) -> bool:
    return page_role(page) in {PageRole.MLP, PageRole.EXPERT} and is_body_weight(page)


def is_native_packed_weight(page: Mapping[str, Any]) -> bool:
    return bool(page.get("bits_per_weight")) and bool(page.get("quant_scheme"))


def _is_auxiliary_quant_page(page: Mapping[str, Any]) -> bool:
    page_id = str(page.get("id") or "")
    return page_id.endswith(("_scales", ".scales", ".scale", "qzeros", "g_idx"))


def _legacy_name_role(page_id: str) -> PageRole:
    # Compatibility for archives produced before op metadata was complete.
    if ".experts." in page_id or ".block_sparse_moe.experts." in page_id:
        return PageRole.EXPERT
    if any(token in page_id for token in ("self_attn.", ".attn.", "linear_attn.")):
        return PageRole.ATTENTION
    if ".mlp." in page_id:
        return PageRole.MLP
    if any(token in page_id for token in ("layernorm", ".norm.")):
        return PageRole.NORM
    return PageRole.OTHER

def is_router_weight(page: Mapping[str, Any]) -> bool:
    return page_role(page) == PageRole.ROUTER
