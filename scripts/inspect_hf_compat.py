#!/usr/bin/env python3
"""Inspect a Hugging Face causal-LM schema before conversion or execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinruntime.archive import ThinArchive
from thinruntime.execution_plan import compile_execution_plan
from thinruntime.model_arch import (
    descriptor_from_hf_config,
    descriptor_from_manifest,
)
from thinruntime.operator_registry import unsupported_operators
from thinruntime.quantization import precision_ladder


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-model")
    source.add_argument("--archive")
    parser.add_argument(
        "--quantization",
        default="auto",
        help="auto, preserve, bf16, fp16, fp8, fp4, q8, q4, or q2",
    )
    parser.add_argument("--allow-requantize", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    archive = None
    if args.hf_model:
        path = Path(args.hf_model)
        descriptor = descriptor_from_hf_config(path)
        source_tensor_names = hf_tensor_names(path)
        tensor_names = canonical_hf_tensor_names(path, source_tensor_names)
        source_label = str(path)
    else:
        archive = ThinArchive(args.archive)
        descriptor = descriptor_from_manifest(archive.manifest)
        tensor_names = tuple(archive.manifest_pages)
        source_tensor_names = tensor_names
        source_label = str(args.archive)

    kernels = available_quant_kernels()
    capability = (
        torch.cuda.get_device_capability()
        if torch.cuda.is_available()
        else None
    )
    plan = compile_execution_plan(
        descriptor,
        tensor_names,
        requested_quantization=args.quantization,
        cuda_capability=capability,
        available_quant_kernels=kernels,
        allow_requantize=args.allow_requantize,
    )
    executor_reasons = current_executor_gaps(plan)
    report = {
        "source": source_label,
        "tensor_count": len(tensor_names),
        "source_tensor_count": len(source_tensor_names),
        "excluded_non_text_tensor_count": (
            len(source_tensor_names) - len(tensor_names)
        ),
        "schema_plan_compiles": plan.supported,
        "schema_plan_errors": list(plan.unsupported_reasons),
        "current_native_executor_ready": plan.supported and not executor_reasons,
        "current_native_executor_gaps": executor_reasons,
        "available_quant_kernels": sorted(kernels),
        "cuda_capability": list(capability) if capability else None,
        "execution_plan": plan.as_dict(),
        "performance_contract": {
            "minimum_speedup_vs_hf": 1.0,
            "target_speedup_vs_hf": 3.0,
            "claim_rule": (
                "claim a win only after correctness-gated real decode beats "
                "the same-model same-precision HF baseline"
            ),
            "candidate_optimizations": list(
                plan.optimization_candidates
            ),
        },
        "precision_ladder": list(
            precision_ladder(descriptor.quantization)
        ),
    }
    if archive is not None:
        archive.close()

    if args.out:
        out = Path(args.out)
        if out.suffix.lower() == ".json":
            out.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            out.write_text(render_markdown(report), encoding="utf-8")
            out.with_suffix(".json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")


def hf_tensor_names(path: Path) -> tuple[str, ...]:
    if path.is_file():
        path = path.parent
    index = path / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        return tuple(payload.get("weight_map", {}))
    single = path / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            return tuple(handle.keys())
    raise FileNotFoundError(
        f"{path} has no model.safetensors or model.safetensors.index.json"
    )


def canonical_hf_tensor_names(
    path: Path, tensor_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Apply the converter's multimodal text-only name projection.

    Compatibility inspection must compile the schema that conversion will
    actually write.  Vision-language repositories such as Qwen3.6 keep their
    decoder under ``model.language_model``; compiling the raw repository names
    reports false missing-role failures even though the converter deliberately
    strips that wrapper and excludes vision pages.
    """
    if path.is_file():
        path = path.parent
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if "text_config" not in config:
        return tensor_names

    raw_cross_layers = config.get("text_config", {}).get(
        "cross_attention_layers", []
    )
    cross_layers = {
        int(layer)
        for layer in raw_cross_layers
        if isinstance(layer, int) and layer >= 0
    }
    prefixes = ("model.language_model.", "language_model.")
    canonical: set[str] = set()
    for name in tensor_names:
        if name == "lm_head.weight":
            canonical.add(name)
            continue
        suffix = next(
            (name[len(prefix) :] for prefix in prefixes if name.startswith(prefix)),
            None,
        )
        if suffix is None or _is_text_adapter_tensor(suffix):
            continue
        normalized = suffix if suffix.startswith("model.") else f"model.{suffix}"
        layer_prefix = "model.layers."
        if not normalized.startswith(layer_prefix):
            canonical.add(normalized)
            continue
        rest = normalized[len(layer_prefix) :]
        layer_text, separator, layer_suffix = rest.partition(".")
        if not separator or not layer_text.isdigit():
            canonical.add(normalized)
            continue
        layer = int(layer_text)
        if layer in cross_layers:
            continue
        compact_layer = layer - sum(
            skipped < layer for skipped in cross_layers
        )
        canonical.add(f"model.layers.{compact_layer}.{layer_suffix}")
    return tuple(sorted(canonical))


def _is_text_adapter_tensor(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".cross_attn.",
            ".cross_attn_attn_gate",
            ".cross_attn_mlp_gate",
        )
    )


def available_quant_kernels() -> set[str]:
    kernels = {"fp8_scaled_matvec"}
    if torch.cuda.is_available():
        kernels.add("mxfp4_matvec")
    return kernels


def current_executor_gaps(plan: Any) -> list[str]:
    descriptor = plan.descriptor
    gaps = []
    supported_schemas = {
        "separate_qkv_gated_dense",
        "fused_qkv_dense",
        "separate_qkv_fused_gate_up_dense",
        "fused_qkv_fused_gate_up_dense",
        "packed_expert_moe",
        "separate_expert_moe",
        "gated_delta_net_dense",
    }
    for layer in plan.layers:
        if layer.schema not in supported_schemas:
            gaps.append(
                f"layer {layer.layer} schema {layer.schema!r} is not implemented"
            )
        for operator in unsupported_operators(layer.operators):
            gaps.append(
                f"layer {layer.layer} operator {operator!r} is not registered"
            )
    if descriptor.quantization.is_quantized and plan.quantization.storage_action not in {
        "preserve",
        "dequantize_selected_experts_on_demand",
    }:
        gaps.append(
            "source quantization lacks a native kernel; BF16 expansion would "
            "defeat low-memory execution"
        )
    return gaps


def render_markdown(report: dict[str, Any]) -> str:
    plan = report["execution_plan"]
    descriptor = plan["descriptor"]
    quant = plan["quantization"]
    lines = [
        "# ThinTensor HF compatibility report",
        "",
        f"- Source: `{report['source']}`",
        f"- Model type label: `{descriptor['model_type']}`",
        f"- Semantic schema: `{plan['schema']}`",
        f"- Architecture family: `{descriptor['architecture_family']}`",
        f"- Attention: `{descriptor['attention_kind']}`",
        f"- MLP: `{descriptor['mlp_kind']}`",
        f"- Source quantization: `{quant['source']['method']}`",
        f"- Quant execution: `{quant['kernel_family']}`",
        f"- Schema plan compiles: `{report['schema_plan_compiles']}`",
        f"- Current native executor ready: `{report['current_native_executor_ready']}`",
        "",
        "## Executor gaps",
        "",
    ]
    gaps = report["current_native_executor_gaps"]
    lines.extend(
        [f"- {value}" for value in gaps]
        if gaps
        else ["- None"]
    )
    lines.extend(
        [
            "",
            "## Optimization candidates",
            "",
            *[
                f"- `{value}`"
                for value in plan["optimization_candidates"]
            ],
            "",
            "## Precision ladder",
            "",
            *[
                f"- `{row['mode']}` ({row['safety']}): {row['note']}"
                for row in report["precision_ladder"]
            ],
            "",
            "A 3x speedup is a benchmark target, not a compatibility claim.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
