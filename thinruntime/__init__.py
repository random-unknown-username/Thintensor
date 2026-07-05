"""ThinTensor Python runtime.

Public objects are loaded lazily so archive-only CLI commands do not require
PyTorch, Triton, or Transformers to be installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.0"

_EXPORTS = {
    "ThinArchive": (".archive", "ThinArchive"),
    "NativeSupport": (".capabilities", "NativeSupport"),
    "analyze_hf_directory": (".capabilities", "analyze_hf_directory"),
    "analyze_archive_model": (".capabilities", "analyze_archive_model"),
    "ArchitectureStatus": (".architectures", "ArchitectureStatus"),
    "architecture_status": (".architectures", "architecture_status"),
    "ModelDescriptor": (".model_arch", "ModelDescriptor"),
    "descriptor_from_hf_config": (".model_arch", "descriptor_from_hf_config"),
    "descriptor_from_manifest": (".model_arch", "descriptor_from_manifest"),
    "CompiledExecutionPlan": (".execution_plan", "CompiledExecutionPlan"),
    "compile_execution_plan": (".execution_plan", "compile_execution_plan"),
    "QuantExecutionPlan": (".quantization", "QuantExecutionPlan"),
    "QuantizationDescriptor": (".quantization", "QuantizationDescriptor"),
    "plan_quantization": (".quantization", "plan_quantization"),
    "precision_ladder": (".quantization", "precision_ladder"),
    "map_dtype": (".torch_loader", "map_dtype"),
    "load_tensor_view": (".torch_loader", "load_tensor_view"),
    "load_thin_model": (".hf_loader", "load_thin_model"),
    "PagedKVCache": (".gpu_runtime", "PagedKVCache"),
    "ThinGpuPagePool": (".gpu_runtime", "ThinGpuPagePool"),
    "ThinGpuCausalLMRuntime": (".gpu_runtime", "ThinGpuCausalLMRuntime"),
    "ThinGpuQwenRuntime": (".gpu_runtime", "ThinGpuQwenRuntime"),
    "ThinGpuWeights": (".gpu_runtime", "ThinGpuWeights"),
    "benchmark_gpu_runtime": (".gpu_runtime", "benchmark_gpu_runtime"),
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
