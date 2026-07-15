"""Component-level runtime memory ownership reporting.

The report deliberately distinguishes owned tensors from allocator reserve and
mapped archive bytes.  Components that are subcategories of a pool are split
instead of summed twice, which keeps the steady total suitable for benchmark
and disk/VRAM claims.
"""

from __future__ import annotations

from typing import Any


def build_memory_report(
    runtime: Any,
    weights: Any,
    kv_cache: Any,
    *,
    device: str,
    rss_before_bytes: int | None = None,
    rss_after_bytes: int | None = None,
) -> dict[str, Any]:
    """Return non-overlapping steady ownership and observed process peaks."""

    telemetry = weights.telemetry() if hasattr(weights, "telemetry") else {}
    gpu_cache = telemetry.get("gpu_cache") or {}
    cpu_store = telemetry.get("cpu_store") or {}
    kv = kv_cache.telemetry() if hasattr(kv_cache, "telemetry") else {}

    pool_resident = _integer(
        gpu_cache.get(
            "resident_bytes",
            telemetry.get("resident_bytes", getattr(weights, "resident_weight_bytes", 0)),
        )
    )
    external_resident = min(
        pool_resident,
        _integer(telemetry.get("external_resident_bytes", 0)),
    )
    page_weights = max(0, pool_resident - external_resident)
    retired = _integer(gpu_cache.get("retired_tensor_bytes", 0))
    expert_cache = _integer(gpu_cache.get("mxfp4_expert_slice_cache_bytes", 0))
    runtime_aux = sum(
        _integer(getattr(runtime, name, 0))
        for name in (
            "runtime_fusion_extra_bytes",
            "fused_mlp_extra_bytes",
            "fp8_sparse_residual_bytes",
            "mxfp4_binary_residual_bytes",
            "mxfp4_int8_row_override_bytes",
            "mxfp4_row_postscale_bytes",
            "adaptive_exact_resident_bytes",
        )
    )
    temp_buffers = _integer(getattr(runtime, "temp_buffer_bytes", 0))
    kv_gpu = _integer(kv.get("kv_gpu_bytes", kv.get("gpu_bytes", 0)))
    kv_cpu = _integer(kv.get("kv_cpu_bytes", kv.get("cpu_bytes", 0)))
    cpu_resident = _integer(
        cpu_store.get("resident_bytes", telemetry.get("cpu_resident_bytes", 0))
    )
    cpu_pinned = min(
        cpu_resident,
        _integer(cpu_store.get("pinned_bytes", telemetry.get("cpu_pinned_bytes", 0))),
    )
    cpu_int4_packed = _integer(telemetry.get("cpu_int4_packed_bytes", 0))

    gpu_components = [
        _component("weight_pages", "page_pool", "gpu", page_weights),
        _component(
            "external_registered_allocations",
            "runtime_registered_with_page_pool",
            "gpu",
            external_resident,
            notes=(
                "Runtime allocations registered against the page-pool budget, "
                "including separate execution heads and expert materialization buffers."
            ),
        ),
        _component("retired_async_weights", "page_pool", "gpu", retired),
        _component("expert_slice_cache", "page_pool", "gpu", expert_cache),
        _component("runtime_quantization_sidecars", "runtime", "gpu", runtime_aux),
        _component("decode_scratch", "runtime", "gpu", temp_buffers),
        _component("kv_cache", "kv_cache", "gpu", kv_gpu),
    ]
    gpu_owned = sum(item["steady_bytes"] for item in gpu_components)
    peak_allocated, peak_reserved, current_allocated, current_reserved = (
        _cuda_allocator_values(device)
    )
    gpu_components.append(
        _component(
            "allocator_or_library_unattributed",
            "cuda_allocator",
            "gpu",
            max(0, current_allocated - gpu_owned),
            notes="Allocated CUDA bytes not represented by explicit runtime owners.",
        )
    )
    gpu_owned = sum(item["steady_bytes"] for item in gpu_components)

    cpu_components = [
        _component(
            "page_cache_unpinned",
            "page_pool_cpu_store",
            "cpu",
            max(0, cpu_resident - cpu_pinned),
        ),
        _component("pinned_staging_and_cache", "page_pool_cpu_store", "cpu", cpu_pinned),
        _component(
            "packed_int4_weights_and_scales",
            "cpu_int4_kernel_cache",
            "cpu",
            cpu_int4_packed,
        ),
        _component("kv_cache", "kv_cache", "cpu", kv_cpu),
    ]
    cpu_owned = sum(item["steady_bytes"] for item in cpu_components)
    archive_mapped = _archive_payload_bytes(weights)

    return {
        "schema": "thintensor.memory_ownership.v1",
        "gpu": {
            "components": gpu_components,
            "explicit_steady_owned_bytes": gpu_owned,
            "allocator_current_allocated_bytes": current_allocated,
            "allocator_current_reserved_bytes": current_reserved,
            "allocator_peak_allocated_bytes": peak_allocated,
            "allocator_peak_reserved_bytes": peak_reserved,
            "page_pool_peak_resident_bytes": _integer(
                gpu_cache.get("peak_resident_bytes", telemetry.get("peak_resident_bytes", 0))
            ),
        },
        "cpu": {
            "components": cpu_components,
            "explicit_steady_owned_bytes": cpu_owned,
            "process_rss_before_bytes": rss_before_bytes,
            "process_rss_after_bytes": rss_after_bytes,
            "process_rss_delta_bytes": (
                None
                if rss_before_bytes is None or rss_after_bytes is None
                else rss_after_bytes - rss_before_bytes
            ),
            "archive_mapped_virtual_payload_bytes": archive_mapped,
            "archive_mapping_note": (
                "Mapped virtual bytes are not added to resident ownership; RSS reflects faulted pages."
            ),
        },
        "accounting_notes": [
            "Page-pool external bytes are split from page weights, not added a second time.",
            "Pinned bytes are a subcategory of CPU resident cache bytes.",
            "Packed CPU INT4 weights include their affine scales/offsets and are separate from the archive mmap.",
            "CUDA allocator reserve is reported separately from allocated ownership.",
        ],
    }


def _component(
    name: str,
    owner: str,
    location: str,
    steady_bytes: int,
    *,
    notes: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "owner": owner,
        "location": location,
        "steady_bytes": max(0, int(steady_bytes)),
    }
    if notes:
        result["notes"] = notes
    return result


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _cuda_allocator_values(device: str) -> tuple[int, int, int, int]:
    if device != "cuda":
        return 0, 0, 0, 0
    import torch

    return (
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
        int(torch.cuda.memory_allocated()),
        int(torch.cuda.memory_reserved()),
    )


def _archive_payload_bytes(weights: Any) -> int:
    archive = getattr(weights, "archive", None)
    pages = getattr(archive, "pages", None)
    if not isinstance(pages, dict):
        return 0
    return sum(_integer(page.get("size", 0)) for page in pages.values())
