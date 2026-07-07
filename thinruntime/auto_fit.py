"""Device-aware, layer-preserving weight residency planning.

The planner deliberately protects global tensors and the first/last decoder
layers.  It only reduces precision from the middle outwards, where a local
error has the most opportunities to be attenuated before logits are produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from .archive import ThinArchive
from .torch_loader import map_dtype


MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class AutoFitPlan:
    mode: str
    total_device_bytes: int
    total_budget_bytes: int
    runtime_reserve_bytes: int
    weight_budget_bytes: int
    source_weight_bytes: int
    estimated_weight_bytes: int
    kv_cache_bytes: int
    residency: str
    expert_int4: bool
    expert_int4_layers: tuple[int, ...]
    expert_int4_group_size: int
    dense_fp8: bool
    dense_fp8_layers: tuple[int, ...]
    dense_int4: bool
    dense_int4_layers: tuple[int, ...]
    dense_int4_group_size: int
    preserved_layers: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def expert_int4_layer_spec(self) -> str | None:
        return _layers_to_spec(self.expert_int4_layers)

    @property
    def dense_fp8_layer_spec(self) -> str | None:
        return _layers_to_spec(self.dense_fp8_layers)

    @property
    def dense_int4_layer_spec(self) -> str | None:
        return _layers_to_spec(self.dense_int4_layers)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expert_int4_layer_spec"] = self.expert_int4_layer_spec
        payload["dense_fp8_layer_spec"] = self.dense_fp8_layer_spec
        payload["dense_int4_layer_spec"] = self.dense_int4_layer_spec
        return payload


def plan_auto_fit(
    archive_path: str | Path,
    *,
    total_device_bytes: int,
    context_tokens: int,
    total_budget_bytes: int = 0,
    mode: str = "on",
    dtype_bytes: int = 2,
    int4_group_size: int | None = None,
) -> AutoFitPlan:
    """Build a deterministic fit plan without loading model tensors.

    ``total_budget_bytes`` is a whole-device budget, not merely a weight-cache
    limit.  KV storage and a conservative execution reserve are subtracted
    before selecting a weight plan.
    """
    if mode not in {"off", "on", "aggressive"}:
        raise ValueError("auto quant mode must be off, on, or aggressive")
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    if total_device_bytes <= 0:
        raise ValueError("total_device_bytes must be positive")

    archive = ThinArchive(archive_path)
    try:
        constrained_budget = total_budget_bytes > 0
        effective_int4_group_size = (
            int4_group_size
            if int4_group_size is not None
            else (
                32
                if mode == "aggressive"
                else 4
                if constrained_budget
                else 16
            )
        )
        manifest = archive.manifest
        model = manifest["model"]
        pages = [
            page
            for page in manifest.get("pages", [])
            if page.get("kind") != "fused_physical"
            and page.get("fused_to") is None
        ]
        source_bytes = sum(_resident_page_bytes(page, dtype_bytes) for page in pages)
        physical_source_bytes = source_bytes
        ple_compressed = False
        if str(model.get("model_type") or "").startswith("gemma4"):
            ple_page = next(
                (
                    page
                    for page in pages
                    if page.get("id")
                    == "model.embed_tokens_per_layer.weight"
                ),
                None,
            )
            if ple_page is not None:
                ple_original = _resident_page_bytes(
                    ple_page, dtype_bytes
                )
                ple_rows = int(ple_page["shape"][0])
                source_bytes = (
                    source_bytes
                    - ple_original
                    + ple_original // dtype_bytes
                    + ple_rows * 4
                )
                ple_compressed = True
        layers = int(model["layers"])
        kv_heads = int(model.get("kv_heads") or model.get("heads") or 1)
        heads = int(model.get("heads") or 1)
        hidden = int(model["hidden_size"])
        head_dim = int(model.get("head_dim") or hidden // heads)
        kv_bytes = (
            2 * layers * kv_heads * head_dim * context_tokens * dtype_bytes
        )
        physical_budget = (
            min(total_budget_bytes, total_device_bytes)
            if total_budget_bytes > 0
            else total_device_bytes
        )
        # Covers activation buffers, Triton workspaces, allocator fragmentation,
        # CUDA context growth, and a bounded amount of in-flight staging.
        execution_reserve = max(384 * MIB, int(physical_budget * 0.08))
        runtime_reserve = min(
            physical_budget,
            kv_bytes + execution_reserve,
        )
        weight_budget = max(0, physical_budget - runtime_reserve)
        is_moe = (
            str(model.get("mlp_kind") or "") == "sparse_moe"
            or int(model.get("num_local_experts") or 0) > 0
            or any(_is_expert_weight(str(page["id"])) for page in pages)
        )

        reasons: list[str] = [
            "global tensors, attention, routers, and normalization stay exact",
            "precision is reduced from middle layers outward",
            (
                "respecting an explicit constrained-device budget"
                if constrained_budget
                else "using the full safe device-memory envelope"
            ),
        ]
        if ple_compressed:
            reasons.append(
                "Gemma 4 token-indexed per-layer embeddings use row-scaled INT8 while all hot projections remain exact"
            )
        if mode == "off" or source_bytes <= weight_budget:
            return AutoFitPlan(
                mode=(
                    "ple_int8_full_vram"
                    if ple_compressed and source_bytes <= weight_budget
                    else "exact"
                    if source_bytes <= weight_budget
                    else "exact_stream"
                ),
                total_device_bytes=total_device_bytes,
                total_budget_bytes=physical_budget,
                runtime_reserve_bytes=runtime_reserve,
                weight_budget_bytes=weight_budget,
                source_weight_bytes=physical_source_bytes,
                estimated_weight_bytes=source_bytes,
                kv_cache_bytes=kv_bytes,
                residency="all" if source_bytes <= weight_budget else "stream",
                expert_int4=False,
                expert_int4_layers=(),
                expert_int4_group_size=effective_int4_group_size,
                dense_fp8=False,
                dense_fp8_layers=(),
                dense_int4=False,
                dense_int4_layers=(),
                dense_int4_group_size=effective_int4_group_size,
                preserved_layers=tuple(range(layers)),
                reasons=tuple(reasons),
            )

        candidates = _middle_out_layers(layers)
        # Neither automatic mode quantizes the first or final layer. The
        # quality-biased mode additionally protects the early routing stack and
        # the final two layers; aggressive mode quantizes every middle layer.
        protected = {0, layers - 1} if layers > 1 else {0}
        if mode == "on" and is_moe and constrained_budget:
            first_quantized = max(1, int(round(layers * 0.44)))
            last_quantized = max(first_quantized, layers - 2)
            expert_selectable = list(range(first_quantized, last_quantized))
        else:
            expert_selectable = [
                layer for layer in candidates if layer not in protected
            ]
        dense_selectable = [
            layer for layer in candidates if layer not in protected
        ]

        if is_moe:
            selected_experts: list[int] = []
            selected_fp8: list[int] = []
            selected_dense_int4: list[int] = []
            estimated = source_bytes

            expert_bytes = _bytes_by_layer(
                pages,
                lambda page: _is_expert_weight(str(page["id"])),
                dtype_bytes,
            )
            projection_bytes = _bytes_by_layer(
                pages,
                lambda page: _is_dense_projection(str(page["id"])),
                dtype_bytes,
            )

            # Step 1: Expert INT4 middle-out using expert_selectable
            for layer in expert_selectable:
                if estimated <= weight_budget:
                    break
                original = expert_bytes.get(layer, 0)
                if original <= 0:
                    continue
                int4_b = _int4_bytes(
                    original,
                    dtype_bytes=dtype_bytes,
                    group_size=effective_int4_group_size,
                )
                estimated -= original - int4_b
                selected_experts.append(layer)

            # Step 1b: Expand to all selectable expert layers if needed
            if estimated > weight_budget and (
                mode == "aggressive" or not constrained_budget
            ):
                all_selectable_experts = [layer for layer in candidates if layer not in protected]
                for layer in all_selectable_experts:
                    if layer in selected_experts:
                        continue
                    if estimated <= weight_budget:
                        break
                    original = expert_bytes.get(layer, 0)
                    if original <= 0:
                        continue
                    int4_b = _int4_bytes(
                        original,
                        dtype_bytes=dtype_bytes,
                        group_size=effective_int4_group_size,
                    )
                    estimated -= original - int4_b
                    selected_experts.append(layer)

            reasons.append(
                "only expert matrices use grouped INT4; active routing remains exact"
            )
            if estimated > weight_budget:
                reasons.append(
                    "compressed archive exceeds the weight budget, so selected experts stream through a bounded LRU"
                )
            preserved = tuple(
                layer for layer in range(layers)
                if layer not in selected_experts and layer not in selected_fp8 and layer not in selected_dense_int4
            )

            # Rerun check if group_size < 32 and it exceeds budget
            if (
                estimated > weight_budget
                and int4_group_size is None
                and effective_int4_group_size < 32
                and (mode == "aggressive" or not constrained_budget)
            ):
                return plan_auto_fit(
                    archive_path,
                    total_device_bytes=total_device_bytes,
                    context_tokens=context_tokens,
                    total_budget_bytes=total_budget_bytes,
                    mode=mode,
                    dtype_bytes=dtype_bytes,
                    int4_group_size=32,
                )

            # Mode name
            if estimated <= weight_budget:
                mode_str = (
                    "expert_int4_full_vram"
                    if not constrained_budget
                    else "expert_int4"
                )
                if selected_fp8 or selected_dense_int4:
                    mode_str = "moe_quant_fit"
            else:
                mode_str = "expert_int4_stream"

            return AutoFitPlan(
                mode=mode_str,
                total_device_bytes=total_device_bytes,
                total_budget_bytes=physical_budget,
                runtime_reserve_bytes=runtime_reserve,
                weight_budget_bytes=weight_budget,
                source_weight_bytes=physical_source_bytes,
                estimated_weight_bytes=estimated,
                kv_cache_bytes=kv_bytes,
                residency="stream",
                expert_int4=bool(selected_experts),
                expert_int4_layers=tuple(sorted(selected_experts)),
                expert_int4_group_size=effective_int4_group_size,
                dense_fp8=bool(selected_fp8),
                dense_fp8_layers=tuple(sorted(selected_fp8)),
                dense_int4=bool(selected_dense_int4),
                dense_int4_layers=tuple(sorted(selected_dense_int4)),
                dense_int4_group_size=effective_int4_group_size,
                preserved_layers=preserved,
                reasons=tuple(reasons),
            )

        selected_fp8 = []
        selected_int4 = []
        estimated = source_bytes
        projection_bytes = _bytes_by_layer(
            pages,
            lambda page: _is_dense_projection(str(page["id"])),
            dtype_bytes,
        )

        # Step 1: Dense FP8 middle-out
        for layer in dense_selectable:
            if estimated <= weight_budget:
                break
            original = projection_bytes.get(layer, 0)
            if original <= 0:
                continue
            fp8_b = _fp8_bytes(original, dtype_bytes, hidden)
            estimated -= original - fp8_b
            selected_fp8.append(layer)

        # Step 2: Dense INT4 middle-out
        if estimated > weight_budget:
            for layer in dense_selectable:
                if estimated <= weight_budget:
                    break
                original = projection_bytes.get(layer, 0)
                if original <= 0:
                    continue
                current_size = _fp8_bytes(original, dtype_bytes, hidden) if layer in selected_fp8 else original
                int4_b = _int4_bytes(
                    original,
                    dtype_bytes=dtype_bytes,
                    group_size=effective_int4_group_size,
                )
                estimated -= current_size - int4_b
                if layer in selected_fp8:
                    selected_fp8.remove(layer)
                selected_int4.append(layer)

        reasons.append("dense middle-layer projections use row-scaled FP8 or INT4")
        if estimated > weight_budget:
            reasons.append("compressed weights use bounded streaming residency")

        # Rerun check if group_size < 32 and it exceeds budget
        if estimated > weight_budget and int4_group_size is None and effective_int4_group_size < 32:
            return plan_auto_fit(
                archive_path,
                total_device_bytes=total_device_bytes,
                context_tokens=context_tokens,
                total_budget_bytes=total_budget_bytes,
                mode=mode,
                dtype_bytes=dtype_bytes,
                int4_group_size=32,
            )

        # Mode name
        if estimated <= weight_budget:
            mode_str = "dense_int4" if selected_int4 else "dense_fp8"
        else:
            mode_str = "dense_int4_stream" if selected_int4 else "dense_fp8_stream"

        preserved = tuple(
            layer for layer in range(layers)
            if layer not in selected_fp8 and layer not in selected_int4
        )

        return AutoFitPlan(
            mode=mode_str,
            total_device_bytes=total_device_bytes,
            total_budget_bytes=physical_budget,
            runtime_reserve_bytes=runtime_reserve,
            weight_budget_bytes=weight_budget,
            source_weight_bytes=physical_source_bytes,
            estimated_weight_bytes=estimated,
            kv_cache_bytes=kv_bytes,
            residency="stream",
            expert_int4=False,
            expert_int4_layers=(),
            expert_int4_group_size=effective_int4_group_size,
            dense_fp8=bool(selected_fp8),
            dense_fp8_layers=tuple(sorted(selected_fp8)),
            dense_int4=bool(selected_int4),
            dense_int4_layers=tuple(sorted(selected_int4)),
            dense_int4_group_size=effective_int4_group_size,
            preserved_layers=preserved,
            reasons=tuple(reasons),
        )
    finally:
        archive.close()


def _resident_page_bytes(page: dict[str, Any], dtype_bytes: int) -> int:
    elements = math.prod(int(dim) for dim in page.get("shape", ()))
    try:
        source = map_dtype(str(page["dtype"]))
        source_bytes = source.itemsize
    except Exception:
        source_bytes = dtype_bytes
    if str(page.get("dtype", "")).startswith(("torch.", "float", "bfloat")):
        source_bytes = dtype_bytes
    return elements * source_bytes


def _bytes_by_layer(
    pages: Iterable[dict[str, Any]],
    predicate: Any,
    dtype_bytes: int,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for page in pages:
        layer = page.get("layer")
        if layer is None or not predicate(page):
            continue
        key = int(layer)
        result[key] = result.get(key, 0) + _resident_page_bytes(page, dtype_bytes)
    return result


def _int4_bytes(source_bytes: int, *, dtype_bytes: int, group_size: int) -> int:
    weights = math.ceil(source_bytes / max(dtype_bytes, 1) / 2)
    scales = math.ceil(source_bytes / max(dtype_bytes, 1) / group_size) * 4
    return weights + scales


def _fp8_bytes(original: int, dtype_bytes: int, hidden: int) -> int:
    if original <= 0:
        return 0
    estimated_fp8 = original // max(dtype_bytes, 1)
    estimated_fp8 += max(4, original // max(hidden * dtype_bytes, 1) * 4)
    return estimated_fp8


def _middle_out_layers(layers: int) -> list[int]:
    center = (layers - 1) / 2.0
    return sorted(range(layers), key=lambda layer: (abs(layer - center), layer))


def _is_expert_weight(page_id: str) -> bool:
    return (
        ".mlp.experts." in page_id
        or ".block_sparse_moe.experts." in page_id
    ) and page_id.endswith(".weight")


def _is_dense_projection(page_id: str) -> bool:
    return page_id.endswith(
        (
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "mlp.gate_up_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight",
        )
    ) and ".experts." not in page_id


def _layers_to_spec(layers: tuple[int, ...]) -> str | None:
    if not layers:
        return None
    runs: list[str] = []
    start = previous = layers[0]
    for layer in layers[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        runs.append(str(start) if start == previous else f"{start}:{previous + 1}")
        start = previous = layer
    runs.append(str(start) if start == previous else f"{start}:{previous + 1}")
    return ",".join(runs)
