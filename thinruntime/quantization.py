"""Schema-driven source quantization discovery and execution planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class QuantizationDescriptor:
    method: str
    storage_dtype: str
    bits: int | None
    group_size: int | None
    scale_dtype: str | None
    zero_point: bool | None
    modules_to_not_convert: tuple[str, ...]
    source_config: dict[str, Any]

    @property
    def is_quantized(self) -> bool:
        return self.method not in {"none", "bf16", "fp16", "fp32"}

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["modules_to_not_convert"] = list(
            self.modules_to_not_convert
        )
        return result


@dataclass(frozen=True)
class QuantExecutionPlan:
    source: QuantizationDescriptor
    requested: str
    storage_action: str
    compute_dtype: str
    kernel_family: str
    exact_storage_preserved: bool
    requires_requantization: bool
    supported: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source"] = self.source.as_dict()
        return result


def precision_ladder(
    source: QuantizationDescriptor,
) -> tuple[dict[str, Any], ...]:
    """Return honest storage/compute choices without silently lowering bits."""
    rows: list[dict[str, Any]] = []

    def add(
        mode: str,
        *,
        safety: str,
        conversion: str,
        note: str,
    ) -> None:
        rows.append(
            {
                "mode": mode,
                "safety": safety,
                "conversion": conversion,
                "note": note,
            }
        )

    if not source.is_quantized:
        add(
            "bf16",
            safety="safe_default",
            conversion="none",
            note="reference execution precision",
        )
        add(
            "fp8",
            safety="opt_in_validate",
            conversion="scaled_quantization",
            note="use only on selected tensor roles after correctness testing",
        )
        for mode in ("q8", "q4", "q2"):
            add(
                mode,
                safety="experimental_validate",
                conversion=f"quantize_to_{mode}",
                note="requires a native packed kernel and model validation",
            )
        return tuple(rows)

    add(
        f"native_{source.method}",
        safety="preferred",
        conversion="preserve",
        note="preserve source bits, scales, zero-point, and grouping",
    )
    add(
        "bf16",
        safety="fallback",
        conversion="dequantize",
        note="correctness fallback when no native kernel exists",
    )
    if source.bits == 8:
        add(
            "fp8",
            safety="peer_format_validate",
            conversion="requantize",
            note=(
                "8-bit integer and FP8 have different ranges; conversion is "
                "not automatically safer or smaller"
            ),
        )
        add(
            "q4",
            safety="lossy_opt_in",
            conversion="requantize",
            note="4-bit storage is allowed only after correctness validation",
        )
        add(
            "q2",
            safety="lossy_opt_in",
            conversion="requantize",
            note="2-bit storage is a separate experimental mode",
        )
    elif source.bits == 4:
        if source.method in {"mxfp4", "fp4"}:
            add(
                "fp4",
                safety="native_or_layout_convert",
                conversion="preserve_numeric_format",
                note="keep block scales and use a hardware-compatible layout",
            )
        else:
            add(
                "fp4",
                safety="cross_format_validate",
                conversion="requantize",
                note="integer Q4 and FP4 are not interchangeable encodings",
            )
        add(
            "q2",
            safety="lossy_opt_in",
            conversion="requantize",
            note="never enabled automatically",
        )
    elif source.bits == 2:
        add(
            "q2",
            safety="preferred",
            conversion="preserve",
            note="do not reduce precision below the source automatically",
        )
    return tuple(rows)


def descriptor_from_config(
    config: dict[str, Any],
    tensor_names: Iterable[str] = (),
) -> QuantizationDescriptor:
    raw = config.get("quantization_config")
    quant = dict(raw) if isinstance(raw, dict) else {}
    names = tuple(tensor_names)
    method = _normalize_method(
        quant.get("quant_method")
        or quant.get("format")
        or quant.get("scheme")
        or _method_from_tensor_names(names)
        or "none"
    )
    bits = _optional_int(
        quant.get("bits")
        or quant.get("weight_bits")
        or quant.get("num_bits")
    )
    if bits is None:
        bits = {
            "mxfp4": 4,
            "fp4": 4,
            "int8": 8,
            "q8": 8,
            "int4": 4,
            "q4": 4,
            "q2": 2,
        }.get(method)
    storage_dtype = _storage_dtype(method, bits, config, names)
    modules = quant.get("modules_to_not_convert") or ()
    if not isinstance(modules, (list, tuple)):
        modules = ()
    return QuantizationDescriptor(
        method=method,
        storage_dtype=storage_dtype,
        bits=bits,
        group_size=_optional_int(
            quant.get("group_size")
            or quant.get("group_size_weights")
            or quant.get("block_size")
        ),
        scale_dtype=_optional_string(
            quant.get("scale_dtype") or quant.get("scales_dtype")
        ),
        zero_point=(
            _optional_bool(quant.get("zero_point"))
            if "zero_point" in quant
            else (
                not bool(quant["sym"])
                if "sym" in quant
                else None
            )
        ),
        modules_to_not_convert=tuple(str(value) for value in modules),
        source_config=quant,
    )


def plan_quantization(
    source: QuantizationDescriptor,
    *,
    requested: str = "auto",
    cuda_capability: tuple[int, int] | None = None,
    available_kernels: Iterable[str] = (),
    allow_requantize: bool = False,
) -> QuantExecutionPlan:
    requested = requested.lower().replace("-", "_")
    kernels = set(available_kernels)
    capability = cuda_capability or (0, 0)

    if requested == "auto":
        return _automatic_plan(source, capability, kernels)
    if requested in {"preserve", "native"}:
        plan = _automatic_plan(source, capability, kernels)
        if not plan.exact_storage_preserved:
            return _unsupported(
                source,
                requested,
                "no native kernel is available for the source encoding",
            )
        return plan
    if requested in {"bf16", "fp16"}:
        return QuantExecutionPlan(
            source=source,
            requested=requested,
            storage_action="dequantize_on_load",
            compute_dtype=requested,
            kernel_family=f"{requested}_matvec",
            exact_storage_preserved=False,
            requires_requantization=False,
            supported=True,
            reason="explicit high-precision execution fallback",
        )
    target_bits = {"fp8": 8, "fp4": 4, "q8": 8, "q4": 4, "q2": 2}.get(
        requested
    )
    if target_bits is None:
        return _unsupported(
            source,
            requested,
            f"unknown requested quantization mode {requested!r}",
        )
    if not allow_requantize:
        return _unsupported(
            source,
            requested,
            "cross-format requantization is opt-in because it changes numerics",
        )
    if source.bits is not None and target_bits < source.bits:
        reason = (
            f"explicit lossy requantization from {source.bits} to "
            f"{target_bits} bits"
        )
    else:
        reason = "explicit cross-format requantization"
    required_kernel = f"{requested}_matvec"
    if required_kernel not in kernels:
        return _unsupported(
            source,
            requested,
            f"required kernel {required_kernel!r} is unavailable",
        )
    return QuantExecutionPlan(
        source=source,
        requested=requested,
        storage_action=f"requantize_to_{requested}",
        compute_dtype=requested,
        kernel_family=required_kernel,
        exact_storage_preserved=False,
        requires_requantization=True,
        supported=True,
        reason=reason,
    )


def _automatic_plan(
    source: QuantizationDescriptor,
    capability: tuple[int, int],
    kernels: set[str],
) -> QuantExecutionPlan:
    native_kernel = {
        "mxfp4": "mxfp4_matvec",
        "fp4": "fp4_matvec",
        "q8": "q8_matvec",
        "int8": "q8_matvec",
        "q4": "q4_matvec",
        "int4": "q4_matvec",
        "q2": "q2_matvec",
        "fp8": "fp8_scaled_matvec",
    }.get(source.method)
    hardware_allows = (
        source.method not in {"mxfp4", "fp4"} or capability >= (7, 5)
    )
    if native_kernel and native_kernel in kernels and hardware_allows:
        return QuantExecutionPlan(
            source=source,
            requested="auto",
            storage_action="preserve",
            compute_dtype=source.storage_dtype,
            kernel_family=native_kernel,
            exact_storage_preserved=True,
            requires_requantization=False,
            supported=True,
            reason="native source-format kernel selected",
        )
    if source.method == "none":
        return QuantExecutionPlan(
            source=source,
            requested="auto",
            storage_action="preserve",
            compute_dtype="bf16",
            kernel_family="bf16_matvec",
            exact_storage_preserved=True,
            requires_requantization=False,
            supported=True,
            reason="unquantized source uses the BF16 reference path",
        )
    return QuantExecutionPlan(
        source=source,
        requested="auto",
        storage_action="dequantize_on_load",
        compute_dtype="bf16",
        kernel_family="bf16_matvec",
        exact_storage_preserved=False,
        requires_requantization=False,
        supported=True,
        reason=(
            "source encoding is preserved in the archive but lacks a native "
            "runtime kernel; use BF16 execution fallback"
        ),
    )


def _unsupported(
    source: QuantizationDescriptor,
    requested: str,
    reason: str,
) -> QuantExecutionPlan:
    return QuantExecutionPlan(
        source=source,
        requested=requested,
        storage_action="none",
        compute_dtype="none",
        kernel_family="none",
        exact_storage_preserved=False,
        requires_requantization=False,
        supported=False,
        reason=reason,
    )


def _normalize_method(value: Any) -> str:
    compact = str(value).lower().replace("-", "").replace("_", "")
    aliases = {
        "none": "none",
        "mxfp4": "mxfp4",
        "nvfp4": "fp4",
        "fp4": "fp4",
        "float8": "fp8",
        "fp8": "fp8",
        "int8": "int8",
        "q8": "q8",
        "q80": "q8",
        "int4": "int4",
        "q4": "q4",
        "q4km": "q4",
        "q4ks": "q4",
        "int2": "q2",
        "q2": "q2",
    }
    return aliases.get(compact, str(value).lower())


def _method_from_tensor_names(names: tuple[str, ...]) -> str | None:
    if any(name.endswith("_blocks") for name in names) and any(
        name.endswith("_scales") for name in names
    ):
        return "mxfp4"
    return None


def _storage_dtype(
    method: str,
    bits: int | None,
    config: dict[str, Any],
    names: tuple[str, ...],
) -> str:
    if method in {"mxfp4", "fp4", "fp8"}:
        return method
    if bits is not None:
        return f"int{bits}"
    dtype = config.get("torch_dtype") or config.get("dtype")
    if dtype:
        return str(dtype).replace("torch.", "")
    if names:
        return "archive"
    return "bf16"


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
